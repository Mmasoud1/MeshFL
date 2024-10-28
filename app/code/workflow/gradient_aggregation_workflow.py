from nvflare.apis.impl.controller import Controller, Task, ClientTask
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable

class GradientAggregationWorkflow(Controller):
    def __init__(
        self,
        aggregator_id="gradient_aggregator",
        min_clients: int = 2,
        num_rounds: int = 50, # <<<<<<<<<<<
        start_round: int = 0,
        wait_time_after_min_received: int = 10,
        train_timeout: int = 0,
        ignore_result_error: bool = False,
        task_check_period: float = 0.5,
        persist_every_n_rounds: int = 1,
        snapshot_every_n_rounds: int = 1,
    ):
        super().__init__()
        self.aggregator_id = aggregator_id
        self.aggregator = None
        self._train_timeout = train_timeout
        self._min_clients = min_clients
        self._num_rounds = num_rounds
        self._start_round = start_round
        self._wait_time_after_min_received = wait_time_after_min_received
        self._ignore_result_error = ignore_result_error
        self._task_check_period = task_check_period
        self._persist_every_n_rounds = persist_every_n_rounds
        self._snapshot_every_n_rounds = snapshot_every_n_rounds

    def start_controller(self, fl_ctx: FLContext) -> None:
        self.aggregator = self._engine.get_component(self.aggregator_id)

    def stop_controller(self, fl_ctx: FLContext):
        pass

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext) -> None:
        current_round = self._start_round
        fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round)

        while current_round < self._num_rounds:
            self.log_info(fl_ctx, f"Starting round {current_round + 1}/{self._num_rounds}")

            # Task to collect gradients from all sites
            get_gradients_task = Task(
                name="train_and_get_gradients",
                data=Shareable(),
                props={},
                timeout=self._train_timeout,
                result_received_cb=self._accept_site_result,
            )

            # Broadcast the task to all sites and wait for the results
            self.broadcast_and_wait(
                task=get_gradients_task,
                min_responses=self._min_clients,
                wait_time_after_min_received=self._wait_time_after_min_received,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            # Perform gradient aggregation
            self.log_info(fl_ctx, "Start gradient aggregation.")
            aggregated_shareable = self.aggregator.aggregate(fl_ctx)
            self.log_info(fl_ctx, "End gradient aggregation.")

            # Task to send the aggregated gradients back to the sites
            accept_aggregated_gradients_task = Task(
                name="accept_aggregated_gradients",
                data=aggregated_shareable,
                props={},
                timeout=self._train_timeout,
            )

            # Broadcast aggregated gradients to sites
            self.broadcast_and_wait(
                task=accept_aggregated_gradients_task,
                min_responses=self._min_clients,
                wait_time_after_min_received=self._wait_time_after_min_received,
                fl_ctx=fl_ctx,
                abort_signal=abort_signal,
            )

            # Increment the round (epoch)
            current_round += 1
            fl_ctx.set_prop(key="CURRENT_ROUND", value=current_round)
      

    def _accept_site_result(self, client_task: ClientTask, fl_ctx: FLContext) -> bool:
        # Accept gradients from each client and forward them to the aggregator
        accepted = self.aggregator.accept(client_task.result, fl_ctx)
        return accepted

    def process_result_of_unknown_task(self, task: Task, fl_ctx: FLContext) -> None:
        pass
