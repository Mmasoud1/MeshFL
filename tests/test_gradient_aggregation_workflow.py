import pytest
from unittest.mock import MagicMock
from nvflare.apis.shareable import Shareable
from app.code.workflow.gradient_aggregation_workflow import GradientAggregationWorkflow

# ----- Dummy Classes for Testing -----

class DummyFLContext:
    def __init__(self):
        self.props = {}
    def set_prop(self, key, value):
        self.props[key] = value
    def get_prop(self, key):
        return self.props.get(key, None)

class DummySignal:
    pass

class DummyAggregator:
    def __init__(self):
        self.accept_called = 0
    def accept(self, result, fl_ctx):
        self.accept_called += 1
        # Always accept for testing purposes.
        return True
    def aggregate(self, fl_ctx):
        s = Shareable()
        s["aggregated_gradients"] = [0, 0]
        return s

# ----- Fixtures for the Workflow and Dummy Contexts -----

@pytest.fixture
def dummy_workflow():
    # Creating a workflow instance with a min number of rounds for testing.
    wf = GradientAggregationWorkflow(
        num_rounds=2,
        start_round=0,
        min_clients=1,
        wait_time_after_min_received=0.1,  # Shorten wait times for tests
        task_check_period=0.1
    )
    # Patch the _engine to return the dummy aggregator when requested.
    dummy_engine = MagicMock()
    dummy_engine.get_component.return_value = DummyAggregator()
    wf._engine = dummy_engine
    # Overriding broadcast_and_wait and log_info to avoid delays and excessive logging.
    wf.broadcast_and_wait = MagicMock()
    wf.log_info = MagicMock()
    return wf

@pytest.fixture
def dummy_fl_ctx():
    return DummyFLContext()

@pytest.fixture
def dummy_signal():
    return DummySignal()

# ----- Test Cases -----

def test_start_controller(dummy_workflow, dummy_fl_ctx):
    dummy_workflow.start_controller(dummy_fl_ctx)
    assert dummy_workflow.aggregator is not None
    # Check that the aggregator is an instance of DummyAggregator.
    from app.code.workflow.gradient_aggregation_workflow import GradientAggregationWorkflow
    assert isinstance(dummy_workflow.aggregator, DummyAggregator)

def test_control_flow(dummy_workflow, dummy_fl_ctx, dummy_signal):
    # Before calling control_flow, ensure aggregator is set.
    dummy_workflow.aggregator = DummyAggregator()

    # Replace broadcast_and_wait with a MagicMock to count calls.
    dummy_workflow.broadcast_and_wait = MagicMock()
    
    # Call control_flow; with num_rounds=2, expect two iterations.
    dummy_workflow.control_flow(dummy_signal, dummy_fl_ctx)
    
    # Each round should trigger two calls to broadcast_and_wait:
    # one for the "train_and_get_gradients" task and one for "accept_aggregated_gradients".
    expected_calls = 2 * 2  # 2 rounds * 2 calls per round
    assert dummy_workflow.broadcast_and_wait.call_count == expected_calls
    # The FLContext property "CURRENT_ROUND" should be updated to 2.
    assert dummy_fl_ctx.get_prop("CURRENT_ROUND") == 2

def test_accept_site_result(dummy_workflow, dummy_fl_ctx):
    # Create a dummy client task with a result.
    dummy_result = "dummy_gradients"
    client_task = MagicMock()
    client_task.result = dummy_result
    dummy_workflow.aggregator = DummyAggregator()
    
    accepted = dummy_workflow._accept_site_result(client_task, dummy_fl_ctx)
    assert accepted is True
    # Ensure the aggregator's accept method was called exactly once.
    assert dummy_workflow.aggregator.accept_called == 1


