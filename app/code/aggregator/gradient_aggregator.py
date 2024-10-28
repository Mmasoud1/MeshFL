import torch
import numpy as np
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.aggregator import Aggregator
from ..executor.dist import GenericLogger

class GradientAggregator(Aggregator):
    def __init__(self):
        self.gradients_list = []  # initializes an empty list to store gradients from different clients.


        # Initialize the logger  will be located in /simulator_workspace/gradient_aggregator.log
        self.logger = GenericLogger(log_file_path='gradient_aggregator.log')  # Customize the log file path

    def accept(self, shareable: Shareable, fl_ctx: FLContext):
        # This fun receives a Shareable object containing the gradients from a client

        if "gradients" not in shareable:
            return False  # Reject invalid results: e.g. Shareable object doesn't contain gradients (i.e., "gradients" key is missing)        

        # Accept collects the gradients from client nodes and stores them in a list.
        gradients = shareable["gradients"]
        
        if not gradients:
            return False  # Reject empty gradients
        
        self.gradients_list.append(gradients)

        return True  # True: indicates that the gradients from the client have been successfully accepted.

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        # Perform gradient aggregation by averaging the gradients across clients and sends the aggregated gradients back to the clients.
        num_clients = len(self.gradients_list)

        self.logger.log_message(f"Num of clients to aggregate :  {num_clients}")

        # Perform the aggregation 
        aggregated_gradients = self.average_gradients(self.gradients_list)

        # Clear the gradients list for the next round
        self.gradients_list = []

        # Return the aggregated gradients
        result = Shareable()
        result["aggregated_gradients"] = aggregated_gradients
        return result

    def average_gradients(self, gradients_list):
        # Convert gradients to numpy arrays and perform averaging
        n = len(gradients_list)

        # Initialize Empty Arrays: sum_arrays is created as a list of arrays with the same shape as the gradients. 
        # These arrays will accumulate the gradients from all clients.
        sum_arrays = [np.zeros_like(arr) for arr in gradients_list[0]]


        # It loops through the list of client gradients, summing them element-wise. 
        # The gradients from each client are added to the corresponding position in the sum_arrays.
        for gradients in gradients_list:
            for i, grad in enumerate(gradients):
                sum_arrays[i] += grad
        
        # Averaging: Once all gradients are summed, each array in sum_arrays is divided 
        # by the number of clients n to compute the average gradient.
        average_arrays = [grad / n for grad in sum_arrays]

        # The list of averaged gradients (in the form of NumPy arrays) is returned.
        return average_arrays
