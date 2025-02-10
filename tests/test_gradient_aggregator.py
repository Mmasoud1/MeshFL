import numpy as np
import pytest
import torch
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_context import FLContext
from app.code.aggregator.gradient_aggregator import GradientAggregator
from unittest.mock import MagicMock

# Define dummy classes for FLContext and (if needed) for logging.
class DummyFLContext(FLContext):
    pass

class DummyLogger:
    def log_message(self, message, level=None):
        # Dummy logger that does nothing.
        pass


def test_accept_no_gradients():
    aggregator = GradientAggregator()
    shareable = Shareable()  # Missing "gradients" key
    dummy_fl_ctx = DummyFLContext()
    result = aggregator.accept(shareable, dummy_fl_ctx)
    assert result is False
    assert aggregator.gradients_list == []

def test_accept_empty_gradients():
    aggregator = GradientAggregator()
    shareable = Shareable({"gradients": []})
    dummy_fl_ctx = DummyFLContext()
    result = aggregator.accept(shareable, dummy_fl_ctx)
    assert result is False
    assert aggregator.gradients_list == []

def test_accept_valid_gradients():
    aggregator = GradientAggregator()
    # Create a dummy gradient: a list of numpy arrays.
    gradients = [np.array([1, 2]), np.array([3, 4])]
    shareable = Shareable({"gradients": gradients})
    dummy_fl_ctx = DummyFLContext()
    result = aggregator.accept(shareable, dummy_fl_ctx)
    assert result is True
    # gradients_list should now have one element (the gradients from this shareable)
    assert len(aggregator.gradients_list) == 1
    np.testing.assert_array_equal(aggregator.gradients_list[0][0], np.array([1, 2]))
    np.testing.assert_array_equal(aggregator.gradients_list[0][1], np.array([3, 4]))

def test_average_gradients():
    aggregator = GradientAggregator()
    # Simulating gradients from two clients.
    gradients_client1 = [np.array([1, 2]), np.array([3, 4])]
    gradients_client2 = [np.array([3, 4]), np.array([5, 6])]
    aggregator.gradients_list = [gradients_client1, gradients_client2]
    averaged = aggregator.average_gradients(aggregator.gradients_list)
    # Expected: first gradient: ([1+3]/2 = 2, [2+4]/2 = 3); second: ([3+5]/2 = 4, [4+6]/2 = 5)
    np.testing.assert_array_almost_equal(averaged[0], np.array([2, 3]))
    np.testing.assert_array_almost_equal(averaged[1], np.array([4, 5]))

def test_aggregate():
    aggregator = GradientAggregator()
    dummy_fl_ctx = DummyFLContext()
    # Simulating two clients providing gradients.
    gradients_client1 = [np.array([1, 2]), np.array([3, 4])]
    gradients_client2 = [np.array([3, 4]), np.array([5, 6])]
    shareable1 = Shareable({"gradients": gradients_client1})
    shareable2 = Shareable({"gradients": gradients_client2})
    aggregator.accept(shareable1, dummy_fl_ctx)
    aggregator.accept(shareable2, dummy_fl_ctx)
    # Aggregating the gradients.
    aggregated_shareable = aggregator.aggregate(dummy_fl_ctx)
    # After aggregation, the gradients list is cleared.
    assert aggregator.gradients_list == []
    # The returned shareable must contain key "aggregated_gradients".
    assert "aggregated_gradients" in aggregated_shareable
    aggregated = aggregated_shareable["aggregated_gradients"]
    # Expected averages are the same as in test_average_gradients.
    np.testing.assert_array_almost_equal(aggregated[0], np.array([2, 3]))
    np.testing.assert_array_almost_equal(aggregated[1], np.array([4, 5]))

