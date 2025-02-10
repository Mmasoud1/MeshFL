import sys
from types import ModuleType

# ----- Existing dummy nvflare modules (if not already defined) -----
if "nvflare" not in sys.modules:
    # Create the top-level nvflare module.
    nvflare = ModuleType("nvflare")
    sys.modules["nvflare"] = nvflare

    # Create the nvflare.apis module.
    apis = ModuleType("nvflare.apis")
    sys.modules["nvflare.apis"] = apis

    # Dummy module for nvflare.apis.fl_constant.
    fl_constant = ModuleType("nvflare.apis.fl_constant")
    class DummyFLContextKey:
        CLIENT_NAME = "CLIENT_NAME"
    fl_constant.FLContextKey = DummyFLContextKey
    sys.modules["nvflare.apis.fl_constant"] = fl_constant

    # Dummy module for nvflare.apis.fl_context.
    fl_context = ModuleType("nvflare.apis.fl_context")
    class DummyFLContext:
        pass
    fl_context.FLContext = DummyFLContext
    sys.modules["nvflare.apis.fl_context"] = fl_context

    # Dummy module for nvflare.apis.executor.
    fl_executor = ModuleType("nvflare.apis.executor")
    class DummyExecutor:
        def __init__(self):
            pass
    fl_executor.Executor = DummyExecutor
    sys.modules["nvflare.apis.executor"] = fl_executor

    # Dummy module for nvflare.apis.shareable.
    fl_shareable = ModuleType("nvflare.apis.shareable")
    class DummyShareable(dict):
        pass
    fl_shareable.Shareable = DummyShareable
    sys.modules["nvflare.apis.shareable"] = fl_shareable

    # Dummy module for nvflare.apis.signal.
    fl_signal = ModuleType("nvflare.apis.signal")
    class DummySignal:
        pass
    fl_signal.Signal = DummySignal
    sys.modules["nvflare.apis.signal"] = fl_signal

    # ----- New dummy modules for nvflare.app_common.abstract.aggregator -----
    # Create the nvflare.app_common module.
    app_common = ModuleType("nvflare.app_common")
    sys.modules["nvflare.app_common"] = app_common

    # Create the nvflare.app_common.abstract module.
    abstract = ModuleType("nvflare.app_common.abstract")
    sys.modules["nvflare.app_common.abstract"] = abstract

    # Create the nvflare.app_common.abstract.aggregator module.
    aggregator_mod = ModuleType("nvflare.app_common.abstract.aggregator")
    # Define a dummy Aggregator class.
    class DummyAggregator:
        def __init__(self):
            pass
    aggregator_mod.Aggregator = DummyAggregator
    sys.modules["nvflare.app_common.abstract.aggregator"] = aggregator_mod

