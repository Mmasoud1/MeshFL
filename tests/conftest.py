import sys
from types import ModuleType

# ----- Existing dummy modules for nvflare  -----
if "nvflare" not in sys.modules:
    # Top-level nvflare
    nvflare = ModuleType("nvflare")
    sys.modules["nvflare"] = nvflare

    # nvflare.apis module
    apis = ModuleType("nvflare.apis")
    sys.modules["nvflare.apis"] = apis

    # Dummy nvflare.apis.fl_constant
    fl_constant = ModuleType("nvflare.apis.fl_constant")
    class DummyFLContextKey:
        CLIENT_NAME = "CLIENT_NAME"
    fl_constant.FLContextKey = DummyFLContextKey
    sys.modules["nvflare.apis.fl_constant"] = fl_constant

    # Dummy nvflare.apis.fl_context
    fl_context = ModuleType("nvflare.apis.fl_context")
    class DummyFLContext:
        pass
    fl_context.FLContext = DummyFLContext
    sys.modules["nvflare.apis.fl_context"] = fl_context

    # Dummy nvflare.apis.executor
    fl_executor = ModuleType("nvflare.apis.executor")
    class DummyExecutor:
        def __init__(self):
            pass
    fl_executor.Executor = DummyExecutor
    sys.modules["nvflare.apis.executor"] = fl_executor

    # Dummy nvflare.apis.shareable
    fl_shareable = ModuleType("nvflare.apis.shareable")
    class DummyShareable(dict):
        pass
    fl_shareable.Shareable = DummyShareable
    sys.modules["nvflare.apis.shareable"] = fl_shareable

    # Dummy nvflare.apis.signal
    fl_signal = ModuleType("nvflare.apis.signal")
    class DummySignal:
        pass
    fl_signal.Signal = DummySignal
    sys.modules["nvflare.apis.signal"] = fl_signal

    # Dummy nvflare.app_common.abstract.aggregator
    app_common = ModuleType("nvflare.app_common")
    sys.modules["nvflare.app_common"] = app_common
    abstract = ModuleType("nvflare.app_common.abstract")
    sys.modules["nvflare.app_common.abstract"] = abstract
    aggregator_mod = ModuleType("nvflare.app_common.abstract.aggregator")
    class DummyAggregator:
        def __init__(self):
            pass
    aggregator_mod.Aggregator = DummyAggregator
    sys.modules["nvflare.app_common.abstract.aggregator"] = aggregator_mod

# ----- Add dummy modules for nvflare.apis.impl.controller -----

# Create dummy module for nvflare.apis.impl if not already created.
if "nvflare.apis.impl" not in sys.modules:
    impl = ModuleType("nvflare.apis.impl")
    sys.modules["nvflare.apis.impl"] = impl

# Create dummy module for nvflare.apis.impl.controller.
impl_controller = ModuleType("nvflare.apis.impl.controller")
# Define a dummy Controller class.
class DummyController:
    def __init__(self):
        pass
    def log_info(self, fl_ctx, message):
        pass
impl_controller.Controller = DummyController

# Define a dummy Task class.
class DummyTask:
    def __init__(self, name, data, props, timeout, result_received_cb=None):
        self.name = name
        self.data = data
        self.props = props
        self.timeout = timeout
        self.result_received_cb = result_received_cb
impl_controller.Task = DummyTask

# Define a dummy ClientTask class.
class DummyClientTask:
    def __init__(self, result):
        self.result = result
impl_controller.ClientTask = DummyClientTask

sys.modules["nvflare.apis.impl.controller"] = impl_controller

