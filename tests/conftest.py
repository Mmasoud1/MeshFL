import sys
from types import ModuleType

# If nvflare is not installed, create dummy modules.
if "nvflare" not in sys.modules:
    # Create the top-level nvflare module.
    nvflare = ModuleType("nvflare")
    sys.modules["nvflare"] = nvflare

    # Create the nvflare.apis module.
    apis = ModuleType("nvflare.apis")
    sys.modules["nvflare.apis"] = apis

    # Create dummy module for nvflare.apis.fl_constant.
    fl_constant = ModuleType("nvflare.apis.fl_constant")
    # Define a dummy FLContextKey with a CLIENT_NAME attribute.
    class DummyFLContextKey:
        CLIENT_NAME = "CLIENT_NAME"
    fl_constant.FLContextKey = DummyFLContextKey
    sys.modules["nvflare.apis.fl_constant"] = fl_constant

    # Create dummy module for nvflare.apis.fl_context.
    fl_context = ModuleType("nvflare.apis.fl_context")
    class DummyFLContext:
        pass
    fl_context.FLContext = DummyFLContext
    sys.modules["nvflare.apis.fl_context"] = fl_context

    # Create dummy module for nvflare.apis.executor.
    fl_executor = ModuleType("nvflare.apis.executor")
    # Define a dummy Executor base class.
    class DummyExecutor:
        def __init__(self):
            pass
    fl_executor.Executor = DummyExecutor
    sys.modules["nvflare.apis.executor"] = fl_executor

    # Create dummy module for nvflare.apis.shareable.
    fl_shareable = ModuleType("nvflare.apis.shareable")
    # Define a dummy Shareable class (here we simply subclass dict).
    class DummyShareable(dict):
        pass
    fl_shareable.Shareable = DummyShareable
    sys.modules["nvflare.apis.shareable"] = fl_shareable

    # Optionally, create dummy module for nvflare.apis.signal if needed.
    fl_signal = ModuleType("nvflare.apis.signal")
    class DummySignal:
        pass
    fl_signal.Signal = DummySignal
    sys.modules["nvflare.apis.signal"] = fl_signal

