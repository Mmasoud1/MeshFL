import sys
from types import ModuleType

# Check if nvflare is already installed; if not, create dummy modules.
if "nvflare" not in sys.modules:
    # Create the nvflare module.
    nvflare = ModuleType("nvflare")
    apis = ModuleType("nvflare.apis")
    fl_constant = ModuleType("nvflare.apis.fl_constant")
    fl_context = ModuleType("nvflare.apis.fl_context")

    # Create a dummy FLContextKey with the attribute CLIENT_NAME.
    class DummyFLContextKey:
        CLIENT_NAME = "CLIENT_NAME"
    fl_constant.FLContextKey = DummyFLContextKey

    # Create a dummy FLContext class (if needed by your tests).
    class DummyFLContext:
        pass
    fl_context.FLContext = DummyFLContext

    # Register these dummy modules in sys.modules.
    sys.modules["nvflare"] = nvflare
    sys.modules["nvflare.apis"] = apis
    sys.modules["nvflare.apis.fl_constant"] = fl_constant
    sys.modules["nvflare.apis.fl_context"] = fl_context

