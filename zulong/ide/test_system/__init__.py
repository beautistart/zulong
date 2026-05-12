__all__ = ["TestSystemApp"]

def __getattr__(name):
    if name == "TestSystemApp":
        from .app import TestSystemApp
        return TestSystemApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
