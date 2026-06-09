# File: zulong/l0/__init__.py
"""L0 package exports.

Keep package import side-effect free. Importing zulong.l0.audio or
zulong.l0.devices must not instantiate actuators or subscribe EventBus handlers.
"""

__all__ = ["actuator_simulator"]


def __getattr__(name):
    if name == "actuator_simulator":
        from .actuator_simulator import actuator_simulator

        return actuator_simulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
