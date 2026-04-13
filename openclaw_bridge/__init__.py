"""OpenClaw Bridge 完整模块"""

from openclaw_bridge.openclaw_types import (
    ZulongEvent,
    OpenClawEventType,
    OpenClawEventPriority,
    create_user_speech_event,
    create_vision_event,
    create_execute_event,
    create_action_result_event,
    create_speak_event
)

from openclaw_bridge.event_bus_client import (
    EventBusClient,
    EventBusConfig,
    get_event_bus_client
)

from openclaw_bridge.adapters import (
    OpenClawMicAdapter,
    OpenClawVisionReporter
)

from openclaw_bridge.listeners import (
    ExecuteListener,
    SpeakListener
)

__version__ = "1.0.0"
__author__ = "ZULONG Team"

__all__ = [
    # 事件模型
    "ZulongEvent",
    "OpenClawEventType",
    "OpenClawEventPriority",
    "create_user_speech_event",
    "create_vision_event",
    "create_execute_event",
    "create_action_result_event",
    "create_speak_event",
    
    # EventBus 客户端
    "EventBusClient",
    "EventBusConfig",
    "get_event_bus_client",
    
    # 适配器
    "OpenClawMicAdapter",
    "OpenClawVisionReporter",
    
    # 监听器
    "ExecuteListener",
    "SpeakListener"
]
