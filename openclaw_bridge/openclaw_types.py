# File: openclaw_bridge/types.py
"""
OpenClaw Bridge 事件模型定义

对应 TSD v2.2 第 3.1 节（事件模型）
确保 OpenClaw 和祖龙系统说"同一种语言"
"""

from enum import Enum
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


class OpenClawEventType(Enum):
    """OpenClaw 事件类型（映射到祖龙 EventType）"""
    
    # 用户事件
    USER_SPEECH = "USER_SPEECH"  # 用户语音（语音识别输入）
    USER_TEXT = "USER_TEXT"  # 用户文本（Web/键盘输入）
    USER_COMMAND = "USER_COMMAND"  # 用户命令
    
    # 传感器事件
    SENSOR_VISION = "SENSOR_VISION"  # 视觉传感器
    SENSOR_VISION_STATE = "SENSOR_VISION_STATE"  # 视觉状态变更
    
    # 执行事件
    TASK_EXECUTE = "TASK_EXECUTE"  # 任务执行
    ACTION_RESULT = "ACTION_RESULT"  # 动作结果
    
    # 语音事件
    ACTION_SPEAK = "ACTION_SPEAK"  # 语音播报
    L2_OUTPUT = "L2_OUTPUT"  # L2 文本输出（用于 Web 响应）
    
    # 系统事件
    SYSTEM_STATUS = "SYSTEM_STATUS"  # 系统状态


class OpenClawEventPriority(Enum):
    """OpenClaw 事件优先级"""
    LOW = 0  # 低优先级
    NORMAL = 1  # 正常优先级
    HIGH = 2  # 高优先级
    CRITICAL = 3  # 临界优先级（紧急事件）


@dataclass
class ZulongEvent:
    """
    祖龙系统事件模型（OpenClaw 桥接版）
    
    字段说明:
    - id: 事件唯一标识（UUID）
    - type: 事件类型（OpenClawEventType）
    - priority: 事件优先级
    - source: 事件来源（如 "openclaw/mic", "openclaw/camera"）
    - payload: 事件载荷（字典）
    - timestamp: 事件时间戳（Unix 时间）
    """
    type: OpenClawEventType  # 事件类型
    source: str  # 事件源
    payload: Dict[str, Any]  # 事件载荷
    priority: OpenClawEventPriority = OpenClawEventPriority.NORMAL  # 事件优先级
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 事件 ID
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZulongEvent":
        """从字典创建（用于反序列化）"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=OpenClawEventType(data["type"]),
            priority=OpenClawEventPriority(data.get("priority", 1)),
            source=data["source"],
            payload=data["payload"],
            timestamp=data.get("timestamp", time.time())
        )


# 便捷事件创建函数

def create_user_speech_event(text: str, source: str = "openclaw/mic") -> ZulongEvent:
    """
    创建用户语音事件
    
    Args:
        text: 语音文本内容
        source: 事件来源（默认 "openclaw/mic"）
    
    Returns:
        ZulongEvent: 用户语音事件
    """
    return ZulongEvent(
        type=OpenClawEventType.USER_SPEECH,
        source=source,
        payload={"text": text},
        priority=OpenClawEventPriority.NORMAL
    )


def create_vision_event(objects: list, source: str = "openclaw/camera") -> ZulongEvent:
    """
    创建视觉事件
    
    Args:
        objects: 物体列表，每个物体包含 {object_id, name, position, status}
        source: 事件来源（默认 "openclaw/camera"）
    
    Returns:
        ZulongEvent: 视觉事件
    """
    return ZulongEvent(
        type=OpenClawEventType.SENSOR_VISION,
        source=source,
        payload={"objects": objects},
        priority=OpenClawEventPriority.NORMAL
    )


def create_execute_event(action_name: str, arguments: dict) -> ZulongEvent:
    """
    创建执行指令事件
    
    Args:
        action_name: 动作名称（如 "grasp", "move_arm"）
        arguments: 动作参数
    
    Returns:
        ZulongEvent: 执行指令事件
    """
    return ZulongEvent(
        type=OpenClawEventType.TASK_EXECUTE,
        source="l1b/scheduler",
        payload={
            "name": action_name,
            "arguments": arguments
        },
        priority=OpenClawEventPriority.HIGH
    )


def create_action_result_event(action_name: str, success: bool, result: Any = None) -> ZulongEvent:
    """
    创建动作结果事件
    
    Args:
        action_name: 动作名称
        success: 是否成功
        result: 结果数据
    
    Returns:
        ZulongEvent: 动作结果事件
    """
    return ZulongEvent(
        type=OpenClawEventType.ACTION_RESULT,
        source="openclaw/executor",
        payload={
            "action": action_name,
            "success": success,
            "result": result
        },
        priority=OpenClawEventPriority.NORMAL
    )


def create_speak_event(text: str) -> ZulongEvent:
    """
    创建语音播报事件
    
    Args:
        text: 播报文本
    
    Returns:
        ZulongEvent: 语音播报事件
    """
    return ZulongEvent(
        type=OpenClawEventType.ACTION_SPEAK,
        source="l1b/scheduler",
        payload={"text": text},
        priority=OpenClawEventPriority.NORMAL
    )
