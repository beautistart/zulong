# File: zulong/l2/intent_schema.py
# 定义意图分类体系

from pydantic import BaseModel
from enum import Enum


class Intent(Enum):
    """意图枚举"""
    MOVE_FORWARD = "MOVE_FORWARD"
    MOVE_BACKWARD = "MOVE_BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    QUERY_STATUS = "QUERY_STATUS"
    UNKNOWN = "UNKNOWN"


# 支持的意图列表
SUPPORTED_INTENTS = [
    Intent.MOVE_FORWARD.value,
    Intent.MOVE_BACKWARD.value,
    Intent.TURN_LEFT.value,
    Intent.TURN_RIGHT.value,
    Intent.STOP.value,
    Intent.QUERY_STATUS.value,
    Intent.UNKNOWN.value
]


class IntentResult(BaseModel):
    """意图识别结果"""
    intent: str  # 意图标签
    confidence: float  # 置信度
    parameters: dict  # 参数，例如 {"distance": 1.0, "unit": "meter"}
    original_text: str  # 原始文本
