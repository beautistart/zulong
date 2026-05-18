"""
LLM自主注意力模式选择 - 类型定义模块

定义压力检测、决策请求/响应、切换记录等核心数据类型
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime


class PressureTrend(Enum):
    """压力趋势枚举"""
    RISING = "rising"      # 压力上升中
    STABLE = "stable"      # 压力稳定
    FALLING = "falling"    # 压力下降中


class TriggerType(Enum):
    """触发类型枚举"""
    TOOL_DRIVEN = "tool_driven"          # 工具驱动触发
    LLM_AUTONOMOUS = "llm_autonomous"    # LLM自主触发
    FALLBACK = "fallback"                # Fallback触发


class OscillationLevel(Enum):
    """震荡级别枚举"""
    NONE = "none"        # 无震荡
    SLIGHT = "slight"    # 轻微震荡
    OBVIOUS = "obvious"  # 明显震荡
    SEVERE = "severe"    # 严重震荡


@dataclass
class PressureMetrics:
    """压力指标数据类"""
    current_pressure: float              # 当前压力值 (tokens/budget)
    pressure_trend: PressureTrend        # 压力趋势
    pressure_velocity: float             # 压力变化速率
    predicted_pressure_5s: float         # 预测5秒后压力值
    timestamp: datetime = field(default_factory=datetime.now)
    
    budget_usage: float = 0.0            # 预算使用率
    message_count: int = 0               # 消息数量
    token_density: float = 0.0           # token密度
    
    def __post_init__(self):
        if self.current_pressure < 0:
            self.current_pressure = 0.0
        if self.current_pressure > 999.0:
            self.current_pressure = 999.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_pressure": self.current_pressure,
            "pressure_trend": self.pressure_trend.value,
            "pressure_velocity": self.pressure_velocity,
            "predicted_pressure_5s": self.predicted_pressure_5s,
            "timestamp": self.timestamp.isoformat(),
            "budget_usage": self.budget_usage,
            "message_count": self.message_count,
            "token_density": self.token_density,
        }


@dataclass
class DecisionRequest:
    """决策请求数据类"""
    pressure_metrics: PressureMetrics              # 压力指标
    current_mode: str                              # 当前注意力模式
    task_context: str                              # 任务上下文摘要
    mode_options: List[str] = field(default_factory=list)  # 可选模式列表
    switch_history: List[Dict] = field(default_factory=list)  # 切换历史
    prompt: str = ""                               # 构建的Prompt
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pressure_metrics": self.pressure_metrics.to_dict(),
            "current_mode": self.current_mode,
            "task_context": self.task_context,
            "mode_options": self.mode_options,
            "switch_history": self.switch_history,
            "prompt": self.prompt,
        }


@dataclass
class DecisionResponse:
    """决策响应数据类"""
    mode: str                          # LLM选择的模式
    reason: str                        # 选择理由
    confidence: float = 0.5            # 置信度 (0-1)
    is_fallback: bool = False          # 是否为Fallback响应
    
    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        if self.confidence > 1.0:
            self.confidence = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "confidence": self.confidence,
            "is_fallback": self.is_fallback,
        }


@dataclass
class SwitchRecord:
    """切换记录数据类"""
    old_mode: str                      # 切换前模式
    new_mode: str                      # 切换后模式
    trigger_type: TriggerType          # 触发类型
    pressure_at_switch: float          # 切换时压力值
    timestamp: datetime = field(default_factory=datetime.now)
    reason: str = ""                   # 切换原因
    confidence: float = 0.5            # LLM决策置信度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_mode": self.old_mode,
            "new_mode": self.new_mode,
            "trigger_type": self.trigger_type.value,
            "pressure_at_switch": self.pressure_at_switch,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class OscillationState:
    """震荡状态数据类"""
    is_oscillating: bool = False                # 是否震荡
    oscillation_level: OscillationLevel = OscillationLevel.NONE  # 震荡级别
    oscillation_pattern: str = ""               # 震荡模式 (如 "ABA", "ABAB")
    adjusted_cooldown_factor: float = 1.0       # 调整后的冷却因子
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_oscillating": self.is_oscillating,
            "oscillation_level": self.oscillation_level.value,
            "oscillation_pattern": self.oscillation_pattern,
            "adjusted_cooldown_factor": self.adjusted_cooldown_factor,
        }


@dataclass
class ThresholdCheckResult:
    """阈值检查结果数据类"""
    should_trigger: bool             # 是否应触发LLM选择
    trigger_type: str                # 触发类型描述
    message: str                     # 检查结果消息
    pressure_value: float = 0.0      # 触发时的压力值
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_trigger": self.should_trigger,
            "trigger_type": self.trigger_type,
            "message": self.message,
            "pressure_value": self.pressure_value,
        }


@dataclass
class CooldownCheckResult:
    """冷却检查结果数据类"""
    is_allowed: bool                 # 是否允许切换
    remaining_time: float            # 剩余冷却时间 (秒)
    message: str = ""                # 检查结果消息
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "remaining_time": self.remaining_time,
            "message": self.message,
        }


@dataclass
class ModeSwitchResult:
    """模式切换结果数据类"""
    success: bool                    # 切换是否成功
    new_mode: str                    # 新模式
    switched: bool                   # 是否实际发生切换
    message: str = ""                # 结果消息
    switch_record: Optional[SwitchRecord] = None  # 切换记录
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "new_mode": self.new_mode,
            "switched": self.switched,
            "message": self.message,
            "switch_record": self.switch_record.to_dict() if self.switch_record else None,
        }
