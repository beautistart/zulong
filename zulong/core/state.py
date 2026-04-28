# File: zulong/core/state.py
# 全局状态管理

from .types import PowerState, L2Status
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class GlobalState:
    """全局状态"""
    power_state: PowerState = PowerState.ACTIVE  # 电源状态
    l2_status: L2Status = L2Status.IDLE  # L2 中枢状态
    context: Dict[str, Any] = field(default_factory=dict)  # 系统上下文
    interrupt_flag: bool = False  # 中断标志


class StateManager:
    """状态管理器（单例模式）"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance
    
    def initialize(self):
        """初始化状态"""
        self.global_state = GlobalState()
    
    def get_power_state(self) -> PowerState:
        """获取电源状态"""
        return self.global_state.power_state
    
    def set_power_state(self, state: PowerState):
        """设置电源状态"""
        self.global_state.power_state = state
    
    def get_l2_status(self) -> L2Status:
        """获取 L2 状态"""
        return self.global_state.l2_status
    
    def set_l2_status(self, status: L2Status):
        """设置 L2 状态"""
        self.global_state.l2_status = status
    
    def get_context(self) -> Dict[str, Any]:
        """获取上下文"""
        return self.global_state.context
    
    def update_context(self, key: str, value: Any):
        """更新上下文"""
        self.global_state.context[key] = value
    
    def set_interrupt_flag(self, flag: bool):
        """设置中断标志"""
        self.global_state.interrupt_flag = flag
    
    def get_interrupt_flag(self) -> bool:
        """获取中断标志"""
        return self.global_state.interrupt_flag


# 全局状态管理器实例
state_manager = StateManager()
