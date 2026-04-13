# File: zulong/core/state_manager.py
# 状态管理器 - 第五阶段总装
# 对应 TSD v1.7: 增强版全局状态

import threading
from typing import Dict, Any, Optional

from zulong.core.types import PowerState, L2Status

import logging
logger = logging.getLogger(__name__)


class StateManager:
    """状态管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化状态管理器"""
        if not hasattr(self, '_initialized'):
            self._power_state = PowerState.ACTIVE  # 默认活跃状态
            self._l2_status = L2Status.IDLE  # 默认空闲状态
            self._active_task_id = None  # 记录当前是否有挂起的任务ID
            self._context = {}
            self._lock = threading.Lock()
            self._initialized = True
            logger.info("StateManager initialized")
    
    def get_power_state(self) -> PowerState:
        """获取电源状态
        
        Returns:
            PowerState: 当前电源状态
        """
        with self._lock:
            return self._power_state
    
    def set_power_state(self, state: PowerState):
        """设置电源状态
        
        Args:
            state: 电源状态
        """
        with self._lock:
            self._power_state = state
            logger.info(f"Power state changed to: {state.name}")
    
    def get_l2_status(self) -> L2Status:
        """获取 L2 状态
        
        Returns:
            L2Status: 当前 L2 状态
        """
        with self._lock:
            return self._l2_status
    
    def set_l2_status(self, status: L2Status, task_id: str = None):
        """设置 L2 状态
        
        Args:
            status: L2 状态
            task_id: 任务 ID
        """
        with self._lock:
            old_status = self._l2_status
            self._l2_status = status
            if task_id:
                self._active_task_id = task_id
            
            logger.info(f"L2 status changed to: {status.name} (Task: {task_id})")
            
            # 关键逻辑：如果是分段任务完成，进入 WAITING 而不是 IDLE
            if status == L2Status.IDLE and self._active_task_id is not None:
                # 如果还有任务ID挂着，说明任务没做完，只是暂停了
                self._l2_status = L2Status.WAITING
                logger.warning(f"⚠️  Task {self._active_task_id} is suspended. Status forced to WAITING.")
    
    def clear_task(self):
        """当任务彻底完成或取消时调用"""
        with self._lock:
            self._active_task_id = None
            self._l2_status = L2Status.IDLE
            logger.info("Task cleared. Status set to IDLE.")
    
    def get_effective_status(self):
        """
        【核心工具函数】供 Gatekeeper 使用
        将 WAITING 视为 BUSY 处理
        """
        with self._lock:
            if self._l2_status in [L2Status.BUSY, L2Status.WAITING]:
                return "ACTIVE_TASK"  # 统一视为“有任务在身”
            return "IDLE"
    
    def get_active_task_id(self) -> Optional[str]:
        """获取当前活跃任务 ID
        
        Returns:
            Optional[str]: 活跃任务 ID
        """
        with self._lock:
            return self._active_task_id
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文值
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            Any: 上下文值
        """
        with self._lock:
            return self._context.get(key, default)
    
    def set_context(self, key: str, value: Any):
        """设置上下文值
        
        Args:
            key: 键
            value: 值
        """
        with self._lock:
            self._context[key] = value
    
    def clear_context(self):
        """清空上下文"""
        with self._lock:
            self._context.clear()


# 全局状态管理器实例
state_manager = StateManager()
