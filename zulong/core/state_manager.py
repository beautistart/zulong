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
            self._power_state = PowerState.ACTIVE
            self._l2_status = L2Status.IDLE
            self._active_task_id = None
            self._interrupt_flag = False
            self._context = {}
            self._lock = threading.Lock()
            self._last_activity_time = 0.0
            self._fc_loop_running = False
            self._initialized = True
            import time
            self.touch_activity()
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
            
            if status == L2Status.IDLE and self._active_task_id is not None:
                if task_id and task_id == self._active_task_id:
                    self._active_task_id = None
                    logger.info(f"Task {task_id} completed, cleared active_task_id.")
                else:
                    self._l2_status = L2Status.WAITING
                    logger.warning(f"Task {self._active_task_id} is suspended (different from completed {task_id}). Status forced to WAITING.")
    
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
    
    def set_interrupt_flag(self, flag: bool):
        """设置中断标志
        
        供 InterruptHandler 使用，在生成循环中通过 get_interrupt_flag() 检测中断。
        
        Args:
            flag: True 表示请求中断，False 表示清除中断
        """
        with self._lock:
            self._interrupt_flag = flag
            if flag:
                logger.info("Interrupt flag set to TRUE")
            else:
                logger.debug("Interrupt flag cleared")
    
    def get_interrupt_flag(self) -> bool:
        """获取中断标志
        
        Returns:
            bool: 当前中断标志状态
        """
        with self._lock:
            return self._interrupt_flag
    
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

    def touch_activity(self):
        """更新最后活动时间（任何用户交互或FC循环活动时调用）"""
        import time
        with self._lock:
            self._last_activity_time = time.time()

    def get_idle_seconds(self) -> float:
        """获取系统空闲秒数"""
        import time
        with self._lock:
            if self._last_activity_time == 0.0:
                import time as _t
                self._last_activity_time = _t.time()
            return time.time() - self._last_activity_time

    def is_system_idle(self, threshold: float = 300.0) -> bool:
        """判断系统是否空闲（超过阈值时间无用户活动即视为空闲，不检查任务状态）
        
        FC循环可能在长时间等待工具结果，此时虽L2=BUSY但无用户活动，
        应视为空闲以允许节点审批利用等待时间。
        """
        import time
        with self._lock:
            if self._last_activity_time == 0.0:
                self._last_activity_time = time.time()
            return (time.time() - self._last_activity_time) >= threshold
    
    def is_fc_loop_running(self) -> bool:
        """判断FC循环是否正在运行
        
        用于节点审查循环判断是否应该暂停提交审查任务。
        当FC循环运行时，禁止节点审查提交以避免竞争LLM资源。
        
        Returns:
            bool: True表示FC循环正在运行，False表示已停止
        """
        with self._lock:
            return self._fc_loop_running
    
    def set_fc_loop_running(self, running: bool):
        """设置FC循环运行状态
        
        Args:
            running: True表示FC循环启动，False表示停止
        """
        with self._lock:
            old_state = self._fc_loop_running
            self._fc_loop_running = running
            if old_state != running:
                logger.info(f"FC loop running state changed: {old_state} -> {running}")


# 全局状态管理器实例
state_manager = StateManager()
