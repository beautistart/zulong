# File: zulong/l1b/power_manager.py
# 电源管理器 - 管理 L2 模型的加载和卸载

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, L2Status, ZulongEvent
from zulong.core.state_manager import state_manager
import threading
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[PowerManager] %(message)s')
logger = logging.getLogger(__name__)


class PowerManager:
    """电源管理器（单例）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PowerManager, cls).__new__(cls)
                cls._instance.initialize()
            return cls._instance
    
    def initialize(self):
        """初始化电源管理器"""
        # 内部状态（仅用于跟踪，实际状态以 StateManager 为准）
        self._l2_status = L2Status.UNLOADED
        self._lock = threading.RLock()
        logger.info(f"PowerManager initialized, L2 status: {self._l2_status.name}")
    
    def wake_up(self):
        """唤醒 L2 模型
        
        Returns:
            bool: 是否成功唤醒
        """
        with self._lock:
            if self._l2_status == L2Status.ACTIVE:
                logger.info("L2 is already ACTIVE, no action needed")
                return True
            
            if self._l2_status == L2Status.LOADING:
                logger.info("L2 is already LOADING, waiting...")
                return False
            
            # 设为 LOADING 状态
            self._l2_status = L2Status.LOADING
            logger.info("L2 status: LOADING")
            
            # 启动后台线程模拟加载
            def load_task():
                try:
                    # 模拟加载过程（1.0 秒）
                    time.sleep(1.0)
                    
                    with self._lock:
                        self._l2_status = L2Status.ACTIVE
                        logger.info("L2 status: ACTIVE")
                    
                    # 同步到全局状态管理器 - 检查是否有 WAITING 状态的任务
                    # 如果有，保持 WAITING 状态；否则设置为 IDLE
                    current_l2 = state_manager.get_l2_status()
                    active_task = state_manager.get_active_task_id()
                    
                    if current_l2 == L2Status.WAITING or (active_task and current_l2 != L2Status.IDLE):
                        # 保持 WAITING 状态，让 Gatekeeper 处理
                        logger.info(f"StateManager L2 status remains: WAITING (active task: {active_task})")
                    else:
                        state_manager.set_l2_status(L2Status.IDLE)
                        logger.info("StateManager L2 status set to: IDLE")
                    
                    # 发布 L2 就绪事件
                    ready_event = ZulongEvent(
                        type=EventType.SYSTEM_L2_READY,
                        priority=EventPriority.NORMAL,
                        source="PowerManager",
                        payload={}
                    )
                    event_bus.publish(ready_event)
                    logger.info("Published SYSTEM_L2_READY event")
                except Exception as e:
                    logger.error(f"Error during L2 loading: {e}")
                    with self._lock:
                        self._l2_status = L2Status.UNLOADED
            
            threading.Thread(target=load_task, daemon=True).start()
            return True
    
    def enter_silent(self):
        """进入安静模式，卸载 L2 模型
        
        Returns:
            bool: 是否成功卸载
        """
        with self._lock:
            if self._l2_status == L2Status.UNLOADED:
                logger.info("L2 is already UNLOADED, no action needed")
                return True
            
            # 设为 UNLOADED 状态（模拟释放显存）
            self._l2_status = L2Status.UNLOADED
            logger.info("L2 status: UNLOADED")
            
            # 同步到全局状态管理器
            state_manager.set_l2_status(L2Status.UNLOADED)
            logger.info("StateManager L2 status set to: UNLOADED")
            
            # 发布 L2 卸载事件
            unloaded_event = ZulongEvent(
                type=EventType.SYSTEM_L2_UNLOADED,
                priority=EventPriority.NORMAL,
                source="PowerManager",
                payload={}
            )
            event_bus.publish(unloaded_event)
            logger.info("Published SYSTEM_L2_UNLOADED event")
            return True
    
    def is_ready(self) -> bool:
        """检查 L2 是否就绪
        
        Returns:
            bool: L2 是否处于 ACTIVE 状态
        """
        with self._lock:
            return self._l2_status == L2Status.ACTIVE
    
    def get_l2_status(self) -> L2Status:
        """获取 L2 状态
        
        Returns:
            L2Status: 当前 L2 状态
        """
        with self._lock:
            return self._l2_status


# 全局电源管理器实例
power_manager = PowerManager()
