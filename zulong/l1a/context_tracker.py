# File: zulong/l1a/context_tracker.py
# 上下文追踪器 - 维护机器人状态

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, ZulongEvent
from typing import Dict, Any
import threading
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[ContextTracker] %(message)s')
logger = logging.getLogger(__name__)


class ContextTracker:
    """上下文追踪器（单例）"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ContextTracker, cls).__new__(cls)
                cls._instance.initialize()
            return cls._instance
    
    def initialize(self):
        """初始化上下文"""
        # 内部维护状态字典
        self._state: Dict[str, Any] = {
            'activity': 'IDLE',  # IDLE, MOVING, CHARGING
            'speed': 0.0,        # 移动速度
            'is_charging': False,  # 是否正在充电
            'last_trigger_time': 0.0  # 上次触发时间（用于去抖）
        }
        # 订阅系统状态事件
        event_bus.subscribe(EventType.SYSTEM_STATUS, self.update_state)
        # 订阅运动传感器事件
        event_bus.subscribe(EventType.SENSOR_MOTION, self.update_state)
        logger.info("ContextTracker initialized")
    
    def update_state(self, event: ZulongEvent):
        """更新内部状态
        
        Args:
            event: 事件对象
        """
        with self._lock:
            if event.type == EventType.SYSTEM_STATUS:
                # 更新系统状态
                status_data = event.payload
                for key, value in status_data.items():
                    if key in self._state:
                        self._state[key] = value
                        logger.debug(f"Updated state: {key} = {value}")
            elif event.type == EventType.SENSOR_MOTION:
                # 更新速度
                speed = event.payload.get('speed', 0.0)
                self._state['speed'] = speed
                # 根据速度更新活动状态
                if speed > 0.1:
                    self._state['activity'] = 'MOVING'
                else:
                    self._state['activity'] = 'IDLE'
                logger.debug(f"Updated motion: speed={speed}, activity={self._state['activity']}")
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前状态快照
        
        Returns:
            状态字典的副本
        """
        with self._lock:
            return self._state.copy()
    
    def set_robot_state(self, activity: str = None, speed: float = None, is_charging: bool = None):
        """设置机器人状态（用于测试）
        
        Args:
            activity: 活动状态
            speed: 速度
            is_charging: 是否充电
        """
        with self._lock:
            if activity is not None:
                self._state['activity'] = activity
            if speed is not None:
                self._state['speed'] = speed
            if is_charging is not None:
                self._state['is_charging'] = is_charging
            logger.info(f"Set robot state: activity={activity}, speed={speed}, is_charging={is_charging}")
    
    def update_last_trigger_time(self):
        """更新上次触发时间"""
        import time
        with self._lock:
            self._state['last_trigger_time'] = time.time()
    
    def get_last_trigger_time(self) -> float:
        """获取上次触发时间"""
        with self._lock:
            return self._state['last_trigger_time']


# 全局上下文追踪器实例
context_tracker = ContextTracker()
