# File: zulong/core/event_bus.py
# 事件总线 - 第五阶段总装
# 对应 TSD v1.7: 增强版事件总线

import threading
import time
from typing import Dict, List, Callable, Optional

from zulong.core.types import ZulongEvent, EventType, PowerState, L2Status
from zulong.core.state_manager import state_manager

import logging
logger = logging.getLogger(__name__)


class EventBus:
    """事件总线"""
    
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
        """初始化事件总线"""
        if not hasattr(self, '_initialized'):
            self._subscribers = {}
            self._event_queue = []
            self._running = False
            self._thread = None
            self._lock = threading.Lock()
            self._initialized = True
            self._start_dispatch_thread()
            logger.info("EventBus initialized with background dispatch thread")
    
    def _start_dispatch_thread(self):
        """启动事件分发线程"""
        self._running = True
        self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._thread.start()
    
    def _dispatch_loop(self):
        """事件分发循环"""
        while self._running:
            if self._event_queue:
                event = self._event_queue.pop(0)
                self._dispatch_event(event)
            else:
                time.sleep(0.01)  # 避免忙等
    
    def subscribe(self, event_type: EventType, handler: Callable, subscriber: str):
        """订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            subscriber: 订阅者名称
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append((handler, subscriber))
        logger.debug(f"Subscriber {subscriber} subscribed to {event_type.name}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """取消订阅
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [(h, s) for h, s in self._subscribers[event_type] if h != handler]
    
    def publish(self, event: ZulongEvent):
        """发布事件
        
        Args:
            event: 事件对象
        """
        # 流式输出事件静默处理，不输出日志
        if event.type != EventType.L2_OUTPUT_STREAM:
            logger.debug(f"\n{'='*80}")
            logger.debug(f"📡 [EventBus] 发布事件！")
            logger.debug(f"📡 [EventBus] 事件类型：{event.type}")
            logger.debug(f"📡 [EventBus] 事件优先级：{event.priority}")
            logger.debug(f"📡 [EventBus] 事件来源：{event.source}")
            logger.debug(f"📡 [EventBus] Payload: {event.payload}")
            logger.debug(f"{'='*80}\n")
        
        # 🎯 [核心架构] 所有事件统一路由到 L1-B (TSD v1.7 增强版)
        # L1-B Gatekeeper 负责所有事件的判断、过滤、优先级排序和转发
        # EventBus 不再判断 L2 状态，不再做路由决策
        
        # 特殊事件类型处理
        if event.type == EventType.INTERACTION_TRIGGER:
            # 交互触发事件 -> 路由给 L1-B
            logger.info(f"📡 [EventBus] INTERACTION_TRIGGER 事件 -> 路由给 L1-B")
            self._route_to_l1b(event)
            return
        elif event.type == EventType.DIRECT_WAKEUP:
            # 直接唤醒事件 -> 路由给 L1-B
            logger.info(f"📡 [EventBus] DIRECT_WAKEUP 事件 -> 路由给 L1-B")
            self._route_to_l1b(event)
            return
        
        # 用户事件统一路由到 L1-B
        if event.type in [EventType.USER_SPEECH, EventType.USER_VOICE, EventType.USER_COMMAND, EventType.USER_TEXT]:
            logger.info(f"📡 [EventBus] 用户事件 {event.type.name} -> 路由给 L1-B")
            self._route_to_l1b(event)
            return
        
        # 传感器事件路由到 L1-B
        if event.type in [EventType.SENSOR_VISION, EventType.SENSOR_VISION_STATE, 
                          EventType.SENSOR_VIDEO_MOTION, EventType.SENSOR_VIDEO_FRAME,
                          EventType.SENSOR_OBSTACLE, EventType.SENSOR_MOTION, 
                          EventType.SENSOR_SOUND, EventType.SENSOR_FALL]:
            logger.info(f"📡 [EventBus] 传感器事件 {event.type.name} -> 路由给 L1-B")
            self._route_to_l1b(event)
            return
        
        # 流式输出事件直接分发，不需要进入队列（静默处理，不输出日志）
        if event.type == EventType.L2_OUTPUT_STREAM:
            self._dispatch_event(event)
            return
        
        # 系统事件放入队列，由后台线程分发
        with self._lock:
            self._event_queue.append(event)
        logger.debug(f"📡 [EventBus] 系统事件 {event.type.name} 放入队列等待分发")
    
    def _route_to_l1b(self, event: ZulongEvent):
        """路由事件到 L1-B
        
        Args:
            event: 事件对象
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📡 [EventBus] _route_to_l1b 被调用")
        logger.info(f"📡 [EventBus] 事件类型：{event.type}")
        logger.info(f"{'='*80}\n")
        
        # 直接分发给 L1-B 订阅者
        if event.type in self._subscribers:
            logger.info(f"📡 [EventBus] 找到 {len(self._subscribers[event.type])} 个订阅者")
            for handler, subscriber in self._subscribers[event.type]:
                logger.info(f"📡 [EventBus]   检查订阅者：{subscriber}")
                if "L1-B" in subscriber or "Gatekeeper" in subscriber:
                    logger.info(f"📡 [EventBus]   ✅ 匹配 L1-B，调用处理器：{handler.__name__}")
                    try:
                        handler(event)
                        logger.info(f"📡 [EventBus]   ✅ L1-B 处理器执行完成")
                    except Exception as e:
                        logger.error(f"❌ [EventBus] Error in L1-B handler: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    logger.info(f"📡 [EventBus]   ❌ 不匹配 L1-B，跳过")
        else:
            logger.warning(f"📡 [EventBus] 事件 {event.type.name} 没有订阅者")
    
    def _route_to_l2(self, event: ZulongEvent):
        """路由事件到 L2
        
        Args:
            event: 事件对象
        """
        # 直接分发给 L2 订阅者
        if event.type in self._subscribers:
            for handler, subscriber in self._subscribers[event.type]:
                if "L2" in subscriber or "InferenceEngine" in subscriber:
                    try:
                        handler(event)
                    except Exception as e:
                        logger.error(f"Error in L2 handler: {e}")
    
    def _dispatch_event(self, event: ZulongEvent):
        """分发事件
        
        Args:
            event: 事件对象
        """
        if event.type in self._subscribers:
            for handler, subscriber in self._subscribers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"❌ [EventBus] Error in {subscriber} handler: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        else:
            logger.warning(f"📡 [EventBus] 事件 {event.type.name} 没有订阅者")
    
    def stop(self):
        """停止事件总线"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("EventBus stopped")


# 全局事件总线实例
event_bus = EventBus()
