# File: zulong/l1a/reflex_controller.py
# L1-A 反射控制器 - 第五阶段总装
# 对应 TSD v1.7: 智能感知与受控反射
# 集成 AT-03: L1-A反射层日志扩展

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
from zulong.core.state_manager import state_manager
from zulong.replay.l1a_logger import get_l1a_logger

import logging
logger = logging.getLogger(__name__)


class ReflexController:
    """L1-A 反射控制器"""
    
    def __init__(self):
        """初始化反射控制器"""
        self.l1a_logger = get_l1a_logger()
        self._register_event_handlers()
        logger.info("L1-A ReflexController initialized and subscribed to sensor events")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        event_bus.subscribe(EventType.SENSOR_OBSTACLE, self.on_obstacle, "L1-A")
        event_bus.subscribe(EventType.SENSOR_MOTION, self.on_motion, "L1-A")
        event_bus.subscribe(EventType.SENSOR_SOUND, self.on_sound, "L1-A")
        event_bus.subscribe(EventType.SENSOR_FALL, self.on_fall, "L1-A")
        event_bus.subscribe(EventType.SENSOR_VISION, self.on_vision, "L1-A")
    
    def evaluate_reflex(self, event: ZulongEvent, context: str) -> str:
        """评估反射反应
        
        Args:
            event: 传感器事件
            context: 当前上下文
            
        Returns:
            str: 反射命令，None 表示抑制
        """
        if context == 'CHARGING' and event.type == EventType.SENSOR_OBSTACLE:
            logger.info(f"Reflex Suppressed (Context: {context})")
            return None
        
        if context == 'MOVING_FAST' and event.type == EventType.SENSOR_OBSTACLE:
            logger.info(f"Reflex Triggered (Context: {context}) - STOP_COMMAND")
            return "STOP_COMMAND"
        
        if event.type == EventType.SENSOR_FALL:
            logger.info(f"Reflex Triggered - EMERGENCY_STOP")
            return "EMERGENCY_STOP"
        
        if event.type == EventType.SENSOR_SOUND:
            level = event.payload.get("level", 0)
            if level > 70:
                logger.info(f"Reflex Triggered - ALERT")
                return "ALERT"
        
        return None
    
    def on_obstacle(self, event: ZulongEvent):
        """处理障碍传感器事件"""
        self._log_sensor_event(event)
        context = self._get_current_context()
        command = self.evaluate_reflex(event, context)
        if command:
            self._execute_reflex(command, event)
    
    def on_motion(self, event: ZulongEvent):
        """处理运动传感器事件"""
        self._log_sensor_event(event)
        context = self._get_current_context()
        command = self.evaluate_reflex(event, context)
        if command:
            self._execute_reflex(command, event)
    
    def on_sound(self, event: ZulongEvent):
        """处理声音传感器事件 (AT-03: VAD日志)"""
        level = event.payload.get("level", 0)
        duration_ms = event.payload.get("duration_ms", 0)
        
        self.l1a_logger.log_vad_trigger(
            confidence=min(1.0, level / 100.0),
            energy=level / 100.0,
            duration_ms=duration_ms
        )
        
        context = self._get_current_context()
        command = self.evaluate_reflex(event, context)
        if command:
            self._execute_reflex(command, event)
    
    def on_fall(self, event: ZulongEvent):
        """处理摔倒传感器事件"""
        self._log_sensor_event(event)
        context = self._get_current_context()
        command = self.evaluate_reflex(event, context)
        if command:
            self._execute_reflex(command, event)
    
    def on_vision(self, event: ZulongEvent):
        """处理视觉传感器事件 (AT-03: 视觉检测日志)"""
        payload = event.payload
        
        self.l1a_logger.log_vision_detection(
            object_type=payload.get("object_type", "unknown"),
            confidence=payload.get("confidence", 0.0),
            bbox=payload.get("bbox", [0, 0, 0, 0]),
            position_3d=payload.get("position_3d")
        )
        
        context = self._get_current_context()
        command = self.evaluate_reflex(event, context)
        if command:
            self._execute_reflex(command, event)
    
    def _log_sensor_event(self, event: ZulongEvent):
        """记录传感器事件 (AT-03)"""
        self.l1a_logger.log_sensor_data(
            sensor_type=event.type.name,
            data=event.payload
        )
    
    def _get_current_context(self) -> str:
        """获取当前上下文"""
        return "NORMAL"
    
    def _execute_reflex(self, command: str, event: ZulongEvent):
        """执行反射命令 - 直接发布到 L0 执行器 (AT-03: 反射日志)"""
        reflex_id = self.l1a_logger.log_reflex_triggered(
            reflex_type=command,
            trigger_source=event.type.name,
            action_taken=command
        )
        
        if command == "EMERGENCY_STOP":
            cmd_event = ZulongEvent(
                type=EventType.CMD_EMERGENCY_STOP,
                priority=EventPriority.CRITICAL,
                source="L1-A-Reflex",
                payload={
                    "reason": f"Reflex: {event.type.name}",
                    "event_payload": event.payload,
                    "reflex_id": reflex_id
                }
            )
            logger.info(f"Executing reflex: {command} -> CMD_EMERGENCY_STOP")
            event_bus.publish(cmd_event)
        
        elif command == "ALERT":
            cmd_event = ZulongEvent(
                type=EventType.CMD_BRAKE,
                priority=EventPriority.HIGH,
                source="L1-A-Reflex",
                payload={
                    "reason": f"Reflex: {event.type.name}",
                    "event_payload": event.payload,
                    "reflex_id": reflex_id
                }
            )
            logger.info(f"Executing reflex: {command} -> CMD_BRAKE")
            event_bus.publish(cmd_event)
        
        elif command == "STOP_COMMAND":
            cmd_event = ZulongEvent(
                type=EventType.CMD_BRAKE,
                priority=EventPriority.HIGH,
                source="L1-A-Reflex",
                payload={
                    "reason": f"Reflex: {event.type.name}",
                    "event_payload": event.payload,
                    "reflex_id": reflex_id
                }
            )
            logger.info(f"Executing reflex: {command} -> CMD_BRAKE")
            event_bus.publish(cmd_event)
        
        reflex_event = ZulongEvent(
            type=EventType.SYSTEM_REFLEX,
            priority=EventPriority.HIGH,
            source="L1-A-Reflex",
            payload={
                "command": command,
                "event_type": event.type.name,
                "event_payload": event.payload,
                "reflex_id": reflex_id
            }
        )
        event_bus.publish(reflex_event)


reflex_controller = ReflexController()
