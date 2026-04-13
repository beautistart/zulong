# File: zulong/perception/sensors_mock.py
# 传感器模拟器 - 第五阶段总装
# 对应 TSD v1.7: 模拟传感器数据

import threading
import time
import random

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent

import logging
logger = logging.getLogger(__name__)


class SensorSimulator:
    """传感器模拟器"""
    
    def __init__(self):
        self._running = False
        self._thread = None
    
    def start(self):
        """启动传感器模拟器"""
        self._running = True
        self._thread = threading.Thread(target=self._simulate, daemon=True)
        self._thread.start()
        logger.info("SensorSimulator started")
    
    def stop(self):
        """停止传感器模拟器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("SensorSimulator stopped")
    
    def _simulate(self):
        """模拟传感器数据"""
        while self._running:
            # 模拟障碍传感器
            self._simulate_obstacle_sensor()
            
            # 模拟运动传感器
            self._simulate_motion_sensor()
            
            # 模拟声音传感器
            self._simulate_sound_sensor()
            
            # 模拟摔倒传感器
            self._simulate_fall_sensor()
            
            # 等待一段时间
            time.sleep(5)
    
    def _simulate_obstacle_sensor(self):
        """模拟障碍传感器"""
        if random.random() < 0.3:  # 30% 概率触发
            distance = random.uniform(0.1, 2.0)
            direction = random.choice(["front", "back", "left", "right"])
            
            event = ZulongEvent(
                type=EventType.SENSOR_OBSTACLE,
                priority=EventPriority.NORMAL,
                source="SensorSimulator",
                payload={"distance": distance, "direction": direction}
            )
            
            logger.debug(f"Simulated obstacle: {distance}m {direction}")
            event_bus.publish(event)
    
    def _simulate_motion_sensor(self):
        """模拟运动传感器"""
        if random.random() < 0.2:  # 20% 概率触发
            detected = random.choice([True, False])
            location = random.choice(["living_room", "kitchen", "bedroom", "hallway"])
            
            event = ZulongEvent(
                type=EventType.SENSOR_MOTION,
                priority=EventPriority.NORMAL,
                source="SensorSimulator",
                payload={"detected": detected, "location": location}
            )
            
            logger.debug(f"Simulated motion: {detected} at {location}")
            event_bus.publish(event)
    
    def _simulate_sound_sensor(self):
        """模拟声音传感器"""
        if random.random() < 0.4:  # 40% 概率触发
            level = random.uniform(30, 80)
            duration = random.uniform(0.5, 2.0)
            
            event = ZulongEvent(
                type=EventType.SENSOR_SOUND,
                priority=EventPriority.NORMAL,
                source="SensorSimulator",
                payload={"level": level, "duration": duration}
            )
            
            logger.debug(f"Simulated sound: {level}dB for {duration}s")
            event_bus.publish(event)
    
    def _simulate_fall_sensor(self):
        """模拟摔倒传感器"""
        if random.random() < 0.05:  # 5% 概率触发
            severity = random.choice(["low", "medium", "high"])
            location = random.choice(["living_room", "kitchen", "bedroom", "hallway"])
            
            event = ZulongEvent(
                type=EventType.SENSOR_FALL,
                priority=EventPriority.CRITICAL,
                source="SensorSimulator",
                payload={"severity": severity, "location": location}
            )
            
            logger.debug(f"Simulated fall: {severity} at {location}")
            event_bus.publish(event)
