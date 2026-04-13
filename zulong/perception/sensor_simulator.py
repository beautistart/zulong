# File: zulong/perception/sensor_simulator.py
# 模拟感知节点 - 生成测试事件

from zulong.core.event_bus import event_bus
from zulong.core.types import EventType, EventPriority, ZulongEvent
import threading
import time
import random
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='[SensorSimulator] %(message)s')
logger = logging.getLogger(__name__)


class SensorSimulator:
    """传感器模拟器"""
    
    def __init__(self):
        self._running = False
        self._thread = None
    
    def start(self):
        """启动模拟器"""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._simulate, daemon=True)
            self._thread.start()
            logger.info("Sensor simulator started")
    
    def stop(self):
        """停止模拟器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Sensor simulator stopped")
    
    def set_robot_state(self, activity: str = None, speed: float = None, is_charging: bool = None):
        """设置机器人状态（用于测试）
        
        Args:
            activity: 活动状态
            speed: 速度
            is_charging: 是否充电
        """
        # 发布系统状态事件
        status_data = {}
        if activity is not None:
            status_data['activity'] = activity
        if speed is not None:
            status_data['speed'] = speed
        if is_charging is not None:
            status_data['is_charging'] = is_charging
        
        if status_data:
            event = ZulongEvent(
                type=EventType.SYSTEM_STATUS,
                priority=EventPriority.NORMAL,
                source="simulator",
                payload=status_data
            )
            event_bus.publish(event)
            logger.info(f"Set robot state: {status_data}")
    
    def _simulate(self):
        """模拟传感器事件"""
        user_speech_counter = 0
        while self._running:
            try:
                # 每隔 5 秒生成一个 USER_SPEECH 事件
                if user_speech_counter % 5 == 0:
                    self._generate_user_speech()
                
                # 随机生成 SENSOR_OBSTACLE 事件
                if random.random() < 0.1:  # 10% 概率
                    self._generate_obstacle()
                
                # 随机生成 SENSOR_MOTION 事件
                if random.random() < 0.15:  # 15% 概率
                    self._generate_motion()
                
                # 随机生成 SENSOR_FALL 事件 (低概率)
                if random.random() < 0.01:  # 1% 概率
                    self._generate_fall()
                
                # 随机生成 SENSOR_IMPACT 事件
                if random.random() < 0.05:  # 5% 概率
                    self._generate_impact()
                
                # 睡眠 1 秒，模拟传感器采样间隔
                # 这不是轮询，而是模拟真实传感器的采样频率
                time.sleep(1)
                user_speech_counter += 1
            except Exception as e:
                logger.error(f"Error in sensor simulation: {e}")
                time.sleep(1)
    
    def _generate_user_speech(self):
        """生成用户语音事件"""
        texts = ["你好", "祖龙", "帮我拿杯水", "今天天气怎么样", "安静"]
        text = random.choice(texts)
        event = ZulongEvent(
            type=EventType.USER_SPEECH,
            priority=EventPriority.NORMAL,
            source="simulator",
            payload={"text": text}
        )
        event_bus.publish(event)
        logger.debug(f"Generated USER_SPEECH: {text}")
    
    def _generate_obstacle(self):
        """生成障碍物事件"""
        distance = random.uniform(0.1, 2.0)
        event = ZulongEvent(
            type=EventType.SENSOR_OBSTACLE,
            priority=EventPriority.HIGH if distance < 0.5 else EventPriority.NORMAL,
            source="simulator",
            payload={"distance": distance}
        )
        event_bus.publish(event)
        logger.debug(f"Generated SENSOR_OBSTACLE: {distance:.2f}m")
    
    def _generate_motion(self):
        """生成运动事件"""
        speed = random.uniform(0.0, 1.0)
        event = ZulongEvent(
            type=EventType.SENSOR_MOTION,
            priority=EventPriority.NORMAL,
            source="simulator",
            payload={"speed": speed}
        )
        event_bus.publish(event)
        logger.debug(f"Generated SENSOR_MOTION: {speed:.2f}m/s")
    
    def _generate_fall(self):
        """生成摔倒事件"""
        event = ZulongEvent(
            type=EventType.SENSOR_FALL,
            priority=EventPriority.CRITICAL,
            source="simulator",
            payload={}
        )
        event_bus.publish(event)
        logger.debug("Generated SENSOR_FALL")
    
    def _generate_impact(self):
        """生成碰撞事件"""
        event = ZulongEvent(
            type=EventType.SENSOR_IMPACT,
            priority=EventPriority.HIGH,
            source="simulator",
            payload={}
        )
        event_bus.publish(event)
        logger.debug("Generated SENSOR_IMPACT")
    
    def generate_obstacle_event(self, distance: float = 0.3):
        """生成障碍物事件（用于测试）"""
        event = ZulongEvent(
            type=EventType.SENSOR_OBSTACLE,
            priority=EventPriority.HIGH,
            source="simulator",
            payload={"distance": distance}
        )
        event_bus.publish(event)
        logger.debug(f"Generated SENSOR_OBSTACLE: {distance:.2f}m")
    
    def generate_fall_event(self):
        """生成摔倒事件（用于测试）"""
        event = ZulongEvent(
            type=EventType.SENSOR_FALL,
            priority=EventPriority.CRITICAL,
            source="simulator",
            payload={}
        )
        event_bus.publish(event)
        logger.debug("Generated SENSOR_FALL")
    
    def generate_impact_event(self):
        """生成碰撞事件（用于测试）"""
        event = ZulongEvent(
            type=EventType.SENSOR_IMPACT,
            priority=EventPriority.HIGH,
            source="simulator",
            payload={}
        )
        event_bus.publish(event)
        logger.debug("Generated SENSOR_IMPACT")


# 全局传感器模拟器实例
sensor_simulator = SensorSimulator()
