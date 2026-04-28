# File: tests/scripts/e2e_scenarios.py
# 自动化演练剧本 - 第五阶段总装
# 对应 TSD v1.7: 剧本驱动测试

import sys
import os
import time
import threading

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.core.event_bus import event_bus
from zulong.core.state_manager import state_manager
from zulong.core.types import EventType, EventPriority, ZulongEvent, PowerState
from zulong.utils.monitor import TraceManager

import logging
logger = logging.getLogger(__name__)


class ScenarioRunner:
    """剧本运行器"""
    
    def __init__(self):
        self._running = False
        self._thread = None
    
    def run_scenario(self, scenario_name, events):
        """运行指定剧本
        
        Args:
            scenario_name: 剧本名称
            events: 事件列表，每个元素为 (delay, event)
        """
        logger.info(f"=== Running Scenario: {scenario_name} ===")
        
        self._running = True
        self._thread = threading.Thread(
            target=self._run_events, 
            args=(scenario_name, events),
            daemon=True
        )
        self._thread.start()
        
        # 等待剧本完成
        total_duration = max([event[0] for event in events]) + 1
        time.sleep(total_duration)
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        
        logger.info(f"=== Scenario {scenario_name} Completed ===")
    
    def _run_events(self, scenario_name, events):
        """按时间轴运行事件"""
        for delay, event in events:
            if not self._running:
                break
            
            # 等待指定时间
            time.sleep(delay)
            
            # 生成 TraceID
            trace_id = TraceManager.new_trace_id()
            
            # 发布事件
            logger.info(f"[{scenario_name}] Sending event: {event.type.name} (TraceID: {trace_id})")
            event_bus.publish(event)
    
    def scenario_a_silent_wakeup(self):
        """剧本 A: 静默与唤醒"""
        events = [
            # T=0s: 设置 SILENT
            (0.0, ZulongEvent(
                type=EventType.USER_VOICE,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload={"text": "去睡觉", "confidence": 0.95}
            )),
            
            # T=1s: 发送普通语音
            (1.0, ZulongEvent(
                type=EventType.USER_VOICE,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload={"text": "你好", "confidence": 0.9}
            )),
            
            # T=2s: 发送障碍传感器数据
            (2.0, ZulongEvent(
                type=EventType.SENSOR_OBSTACLE,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload={"distance": 0.5, "direction": "front"}
            )),
            
            # T=3s: 发送紧急语音 "救命"
            (3.0, ZulongEvent(
                type=EventType.USER_VOICE,
                priority=EventPriority.HIGH,
                source="ScenarioRunner",
                payload={"text": "救命", "confidence": 0.98}
            )),
            
            # T=8s: 发送 "去睡觉"
            (8.0, ZulongEvent(
                type=EventType.USER_VOICE,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload={"text": "去睡觉", "confidence": 0.95}
            )),
        ]
        
        self.run_scenario("Silent Wakeup", events)
    
    def scenario_b_interrupt_resume(self):
        """剧本 B: 中断与恢复"""
        events = [
            # T=0s: 发送 "讲火星故事"
            (0.0, ZulongEvent(
                type=EventType.USER_VOICE,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload={"text": "讲一个火星的故事", "confidence": 0.95}
            )),
            
            # T=3s: (模拟 L2 生成中) 发送 "摔倒" 传感器事件
            (3.0, ZulongEvent(
                type=EventType.SENSOR_FALL,
                priority=EventPriority.CRITICAL,
                source="ScenarioRunner",
                payload={"severity": "high", "location": "living_room"}
            )),
            
            # T=5s: 发送 "继续"
            (5.0, ZulongEvent(
                type=EventType.USER_VOICE,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload={"text": "继续讲故事", "confidence": 0.9}
            )),
        ]
        
        self.run_scenario("Interrupt Resume", events)
    
    def scenario_c_stress_test(self):
        """剧本 C: 压力测试"""
        events = []
        
        # 生成 100 个混合事件
        for i in range(100):
            delay = i * 0.01  # 间隔 10ms
            
            # 随机选择事件类型
            event_type = None
            payload = {}
            
            if i % 5 == 0:
                # 用户语音事件
                event_type = EventType.USER_VOICE
                payload = {"text": f"测试消息 {i}", "confidence": 0.9}
            elif i % 5 == 1:
                # 障碍传感器事件
                event_type = EventType.SENSOR_OBSTACLE
                payload = {"distance": 0.5 + (i % 10) * 0.1, "direction": "front"}
            elif i % 5 == 2:
                # 运动传感器事件
                event_type = EventType.SENSOR_MOTION
                payload = {"detected": True, "location": "living_room"}
            elif i % 5 == 3:
                # 声音传感器事件
                event_type = EventType.SENSOR_SOUND
                payload = {"level": 60 + (i % 20), "duration": 1.0}
            else:
                # 系统事件
                event_type = EventType.SYSTEM_L2_READY
                payload = {"status": "ready"}
            
            # 创建事件
            event = ZulongEvent(
                type=event_type,
                priority=EventPriority.NORMAL,
                source="ScenarioRunner",
                payload=payload
            )
            
            events.append((delay, event))
        
        self.run_scenario("Stress Test", events)


if __name__ == "__main__":
    # 测试剧本运行器
    runner = ScenarioRunner()
    
    # 运行剧本 A
    runner.scenario_a_silent_wakeup()
    time.sleep(2)
    
    # 运行剧本 B
    runner.scenario_b_interrupt_resume()
    time.sleep(2)
    
    # 运行剧本 C
    runner.scenario_c_stress_test()
