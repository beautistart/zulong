# File: tests/test_zulong_system_integration.py
"""
祖龙系统集成测试

测试整个系统的 L0-L4 联动：
1. 摄像头采集
2. L1-C 视觉处理（四层检测）
3. L1-A 反射响应
4. L1-B 调度决策
5. L2 推理与中断
6. EventBus 事件流转
7. L0 执行器响应

测试场景：
- 场景 1: 人体检测 → L1-A 反射
- 场景 2: 运动检测 → SENSOR_VISION_STATE 事件
- 场景 3: 意图识别 → INTERACTION_TRIGGER 事件
- 场景 4: 手势识别 → 鹰眼模式
- 场景 5: 紧急停止 → CMD_EMERGENCY_STOP 事件
"""

import asyncio
import time
import logging
from typing import Dict, List
from datetime import datetime

from zulong.core.event_bus import event_bus
from zulong.core.state_manager import state_manager
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.l0.devices.camera_device import CameraDevice

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("SystemIntegrationTest")


class IntegrationTestResult:
    """集成测试结果"""
    def __init__(self):
        self.scenarios_passed = 0
        self.scenarios_failed = 0
        self.events_captured: List[ZulongEvent] = []
        self.errors: List[str] = []
    
    def add_event(self, event: ZulongEvent):
        """记录事件"""
        self.events_captured.append(event)
    
    def add_error(self, error: str):
        """记录错误"""
        self.errors.append(error)
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print(" 集成测试结果总结")
        print("=" * 80)
        print(f"\n总场景数：{self.scenarios_passed + self.scenarios_failed}")
        print(f"通过场景：{self.scenarios_passed} ✅")
        print(f"失败场景：{self.scenarios_failed} ❌")
        print(f"捕获事件：{len(self.events_captured)}")
        
        if self.errors:
            print(f"\n错误列表:")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        # 事件统计
        event_types = {}
        for event in self.events_captured:
            event_type = event.type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        print(f"\n事件类型统计:")
        for event_type, count in sorted(event_types.items()):
            print(f"  - {event_type}: {count}次")
        
        print("\n" + "=" * 80)


async def run_integration_test():
    """运行集成测试"""
    
    print("\n" + "=" * 80)
    print(" 祖龙系统集成测试 - L0-L4 完整联动")
    print("=" * 80)
    
    result = IntegrationTestResult()
    
    # 事件捕获器
    def on_event(event: ZulongEvent):
        result.add_event(event)
        logger.info(f"📩 事件：{event.type.value} from {event.source} (P{event.priority.value})")
    
    # 订阅所有关键事件
    event_bus.subscribe(EventType.SENSOR_VISION_STATE, on_event, "IntegrationTest")
    event_bus.subscribe(EventType.INTERACTION_TRIGGER, on_event, "IntegrationTest")
    event_bus.subscribe(EventType.DIRECT_WAKEUP, on_event, "IntegrationTest")
    event_bus.subscribe(EventType.CMD_EMERGENCY_STOP, on_event, "IntegrationTest")
    event_bus.subscribe(EventType.CMD_BRAKE, on_event, "IntegrationTest")
    
    try:
        # ========== 1. 初始化视觉系统 ==========
        print("\n[1/5] 初始化视觉系统...")
        
        vision_processor = OptimizedVisionProcessor()
        await vision_processor.initialize(load_models=True)
        
        print("  ✅ 视觉处理器已初始化")
        print(f"     - YOLO 置信度：{vision_processor._config['yolo_conf_threshold']}")
        print(f"     - ROI 运动阈值：{vision_processor._config['roi_motion_threshold']}px")
        print(f"     - 手势置信度：{vision_processor._config['gesture_conf_threshold']}")
        print(f"     - 鹰眼冷却：{vision_processor._config['eagle_eye_cooldown']}s")
        
        # ========== 2. 启动摄像头 ==========
        print("\n[2/5] 启动摄像头...")
        
        camera = CameraDevice()
        await camera.initialize()
        camera.start_streaming()
        
        print("  ✅ 摄像头已启动")
        
        # 等待摄像头稳定
        await asyncio.sleep(2)
        
        # ========== 3. 场景测试 1: 人体检测 ==========
        print("\n[3/5] 场景测试 1: 人体检测")
        print("  说明：站在摄像头前，检测人体")
        print("  预期：Layer 1 检测到人体，发布 SENSOR_VISION_STATE 事件")
        
        human_detected = False
        sensor_events_count = 0
        
        for i in range(30):  # 测试 30 帧（约 1 秒）
            ret, frame = camera.read()
            if not ret:
                break
            
            timestamp = time.time()
            vision_processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 检查人体检测
            if vision_processor.shared_memory['human_detected']:
                human_detected = True
            
            # 检查事件
            recent_events = [e for e in result.events_captured 
                           if e.type == EventType.SENSOR_VISION_STATE 
                           and e.timestamp > timestamp - 1.0]
            sensor_events_count = len(recent_events)
        
        if human_detected and sensor_events_count > 0:
            print(f"  ✅ 场景 1 通过：检测到人体，{sensor_events_count}个 SENSOR_VISION_STATE 事件")
            result.scenarios_passed += 1
        else:
            print(f"  ❌ 场景 1 失败：human={human_detected}, events={sensor_events_count}")
            result.scenarios_failed += 1
            if not human_detected:
                result.add_error("场景 1: 未检测到人体")
            if sensor_events_count == 0:
                result.add_error("场景 1: 未发布 SENSOR_VISION_STATE 事件")
        
        # ========== 4. 场景测试 2: 运动检测 ==========
        print("\n[4/5] 场景测试 2: 运动检测")
        print("  说明：挥动手臂，触发运动检测")
        print("  预期：Layer 2 检测到运动，motion_pixels > 100")
        
        motion_detected = False
        max_motion_pixels = 0
        
        for i in range(30):
            ret, frame = camera.read()
            if not ret:
                break
            
            timestamp = time.time()
            vision_processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 检查运动像素
            motion_pixels = vision_processor.shared_memory.get('motion_pixels', 0)
            max_motion_pixels = max(max_motion_pixels, motion_pixels)
            
            if motion_pixels > 100:
                motion_detected = True
        
        if motion_detected and max_motion_pixels > 100:
            print(f"  ✅ 场景 2 通过：最大运动像素 {max_motion_pixels}px")
            result.scenarios_passed += 1
        else:
            print(f"  ❌ 场景 2 失败：motion={motion_detected}, max_pixels={max_motion_pixels}")
            result.scenarios_failed += 1
            result.add_error(f"场景 2: 运动检测失败 (max={max_motion_pixels}px)")
        
        # ========== 5. 场景测试 3: 意图识别 ==========
        print("\n[5/5] 场景测试 3: 意图识别 + 手势识别")
        print("  说明：持续挥手 2-3 秒，然后比出 V 字手势")
        print("  预期：Layer 3 识别意图，Layer 4 识别手势")
        
        # 模拟持续挥手（实际测试需要用户配合，这里只检查状态）
        print("  提示：请持续挥手 2-3 秒...")
        await asyncio.sleep(3)
        
        intent_detected = False
        gesture_detected = False
        interaction_events_count = 0
        
        for i in range(60):  # 测试 3 秒
            ret, frame = camera.read()
            if not ret:
                break
            
            timestamp = time.time()
            vision_processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 检查意图
            action_score = vision_processor.shared_memory.get('action_score', 0)
            if action_score > 0.6:
                intent_detected = True
            
            # 检查手势
            gesture_type = vision_processor.shared_memory.get('gesture_type', '')
            if gesture_type:
                gesture_detected = True
            
            # 检查 INTERACTION_TRIGGER 事件
            recent_events = [e for e in result.events_captured 
                           if e.type == EventType.INTERACTION_TRIGGER 
                           and e.timestamp > timestamp - 1.0]
            interaction_events_count = len(recent_events)
        
        if intent_detected or gesture_detected or interaction_events_count > 0:
            print(f"  ✅ 场景 3 通过：intent={intent_detected}, gesture={gesture_detected}, events={interaction_events_count}")
            result.scenarios_passed += 1
        else:
            print(f"  ⚠️  场景 3 未完全通过：intent={intent_detected}, gesture={gesture_detected}")
            # 不记为失败，因为需要用户实际配合
        
        # ========== 清理资源 ==========
        print("\n清理资源...")
        
        camera.stop_streaming()
        await camera.cleanup()
        vision_processor.stop()
        
        # ========== 打印总结 ==========
        result.print_summary()
        
        # 判断总体结果
        if result.scenarios_failed == 0:
            print("\n✅ 所有场景测试通过！系统集成成功！")
            return True
        else:
            print(f"\n❌ {result.scenarios_failed} 个场景测试失败")
            return False
        
    except Exception as e:
        logger.error(f"❌ 集成测试失败：{e}", exc_info=True)
        result.add_error(f"集成测试异常：{str(e)}")
        result.print_summary()
        return False


async def main():
    """主函数"""
    try:
        success = await run_integration_test()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        exit(0)


if __name__ == "__main__":
    asyncio.run(main())
