# File: tests/test_layer3_4_action_gesture.py
"""
视觉模块 Layer 3 & Layer 4 单独测试

测试内容：
1. Layer 3: 动作分类（MobileNetV4-TSM）
   - 测试挥手检测
   - 测试意图识别（INTERACTION_TRIGGER）
   - 测试动作分数阈值

2. Layer 4: 手势识别（MediaPipe + EfficientNet）
   - 测试 V 字手势
   - 测试 OK 手势
   - 测试点赞手势
   - 测试鹰眼模式触发

测试配置：
- 动作分类阈值：0.6
- 手势置信度：0.25
- 鹰眼冷却：0.5 秒
- 鹰眼放大：5.0 倍
"""

import asyncio
import time
import cv2
import logging
from typing import Dict, List

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.core.event_bus import event_bus
from zulong.core.types import ZulongEvent, EventType

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Layer3_4_Test")


class TestResult:
    """测试结果记录"""
    def __init__(self):
        self.layer3_detected = False
        self.layer3_max_score = 0.0
        self.layer3_intent_types: List[str] = []
        self.layer4_detected = False
        self.layer4_gestures: List[str] = []
        self.eagle_eye_triggered = False
        self.events_captured: List[ZulongEvent] = []
    
    def add_event(self, event: ZulongEvent):
        self.events_captured.append(event)


async def test_layer3_action_classification():
    """测试 Layer 3: 动作分类"""
    
    print("\n" + "=" * 80)
    print(" Layer 3: 动作分类测试")
    print("=" * 80)
    print("\n测试说明:")
    print("  - 请在摄像头前持续挥手 2-3 秒")
    print("  - 系统会检测动作分数和意图类型")
    print("  - 阈值：action_score > 0.6 触发意图")
    print("\n准备动作：挥手（左右或上下）")
    print("按 'q' 退出测试\n")
    
    result = TestResult()
    
    # 初始化视觉处理器
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    # 订阅事件
    def on_event(event: ZulongEvent):
        result.add_event(event)
        logger.info(f"📩 事件：{event.type.value} | score={processor.shared_memory.get('action_score', 0):.2f}")
    
    event_bus.subscribe(EventType.SENSOR_VISION_STATE, on_event, "Layer3Test")
    event_bus.subscribe(EventType.INTERACTION_TRIGGER, on_event, "Layer3Test")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 初始化完成，请开始挥手...")
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    last_print = 0
    
    try:
        while time.time() - start_time < 30:  # 测试 30 秒
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧
            processor.feed_frame(frame, timestamp)
            
            # 等待处理
            await asyncio.sleep(0.05)
            
            # 获取状态
            mem = processor.shared_memory
            state = processor.state_machine
            
            # 检查 Layer 3 状态
            action_score = mem.get('action_score', 0)
            intent_type = mem.get('intent_type', 'UNKNOWN')
            interact_type = mem.get('interact_type', 'NONE')
            
            # 更新最大值
            if action_score > result.layer3_max_score:
                result.layer3_max_score = action_score
            
            # 检测是否触发
            if action_score > 0.6:
                result.layer3_detected = True
                if intent_type and intent_type not in result.layer3_intent_types:
                    result.layer3_intent_types.append(intent_type)
            
            # 每 1 秒打印状态
            if timestamp - last_print >= 1.0:
                last_print = timestamp
                print(f"\n[时间 {time.time() - start_time:.1f}s | 帧 {frame_count}]")
                print(f"  L3 状态：{state['layer3_state']}")
                print(f"  动作分数：{action_score:.3f} (最大：{result.layer3_max_score:.3f})")
                print(f"  意图类型：{intent_type}")
                print(f"  交互类型：{interact_type}")
                print(f"  触发状态：{'✅ 已触发' if result.layer3_detected else '❌ 未触发'}")
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 打印测试结果
        print("\n" + "=" * 80)
        print(" Layer 3 测试结果")
        print("=" * 80)
        print(f"动作检测：{'✅ 通过' if result.layer3_detected else '❌ 失败'}")
        print(f"最大动作分数：{result.layer3_max_score:.3f}")
        print(f"检测到的意图：{result.layer3_intent_types}")
        print(f"捕获事件数：{len(result.events_captured)}")
        
        # 显示最高分的帧
        if result.layer3_max_score > 0.3:
            print(f"\n✅ 动作分类模型工作正常！")
            return True
        else:
            print(f"\n⚠️  动作分数较低，可能需要更大幅度的动作")
            return result.layer3_max_score > 0.2
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()


async def test_layer4_gesture_recognition():
    """测试 Layer 4: 手势识别"""
    
    print("\n" + "=" * 80)
    print(" Layer 4: 手势识别测试")
    print("=" * 80)
    print("\n测试说明:")
    print("  - 请在摄像头前比出手势")
    print("  - 系统会检测手势类型和置信度")
    print("  - 鹰眼模式会在检测到手势时自动触发")
    print("\n测试手势:")
    print("  1. V 字手势（剪刀手）")
    print("  2. OK 手势")
    print("  3. 点赞手势（大拇指向上）")
    print("  4. 张开手掌")
    print("\n按 'q' 退出测试\n")
    
    result = TestResult()
    
    # 初始化视觉处理器
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    # 订阅事件
    def on_event(event: ZulongEvent):
        result.add_event(event)
        gesture = processor.shared_memory.get('gesture_type', 'NONE')
        logger.info(f"📩 事件：{event.type.value} | gesture={gesture}")
    
    event_bus.subscribe(EventType.SENSOR_VISION_STATE, on_event, "Layer4Test")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 初始化完成，请开始比手势...")
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    last_print = 0
    gesture_history: Dict[str, int] = {}
    
    try:
        while time.time() - start_time < 30:  # 测试 30 秒
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧
            processor.feed_frame(frame, timestamp)
            
            # 等待处理
            await asyncio.sleep(0.05)
            
            # 获取状态
            mem = processor.shared_memory
            state = processor.state_machine
            
            # 检查 Layer 4 状态
            gesture_type = mem.get('gesture_type', 'NONE')
            gesture_conf = mem.get('gesture_confidence', 0)
            eagle_eye_active = mem.get('eagle_eye_active', False)
            
            # 检测手势
            if gesture_type and gesture_type != 'NONE':
                result.layer4_detected = True
                if gesture_type not in result.layer4_gestures:
                    result.layer4_gestures.append(gesture_type)
                gesture_history[gesture_type] = gesture_history.get(gesture_type, 0) + 1
            
            # 检测鹰眼模式
            if eagle_eye_active:
                result.eagle_eye_triggered = True
            
            # 每 1 秒打印状态
            if timestamp - last_print >= 1.0:
                last_print = timestamp
                print(f"\n[时间 {time.time() - start_time:.1f}s | 帧 {frame_count}]")
                print(f"  L4 状态：{state['layer4_state']}")
                print(f"  手势类型：{gesture_type}")
                print(f"  手势置信度：{gesture_conf:.3f}")
                print(f"  鹰眼模式：{'✅ 激活' if eagle_eye_active else '❌ 未激活'}")
                print(f"  检测到的手势：{result.layer4_gestures}")
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 打印测试结果
        print("\n" + "=" * 80)
        print(" Layer 4 测试结果")
        print("=" * 80)
        print(f"手势检测：{'✅ 通过' if result.layer4_detected else '❌ 失败'}")
        print(f"检测到的手势：{result.layer4_gestures}")
        print(f"鹰眼模式触发：{'✅ 是' if result.eagle_eye_triggered else '❌ 否'}")
        print(f"手势历史统计:")
        for gesture, count in sorted(gesture_history.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {gesture}: {count}次")
        print(f"捕获事件数：{len(result.events_captured)}")
        
        # 判断结果
        if result.layer4_detected and len(result.layer4_gestures) > 0:
            print(f"\n✅ 手势识别模型工作正常！")
            return True
        else:
            print(f"\n⚠️  未检测到手势，请确保手势清晰且在光线充足环境下")
            return False
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()


async def test_layer3_4_combined():
    """联合测试 Layer 3 + Layer 4"""
    
    print("\n" + "=" * 80)
    print(" Layer 3 + Layer 4: 联合测试")
    print("=" * 80)
    print("\n测试说明:")
    print("  - 先持续挥手 2-3 秒（触发 Layer 3）")
    print("  - 然后比出手势（触发 Layer 4）")
    print("  - 测试意图识别到手势识别的转换")
    print("\n按 'q' 退出测试\n")
    
    result = TestResult()
    
    # 初始化视觉处理器
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 初始化完成")
    print("阶段 1: 请持续挥手 2-3 秒...")
    await asyncio.sleep(2)
    
    start_time = time.time()
    frame_count = 0
    last_print = 0
    phase = 1  # 1: 挥手，2: 手势
    
    try:
        while time.time() - start_time < 40:  # 测试 40 秒
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            # 喂食帧
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 获取状态
            mem = processor.shared_memory
            
            # Layer 3 状态
            action_score = mem.get('action_score', 0)
            intent_type = mem.get('intent_type', 'UNKNOWN')
            
            # Layer 4 状态
            gesture_type = mem.get('gesture_type', 'NONE')
            gesture_conf = mem.get('gesture_confidence', 0)
            
            # 更新结果
            if action_score > result.layer3_max_score:
                result.layer3_max_score = action_score
            if action_score > 0.6:
                result.layer3_detected = True
                if intent_type and intent_type not in result.layer3_intent_types:
                    result.layer3_intent_types.append(intent_type)
            if gesture_type and gesture_type != 'NONE':
                result.layer4_detected = True
                if gesture_type not in result.layer4_gestures:
                    result.layer4_gestures.append(gesture_type)
            
            # 自动切换阶段
            if phase == 1 and result.layer3_detected:
                print("\n✅ 阶段 1 完成！现在请比出手势...")
                phase = 2
                await asyncio.sleep(1)
            
            # 每 2 秒打印状态
            if timestamp - last_print >= 2.0:
                last_print = timestamp
                print(f"\n[时间 {time.time() - start_time:.1f}s | 阶段 {phase}]")
                print(f"  L3: score={action_score:.3f}, intent={intent_type}")
                print(f"  L4: gesture={gesture_type}, conf={gesture_conf:.3f}")
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 打印测试结果
        print("\n" + "=" * 80)
        print(" Layer 3 + Layer 4 联合测试结果")
        print("=" * 80)
        print(f"Layer 3 (动作分类): {'✅ 通过' if result.layer3_detected else '❌ 失败'}")
        print(f"  最大动作分数：{result.layer3_max_score:.3f}")
        print(f"  检测到的意图：{result.layer3_intent_types}")
        print(f"\nLayer 4 (手势识别): {'✅ 通过' if result.layer4_detected else '❌ 失败'}")
        print(f"  检测到的手势：{result.layer4_gestures}")
        print(f"\n总体评价:")
        if result.layer3_detected and result.layer4_detected:
            print("  ✅ 两层都工作正常！系统可以正确识别意图和手势！")
            return True
        elif result.layer3_detected:
            print("  ⚠️  Layer 3 正常，Layer 4 需要更多测试")
            return True
        elif result.layer4_detected:
            print("  ⚠️  Layer 4 正常，Layer 3 需要更大幅度的动作")
            return True
        else:
            print("  ❌ 两层都未检测到，请检查摄像头和光线条件")
            return False
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()


async def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" 视觉模块 Layer 3 & Layer 4 测试套件")
    print("=" * 80)
    print("\n请选择测试模式:")
    print("1. 单独测试 Layer 3 (动作分类)")
    print("2. 单独测试 Layer 4 (手势识别)")
    print("3. 联合测试 Layer 3 + Layer 4")
    print("4. 运行所有测试")
    print()
    
    choice = input("请输入选项 (1-4): ").strip()
    
    results = []
    
    if choice == '1':
        results.append(await test_layer3_action_classification())
    elif choice == '2':
        results.append(await test_layer4_gesture_recognition())
    elif choice == '3':
        results.append(await test_layer3_4_combined())
    elif choice == '4':
        results.append(await test_layer3_action_classification())
        await asyncio.sleep(2)
        results.append(await test_layer4_gesture_recognition())
        await asyncio.sleep(2)
        results.append(await test_layer3_4_combined())
    else:
        print("❌ 无效选项")
        return
    
    # 打印总结
    print("\n" + "=" * 80)
    print(" 测试总结")
    print("=" * 80)
    if all(results):
        print("✅ 所有测试通过！")
    else:
        print(f"❌ {results.count(False)} 个测试失败")


if __name__ == "__main__":
    asyncio.run(main())
