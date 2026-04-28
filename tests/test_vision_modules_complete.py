# File: tests/test_vision_modules_complete.py
"""
完整视觉模块测试（基于成功测试配置）

测试顺序：
1. Layer 1: YOLO 人体检测
2. Layer 2: ROI 运动检测  
3. Layer 3: 动作分类
4. Layer 4: 手势识别
5. L0-L4 完整联动

配置说明（基于成功测试）：
- YOLO 置信度：0.25
- MediaPipe 阈值：0.3
- 鹰眼冷却：0.5 秒
- 手势置信度：0.25
"""

import cv2
import time
import asyncio
import logging
from typing import Dict, Any

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.l1c.vision_model_loader import get_vision_model_loader
from zulong.l1c.action_classifier import MobileNetV4_TSM
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer
from zulong.core.types import ZulongEvent, EventType, EventPriority
from zulong.core.event_bus import event_bus

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("VisionTest")


class TestResult:
    """测试结果记录"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details: Dict[str, Any] = {}
        self.errors = []
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} - {self.name}"


# ============== Layer 1: YOLO 人体检测 ==============
async def test_layer1_yolo() -> TestResult:
    """测试 Layer 1: YOLO 人体检测"""
    result = TestResult("Layer 1 - YOLO 人体检测")
    
    print("\n" + "=" * 80)
    print(" Layer 1: YOLO 人体检测测试")
    print("=" * 80)
    
    try:
        # 加载模型
        print("加载 YOLO 模型...")
        loader = get_vision_model_loader()
        loader.load_all_models()
        
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        print("✅ 初始化完成，请站在摄像头前")
        print("测试说明：")
        print("  - 确保人体在画面中")
        print("  - 观察绿色人体检测框")
        print("  - 按 'q' 退出进入下一层")
        print("\n3 秒后开始...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1)
        
        start_time = time.time()
        frame_count = 0
        detect_count = 0
        max_confidence = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO 检测
            human_detected, bbox_info = loader.detect_human(frame)
            
            if human_detected and bbox_info:
                detect_count += 1
                conf = bbox_info.get('confidence', 0)
                max_confidence = max(max_confidence, conf)
                
                # 绘制框
                x1, y1, x2, y2 = [int(c) for c in bbox_info['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示统计
            elapsed = frame_count / (time.time() - start_time + 0.001)
            cv2.putText(frame, f"Detect: {detect_count}/{frame_count} ({detect_count/frame_count*100:.1f}%)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {elapsed:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Layer 1 Test", frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        # 评估结果
        detection_rate = detect_count / frame_count if frame_count > 0 else 0
        result.details = {
            'total_frames': frame_count,
            'detected_frames': detect_count,
            'detection_rate': f"{detection_rate*100:.1f}%",
            'max_confidence': f"{max_confidence:.2f}"
        }
        
        if detection_rate >= 0.5 and max_confidence >= 0.4:
            result.passed = True
            print(f"\n✅ Layer 1 测试通过！")
        else:
            print(f"\n❌ Layer 1 测试失败！")
            result.errors.append(f"检测率 {detection_rate*100:.1f}% < 50% 或 置信度 {max_confidence:.2f} < 0.4")
        
        print(f"  - 总帧数：{frame_count}")
        print(f"  - 检测帧数：{detect_count}")
        print(f"  - 检测率：{detection_rate*100:.1f}%")
        print(f"  - 最高置信度：{max_confidence:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"❌ Layer 1 测试失败：{e}", exc_info=True)
    
    return result


# ============== Layer 2: ROI 运动检测 ==============
async def test_layer2_motion() -> TestResult:
    """测试 Layer 2: ROI 运动检测"""
    result = TestResult("Layer 2 - ROI 运动检测")
    
    print("\n" + "=" * 80)
    print(" Layer 2: ROI 运动检测测试")
    print("=" * 80)
    
    try:
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=False)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        print("✅ 初始化完成，请挥动手臂")
        print("测试说明：")
        print("  - 在摄像头前挥动手臂")
        print("  - 观察红色/绿色运动指示")
        print("  - 按 'q' 退出进入下一层")
        print("\n3 秒后开始...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1)
        
        start_time = time.time()
        frame_count = 0
        motion_detected_count = 0
        max_motion_pixels = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            motion_pixels = processor.shared_memory.get('motion_pixels', 0)
            if motion_pixels > 0:
                motion_detected_count += 1
                max_motion_pixels = max(max_motion_pixels, motion_pixels)
            
            # 显示
            h, w = frame.shape[:2]
            color = (0, 255, 0) if motion_pixels > 0 else (0, 0, 255)
            cv2.putText(frame, f"Motion: {motion_pixels}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Layer 2 Test", frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        # 评估结果
        motion_rate = motion_detected_count / frame_count if frame_count > 0 else 0
        result.details = {
            'total_frames': frame_count,
            'motion_frames': motion_detected_count,
            'motion_rate': f"{motion_rate*100:.1f}%",
            'max_motion_pixels': max_motion_pixels
        }
        
        if motion_rate >= 0.3 and max_motion_pixels > 100:
            result.passed = True
            print(f"\n✅ Layer 2 测试通过！")
        else:
            print(f"\n❌ Layer 2 测试失败！")
            result.errors.append(f"运动检测率 {motion_rate*100:.1f}% < 30% 或 最大运动像素 {max_motion_pixels} < 100")
        
        print(f"  - 总帧数：{frame_count}")
        print(f"  - 检测到运动帧数：{motion_detected_count}")
        print(f"  - 运动检测率：{motion_rate*100:.1f}%")
        print(f"  - 最大运动像素：{max_motion_pixels}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"❌ Layer 2 测试失败：{e}", exc_info=True)
    
    return result


# ============== Layer 3: 动作分类 ==============
async def test_layer3_action() -> TestResult:
    """测试 Layer 3: 动作分类"""
    result = TestResult("Layer 3 - 动作分类")
    
    print("\n" + "=" * 80)
    print(" Layer 3: 动作分类测试")
    print("=" * 80)
    
    try:
        print("加载 MobileNetV4-TSM 动作分类器...")
        action_classifier = MobileNetV4_TSM()
        await action_classifier.initialize()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        print("✅ 初始化完成，请挥手/靠近/注视")
        print("测试说明：")
        print("  - 挥手、靠近摄像头或注视")
        print("  - 观察意图识别结果")
        print("  - 按 'q' 退出进入下一层")
        print("\n3 秒后开始...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1)
        
        start_time = time.time()
        frame_count = 0
        intent_detected_count = 0
        intent_types = {}
        max_score = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            score, intent_type = await action_classifier.classify_action(frame)
            
            if score > 0.3:
                intent_detected_count += 1
                max_score = max(max_score, score)
                intent_types[intent_type] = intent_types.get(intent_type, 0) + 1
            
            # 显示
            color = (0, 0, 255) if score < 0.6 else (0, 255, 255) if score < 0.8 else (0, 255, 0)
            cv2.putText(frame, f"{intent_type} ({score:.2f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Layer 3 Test", frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        # 评估结果
        intent_rate = intent_detected_count / frame_count if frame_count > 0 else 0
        result.details = {
            'total_frames': frame_count,
            'intent_frames': intent_detected_count,
            'intent_rate': f"{intent_rate*100:.1f}%",
            'max_score': f"{max_score:.2f}",
            'intent_distribution': intent_types
        }
        
        if intent_rate >= 0.3 and max_score >= 0.5:
            result.passed = True
            print(f"\n✅ Layer 3 测试通过！")
        else:
            print(f"\n❌ Layer 3 测试失败！")
            result.errors.append(f"意图检测率 {intent_rate*100:.1f}% < 30% 或 最高分数 {max_score:.2f} < 0.5")
        
        print(f"  - 总帧数：{frame_count}")
        print(f"  - 检测到意图帧数：{intent_detected_count}")
        print(f"  - 意图检测率：{intent_rate*100:.1f}%")
        print(f"  - 最高分数：{max_score:.2f}")
        print(f"  - 意图分布：{intent_types}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"❌ Layer 3 测试失败：{e}", exc_info=True)
    
    return result


# ============== Layer 4: 手势识别 ==============
async def test_layer4_gesture() -> TestResult:
    """测试 Layer 4: 手势识别"""
    result = TestResult("Layer 4 - 手势识别")
    
    print("\n" + "=" * 80)
    print(" Layer 4: 手势识别测试")
    print("=" * 80)
    
    try:
        print("加载 MediaPipe Gesture Recognizer...")
        gesture_recognizer = MediaPipeGestureRecognizer(confidence_threshold=0.3)
        await asyncio.sleep(0.5)  # 等待初始化
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        print("✅ 初始化完成，请比出手势")
        print("测试说明：")
        print("  - 张开手掌 ✋")
        print("  - V 字手势 ✌️")
        print("  - OK 手势 👌")
        print("  - 拳头 ✊")
        print("  - 按 'q' 退出进入下一层")
        print("\n3 秒后开始...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1)
        
        start_time = time.time()
        frame_count = 0
        gesture_detected_count = 0
        gesture_types = {}
        max_confidence = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 手势识别
            gesture_type, confidence, _ = gesture_recognizer.classify_gesture(frame)
            
            if gesture_type and confidence > 0.3:
                gesture_detected_count += 1
                max_confidence = max(max_confidence, confidence)
                gesture_types[gesture_type] = gesture_types.get(gesture_type, 0) + 1
                
                # 绘制
                cv2.putText(frame, f"{gesture_type} ({confidence:.2f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("Layer 4 Test", frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        # 评估结果
        gesture_rate = gesture_detected_count / frame_count if frame_count > 0 else 0
        result.details = {
            'total_frames': frame_count,
            'gesture_frames': gesture_detected_count,
            'gesture_rate': f"{gesture_rate*100:.1f}%",
            'max_confidence': f"{max_confidence:.2f}",
            'gesture_distribution': gesture_types
        }
        
        if gesture_rate >= 0.3 and max_confidence >= 0.4:
            result.passed = True
            print(f"\n✅ Layer 4 测试通过！")
        else:
            print(f"\n❌ Layer 4 测试失败！")
            result.errors.append(f"手势检测率 {gesture_rate*100:.1f}% < 30% 或 最高置信度 {max_confidence:.2f} < 0.4")
        
        print(f"  - 总帧数：{frame_count}")
        print(f"  - 检测到手势帧数：{gesture_detected_count}")
        print(f"  - 手势检测率：{gesture_rate*100:.1f}%")
        print(f"  - 最高置信度：{max_confidence:.2f}")
        print(f"  - 手势分布：{gesture_types}")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"❌ Layer 4 测试失败：{e}", exc_info=True)
    
    return result


# ============== L0-L4 联合测试 ==============
async def test_full_integration() -> TestResult:
    """测试 L0-L4 完整联合处理"""
    result = TestResult("L0-L4 - 完整联合测试")
    
    print("\n" + "=" * 80)
    print(" L0-L4: 完整系统联合测试")
    print("=" * 80)
    
    events_received = []
    
    def on_event(event: ZulongEvent):
        events_received.append(event)
        print(f"  📩 事件：{event.type.value} from {event.source}")
    
    event_bus.subscribe(EventType.SENSOR_VISION_STATE, on_event, "Test")
    event_bus.subscribe(EventType.INTERACTION_TRIGGER, on_event, "Test")
    
    try:
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        print("✅ 初始化完成")
        print("\n请按以下步骤操作：")
        print("1. 坐在摄像头前静止 3 秒")
        print("2. 挥动手臂 2-3 秒")
        print("3. 继续挥手并比出 V 字手势")
        print("4. 按 'q' 退出测试")
        print("\n3 秒后开始...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            await asyncio.sleep(1)
        
        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = time.time()
            
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 显示
            h, w = frame.shape[:2]
            mem = processor.shared_memory
            
            if mem['human_detected'] and mem.get('human_bbox'):
                x1, y1, x2, y2 = [int(c) for c in mem['human_bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if mem.get('motion_pixels', 0) > 0:
                cv2.putText(frame, f"L2: {mem['motion_pixels']}px", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            score = mem.get('action_score', 0)
            intent = mem.get('intent_type', 'UNKNOWN')
            color = (0, 0, 255) if score < 0.6 else (0, 255, 255) if score < 0.8 else (0, 255, 0)
            cv2.putText(frame, f"L3: {intent} ({score:.2f})", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            gesture = mem.get('gesture_type', '')
            if gesture:
                cv2.putText(frame, f"L4: {gesture}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            cv2.imshow("Full Integration Test", frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        
        # 评估结果
        result.details = {
            'total_frames': frame_count,
            'events_received': len(events_received),
            'event_types': [e.type.value for e in events_received]
        }
        
        sensor_events = [e for e in events_received if e.type == EventType.SENSOR_VISION_STATE]
        interaction_events = [e for e in events_received if e.type == EventType.INTERACTION_TRIGGER]
        
        if len(sensor_events) > 0 and len(interaction_events) > 0:
            result.passed = True
            print(f"\n✅ L0-L4 联合测试通过！")
        else:
            print(f"\n❌ L0-L4 联合测试失败！")
            if len(sensor_events) == 0:
                result.errors.append("未收到 SENSOR_VISION_STATE 事件")
            if len(interaction_events) == 0:
                result.errors.append("未收到 INTERACTION_TRIGGER 事件")
        
        print(f"  - 总帧数：{frame_count}")
        print(f"  - 收到事件数：{len(events_received)}")
        print(f"  - SENSOR_VISION_STATE: {len(sensor_events)}次")
        print(f"  - INTERACTION_TRIGGER: {len(interaction_events)}次")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"❌ 联合测试失败：{e}", exc_info=True)
    
    return result


# ============== 主测试流程 ==============
async def run_all_tests():
    """运行所有测试"""
    
    print("\n" + "=" * 80)
    print(" 祖龙视觉系统 - 完整测试套件")
    print("=" * 80)
    print("\n测试配置（基于成功测试）：")
    print("  - YOLO 置信度：0.25")
    print("  - MediaPipe 阈值：0.3")
    print("  - 鹰眼冷却：0.5 秒")
    print("  - 手势置信度：0.25")
    print("\n测试将按顺序执行：")
    print("1. Layer 1: YOLO 人体检测")
    print("2. Layer 2: ROI 运动检测")
    print("3. Layer 3: 动作分类")
    print("4. Layer 4: 手势识别")
    print("5. L0-L4: 完整联合测试")
    print("\n每层测试完成后按 'q' 进入下一层\n")
    
    results = []
    
    # Layer 1
    results.append(await test_layer1_yolo())
    await asyncio.sleep(1)
    
    # Layer 2
    results.append(await test_layer2_motion())
    await asyncio.sleep(1)
    
    # Layer 3
    results.append(await test_layer3_action())
    await asyncio.sleep(1)
    
    # Layer 4
    results.append(await test_layer4_gesture())
    await asyncio.sleep(1)
    
    # Full Integration
    results.append(await test_full_integration())
    
    # 打印总结
    print("\n" + "=" * 80)
    print(" 测试结果总结")
    print("=" * 80)
    
    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)
    
    for result in results:
        print(f"\n{result}")
        if result.details:
            for key, value in result.details.items():
                print(f"  - {key}: {value}")
        if result.errors:
            for error in result.errors:
                print(f"  ❌ {error}")
    
    print(f"\n{'=' * 80}")
    print(f" 总计：{passed_count}/{total_count} 测试通过")
    print(f"{'=' * 80}\n")
    
    return passed_count == total_count


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        exit(0)
