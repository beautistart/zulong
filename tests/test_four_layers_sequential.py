# File: tests/test_four_layers_sequential.py
"""
四层视觉系统顺序测试

按顺序测试 Layer 1 → Layer 2 → Layer 3 → Layer 4
每层独立测试，确保各层功能正常。
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.l1c.vision_model_loader import VisionModelLoader


async def test_layer1_human_detection(processor: OptimizedVisionProcessor, cap: cv2.VideoCapture):
    """
    Layer 1: YOLO 人体检测测试
    """
    print("\n" + "=" * 60)
    print(" Layer 1: YOLO 人体检测测试")
    print("=" * 60)
    
    frame_count = 0
    human_count = 0
    
    print("\n 测试说明:")
    print("   - 站在摄像头前，确保人体在画面中")
    print("   - 观察绿色人体检测框")
    print("   - 按 'q' 退出本层测试，进入下一层")
    print("\n 3 秒后开始...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = time.time()
        
        # Layer 1: 人体检测
        human_bboxes = processor._layer1_human_detection(frame)
        
        # 绘制结果
        if human_bboxes:
            human_count += 1
            for bbox in human_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "HUMAN", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            status = "Human Detected!"
            status_color = (0, 255, 0)
        else:
            status = "No Human"
            status_color = (0, 0, 255)
        
        # 显示统计
        elapsed = frame_count / (time.time() - start_time + 0.001)
        cv2.putText(frame, f"Layer 1: {status}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Human: {human_count}/{frame_count} ({human_count/frame_count*100:.1f}%)", 
                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {elapsed:.1f}", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Four Layers Test", frame)
        
        key = cv2.waitKey(10) & 0xFF  # 增加到 10ms 确保按键响应
        if key in [ord('q'), ord('Q'), 27]:  # q, Q, 或 ESC
            print("\n 用户退出当前层测试")
            break
    
    print(f"\n Layer 1 测试结果:")
    print(f"   总帧数：{frame_count}")
    print(f"   检测到人体：{human_count}")
    print(f"   检测率：{human_count/frame_count*100:.1f}%")
    
    return human_count > 0


async def test_layer2_motion_detection(processor: OptimizedVisionProcessor, cap: cv2.VideoCapture):
    """
    Layer 2: ROI 运动检测测试
    """
    print("\n" + "=" * 60)
    print(" Layer 2: ROI 运动检测测试")
    print("=" * 60)
    
    frame_count = 0
    motion_count = 0
    
    print("\n 测试说明:")
    print("   - 确保人体在画面中（Layer 1 检测）")
    print("   - 挥动手臂或移动身体触发运动检测")
    print("   - 观察蓝色 ROI 区域和绿色运动检测框")
    print("   - 按 'q' 退出本层测试，进入下一层")
    print("\n 3 秒后开始...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = time.time()
        
        # 添加帧到缓冲区（Layer 2 需要帧间差分）
        processor.frame_buffer.append(frame.copy())
        
        # Layer 1: 人体检测
        human_bboxes = processor._layer1_human_detection(frame)
        
        # Layer 2: 运动检测
        motion_detected, motion_pixels = processor._layer2_roi_motion_detection(
            frame, human_bboxes[0] if human_bboxes else None
        )
        motion_bbox = None  # Layer 2 不返回 bbox
        
        # 绘制结果
        if human_bboxes:
            for bbox in human_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if motion_detected:
            motion_count += 1
            if motion_bbox:
                x1, y1, x2, y2 = map(int, motion_bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "MOTION", (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            status = "Motion Detected!"
            status_color = (255, 255, 0)
        else:
            status = "No Motion"
            status_color = (0, 0, 255)
        
        # 显示统计
        elapsed = frame_count / (time.time() - start_time + 0.001)
        cv2.putText(frame, f"Layer 1: {'Human' if human_bboxes else 'None'}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if human_bboxes else (0, 0, 255), 2)
        cv2.putText(frame, f"Layer 2: {status}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Motion: {motion_count}/{frame_count} ({motion_count/frame_count*100:.1f}%)", 
                  (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {elapsed:.1f}", (10, 150), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Four Layers Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    print(f"\n Layer 2 测试结果:")
    print(f"   总帧数：{frame_count}")
    print(f"   检测到运动：{motion_count}")
    print(f"   运动检测率：{motion_count/frame_count*100:.1f}%")
    
    return motion_count > 0


async def test_layer3_action_classification(processor: OptimizedVisionProcessor, cap: cv2.VideoCapture):
    """
    Layer 3: MobileNetV4-TSM 动作分类测试
    """
    print("\n" + "=" * 60)
    print(" Layer 3: 动作分类测试")
    print("=" * 60)
    
    frame_count = 0
    intent_detected_count = 0
    
    print("\n 测试说明:")
    print("   - 保持一个动作 2-3 秒（如伸手、指向）")
    print("   - 观察意图分类结果")
    print("   - 按 'q' 退出本层测试，进入下一层")
    print("\n 3 秒后开始...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = time.time()
        
        # Layer 1: 人体检测
        human_bboxes = processor._layer1_human_detection(frame)
        
        # Layer 2: 运动检测
        motion_detected, motion_pixels = processor._layer2_roi_motion_detection(
            frame, human_bboxes[0] if human_bboxes else None
        )
        motion_bbox = None  # Layer 2 不返回 bbox
        
        # Layer 3: 动作分类（使用 action_classifier，需要填充缓冲区）
        # 先添加帧到 action_classifier 缓冲区
        if processor._action_classifier:
            processor._action_classifier.add_frame(frame, timestamp)
        
        intent_score, intent_type = processor._layer3_action_classification(
            frame
        )
        
        # 绘制结果
        if human_bboxes:
            for bbox in human_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if intent_type and intent_score > 0.5:
            intent_detected_count += 1
            status = f"Intent: {intent_type} ({intent_score:.2f})"
            status_color = (0, 255, 255)
        else:
            status = "No Intent"
            status_color = (0, 0, 255)
        
        # 显示统计
        elapsed = frame_count / (time.time() - start_time + 0.001)
        cv2.putText(frame, f"L1: {'Human' if human_bboxes else 'None'}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if human_bboxes else (0, 0, 255), 2)
        cv2.putText(frame, f"L2: {'Motion' if motion_detected else 'Still'}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0) if motion_detected else (0, 0, 255), 2)
        cv2.putText(frame, f"L3: {status}", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Intent: {intent_detected_count}/{frame_count}", 
                  (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {elapsed:.1f}", (10, 180), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Four Layers Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    print(f"\n Layer 3 测试结果:")
    print(f"   总帧数：{frame_count}")
    print(f"   检测到意图：{intent_detected_count}")
    print(f"   意图检测率：{intent_detected_count/frame_count*100:.1f}%")
    
    return intent_detected_count > 0


async def test_layer4_gesture_recognition(processor: OptimizedVisionProcessor, cap: cv2.VideoCapture):
    """
    Layer 4: MediaPipe 手势识别测试
    """
    print("\n" + "=" * 60)
    print(" Layer 4: 手势识别测试")
    print("=" * 60)
    
    frame_count = 0
    gesture_count = 0
    last_gesture = None
    
    print("\n 测试说明:")
    print("   - 展示以下手势:")
    print("     • Open_Palm (张开手掌) ✋")
    print("     • Victory_Sign (V 字手势) ✌️")
    print("     • Thumb_Up (竖起大拇指) 👍")
    print("     • OK_Gesture (OK 手势) 👌")
    print("   - 观察手势识别结果")
    print("   - 按 'q' 退出测试")
    print("\n 3 秒后开始...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        timestamp = time.time()
        
        # Layer 1: 人体检测（获取 bbox）
        human_detected, human_info = processor._model_loader.detect_human(frame)
        human_bbox = human_info['bbox'] if human_info else None
        
        # Layer 4: 手势识别（鹰眼模式）
        gesture_result = None
        if human_bbox:
            gesture_result = processor._layer4_eagle_eye_mode(frame, human_bbox, timestamp)
        
        # 绘制结果
        gesture_name = None
        confidence = 0.0
        
        if gesture_result:
            gesture_name = gesture_result.get('gesture', 'UNKNOWN')
            confidence = gesture_result.get('confidence', 0.0)
        
        if gesture_name and confidence > 0.3:
            gesture_count += 1
            status = f"{gesture_name} ({confidence:.2f})"
            status_color = (0, 255, 0)
            
            if gesture_name != last_gesture:
                print(f"\n 帧 {frame_count}: 识别手势 - {gesture_name} (置信度：{confidence:.2f})")
                last_gesture = gesture_name
        else:
            status = "No Gesture"
            status_color = (0, 0, 255)
        
        # 显示统计
        elapsed = frame_count / (time.time() - start_time + 0.001)
        cv2.putText(frame, f"L4: {status}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Gestures: {gesture_count}/{frame_count} ({gesture_count/frame_count*100:.1f}%)", 
                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {elapsed:.1f}", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Four Layers Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    print(f"\n Layer 4 测试结果:")
    print(f"   总帧数：{frame_count}")
    print(f"   识别手势：{gesture_count}")
    print(f"   手势识别率：{gesture_count/frame_count*100:.1f}%")
    
    return gesture_count > 0


async def main():
    """主测试流程"""
    print("=" * 60)
    print(" 四层视觉系统顺序测试")
    print("=" * 60)
    
    # 初始化处理器
    print("\n 初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    print(" 处理器初始化成功")
    
    # 打开摄像头（设备 0）
    print("\n 打开摄像头 (设备 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" 无法打开摄像头")
        return
    
    print(" 摄像头已启动")
    
    # 模型状态检查
    print("\n 模型加载状态:")
    print(f"   - YOLO: {'✅' if processor._model_loader and processor._model_loader.yolo_model else '❌'}")
    print(f"   - MobileNetV3: {'✅' if processor._action_classifier and processor._action_classifier._model else '❌'}")
    print(f"   - MediaPipe: {'✅' if processor._gesture_classifier else '❌'}")
    
    try:
        # Layer 1 测试
        layer1_passed = await test_layer1_human_detection(processor, cap)
        
        # Layer 2 测试
        layer2_passed = await test_layer2_motion_detection(processor, cap)
        
        # Layer 3 测试
        layer3_passed = await test_layer3_action_classification(processor, cap)
        
        # Layer 4 测试
        layer4_passed = await test_layer4_gesture_recognition(processor, cap)
        
        # 总结
        print("\n" + "=" * 60)
        print(" 测试总结")
        print("=" * 60)
        print(f" Layer 1 (人体检测): {'✅ 通过' if layer1_passed else '❌ 失败'}")
        print(f" Layer 2 (运动检测): {'✅ 通过' if layer2_passed else '❌ 失败'}")
        print(f" Layer 3 (动作分类): {'✅ 通过' if layer3_passed else '❌ 失败'}")
        print(f" Layer 4 (手势识别): {'✅ 通过' if layer4_passed else '❌ 失败'}")
        
        if all([layer1_passed, layer2_passed, layer3_passed, layer4_passed]):
            print("\n 🎉 所有层测试通过！")
        else:
            print("\n ⚠️ 部分层测试失败，请检查日志")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n 测试中断")
    except Exception as e:
        print(f"\n 测试失败：{e}")
        import traceback
        traceback.print_exc()
