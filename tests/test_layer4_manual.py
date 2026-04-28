# File: tests/test_layer4_manual.py
"""
Layer 4 手势识别手动触发测试

绕过 Layer 3 阈值限制，直接测试 MediaPipe 手势识别
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
import cv2
import logging
from datetime import datetime

from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer

logging.basicConfig(level=logging.INFO)

async def test():
    print("\n" + "=" * 80)
    print(" Layer 4: 手势识别手动触发测试")
    print("=" * 80)
    print("\n说明:")
    print("  - 绕过 Layer 3 阈值限制")
    print("  - 直接调用 MediaPipe 手势识别")
    print("  - 视频窗口实时显示检测结果")
    print("  - 按 'q' 键退出测试")
    print("\n测试手势:")
    print("  1. V 字手势（剪刀手）")
    print("  2. OK 手势")
    print("  3. 点赞手势（大拇指向上）")
    print("  4. 张开手掌")
    print("\n准备...")
    
    # 初始化
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    # 检查手势分类器
    print("\n[检查] 手势分类器状态...")
    if not processor._gesture_classifier:
        print("❌ 手势分类器未加载")
        return False
    
    if not isinstance(processor._gesture_classifier, MediaPipeGestureRecognizer):
        print(f"⚠️  不是 MediaPipe 识别器：{type(processor._gesture_classifier)}")
        return False
    
    print(f"✅ MediaPipe 手势识别器已加载")
    print(f"   识别器状态：{'✅' if processor._gesture_classifier._recognizer else '❌'}")
    print(f"   置信度阈值：{processor._gesture_classifier._confidence_threshold}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 摄像头已打开")
    print("\n开始测试，请比出手势...\n")
    await asyncio.sleep(2)
    
    start_time = time.time()
    gestures_detected = set()
    max_confidence = 0
    best_gesture = "NONE"
    frame_count = 0
    detect_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            
            # 直接调用 MediaPipe 识别手势（绕过 Layer 3）
            timestamp = time.time()
            
            if isinstance(processor._gesture_classifier, MediaPipeGestureRecognizer):
                # 直接调用 classify_gesture 方法
                gesture_name, conf, details = processor._gesture_classifier.classify_gesture(frame)
                
                if gesture_name:
                    detect_count += 1
                    gestures_detected.add(gesture_name)
                    
                    if conf > max_confidence:
                        max_confidence = conf
                        best_gesture = gesture_name
                    
                    # 每 30 帧打印一次
                    if frame_count % 30 == 0:
                        print(f"[Frame {frame_count}] 🎯 手势：{gesture_name}, 置信度：{conf:.3f}")
                else:
                    gesture_name = "NONE"
                    conf = 0.0
            else:
                gesture_name = "NONE"
                conf = 0.0
            
            # 在视频上绘制信息
            display_frame = frame.copy()
            
            # 手势信息（大字体，红色）
            gesture_display = gesture_name if gesture_name else 'NONE'
            cv2.putText(display_frame, f"Gesture: {gesture_display}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Conf: {conf:.3f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # 检测到的所有手势（蓝色）
            if gestures_detected:
                y_offset = 150
                for i, gesture in enumerate(sorted(gestures_detected)):
                    cv2.putText(display_frame, f"Detected: {gesture}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_offset += 30
            
            # 统计信息（白色）
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"Time: {elapsed:4.1f}s", (10, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detect: {detect_count}", (10, 310),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示视频窗口
            cv2.imshow('Layer 4 Manual Test - Press q to exit', display_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 用户请求退出...")
                break
        
        # 打印结果
        print("\n\n" + "=" * 80)
        print(" 测试结果")
        print("=" * 80)
        print(f"测试时长：{time.time() - start_time:.1f}秒")
        print(f"总帧数：{frame_count}帧")
        print(f"检测到手势的帧数：{detect_count}帧")
        print(f"检测成功率：{detect_count / frame_count * 100:.1f}%" if frame_count > 0 else "N/A")
        print(f"\nLayer 4 (手势识别):")
        print(f"  检测到的手势：{gestures_detected}")
        print(f"  最佳手势：{best_gesture}")
        print(f"  最高置信度：{max_confidence:.3f}")
        print(f"  状态：{'✅ 通过' if len(gestures_detected) > 0 else '❌ 失败'}")
        print("\n" + "=" * 80)
        
        # 保存截图
        screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"layer4_manual_{timestamp}.png")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 截图已保存：{screenshot_path}")
        
        if len(gestures_detected) > 0:
            print("✅ Layer 4 手势识别通过！")
            return True
        else:
            print("❌ Layer 4 未检测到手势")
            return False
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test())
