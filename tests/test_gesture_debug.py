# File: tests/test_gesture_debug.py
"""
手势识别调试测试

详细记录 MediaPipe 的每一步输出，帮助诊断问题
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

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(name)s] %(message)s'
)

async def test():
    print("\n" + "=" * 80)
    print(" 手势识别调试测试")
    print("=" * 80)
    print("\n说明:")
    print("  - 详细记录 MediaPipe 的每一步输出")
    print("  - 视频窗口实时显示检测结果")
    print("  - 按 'q' 键退出测试")
    print("\n测试手势:")
    print("  1. V 字手势（剪刀手）")
    print("  2. OK 手势")
    print("  3. 点赞手势（大拇指向上）")
    print("  4. 张开手掌")
    print("\n提示:")
    print("  - 确保手部光线充足")
    print("  - 手掌面向摄像头")
    print("  - 手指完全伸展")
    print("  - 距离摄像头 50-80cm")
    print("\n准备...")
    
    # 初始化
    print("\n[初始化] 加载视觉处理器...")
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    # 检查手势分类器
    print("\n[检查] 手势分类器状态...")
    if processor._gesture_classifier:
        print(f"✅ 手势分类器已加载")
        print(f"   类型：{type(processor._gesture_classifier).__name__}")
        
        # 检查是否是 MediaPipe
        from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer
        if isinstance(processor._gesture_classifier, MediaPipeGestureRecognizer):
            print(f"✅ MediaPipe 手势识别器")
            print(f"   识别器状态：{'✅ 已加载' if processor._gesture_classifier._recognizer else '❌ 未加载'}")
            print(f"   置信度阈值：{processor._gesture_classifier._confidence_threshold}")
        else:
            print(f"⚠️  使用其他手势识别器")
    else:
        print(f"❌ 手势分类器未加载")
        return False
    
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
            timestamp = time.time()
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.03)
            
            # 获取状态
            mem = processor.shared_memory
            
            gesture_type = mem.get('gesture_type', 'NONE')
            gesture_conf = mem.get('gesture_confidence', 0)
            gesture_method = mem.get('gesture_method', 'unknown')
            
            # 记录手势
            if gesture_type and gesture_type != 'NONE':
                detect_count += 1
                gestures_detected.add(gesture_type)
                if gesture_conf > max_confidence:
                    max_confidence = gesture_conf
                    best_gesture = gesture_type
                
                # 打印详细信息
                if frame_count % 30 == 0:  # 每 30 帧打印一次
                    print(f"[Frame {frame_count}] 🎯 手势：{gesture_type}, 置信度：{gesture_conf:.3f}, 方法：{gesture_method}")
            
            # 在视频上绘制信息
            display_frame = frame.copy()
            
            # 手势信息（大字体，红色）
            gesture_display = gesture_type if gesture_type else 'NONE'
            cv2.putText(display_frame, f"Gesture: {gesture_display}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Conf: {gesture_conf:.3f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Method: {gesture_method}", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 检测到的所有手势（蓝色）
            if gestures_detected:
                y_offset = 190
                for i, gesture in enumerate(sorted(gestures_detected)):
                    cv2.putText(display_frame, f"Detected: {gesture}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_offset += 30
            
            # 统计信息（白色）
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"Time: {elapsed:4.1f}s", (10, 350),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 380),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detect: {detect_count}", (10, 410),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示视频窗口
            cv2.imshow('Gesture Debug Test - Press q to exit', display_frame)
            
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
        screenshot_path = os.path.join(screenshot_dir, f"gesture_debug_{timestamp}.png")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 截图已保存：{screenshot_path}")
        
        if len(gestures_detected) > 0:
            print("✅ Layer 4 手势识别通过！")
            return True
        else:
            print("❌ Layer 4 未检测到手势")
            print("\n调试建议:")
            print("  1. 检查 MediaPipe 是否正确加载")
            print("  2. 检查模型文件是否存在")
            print("  3. 运行 test_mediapipe_diagnosis.py 进行详细诊断")
            print("  4. 确保光线充足，手部清晰")
            return False
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test())
