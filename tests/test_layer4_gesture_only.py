# File: tests/test_layer4_gesture_only.py
"""
Layer 4 手势识别单独测试（优化版）

专门用于测试手势识别，降低阈值提高灵敏度
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

logging.basicConfig(level=logging.WARNING)

async def test():
    print("\n" + "=" * 80)
    print(" Layer 4: 手势识别单独测试（优化版）")
    print("=" * 80)
    print("\n说明:")
    print("  - 降低手势置信度阈值到 0.15（默认 0.25）")
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
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    # 临时降低手势阈值
    processor._config['gesture_conf_threshold'] = 0.15
    print(f"✅ 手势置信度阈值已降低到：0.15")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 初始化完成，请开始比手势！\n")
    await asyncio.sleep(2)
    
    start_time = time.time()
    gestures_detected = set()
    max_confidence = 0
    best_gesture = "NONE"
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            timestamp = time.time()
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.03)  # 更快响应
            
            # 获取状态
            mem = processor.shared_memory
            
            gesture_type = mem.get('gesture_type', 'NONE')
            gesture_conf = mem.get('gesture_confidence', 0)
            
            # 记录手势
            if gesture_type and gesture_type != 'NONE':
                gestures_detected.add(gesture_type)
                if gesture_conf > max_confidence:
                    max_confidence = gesture_conf
                    best_gesture = gesture_type
            
            # 在视频上绘制信息
            display_frame = frame.copy()
            
            # 手势信息（大字体，红色）
            cv2.putText(display_frame, f"Gesture: {gesture_type}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Conf: {gesture_conf:.3f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # 检测到的所有手势（蓝色）
            if gestures_detected:
                y_offset = 150
                for i, gesture in enumerate(sorted(gestures_detected)):
                    cv2.putText(display_frame, f"Detected: {gesture}", (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    y_offset += 30
            
            # 时间显示（白色）
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"Time: {elapsed:4.1f}s", (10, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 330),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示视频窗口
            cv2.imshow('Layer 4 Gesture Test - Press q to exit', display_frame)
            
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
        screenshot_path = os.path.join(screenshot_dir, f"layer4_gesture_{timestamp}.png")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 截图已保存：{screenshot_path}")
        
        if len(gestures_detected) > 0:
            print("✅ Layer 4 手势识别通过！")
            return True
        else:
            print("❌ Layer 4 未检测到手势")
            print("\n建议:")
            print("  1. 确保光线充足，手部清晰")
            print("  2. 手掌完全面向摄像头")
            print("  3. 手指完全伸展")
            print("  4. 尝试不同的手势角度")
            print("  5. 距离摄像头 50-80cm")
            return False
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test())
