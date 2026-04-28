# File: tests/test_ok_gesture.py
"""
OK 手势专项测试

针对 OK 手势识别进行优化测试
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
    print(" OK 手势专项测试")
    print("=" * 80)
    print("\n说明:")
    print("  - 专门测试 OK 手势识别")
    print("  - 视频窗口实时显示检测结果")
    print("  - 按 'q' 键退出测试")
    print("\nOK 手势要点:")
    print("  1. 拇指和食指形成完整圆圈")
    print("  2. 其他三指伸直")
    print("  3. 手掌面向摄像头")
    print("  4. 距离 50-80cm")
    print("\n提示:")
    print("  - 先挥手提高 L3 分数")
    print("  - 然后保持动作比 OK 手势")
    print("  - 确保光线充足")
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
    print(f"   交互阈值：{processor._config['interact_threshold']}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 摄像头已打开")
    print("\n开始测试，请先挥手然后比 OK 手势...\n")
    await asyncio.sleep(2)
    
    start_time = time.time()
    ok_count = 0
    total_detect = 0
    gestures_detected = set()
    max_ok_confidence = 0
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
            await asyncio.sleep(0.03)
            
            # 获取状态
            mem = processor.shared_memory
            
            action_score = mem.get('action_score', 0)
            intent_type = mem.get('intent_type', 'UNKNOWN')
            gesture_type = mem.get('gesture_type', 'NONE')
            gesture_conf = mem.get('gesture_confidence', 0)
            
            # 统计 OK 手势
            if gesture_type and gesture_type != 'NONE':
                total_detect += 1
                gestures_detected.add(gesture_type)
                
                if gesture_type == "OK_Gesture":
                    ok_count += 1
                    if gesture_conf > max_ok_confidence:
                        max_ok_confidence = gesture_conf
                        print(f"\n🎯 OK 手势！置信度：{gesture_conf:.3f}")
                
                # 每 30 帧打印一次
                if frame_count % 30 == 0:
                    print(f"[Frame {frame_count}] L3: {action_score:.3f}, 手势：{gesture_type}, 置信度：{gesture_conf:.3f}")
            
            # 在视频上绘制信息
            display_frame = frame.copy()
            
            # Layer 3 信息
            score_color = (0, 255, 0) if action_score >= 0.35 else (0, 255, 255)
            cv2.putText(display_frame, f"L3 Score: {action_score:.3f}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, score_color, 2)
            cv2.putText(display_frame, f"Intent: {intent_type}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Layer 4 信息
            gesture_display = gesture_type if gesture_type else 'NONE'
            color = (0, 255, 0) if gesture_type == "OK_Gesture" else (0, 0, 255)
            cv2.putText(display_frame, f"Gesture: {gesture_display}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            cv2.putText(display_frame, f"Conf: {gesture_conf:.3f}", (10, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # OK 手势统计
            cv2.putText(display_frame, f"OK Count: {ok_count}", (10, 200),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Total: {total_detect}", (10, 230),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 统计信息
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"Time: {elapsed:4.1f}s", (10, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 310),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示视频窗口
            cv2.imshow('OK Gesture Test - Press q to exit', display_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n👋 用户请求退出...")
                break
        
        # 打印结果
        print("\n\n" + "=" * 80)
        print(" OK 手势测试结果")
        print("=" * 80)
        print(f"测试时长：{time.time() - start_time:.1f}秒")
        print(f"总帧数：{frame_count}帧")
        print(f"检测到手势：{total_detect}帧")
        print(f"OK 手势次数：{ok_count}帧")
        print(f"OK 手势占比：{ok_count / total_detect * 100:.1f}%" if total_detect > 0 else "N/A")
        print(f"\n检测到的所有手势：{gestures_detected}")
        print(f"OK 手势最高置信度：{max_ok_confidence:.3f}")
        print("\n" + "=" * 80)
        
        # 分析
        if ok_count > 0:
            print("✅ OK 手势识别成功！")
            if max_ok_confidence >= 0.5:
                print("✅ 置信度良好")
            elif max_ok_confidence >= 0.3:
                print("⚠️  置信度一般，可以降低阈值")
            else:
                print("⚠️  置信度较低，建议检查手势标准度")
        else:
            print("❌ 未检测到 OK 手势")
            print("\n可能原因:")
            print("  1. OK 手势不够标准")
            print("  2. Layer 3 分数未达到 0.35")
            print("  3. 光线不足或角度不对")
            print("  4. MediaPipe 对 OK 手势识别能力有限")
            print("\n建议:")
            print("  - 确保拇指和食指形成完整圆圈")
            print("  - 其他三指伸直")
            print("  - 手掌完全面向摄像头")
            print("  - 先挥手提高 L3 分数")
        
        print("\n" + "=" * 80)
        
        # 保存截图
        screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"ok_gesture_test_{timestamp}.png")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 截图已保存：{screenshot_path}")
        
        return ok_count > 0
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(test())
