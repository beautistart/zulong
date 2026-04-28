# File: tests/test_layer3_4_quick.py
"""
Layer 3 & Layer 4 快速测试（简化版）

用途：快速验证动作分类和手势识别是否工作
特点：
- 无需选择菜单，直接运行
- 实时显示状态
- 自动检测并记录最高分数
- 10 秒快速测试
- 视频窗口显示实时检测结果
- 自动保存截图
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("QuickTest")


async def quick_test():
    """快速测试 Layer 3 和 Layer 4"""
    
    print("\n" + "=" * 80)
    print(" Layer 3 & Layer 4 快速测试")
    print("=" * 80)
    print("\n说明:")
    print("  - 视频窗口会实时显示检测结果")
    print("  - 请先挥手 2-3 秒，然后比出手势")
    print("  - 按 'q' 键退出测试")
    print("  - 截图会在退出时自动保存")
    print("\n准备...")
    
    # 初始化
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return False
    
    print("✅ 初始化完成，请开始挥手！\n")
    await asyncio.sleep(2)
    
    start_time = time.time()
    max_action_score = 0
    best_intent = "UNKNOWN"
    gestures_detected = set()
    frame_count = 0
    
    try:
        while True:  # 无限循环，直到用户按'q'退出
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取摄像头画面")
                break
            
            frame_count += 1
            timestamp = time.time()
            processor.feed_frame(frame, timestamp)
            await asyncio.sleep(0.05)
            
            # 获取状态
            mem = processor.shared_memory
            
            action_score = mem.get('action_score', 0)
            intent_type = mem.get('intent_type', 'UNKNOWN')
            gesture_type = mem.get('gesture_type', 'NONE')
            gesture_conf = mem.get('gesture_confidence', 0)
            
            # 更新最大值
            if action_score > max_action_score:
                max_action_score = action_score
                best_intent = intent_type
            
            # 记录手势
            if gesture_type and gesture_type != 'NONE':
                gestures_detected.add(gesture_type)
            
            # 在视频上绘制信息
            display_frame = frame.copy()
            
            # Layer 3 信息（绿色）
            cv2.putText(display_frame, f"L3 Score: {action_score:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Intent: {intent_type}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Layer 4 信息（红色）
            gesture_display = gesture_type if gesture_type else 'NONE'
            cv2.putText(display_frame, f"Gesture: {gesture_display}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"Conf: {gesture_conf:.2f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 时间显示（白色）
            elapsed = time.time() - start_time
            cv2.putText(display_frame, f"Time: {elapsed:4.1f}s", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 帧数显示
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 180),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示视频窗口
            cv2.imshow('Layer 3 & 4 Test - Press q to exit', display_frame)
            
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
        print(f"\nLayer 3 (动作分类):")
        print(f"  最大分数：{max_action_score:.3f}")
        print(f"  最佳意图：{best_intent}")
        print(f"  状态：{'✅ 通过' if max_action_score > 0.5 else '❌ 失败'}")
        print(f"\nLayer 4 (手势识别):")
        print(f"  检测到的手势：{gestures_detected}")
        print(f"  状态：{'✅ 通过' if len(gestures_detected) > 0 else '❌ 失败'}")
        print("\n" + "=" * 80)
        
        # 保存最后一帧作为截图
        import os
        from datetime import datetime
        screenshot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'screenshots')
        os.makedirs(screenshot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(screenshot_dir, f"layer3_4_test_{timestamp}.png")
        cv2.imwrite(screenshot_path, display_frame)
        print(f"\n📸 截图已保存：{screenshot_path}")
        
        if max_action_score > 0.5 or len(gestures_detected) > 0:
            print("✅ 测试通过！视觉模块工作正常！")
            return True
        else:
            print("⚠️  未检测到明显动作或手势")
            print("提示:")
            print("  - 确保光线充足")
            print("  - 动作幅度大一些")
            print("  - 手势清晰、稳定")
            return False
        
    finally:
        cap.release()
        processor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(quick_test())
