# File: tests/test_gesture_with_motion.py
"""
真实手势识别测试（带动作指导）

核心逻辑:
根据 TSD v1.7 架构，手势识别需要:
1. Layer 1: YOLO 检测人体 ✅ (已验证 100% 检测率)
2. Layer 2: ROI 检测到运动 ⚠️ (需要挥动手臂)
3. Layer 3: 动作分类判断意图
4. Layer 4: 鹰眼模式 + 手势识别

关键：展示手势时必须有明显的手臂/身体移动！

测试方法:
1. 站在摄像头前 1-2 米
2. 听到提示后，先静止 1 秒
3. 然后**挥动手臂**展示手势 2 秒
4. 系统会检测到运动并触发手势识别
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_gesture_with_motion():
    """带动作指导的手势测试"""
    
    print("="*60)
    print("🎯 真实手势识别测试（带动作指导）")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        from zulong.l0.usb_camera import USBCamera
        from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 等待系统稳定
        print("\n⏳ 等待系统稳定 (3 秒)...")
        time.sleep(3)
        
        # 测试 5 种手势
        gestures = ['open_palm', 'fist', 'v_sign', 'ok_sign', 'thumbs_up']
        
        for gesture_name in gestures:
            print("\n" + "="*60)
            print(f"🎯 测试手势：{gesture_name}")
            print("="*60)
            
            print(f"\n📋 动作指导:")
            print(f"   1. 站在摄像头前 1-2 米")
            print(f"   2. 准备展示 '{gesture_name}' 手势")
            print(f"   3. **关键：展示时挥动手臂** (像打招呼一样)")
            print(f"   4. 保持手势 2 秒")
            
            # 倒计时
            for i in range(3, 0, -1):
                print(f"   ⏳ {i}秒后开始...")
                time.sleep(1)
            
            print(f"\n👉 现在请展示 '{gesture_name}' 手势并挥动手臂！")
            print(f"📸 开始采集 (5 秒)...")
            
            # 采集 5 秒
            start_time = time.time()
            success_count = 0
            total_frames = 0
            
            while (time.time() - start_time) < 5.0:
                frame, timestamp = camera.get_latest_frame()
                
                if frame is None:
                    continue
                
                total_frames += 1
                
                # 🎯 关键：将帧推送给处理器
                processor.feed_frame(frame, timestamp)
                
                # 等待处理完成（增加等待时间）
                await asyncio.sleep(0.1)  # 给更多时间让异步处理完成
                
                # 获取检测结果
                human_detected = processor.shared_memory.get('human_detected', False)
                human_bbox = processor.shared_memory.get('human_bbox')
                motion_pixels = processor.shared_memory.get('motion_pixels', 0)
                motion_detected = motion_pixels > 0
                gesture_type = processor.shared_memory.get('gesture_type')
                
                # 🐛 调试：打印原始值
                if total_frames % 30 == 1:  # 每秒打印一次
                    print(f"   [Debug] human_detected={human_detected} (type={type(human_detected)}), motion_pixels={motion_pixels}")
                
                # 🎨 绘制 YOLO 检测框（绿色）
                if human_detected and human_bbox:
                    x1, y1, x2, y2 = [int(coord) for coord in human_bbox]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 🎨 绘制 ROI 区域（蓝色）
                    margin = 50
                    roi_x1, roi_y1 = max(0, x1 - margin), max(0, y1 - margin)
                    roi_x2, roi_y2 = min(640, x2 + margin), min(480, y2 + margin)
                    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 1)
                
                # 显示状态
                status_lines = []
                status_lines.append(f"Human: {'✅' if human_detected else '❌'}")
                status_lines.append(f"Motion: {'✅' if motion_detected else '❌'}")
                status_lines.append(f"Gesture: {gesture_type if gesture_type else 'None'}")
                status_lines.append(f"Motion Pixels: {processor.shared_memory.get('motion_pixels', 0)}")
                
                # 在帧上显示状态
                for i, line in enumerate(status_lines):
                    y = 30 + (i * 30)
                    cv2.putText(frame, line, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 如果识别到手势
                if gesture_type:
                    success_count += 1
                    cv2.putText(frame, f"SUCCESS: {gesture_type}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 显示帧（按 Q 退出）
                cv2.imshow(f"Gesture Test - {gesture_name}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.033)  # 30 FPS
            
            cv2.destroyWindow(f"Gesture Test - {gesture_name}")
            
            # 统计结果
            print(f"\n📊 {gesture_name} 识别结果:")
            print(f"   - 总帧数：{total_frames}")
            print(f"   - 成功识别：{success_count}")
            if total_frames > 0:
                print(f"   - 识别率：{success_count/total_frames*100:.1f}%")
            
            # 休息
            print(f"\n⏳ 休息 2 秒，准备下一个手势...")
            time.sleep(2)
        
        camera.stop()
        camera.disconnect()
        
        print("\n" + "="*60)
        print("✅ 所有手势测试完成！")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gesture_with_motion())
    sys.exit(0 if success else 1)
