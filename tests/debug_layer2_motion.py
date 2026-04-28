# File: tests/debug_layer2_motion.py
"""
Layer 2 ROI 运动检测实时调试

目标:
1. 实时显示 YOLO 检测框
2. 实时显示 ROI 区域
3. 实时显示运动像素数
4. 帮助理解为什么 Layer 2 未触发

TSD v1.7 对应:
- 4.4 感知预处理
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def debug_layer2_motion():
    """实时调试 Layer 2 ROI 运动检测"""
    
    print("="*60)
    print("🔍 Layer 2 ROI 运动检测实时调试")
    print("="*60)
    
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
        
        print("\n📋 测试说明:")
        print("   1. 站在摄像头前 1-2 米")
        print("   2. 先静止 1 秒（让系统建立 baseline）")
        print("   3. 然后挥动手臂（像打招呼一样）")
        print("   4. 观察窗口中的实时数据")
        print("   5. 按 Q 键退出")
        
        print("\n🎯 开始测试...")
        time.sleep(2)
        
        # 实时显示 30 秒
        start_time = time.time()
        frame_count = 0
        motion_detected_count = 0
        
        while (time.time() - start_time) < 30.0:
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                continue
            
            frame_count += 1
            
            # 🎯 关键：将帧推送给处理器
            processor.feed_frame(frame, timestamp)
            
            # 等待处理完成
            await asyncio.sleep(0.033)  # 30FPS
            
            # 获取共享内存状态
            human_detected = processor.shared_memory.get('human_detected', False)
            human_bbox = processor.shared_memory.get('human_bbox')
            motion_pixels = processor.shared_memory.get('motion_pixels', 0)
            gesture_type = processor.shared_memory.get('gesture_type')
            
            # 绘制 YOLO 检测框
            if human_detected and human_bbox:
                x1, y1, x2, y2 = [int(coord) for coord in human_bbox]
                
                # 绘制人体检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制 ROI 扩展区域
                margin = 50
                roi_x1, roi_y1 = max(0, x1 - margin), max(0, y1 - margin)
                roi_x2, roi_y2 = min(640, x2 + margin), min(480, y2 + margin)
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 1)
                
                # 标注 ROI 区域
                cv2.putText(frame, "ROI Region", (roi_x1, roi_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 显示状态信息
            status_lines = [
                f"Frame: {frame_count}",
                f"Human: {'YES' if human_detected else 'NO'}",
                f"Motion Pixels: {motion_pixels}",
                f"Motion Detected: {'YES' if motion_pixels > 0 else 'NO'}",
                f"Gesture: {gesture_type if gesture_type else 'None'}",
            ]
            
            # 如果检测到运动，用红色显示
            if motion_pixels > 0:
                motion_detected_count += 1
                color = (0, 0, 255)  # 红色
            else:
                color = (0, 255, 0)  # 绿色
            
            for i, line in enumerate(status_lines):
                y = 30 + (i * 35)
                cv2.putText(frame, line, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 显示帧
            cv2.imshow("Layer 2 Motion Debug - Press Q to Exit", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            time.sleep(0.033)  # 30 FPS
        
        cv2.destroyWindow("Layer 2 Motion Debug - Press Q to Exit")
        
        # 统计结果
        print("\n" + "="*60)
        print("📊 测试结果统计")
        print("="*60)
        print(f"总帧数：{frame_count}")
        print(f"检测到运动的帧数：{motion_detected_count}")
        print(f"运动检测率：{motion_detected_count/frame_count*100:.1f}%")
        
        camera.stop()
        camera.disconnect()
        
        return motion_detected_count > 0
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(debug_layer2_motion())
    
    print("\n" + "="*60)
    if success:
        print("✅ Layer 2 检测到运动！")
        print("="*60)
    else:
        print("❌ Layer 2 未检测到运动")
        print("="*60)
        print("\n📋 可能原因:")
        print("   1. 没有挥动手臂（静止不动）")
        print("   2. ROI 区域没有包含手部")
        print("   3. 运动阈值过高")
        print("   4. 光线太暗，摄像头看不清运动")
    
    sys.exit(0 if success else 1)
