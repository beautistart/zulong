# File: tests/test_yolo_sync.py
"""
YOLO 同步测试

直接调用 Layer 1 方法，验证 YOLO 是否工作。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor


async def test_yolo_sync():
    """同步测试 YOLO 人体检测"""
    print("=" * 60)
    print(" YOLO 同步测试")
    print("=" * 60)
    
    # 初始化处理器
    print("\n 初始化 OptimizedVisionProcessor...")
    processor = OptimizedVisionProcessor()
    await processor.initialize(load_models=True)
    print(" 处理器初始化成功")
    
    # 打开摄像头（使用设备 0）
    print("\n 打开摄像头 (设备 0)...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" 无法打开摄像头")
        return
    
    print(" 摄像头已启动")
    print("\n 按 'q' 退出测试")
    
    frame_count = 0
    human_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        timestamp = time.time()
        
        # 直接调用 Layer 1 方法（同步）
        human_bboxes = processor._layer1_human_detection(frame)
        
        if human_bboxes:
            human_count += 1
            
            # 绘制检测框
            for bbox in human_bboxes:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "HUMAN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            status = f"Human Detected! ({len(human_bboxes)})"
            color = (0, 255, 0)
        else:
            status = "No Human"
            color = (0, 0, 255)
        
        # 显示统计
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Human Frames: {human_count}/{frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Rate: {human_count/frame_count*100:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("YOLO Sync Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n 测试完成:")
    print(f"   总帧数：{frame_count}")
    print(f"   检测到人体：{human_count}")
    if frame_count > 0:
        print(f"   检测率：{human_count/frame_count*100:.1f}%")


if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(test_yolo_sync())
    except KeyboardInterrupt:
        print("\n\n 测试中断")
    except Exception as e:
        print(f"\n 测试失败：{e}")
        import traceback
        traceback.print_exc()
