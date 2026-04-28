# File: tests/test_yolo_direct.py
"""
YOLO 模型直接测试

验证 YOLO 模型是否能正常检测人体。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO


def test_yolo_human_detection():
    """测试 YOLO 人体检测"""
    print("=" * 60)
    print(" YOLO 人体检测测试")
    print("=" * 60)
    
    # 加载 YOLO 模型
    print("\n 加载 YOLOv8n 模型...")
    yolo_path = Path(__file__).parent.parent / "yolov8n.pt"
    
    if not yolo_path.exists():
        print(f" 模型文件不存在：{yolo_path}")
        return
    
    model = YOLO(str(yolo_path))
    print(" YOLO 模型加载成功")
    
    # COCO 数据集中的人体类别 ID 是 0
    print(" 人体类别 ID: 0 (person)")
    
    # 打开摄像头
    print("\n 打开摄像头...")
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
        
        # YOLO 推理
        results = model(frame, verbose=False, conf=0.5)
        
        # 解析结果
        human_detected = False
        bboxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 只检测人体 (class 0)
                    if cls == 0:
                        human_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bboxes.append((x1, y1, x2, y2, conf))
        
        if human_detected:
            human_count += 1
            
            # 绘制检测框
            for x1, y1, x2, y2, conf in bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Human {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            status = f"Human Detected! ({len(bboxes)})"
            color = (0, 255, 0)
        else:
            status = "No Human"
            color = (0, 0, 255)
        
        # 显示统计
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Human Frames: {human_count}/{frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Rate: {human_count/frame_count*100:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("YOLO Human Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if frame_count > 0:
        print(f"\n 测试完成:")
        print(f"   总帧数：{frame_count}")
        print(f"   检测到人体：{human_count}")
        print(f"   检测率：{human_count/frame_count*100:.1f}%")
    else:
        print("\n 未捕获到任何帧")
        print(" 可能原因:")
        print("   1. 摄像头被其他程序占用")
        print("   2. 摄像头设备号不正确")
        print("   3. 摄像头驱动问题")
        print("\n 建议:")
        print("   - 关闭所有摄像头应用")
        print("   - 尝试其他摄像头设备号 (0, 1, 2...)")


if __name__ == "__main__":
    try:
        test_yolo_human_detection()
    except KeyboardInterrupt:
        print("\n\n 测试中断")
    except Exception as e:
        print(f"\n 测试失败：{e}")
        import traceback
        traceback.print_exc()
