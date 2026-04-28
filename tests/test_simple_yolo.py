# File: tests/test_simple_yolo.py
"""
简单 YOLO 人体检测测试
"""

import cv2
import time
from zulong.l1c.vision_model_loader import get_vision_model_loader

print("=" * 80)
print(" YOLO 人体检测测试")
print("=" * 80)

# 加载模型
loader = get_vision_model_loader()
loader.load_all_models()

print("\n✅ 模型加载完成")

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

print("✅ 摄像头已打开")
print("\n请坐在摄像头前，测试将检测人体 10 秒...\n")

time.sleep(2)

start_time = time.time()
frame_count = 0
detect_count = 0

try:
    while time.time() - start_time < 10:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO 检测
        human_detected, bbox_info = loader.detect_human(frame)
        
        if human_detected and bbox_info:
            detect_count += 1
            conf = bbox_info.get('confidence', 0)
            print(f"帧 {frame_count}: ✅ 检测到人 (置信度：{conf:.2f})")
            
            # 绘制框
            x1, y1, x2, y2 = [int(c) for c in bbox_info['bbox']]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            print(f"帧 {frame_count}: ❌ 未检测到人")
        
        # 显示
        cv2.putText(frame, f"Detect: {detect_count}/{frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("YOLO Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print(f"\n测试结果：检测到 {detect_count}/{frame_count} 帧")
    print(f"检测率：{detect_count/frame_count*100:.1f}%")
    
finally:
    cap.release()
    cv2.destroyAllWindows()
