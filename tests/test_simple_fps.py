# File: tests/test_simple_fps.py
"""
简单 FPS 测试
"""

import time
import cv2

print("=" * 60)
print(" 简单 FPS 测试 (仅摄像头采集)")
print("=" * 60)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

start_time = time.time()
frame_count = 0
test_duration = 5  # 秒

print(f"\n开始测试 ({test_duration}秒)...")

while time.time() - start_time < test_duration:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # 显示 FPS
    elapsed = time.time() - start_time
    current_fps = frame_count / elapsed if elapsed > 0 else 0
    
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Simple FPS Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\n总帧数：{frame_count}")
print(f"总时间：{total_time:.2f}秒")
print(f"平均 FPS: {avg_fps:.2f}")

cap.release()
cv2.destroyAllWindows()
