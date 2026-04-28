# File: tests/test_camera_settings.py
"""
检查摄像头设置
"""

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 无法打开摄像头")
    exit(1)

# 获取当前设置
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 60)
print(" 摄像头当前设置")
print("=" * 60)
print(f"分辨率：{width}x{height}")
print(f"FPS: {fps}")

# 尝试设置更高 FPS
print("\n尝试设置 640x480 @ 30fps...")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# 重新读取
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"分辨率：{width}x{height}")
print(f"FPS: {fps}")

# 测试实际 FPS
import time
start_time = time.time()
frame_count = 0
test_duration = 3

print(f"\n开始测试 ({test_duration}秒)...")

while time.time() - start_time < test_duration:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\n实际 FPS: {avg_fps:.2f} (帧数：{frame_count}, 时间：{total_time:.2f}秒)")

cap.release()
