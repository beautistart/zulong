# File: tests/test_camera_diagnostic.py
"""
摄像头诊断脚本

检查所有可用的摄像头设备。
"""

import cv2

def test_camera(device_id):
    """测试指定设备 ID 的摄像头"""
    print(f"\n 测试摄像头设备 {device_id}...")
    
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"  设备 {device_id}: ❌ 无法打开")
        return False
    
    # 尝试读取一帧
    ret, frame = cap.read()
    
    if not ret:
        print(f"  设备 {device_id}: ❌ 无法读取帧")
        cap.release()
        return False
    
    # 获取摄像头属性
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  设备 {device_id}: ✅ 成功")
    print(f"    - 分辨率：{width}x{height}")
    print(f"    - FPS: {fps}")
    print(f"    - 帧形状：{frame.shape}")
    
    cap.release()
    return True


if __name__ == "__main__":
    print("=" * 60)
    print(" 摄像头诊断")
    print("=" * 60)
    
    # 测试设备 0-3
    for device_id in range(4):
        test_camera(device_id)
    
    print("\n 诊断完成")
