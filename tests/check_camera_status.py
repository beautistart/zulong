# File: tests/check_camera_status.py
"""
检查摄像头状态
"""
import cv2
import numpy as np

def check_camera(device_index=0):
    """检查摄像头状态"""
    print(f"\n检查摄像头 #{device_index}")
    print("=" * 60)
    
    # 尝试不同后端
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto"),
    ]
    
    for backend, name in backends:
        print(f"\n尝试后端：{name}")
        cap = cv2.VideoCapture(device_index, backend)
        
        if cap.isOpened():
            print(f"  ✅ 摄像头已打开")
            
            # 尝试读取帧
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✅ 读取帧成功：{frame.shape}")
                print(f"  📊 亮度：{np.mean(frame):.2f}")
            else:
                print(f"  ❌ 读取帧失败")
            
            cap.release()
        else:
            print(f"  ❌ 无法打开摄像头")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # 检查摄像头 0 和 1
    for i in range(2):
        check_camera(i)
