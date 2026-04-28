# File: tests/test_software_brightness.py
"""
软件级亮度增强测试

在捕获帧后自动提升亮度，解决摄像头硬件无法调整的问题。
"""

import cv2
import numpy as np
import time


def enhance_brightness(frame: np.ndarray, alpha: float = 1.5, beta: int = 30) -> np.ndarray:
    """
    软件级亮度增强
    
    formula: new_image = alpha * old_image + beta
    
    Args:
        frame: 原始帧
        alpha: 对比度控制 (1.0-3.0)
        beta: 亮度控制 (0-100)
    
    Returns:
        增强后的帧
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def test_software_brightness(device_id=1):
    """
    测试软件亮度增强效果
    """
    print("=" * 60)
    print(" 软件级亮度增强测试")
    print("=" * 60)
    
    print(f"\n 打开摄像头 (设备 {device_id})...")
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f" 无法打开设备 {device_id}")
        return
    
    print(" 摄像头已启动")
    
    print("\n 按 'q' 退出")
    print(" 按 '1-5' 切换增强级别:")
    print("   1: 原始帧")
    print("   2: 亮度 +30")
    print("   3: 亮度 +50")
    print("   4: 亮度 +30, 对比度 1.5x")
    print("   5: 亮度 +70, 对比度 2.0x")
    
    level = 4
    brightness_settings = {
        1: (1.0, 0),    # 原始
        2: (1.0, 30),   # 亮度 +30
        3: (1.0, 50),   # 亮度 +50
        4: (1.5, 30),   # 亮度 +30, 对比度 1.5x
        5: (2.0, 70),   # 亮度 +70, 对比度 2.0x
    }
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # 应用亮度增强
        alpha, beta = brightness_settings[level]
        enhanced_frame = enhance_brightness(frame, alpha, beta)
        
        # 计算亮度
        orig_brightness = frame.mean()
        enhanced_brightness = enhanced_frame.mean()
        
        # 显示信息
        info_text = [
            f"Level: {level} (alpha={alpha}, beta={beta})",
            f"Frame: {frame_count}",
            f"Original Brightness: {orig_brightness:.1f}",
            f"Enhanced Brightness: {enhanced_brightness:.1f}",
        ]
        
        # 上半部分显示原始帧，下半部分显示增强帧
        combined = np.vstack([
            cv2.resize(frame, (640, 240)),
            cv2.resize(enhanced_frame, (640, 240))
        ])
        
        for i, text in enumerate(info_text):
            cv2.putText(combined, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(combined, "Original (Top) / Enhanced (Bottom)", (10, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Software Brightness Test", combined)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key in [ord(str(i)) for i in range(1, 6)]:
            level = int(chr(key))
            print(f"\n 切换到级别 {level}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n 测试完成")


if __name__ == "__main__":
    try:
        for device_id in [1, 0]:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                cap.release()
                test_software_brightness(device_id)
                break
    except KeyboardInterrupt:
        print("\n\n 测试中断")
    except Exception as e:
        print(f"\n 测试失败：{e}")
        import traceback
        traceback.print_exc()
