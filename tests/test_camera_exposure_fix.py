# File: tests/test_camera_exposure_fix.py
"""
摄像头曝光修复测试

尝试调整摄像头参数来改善画面质量。
"""

import cv2
import time


def adjust_camera_settings(cap: cv2.VideoCapture):
    """
    调整摄像头参数以改善画面质量
    """
    print("\n 当前摄像头参数:")
    print(f"  亮度 (CAP_PROP_BRIGHTNESS): {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"  对比度 (CAP_PROP_CONTRAST): {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"  饱和度 (CAP_PROP_SATURATION): {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"  色调 (CAP_PROP_HUE): {cap.get(cv2.CAP_PROP_HUE)}")
    print(f"  增益 (CAP_PROP_GAIN): {cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"  曝光 (CAP_PROP_EXPOSURE): {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"  自动曝光 (CAP_PROP_AUTO_EXPOSURE): {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    
    # 尝试调整参数
    print("\n 调整摄像头参数...")
    
    # 方法 1: 强制启用自动曝光（关键！）
    print("  强制启用自动曝光...")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = 自动模式（某些摄像头）
    time.sleep(0.5)
    
    # 方法 2: 设置曝光时间为负值（自动）
    print("  设置自动曝光时间...")
    cap.set(cv2.CAP_PROP_EXPOSURE, -13)  # 更负的值 = 更亮
    time.sleep(0.5)
    
    # 方法 3: 手动增加亮度
    print("  手动增加亮度...")
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)  # 增加亮度值
    time.sleep(0.3)
    
    # 方法 4: 调整对比度
    print("  调整对比度...")
    cap.set(cv2.CAP_PROP_CONTRAST, 50)
    time.sleep(0.3)
    
    # 方法 5: 调整增益
    print("  调整增益...")
    cap.set(cv2.CAP_PROP_GAIN, 100)  # 增加增益
    time.sleep(0.3)
    
    print("\n 调整后的参数:")
    print(f"  亮度：{cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"  对比度：{cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"  增益：{cap.get(cv2.CAP_PROP_GAIN)}")
    print(f"  曝光：{cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"  自动曝光：{cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")


def test_camera_with_fix(device_id=1):
    """
    测试摄像头并应用曝光修复
    """
    print("=" * 60)
    print(" 摄像头曝光修复测试")
    print("=" * 60)
    
    print(f"\n 打开摄像头 (设备 {device_id})...")
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f" 无法打开设备 {device_id}")
        return
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(" 摄像头已启动")
    
    # 调整参数
    adjust_camera_settings(cap)
    
    print("\n 按 'q' 退出测试")
    print(" 按 'r' 重置参数")
    print(" 按 'a' 启用自动曝光")
    print(" 按 'm' 手动增加亮度")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # 显示帧信息
        avg_brightness = frame.mean()
        elapsed = frame_count / (time.time() - start_time + 0.001)
        
        info_text = [
            f"Frame: {frame_count}",
            f"FPS: {elapsed:.1f}",
            f"Avg Brightness: {avg_brightness:.1f}",
            f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS):.2f}",
            f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE):.2f}",
            f"Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Camera Exposure Fix Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n 重置参数...")
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            cap.set(cv2.CAP_PROP_GAIN, 0.5)
        elif key == ord('a'):
            print("\n 启用自动曝光...")
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        elif key == ord('m'):
            print("\n 手动增加亮度...")
            current = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, min(1.0, current + 0.1))
            cap.set(cv2.CAP_PROP_GAIN, min(1.0, current + 0.1))
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n 测试完成:")
    print(f"   总帧数：{frame_count}")


if __name__ == "__main__":
    try:
        # 先尝试设备 1，如果失败则尝试设备 0
        for device_id in [1, 0]:
            cap = cv2.VideoCapture(device_id)
            if cap.isOpened():
                cap.release()
                test_camera_with_fix(device_id)
                break
            else:
                cap.release()
                print(f"设备 {device_id} 不可用，尝试下一个...")
    except KeyboardInterrupt:
        print("\n\n 测试中断")
    except Exception as e:
        print(f"\n 测试失败：{e}")
        import traceback
        traceback.print_exc()
