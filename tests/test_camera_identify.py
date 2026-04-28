# File: tests/test_camera_identify.py
"""
摄像头识别测试

同时打开摄像头 #0 和 #1，显示实时画面，帮助识别具体是哪个物理设备

功能:
1. 同时显示两个摄像头的画面
2. 在画面上显示设备 ID
3. 显示摄像头详细参数
4. 按 Q 键退出

使用方法:
- 运行脚本后，会看到两个窗口
- 窗口标题显示设备 ID
- 观察画面内容判断哪个是需要的摄像头
- 按 Q 键退出
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_camera_details(device_id: int) -> dict:
    """获取摄像头详细信息"""
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        return None
    
    details = {
        'device_id': device_id,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
        'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
        'saturation': cap.get(cv2.CAP_PROP_SATURATION),
        'autofocus': cap.get(cv2.CAP_PROP_AUTOFOCUS),
    }
    
    cap.release()
    return details


def add_info_overlay(frame: np.ndarray, device_id: int, details: dict) -> np.ndarray:
    """在帧上添加信息叠加层"""
    # 创建信息文本
    info_lines = [
        f"Camera #{device_id}",
        f"Resolution: {details['width']}x{details['height']}",
        f"Brightness: {details['brightness']}",
        f"Contrast: {details['contrast']}",
        f"Saturation: {details['saturation']}",
        f"AutoFocus: {'ON' if details['autofocus'] else 'OFF'}",
    ]
    
    # 在帧顶部添加黑色信息栏
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
    
    # 添加半透明效果
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 显示信息
    y_offset = 25
    for i, line in enumerate(info_lines):
        y = y_offset + (i * 20)
        cv2.putText(frame, line, (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame


def test_cameras():
    """测试并显示两个摄像头"""
    print("="*60)
    print("📷 摄像头识别测试")
    print("="*60)
    
    from zulong.l0.usb_camera import detect_available_cameras
    
    # 检测可用摄像头
    cameras = detect_available_cameras()
    
    if not cameras:
        print("\n❌ 未检测到任何摄像头")
        return
    
    print(f"\n✅ 检测到 {len(cameras)} 个摄像头")
    
    # 获取详细信息
    camera_details = {}
    for device_id in cameras[:2]:  # 最多测试前 2 个
        print(f"\n📹 摄像头 #{device_id} 详细信息:")
        details = get_camera_details(device_id)
        if details:
            camera_details[device_id] = details
            print(f"   分辨率：{details['width']}x{details['height']}")
            print(f"   亮度：{details['brightness']}")
            print(f"   对比度：{details['contrast']}")
            print(f"   饱和度：{details['saturation']}")
            print(f"   自动对焦：{'✅' if details['autofocus'] else '❌'}")
        else:
            print(f"   ❌ 无法获取详细信息")
    
    print("\n" + "="*60)
    print("🎥 正在打开摄像头...")
    print("按 Q 键退出测试")
    print("="*60)
    
    # 打开摄像头
    caps = []
    for device_id in cameras[:2]:
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if cap.isOpened():
            # 设置参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            caps.append((device_id, cap))
        else:
            print(f"❌ 无法打开摄像头 #{device_id}")
    
    if not caps:
        print("\n❌ 无法打开任何摄像头")
        return
    
    print(f"\n✅ 成功打开 {len(caps)} 个摄像头")
    
    # 显示画面
    print("\n📺 显示摄像头画面...")
    print("提示:")
    print("  - 观察画面内容判断摄像头位置")
    print("  - 画面上方显示设备 ID 和参数")
    print("  - 按 Q 键退出")
    
    try:
        while True:
            frames = []
            
            # 读取所有摄像头的帧
            for device_id, cap in caps:
                ret, frame = cap.read()
                
                if ret:
                    # 添加信息叠加层
                    frame = add_info_overlay(frame, device_id, camera_details.get(device_id, {}))
                    frames.append((device_id, frame))
                else:
                    print(f"⚠️ 摄像头 #{device_id} 读取失败")
            
            if not frames:
                print("❌ 所有摄像头都无法读取帧")
                break
            
            # 显示画面
            if len(frames) == 1:
                cv2.imshow(f"Camera {frames[0][0]}", frames[0][1])
            else:
                # 并排显示两个摄像头
                frame0 = frames[0][1]
                frame1 = frames[1][1]
                
                # 确保两个帧高度相同
                height = min(frame0.shape[0], frame1.shape[0])
                frame0 = cv2.resize(frame0, (640, height))
                frame1 = cv2.resize(frame1, (640, height))
                
                # 水平拼接
                combined = np.hstack([frame0, frame1])
                
                # 添加标题
                cv2.putText(combined, f"Camera #{frames[0][0]} (Left)", (10, height + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(combined, f"Camera #{frames[1][0]} (Right)", (650, height + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Camera Comparison - Press Q to Exit", combined)
            
            # 检查退出键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except Exception as e:
        print(f"\n❌ 测试过程中出错：{e}")
    
    finally:
        # 清理
        print("\n🧹 清理资源...")
        for device_id, cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("✅ 测试结束")


if __name__ == "__main__":
    test_cameras()
