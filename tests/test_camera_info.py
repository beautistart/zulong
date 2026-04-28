# File: tests/test_camera_info.py
"""
摄像头信息查询工具

快速查看系统中所有可用的 USB 摄像头及其详细信息
"""

import sys
import cv2
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_camera_info(device_id: int) -> dict:
    """获取摄像头详细信息"""
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        return None
    
    info = {
        'device_id': device_id,
        'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'backend': cap.getBackendName(),
        'brightness': cap.get(cv2.CAP_PROP_BRIGHTNESS),
        'contrast': cap.get(cv2.CAP_PROP_CONTRAST),
        'saturation': cap.get(cv2.CAP_PROP_SATURATION),
        'autofocus': cap.get(cv2.CAP_PROP_AUTOFOCUS),
        'focus': cap.get(cv2.CAP_PROP_FOCUS),
    }
    
    cap.release()
    return info


def main():
    """主函数"""
    print("="*60)
    print("📷 USB 摄像头信息查询")
    print("="*60)
    
    from zulong.l0.usb_camera import detect_available_cameras
    
    # 检测可用摄像头
    cameras = detect_available_cameras()
    
    if not cameras:
        print("\n❌ 未检测到任何 USB 摄像头")
        return
    
    print(f"\n✅ 检测到 {len(cameras)} 个摄像头\n")
    
    # 显示每个摄像头的详细信息
    for device_id in cameras:
        print("="*60)
        print(f"📹 摄像头 #{device_id}")
        print("="*60)
        
        info = get_camera_info(device_id)
        
        if info:
            print(f"设备 ID:    {info['device_id']}")
            print(f"分辨率：    {info['width']:.0f} x {info['height']:.0f}")
            print(f"帧率：      {info['fps']:.0f} FPS")
            print(f"后端：      {info['backend']}")
            print(f"亮度：      {info['brightness']:.0f}")
            print(f"对比度：    {info['contrast']:.0f}")
            print(f"饱和度：    {info['saturation']:.0f}")
            print(f"自动对焦：  {'✅' if info['autofocus'] else '❌'}")
            if not info['autofocus']:
                print(f"对焦位置：  {info['focus']:.0f}")
        else:
            print("❌ 无法获取摄像头详细信息")
        
        print()
    
    print("="*60)
    print("💡 当前视觉模块使用的摄像头:")
    print("="*60)
    if cameras:
        print(f"   设备 ID: {cameras[0]} (第一个可用摄像头)")
        print(f"   配置：640x480@30fps")
        print(f"   用途：人体检测、动作分类、手势识别")
    else:
        print("   ❌ 未检测到摄像头")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
