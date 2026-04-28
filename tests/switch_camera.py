# File: tests/switch_camera.py
"""
快速切换摄像头配置

用于在摄像头 #0 和 #1 之间快速切换
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def switch_to_camera(camera_id: int):
    """切换所有测试脚本到指定摄像头"""
    
    test_files = [
        'tests/test_real_gesture_auto.py',
        'tests/test_real_gesture_recognition.py',
        'tests/test_usb_camera_stream.py',
    ]
    
    for file_path in test_files:
        full_path = Path(__file__).parent.parent / file_path
        
        if not full_path.exists():
            print(f"⚠️ 文件不存在：{file_path}")
            continue
        
        # 读取文件
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换摄像头 ID
        old_line = f'camera = USBCamera(device_id=0, width=640, height=480, fps=30)'
        new_line = f'camera = USBCamera(device_id={camera_id}, width=640, height=480, fps=30)'
        
        if old_line in content:
            content = content.replace(old_line, new_line)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 已切换 {file_path} 到摄像头 #{camera_id}")
        else:
            print(f"⚠️ {file_path} 中未找到摄像头配置")


if __name__ == "__main__":
    print("="*60)
    print("🔄 切换摄像头配置")
    print("="*60)
    
    print("\n当前可用选项:")
    print("  1. 摄像头 #0 (左侧，曝光正常)")
    print("  2. 摄像头 #1 (右侧，较暗)")
    
    choice = input("\n请输入要切换的摄像头编号 (0 或 1): ").strip()
    
    if choice in ['0', '1']:
        camera_id = int(choice)
        switch_to_camera(camera_id)
        
        print(f"\n✅ 所有测试脚本已切换到摄像头 #{camera_id}")
        print("\n📋 下一步:")
        print(f"   运行：python tests/test_real_gesture_auto.py")
        print(f"   将使用摄像头 #{camera_id}")
    else:
        print("❌ 无效输入")
