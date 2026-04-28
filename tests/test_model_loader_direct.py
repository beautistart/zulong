# File: tests/test_model_loader_direct.py
"""
直接测试 VisionModelLoader

目标：验证模型加载器是否正常工作
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time
import asyncio

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_model_loader():
    """直接测试模型加载器"""
    
    print("="*60)
    print("🧪 直接测试 VisionModelLoader")
    print("="*60)
    
    try:
        from zulong.l0.usb_camera import USBCamera
        from zulong.l1c.vision_model_loader import get_vision_model_loader
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 加载模型
        print("\n📦 加载模型...")
        model_loader = get_vision_model_loader()
        model_loader.load_all_models()
        
        print(f"\n✅ 模型加载状态:")
        print(f"   - YOLO: {model_loader.yolo_model is not None}")
        print(f"   - MobileNet: {model_loader.mobilenet_model is not None}")
        print(f"   - EfficientNet: {model_loader.efficientnet_model is not None}")
        print(f"   - 设备：{model_loader.device}")
        
        # 测试 20 帧
        print("\n📸 测试 20 帧 YOLO 检测...")
        detect_count = 0
        
        for i in range(20):
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                print(f"   帧 {i+1}: ❌ 无法获取帧")
                continue
            
            # 使用模型加载器检测
            human_detected, bbox_info = model_loader.detect_human(frame)
            
            if human_detected and bbox_info:
                detect_count += 1
                conf = bbox_info.get('confidence', 0)
                bbox = bbox_info.get('bbox', [])
                print(f"   帧 {i+1}: ✅ 检测到人体 (conf={conf:.2f}, bbox={bbox})")
            else:
                print(f"   帧 {i+1}: ❌ 未检测到人体")
            
            time.sleep(0.1)
        
        print(f"\n📊 检测结果:")
        print(f"   - 总帧数：20")
        print(f"   - 检测到人体：{detect_count}")
        print(f"   - 检测率：{detect_count/20*100:.0f}%")
        
        camera.stop()
        camera.disconnect()
        
        return detect_count > 0
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_model_loader())
    
    print("\n" + "="*60)
    if success:
        print("✅ VisionModelLoader 工作正常!")
    else:
        print("❌ VisionModelLoader 未检测到人体")
    print("="*60)
    
    sys.exit(0 if success else 1)
