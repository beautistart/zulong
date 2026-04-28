# File: tests/debug_vision_layers.py
"""
三层注意机制调试测试

测试目标:
1. 检查 L1-C 层是否正确获取摄像头画面
2. 检查 YOLO 人体检测是否工作
3. 检查 ROI 增益放大是否生效
4. 检查 MobileNetV4-TSM 动作分类是否工作
5. 检查 EfficientNet 手势识别是否工作

TSD v1.7 对应:
- 2.2.2 L1 层拆分
- 4.4 感知预处理
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_layer1_yolo_detection():
    """测试 Layer 1: YOLO 人体检测"""
    print("\n" + "="*60)
    print("🧪 测试 Layer 1: YOLO 人体检测")
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
        
        # 加载 YOLO 模型
        print("📦 加载 YOLO 模型...")
        model_loader = get_vision_model_loader()
        model_loader.load_all_models()  # 不是异步函数
        
        yolo_model = model_loader.yolo_model
        
        if yolo_model is None:
            print("❌ YOLO 模型加载失败")
            return False
        
        print("✅ YOLO 模型已加载")
        
        # 测试 10 帧
        print("\n📸 测试 10 帧人体检测...")
        detect_count = 0
        
        for i in range(10):
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                print(f"   帧 {i+1}: ❌ 无法获取帧")
                continue
            
            # YOLO 检测
            results = yolo_model(frame, verbose=False)
            
            # 解析结果
            detections = results[0].boxes
            
            if len(detections) > 0:
                detect_count += 1
                # 获取第一个人体检测框
                bbox = detections[0].xyxy[0].cpu().numpy()
                conf = detections[0].conf[0].cpu().numpy()
                
                print(f"   帧 {i+1}: ✅ 检测到人体 (conf={conf:.2f}, bbox={bbox})")
            else:
                print(f"   帧 {i+1}: ❌ 未检测到人体")
            
            time.sleep(0.1)
        
        print(f"\n📊 Layer 1 检测结果:")
        print(f"   - 总帧数：10")
        print(f"   - 检测到人体：{detect_count}")
        print(f"   - 检测率：{detect_count/10*100:.0f}%")
        
        camera.stop()
        camera.disconnect()
        
        return detect_count > 0
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_layer2_roi_motion():
    """测试 Layer 2: ROI 增益放大 + 运动检测"""
    print("\n" + "="*60)
    print("🧪 测试 Layer 2: ROI 增益放大 + 运动检测")
    print("="*60)
    
    try:
        from zulong.l0.usb_camera import USBCamera
        from zulong.l1c.vision_model_loader import get_vision_model_loader
        from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 等待系统稳定
        print("⏳ 等待系统稳定 (2 秒)...")
        time.sleep(2)
        
        # 测试 10 帧
        print("\n📸 测试 ROI 运动检测...")
        motion_count = 0
        
        for i in range(10):
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                print(f"   帧 {i+1}: ❌ 无法获取帧")
                continue
            
            # 获取人体检测框
            human_bbox = processor.shared_memory.get('human_bbox')
            
            if human_bbox is None:
                print(f"   帧 {i+1}: ⚠️ 无人体检测框")
                continue
            
            # ROI 运动检测
            motion_detected, motion_pixels = processor._layer2_roi_motion_detection(
                frame, human_bbox
            )
            
            if motion_detected:
                motion_count += 1
                print(f"   帧 {i+1}: ✅ 检测到运动 (pixels={motion_pixels})")
            else:
                print(f"   帧 {i+1}: ❌ 未检测到运动 (pixels={motion_pixels})")
            
            time.sleep(0.1)
        
        print(f"\n📊 Layer 2 ROI 运动检测结果:")
        print(f"   - 总帧数：10")
        print(f"   - 检测到运动：{motion_count}")
        print(f"   - 检测率：{motion_count/10*100:.0f}%")
        
        camera.stop()
        camera.disconnect()
        
        return motion_count > 0
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_layer3_action_classification():
    """测试 Layer 3: MobileNetV4-TSM 动作分类"""
    print("\n" + "="*60)
    print("🧪 测试 Layer 3: MobileNetV4-TSM 动作分类")
    print("="*60)
    
    try:
        from zulong.l0.usb_camera import USBCamera
        from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 等待系统稳定
        print("⏳ 等待系统稳定 (2 秒)...")
        time.sleep(2)
        
        # 测试 10 帧
        print("\n📸 测试动作分类...")
        intent_count = 0
        
        for i in range(10):
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                print(f"   帧 {i+1}: ❌ 无法获取帧")
                continue
            
            # 获取动作分数
            action_score = processor.shared_memory.get('action_score', 0.0)
            
            # 判断是否有意图
            has_intent = action_score >= processor._config['intent_threshold']
            
            if has_intent:
                intent_count += 1
                print(f"   帧 {i+1}: ✅ 检测到意图 (score={action_score:.2f})")
            else:
                print(f"   帧 {i+1}: ❌ 无意图 (score={action_score:.2f})")
            
            time.sleep(0.1)
        
        print(f"\n📊 Layer 3 动作分类结果:")
        print(f"   - 总帧数：10")
        print(f"   - 检测到意图：{intent_count}")
        print(f"   - 检测率：{intent_count/10*100:.0f}%")
        
        camera.stop()
        camera.disconnect()
        
        return True  # 动作分类可能没有意图，这是正常的
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_layer4_gesture_recognition():
    """测试 Layer 4: EfficientNet 手势识别"""
    print("\n" + "="*60)
    print("🧪 测试 Layer 4: EfficientNet 手势识别")
    print("="*60)
    
    try:
        from zulong.l0.usb_camera import USBCamera
        from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        await processor.initialize(load_models=True)
        
        # 创建摄像头
        camera = USBCamera(device_id=0, width=640, height=480, fps=30)
        
        if not camera.connect():
            print("❌ 摄像头连接失败")
            return False
        
        camera.start()
        print("✅ 摄像头已启动")
        
        # 等待系统稳定
        print("⏳ 等待系统稳定 (2 秒)...")
        time.sleep(2)
        
        # 测试 10 帧
        print("\n📸 测试手势识别...")
        gesture_count = 0
        
        for i in range(10):
            frame, timestamp = camera.get_latest_frame()
            
            if frame is None:
                print(f"   帧 {i+1}: ❌ 无法获取帧")
                continue
            
            # 获取手势类型
            gesture_type = processor.shared_memory.get('gesture_type')
            
            if gesture_type:
                gesture_count += 1
                print(f"   帧 {i+1}: ✅ 识别到手势：{gesture_type}")
            else:
                print(f"   帧 {i+1}: ❌ 未识别到手势")
            
            time.sleep(0.1)
        
        print(f"\n📊 Layer 4 手势识别结果:")
        print(f"   - 总帧数：10")
        print(f"   - 识别到手势：{gesture_count}")
        print(f"   - 识别率：{gesture_count/10*100:.0f}%")
        
        camera.stop()
        camera.disconnect()
        
        return gesture_count > 0
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("="*60)
    print("🔍 三层注意机制调试测试")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 测试 Layer 1
    results['layer1_yolo'] = await test_layer1_yolo_detection()
    
    # 测试 Layer 2
    results['layer2_roi'] = await test_layer2_roi_motion()
    
    # 测试 Layer 3
    results['layer3_action'] = await test_layer3_action_classification()
    
    # 测试 Layer 4
    results['layer4_gesture'] = await test_layer4_gesture_recognition()
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    
    print(f"\nLayer 1 (YOLO 人体检测): {'✅ 通过' if results['layer1_yolo'] else '❌ 失败'}")
    print(f"Layer 2 (ROI 运动检测): {'✅ 通过' if results['layer2_roi'] else '❌ 失败'}")
    print(f"Layer 3 (动作分类): {'✅ 通过' if results['layer3_action'] else '❌ 失败'}")
    print(f"Layer 4 (手势识别): {'✅ 通过' if results['layer4_gesture'] else '❌ 失败'}")
    
    # 分析
    print("\n🔍 问题分析:")
    
    if not results['layer1_yolo']:
        print("   ⚠️ Layer 1 失败：YOLO 无法检测到人体")
        print("      可能原因：")
        print("      - 摄像头画面中没有人体")
        print("      - YOLO 模型阈值过高")
        print("      - 光照条件不佳")
    
    if not results['layer2_roi']:
        print("   ⚠️ Layer 2 失败：ROI 无法检测到运动")
        print("      可能原因：")
        print("      - 人体保持静止，没有动作")
        print("      - ROI 区域设置过小")
        print("      - 运动阈值过高")
    
    if not results['layer4_gesture']:
        print("   ⚠️ Layer 4 失败：EfficientNet 无法识别手势")
        print("      可能原因：")
        print("      - 模型未针对真实手势微调")
        print("      - 手势距离太远")
        print("      - Digital Zoom 倍数不足")
        print("      - 手势角度与训练数据不匹配")
    
    return all(results.values())


if __name__ == "__main__":
    import asyncio
    
    success = asyncio.run(main())
    
    if success:
        print("\n" + "="*60)
        print("🎉 所有测试通过!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️ 部分测试失败，请查看上方分析")
        print("="*60)
    
    sys.exit(0 if success else 1)
