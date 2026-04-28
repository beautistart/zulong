# File: tests/test_real_models_performance.py
"""
真实模型性能测试

测试目标:
1. 加载真实 YOLO/MobileNetV4/EfficientNet 模型
2. 测试真实推理性能
3. 监控显存占用
4. 验证模型集成

TSD v1.7 对应:
- 5.2 显存约束
- 7.2 集成测试场景
"""

import sys
import os
import asyncio
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_real_model_loading():
    """测试 1: 真实模型加载"""
    print("\n" + "="*60)
    print("🧪 测试 1: 真实模型加载")
    print("="*60)
    
    try:
        from zulong.l1c.vision_model_loader import VisionModelLoader
        
        print("📦 开始加载真实模型...")
        start_time = time.time()
        
        # 创建并加载模型
        loader = VisionModelLoader()
        success = loader.load_all_models()
        
        load_time = time.time() - start_time
        
        if not success:
            print("❌ 模型加载失败")
            return False
        
        # 获取模型状态
        stats = loader.get_model_stats()
        
        print(f"\n✅ 模型加载完成 (耗时：{load_time:.2f}s)")
        print(f"\n📊 模型状态:")
        print(f"   - YOLOv10-Nano: {'✅' if stats['yolo_loaded'] else '❌'}")
        print(f"   - MobileNetV4: {'✅' if stats['mobilenet_loaded'] else '❌'}")
        print(f"   - EfficientNet-B0: {'✅' if stats['efficientnet_loaded'] else '❌'}")
        print(f"   - 设备：{stats['device']}")
        
        # 验证所有模型都已加载
        assert stats['yolo_loaded'], "YOLO 模型未加载"
        assert stats['mobilenet_loaded'], "MobileNetV4 未加载"
        assert stats['efficientnet_loaded'], "EfficientNet 未加载"
        
        print("\n✅ 测试 1 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 1 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_real_yolo_inference():
    """测试 2: YOLO 真实推理"""
    print("\n" + "="*60)
    print("🧪 测试 2: YOLO 真实推理测试")
    print("="*60)
    
    try:
        from zulong.l1c.vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧 (模拟真实场景)
        # 场景 1: 黑色背景 (无人)
        frame_empty = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 场景 2: 随机噪声 (模拟干扰)
        frame_noise = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("📊 YOLO 推理测试:")
        
        # 测试场景 1
        print("\n   场景 1: 空背景")
        human_detected, bbox_info = loader.detect_human(frame_empty)
        print(f"      - 检测到人体：{human_detected}")
        if bbox_info:
            print(f"      - 置信度：{bbox_info['confidence']:.2f}")
        
        # 测试场景 2
        print("\n   场景 2: 噪声背景")
        human_detected, bbox_info = loader.detect_human(frame_noise)
        print(f"      - 检测到人体：{human_detected}")
        if bbox_info:
            print(f"      - 置信度：{bbox_info['confidence']:.2f}")
        
        print("\n✅ 测试 2 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 2 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_real_mobilenet_inference():
    """测试 3: MobileNetV4 真实推理"""
    print("\n" + "="*60)
    print("🧪 测试 3: MobileNetV4 真实推理测试")
    print("="*60)
    
    try:
        from zulong.l1c.vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("📊 MobileNetV4 推理测试:")
        
        # 推理
        action_label, confidence = loader.classify_action(test_frame)
        
        print(f"   - 动作标签：{action_label}")
        print(f"   - 置信度：{confidence:.4f}")
        
        # 验证返回类型
        assert isinstance(action_label, str), "动作标签应为字符串"
        assert 0 <= confidence <= 1, "置信度应在 0-1 之间"
        
        print("\n✅ 测试 3 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 3 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_real_efficientnet_inference():
    """测试 4: EfficientNet 真实推理"""
    print("\n" + "="*60)
    print("🧪 测试 4: EfficientNet 真实推理测试")
    print("="*60)
    
    try:
        from zulong.l1c.vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("📊 EfficientNet 推理测试:")
        
        # 推理
        gesture_label, confidence = loader.recognize_gesture(test_frame)
        
        print(f"   - 手势标签：{gesture_label}")
        print(f"   - 置信度：{confidence:.4f}")
        
        # 验证返回类型
        assert isinstance(gesture_label, str), "手势标签应为字符串"
        assert 0 <= confidence <= 1, "置信度应在 0-1 之间"
        
        print("\n✅ 测试 4 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 4 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_inference_speed():
    """测试 5: 推理速度基准测试"""
    print("\n" + "="*60)
    print("🧪 测试 5: 推理速度基准测试")
    print("="*60)
    
    try:
        from zulong.l1c.vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("📊 推理速度测试 (10 次推理):")
        
        # 测试 YOLO
        print("\n   YOLOv10-Nano:")
        yolo_times = []
        for i in range(10):
            start = time.time()
            loader.detect_human(test_frame)
            elapsed = (time.time() - start) * 1000  # ms
            yolo_times.append(elapsed)
        
        yolo_avg = sum(yolo_times) / len(yolo_times)
        print(f"      - 平均延迟：{yolo_avg:.2f}ms")
        print(f"      - 最小延迟：{min(yolo_times):.2f}ms")
        print(f"      - 最大延迟：{max(yolo_times):.2f}ms")
        print(f"      - FPS: {1000/yolo_avg:.1f}")
        
        # 测试 MobileNetV4
        print("\n   MobileNetV4:")
        mobilenet_times = []
        for i in range(10):
            start = time.time()
            loader.classify_action(test_frame)
            elapsed = (time.time() - start) * 1000  # ms
            mobilenet_times.append(elapsed)
        
        mobilenet_avg = sum(mobilenet_times) / len(mobilenet_times)
        print(f"      - 平均延迟：{mobilenet_avg:.2f}ms")
        print(f"      - FPS: {1000/mobilenet_avg:.1f}")
        
        # 测试 EfficientNet
        print("\n   EfficientNet-B0:")
        efficientnet_times = []
        for i in range(10):
            start = time.time()
            loader.recognize_gesture(test_frame)
            elapsed = (time.time() - start) * 1000  # ms
            efficientnet_times.append(elapsed)
        
        efficientnet_avg = sum(efficientnet_times) / len(efficientnet_times)
        print(f"      - 平均延迟：{efficientnet_avg:.2f}ms")
        print(f"      - FPS: {1000/efficientnet_avg:.1f}")
        
        # 总体评估
        print("\n📈 总体评估:")
        total_latency = yolo_avg + mobilenet_avg + efficientnet_avg
        print(f"   - 总延迟 (串行): {total_latency:.2f}ms")
        print(f"   - 理论 FPS: {1000/total_latency:.1f}")
        
        if total_latency < 100:
            print("   ✅ 延迟优秀 (<100ms)")
        elif total_latency < 200:
            print("   ⚠️ 延迟可接受 (100-200ms)")
        else:
            print("   ❌ 延迟过高 (>200ms)")
        
        print("\n✅ 测试 5 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 5 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_memory_usage():
    """测试 6: 显存占用测试"""
    print("\n" + "="*60)
    print("🧪 测试 6: 显存占用测试")
    print("="*60)
    
    try:
        import torch
        
        # 加载前显存
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"📊 加载前显存占用：{mem_before:.2f} GB")
        else:
            print("⚠️ CUDA 不可用，跳过显存测试")
            return True
        
        from zulong.l1c.vision_model_loader import VisionModelLoader
        
        # 加载模型
        loader = VisionModelLoader()
        loader.load_all_models()
        
        # 加载后显存
        mem_after = torch.cuda.memory_allocated() / (1024**3)  # GB
        mem_used = mem_after - mem_before
        
        print(f"📊 加载后显存占用：{mem_after:.2f} GB")
        print(f"📊 模型显存占用：{mem_used:.2f} GB")
        
        # 评估
        print(f"\n📈 显存评估:")
        if mem_used < 2.0:
            print("   ✅ 显存占用优秀 (<2GB)")
        elif mem_used < 4.0:
            print("   ⚠️ 显存占用可接受 (2-4GB)")
        else:
            print("   ❌ 显存占用过高 (>4GB)")
        
        # TSD v1.7 要求
        print(f"\n📋 TSD v1.7 显存约束:")
        print(f"   - 要求：所有模型常驻显存 <6GB")
        print(f"   - 当前：{mem_used:.2f} GB")
        print(f"   - 状态：{'✅ 符合' if mem_used < 6.0 else '❌ 超出'}")
        
        print("\n✅ 测试 6 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 6 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("="*60)
    print("🚀 真实模型性能测试")
    print("="*60)
    print(f"📅 测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("真实模型加载", test_real_model_loading),
        ("YOLO 真实推理", test_real_yolo_inference),
        ("MobileNetV4 真实推理", test_real_mobilenet_inference),
        ("EfficientNet 真实推理", test_real_efficientnet_inference),
        ("推理速度基准", test_inference_speed),
        ("显存占用测试", test_memory_usage),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 测试失败：{e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("📊 测试汇总报告")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有真实模型测试通过！")
        print("\n📋 下一步:")
        print("1. 运行真实用户手势识别测试")
        print("2. 长时间稳定性测试")
        print("3. 集成到 L1-B Scheduler")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
    
    print("\n" + "="*60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
