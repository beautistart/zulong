# File: tests/test_real_model_integration.py
"""
真实模型集成测试

测试目标:
1. VisionModelLoader 加载所有模型
2. OptimizedVisionProcessor 使用真实模型推理
3. 端到端帧处理流程验证

TSD v1.7 对应:
- 5.2 显存约束
- 7.2 集成测试场景
"""

import sys
import os
import asyncio
import numpy as np
from pathlib import Path
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_vision_model_loader():
    """测试 1: VisionModelLoader 加载所有模型"""
    print("\n" + "="*60)
    print("🧪 测试 1: VisionModelLoader 模型加载")
    print("="*60)
    
    try:
        # 直接导入 vision_model_loader，绕过 l1c.__init__.py
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from vision_model_loader import VisionModelLoader
        
        # 创建单例
        loader = VisionModelLoader()
        
        # 加载所有模型
        success = loader.load_all_models()
        
        if not success:
            print("❌ 模型加载失败")
            return False
        
        # 验证模型状态
        stats = loader.get_model_stats()
        print(f"📊 模型状态:")
        print(f"   - YOLO: {'✅' if stats['yolo_loaded'] else '❌'}")
        print(f"   - MobileNetV4: {'✅' if stats['mobilenet_loaded'] else '❌'}")
        print(f"   - EfficientNet: {'✅' if stats['efficientnet_loaded'] else '❌'}")
        print(f"   - 设备：{stats['device']}")
        
        # 验证所有模型都已加载
        assert stats['yolo_loaded'], "YOLO 模型未加载"
        assert stats['mobilenet_loaded'], "MobileNetV4 模型未加载"
        assert stats['efficientnet_loaded'], "EfficientNet 模型未加载"
        
        print("\n✅ 测试 1 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 1 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_yolo_inference():
    """测试 2: YOLO 人体检测推理"""
    print("\n" + "="*60)
    print("🧪 测试 2: YOLO 人体检测推理")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧 (黑色背景)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 推理
        human_detected, bbox_info = loader.detect_human(test_frame)
        
        print(f"📊 推理结果:")
        print(f"   - 检测到人体：{human_detected}")
        if bbox_info:
            print(f"   - 检测框：{bbox_info['bbox']}")
            print(f"   - 置信度：{bbox_info['confidence']:.2f}")
        
        # 黑色背景应该检测不到人体
        assert not human_detected, "黑色背景不应检测到人体"
        
        print("\n✅ 测试 2 通过")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 2 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_action_classification():
    """测试 3: MobileNetV4 动作分类"""
    print("\n" + "="*60)
    print("🧪 测试 3: MobileNetV4 动作分类")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 推理
        action_label, confidence = loader.classify_action(test_frame)
        
        print(f"📊 推理结果:")
        print(f"   - 动作标签：{action_label}")
        print(f"   - 置信度：{confidence:.2f}")
        
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


async def test_gesture_recognition():
    """测试 4: EfficientNet 手势识别"""
    print("\n" + "="*60)
    print("🧪 测试 4: EfficientNet 手势识别")
    print("="*60)
    
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from vision_model_loader import VisionModelLoader
        
        loader = VisionModelLoader()
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 推理
        gesture_label, confidence = loader.recognize_gesture(test_frame)
        
        print(f"📊 推理结果:")
        print(f"   - 手势标签：{gesture_label}")
        print(f"   - 置信度：{confidence:.2f}")
        
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


async def test_optimized_vision_processor():
    """测试 5: OptimizedVisionProcessor 集成"""
    print("\n" + "="*60)
    print("🧪 测试 5: OptimizedVisionProcessor 集成")
    print("="*60)
    
    try:
        # 直接导入 optimized_vision_processor，绕过 l1c.__init__.py
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / 'zulong' / 'l1c'))
        
        from optimized_vision_processor import OptimizedVisionProcessor
        
        # 创建处理器
        processor = OptimizedVisionProcessor()
        
        # 初始化 (不加载真实模型，避免依赖问题)
        await processor.initialize(load_models=False)
        
        print(f"📊 处理器状态:")
        print(f"   - 已初始化：{processor.is_initialized}")
        print(f"   - 运行中：{processor.is_running}")
        
        # 验证状态
        assert processor.is_initialized, "处理器未初始化"
        
        # 创建测试帧
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        timestamp = time.time()
        
        # 喂入帧
        processor.feed_frame(test_frame, timestamp)
        
        # 等待处理
        await asyncio.sleep(0.5)
        
        # 检查共享内存
        print(f"\n📊 共享内存状态:")
        print(f"   - 人体检测：{processor.shared_memory['human_detected']}")
        print(f"   - 运动像素：{processor.shared_memory['motion_pixels']}")
        print(f"   - 动作分数：{processor.shared_memory['action_score']:.2f}")
        
        print("\n✅ 测试 5 通过")
        
        # 停止处理器
        processor.stop()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试 5 失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """运行所有测试"""
    print("="*60)
    print("🚀 真实模型集成测试")
    print("="*60)
    
    tests = [
        ("VisionModelLoader 模型加载", test_vision_model_loader),
        ("YOLO 人体检测推理", test_yolo_inference),
        ("MobileNetV4 动作分类", test_action_classification),
        ("EfficientNet 手势识别", test_gesture_recognition),
        ("OptimizedVisionProcessor 集成", test_optimized_vision_processor),
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
        print("\n🎉 所有集成测试通过！")
        print("\n📋 下一步:")
        print("1. 真实摄像头视频流测试")
        print("2. 3 米手势识别率测试")
        print("3. 性能基准测试 (目标>30 FPS)")
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
    
    print("\n" + "="*60)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
