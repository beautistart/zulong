# File: tests/test_vision_models_validation.py
"""
优化视觉处理器真实模型验证测试

验证目标:
1. YOLOv10-Nano 模型加载
2. MobileNetV4 模型加载
3. EfficientNet-B0 模型加载
4. ModelContainer 集成验证

TSD v1.7 对应:
- 5.2 显存约束
- 7.2 集成测试场景
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_yolo_model():
    """测试 1: YOLOv10-Nano 模型文件验证"""
    print("\n" + "="*60)
    print("🧪 测试 1: YOLOv10-Nano 模型文件验证")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    yolo_path = base_dir / "models" / "yolov10n.pt"
    
    # 检查文件存在
    assert yolo_path.exists(), f"YOLO 模型文件不存在：{yolo_path}"
    print(f"✅ YOLO 模型文件存在：{yolo_path}")
    
    # 检查文件大小 (应该>1MB)
    file_size = yolo_path.stat().st_size / (1024 * 1024)  # MB
    assert file_size > 1.0, f"YOLO 模型文件过小：{file_size:.2f}MB"
    print(f"✅ YOLO 模型文件大小：{file_size:.2f}MB")
    
    print("\n✅ 测试 1 通过")
    return True


def test_mobilenetv4_model():
    """测试 2: MobileNetV4 模型文件验证"""
    print("\n" + "="*60)
    print("🧪 测试 2: MobileNetV4 模型文件验证")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    mobilenet_path = base_dir / "models" / "jaiwei98" / "MobileNetV4"
    
    # 检查目录存在
    assert mobilenet_path.exists(), f"MobileNetV4 目录不存在：{mobilenet_path}"
    print(f"✅ MobileNetV4 目录存在：{mobilenet_path}")
    
    # 检查权重文件
    weight_files = list(mobilenet_path.glob("*.pth"))
    assert len(weight_files) > 0, "未找到 MobileNetV4 权重文件"
    print(f"✅ 找到 {len(weight_files)} 个权重文件:")
    for wf in weight_files:
        size_mb = wf.stat().st_size / (1024 * 1024)
        print(f"   - {wf.name}: {size_mb:.2f}MB")
    
    print("\n✅ 测试 2 通过")
    return True


def test_efficientnet_model():
    """测试 3: EfficientNet-B0 模型文件验证"""
    print("\n" + "="*60)
    print("🧪 测试 3: EfficientNet-B0 模型文件验证")
    print("="*60)
    
    base_dir = Path(__file__).parent.parent
    efficientnet_path = base_dir / "models" / "google" / "efficientnet-b0"
    
    # 检查目录存在
    assert efficientnet_path.exists(), f"EfficientNet 目录不存在：{efficientnet_path}"
    print(f"✅ EfficientNet 目录存在：{efficientnet_path}")
    
    # 检查配置文件
    config_file = efficientnet_path / "config.json"
    assert config_file.exists(), "未找到 config.json"
    print(f"✅ 配置文件存在：{config_file}")
    
    # 检查权重文件
    weight_files = list(efficientnet_path.glob("*.bin"))
    assert len(weight_files) > 0, "未找到 EfficientNet 权重文件"
    weight_file = weight_files[0]
    size_mb = weight_file.stat().st_size / (1024 * 1024)
    print(f"✅ 权重文件：{weight_file.name} ({size_mb:.2f}MB)")
    
    print("\n✅ 测试 3 通过")
    return True


def test_model_container_integration():
    """测试 4: ModelContainer 集成验证"""
    print("\n" + "="*60)
    print("🧪 测试 4: ModelContainer 集成验证")
    print("="*60)
    
    try:
        from zulong.models.config import ModelID, MODEL_CONFIGS
        from zulong.models.container import ModelContainer
        
        # 检查配置
        assert ModelID.VISION_YOLO in MODEL_CONFIGS, "VISION_YOLO 配置缺失"
        assert ModelID.VISION_ACTION in MODEL_CONFIGS, "VISION_ACTION 配置缺失"
        assert ModelID.VISION_GESTURE in MODEL_CONFIGS, "VISION_GESTURE 配置缺失"
        
        print(f"✅ 模型配置完整")
        print(f"   - VISION_YOLO: {MODEL_CONFIGS[ModelID.VISION_YOLO].repo_id}")
        print(f"   - VISION_ACTION: {MODEL_CONFIGS[ModelID.VISION_ACTION].repo_id}")
        print(f"   - VISION_GESTURE: {MODEL_CONFIGS[ModelID.VISION_GESTURE].repo_id}")
        
        # 计算总显存占用
        total_vram = sum(
            MODEL_CONFIGS[mid].estimated_vram_gb 
            for mid in [ModelID.VISION_YOLO, ModelID.VISION_ACTION, ModelID.VISION_GESTURE]
        )
        print(f"✅ 视觉模型总显存占用：{total_vram:.1f}GB")
        
        # 检查 ModelContainer 单例
        container = ModelContainer()
        print(f"✅ ModelContainer 单例创建成功")
        
        print("\n✅ 测试 4 通过")
        return True
        
    except Exception as e:
        print(f"❌ ModelContainer 集成失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有验证测试"""
    print("\n" + "="*60)
    print("🚀 优化视觉处理器真实模型验证")
    print("TSD v1.7 三层注意力机制")
    print("="*60)
    
    tests = [
        ("YOLOv10-Nano 验证", test_yolo_model),
        ("MobileNetV4 验证", test_mobilenetv4_model),
        ("EfficientNet-B0 验证", test_efficientnet_model),
        ("ModelContainer 集成", test_model_container_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 测试失败：{e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*60)
    print("📊 验证汇总报告")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计：{passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有模型验证通过！")
        print("\n📋 下一步:")
        print("1. 在真实环境中加载模型")
        print("2. 运行集成测试：python tests/test_optimized_vision_integration.py")
        print("3. 真实机器人 3 米手势识别测试")
    else:
        print("\n⚠️ 部分验证失败，请检查日志")
    
    print("\n" + "="*60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
