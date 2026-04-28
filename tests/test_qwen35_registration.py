# 测试 qwen3_5 架构注册和模型加载

"""
验证 qwen3_5 架构是否正确注册到 Transformers
并测试模型加载是否正常
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_qwen35_registration():
    """测试 1: 验证 qwen3_5 架构注册"""
    print("="*80)
    print("测试 1: 验证 qwen3_5 架构注册")
    print("="*80)
    
    try:
        from transformers.models.auto import CONFIG_MAPPING
        
        if "qwen3_5" in CONFIG_MAPPING:
            print("✅ qwen3_5 架构已成功注册到 CONFIG_MAPPING")
            print(f"   注册的配置类：{CONFIG_MAPPING['qwen3_5']}")
            return True
        else:
            print("⚠️ qwen3_5 架构未注册到 CONFIG_MAPPING")
            print(f"   已注册的架构：{list(CONFIG_MAPPING.keys())[-10:]}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_container_init():
    """测试 2: 验证 ModelContainer 初始化"""
    print("\n" + "="*80)
    print("测试 2: 验证 ModelContainer 初始化")
    print("="*80)
    
    try:
        from zulong.models.container import ModelContainer
        
        container = ModelContainer()
        print("✅ ModelContainer 初始化成功")
        print(f"   常驻模型数量：{len(container.resident_models)}")
        print(f"   当前显存使用：{container.current_vram_usage:.2f}GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_auto_config_load():
    """测试 3: 验证 AutoConfig 加载 qwen3_5 模型"""
    print("\n" + "="*80)
    print("测试 3: 验证 AutoConfig 加载 qwen3_5 模型")
    print("="*80)
    
    try:
        from transformers import AutoConfig
        
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'Qwen', 'Qwen3___5-2B-AWQ')
        
        if not os.path.exists(model_path):
            print(f"⚠️ 模型路径不存在：{model_path}")
            print("   跳过此测试")
            return True
        
        print(f"加载模型配置：{model_path}")
        
        # 🔥 关键：使用 trust_remote_code=True
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"✅ AutoConfig 加载成功")
        print(f"   模型类型：{config.model_type}")
        print(f"   Transformers 版本：{config.transformers_version}")
        print(f"   词汇表大小：{config.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_model():
    """测试 4: 验证通过 ModelContainer 获取模型"""
    print("\n" + "="*80)
    print("测试 4: 验证通过 ModelContainer 获取模型")
    print("="*80)
    
    try:
        from zulong.models.container import ModelContainer
        from zulong.models.config import ModelID
        
        container = ModelContainer()
        
        # 测试获取 L2_CORE 模型
        if ModelID.L2_CORE in container.resident_models:
            model = container.resident_models[ModelID.L2_CORE]
            print(f"✅ L2_CORE 模型获取成功")
            print(f"   模型类型：{type(model)}")
            print(f"   模型路径：{model.get('path', 'N/A')}")
        else:
            print(f"⚠️ L2_CORE 模型未加载到常驻模型")
            print(f"   常驻模型列表：{list(container.resident_models.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 qwen3_5 架构注册和模型加载测试")
    print("="*80)
    
    results = []
    
    # 测试 1: 架构注册
    results.append(("架构注册", test_qwen35_registration()))
    
    # 测试 2: ModelContainer 初始化
    results.append(("ModelContainer 初始化", test_model_container_init()))
    
    # 测试 3: AutoConfig 加载
    results.append(("AutoConfig 加载", test_auto_config_load()))
    
    # 测试 4: 获取模型
    results.append(("获取模型", test_get_model()))
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, r in results if r)
    total_tests = len(results)
    
    print(f"\n总计：{total_passed}/{total_tests} 测试通过")
    
    if total_passed == total_tests:
        print("\n🎉 所有测试通过！qwen3_5 架构注册成功！")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
