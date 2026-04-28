# 测试 L2-BACKUP vLLM 配置

"""
验证 L2-BACKUP 是否正确配置为使用 vLLM 加载
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_l2_backup_vllm_config():
    """测试 1: 验证 L2_BACKUP 的 vLLM 配置"""
    print("="*80)
    print("测试 1: 验证 L2_BACKUP 的 vLLM 配置")
    print("="*80)
    
    try:
        # 设置环境变量
        os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true"
        
        from zulong.models.container import ModelContainer
        from zulong.models.config import ModelID
        
        container = ModelContainer()
        
        # 检查 L2_BACKUP 是否已注册
        if ModelID.L2_BACKUP in container.resident_models:
            model_config = container.resident_models[ModelID.L2_BACKUP]
            
            if isinstance(model_config, dict) and model_config.get('path') == 'vllm':
                print("✅ L2_BACKUP 已配置为 vLLM 模式")
                print(f"   端点：{model_config.get('endpoint')}")
                print(f"   共享：{model_config.get('shared_with', '无')}")
                return True
            else:
                print("⚠️ L2_BACKUP 使用的是本地加载模式")
                print(f"   类型：{type(model_config)}")
                return False
        else:
            print("❌ L2_BACKUP 未注册到 ModelContainer")
            print(f"   已注册模型：{list(container.resident_models.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_l2_backup_fallback():
    """测试 2: 验证 L2_BACKUP 降级方案"""
    print("\n" + "="*80)
    print("测试 2: 验证 L2_BACKUP 降级方案（不使用 vLLM）")
    print("="*80)
    
    try:
        # 设置环境变量为 False
        os.environ["USE_VLLM_FOR_L2_BACKUP"] = "false"
        
        # 需要重新导入以刷新配置
        import importlib
        import zulong.models.container as container_module
        importlib.reload(container_module)
        
        from zulong.models.container import ModelContainer
        from zulong.models.config import ModelID
        
        container = ModelContainer()
        
        # 检查 L2_BACKUP 是否已注册
        if ModelID.L2_BACKUP in container.resident_models:
            model_config = container.resident_models[ModelID.L2_BACKUP]
            
            # 检查是否为本地加载（RealModelLoader）
            if hasattr(model_config, 'load_model'):
                print("✅ L2_BACKUP 已配置为本地加载模式（降级方案）")
                print(f"   类型：{type(model_config).__name__}")
                return True
            else:
                print("⚠️ L2_BACKUP 配置类型未知")
                print(f"   类型：{type(model_config)}")
                return False
        else:
            print("❌ L2_BACKUP 未注册到 ModelContainer")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

def test_vllm_endpoint():
    """测试 3: 验证 vLLM 端点可用性"""
    print("\n" + "="*80)
    print("测试 3: 验证 vLLM 端点可用性")
    print("="*80)
    
    try:
        import requests
        
        vllm_endpoint = "http://localhost:8000/v1/models"
        
        print(f"请求 vLLM 端点：{vllm_endpoint}")
        response = requests.get(vllm_endpoint, timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ vLLM 端点可用")
            print(f"   可用模型：{len(models.get('data', []))}")
            
            # 打印模型列表
            for model in models.get('data', []):
                print(f"   - {model.get('id')}")
            
            return True
        else:
            print(f"⚠️ vLLM 端点返回异常状态码：{response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ vLLM 端点无法连接")
        print("   请确保 vLLM 服务已启动：vllm serve ...")
        return False
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 L2-BACKUP vLLM 配置测试")
    print("="*80)
    
    results = []
    
    # 测试 1: vLLM 配置
    results.append(("vLLM 配置", test_l2_backup_vllm_config()))
    
    # 测试 2: 降级方案
    results.append(("降级方案", test_l2_backup_fallback()))
    
    # 测试 3: vLLM 端点
    results.append(("vLLM 端点", test_vllm_endpoint()))
    
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
        print("\n🎉 所有测试通过！L2-BACKUP vLLM 配置成功！")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查日志")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
