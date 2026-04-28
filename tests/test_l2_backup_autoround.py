# 验证 L2-BACKUP vLLM 配置（AutoRound 版本）

"""
验证 L2-BACKUP 是否正确配置为使用 vLLM + Qwen3.5-0.8B-int4-AutoRound
"""

import os
import sys
import requests
import subprocess

def test_l2_backup_config():
    """测试 1: 验证 L2_BACKUP 配置"""
    print("="*80)
    print("测试 1: 验证 L2_BACKUP vLLM 配置")
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
            
            if isinstance(model_config, dict):
                if model_config.get('path') == 'vllm' and model_config.get('type') == 'remote':
                    print("✅ L2_BACKUP 已配置为 vLLM 模式")
                    print(f"   端点：{model_config.get('endpoint')}")
                    print(f"   模型：{model_config.get('model_name')}")
                    print(f"   量化：{model_config.get('quantization')}")
                    return True, model_config.get('endpoint')
                    
        print("⚠️ L2_BACKUP 未正确配置为 vLLM 模式")
        return False, None
            
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_vllm_endpoint_8001(endpoint):
    """测试 2: 验证 vLLM 端点可用性（端口 8001）"""
    print("\n" + "="*80)
    print("测试 2: 验证 vLLM 端点可用性（端口 8001）")
    print("="*80)
    
    if not endpoint:
        print("❌ 无法测试：endpoint 为空")
        return False
    
    try:
        # 检查端点
        url = f"{endpoint}/models"
        
        print(f"请求端点：{url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ vLLM 端点可用（端口 8001）")
            print(f"   可用模型数量：{len(models.get('data', []))}")
            
            # 打印模型列表
            for model in models.get('data', []):
                print(f"   - {model.get('id')}")
            
            return True
        else:
            print(f"⚠️ vLLM 端点返回异常状态码：{response.status_code}")
            print(f"   响应内容：{response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ vLLM 端点（8001）无法连接")
        print("   提示：请先启动 L2-BACKUP vLLM 服务：")
        print("   - Windows: 双击运行 start_l2_backup_vllm.bat")
        print("   - Linux/WSL: bash scripts/start_l2_backup_vllm.sh")
        return False
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

def test_l2_core_endpoint():
    """测试 3: 验证 L2_CORE vLLM 端点（端口 8000）"""
    print("\n" + "="*80)
    print("测试 3: 验证 L2_CORE vLLM 端点（端口 8000）")
    print("="*80)
    
    try:
        url = "http://localhost:8000/v1/models"
        print(f"请求端点：{url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✅ L2_CORE vLLM 端点可用（端口 8000）")
            print(f"   可用模型数量：{len(models.get('data', []))}")
            
            for model in models.get('data', []):
                print(f"   - {model.get('id')}")
            
            return True
        else:
            print(f"⚠️ L2_CORE 端点返回异常状态码：{response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ L2_CORE vLLM 端点（8000）无法连接")
        return False
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        return False

def get_startup_instructions():
    """获取启动说明"""
    print("\n" + "="*80)
    print("📋 启动说明")
    print("="*80)
    print("""
要使用 L2-BACKUP vLLM 服务，需要先启动 vLLM 实例：

方案 1: Windows 用户
  1. 双击运行 start_l2_backup_vllm.bat
  2. 等待模型下载完成（首次运行）
  3. 确认服务已启动

方案 2: Linux/WSL 用户
  1. 运行：bash scripts/start_l2_backup_vllm.sh
  2. 等待模型下载完成（首次运行）
  3. 确认服务已启动

方案 3: 直接使用 vLLM CLI
  1. 激活虚拟环境：source ~/vllm-env/bin/activate
  2. 设置环境变量：export VLLM_USE_MODELSCOPE=true
  3. 启动服务：
     vllm serve Intel/Qwen3.5-0.8B-int4-AutoRound \\
         --port 8001 \\
         --tensor-parallel-size 1 \\
         --gpu-memory-utilization 0.8 \\
         --max-model-len 4096 \\
         --trust-remote-code \\
         --dtype auto \\
         --quantization gptq_marlin

验证服务启动：
  curl http://localhost:8001/v1/models
""")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("🧪 L2-BACKUP vLLM + AutoRound 配置测试")
    print("="*80)
    
    results = []
    endpoint = None
    
    # 测试 1: 配置验证
    config_ok, endpoint = test_l2_backup_config()
    results.append(("L2_BACKUP 配置", config_ok))
    
    # 测试 2: L2_BACKUP vLLM 端点
    if endpoint:
        endpoint_ok = test_vllm_endpoint_8001(endpoint)
        results.append(("L2_BACKUP vLLM 端点(8001)", endpoint_ok))
    else:
        results.append(("L2_BACKUP vLLM 端点(8001)", False))
        get_startup_instructions()
    
    # 测试 3: L2_CORE vLLM 端点
    l2_core_ok = test_l2_core_endpoint()
    results.append(("L2_CORE vLLM 端点(8000)", l2_core_ok))
    
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
        print("\n🎉 所有测试通过！L2-BACKUP vLLM + AutoRound 配置成功！")
        return True
    else:
        print("\n⚠️ 部分测试失败，请检查 vLLM 服务是否已启动")
        get_startup_instructions()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
