# -*- coding: utf-8 -*-
# File: scripts/check_l2_model.py
# 检查当前系统使用的是 L2 CORE 还是 L2 BACKUP

import sys
import os

# 设置控制台编码为 UTF-8
sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           L2 模型使用情况检查")
print("=" * 80)
print()

# 1. 检查环境变量
print("1. 环境变量检查")
print("-" * 80)
vllm_base_url = os.environ.get('VLLM_BASE_URL', 'http://localhost:8000/v1')
use_vllm_l2 = os.environ.get('USE_VLLM_FOR_L2', 'false')
use_vllm_backup = os.environ.get('USE_VLLM_FOR_L2_BACKUP', 'false')

print(f"   VLLM_BASE_URL:        {vllm_base_url}")
print(f"   USE_VLLM_FOR_L2:      {use_vllm_l2}")
print(f"   USE_VLLM_FOR_L2_BACKUP: {use_vllm_backup}")
print()

# 判断使用的模型
if '8000' in vllm_base_url:
    print(f"   ✅ 当前使用：L2 CORE (端口 8000)")
    model_in_use = "L2 CORE"
elif '8001' in vllm_base_url:
    print(f"   ⚠️  当前使用：L2 BACKUP (端口 8001)")
    model_in_use = "L2 BACKUP"
else:
    print(f"   ❓ 当前使用：未知服务 ({vllm_base_url})")
    model_in_use = "UNKNOWN"

print()

# 2. 检查推理引擎状态
print("2. 推理引擎状态检查")
print("-" * 80)
try:
    from zulong.l2.inference_engine import InferenceEngine
    
    # 获取实例
    ie = InferenceEngine.get_instance()
    
    if ie:
        print(f"   ✅ 推理引擎实例：已存在")
        
        # 检查 vLLM 客户端
        if hasattr(ie, 'vllm_client') and ie.vllm_client:
            print(f"   ✅ vLLM 客户端：已初始化")
            # 尝试获取客户端地址
            if hasattr(ie.vllm_client, 'base_url'):
                client_url = str(ie.vllm_client.base_url)
                print(f"   📍 客户端地址：{client_url}")
                
                if '8000' in client_url:
                    print(f"   ✅ 实际使用：L2 CORE (端口 8000)")
                elif '8001' in client_url:
                    print(f"   ⚠️  实际使用：L2 BACKUP (端口 8001)")
                else:
                    print(f"   ❓ 实际使用：未知地址")
            else:
                print(f"   ⚠️  无法获取客户端地址")
        else:
            print(f"   ⚠️  vLLM 客户端：未初始化")
            print(f"   💡 提示：系统可能在使用本地模型或尚未进行推理")
    else:
        print(f"   ⚠️  推理引擎实例：不存在")
        print(f"   💡 提示：祖龙系统可能尚未启动")
        
except Exception as e:
    print(f"   ❌ 检查失败：{e}")
    import traceback
    traceback.print_exc()

print()

# 3. 检查模型容器配置
print("3. 模型容器配置检查")
print("-" * 80)
try:
    from zulong.models.container import ModelContainer, ModelID
    
    container = ModelContainer()
    
    # 检查 L2 CORE
    if ModelID.L2_CORE in container.resident_models:
        l2_core_config = container.resident_models[ModelID.L2_CORE]
        print(f"   ✅ L2 CORE 已加载")
        if isinstance(l2_core_config, dict):
            endpoint = l2_core_config.get('endpoint', 'N/A')
            print(f"      端点：{endpoint}")
            if endpoint != 'N/A':
                if '8000' in endpoint:
                    print(f"      ✅ 配置正确：L2 CORE (端口 8000)")
                elif '8001' in endpoint:
                    print(f"      ⚠️  配置异常：指向 L2 BACKUP")
    else:
        print(f"   ⚠️  L2 CORE 未加载到容器")
    
    # 检查 L2 BACKUP
    if ModelID.L2_BACKUP in container.resident_models:
        l2_backup_config = container.resident_models[ModelID.L2_BACKUP]
        print(f"   ✅ L2 BACKUP 已加载")
        if isinstance(l2_backup_config, dict):
            endpoint = l2_backup_config.get('endpoint', 'N/A')
            print(f"      端点：{endpoint}")
            if endpoint != 'N/A':
                if '8001' in endpoint:
                    print(f"      ✅ 配置正确：L2 BACKUP (端口 8001)")
                elif '8000' in endpoint:
                    print(f"      ⚠️  配置异常：指向 L2 CORE")
    else:
        print(f"   ⚠️  L2 BACKUP 未加载到容器")
        
except Exception as e:
    print(f"   ❌ 检查失败：{e}")
    import traceback
    traceback.print_exc()

print()

# 4. 检查 vLLM 服务状态
print("4. vLLM 服务状态检查")
print("-" * 80)
import subprocess

def check_port(port):
    """检查端口是否有进程监听"""
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if f':{port}' in line and 'LISTENING' in line:
                return True
        return False
    except:
        return False

port_8000 = check_port(8000)
port_8001 = check_port(8001)

print(f"   端口 8000 (L2 CORE):    {'✅ 监听中' if port_8000 else '❌ 未监听'}")
print(f"   端口 8001 (L2 BACKUP):  {'✅ 监听中' if port_8001 else '❌ 未监听'}")

print()

# 5. 网络连接统计
print("5. 网络连接统计（最近 10 秒）")
print("-" * 80)
try:
    result = subprocess.run(
        ['netstat', '-ano'],
        capture_output=True,
        text=True,
        timeout=5
    )
    lines = result.stdout.split('\n')
    
    conn_8000 = sum(1 for line in lines if ':8000' in line and 'ESTABLISHED' in line)
    conn_8001 = sum(1 for line in lines if ':8001' in line and 'ESTABLISHED' in line)
    
    print(f"   L2 CORE (8000) 活跃连接数：   {conn_8000}")
    print(f"   L2 BACKUP (8001) 活跃连接数： {conn_8001}")
    
    if conn_8001 > conn_8000:
        print(f"\n   ⚠️  警告：L2 BACKUP 的连接数多于 L2 CORE")
        print(f"   💡 系统可能在使用 L2 BACKUP 进行推理")
    elif conn_8000 > 0:
        print(f"\n   ✅ 正常：L2 CORE 正在处理请求")
    else:
        print(f"\n   💡 提示：没有活跃连接，可能尚未进行推理")
        
except Exception as e:
    print(f"   ❌ 统计失败：{e}")

print()

# 6. 总结
print("=" * 80)
print("总结")
print("=" * 80)
print(f"   配置使用：{model_in_use}")
print(f"   L2 CORE 状态：   {'✅ 运行中' if port_8000 else '❌ 未运行'}")
print(f"   L2 BACKUP 状态： {'✅ 运行中' if port_8001 else '❌ 未运行'}")
print()

if model_in_use == "L2 BACKUP":
    print("   ⚠️  检测到系统正在使用 L2 BACKUP")
    print("   💡 建议检查:")
    print("      1. VLLM_BASE_URL 环境变量是否被覆盖")
    print("      2. 推理引擎是否有故障转移逻辑")
    print("      3. L2 CORE 是否启动失败")
elif model_in_use == "L2 CORE":
    print("   ✅ 系统配置正常，使用 L2 CORE")
else:
    print("   ❓ 无法确定使用的模型")

print()
print("=" * 80)
