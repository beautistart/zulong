# File: scripts/start_zulong.py
# 祖龙系统启动脚本 - 设置环境变量

import os
import sys

# 设置 vLLM 环境变量
# L2 BACKUP 复用 L2 CORE 的模型实例（共享同一个vLLM服务）
os.environ["USE_VLLM_FOR_L2"] = "true"
os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true"  # 启用L2 BACKUP功能
os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"
os.environ["VLLM_BACKUP_BASE_URL"] = "http://localhost:8000/v1"  # 复用L2 CORE的端口

print("=" * 80)
print("                    祖龙系统启动脚本")
print("=" * 80)
print()
print("环境变量配置:")
print(f"  - USE_VLLM_FOR_L2: {os.environ['USE_VLLM_FOR_L2']}")
print(f"  - USE_VLLM_FOR_L2_BACKUP: {os.environ['USE_VLLM_FOR_L2_BACKUP']}")
print(f"  - VLLM_BASE_URL: {os.environ['VLLM_BASE_URL']}")
print(f"  - VLLM_BACKUP_BASE_URL: {os.environ['VLLM_BACKUP_BASE_URL']}")
print()
print("=" * 80)
print()

# 启动祖龙
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zulong import bootstrap

if __name__ == "__main__":
    import logging
    
    # 配置全局日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    bootstrap.main()
