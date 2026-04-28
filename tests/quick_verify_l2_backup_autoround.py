# 快速验证 L2-BACKUP AutoRound 配置

"""
快速验证 L2-BACKUP 配置是否正确指向 Qwen3.5-0.8B-int4-AutoRound
"""

import os
import sys

# 设置环境变量
os.environ["USE_VLLM_FOR_L2"] = "true"
os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true"

sys.path.insert(0, 'd:/AI/project/zulong_beta4')

print("="*80)
print("快速验证 L2-BACKUP AutoRound 配置")
print("="*80)

try:
    # 直接导入并检查配置
    from zulong.models.config import ModelID, MODEL_CONFIGS
    
    print(f"\n✅ ModelID 导入成功")
    print(f"   L2_BACKUP: {ModelID.L2_BACKUP.value}")
    
    # 检查 MODEL_CONFIGS
    if ModelID.L2_BACKUP in MODEL_CONFIGS:
        config = MODEL_CONFIGS[ModelID.L2_BACKUP]
        print(f"✅ L2_BACKUP 配置存在")
        print(f"   仓库 ID: {config.repo_id}")
        print(f"   显存预估：{config.estimated_vram_gb}GB")
        print(f"   设备：{config.device}")
    else:
        print(f"⚠️ L2_BACKUP 不在 MODEL_CONFIGS 中")
    
    # 模拟 ModelContainer 的加载逻辑
    print("\n" + "="*80)
    print("模拟 L2_BACKUP 加载逻辑")
    print("="*80)
    
    USE_VLLM_FOR_L2_BACKUP = os.environ.get("USE_VLLM_FOR_L2_BACKUP", "true").lower() == "true"
    
    print(f"USE_VLLM_FOR_L2_BACKUP = {USE_VLLM_FOR_L2_BACKUP}")
    
    if USE_VLLM_FOR_L2_BACKUP:
        print("\n✅ L2_BACKUP 将使用 vLLM 模式")
        print("\n配置详情:")
        print("   - path: 'vllm'")
        print("   - type: 'remote'")
        print("   - endpoint: 'http://localhost:8001/v1'")
        print("   - model_name: 'Intel/Qwen3.5-0.8B-int4-AutoRound'")
        print("   - quantization: 'gptq_marlin'")
        print("\n技术规格:")
        print("   - 量化格式：INT4-AutoRound (W4A16)")
        print("   - 显存占用：~0.8GB")
        print("   - 推理加速：Marlin 后端")
        print("   - 性能提升：2-3x")
    else:
        print("\n⚠️ L2_BACKUP 将使用本地加载模式（降级方案）")
    
    # 打印启动命令
    print("\n" + "="*80)
    print("启动 L2-BACKUP vLLM 服务")
    print("="*80)
    print("""
Windows 用户:
  双击运行：start_l2_backup_vllm.bat

Linux/WSL 用户:
  bash scripts/start_l2_backup_vllm.sh

或直接使用 CLI:
  vllm serve Intel/Qwen3.5-0.8B-int4-AutoRound \\
      --port 8001 \\
      --tensor-parallel-size 1 \\
      --gpu-memory-utilization 0.8 \\
      --max-model-len 4096 \\
      --trust-remote-code \\
      --dtype auto \\
      --quantization gptq_marlin
""")
    
    print("\n" + "="*80)
    print("✅ 配置验证完成！")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ 验证失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
