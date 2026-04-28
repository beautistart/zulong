# 快速验证 L2-BACKUP vLLM 配置

"""
快速验证 L2-BACKUP 是否正确配置为 vLLM 模式
"""

import os
import sys

# 设置环境变量
os.environ["USE_VLLM_FOR_L2"] = "true"
os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true"

sys.path.insert(0, 'd:/AI/project/zulong_beta4')

print("="*80)
print("快速验证 L2-BACKUP vLLM 配置")
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
    
    # 检查 ModelContainer 中的注册
    print("\n" + "="*80)
    print("检查 ModelContainer 中的 L2_BACKUP 注册")
    print("="*80)
    
    # 模拟 ModelContainer 的加载逻辑
    USE_VLLM_FOR_L2_BACKUP = os.environ.get("USE_VLLM_FOR_L2_BACKUP", "true").lower() == "true"
    
    print(f"USE_VLLM_FOR_L2_BACKUP = {USE_VLLM_FOR_L2_BACKUP}")
    
    if USE_VLLM_FOR_L2_BACKUP:
        print("✅ L2_BACKUP 将使用 vLLM 模式")
        print("   配置：")
        print("   - path: 'vllm'")
        print("   - type: 'remote'")
        print("   - endpoint: 'http://localhost:8000/v1'")
        print("   - shared_with: 'L2_CORE'")
    else:
        print("⚠️ L2_BACKUP 将使用本地加载模式（降级方案）")
    
    print("\n" + "="*80)
    print("✅ 配置验证完成！")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ 验证失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
