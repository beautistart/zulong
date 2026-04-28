# File: tests/test_l2_model.py
"""
测试 L2 模型加载
"""
import sys
import os

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

def test_l2_model():
    """测试 L2 模型加载"""
    print("\n" + "=" * 60)
    print("测试 L2 模型加载")
    print("=" * 60)
    
    try:
        # 创建模型容器（会加载常驻模型）
        print("\n创建模型容器...")
        container = ModelContainer()
        
        print(f"\n已加载的常驻模型:")
        for model_id, model in container.resident_models.items():
            print(f"  - {model_id.value}: {type(model).__name__}")
        
        # 检查 L2_CORE 是否已加载
        print(f"\n检查 L2_CORE 模型...")
        if ModelID.L2_CORE in container.resident_models:
            print(f"✅ L2_CORE 已加载")
            l2_model = container.resident_models[ModelID.L2_CORE]
            print(f"   类型：{type(l2_model)}")
            print(f"   路径：{l2_model.model_name if hasattr(l2_model, 'model_name') else 'N/A'}")
        else:
            print(f"❌ L2_CORE 未加载")
            print(f"   可用模型：{list(container.resident_models.keys())}")
        
        # 检查 L2_BACKUP
        print(f"\n检查 L2_BACKUP 模型...")
        if ModelID.L2_BACKUP in container.resident_models:
            print(f"✅ L2_BACKUP 已加载")
        else:
            print(f"❌ L2_BACKUP 未加载")
        
        print(f"\n当前显存使用：{container.current_vram_usage:.2f} GB")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_l2_model()
