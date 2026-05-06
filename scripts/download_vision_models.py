# File: scripts/download_vision_models.py
"""
优化视觉处理器模型下载脚本

下载 TSD v1.7 三层注意力机制所需的视觉模型：
1. YOLOv10-Nano: 人体检测 (Layer 1)
2. MobileNetV4-TSM: 动作分类 (Layer 3)
3. EfficientNet-B0: 手势识别 (Layer 4)

使用方法:
    python scripts/download_vision_models.py

模型将下载到：models/google/ 目录
"""

import os
import sys
from pathlib import Path

# 检查依赖
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ 缺少依赖：huggingface_hub")
    print("请运行：pip install huggingface_hub")
    sys.exit(1)


def download_model(repo_id: str, local_dir: str):
    """
    从 HuggingFace 下载模型
    
    Args:
        repo_id: HuggingFace 仓库 ID
        local_dir: 本地保存目录
    """
    print(f"\n{'='*60}")
    print(f"📥 下载模型：{repo_id}")
    print(f"📂 保存到：{local_dir}")
    print(f"{'='*60}")
    
    try:
        # 下载模型 (safetensors 格式)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            revision="main",
        )
        print(f"✅ 下载完成：{repo_id}")
        return True
    except Exception as e:
        print(f"❌ 下载失败：{e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 优化视觉处理器模型下载工具")
    print("TSD v1.7 三层注意力机制")
    print("="*60)
    
    # 获取项目根目录
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models" / "google"
    
    # 创建目录
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 模型保存目录：{models_dir}")
    
    # 模型列表
    models = [
        {
            'name': 'MobileNetV4-TSM (动作分类)',
            'repo_id': 'jaiwei98/MobileNetV4',
            'local_dir': str(models_dir.parent / 'jaiwei98' / 'MobileNetV4'),
            'layer': 'Layer 3',
        },
        {
            'name': 'EfficientNet-B0 (手势识别)',
            'repo_id': 'google/efficientnet-b0',
            'local_dir': str(models_dir / 'efficientnet-b0'),
            'layer': 'Layer 4',
        },
    ]
    
    # 检查 YOLOv10-Nano
    yolo_path = base_dir / "models" / "yolov10n.pt"
    if yolo_path.exists():
        print(f"\n✅ YOLOv10-Nano 已存在：{yolo_path}")
        print(f"   Layer 1: 人体检测 (已就绪)")
    else:
        print(f"\n⚠️ YOLOv10-Nano 不存在：{yolo_path}")
        print(f"   请手动下载或使用 ultralytics 下载:")
        print(f"   from ultralytics import YOLO")
        print(f"   YOLO('yolov10n.pt')")
    
    # 下载模型
    success_count = 0
    for model in models:
        print(f"\n📊 进度：{success_count}/{len(models)}")
        print(f"🎯 正在下载：{model['name']} ({model['layer']})")
        
        if download_model(model['repo_id'], model['local_dir']):
            success_count += 1
    
    # 总结
    print("\n" + "="*60)
    print("📊 下载总结")
    print("="*60)
    print(f"✅ 成功：{success_count}/{len(models)}")
    
    if success_count == len(models):
        print("\n🎉 所有模型下载完成！")
        print("\n📋 下一步:")
        print("1. 验证模型文件完整性")
        print("2. 运行测试：python tests/test_optimized_vision_integration.py")
        print("3. 在真实机器人上测试 3 米手势识别")
    else:
        print("\n⚠️ 部分模型下载失败，请检查网络连接")
        print("建议使用镜像源：")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
