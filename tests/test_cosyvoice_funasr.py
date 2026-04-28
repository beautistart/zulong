# File: tests/test_cosyvoice_funasr.py
"""
测试 FunASR 加载 CosyVoice 模型
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cosyvoice_funasr():
    from funasr import AutoModel
    
    model_path = r"d:\AI\project\zulong_beta4\models\CosyVoice3-0.5B\FunAudioLLM\Fun-CosyVoice3-0___5B-2512"
    
    print("\n" + "="*70)
    print("测试 FunASR 加载 CosyVoice 模型")
    print("="*70)
    print(f"\n模型路径: {model_path}")
    
    print("\n步骤 1: 尝试加载模型...")
    try:
        model = AutoModel(
            model=model_path,
            device="cpu",
        )
        print("  ✓ 模型加载成功")
        
        print("\n步骤 2: 测试推理...")
        result = model.generate("你好，我是祖龙机器人。")
        print(f"  结果: {result}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cosyvoice_modelscope_download():
    """使用 ModelScope SDK 下载并加载 CosyVoice"""
    from modelscope import snapshot_download
    
    print("\n" + "="*70)
    print("测试 ModelScope SDK 下载 CosyVoice")
    print("="*70)
    
    model_id = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    local_dir = r"d:\AI\project\zulong_beta4\models\FunAudioLLM\Fun-CosyVoice3-0.5B"
    
    print(f"\n模型 ID: {model_id}")
    print(f"本地目录: {local_dir}")
    
    try:
        print("\n下载中...")
        model_path = snapshot_download(
            model_id,
            local_dir=local_dir,
        )
        print(f"✓ 下载完成: {model_path}")
        return model_path
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return None


if __name__ == "__main__":
    test_cosyvoice_funasr()
