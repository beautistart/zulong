# File: tests/test_cosyvoice_local.py
"""
测试本地 CosyVoice (从整合包)
"""

import os
import sys

sys.path.insert(0, r"d:\AI\project\zulong_beta4\third_party\tn")
sys.path.insert(0, r"d:\AI\project\zulong_beta4\third_party\CosyVoice\CosyVoice")
sys.path.insert(0, r"d:\AI\project\zulong_beta4\third_party\CosyVoice\CosyVoice\third_party\Matcha-TTS")

import torch
import numpy as np


def test_cosyvoice2_local():
    """测试本地 CosyVoice2"""
    print("\n" + "="*70)
    print("测试本地 CosyVoice2 (整合包)")
    print("="*70)
    
    model_dir = r"d:\AI\project\zulong_beta4\third_party\CosyVoice\CosyVoice\iic\CosyVoice2-0.5B"
    
    print(f"\n模型路径: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"  ✗ 模型路径不存在")
        return False
    
    print("\n步骤 1: 检查模型文件...")
    required_files = [
        "cosyvoice2.yaml",
        "campplus.onnx",
        "speech_tokenizer_v2.onnx",
    ]
    
    for f in required_files:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} 不存在")
    
    print("\n步骤 2: 导入 CosyVoice...")
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        print("  ✓ CosyVoice2 导入成功")
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False
    
    print("\n步骤 3: 加载模型 (CPU)...")
    try:
        cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)
        print("  ✓ 模型加载成功")
        print(f"  采样率: {cosy.sample_rate}")
    except Exception as e:
        print(f"  ✗ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n步骤 4: 测试语音合成...")
    try:
        text = "你好，我是祖龙机器人。"
        print(f"  输入文本: {text}")
        
        import torchaudio
        
        output_file = r"d:\AI\project\zulong_beta4\tests\output_cosyvoice2.wav"
        
        for i, j in enumerate(cosy.inference_zero_shot(
            text,
            "希望你以后能够做的比我还好呦。",
            r"d:\AI\project\zulong_beta4\third_party\CosyVoice\CosyVoice\asset\zero_shot_prompt.wav"
        )):
            torchaudio.save(output_file, j['tts_speech'], cosy.sample_rate)
            print(f"  ✓ 音频已保存: {output_file}")
            print(f"  文件大小: {os.path.getsize(output_file)} bytes")
            break
        
        return True
        
    except Exception as e:
        print(f"  ✗ 合成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cosyvoice_dependencies():
    """检查依赖"""
    print("\n" + "="*70)
    print("检查 CosyVoice 依赖")
    print("="*70)
    
    dependencies = [
        "torch",
        "torchaudio",
        "onnxruntime",
        "hyperpyyaml",
        "modelscope",
        "tn",
    ]
    
    for dep in dependencies:
        try:
            mod = __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError as e:
            print(f"  ✗ {dep} 未安装: {e}")


if __name__ == "__main__":
    test_cosyvoice_dependencies()
    test_cosyvoice2_local()
