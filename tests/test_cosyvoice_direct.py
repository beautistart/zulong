# File: tests/test_cosyvoice_direct.py
"""直接调用 CosyVoice2 模型测试"""

import subprocess
import os
import tempfile

def test_cosyvoice_direct():
    """测试直接调用 CosyVoice2"""
    print("\n" + "="*70)
    print("测试 CosyVoice2 直接调用")
    print("="*70)
    
    integrated_python = r'D:\BaiduNetdiskDownload\CosyVoiceV2\python\python.exe'
    model_dir = r'D:\BaiduNetdiskDownload\CosyVoiceV2\CosyVoice\iic\CosyVoice2-0.5B'
    code_path = r'D:\BaiduNetdiskDownload\CosyVoiceV2\CosyVoice'
    output_path = r'd:\AI\project\zulong_beta4\tests\output_cosyvoice_direct.wav'
    
    script = f'''
import os
import sys

sys.path.insert(0, r"{code_path}")
sys.path.insert(0, os.path.join(r"{code_path}", "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

print("Loading CosyVoice2...")
cosy = CosyVoice2(r"{model_dir}", load_jit=False, load_trt=False, fp16=False)
print(f"Model loaded. Sample rate: {{cosy.sample_rate}}")

print("Synthesizing...")
text = "你好，我是祖龙机器人。"
prompt_text = "希望你以后能够做的比我还好呦。"
prompt_audio_path = r"{code_path}\\asset\\zero_shot_prompt.wav"

# 加载音频文件
prompt_speech_16k = load_wav(prompt_audio_path, 16000)
print(f"Prompt audio shape: {{prompt_speech_16k.shape}}")

for i, j in enumerate(cosy.inference_zero_shot(text, prompt_text, prompt_speech_16k)):
    torchaudio.save(r"{output_path}", j["tts_speech"], cosy.sample_rate)
    print(f"Saved: {output_path}")
    break

print("DONE")
'''
    
    print(f"Python: {integrated_python}")
    print(f"Model: {model_dir}")
    print(f"Output: {output_path}")
    
    result = subprocess.run(
        [integrated_python, '-c', script],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=120
    )
    
    print('\n--- STDOUT ---')
    print(result.stdout)
    
    if result.stderr:
        print('\n--- STDERR ---')
        print(result.stderr)
    
    print(f'\nReturn code: {result.returncode}')
    
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f'\n✓ 音频文件已生成: {output_path}')
        print(f'  文件大小: {size} bytes')
        return True
    else:
        print(f'\n✗ 音频文件未生成')
        return False

if __name__ == "__main__":
    test_cosyvoice_direct()
