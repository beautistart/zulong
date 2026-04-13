# File: zulong/l3/tts_cosyvoice_wrapper.py
"""
CosyVoice TTS 包装器
使用整合包的 Python 环境运行 CosyVoice
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CosyVoiceWrapper:
    """
    CosyVoice TTS 包装器
    
    使用整合包的 Python 环境来运行 CosyVoice，
    避免 Python 版本不兼容问题。
    """
    
    def __init__(
        self,
        integrated_python_path: str = ""  # TODO: Set your Python executable path,
        cosyvoice_code_path: str = ""  # TODO: Set your CosyVoice code path,
        model_dir: str = ""  # TODO: Set your CosyVoice model directory,
        device: str = "cpu"
    ):
        """
        初始化 CosyVoice 包装器
        
        Args:
            integrated_python_path: 整合包 Python 路径
            cosyvoice_code_path: CosyVoice 代码路径
            model_dir: 模型目录
            device: 运行设备 (cpu/cuda)
        """
        self.integrated_python_path = integrated_python_path
        self.cosyvoice_code_path = cosyvoice_code_path
        self.model_dir = model_dir
        self.device = device
        
        self.sample_rate = 22050
        
        logger.info(f"🎤 CosyVoice 包装器初始化")
        logger.info(f"   Python: {integrated_python_path}")
        logger.info(f"   模型: {model_dir}")
        logger.info(f"   设备: {device}")
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        prompt_text: Optional[str] = None,
        prompt_audio: Optional[str] = None,
        mode: str = "zero_shot"
    ) -> bool:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            prompt_text: 提示文本 (zero_shot 模式)
            prompt_audio: 提示音频路径 (zero_shot 模式)
            mode: 合成模式 (zero_shot/sft/instruct)
        
        Returns:
            bool: 是否成功
        """
        params = {
            "text": text,
            "output_path": output_path,
            "model_dir": self.model_dir,
            "code_path": self.cosyvoice_code_path,
            "mode": mode,
            "prompt_text": prompt_text or "希望你以后能够做的比我还好呦。",
            "prompt_audio": prompt_audio or os.path.join(self.cosyvoice_code_path, "asset", "zero_shot_prompt.wav")
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
            params_file = f.name
        
        script = '''
import os
import sys
import json

params_file = sys.argv[1]
with open(params_file, 'r', encoding='utf-8') as f:
    params = json.load(f)

sys.path.insert(0, params["code_path"])
sys.path.insert(0, os.path.join(params["code_path"], "third_party", "Matcha-TTS"))

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2

print("Loading CosyVoice2...")
cosy = CosyVoice2(params["model_dir"], load_jit=False, load_trt=False, fp16=False)
print(f"Model loaded. Sample rate: {cosy.sample_rate}")

print(f"Synthesizing: {params['text']}")

if params["mode"] == "zero_shot":
    for i, j in enumerate(cosy.inference_zero_shot(
        params["text"],
        params["prompt_text"],
        params["prompt_audio"]
    )):
        torchaudio.save(params["output_path"], j['tts_speech'], cosy.sample_rate)
        print(f"Saved: {params['output_path']}")
        break
else:
    spks = cosy.list_available_spks()
    if spks:
        for i, j in enumerate(cosy.inference_sft(params["text"], spks[0])):
            torchaudio.save(params["output_path"], j['tts_speech'], cosy.sample_rate)
            print(f"Saved: {params['output_path']}")
            break

print("DONE")
'''
        
        try:
            result = subprocess.run(
                [self.integrated_python_path, "-c", script, params_file],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=120
            )
            
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            
            if result.returncode != 0:
                logger.error(f"TTS 合成失败: {result.stderr}")
                return False
            
            if os.path.exists(output_path):
                logger.info(f"✓ 音频已保存: {output_path}")
                return True
            else:
                logger.error(f"✗ 音频文件未生成")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("TTS 合成超时")
            return False
        finally:
            if os.path.exists(params_file):
                os.unlink(params_file)


def test_cosyvoice_wrapper():
    """测试 CosyVoice 包装器"""
    print("\n" + "="*70)
    print("测试 CosyVoice 包装器")
    print("="*70)
    
    wrapper = CosyVoiceWrapper()
    
    output_path = r"d:\AI\project\zulong_beta4\tests\output_cosyvoice_wrapper.wav"
    
    success = wrapper.synthesize(
        text="你好，我是祖龙机器人。",
        output_path=output_path
    )
    
    if success:
        print(f"\n✓ 测试成功")
        print(f"  输出文件: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path)} bytes")
    else:
        print(f"\n✗ 测试失败")
    
    return success


if __name__ == "__main__":
    test_cosyvoice_wrapper()
