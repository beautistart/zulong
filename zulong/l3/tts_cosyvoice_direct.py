# File: zulong/l3/tts_cosyvoice_direct.py
"""
CosyVoice TTS 直接调用客户端
通过整合包的 Python 环境直接调用 CosyVoice 模型
"""

import os
import sys
import subprocess
import tempfile
import json
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CosyVoiceDirectClient:
    """
    CosyVoice 直接调用客户端
    
    通过整合包的 Python 环境直接调用 CosyVoice 模型进行语音合成
    支持 GPU 加速
    """
    
    def __init__(
        self,
        integrated_python_path: str = ""  # TODO: Set your Python executable path,
        model_dir: str = ""  # TODO: Set your CosyVoice model directory,
        code_path: str = ""  # TODO: Set your CosyVoice code path,
        default_prompt_audio: str = ""  # TODO: Set your prompt audio path,
        default_prompt_text: str = "希望你以后能够做的比我还好呦。",
        use_gpu: bool = True
    ):
        """
        初始化直接调用客户端
        
        Args:
            integrated_python_path: 整合包 Python 路径
            model_dir: 模型目录
            code_path: CosyVoice 代码路径
            default_prompt_audio: 默认提示音频
            default_prompt_text: 默认提示文本
            use_gpu: 是否使用 GPU 加速
        """
        self.integrated_python_path = integrated_python_path
        self.model_dir = model_dir
        self.code_path = code_path
        self.default_prompt_audio = default_prompt_audio
        self.default_prompt_text = default_prompt_text
        self.sample_rate = 24000
        self.use_gpu = use_gpu
        
        logger.info(f"🎤 CosyVoice 直接调用客户端初始化")
        logger.info(f"   Python: {integrated_python_path}")
        logger.info(f"   模型: {model_dir}")
        logger.info(f"   GPU: {'启用' if use_gpu else '禁用'}")
    
    def check_environment(self) -> bool:
        """检查环境是否可用"""
        if not os.path.exists(self.integrated_python_path):
            logger.error(f"Python 不存在: {self.integrated_python_path}")
            return False
        
        if not os.path.exists(self.model_dir):
            logger.error(f"模型目录不存在: {self.model_dir}")
            return False
        
        return True
    
    def synthesize(
        self,
        text: str,
        output_path: str,
        prompt_text: Optional[str] = None,
        prompt_audio_path: Optional[str] = None,
        mode: str = "zero_shot"
    ) -> bool:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            prompt_text: 提示文本 (zero_shot 模式)
            prompt_audio_path: 提示音频路径 (zero_shot 模式)
            mode: 合成模式 (zero_shot)
        
        Returns:
            bool: 是否成功
        """
        if not self.check_environment():
            return False
        
        if prompt_text is None:
            prompt_text = self.default_prompt_text
        
        if prompt_audio_path is None:
            prompt_audio_path = self.default_prompt_audio
        
        if not os.path.exists(prompt_audio_path):
            logger.error(f"提示音频不存在: {prompt_audio_path}")
            return False
        
        script = f'''
import os
import sys

sys.path.insert(0, r"{self.code_path}")
sys.path.insert(0, os.path.join(r"{self.code_path}", "third_party", "Matcha-TTS"))

import torch
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# 设置设备
device = "cuda" if torch.cuda.is_available() and {str(self.use_gpu)} else "cpu"
print(f"Using device: {{device}}")

if device == "cuda":
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")
    print(f"GPU Memory: {{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}} GB")

print("Loading CosyVoice2...")
cosy = CosyVoice2(r"{self.model_dir}", load_jit=False, load_trt=False, fp16=False)
print(f"Model loaded. Sample rate: {{cosy.sample_rate}}")
print(f"Model device: {{cosy.model.device}}")

print("Synthesizing...")
text = "{text}"
prompt_text = "{prompt_text}"
prompt_audio_path = r"{prompt_audio_path}"

# 加载音频保持在 CPU，模型内部会处理设备转换
prompt_speech_16k = load_wav(prompt_audio_path, 16000)
print(f"Prompt audio shape: {{prompt_speech_16k.shape}}")

for i, j in enumerate(cosy.inference_zero_shot(text, prompt_text, prompt_speech_16k)):
    # 输出已经在正确的设备上
    tts_speech = j["tts_speech"]
    if tts_speech.is_cuda:
        tts_speech = tts_speech.cpu()
    torchaudio.save(r"{output_path}", tts_speech, cosy.sample_rate)
    print(f"Saved: {output_path}")
    break

# 清理 GPU 缓存
if device == "cuda":
    torch.cuda.empty_cache()

print("DONE")
'''
        
        try:
            result = subprocess.run(
                [self.integrated_python_path, '-c', script],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=180
            )
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line and not line.startswith('2026-') and 'INFO' not in line:
                        print(f"  {line}")
            
            if result.returncode != 0:
                logger.error(f"TTS 合成失败: 返回码 {result.returncode}")
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if 'FutureWarning' not in line and 'UserWarning' not in line:
                            logger.error(f"  {line}")
                return False
            
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                logger.info(f"✓ 音频已保存: {output_path} ({size} bytes)")
                return True
            else:
                logger.error("✗ 音频文件未生成")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("TTS 合成超时")
            return False
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_direct_client():
    """测试直接调用客户端"""
    print("\n" + "="*70)
    print("测试 CosyVoice 直接调用客户端")
    print("="*70)
    
    client = CosyVoiceDirectClient()
    
    print("\n检查环境...")
    if not client.check_environment():
        print("✗ 环境检查失败")
        return False
    
    print("✓ 环境检查通过")
    
    output_path = r"d:\AI\project\zulong_beta4\tests\output_cosyvoice_direct_client.wav"
    
    print("\n测试语音合成...")
    success = client.synthesize(
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
    test_direct_client()
