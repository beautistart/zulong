# File: zulong/l3/tts_cosyvoice_gradio.py
"""
CosyVoice TTS Gradio 客户端
通过 HTTP 调用整合包的 Gradio WebUI
"""

import os
import sys
import requests
import json
import time
from typing import Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class CosyVoiceGradioClient:
    """
    CosyVoice Gradio 客户端
    
    通过 HTTP 调用整合包的 Gradio WebUI 进行语音合成
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:50000",
        timeout: int = 120
    ):
        """
        初始化 Gradio 客户端
        
        Args:
            server_url: Gradio 服务器地址
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.sample_rate = 22050
        
        logger.info(f"🎤 CosyVoice Gradio 客户端初始化")
        logger.info(f"   服务器: {server_url}")
    
    def check_server(self) -> bool:
        """检查服务器是否可用"""
        try:
            response = requests.get(f"{self.server_url}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def synthesize_sft(
        self,
        text: str,
        output_path: str,
        speaker: Optional[str] = None,
        seed: int = 0,
        speed: float = 1.0
    ) -> bool:
        """
        使用预训练音色合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            speaker: 说话人名称（可选，默认使用第一个可用说话人）
            seed: 随机种子
            speed: 语速
        
        Returns:
            bool: 是否成功
        """
        return self._synthesize_with_gradio_client(
            text=text,
            output_path=output_path,
            mode="预训练音色",
            speaker=speaker,
            seed=seed,
            speed=speed
        )
    
    def synthesize_zero_shot(
        self,
        text: str,
        output_path: str,
        prompt_text: str,
        prompt_audio_path: str,
        seed: int = 0,
        speed: float = 1.0
    ) -> bool:
        """
        使用 3s 极速复刻模式合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出音频路径
            prompt_text: 提示文本
            prompt_audio_path: 提示音频路径
            seed: 随机种子
            speed: 语速
        
        Returns:
            bool: 是否成功
        """
        return self._synthesize_with_gradio_client(
            text=text,
            output_path=output_path,
            mode="3s极速复刻",
            prompt_text=prompt_text,
            prompt_audio_path=prompt_audio_path,
            seed=seed,
            speed=speed
        )
    
    def _synthesize_with_gradio_client(
        self,
        text: str,
        output_path: str,
        mode: str = "预训练音色",
        speaker: Optional[str] = None,
        prompt_text: str = "",
        prompt_audio_path: Optional[str] = None,
        instruct_text: str = "",
        seed: int = 0,
        speed: float = 1.0
    ) -> bool:
        """
        使用 gradio_client 库进行合成
        
        API 参数顺序 (从 webui.py):
        1. tts_text - 合成文本
        2. mode_checkbox_group - 模式
        3. sft_dropdown - 预训练音色
        4. prompt_text - 提示文本
        5. prompt_wav_upload - 上传音频
        6. prompt_wav_record - 录制音频
        7. instruct_text - 指令文本
        8. seed - 随机种子
        9. stream - 是否流式
        10. speed - 语速
        """
        try:
            from gradio_client import Client, handle_file
            import soundfile as sf
            import tempfile
            import shutil
            
            print(f"\n连接 Gradio 服务器: {self.server_url}")
            client = Client(self.server_url, verbose=False)
            
            print(f"模式: {mode}")
            print(f"文本: {text}")
            
            if mode == "预训练音色":
                if speaker is None:
                    speaker = "中文女"
                print(f"说话人: {speaker}")
                
                result = client.predict(
                    text,                    # tts_text
                    mode,                    # mode_checkbox_group
                    speaker,                 # sft_dropdown
                    "",                      # prompt_text
                    None,                    # prompt_wav_upload
                    None,                    # prompt_wav_record
                    "",                      # instruct_text
                    seed,                    # seed
                    False,                   # stream (布尔值)
                    speed,                   # speed
                    api_name="/generate_audio"
                )
                
            elif mode == "3s极速复刻":
                print(f"提示文本: {prompt_text}")
                print(f"提示音频: {prompt_audio_path}")
                
                result = client.predict(
                    text,                    # tts_text
                    mode,                    # mode_checkbox_group
                    "",                      # sft_dropdown
                    prompt_text,             # prompt_text
                    handle_file(prompt_audio_path),  # prompt_wav_upload
                    None,                    # prompt_wav_record
                    "",                      # instruct_text
                    seed,                    # seed
                    False,                   # stream (布尔值)
                    speed,                   # speed
                    api_name="/generate_audio"
                )
            else:
                logger.error(f"不支持的模式: {mode}")
                return False
            
            print(f"结果类型: {type(result)}")
            print(f"结果内容: {result}")
            
            if result:
                if isinstance(result, str):
                    print(f"  检测到文件路径: {result}")
                    
                    if result.endswith('.m3u8') or result.endswith('.m3u'):
                        print(f"  处理 M3U8 播放列表...")
                        with open(result, 'r', encoding='utf-8') as f:
                            m3u_content = f.read()
                        print(f"  播放列表内容:\n{m3u_content}")
                        
                        lines = m3u_content.strip().split('\n')
                        
                        for line in lines:
                            if line and not line.startswith('#'):
                                audio_file = line.strip()
                                print(f"  音频文件: {audio_file}")
                                
                                url = f"{self.server_url}/file={audio_file}"
                                print(f"  下载 URL: {url}")
                                
                                try:
                                    resp = requests.get(url, timeout=30)
                                    print(f"  响应状态: {resp.status_code}, 大小: {len(resp.content)} bytes")
                                    
                                    with tempfile.NamedTemporaryFile(suffix='.aac', delete=False) as tmp:
                                        tmp.write(resp.content)
                                        tmp_path = tmp.name
                                    
                                    try:
                                        data, sr = sf.read(tmp_path)
                                        if sample_rate is None:
                                            sample_rate = sr
                                            print(f"  采样率: {sample_rate}")
                                        all_audio_chunks.append(data)
                                        print(f"  音频块大小: {len(data)} samples")
                                    except Exception as e:
                                        print(f"  直接读取 AAC 失败: {e}")
                                        
                                        import subprocess
                                        wav_path = tmp_path.replace('.aac', '.wav')
                                        result_ffmpeg = subprocess.run(
                                            ['ffmpeg', '-i', tmp_path, wav_path, '-y'],
                                            capture_output=True, text=True
                                        )
                                        if result_ffmpeg.returncode != 0:
                                            print(f"  FFmpeg stderr: {result_ffmpeg.stderr}")
                                        
                                        if os.path.exists(wav_path):
                                            data, sr = sf.read(wav_path)
                                            if sample_rate is None:
                                                sample_rate = sr
                                            all_audio_chunks.append(data)
                                            print(f"  FFmpeg 转换后音频块大小: {len(data)} samples")
                                            os.unlink(wav_path)
                                    
                                    os.unlink(tmp_path)
                                except Exception as e:
                                    print(f"  下载音频失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                    
                    elif result.endswith('.wav'):
                        print(f"  检测到 WAV 文件")
                        data, sr = sf.read(result)
                        sf.write(output_path, data, sr)
                        logger.info(f"✓ 音频已保存: {output_path}")
                        return True
                    
                    elif result.endswith('.aac'):
                        print(f"  检测到 AAC 文件")
                        try:
                            data, sr = sf.read(result)
                            sf.write(output_path, data, sr)
                            logger.info(f"✓ 音频已保存: {output_path}")
                            return True
                        except Exception as e:
                            print(f"  直接读取失败: {e}")
                            import subprocess
                            wav_path = result.rsplit('.', 1)[0] + '.wav'
                            subprocess.run(['ffmpeg', '-i', result, wav_path, '-y'], 
                                          capture_output=True, check=False)
                            if os.path.exists(wav_path):
                                data, sr = sf.read(wav_path)
                                sf.write(output_path, data, sr)
                                logger.info(f"✓ 音频已保存: {output_path}")
                                os.unlink(wav_path)
                                return True
            
            if all_audio_chunks:
                print(f"\n合并 {len(all_audio_chunks)} 个音频块...")
                full_audio = np.concatenate(all_audio_chunks)
                sf.write(output_path, full_audio, sample_rate)
                logger.info(f"✓ 音频已保存: {output_path}")
                return True
            
            logger.error("✗ 未获取到音频数据")
            return False
            
        except ImportError:
            logger.error("请安装 gradio_client: pip install gradio_client")
            return False
        except Exception as e:
            logger.error(f"TTS 合成失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_gradio_client():
    """测试 Gradio 客户端"""
    print("\n" + "="*70)
    print("测试 CosyVoice Gradio 客户端")
    print("="*70)
    
    client = CosyVoiceGradioClient()
    
    print("\n检查服务器...")
    if not client.check_server():
        print("✗ 服务器未运行，请先启动 start.bat")
        print("  访问地址: http://localhost:50000")
        return False
    
    print("✓ 服务器已运行")
    
    output_path = r"d:\AI\project\zulong_beta4\tests\output_cosyvoice_gradio.wav"
    
    print("\n测试 3s 极速复刻模式（CosyVoice2-0.5B 无预训练音色）...")
    
    prompt_audio = ""  # TODO: Set your prompt audio path
    prompt_text = "希望你以后能够做的比我还好呦。"
    
    if not os.path.exists(prompt_audio):
        print(f"✗ 参考音频不存在: {prompt_audio}")
        return False
    
    success = client.synthesize_zero_shot(
        text="你好，我是祖龙机器人。",
        output_path=output_path,
        prompt_text=prompt_text,
        prompt_audio_path=prompt_audio
    )
    
    if success:
        print(f"\n✓ 测试成功")
        print(f"  输出文件: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path)} bytes")
    else:
        print(f"\n✗ 测试失败")
        print("\n请尝试在浏览器中手动测试: http://localhost:50000")
    
    return success


if __name__ == "__main__":
    test_gradio_client()
