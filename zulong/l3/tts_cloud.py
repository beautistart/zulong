# File: zulong/l3/tts_cloud.py
"""
云端 TTS 服务
支持多种在线 TTS API
"""

import os
import asyncio
import aiohttp
import hashlib
import time
from typing import Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TTSProvider(ABC):
    """TTS 服务提供商基类"""
    
    @abstractmethod
    async def synthesize(self, text: str, output_path: str) -> bool:
        """合成语音"""
        pass


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS (免费)"""
    
    VOICES = {
        "zh-CN-XiaoxiaoNeural": "晓晓 (女声，自然)",
        "zh-CN-YunxiNeural": "云希 (男声，年轻)",
        "zh-CN-YunyangNeural": "云扬 (男声，新闻)",
        "zh-CN-XiaoyiNeural": "晓伊 (女声，温柔)",
        "zh-CN-YunjianNeural": "云健 (男声，沉稳)",
        "zh-CN-XiaochenNeural": "晓辰 (女声，甜美)",
        "zh-CN-XiaohanNeural": "晓涵 (女声，温暖)",
        "zh-CN-XiaomengNeural": "晓梦 (女声，可爱)",
        "zh-CN-XiaomoNeural": "晓墨 (女声，知性)",
        "zh-CN-XiaoruiNeural": "晓睿 (女声，活泼)",
        "zh-CN-XiaoshuangNeural": "晓双 (女声，儿童)",
        "zh-CN-XiaoxuanNeural": "晓萱 (女声，优雅)",
        "zh-CN-XiaoyanNeural": "晓颜 (女声，亲切)",
        "zh-CN-XiaoyouNeural": "晓悠 (女声，童声)",
        "zh-CN-YunfengNeural": "云枫 (男声，磁性)",
        "zh-CN-YunhaoNeural": "云皓 (男声，活力)",
        "zh-CN-YunxiaNeural": "云夏 (男声，少年)",
        "zh-CN-YunyeNeural": "云野 (男声，文艺)",
    }
    
    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
    
    async def synthesize(self, text: str, output_path: str) -> bool:
        try:
            import edge_tts
            
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)
            
            if os.path.exists(output_path):
                logger.info(f"✓ Edge TTS 合成成功: {output_path}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Edge TTS 合成失败: {e}")
            return False


class AliyunTTSProvider(TTSProvider):
    """阿里云 TTS"""
    
    def __init__(
        self,
        access_key_id: str = "",
        access_key_secret: str = "",
        app_key: str = "",
        voice: str = "xiaoyun"
    ):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.app_key = app_key
        self.voice = voice
    
    async def synthesize(self, text: str, output_path: str) -> bool:
        try:
            import requests
            
            url = "https://nls-gateway.cn-shanghai.aliyuncs.com/stream/v1/tts"
            
            timestamp = str(int(time.time() * 1000))
            
            params = {
                "appkey": self.app_key,
                "text": text,
                "format": "wav",
                "sample_rate": 16000,
                "voice": self.voice,
                "volume": 50,
                "speech_rate": 0,
                "pitch_rate": 0
            }
            
            headers = {
                "X-NLS-Token": self._generate_token(timestamp),
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"✓ 阿里云 TTS 合成成功: {output_path}")
                return True
            else:
                logger.error(f"阿里云 TTS 失败: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"阿里云 TTS 合成失败: {e}")
            return False
    
    def _generate_token(self, timestamp: str) -> str:
        return hashlib.md5(
            f"{self.access_key_id}{timestamp}{self.access_key_secret}".encode()
        ).hexdigest()


class TencentTTSProvider(TTSProvider):
    """腾讯云 TTS"""
    
    def __init__(
        self,
        secret_id: str = "",
        secret_key: str = "",
        voice_type: int = 1
    ):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.voice_type = voice_type
    
    async def synthesize(self, text: str, output_path: str) -> bool:
        try:
            from tencentcloud.common import credential
            from tencentcloud.tts.v20190823 import tts_client, models
            
            cred = credential.Credential(self.secret_id, self.secret_key)
            client = tts_client.TtsClient(cred, "ap-beijing")
            
            req = models.TextToVoiceRequest()
            req.Text = text
            req.VoiceType = self.voice_type
            req.Codec = "wav"
            req.SampleRate = 16000
            
            resp = client.TextToVoice(req)
            
            import base64
            audio_data = base64.b64decode(resp.Audio)
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            logger.info(f"✓ 腾讯云 TTS 合成成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"腾讯云 TTS 合成失败: {e}")
            return False


class CloudTTSClient:
    """云端 TTS 客户端"""
    
    def __init__(self, provider: str = "edge", **kwargs):
        """
        初始化云端 TTS 客户端
        
        Args:
            provider: 服务提供商 (edge/aliyun/tencent)
            **kwargs: 提供商特定参数
        """
        self.provider_name = provider
        self.provider = self._create_provider(provider, **kwargs)
        
        logger.info(f"☁️ 云端 TTS 客户端初始化")
        logger.info(f"   提供商: {provider}")
    
    def _create_provider(self, provider: str, **kwargs) -> TTSProvider:
        if provider == "edge":
            return EdgeTTSProvider(voice=kwargs.get("voice", "zh-CN-XiaoxiaoNeural"))
        elif provider == "aliyun":
            return AliyunTTSProvider(
                access_key_id=kwargs.get("access_key_id", ""),
                access_key_secret=kwargs.get("access_key_secret", ""),
                app_key=kwargs.get("app_key", ""),
                voice=kwargs.get("voice", "xiaoyun")
            )
        elif provider == "tencent":
            return TencentTTSProvider(
                secret_id=kwargs.get("secret_id", ""),
                secret_key=kwargs.get("secret_key", ""),
                voice_type=kwargs.get("voice_type", 1)
            )
        else:
            raise ValueError(f"不支持的 TTS 提供商: {provider}")
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """同步合成语音"""
        return asyncio.run(self.synthesize_async(text, output_path))
    
    async def synthesize_async(self, text: str, output_path: str) -> bool:
        """异步合成语音"""
        start_time = time.time()
        result = await self.provider.synthesize(text, output_path)
        elapsed = time.time() - start_time
        
        if result:
            size = os.path.getsize(output_path)
            logger.info(f"   耗时: {elapsed:.2f}s, 大小: {size} bytes")
        
        return result
    
    @staticmethod
    def list_voices(provider: str = "edge"):
        """列出可用声音"""
        if provider == "edge":
            print("\n可用声音 (Edge TTS):")
            for voice, desc in EdgeTTSProvider.VOICES.items():
                print(f"  {voice}: {desc}")
        else:
            print(f"提供商 {provider} 的声音列表请参考官方文档")


def test_cloud_tts():
    """测试云端 TTS"""
    print("\n" + "="*60)
    print("云端 TTS 测试")
    print("="*60)
    
    client = CloudTTSClient(provider="edge", voice="zh-CN-XiaoxiaoNeural")
    
    output_dir = r"d:\AI\project\zulong_beta4\tests\tts_output"
    os.makedirs(output_dir, exist_ok=True)
    
    test_cases = [
        "你好，我是祖龙机器人。",
        "今天天气真好。",
        "我可以帮你完成各种任务。",
        "请问有什么可以帮你的吗？",
        "再见，祝你有美好的一天。"
    ]
    
    print(f"\n测试 {len(test_cases)} 个句子...\n")
    
    total_time = 0
    success_count = 0
    
    for i, text in enumerate(test_cases):
        output_path = os.path.join(output_dir, f"cloud_tts_{i+1}.wav")
        
        start_time = time.time()
        success = client.synthesize(text=text, output_path=output_path)
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if success:
            size = os.path.getsize(output_path)
            print(f"[{i+1}] {text[:20]}... ✓ {elapsed:.2f}s ({size} bytes)")
            success_count += 1
        else:
            print(f"[{i+1}] {text[:20]}... ✗")
    
    print(f"\n总计: {total_time:.2f}s, 平均: {total_time/len(test_cases):.2f}s/句")
    print(f"成功率: {success_count}/{len(test_cases)}")


if __name__ == "__main__":
    CloudTTSClient.list_voices()
    test_cloud_tts()
