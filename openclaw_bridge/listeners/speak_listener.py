# File: openclaw_bridge\listeners\speak_listener.py
"""
OpenClaw 语音播报监听器

功能：
1. 监听 ACTION_SPEAK 事件（来自 L1-B）
2. 提取 payload.text
3. 调用 OpenClaw 的 TTS 模块或直接打印文本

对应 TSD v2.2 第 3.1 节（事件模型）、第 4.3 节（L1-B 下行链路）
关键逻辑：支持 Mock 模式（打印文本）和真实 TTS 模式
"""

import asyncio
import time
import logging
from typing import Optional
from dataclasses import dataclass

from openclaw_bridge.openclaw_types import ZulongEvent, OpenClawEventType
from openclaw_bridge.event_bus_client import EventBusClient

logger = logging.getLogger(__name__)


@dataclass
class SpeakConfig:
    """语音播报配置"""
    mock_mode: bool = True  # Mock 模式（打印文本）
    enable_log: bool = True  # 启用日志
    tts_engine: str = "mock"  # TTS 引擎（mock, baidu, azure, etc.）
    volume: float = 1.0  # 音量（0.0-1.0）
    rate: float = 1.0  # 语速（0.5-2.0）


class SpeakListener:
    """
    语音播报监听器
    
    工作流程：
    1. 订阅 ACTION_SPEAK 事件
    2. 提取播报文本
    3. 调用 TTS 引擎播报
    4. 支持 Mock 模式（打印）和真实 TTS
    """
    
    def __init__(
        self,
        event_bus: EventBusClient,
        config: Optional[SpeakConfig] = None
    ):
        """
        初始化语音播报监听器
        
        Args:
            event_bus: EventBus 客户端
            config: 语音播报配置
        """
        self.event_bus = event_bus
        self.config = config or SpeakConfig()
        
        logger.info(
            f"[SpeakListener] 初始化完成，Mock 模式：{self.config.mock_mode}"
        )
    
    async def on_speak_event(self, event: ZulongEvent):
        """
        处理语音播报事件
        
        Args:
            event: 语音播报事件
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔊 [SpeakListener] 收到语音播报事件")
        logger.info(f"🔊 [SpeakListener] 文本：{event.payload.get('text')}")
        logger.info(f"{'='*80}\n")
        
        text = event.payload.get("text")
        
        if not text:
            logger.warning("[SpeakListener] ⚠️ 播报文本为空")
            return
        
        try:
            # 执行播报
            await self._speak(text)
            
        except Exception as e:
            logger.error(f"[SpeakListener] ❌ 播报失败：{e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _speak(self, text: str):
        """
        执行语音播报
        
        Args:
            text: 播报文本
        """
        if self.config.mock_mode:
            # Mock 模式：打印文本
            await self._mock_speak(text)
        else:
            # 真实 TTS 模式
            await self._real_tts_speak(text)
    
    async def _mock_speak(self, text: str):
        """
        Mock 语音播报（打印文本）
        
        Args:
            text: 播报文本
        """
        if self.config.enable_log:
            print("\n" + "=" * 80)
            print(f"🔊 [机器人播报] {text}")
            print("=" * 80 + "\n")
        
        logger.info(f"[SpeakListener] 🔊 Mock 播报：{text}")
        
        # 模拟播报时间（根据文本长度）
        duration = len(text) * 0.1  # 每个字 0.1 秒
        await asyncio.sleep(min(duration, 3.0))
    
    async def _real_tts_speak(self, text: str):
        """
        真实 TTS 语音播报
        
        Args:
            text: 播报文本
        
        TODO: 实现真实 TTS 引擎对接
        """
        if self.config.tts_engine == "mock":
            await self._mock_speak(text)
            return
        
        # 根据 TTS 引擎调用对应的 API
        if self.config.tts_engine == "baidu":
            # TODO: 百度 TTS
            # from aip import AipSpeech
            # client = AipSpeech(app_id, api_key, secret_key)
            # result = client.synthesis(text, 'zh', 1, {'vol': self.config.volume * 100})
            # play_audio(result['audio'])
            raise NotImplementedError("百度 TTS 尚未实现")
        
        elif self.config.tts_engine == "azure":
            # TODO: Azure TTS
            # import azure.cognitiveservices.speech as speechsdk
            # speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
            # synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            # synthesizer.speak_text_async(text).get()
            raise NotImplementedError("Azure TTS 尚未实现")
        
        else:
            logger.warning(f"[SpeakListener] ⚠️ 未知 TTS 引擎：{self.config.tts_engine}")
            await self._mock_speak(text)
    
    def set_volume(self, volume: float):
        """
        设置音量
        
        Args:
            volume: 音量（0.0-1.0）
        """
        self.config.volume = max(0.0, min(1.0, volume))
        logger.info(f"[SpeakListener] 音量已设置为：{self.config.volume}")
    
    def set_rate(self, rate: float):
        """
        设置语速
        
        Args:
            rate: 语速（0.5-2.0）
        """
        self.config.rate = max(0.5, min(2.0, rate))
        logger.info(f"[SpeakListener] 语速已设置为：{self.config.rate}")


# 便捷创建函数

def create_speak_listener(
    event_bus: EventBusClient,
    mock_mode: bool = True
) -> SpeakListener:
    """
    创建语音播报监听器
    
    Args:
        event_bus: EventBus 客户端
        mock_mode: 是否使用 Mock 模式
    
    Returns:
        SpeakListener: 语音播报监听器实例
    """
    config = SpeakConfig(mock_mode=mock_mode)
    listener = SpeakListener(event_bus, config)
    return listener
