# File: openclaw_bridge\adapters\mic_adapter.py
"""
OpenClaw 麦克风适配器

功能：
1. 监听麦克风输入（模拟或真实）
2. 检测语音结束
3. 将语音文本封装成 ZulongEvent
4. 通过 EventBus 发送给 L1-B

对应 TSD v2.2 第 3.1 节（事件模型）
关键配置：type="USER_SPEECH", source="openclaw/mic", priority=5
"""

import asyncio
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass

from openclaw_bridge.openclaw_types import ZulongEvent, OpenClawEventType, OpenClawEventPriority
from openclaw_bridge.event_bus_client import EventBusClient

logger = logging.getLogger(__name__)


@dataclass
class MicConfig:
    """麦克风配置"""
    silence_threshold: float = 0.5  # 静音阈值（秒）
    sample_rate: int = 16000  # 采样率
    mock_mode: bool = True  # Mock 模式（开发测试用）


class OpenClawMicAdapter:
    """
    OpenClaw 麦克风适配器
    
    工作流程：
    1. 监听麦克风输入
    2. 检测语音活动（VAD）
    3. 语音结束后，将文本封装成事件
    4. 通过 EventBus 发送给 L1-B
    """
    
    def __init__(
        self,
        event_bus: EventBusClient,
        config: Optional[MicConfig] = None
    ):
        """
        初始化麦克风适配器
        
        Args:
            event_bus: EventBus 客户端
            config: 麦克风配置
        """
        self.event_bus = event_bus
        self.config = config or MicConfig()
        self._running = False
        self._speech_callback: Optional[Callable] = None
        
        logger.info("[OpenClawMicAdapter] 初始化完成")
    
    def set_speech_callback(self, callback: Callable[[str], Any]):
        """
        设置语音回调函数
        
        Args:
            callback: 回调函数，接收语音文本参数
        """
        self._speech_callback = callback
        logger.info("[OpenClawMicAdapter] 语音回调函数已设置")
    
    async def start(self):
        """启动麦克风监听"""
        logger.info("[OpenClawMicAdapter] 启动麦克风监听...")
        self._running = True
        
        if self.config.mock_mode:
            # Mock 模式：模拟语音输入
            asyncio.create_task(self._mock_speech_loop())
            logger.info("[OpenClawMicAdapter] ✅ Mock 模式已启动")
        else:
            # 真实模式：连接麦克风硬件
            asyncio.create_task(self._real_mic_loop())
            logger.info("[OpenClawMicAdapter] ✅ 真实模式已启动")
    
    async def stop(self):
        """停止麦克风监听"""
        logger.info("[OpenClawMicAdapter] 停止麦克风监听...")
        self._running = False
    
    async def _mock_speech_loop(self):
        """
        Mock 语音输入循环（开发测试用）
        
        监听外部触发，不主动发送测试消息
        """
        logger.info("[OpenClawMicAdapter] Mock 语音循环已启动（被动模式）")
        
        # 被动模式：等待外部触发，不主动发送测试消息
        while self._running:
            await asyncio.sleep(1.0)
    
    async def _real_mic_loop(self):
        """
        真实麦克风输入循环
        
        使用 OpenClaw SDK 或 PyAudio 实现真实麦克风监听
        TODO: 实现真实硬件对接
        """
        logger.warning(
            "[OpenClawMicAdapter] ⚠️ 真实模式尚未实现，使用 Mock 模式"
        )
        await self._mock_speech_loop()
    
    async def process_speech(self, text: str):
        """
        处理语音识别结果
        
        Args:
            text: 识别的语音文本
        """
        logger.info(f"[OpenClawMicAdapter] 处理语音：{text}")
        
        # 检测紧急关键词
        emergency_keywords = ["停下", "停止", "救命", "紧急", "危险"]
        is_emergency = any(keyword in text for keyword in emergency_keywords)
        
        # 设置优先级
        priority = (
            OpenClawEventPriority.CRITICAL
            if is_emergency
            else OpenClawEventPriority.NORMAL
        )
        
        # 创建事件
        event = ZulongEvent(
            type=OpenClawEventType.USER_SPEECH,
            source="openclaw/mic",
            payload={"text": text},
            priority=priority
        )
        
        # 发布到 EventBus
        self.event_bus.publish(event)
        
        logger.info(
            f"[OpenClawMicAdapter] ✅ 语音事件已发布（优先级：{priority.value}）"
        )


# 便捷创建函数

def create_mic_adapter(
    event_bus: EventBusClient,
    mock_mode: bool = True
) -> OpenClawMicAdapter:
    """
    创建麦克风适配器
    
    Args:
        event_bus: EventBus 客户端
        mock_mode: 是否使用 Mock 模式
    
    Returns:
        OpenClawMicAdapter: 麦克风适配器实例
    """
    config = MicConfig(mock_mode=mock_mode)
    adapter = OpenClawMicAdapter(event_bus, config)
    return adapter
