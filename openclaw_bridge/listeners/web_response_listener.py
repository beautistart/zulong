# File: openclaw_bridge/listeners/web_response_listener.py
"""
OpenClaw Web 响应监听器

功能：
监听祖龙系统的 ACTION_SPEAK 事件
将响应推送到 Web 前端

架构说明：
- 订阅 ACTION_SPEAK 事件
- 通过 WebAdapter 广播到前端
"""

import logging
from typing import Optional, Any

from openclaw_bridge.openclaw_types import ZulongEvent, OpenClawEventType
from openclaw_bridge.event_bus_client import EventBusClient
from openclaw_bridge.adapters.web_adapter import OpenClawWebAdapter

logger = logging.getLogger(__name__)


class WebResponseListener:
    """
    Web 响应监听器
    
    工作流程：
    1. 订阅 ACTION_SPEAK 事件
    2. 提取响应文本
    3. 通过 WebAdapter 推送到前端
    """
    
    def __init__(
        self,
        event_bus: EventBusClient,
        web_adapter: OpenClawWebAdapter
    ):
        """
        初始化 Web 响应监听器
        
        Args:
            event_bus: EventBus 客户端
            web_adapter: Web 适配器
        """
        self.event_bus = event_bus
        self.web_adapter = web_adapter
        
        logger.info("[WebResponseListener] 初始化完成")
    
    def on_speak_event(self, event: ZulongEvent):
        """
        处理 ACTION_SPEAK 事件（同步包装器）
        
        Args:
            event: 祖龙事件
        """
        text = event.payload.get("text", "")
        
        if text:
            logger.info(f"[WebResponseListener] 🔊 收到语音播报：{text}")
            
            # 🔥 [关键修复] 在事件循环中调度异步调用
            import asyncio
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    asyncio.create_task(self.web_adapter.broadcast_response(text))
                    logger.info("[WebResponseListener] ✅ 响应已异步推送到前端")
                else:
                    # 如果事件循环未运行，运行它
                    loop.run_until_complete(self.web_adapter.broadcast_response(text))
                    logger.info("[WebResponseListener] ✅ 响应已推送到前端")
            except RuntimeError:
                # 如果没有事件循环，创建一个新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.web_adapter.broadcast_response(text))
                logger.info("[WebResponseListener] ✅ 响应已推送到前端（新循环）")
        else:
            logger.warning("[WebResponseListener] ⚠️ 空文本，忽略")
    
    def on_l2_output_event(self, event: ZulongEvent):
        """
        处理 L2_OUTPUT 事件（文本模式响应）
        
        Args:
            event: 祖龙事件
        """
        text = event.payload.get("text", "")
        is_streaming = event.payload.get("streaming", False)
        
        if text:
            logger.info(f"[WebResponseListener] 📝 收到 L2 输出：{text[:50]}...")
            
            # 🔥 [关键修复] 在事件循环中调度异步调用
            import asyncio
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    if is_streaming:
                        asyncio.create_task(self.web_adapter.broadcast_streaming_response(text))
                        logger.info("[WebResponseListener] ✅ 流式响应已异步推送到前端")
                    else:
                        asyncio.create_task(self.web_adapter.broadcast_response(text))
                        logger.info("[WebResponseListener] ✅ 响应已异步推送到前端")
                else:
                    # 如果事件循环未运行，运行它
                    if is_streaming:
                        loop.run_until_complete(self.web_adapter.broadcast_streaming_response(text))
                        logger.info("[WebResponseListener] ✅ 流式响应已推送到前端")
                    else:
                        loop.run_until_complete(self.web_adapter.broadcast_response(text))
                        logger.info("[WebResponseListener] ✅ 响应已推送到前端")
            except RuntimeError:
                # 如果没有事件循环，创建一个新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                if is_streaming:
                    loop.run_until_complete(self.web_adapter.broadcast_streaming_response(text))
                    logger.info("[WebResponseListener] ✅ 流式响应已推送到前端（新循环）")
                else:
                    loop.run_until_complete(self.web_adapter.broadcast_response(text))
                    logger.info("[WebResponseListener] ✅ 响应已推送到前端（新循环）")
        else:
            logger.warning("[WebResponseListener] ⚠️ 空文本，忽略")
    
    async def on_speak_event_async(self, event: ZulongEvent):
        """
        处理 ACTION_SPEAK 事件（异步版本）
        
        Args:
            event: 祖龙事件
        """
        text = event.payload.get("text", "")
        
        if text:
            logger.info(f"[WebResponseListener] 🔊 收到语音播报：{text}")
            
            # 推送到前端
            await self.web_adapter.broadcast_response(text)
            
            logger.info("[WebResponseListener] ✅ 响应已推送到前端")
        else:
            logger.warning("[WebResponseListener] ⚠️ 空文本，忽略")
