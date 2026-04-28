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
        
        从 EventBusClient 的监听线程调用，
        使用 schedule_coroutine 线程安全地调度到 uvicorn 事件循环。
        
        Args:
            event: 祖龙事件
        """
        text = event.payload.get("text", "")
        
        if text:
            logger.info(f"[WebResponseListener] 收到语音播报：{text}")
            self.web_adapter.schedule_coroutine(
                self.web_adapter.broadcast_response(text)
            )
            logger.info("[WebResponseListener] 响应已调度推送到前端")
        else:
            logger.warning("[WebResponseListener] 空文本，忽略")
    
    def on_l2_output_event(self, event: ZulongEvent):
        """
        处理 L2_OUTPUT 事件（文本模式响应）
        
        从 EventBusClient 的监听线程调用，
        使用 schedule_coroutine 线程安全地调度到 uvicorn 事件循环。
        
        Args:
            event: 祖龙事件
        """
        text = event.payload.get("text", "")
        is_streaming = event.payload.get("streaming", False)
        session_id = event.payload.get("session_id")
        request_id = event.payload.get("request_id")
        
        if text:
            logger.info(f"[WebResponseListener] 收到 L2 输出：{text[:100]}...")
            
            # 保存 AI 响应到测试会话
            if session_id and hasattr(self.web_adapter, '_test_sessions'):
                self._save_ai_response(session_id, text)
            
            # 线程安全地调度到 uvicorn 事件循环
            if is_streaming:
                self.web_adapter.schedule_coroutine(
                    self.web_adapter.broadcast_streaming_response(text)
                )
            else:
                self.web_adapter.schedule_coroutine(
                    self.web_adapter.broadcast_response(text, request_id)
                )
            logger.info("[WebResponseListener] L2 输出已调度推送到前端")
        else:
            logger.warning("[WebResponseListener] 空文本，忽略")
    
    def on_thinking_step_event(self, event: ZulongEvent):
        """
        处理 L2_THINKING_STEP 事件（推理过程步骤实时推送）
        
        从 EventBusClient 的监听线程调用，
        使用 schedule_coroutine 线程安全地调度到 uvicorn 事件循环。
        
        Args:
            event: 祖龙事件
        """
        payload = event.payload
        if not payload:
            return
        
        self.web_adapter.schedule_coroutine(
            self.web_adapter.broadcast_thinking_step(payload)
        )
    
    def on_stream_event(self, event: ZulongEvent):
        """
        处理 L2_OUTPUT_STREAM 事件（流式输出实时推送）
        
        从 EventBusClient 的监听线程调用，
        使用 schedule_coroutine 线程安全地调度到 uvicorn 事件循环。
        
        Args:
            event: 祖龙事件
        """
        text = event.payload.get("text", "")
        chunk = event.payload.get("chunk", "")
        request_id = event.payload.get("request_id")
        
        if not chunk and not text:
            return
        
        self.web_adapter.schedule_coroutine(
            self.web_adapter.broadcast_streaming_response(text, chunk, request_id)
        )
    
    def _save_ai_response(self, session_id: str, text: str):
        """
        保存 AI 响应到测试会话
        
        Args:
            session_id: 会话 ID
            text: AI 响应文本
        """
        if session_id not in self.web_adapter._test_sessions:
            logger.debug(f"[WebResponseListener] ⚠️ 会话 {session_id} 不存在，跳过保存")
            return
        
        # 添加 AI 响应到会话
        self.web_adapter._test_sessions[session_id].append({
            "role": "assistant",
            "content": text,
            "timestamp": __import__('time').time(),
            "is_ai_response": True
        })
        
        # 保存到磁盘
        self.web_adapter._save_session(session_id)
        
        logger.debug(f"💾 [WebResponseListener] AI 响应已保存到会话 {session_id}")
    
    def on_memory_graph_update_event(self, event: ZulongEvent):
        """
        处理 MEMORY_GRAPH_UPDATED 事件
        
        从 EventBusClient 的监听线程调用，
        使用 schedule_coroutine 线程安全地调度到 uvicorn 事件循环。
        
        Args:
            event: 祖龙事件
        """
        payload = event.payload
        if not payload:
            return
        
        self.web_adapter.schedule_coroutine(
            self.web_adapter.broadcast_memory_graph_update(payload)
        )
    
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
