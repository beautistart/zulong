# File: openclaw_bridge/adapters/web_adapter.py
"""
OpenClaw Web 适配器

功能：
1. 提供 FastAPI Web 服务器
2. WebSocket 实时通信
3. 将 Web 聊天消息转换为 ZulongEvent
4. 将祖龙响应推送到前端

架构说明：
- Web 页面作为 OpenClaw 的另一个输入源（类似麦克风）
- 通过 EventBus 与祖龙 L1-B 通信
- source 标记为 "openclaw/web_ui"
"""

import asyncio
import logging
from typing import Dict, Set, Optional
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from openclaw_bridge.openclaw_types import ZulongEvent, OpenClawEventType, OpenClawEventPriority
from openclaw_bridge.event_bus_client import EventBusClient

logger = logging.getLogger(__name__)


@dataclass
class WebConfig:
    """Web 配置"""
    host: str = "localhost"
    port: int = 8080
    static_path: str = "openclaw_bridge/web/static"
    enable_cors: bool = True


class OpenClawWebAdapter:
    """
    OpenClaw Web 适配器
    
    工作流程：
    1. 启动 FastAPI 服务器
    2. WebSocket 连接管理
    3. 接收用户消息 -> 发布到 EventBus
    4. 监听 ACTION_SPEAK -> 推送到前端
    """
    
    def __init__(
        self,
        event_bus: EventBusClient,
        config: Optional[WebConfig] = None
    ):
        """
        初始化 Web 适配器
        
        Args:
            event_bus: EventBus 客户端
            config: Web 配置
        """
        self.event_bus = event_bus
        self.config = config or WebConfig()
        self._running = False
        
        # WebSocket 连接管理
        self._active_connections: Set[WebSocket] = set()
        
        # 创建 FastAPI 应用
        self.app = FastAPI(title="OpenClaw Web UI")
        
        # 设置路由
        self._setup_routes()
        
        logger.info(f"[OpenClawWebAdapter] 初始化完成，监听 {self.config.host}:{self.config.port}")
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/")
        async def get_index():
            """返回聊天页面"""
            try:
                with open(f"{self.config.static_path}/index.html", "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read(), status_code=200)
            except FileNotFoundError:
                return HTMLResponse(content="<h1>404 - Page not found</h1>", status_code=404)
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket 端点"""
            await self._handle_websocket(websocket)
        
        logger.info("[OpenClawWebAdapter] 路由已设置")
    
    async def _handle_websocket(self, websocket: WebSocket):
        """
        处理 WebSocket 连接
        
        Args:
            websocket: WebSocket 连接
        """
        # 接受连接
        await websocket.accept()
        self._active_connections.add(websocket)
        
        logger.info(f"[OpenClawWebAdapter] 🌐 新连接，当前连接数：{len(self._active_connections)}")
        
        try:
            # 发送欢迎消息
            await websocket.send_json({
                "type": "WELCOME",
                "message": "已连接到 OpenClaw Bridge",
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # 监听消息
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_message(data, websocket)
                except WebSocketDisconnect:
                    logger.info("[OpenClawWebAdapter] 客户端断开连接")
                    break
                except Exception as e:
                    logger.error(f"[OpenClawWebAdapter] 消息处理错误：{e}")
        
        finally:
            self._active_connections.remove(websocket)
            logger.info(f"[OpenClawWebAdapter] 连接关闭，当前连接数：{len(self._active_connections)}")
    
    async def _handle_message(self, data: dict, websocket: WebSocket):
        """
        处理收到的消息
        
        Args:
            data: 消息数据
            websocket: WebSocket 连接
        """
        message_type = data.get("type")
        
        if message_type == "CHAT_MESSAGE":
            text = data.get("text", "")
            if text:
                logger.info(f"[OpenClawWebAdapter] 📥 收到消息：{text}")
                await self._publish_chat_message(text)
    
    async def _publish_chat_message(self, text: str):
        """
        发布聊天消息到 EventBus
        
        Args:
            text: 聊天文本
        """
        # 创建祖龙事件
        # 🔥 [修复] Web 文本输入应该使用 USER_TEXT，不是 USER_SPEECH
        # 这样 L1-B 会使用 TEXT_ONLY 模式，不会自动语音播报
        event = ZulongEvent(
            type=OpenClawEventType.USER_TEXT,  # 改为 USER_TEXT
            source="openclaw/web_ui",
            payload={
                "text": text,
                "confidence": 1.0  # Web 文本输入，置信度 100%
            },
            priority=OpenClawEventPriority.NORMAL
        )
        
        # 发布到 EventBus
        self.event_bus.publish(event)
        
        logger.info(f"[OpenClawWebAdapter] ✅ 消息已发布到 EventBus (USER_TEXT, confidence=1.0)")
    
    async def broadcast_response(self, text: str):
        """
        广播响应到所有前端
        
        Args:
            text: 响应文本
        """
        if not self._active_connections:
            logger.warning("[OpenClawWebAdapter] ⚠️ 无活动连接，无法广播")
            return
        
        message = {
            "type": "CHAT_RESPONSE",
            "text": text,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # 广播到所有连接
        disconnected = set()
        for connection in self._active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 广播失败：{e}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self._active_connections.remove(connection)
        
        logger.info(f"[OpenClawWebAdapter] 📤 已广播到 {len(self._active_connections)} 个连接")
    
    async def broadcast_streaming_response(self, text: str):
        """
        广播流式响应到所有前端
        
        Args:
            text: 流式响应文本
        """
        if not self._active_connections:
            logger.warning("[OpenClawWebAdapter] ⚠️ 无活动连接，无法广播流式响应")
            return
        
        message = {
            "type": "STREAMING_RESPONSE",
            "text": text,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # 广播到所有连接
        disconnected = set()
        for connection in self._active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 流式广播失败：{e}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            self._active_connections.remove(connection)
        
        logger.debug(f"[OpenClawWebAdapter] 📤 已流式广播到 {len(self._active_connections)} 个连接")
    
    async def start(self):
        """启动 Web 服务器"""
        logger.info("[OpenClawWebAdapter] 启动 Web 服务器...")
        self._running = True
        
        # 配置 uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # 在后台线程中运行
        import threading
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        
        logger.info(f"[OpenClawWebAdapter] ✅ Web 服务器已启动于 http://{self.config.host}:{self.config.port}")
    
    async def stop(self):
        """停止 Web 服务器"""
        logger.info("[OpenClawWebAdapter] 停止 Web 服务器...")
        self._running = False
        
        # 关闭所有连接
        for connection in self._active_connections:
            try:
                await connection.close()
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 关闭连接失败：{e}")
        
        self._active_connections.clear()
        logger.info("[OpenClawWebAdapter] ✅ Web 服务器已停止")
