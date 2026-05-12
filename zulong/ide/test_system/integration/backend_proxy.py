from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional

import websockets

logger = logging.getLogger(__name__)


class BackendConnection:
    def __init__(self, uri: str = "ws://127.0.0.1:8090/ide"):
        self.uri = uri
        self._ws: Optional[websockets.WebClientProtocol] = None
        self._recv_handlers: dict[str, Callable] = {}
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect = 3
        self._recv_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        for attempt in range(self._max_reconnect):
            try:
                self._ws = await websockets.connect(self.uri, ping_interval=30, ping_timeout=10)
                self._connected = True
                self._reconnect_attempts = 0
                self._recv_task = asyncio.create_task(self._recv_loop())
                logger.info("后端WebSocket连接成功: %s", self.uri)
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning("连接失败(第%d次), %ds后重试: %s", attempt + 1, wait, e)
                await asyncio.sleep(wait)
        raise ConnectionError(f"后端WebSocket连接失败(已重试{self._max_reconnect}次)")

    async def disconnect(self) -> None:
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send(self, message: dict) -> None:
        if not self._ws or not self._connected:
            raise ConnectionError("WebSocket未连接")
        await self._ws.send(json.dumps(message, ensure_ascii=False))

    async def send_session_start(self, session_config: dict) -> None:
        await self.send({"type": "session_start", **session_config})

    async def send_tool_result(self, tool_use_id: str, content: Any) -> None:
        await self.send({"type": "tool_result", "tool_use_id": tool_use_id, "content": content})

    async def send_cancel(self) -> None:
        await self.send({"type": "user_cancel"})

    async def send_session_resume(self, snapshot_data: dict) -> None:
        await self.send({"type": "session_resume", "snapshot": snapshot_data})

    async def send_ping(self) -> None:
        await self.send({"type": "ping"})

    def on_message(self, msg_type: str, handler: Callable) -> None:
        self._recv_handlers[msg_type] = handler

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    msg_type = msg.get("type", "")
                    handler = self._recv_handlers.get(msg_type)
                    if handler:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(msg)
                        else:
                            handler(msg)
                except json.JSONDecodeError:
                    logger.warning("收到非法JSON消息")
        except websockets.ConnectionClosed:
            logger.info("后端WebSocket连接关闭")
            self._connected = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("接收循环异常: %s", e)
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected


class BackendProxy:
    def __init__(self, uri: str = "ws://127.0.0.1:8090/ide"):
        self._conn = BackendConnection(uri)
        self._pending_requests: dict[str, asyncio.Future] = {}

    async def start_session(self, config: dict) -> dict:
        await self._conn.send_session_start(config)
        return await self._wait_response("session_ack", timeout=10)

    async def execute_tool(self, tool_name: str, tool_input: dict, timeout: int = 300) -> dict:
        request_id = f"req_{int(time.time() * 1000)}"
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future
        await self._conn.send({"type": "tool_call", "request_id": request_id, "tool": tool_name, "input": tool_input})
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(f"工具调用超时: {tool_name}")

    async def cancel(self) -> None:
        await self._conn.send_cancel()

    async def resume(self, snapshot: dict) -> dict:
        await self._conn.send_session_resume(snapshot)
        return await self._wait_response("session_ack", timeout=10)

    async def connect(self) -> None:
        await self._conn.connect()

    async def disconnect(self) -> None:
        await self._conn.disconnect()

    def on_event(self, event_type: str, handler: Callable) -> None:
        self._conn.on_message(event_type, handler)

    @property
    def connected(self) -> bool:
        return self._conn.connected

    async def _wait_response(self, expected_type: str, timeout: int = 10) -> dict:
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def handler(msg: dict):
            if not future.done():
                future.set_result(msg)

        self._conn.on_message(expected_type, handler)
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            pass
