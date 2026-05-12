from __future__ import annotations

import asyncio
import enum
import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional

import websockets

logger = logging.getLogger(__name__)


class TestEvent(str, enum.Enum):
    TEST_STARTED = "TEST_STARTED"
    STEP_STARTED = "STEP_STARTED"
    STEP_COMPLETED = "STEP_COMPLETED"
    STEP_FAILED = "STEP_FAILED"
    PROGRESS_UPDATE = "PROGRESS_UPDATE"
    STAGNATION_ALERT = "STAGNATION_ALERT"
    INTERRUPT_TRIGGERED = "INTERRUPT_TRIGGERED"
    RESUME_STARTED = "RESUME_STARTED"
    RESUME_COMPLETED = "RESUME_COMPLETED"
    ATTENTION_SWITCH = "ATTENTION_SWITCH"
    TEST_COMPLETED = "TEST_COMPLETED"


class EventBridge:
    def __init__(self, backend_proxy=None):
        self._backend_proxy = backend_proxy
        self._subscribers: list[Callable[[TestEvent, dict], Any]] = []

    def subscribe(self, handler: Callable[[TestEvent, dict], Any]) -> None:
        self._subscribers.append(handler)

    def unsubscribe(self, handler: Callable[[TestEvent, dict], Any]) -> None:
        self._subscribers = [h for h in self._subscribers if h != handler]

    async def emit(self, event: TestEvent, data: dict) -> None:
        message = {
            "type": event.value,
            "payload": data,
            "timestamp": datetime.now().isoformat(),
        }
        for handler in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event, data)
                else:
                    handler(event, data)
            except Exception as e:
                logger.error("事件处理器异常: %s", e)

    async def listen_backend_monitor(self, monitor_uri: str = "ws://127.0.0.1:8090/monitor") -> None:
        try:
            async with websockets.connect(monitor_uri, ping_interval=30) as ws:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        await self._handle_backend_event(msg)
                    except json.JSONDecodeError:
                        logger.warning("收到非法JSON消息")
        except Exception as e:
            logger.error("Monitor连接异常: %s", e)

    async def _handle_backend_event(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        if msg_type == "PROGRESS_REPORT":
            await self.emit(TestEvent.PROGRESS_UPDATE, msg.get("data", {}))
        elif msg_type == "TASK_GRAPH_UPDATE":
            await self.emit(TestEvent.PROGRESS_UPDATE, msg.get("data", {}))
        elif msg_type == "display_text":
            pass
        elif msg_type == "task_complete":
            await self.emit(TestEvent.TEST_COMPLETED, msg.get("data", {}))


class TestMonitorWSService:
    def __init__(self, event_bridge: EventBridge):
        self._bridge = event_bridge
        self._connections: list[websockets.WebServerProtocol] = []
        self._subscriptions: dict[str, set[str]] = {}
        self._server: Optional[websockets.WebSocketServer] = None

    async def start(self, host: str = "127.0.0.1", port: int = 8091) -> None:
        self._server = await websockets.serve(self._handler, host, port)
        self._bridge.subscribe(self._on_event)
        logger.info("TestMonitor WebSocket服务启动: ws://%s:%d", host, port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        for ws in self._connections:
            await ws.close()
        self._connections.clear()

    async def _handler(self, ws: websockets.WebServerProtocol, path: str) -> None:
        self._connections.append(ws)
        try:
            welcome = {
                "type": "TEST_WELCOME",
                "payload": {"active_executions": [], "test_cases": []},
                "timestamp": datetime.now().isoformat(),
            }
            await ws.send(json.dumps(welcome, ensure_ascii=False))

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_client_message(ws, msg)
                except json.JSONDecodeError:
                    pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self._connections.remove(ws)

    async def _handle_client_message(self, ws, msg: dict) -> None:
        msg_type = msg.get("type", "")
        if msg_type == "SUBSCRIBE":
            execution_id = msg.get("execution_id", "")
            ws_id = id(ws)
            if ws_id not in self._subscriptions:
                self._subscriptions[ws_id] = set()
            self._subscriptions[ws_id].add(execution_id)
        elif msg_type == "UNSUBSCRIBE":
            execution_id = msg.get("execution_id", "")
            ws_id = id(ws)
            if ws_id in self._subscriptions:
                self._subscriptions[ws_id].discard(execution_id)
        elif msg_type == "ping":
            await ws.send(json.dumps({"type": "pong"}))

    async def _on_event(self, event: TestEvent, data: dict) -> None:
        message = {
            "type": event.value,
            "payload": data,
            "timestamp": datetime.now().isoformat(),
        }
        raw = json.dumps(message, ensure_ascii=False, default=str)
        disconnected = []
        for ws in self._connections:
            try:
                await ws.send(raw)
            except websockets.ConnectionClosed:
                disconnected.append(ws)
        for ws in disconnected:
            self._connections.remove(ws)
