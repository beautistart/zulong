# File: zulong/core/websocket_server.py
"""
WebSocket 服务器 - 用于 OpenClaw Bridge 连接

实现基于 websockets 库的 WebSocket 服务器
支持：
1. OpenClaw Bridge 连接
2. 事件发布/订阅
3. 多客户端支持
4. EventBus 远程桥接（/eventbus 端点）
"""

import asyncio
import websockets
import json
import logging
from typing import Set, Optional, Dict, Callable, List
from datetime import datetime

from zulong.core.event_bus import event_bus
from zulong.core.types import ZulongEvent, EventType, EventPriority

logger = logging.getLogger(__name__)


class EventBusBridge:
    """
    EventBus 远程桥接器
    
    功能：
    1. 订阅祖龙本地 EventBus 事件，转发到远程客户端
    2. 接收远程客户端事件，发布到祖龙本地 EventBus
    """
    
    def __init__(self):
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._local_subscribers = {}  # 保存本地订阅引用
    
    def subscribe_remote(self, event_type: str, callback: Callable):
        """订阅远程事件"""
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = []
        self._subscriptions[event_type].append(callback)
    
    def unsubscribe_remote(self, event_type: str, callback: Callable):
        """取消订阅远程事件"""
        if event_type in self._subscriptions:
            try:
                self._subscriptions[event_type].remove(callback)
                logger.debug(f"[EventBusBridge] 已取消订阅：{event_type}")
            except ValueError:
                logger.warning(f"取消订阅失败：{event_type} 没有找到回调")
    
    def attach_to_local_bus(self, event_types: list):
        """
        附加到本地 EventBus
        
        Args:
            event_types: 需要转发的本地事件类型列表
        """
        def create_handler(event_type_str: str):
            def handler(event: ZulongEvent):
                # 转发到所有远程订阅者（L2_OUTPUT 事件不打印日志，避免干扰显示）
                if event_type_str != "L2_OUTPUT":
                    logger.debug(f"[EventBusBridge] 收到本地事件：{event_type_str}")
                if event_type_str in self._subscriptions:
                    if event_type_str != "L2_OUTPUT":
                        logger.debug(f"[EventBusBridge] 找到 {len(self._subscriptions[event_type_str])} 个订阅者")
                    for callback in self._subscriptions[event_type_str]:
                        try:
                            if event_type_str != "L2_OUTPUT":
                                logger.debug(f"[EventBusBridge] 正在调用回调：{callback}")
                            callback(event)
                        except Exception as e:
                            logger.error(f"转发事件失败：{e}")
                else:
                    if event_type_str != "L2_OUTPUT":
                        logger.debug(f"[EventBusBridge] 没有订阅者：{event_type_str}")
            return handler
        
        # 订阅本地事件
        for event_type_str in event_types:
            try:
                event_type = EventType(event_type_str)
                handler = create_handler(event_type_str)
                self._local_subscribers[event_type_str] = handler
                event_bus.subscribe(event_type, handler, "EventBusBridge")
                logger.info(f"[EventBusBridge] 已订阅本地事件：{event_type_str}")
            except ValueError:
                logger.warning(f"无效的事件类型：{event_type_str}")
    
    def publish_to_local(self, event_data: dict):
        """
        发布远程事件到本地 EventBus
        
        Args:
            event_data: 事件数据
        """
        try:
            # 🔥 [关键修复] 去掉 id 和 timestamp 字段，ZulongEvent 不需要这两个参数
            event = ZulongEvent(
                type=EventType(event_data["type"]),
                priority=EventPriority(event_data.get("priority", "NORMAL")),
                source=event_data.get("source", "RemoteClient"),
                payload=event_data.get("payload", {}),
            )
            event_bus.publish(event)
            logger.info(f"[EventBusBridge] 远程事件已发布到本地：{event.type}")
        except Exception as e:
            logger.error(f"发布远程事件失败：{e}")


class WebSocketServer:
    """WebSocket 服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 5555):
        """初始化 WebSocket 服务器"""
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.eventbus_clients: Set[websockets.WebSocketServerProtocol] = set()  # EventBus 专用客户端
        self.server: Optional[websockets.WebSocketServerProtocol] = None
        self._running = False
        self._bridge = EventBusBridge()
        self._event_loop = None  # 🔥 保存 event loop 引用
        
        # 附加到本地 EventBus，转发需要发送到远程的事件
        self._bridge.attach_to_local_bus([
            "ACTION_SPEAK",
            "L2_OUTPUT",
            "SYSTEM_STATUS"
        ])
        
        logger.info(f"[WebSocketServer] 初始化完成，监听 {host}:{port}")
    
    async def start(self):
        """启动 WebSocket 服务器"""
        self._running = True
        self._event_loop = asyncio.get_running_loop()  # 🔥 保存当前 event loop
        
        # 创建多路径处理器
        # 🔥 [关键修复] 添加 ws_server_kwargs 以支持同步客户端连接
        # 设置 ping_interval 和 ping_timeout 为 None 禁用 ping/pong
        # 设置 max_size=None 允许大消息
        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            process_request=self._process_request,
            ping_interval=None,  # 禁用 ping/pong
            ping_timeout=None,   # 禁用超时
            max_size=None,       # 允许大消息
            compression=None,    # 禁用压缩
        ):
            logger.info(f"[WebSocketServer] 已启动于 ws://{self.host}:{self.port}")
            logger.info(f"[WebSocketServer] 普通端点：/ (默认)")
            logger.info(f"[WebSocketServer] EventBus 端点：/eventbus")
            await asyncio.Future()  # 永远运行
    
    def _process_request(self, path: str, headers):
        """
        处理 HTTP 请求（用于路由到不同的 WebSocket 处理器）
        
        Args:
            path: 请求路径
            headers: HTTP 头
        """
        # 返回 None 让 websockets 库处理 WebSocket 握手
        # 我们在 _handle_client 中根据 path 区分不同的处理器
        return None
    
    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """
        处理客户端连接
        
        Args:
            websocket: WebSocket 连接
        """
        # 🔥 [关键修复] 从 websocket.request.path 获取路径
        # websockets 库新版本 API 变更
        try:
            # 尝试新 API (websockets 12+)
            path = websocket.request.path if hasattr(websocket, 'request') else websocket.path
        except AttributeError:
            # 降级到旧 API
            path = "/"
        
        logger.info(f"[WebSocketServer] 新连接，路径：{path}")
        
        if path == "/eventbus":
            await self._handle_eventbus_client(websocket)
        else:
            await self._handle_normal_client(websocket)
    
    async def _handle_eventbus_client(self, websocket: websockets.WebSocketServerProtocol):
        """
        处理 EventBus 客户端连接（双向事件流）
        
        功能：
        1. 接收远程事件，发布到本地 EventBus
        2. 订阅本地事件，转发到远程客户端
        """
        self.eventbus_clients.add(websocket)
        logger.info(f"[WebSocketServer] EventBus 客户端已连接，当前连接数：{len(self.eventbus_clients)}")
        
        # 注册转发函数
        def forward_to_remote(event: ZulongEvent):
            """转发本地事件到远程客户端"""
            # 🔥 [关键修复] 使用保存的 event loop 运行协程
            if self._event_loop:
                try:
                    # 🔥 添加详细日志
                    logger.info(f"[WebSocketServer] 正在转发事件 {event.type} 到远程客户端")
                    future = asyncio.run_coroutine_threadsafe(self._send_event_to_client(websocket, event), self._event_loop)
                    # 等待完成以捕获错误
                    try:
                        future.result(timeout=1.0)  # 等待最多 1 秒
                    except asyncio.TimeoutError:
                        logger.error("转发事件超时")
                    except Exception as e:
                        logger.error(f"转发事件失败：{e}")
                except Exception as e:
                    logger.error(f"run_coroutine_threadsafe 失败：{e}")
            else:
                logger.error("没有可用的 event loop，无法转发事件")
        
        # 订阅所有需要转发的事件
        self._bridge.subscribe_remote("ACTION_SPEAK", forward_to_remote)
        self._bridge.subscribe_remote("L2_OUTPUT", forward_to_remote)
        self._bridge.subscribe_remote("SYSTEM_STATUS", forward_to_remote)
        
        logger.info(f"[WebSocketServer] 已注册转发函数到 EventBusBridge")
        logger.info(f"[WebSocketServer] EventBusBridge 订阅状态：{list(self._bridge._subscriptions.keys())}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_eventbus_message(data, websocket)
                except json.JSONDecodeError:
                    logger.error(f"无效的 JSON: {message}")
                except Exception as e:
                    logger.error(f"处理 EventBus 消息失败：{e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("EventBus 客户端断开连接")
        finally:
            self.eventbus_clients.remove(websocket)
            # 🔥 [关键修复] 取消订阅，避免内存泄漏和重复转发
            logger.info(f"[WebSocketServer] 取消 EventBusBridge 订阅")
            self._bridge.unsubscribe_remote("ACTION_SPEAK", forward_to_remote)
            self._bridge.unsubscribe_remote("L2_OUTPUT", forward_to_remote)
            self._bridge.unsubscribe_remote("SYSTEM_STATUS", forward_to_remote)
            logger.info(f"EventBus 客户端已断开，当前连接数：{len(self.eventbus_clients)}")
    
    async def _process_eventbus_message(self, data: dict, websocket: websockets.WebSocketServerProtocol):
        """
        处理 EventBus 消息
        
        Args:
            data: 消息数据
            websocket: WebSocket 连接
        """
        message_type = data.get("type")
        
        if message_type == "PUBLISH":
            # 发布远程事件到本地 EventBus
            event_data = data.get("event", {})
            self._bridge.publish_to_local(event_data)
            
            # 发送确认
            await websocket.send(json.dumps({
                "type": "ACK",
                "event_type": event_data.get("type"),
                "timestamp": datetime.now().isoformat(),
                "message": "Event published successfully"
            }))
        
        elif message_type == "SUBSCRIBE":
            # 订阅本地事件（已经在连接时自动订阅）
            event_types = data.get("event_types", [])
            logger.info(f"远程客户端订阅事件：{event_types}")
        
        else:
            logger.warning(f"未知的 EventBus 消息类型：{message_type}")
            await websocket.send(json.dumps({
                "type": "ERROR",
                "message": f"Unknown message type: {message_type}"
            }))
    
    async def _send_event_to_client(self, websocket: websockets.WebSocketServerProtocol, event: ZulongEvent):
        """
        发送事件到远程客户端
        
        Args:
            websocket: WebSocket 连接
            event: 祖龙事件
        """
        try:
            # 🔥 [关键修复] 去掉 id 和 timestamp 字段，ZulongEvent 没有这两个属性
            message = {
                "type": "SUBSCRIBE",
                "event": {
                    "type": event.type.value,
                    "priority": event.priority.value,
                    "source": event.source,
                    "payload": event.payload,
                }
            }
            logger.info(f"[WebSocketServer] 准备发送事件到远程：{event.type}")
            logger.info(f"[WebSocketServer] 发送消息：{message}")
            await websocket.send(json.dumps(message))
            logger.info(f"[WebSocketServer] ✅ 已转发事件 {event.type} 到远程客户端")
        except Exception as e:
            logger.error(f"转发事件到远程失败：{e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def _handle_normal_client(self, websocket: websockets.WebSocketServerProtocol):
        """处理普通客户端连接（兼容旧版本）"""
        self.clients.add(websocket)
        logger.info(f"[WebSocketServer] 普通客户端已连接，当前连接数：{len(self.clients)}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_message(data, websocket)
                except json.JSONDecodeError:
                    logger.error(f"无效的 JSON: {message}")
                except Exception as e:
                    logger.error(f"处理消息失败：{e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("普通客户端断开连接")
        finally:
            self.clients.remove(websocket)
            logger.info(f"普通客户端已断开，当前连接数：{len(self.clients)}")
    
    async def _process_message(self, data: dict, websocket: websockets.WebSocketServerProtocol):
        """处理收到的消息"""
        event_type = data.get("type")
        payload = data.get("payload", {})
        source = data.get("source", "WebSocket")
        
        logger.info(f"[WebSocketServer] 收到事件：{event_type} from {source}")
        
        # 映射事件类型
        try:
            zulong_event_type = EventType(event_type)
        except ValueError:
            logger.warning(f"未知的事件类型：{event_type}")
            await websocket.send(json.dumps({
                "type": "ERROR",
                "message": f"Unknown event type: {event_type}"
            }))
            return
        
        # 创建祖龙事件
        event = ZulongEvent(
            type=zulong_event_type,
            priority=EventPriority.NORMAL,
            source=source,
            payload=payload
        )
        
        # 发布到 EventBus
        event_bus.publish(event)
        logger.info(f"[WebSocketServer] 事件已发布到 EventBus: {event_type}")
        
        # 发送确认
        await websocket.send(json.dumps({
            "type": "ACK",
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "message": "Event published successfully"
        }))
    
    async def stop(self):
        """停止 WebSocket 服务器"""
        self._running = False
        
        # 关闭所有客户端连接
        for client in self.clients:
            await client.close()
        
        self.clients.clear()
        logger.info("[WebSocketServer] 已停止")


# 全局 WebSocket 服务器实例
websocket_server: Optional[WebSocketServer] = None


def get_websocket_server() -> WebSocketServer:
    """获取全局 WebSocket 服务器实例"""
    global websocket_server
    if websocket_server is None:
        websocket_server = WebSocketServer()
    return websocket_server


async def start_websocket_server(host: str = "localhost", port: int = 5555):
    """启动 WebSocket 服务器的便捷函数"""
    server = get_websocket_server()
    await server.start()


if __name__ == "__main__":
    # 测试 WebSocket 服务器
    logging.basicConfig(level=logging.INFO)
    
    try:
        asyncio.run(start_websocket_server())
    except KeyboardInterrupt:
        logger.info("WebSocket 服务器已停止")
