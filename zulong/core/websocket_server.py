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
import socket
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
                # 转发到所有远程订阅者（高频事件不打印日志，避免干扰显示）
                _silent = event_type_str in ("L2_OUTPUT", "L2_OUTPUT_STREAM", "L2_THINKING_STEP")
                if not _silent:
                    logger.debug(f"[EventBusBridge] 收到本地事件：{event_type_str}")
                if event_type_str in self._subscriptions:
                    if not _silent:
                        logger.debug(f"[EventBusBridge] 找到 {len(self._subscriptions[event_type_str])} 个订阅者")
                    for callback in self._subscriptions[event_type_str]:
                        try:
                            if not _silent:
                                logger.debug(f"[EventBusBridge] 正在调用回调：{callback}")
                            callback(event)
                        except Exception as e:
                            logger.error(f"转发事件失败：{e}")
                else:
                    if not _silent:
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
            # 🔥 [BUG-9 修复] 正确解析优先级字符串
            # EventPriority 枚举的值是整数 (NORMAL=1)，不能用字符串 "NORMAL" 构造
            priority_raw = event_data.get("priority", "NORMAL")
            if isinstance(priority_raw, str):
                # 按名称查找枚举成员
                priority = EventPriority[priority_raw] if priority_raw in EventPriority.__members__ else EventPriority.NORMAL
            else:
                priority = EventPriority(priority_raw)
            
            event = ZulongEvent(
                type=EventType(event_data["type"]),
                priority=priority,
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
            "L2_OUTPUT_STREAM",
            "L2_THINKING_STEP",
            "SYSTEM_STATUS",
            "MEMORY_GRAPH_UPDATED",
        ])
        
        logger.info(f"[WebSocketServer] 初始化完成，监听 {host}:{port}")
    
    async def start(self):
        """启动 WebSocket 服务器"""
        self._running = True
        self._event_loop = asyncio.get_running_loop()  # 🔥 保存当前 event loop
        
        # 🔥 [BUG-8 修复] 强制使用 IPv4 避免 Windows 双栈绑定冲突
        # Windows 上 "localhost" 会同时解析为 127.0.0.1 和 ::1，
        # 如果 IPv6 端口被占用会导致整个 serve() 失败
        bind_host = "127.0.0.1" if self.host == "localhost" else self.host
        
        async with websockets.serve(
            self._handle_client,
            bind_host,
            self.port,
            process_request=self._process_request,
            ping_interval=None,  # 禁用 ping/pong
            ping_timeout=None,   # 禁用超时
            max_size=None,       # 允许大消息
            compression=None,    # 禁用压缩
            family=socket.AF_INET,  # 🔥 [BUG-8] 强制 IPv4，避免双栈绑定冲突
        ):
            logger.info(f"[WebSocketServer] 已启动于 ws://{bind_host}:{self.port} (IPv4)")
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
            # 🔥 [BUG-7 修复] 使用非阻塞方式转发事件
            # 原来的 future.result(timeout=1.0) 会阻塞 EventBus 分发线程，
            # 如果 ws 事件循环忙碌（如正在处理 asyncio.create_task），
            # 超时后事件会丢失，且阻塞后续事件的分发
            if self._event_loop and not self._event_loop.is_closed():
                try:
                    logger.info(f"[WebSocketServer] 正在转发事件 {event.type} 到远程客户端")
                    future = asyncio.run_coroutine_threadsafe(
                        self._send_event_to_client(websocket, event), 
                        self._event_loop
                    )
                    # 非阻塞：使用回调捕获错误，不阻塞分发线程
                    def _on_forward_done(f):
                        try:
                            f.result()
                        except Exception as e:
                            logger.error(f"[WebSocketServer] 转发事件异步完成失败：{e}")
                    future.add_done_callback(_on_forward_done)
                except Exception as e:
                    logger.error(f"[WebSocketServer] run_coroutine_threadsafe 失败：{e}")
            else:
                logger.error(f"[WebSocketServer] event loop 不可用 (loop={self._event_loop}, closed={self._event_loop.is_closed() if self._event_loop else 'N/A'})")
        
        # 订阅所有需要转发的事件
        self._bridge.subscribe_remote("ACTION_SPEAK", forward_to_remote)
        self._bridge.subscribe_remote("L2_OUTPUT", forward_to_remote)
        self._bridge.subscribe_remote("L2_OUTPUT_STREAM", forward_to_remote)
        self._bridge.subscribe_remote("L2_THINKING_STEP", forward_to_remote)
        self._bridge.subscribe_remote("SYSTEM_STATUS", forward_to_remote)
        self._bridge.subscribe_remote("MEMORY_GRAPH_UPDATED", forward_to_remote)
        
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
            self._bridge.unsubscribe_remote("L2_OUTPUT_STREAM", forward_to_remote)
            self._bridge.unsubscribe_remote("L2_THINKING_STEP", forward_to_remote)
            self._bridge.unsubscribe_remote("SYSTEM_STATUS", forward_to_remote)
            self._bridge.unsubscribe_remote("MEMORY_GRAPH_UPDATED", forward_to_remote)
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
        
        # ✅ 特殊处理 L2_COMMAND：如果启用 streaming，使用流式输出
        if event_type == "L2_COMMAND":
            enable_streaming = payload.get("enable_streaming", True)
            
            # 检查配置是否启用 streaming
            try:
                from zulong.config.config_manager import get_l2_inference_config
                config = get_l2_inference_config()
                orch_config = config.get("orchestrator", {})
                streaming_enabled = orch_config.get("enable_streaming", False)
            except Exception as e:
                logger.warning(f"[WebSocketServer] 读取 streaming 配置失败: {e}，默认禁用")
                streaming_enabled = False
            
            if streaming_enabled and enable_streaming:
                logger.info("[WebSocketServer] ✅ 使用流式模式处理 L2_COMMAND")
                await self._handle_l2_command_streaming(payload, websocket)
                return
        
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
    
    async def _handle_l2_command_streaming(
        self, 
        payload: dict, 
        websocket: websockets.WebSocketServerProtocol
    ):
        """处理 L2_COMMAND 流式输出
        
        Args:
            payload: 命令载荷
            websocket: WebSocket 连接
        """
        try:
            # 提取命令参数
            text = payload.get("text", "")
            vllm_model_id = payload.get("vllm_model_id", "qwen3.5:4b")
            
            logger.info(f"[WebSocketServer] 开始流式处理 L2 命令: '{text[:50]}...'")
            
            # 发送开始事件
            await websocket.send(json.dumps({
                "type": "L2_STREAM_START",
                "timestamp": datetime.now().isoformat(),
                "message": "Streaming started"
            }))
            
            # 获取 InferenceEngine 实例
            from zulong.l2.inference_engine import InferenceEngine
            engine = InferenceEngine()
            
            # 检查是否启用 LangGraph Orchestrator
            from zulong.config.config_manager import get_l2_inference_config
            config = get_l2_inference_config()
            orch_config = config.get("orchestrator", {})
            use_langgraph = orch_config.get("use_langgraph", False)
            
            if not use_langgraph:
                logger.warning("[WebSocketServer] LangGraph 未启用，降级为非流式模式")
                await websocket.send(json.dumps({
                    "type": "L2_STREAM_ERROR",
                    "error": "LangGraph not enabled",
                    "message": "Streaming requires LangGraph to be enabled"
                }))
                return
            
            # 创建或复用编排器实例
            if not hasattr(engine, '_orchestrator_langgraph'):
                from zulong.l2.orchestrator_graph import OrchestratorWithLangGraph
                engine._orchestrator_langgraph = OrchestratorWithLangGraph(engine)
            
            orchestrator = engine._orchestrator_langgraph
            
            # 准备消息和工具定义
            messages = [
                {"role": "user", "content": text}
            ]
            tool_definitions = []  # TODO: 从引擎获取实际的工具定义
            
            # ✅ 使用 stream_run 方法进行流式执行
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async for event in orchestrator.stream_run(
                    user_input=text,
                    messages=messages,
                    tool_definitions=tool_definitions,
                    vllm_model_id=vllm_model_id,
                ):
                    # 转发每个流式事件到 WebSocket 客户端
                    await websocket.send(json.dumps(event))
                    
            finally:
                loop.close()
            
            # 发送结束事件
            await websocket.send(json.dumps({
                "type": "L2_STREAM_END",
                "timestamp": datetime.now().isoformat(),
                "message": "Streaming completed"
            }))
            
            logger.info("[WebSocketServer] L2 命令流式处理完成")
            
        except Exception as e:
            logger.error(f"[WebSocketServer] 流式处理失败: {e}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "L2_STREAM_ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
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
