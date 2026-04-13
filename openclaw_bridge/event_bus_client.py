# File: openclaw_bridge\event_bus_client.py
"""
OpenClaw EventBus 客户端

实现与祖龙系统 EventBuses 的连接和通信
架构说明：
- OpenClaw 作为事件收发端，通过 WebSocket 连接到祖龙 EventBus
- 所有事件统一路由到 L1-B，由 L1-B Gatekeeper 负责分发
- 支持发布/订阅模式
- 🔥 使用 WebSocket 远程连接到祖龙主系统（共享记忆）

对应 TSD v2.2 第 3.1 节（事件模型）、第 4.1 节（L1-B 路由逻辑）
"""

import asyncio
import threading
import time
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
import websocket  # WebSocket 客户端库
import json
from datetime import datetime

from openclaw_bridge.openclaw_types import ZulongEvent, OpenClawEventType, OpenClawEventPriority

logger = logging.getLogger(__name__)


@dataclass
class EventBusConfig:
    """EventBus 客户端配置"""
    host: str = "localhost"  # 祖龙系统主机
    port: int = 5555  # EventBus 端口
    client_name: str = "OpenClaw_Bridge"  # 客户端名称
    reconnect_interval: float = 5.0  # 重连间隔（秒）
    max_reconnect_attempts: int = 10  # 最大重连次数


class EventBusClient:
    """
    OpenClaw EventBus 客户端（WebSocket 远程连接）
    
    功能：
    1. 通过 WebSocket 连接祖龙系统 EventBus（ws://localhost:5555/eventbus）
    2. 发布事件到祖龙系统（所有事件路由到 L1-B）
    3. 订阅祖龙系统事件（接收 L1-B 分发的下行事件）
    4. 自动重连机制
    """
    
    def __init__(self, config: Optional[EventBusConfig] = None):
        """
        初始化 EventBus 客户端
        
        Args:
            config: 客户端配置
        """
        self.config = config or EventBusConfig()
        self._connected = False
        self._subscribers: Dict[OpenClawEventType, List[tuple]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._reconnect_attempts = 0
        
        # 🔥 WebSocket 客户端
        self._ws_url = f"ws://{self.config.host}:{self.config.port}/eventbus"
        self._ws: Optional[websocket.WebSocket] = None
        self._ws_thread: Optional[threading.Thread] = None
        
        logger.info(f"[EventBusClient] 初始化完成，客户端名称：{self.config.client_name}")
        logger.info(f"[EventBusClient] WebSocket URL: {self._ws_url}")
    
    async def connect(self) -> bool:
        """
        连接到祖龙系统 EventBus（通过 WebSocket）
        
        Returns:
            bool: 是否连接成功
        """
        logger.info(f"[EventBusClient] 🔌 正在连接到 {self._ws_url}...")
        logger.info(f"[EventBusClient] 启动 WebSocket 客户端线程")
        
        try:
            # 🔥 [关键修复] 设置_running 为 True，启动监听循环
            self._running = True
            
            # 启动 WebSocket 客户端线程
            self._ws_thread = threading.Thread(target=self._run_websocket_client, daemon=True)
            self._ws_thread.start()
            logger.info(f"[EventBusClient] WebSocket 线程已启动，线程 ID: {self._ws_thread.ident}")
            
            # 等待连接建立（最多 5 秒）
            for i in range(50):
                if self._connected:
                    logger.info(f"[EventBusClient] ✅ WebSocket 连接成功！ (等待 {i*0.1:.1f}秒)")
                    return True
                await asyncio.sleep(0.1)
            
            logger.error(f"[EventBusClient] ❌ 连接超时 (5 秒)")
            logger.error(f"[EventBusClient] 连接状态：_connected={self._connected}, _ws={self._ws}")
            return False
            
        except Exception as e:
            logger.error(f"[EventBusClient] ❌ 连接失败：{e}")
            await self._reconnect()
            return False
    
    def _run_websocket_client(self):
        """运行 WebSocket 客户端（在独立线程中）"""
        try:
            logger.info(f"[EventBusClient] 正在连接 WebSocket：{self._ws_url}")
            # 创建 WebSocket 连接
            self._ws = websocket.create_connection(
                self._ws_url,
                timeout=10,
                enable_multithread=True
            )
            
            logger.info(f"[EventBusClient] ✅ WebSocket 已连接到：{self._ws_url}")
            logger.info(f"[EventBusClient] WebSocket 连接 ID: {id(self._ws)}")
            self._connected = True
            self._reconnect_attempts = 0
            
            # 发送订阅请求
            subscribe_msg = {
                "type": "SUBSCRIBE",
                "event_types": ["ACTION_SPEAK", "L2_OUTPUT", "SYSTEM_STATUS"]
            }
            self._ws.send(json.dumps(subscribe_msg))
            logger.info(f"[EventBusClient] 已发送订阅消息：{subscribe_msg}")
            logger.info(f"[EventBusClient] 已订阅远程事件")
            
            # 监听消息循环
            self._listen_loop()
            
        except Exception as e:
            logger.error(f"[EventBusClient] ❌ WebSocket 连接失败：{e}")
            logger.error(f"[EventBusClient] 目标 URL: {self._ws_url}")
            import traceback
            logger.error(traceback.format_exc())
            self._connected = False
            asyncio.create_task(self._reconnect())
    
    def _listen_loop(self):
        """监听 WebSocket 消息循环"""
        logger.info(f"[EventBusClient] 开始监听 WebSocket 消息... (连接：{id(self._ws)})")
        listen_count = 0
        while self._running:
            try:
                listen_count += 1
                if listen_count % 10 == 0:
                    logger.info(f"[EventBusClient] 监听循环 #{listen_count}...")
                # 🔥 添加超时，避免永久阻塞
                self._ws.settimeout(5.0)  # 5 秒超时
                message = self._ws.recv()
                logger.info(f"[EventBusClient] 📥 收到 WebSocket 消息：{message}")
                if message:
                    data = json.loads(message)
                    logger.info(f"[EventBusClient] 🔍 解析消息：{data}")
                    logger.info(f"[EventBusClient] 消息类型：{data.get('type')}")
                    self._handle_remote_message(data)
            except websocket.WebSocketTimeoutException:
                # 超时是正常的，继续监听
                if listen_count % 10 == 0:
                    logger.info(f"[EventBusClient] 接收超时，继续监听...")
                pass
            except websocket.WebSocketConnectionClosedException:
                logger.error("WebSocket 连接已关闭")
                self._connected = False
                break
            except Exception as e:
                logger.error(f"监听 WebSocket 失败：{e}")
                time.sleep(1)
    
    def _handle_remote_message(self, data: dict):
        """
        处理远程消息
        
        Args:
            data: 消息数据
        """
        message_type = data.get("type")
        
        if message_type == "SUBSCRIBE":
            # 收到远程事件（从祖龙系统转发过来）
            event_data = data.get("event", {})
            self._dispatch_remote_event(event_data)
        
        elif message_type == "ACK":
            # 发布确认
            logger.debug(f"[EventBusClient] 发布确认：{data.get('event_type')}")
        
        elif message_type == "ERROR":
            logger.error(f"[EventBusClient] 远程错误：{data.get('message')}")
        
        else:
            logger.warning(f"[EventBusClient] 未知的消息类型：{message_type}")
    
    def _dispatch_remote_event(self, event_data: dict):
        """
        分发远程事件到本地订阅者
        
        Args:
            event_data: 事件数据
        """
        try:
            # 转换为 OpenClaw 事件类型
            event_type_map = {
                "ACTION_SPEAK": OpenClawEventType.ACTION_SPEAK,
                "L2_OUTPUT": OpenClawEventType.L2_OUTPUT,
                "SYSTEM_STATUS": OpenClawEventType.SYSTEM_STATUS,
            }
            
            event_type_str = event_data.get("type")
            if event_type_str not in event_type_map:
                logger.debug(f"忽略未订阅的事件类型：{event_type_str}")
                return
            
            oc_event_type = event_type_map[event_type_str]
            
            # 🔥 [关键修复] 去掉 id 和 timestamp 字段，祖龙系统不再提供这两个字段
            # 创建 OpenClaw 事件
            event = ZulongEvent(
                type=oc_event_type,
                priority=OpenClawEventPriority.NORMAL,
                source=event_data.get("source", "Remote"),
                payload=event_data.get("payload", {}),
            )
            
            # 通知本地订阅者
            if oc_event_type in self._subscribers:
                for callback, name in self._subscribers[oc_event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"订阅者 {name} 处理失败：{e}")
            
        except Exception as e:
            logger.error(f"分发远程事件失败：{e}")
    
    async def disconnect(self):
        """断开连接"""
        logger.info("[EventBusClient] 正在断开连接...")
        self._running = False
        self._connected = False
        
        # 等待事件队列清空
        await asyncio.sleep(0.5)
        logger.info("[EventBusClient] ✅ 已断开连接")
    
    async def _reconnect(self):
        """重连机制"""
        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(f"[EventBusClient] ❌ 达到最大重连次数，放弃重连")
            return
        
        self._reconnect_attempts += 1
        logger.info(
            f"[EventBusClient] {self._reconnect_attempts}/{self.config.max_reconnect_attempts} "
            f"秒后重试..."
        )
        
        await asyncio.sleep(self.config.reconnect_interval)
        await self.connect()
    
    def publish(self, event: ZulongEvent):
        """
        发布事件到祖龙系统（通过 WebSocket）
        
        架构说明：
        - 所有事件统一路由到 L1-B
        - L1-B Gatekeeper 负责判断、过滤、优先级排序和转发
        
        Args:
            event: 事件对象
        """
        if not self._connected:
            logger.warning(f"[EventBusClient] ⚠️ 未连接，事件已加入队列：{event.type}")
            asyncio.create_task(self._event_queue.put(event))
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📡 [EventBusClient] 发布事件")
        logger.info(f"📡 [EventBusClient] 事件类型：{event.type.value}")
        logger.info(f"📡 [EventBusClient] 事件优先级：{event.priority.value}")
        logger.info(f"📡 [EventBusClient] 事件来源：{event.source}")
        logger.info(f"📡 [EventBusClient] Payload: {event.payload}")
        logger.info(f"{'='*80}\n")
        
        # 通过 WebSocket 发送到远程
        self._publish_to_remote(event)
    
    def _publish_to_remote(self, event: ZulongEvent):
        """
        发布到远程 EventBus（通过 WebSocket）
        
        Args:
            event: 事件对象
        """
        try:
            # 转换为祖龙系统 EventType
            from zulong.core.types import EventType as ZulongEventType, EventPriority as ZulongEventPriority
            
            event_type_map = {
                OpenClawEventType.USER_SPEECH: ZulongEventType.USER_SPEECH,
                OpenClawEventType.USER_TEXT: ZulongEventType.USER_TEXT,
                OpenClawEventType.USER_COMMAND: ZulongEventType.USER_COMMAND,
                OpenClawEventType.SENSOR_VISION: ZulongEventType.SENSOR_VISION,
                OpenClawEventType.SENSOR_VISION_STATE: ZulongEventType.SENSOR_VISION_STATE,
                OpenClawEventType.TASK_EXECUTE: ZulongEventType.TASK_EXECUTE,
                OpenClawEventType.ACTION_RESULT: ZulongEventType.ACTION_RESULT,
                OpenClawEventType.ACTION_SPEAK: ZulongEventType.ACTION_SPEAK,
                OpenClawEventType.L2_OUTPUT: ZulongEventType.L2_OUTPUT,
                OpenClawEventType.SYSTEM_STATUS: ZulongEventType.SYSTEM_STATUS,
            }
            
            # 优先级转换
            zulong_priority = ZulongEventPriority(event.priority.value)
            
            # 构建远程消息
            # 🔥 [关键修复] 去掉 id 和 timestamp 字段，祖龙系统会自动生成
            message = {
                "type": "PUBLISH",
                "event": {
                    "type": event_type_map[event.type].value,
                    "priority": zulong_priority.value,
                    "source": event.source,
                    "payload": event.payload,
                }
            }
            
            # 通过 WebSocket 发送
            if self._ws:
                ws_message = json.dumps(message)
                logger.debug(f"[EventBusClient] 发送 WebSocket 消息：{ws_message}")
                self._ws.send(ws_message)
                logger.info(f"[EventBusClient] ✅ 事件已发布到远程 EventBus (WebSocket 连接：{id(self._ws)})")
            else:
                logger.error("[EventBusClient] ❌ WebSocket 未连接，无法发布事件")
                logger.error(f"[EventBusClient] 连接状态：_connected={self._connected}, _ws={self._ws}")
            
        except Exception as e:
            logger.error(f"[EventBusClient] ❌ 发布失败：{e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def subscribe(
        self,
        event_type: OpenClawEventType,
        handler: Callable[[ZulongEvent], Any],
        subscriber: str = "OpenClaw"
    ):
        """
        订阅事件
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
            subscriber: 订阅者名称
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append((handler, subscriber))
        
        logger.info(f"[EventBusClient] ✅ 本地已订阅 {event_type.name} (订阅者：{subscriber})")
    
    def unsubscribe(self, event_type: OpenClawEventType, handler: Callable):
        """
        取消订阅
        
        Args:
            event_type: 事件类型
            handler: 事件处理函数
        """
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                (h, s) for h, s in self._subscribers[event_type] if h != handler
            ]
    
    async def _dispatch_loop(self):
        """事件分发循环"""
        logger.info("[EventBusClient] 启动事件分发循环...")
        
        while self._running:
            try:
                # 处理队列中的事件
                while not self._event_queue.empty():
                    event = await self._event_queue.get()
                    self._publish_to_local_bus(event)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"[EventBusClient] 分发循环错误：{e}")
                await asyncio.sleep(1.0)
    
    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._connected


# 全局单例
_event_bus_client: Optional[EventBusClient] = None


def get_event_bus_client(config: Optional[EventBusConfig] = None) -> EventBusClient:
    """
    获取 EventBus 客户端单例
    
    Args:
        config: 客户端配置
    
    Returns:
        EventBusClient: 客户端实例
    """
    global _event_bus_client
    
    if _event_bus_client is None:
        _event_bus_client = EventBusClient(config)
    
    return _event_bus_client
