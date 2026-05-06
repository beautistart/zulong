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
import mimetypes
from typing import Dict, Set, Optional
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import time
from pathlib import Path

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
    test_storage_path: str = "openclaw_bridge/web/test_storage"


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
        
        # uvicorn 事件循环引用（供跨线程调度使用）
        self._uvicorn_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 测试对话存储
        self._test_sessions: Dict[str, list] = {}  # session_id -> messages
        self._storage_path = Path(self.config.test_storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        # 创建 FastAPI 应用
        self.app = FastAPI(title="OpenClaw Web UI")
        
        # 启用 CORS
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # 捕获 uvicorn 事件循环引用（供跨线程安全调度）
        @self.app.on_event("startup")
        async def _capture_event_loop():
            self._uvicorn_loop = asyncio.get_running_loop()
            logger.info("[OpenClawWebAdapter] 已捕获 uvicorn 事件循环引用")
        
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
        
        # ========== 🔥 新增：复杂任务测试 API ==========
        
        @self.app.post("/api/test/start")
        async def start_test_session(data: dict):
            """
            开始一个新的测试会话
            
            Args:
                data: {"session_name": "测试名称", "description": "测试描述"}
            
            Returns:
                {"session_id": "xxx", "status": "started"}
            """
            import uuid
            session_id = str(uuid.uuid4())
            self._test_sessions[session_id] = []
            
            logger.info(f"🧪 [测试 API] 开始测试会话：{session_id}")
            
            return JSONResponse({
                "session_id": session_id,
                "status": "started",
                "message": "测试会话已创建"
            })
        
        @self.app.post("/api/test/send")
        async def send_test_message(data: dict):
            """
            发送测试消息到祖龙系统
            
            Args:
                data: {
                    "session_id": "会话 ID",
                    "message": "测试消息内容"
                }
            
            Returns:
                {"status": "sent", "trace_id": "xxx"}
            """
            session_id = data.get("session_id")
            message = data.get("message")
            
            if not session_id or not message:
                raise HTTPException(status_code=400, detail="缺少 session_id 或 message")
            
            if session_id not in self._test_sessions:
                raise HTTPException(status_code=404, detail="会话不存在")
            
            logger.info(f"🧪 [测试 API] 收到测试消息：{message[:50]}...")
            
            # 保存测试消息到会话
            self._test_sessions[session_id].append({
                "role": "user",
                "content": message,
                "timestamp": time.time()
            })
            
            # 创建祖龙事件
            event = ZulongEvent(
                type=OpenClawEventType.USER_TEXT,
                source="openclaw/test_api",
                payload={
                    "text": message,
                    "confidence": 1.0,
                    "session_id": session_id
                },
                priority=OpenClawEventPriority.NORMAL
            )
            
            # 发布到 EventBus
            self.event_bus.publish(event)

            # 广播用户消息到前端 WebSocket（让 Web 端可见）
            import asyncio
            await self.broadcast_user_message(message, session_id)

            # 保存会话到磁盘
            self._save_session(session_id)
            
            logger.info(f"🧪 [测试 API] 消息已发布到 EventBus")
            
            return JSONResponse({
                "status": "sent",
                "session_id": session_id,
                "message_id": len(self._test_sessions[session_id]) - 1
            })
        
        @self.app.get("/api/test/session/{session_id}")
        async def get_session(session_id: str):
            """
            获取测试会话的完整对话历史
            
            Args:
                session_id: 会话 ID
            
            Returns:
                {
                    "session_id": "xxx",
                    "messages": [...],
                    "ai_responses": [...]
                }
            """
            if session_id not in self._test_sessions:
                # 尝试从磁盘加载
                session_file = self._storage_path / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file, 'r', encoding='utf-8') as f:
                        self._test_sessions[session_id] = json.load(f)
                else:
                    raise HTTPException(status_code=404, detail="会话不存在")
            
            return JSONResponse({
                "session_id": session_id,
                "messages": self._test_sessions[session_id]
            })
        
        @self.app.get("/api/test/sessions")
        async def list_sessions():
            """
            列出所有测试会话
            
            Returns:
                {"sessions": [{"session_id": "xxx", "message_count": 10, "created_at": 123}]}
            """
            sessions = []
            
            # 内存中的会话
            for session_id, messages in self._test_sessions.items():
                sessions.append({
                    "session_id": session_id,
                    "message_count": len(messages),
                    "created_at": messages[0]["timestamp"] if messages else 0,
                    "last_updated": messages[-1]["timestamp"] if messages else 0
                })
            
            # 磁盘上的会话
            if self._storage_path.exists():
                for file in self._storage_path.glob("*.json"):
                    session_id = file.stem
                    if session_id not in self._test_sessions:
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                sessions.append({
                                    "session_id": session_id,
                                    "message_count": len(data),
                                    "created_at": data[0]["timestamp"] if data else 0,
                                    "last_updated": data[-1]["timestamp"] if data else 0,
                                    "from_disk": True
                                })
                        except Exception as e:
                            logger.warning(f"加载会话文件失败：{e}")
            
            return JSONResponse({"sessions": sessions})
        
        @self.app.delete("/api/test/session/{session_id}")
        async def delete_session(session_id: str):
            """删除测试会话"""
            if session_id in self._test_sessions:
                del self._test_sessions[session_id]
            
            # 删除磁盘文件
            session_file = self._storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            return JSONResponse({"status": "deleted"})
        
        @self.app.post("/api/test/export/{session_id}")
        async def export_session(session_id: str):
            """导出测试会话为 JSON 文件"""
            if session_id not in self._test_sessions:
                # 尝试从磁盘加载
                session_file = self._storage_path / f"{session_id}.json"
                if session_file.exists():
                    with open(session_file, 'r', encoding='utf-8') as f:
                        self._test_sessions[session_id] = json.load(f)
                else:
                    raise HTTPException(status_code=404, detail="会话不存在")
            
            # 导出到文件
            export_file = self._storage_path / f"{session_id}_export.json"
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self._test_sessions[session_id], f, indent=2, ensure_ascii=False)
            
            return JSONResponse({
                "status": "exported",
                "file": str(export_file)
            })
        
        # ========== 聊天会话持久化 API ==========
        # 会话数据存储在服务端 JSON 文件，跨端口/浏览器持久化
        
        self._chat_sessions_path = Path(self.config.static_path).parent / "chat_sessions.json"
        
        @self.app.get("/api/chat/sessions")
        async def get_chat_sessions():
            """读取服务端持久化的聊天会话数据"""
            try:
                if self._chat_sessions_path.exists():
                    with open(self._chat_sessions_path, 'r', encoding='utf-8') as f:
                        return JSONResponse(json.load(f))
            except Exception as e:
                logger.error(f"[WebAdapter] 读取聊天会话失败: {e}")
            return JSONResponse({"activeSessionId": None, "sessions": []})
        
        @self.app.post("/api/chat/sessions")
        async def save_chat_sessions(data: dict):
            """保存聊天会话数据到服务端"""
            try:
                self._chat_sessions_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._chat_sessions_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return JSONResponse({"status": "ok"})
            except Exception as e:
                logger.error(f"[WebAdapter] 保存聊天会话失败: {e}")
                return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
        
        # ========== 文件服务 API ==========
        
        @self.app.get("/api/files/serve")
        async def serve_file(path: str = ""):
            """提供工作区文件访问（图片预览、文件下载）"""
            safe_path = self._validate_file_path(path)
            if not safe_path:
                return JSONResponse({"error": "access denied"}, status_code=403)
            if not safe_path.exists() or not safe_path.is_file():
                return JSONResponse({"error": "not found"}, status_code=404)
            mime, _ = mimetypes.guess_type(str(safe_path))
            return FileResponse(
                str(safe_path),
                media_type=mime or "application/octet-stream",
            )
        
        @self.app.get("/api/files/info")
        async def file_info(path: str = ""):
            """返回文件元信息"""
            safe_path = self._validate_file_path(path)
            if not safe_path or not safe_path.exists():
                return JSONResponse({"exists": False})
            mime, _ = mimetypes.guess_type(str(safe_path))
            stat = safe_path.stat()
            return JSONResponse({
                "exists": True,
                "name": safe_path.name,
                "path": str(safe_path),
                "size": stat.st_size,
                "is_image": (mime or "").startswith("image/"),
                "mime_type": mime or "unknown",
                "modified_at": stat.st_mtime,
            })
        
        logger.info("[OpenClawWebAdapter] 路由已设置（包含测试 API + 文件服务）")
    
    def _validate_file_path(self, path: str) -> Optional[Path]:
        """校验文件路径安全性（白名单 + 路径遍历防护）"""
        if not path or '..' in path:
            return None
        project_root = Path(__file__).resolve().parent.parent.parent
        allowed_roots = [
            project_root,
            project_root / "agent_workspace",
        ]
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = project_root / path
        try:
            candidate = candidate.resolve()
        except (OSError, ValueError):
            return None
        for root in allowed_roots:
            try:
                candidate.relative_to(root.resolve())
                return candidate
            except ValueError:
                continue
        return None
    
    def _save_session(self, session_id: str):
        """
        保存会话到磁盘
        
        Args:
            session_id: 会话 ID
        """
        if session_id not in self._test_sessions:
            return
        
        session_file = self._storage_path / f"{session_id}.json"
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self._test_sessions[session_id], f, indent=2, ensure_ascii=False)
            logger.debug(f"💾 [测试 API] 会话已保存：{session_file}")
        except Exception as e:
            logger.error(f"❌ [测试 API] 保存会话失败：{e}")
    
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
            
            # 推送当前记忆图谱快照（重启/刷新后前端立即可见）
            await self._push_memory_graph_snapshot(websocket)
            
            # 首次连接时检查可恢复任务（延迟 1 秒确保前端就绪）
            if not getattr(self, '_recovery_checked', False):
                self._recovery_checked = True
                import threading as _th
                def _delayed_recovery_check():
                    import time as _time
                    _time.sleep(1)
                    try:
                        from zulong.l2.recovery_notifier import RecoveryNotifier
                        RecoveryNotifier.check_and_notify()
                    except Exception:
                        pass
                _th.Thread(target=_delayed_recovery_check, daemon=True).start()
            
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
            self._active_connections.discard(websocket)
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
            session_id = data.get("session_id", None)
            request_id = data.get("request_id", None)
            referenced_nodes = data.get("referenced_nodes", [])
            
            if text:
                logger.info(f"[OpenClawWebAdapter] 📥 收到消息：{text}")
                logger.info(f"[OpenClawWebAdapter] 🔍 完整数据：{data}")
                await self._publish_chat_message(text, session_id, request_id, referenced_nodes)
        
        elif message_type == "REQUEST_MEMORY_GRAPH":
            # 前端主动请求记忆图谱快照（切换 tab 时、手动刷新时）
            await self._push_memory_graph_snapshot(websocket)
        
        elif message_type == "EXPAND_NODE":
            # 前端请求展开某节点的子节点
            node_id = data.get("node_id")
            if node_id:
                await self._handle_expand_node(node_id, websocket)
        
        elif message_type == "STOP_GENERATION":
            request_id = data.get("request_id", None)
            logger.info(f"[OpenClawWebAdapter] 🛑 收到停止请求: request_id={request_id}")
            event = ZulongEvent(
                type=OpenClawEventType.USER_TEXT,
                source="openclaw/web_ui",
                payload={
                    "action": "stop_generation",
                    "request_id": request_id,
                },
                priority=OpenClawEventPriority.HIGH,
            )
            self.event_bus.publish(event)
    
    async def _publish_chat_message(self, text: str, session_id: str = None, request_id: str = None, referenced_nodes: list = None):
        """
        发布聊天消息到 EventBus
        
        Args:
            text: 聊天文本
            session_id: 会话 ID（用于测试追踪）
            request_id: 请求 ID
            referenced_nodes: 引用的任务节点 ID 列表
        """
        # 创建祖龙事件
        # 🔥 [修复] Web 文本输入应该使用 USER_TEXT，不是 USER_SPEECH
        # 这样 L1-B 会使用 TEXT_ONLY 模式，不会自动语音播报
        payload = {
            "text": text,
            "confidence": 1.0  # Web 文本输入，置信度 100%
        }
        
        # 如果提供了 session_id，添加到 payload
        if session_id:
            payload["session_id"] = session_id
            logger.info(f"[OpenClawWebAdapter] 🧪 会话 ID: {session_id}")
        
        # 添加 request_id（用于思考过程关联）
        if request_id:
            payload["request_id"] = request_id
        
        # 添加引用节点列表
        if referenced_nodes:
            payload["referenced_nodes"] = referenced_nodes
        
        event = ZulongEvent(
            type=OpenClawEventType.USER_TEXT,
            source="openclaw/web_ui",
            payload=payload,
            priority=OpenClawEventPriority.NORMAL
        )
        
        # 发布到 EventBus
        self.event_bus.publish(event)
        
        logger.info(f"[OpenClawWebAdapter] ✅ 消息已发布到 EventBus (USER_TEXT, confidence=1.0)")
    
    def schedule_coroutine(self, coro):
        """线程安全地调度协程到 uvicorn 事件循环
        
        供 WebResponseListener 等从非 asyncio 线程调用。
        
        Args:
            coro: 要调度的协程对象
            
        Returns:
            concurrent.futures.Future 或 None
        """
        loop = self._uvicorn_loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future
        else:
            logger.warning("[OpenClawWebAdapter] uvicorn 事件循环不可用，尝试新建循环执行")
            try:
                new_loop = asyncio.new_event_loop()
                new_loop.run_until_complete(coro)
                new_loop.close()
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 备用循环执行失败: {e}")
            return None

    async def broadcast_user_message(self, text: str, session_id: str = None):
        """广播用户消息到前端 WebSocket（用于 API 发送的消息在 Web 端可见）

        Args:
            text: 用户消息文本
            session_id: 会话 ID
        """
        if not self._active_connections:
            return

        message = {
            "type": "USER_MESSAGE",
            "text": text,
            "session_id": session_id,
            "timestamp": asyncio.get_event_loop().time()
        }

        disconnected = set()
        for connection in self._active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 用户消息广播失败：{e}")
                disconnected.add(connection)

        for connection in disconnected:
            self._active_connections.remove(connection)

        logger.info(f"[OpenClawWebAdapter] 用户消息已广播到 {len(self._active_connections)} 个连接")

    async def broadcast_response(self, text: str, request_id: str = None):
        """
        广播响应到所有前端
        
        Args:
            text: 响应文本
            request_id: 请求 ID（用于关联思考过程面板）
        """
        if not self._active_connections:
            logger.warning("[OpenClawWebAdapter] ⚠️ 无活动连接，无法广播")
            return
        
        message = {
            "type": "CHAT_RESPONSE",
            "text": text,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        if request_id:
            message["request_id"] = request_id
        
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
    
    async def broadcast_streaming_response(self, text: str, chunk: str = "", request_id: str = None):
        """
        广播流式响应到所有前端
        
        Args:
            text: 累积的完整文本
            chunk: 本次增量文本片段
            request_id: 请求 ID（用于关联思考过程面板）
        """
        if not self._active_connections:
            return
        
        message = {
            "type": "STREAMING_RESPONSE",
            "text": text,
            "chunk": chunk,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        if request_id:
            message["request_id"] = request_id
        
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
    
    async def broadcast_thinking_step(self, payload: dict):
        """
        广播思考步骤到所有前端
        
        Args:
            payload: 思考步骤数据
        """
        if not self._active_connections:
            return
        
        message = {
            "type": "THINKING_STEP",
            **payload
        }
        
        disconnected = set()
        for connection in self._active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 思考步骤广播失败：{e}")
                disconnected.add(connection)
        
        for connection in disconnected:
            self._active_connections.remove(connection)
    
    async def _push_memory_graph_snapshot(self, websocket: WebSocket):
        """向单个 WebSocket 连接推送当前记忆图谱快照（仅顶层节点）
        
        在新连接建立或前端主动请求时调用，使用 depth=0 仅推送
        根节点，前端通过 EXPAND_NODE 按需加载子节点。
        """
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            payload = mg.to_frontend_dict(depth=0)
            if payload and payload.get("nodes"):
                message = {
                    "type": "MEMORY_GRAPH_UPDATE",
                    "update_type": "full",
                    **payload,
                }
                await websocket.send_json(message)
                logger.info(
                    f"[OpenClawWebAdapter] 已推送记忆图谱快照(depth=0): "
                    f"{len(payload.get('nodes', []))} 节点, "
                    f"{len(payload.get('edges', []))} 边"
                )
        except ImportError:
            pass  # 记忆图谱模块不可用，静默降级
        except Exception as e:
            logger.debug(f"[OpenClawWebAdapter] 推送记忆图谱快照失败: {e}")

    async def _handle_expand_node(self, node_id: str, websocket: WebSocket):
        """处理前端展开节点请求，返回子节点数据"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            result = mg.get_node_children_for_frontend(node_id)
            message = {
                "type": "MEMORY_GRAPH_EXPAND_RESULT",
                **result,
            }
            await websocket.send_json(message)
            logger.debug(
                f"[OpenClawWebAdapter] 展开节点 {node_id}: "
                f"{len(result.get('nodes', []))} 子节点"
            )
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"[OpenClawWebAdapter] 展开节点失败 ({node_id}): {e}")
    
    async def broadcast_memory_graph_update(self, payload: dict):
        """
        广播记忆图谱更新到所有前端
        
        Args:
            payload: 记忆图谱更新数据 (包含 nodes, edges, stats, update_type)
        """
        if not self._active_connections:
            return
        
        message = {
            "type": "MEMORY_GRAPH_UPDATE",
            **payload
        }
        
        disconnected = set()
        for connection in self._active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"[OpenClawWebAdapter] 记忆图谱广播失败：{e}")
                disconnected.add(connection)
        
        for connection in disconnected:
            self._active_connections.remove(connection)
    
    async def start(self):
        """启动 Web 服务器"""
        logger.info("[OpenClawWebAdapter] 启动 Web 服务器...")
        self._running = True
        
        # 配置 uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
            ws_ping_interval=None,
            ws_ping_timeout=None,
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
        
        # 清理连接集合（不尝试跨事件循环 close，uvicorn 关闭时会自动断连）
        self._active_connections.clear()
        logger.info("[OpenClawWebAdapter] ✅ Web 服务器已停止")
