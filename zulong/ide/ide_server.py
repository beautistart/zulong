"""
祖龙 IDE WebSocket 服务端

替代 ide_api_server.py 的 HTTP 代理模式，通过 WebSocket 实现双向实时通信。
祖龙 Python 后端作为唯一 Agent 大脑，VS Code 插件仅作 UI + 工具执行层。

协议：
  插件 → 后端: session_start / tool_result / user_cancel
  后端 → 插件: tool_request / display_text / display_reasoning / task_complete / task_error / status_update
"""

import asyncio
import json
import logging
import logging.handlers
import os
import time
import uuid
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger(__name__)

from zulong.core.unified_protocol import (
    MessageType,
    ProtocolBridge,
    make_unified_message,
)

_protocol_bridge = ProtocolBridge()


# ── 待发送消息缓存（断线重连恢复）──────────────────────────

class PendingMessageCache:
    """WebSocket断线后缓存待发送消息，重连后恢复"""
    _cache: Dict[str, list] = {}  # session_id → [{"msg": dict, "timestamp": float}, ...]
    _ttl = 300  # 5分钟过期
    
    @classmethod
    def cache(cls, session_id: str, msg: Dict[str, Any]) -> None:
        """缓存消息"""
        if session_id not in cls._cache:
            cls._cache[session_id] = []
        cls._cache[session_id].append({
            "msg": msg,
            "timestamp": time.time(),
        })
        # 限制缓存大小
        if len(cls._cache[session_id]) > 10:
            cls._cache[session_id] = cls._cache[session_id][-10:]
        logger.debug(f"[PendingMessageCache] 缓存消息: session={session_id[:12]}, type={msg.get('type')}")
    
    @classmethod
    def pop(cls, session_id: str) -> list:
        """取出并清除缓存消息（过滤过期）"""
        msgs = cls._cache.get(session_id, [])
        now = time.time()
        valid = [m["msg"] for m in msgs if now - m["timestamp"] < cls._ttl]
        cls._cache[session_id] = []
        if valid:
            logger.info(f"[PendingMessageCache] 恢复 {len(valid)} 条消息: session={session_id[:12]}")
        return valid
    
    @classmethod
    def clear(cls, session_id: str) -> None:
        """清除缓存"""
        cls._cache.pop(session_id, None)

# ── 模块级日志持久化 ──────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(
    _LOG_DIR, f"zulong_ide_{time.strftime('%Y%m%d_%H%M%S')}.log")
_root = logging.getLogger()
if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in _root.handlers):
    _root.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in _root.handlers):
        _sh = logging.StreamHandler()
        _sh.setFormatter(_fmt)
        _root.addHandler(_sh)
    _fh = logging.handlers.RotatingFileHandler(
        _LOG_FILE, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _root.addHandler(_fh)
    logger.info(f"[ZulongIDE] 日志文件: {os.path.abspath(_LOG_FILE)}")

app = FastAPI(title="Zulong IDE Server")
_main_loop: Optional[asyncio.AbstractEventLoop] = None

from zulong.ide.ide_session import AgentSessionStore
_global_session_store = AgentSessionStore()

def get_session_store() -> AgentSessionStore:
    return _global_session_store

# IDE 路由器 — 可独立使用或由 Launcher 挂载到其 FastAPI app 上
ide_router = APIRouter()


# ── 会话管理 ──────────────────────────────────────────

class IDESession:
    """单个 WebSocket 连接对应一个会话"""

    def __init__(self, session_id: str, ws: WebSocket):
        self.session_id = session_id
        self.ws = ws
        self.created_at = time.time()
        # 统一协议: 客户端类型和 API 版本（handshake 时设置）
        self.client_type: str = "unknown"
        self.api_version: str = "1.0"
        # 用于 FC 循环向 WS 发消息
        self.outbound_queue: asyncio.Queue = asyncio.Queue()
        # 用于 WS 读循环向 FC 循环传递工具结果
        self.tool_result_queue: asyncio.Queue = asyncio.Queue()
        # 取消信号
        self.cancel_event = asyncio.Event()
        # FC 循环 task 引用
        self.fc_task: Optional[asyncio.Task] = None
        # Runner 实例（FC 循环期间持续存活）
        self.runner = None
        # 会话元数据（任务-项目关联）
        self.cwd: Optional[str] = None
        self.project_id: Optional[str] = None
        self.task_graph_id: Optional[str] = None
        self.conversation_id: Optional[str] = None
        self.web_turn_id: Optional[str] = None
        self.workspace_trusted: Optional[bool] = None
        self.trust_required: bool = False

    def to_info_dict(self) -> Dict[str, Any]:
        """序列化会话元数据（用于 REST API / WELCOME 消息）"""
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "project_id": self.project_id,
            "task_graph_id": self.task_graph_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.web_turn_id,
            "created_at": self.created_at,
            "has_fc_task": self.fc_task is not None and not self.fc_task.done(),
            "workspace_trusted": self.workspace_trusted,
            "trust_required": self.trust_required,
        }

    async def send_msg(self, msg_type: str, payload: Dict[str, Any]) -> None:
        """向插件发送消息"""
        msg = make_unified_message(
            msg_type,
            payload,
            session_id=self.session_id,
        )
        await self.outbound_queue.put(msg)

    async def send_unified_msg(self, msg_type: str, payload: Dict[str, Any]) -> None:
        """按统一协议类型发送消息。

        api_version=2.0 的客户端收到 namespace:action；旧客户端保持 IDE
        legacy 类型，便于渐进迁移。
        """
        unified = make_unified_message(
            msg_type,
            payload,
            session_id=self.session_id,
        )
        if self.api_version.startswith("2"):
            await self.outbound_queue.put(unified)
            return
        await self.outbound_queue.put(
            _protocol_bridge.from_unified(unified, target_format="ide")
        )


# IDEClientConnection is the long-lived VS Code backend bridge.
IDEClientConnection = IDESession


class AgentRunSession:
    """A single agent run initiated by Web, voice, or another user-facing surface."""

    def __init__(
        self,
        *,
        conversation_id: str,
        turn_id: str,
        task_text: str,
        workspace_path: str,
        project_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
        source: str = "web_chat",
    ):
        self.run_id = f"run_{uuid.uuid4().hex}"
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.task_text = task_text
        self.workspace_path = workspace_path
        self.project_id = project_id
        self.task_graph_id = task_graph_id
        self.source = source
        self.created_at = time.time()
        self.ide_session_id: Optional[str] = None
        self.status = "queued"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "workspace_path": self.workspace_path,
            "project_id": self.project_id,
            "task_graph_id": self.task_graph_id,
            "source": self.source,
            "created_at": self.created_at,
            "ide_session_id": self.ide_session_id,
            "status": self.status,
        }


# 活跃会话
_sessions: Dict[str, IDESession] = {}
_agent_runs: Dict[str, AgentRunSession] = {}
_workspace_trust_waiters: Dict[str, list[asyncio.Future]] = {}

# ── Web 监控连接 ──────────────────────────────────────
_monitor_connections: Set[WebSocket] = set()


def _resolve_active_task_workspace(
    preferred: Optional[str] = None,
    task_graph_id: Optional[str] = None,
) -> str:
    """Resolve the current task workspace; never use repo cwd as an implicit task dir."""
    candidates = []

    def add_candidate(value: Optional[str]) -> None:
        if value:
            candidates.append(value)

    if task_graph_id:
        try:
            from zulong.workspace.project_registry import get_project_registry
            project = get_project_registry().get_project_by_graph_id(task_graph_id)
            if project and project.path:
                add_candidate(project.path)
        except Exception:
            pass

    try:
        from zulong.tools.task_tools import get_active_task_graph, get_active_workspace_dir
        active_workspace = get_active_workspace_dir()
        active_graph = get_active_task_graph()
        active_graph_id = getattr(active_graph, "id", None) if active_graph else None
        if active_workspace and (not task_graph_id or active_graph_id == task_graph_id):
            add_candidate(active_workspace)
        if active_graph is not None:
            meta_workspace = getattr(active_graph, "metadata", {}).get("workspace_dir")
            if meta_workspace and (not task_graph_id or active_graph_id == task_graph_id):
                add_candidate(meta_workspace)
    except Exception:
        pass

    # The request payload may contain stale IDE context (for example a parent
    # folder). It is only a final fallback after TaskGraph/ProjectRegistry data.
    add_candidate(preferred)

    for candidate in candidates:
        try:
            path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(candidate))))
            if os.path.exists(path):
                return path
        except Exception:
            continue
    return ""


def _find_run_by_ide_session(ide_session_id: str) -> Optional[AgentRunSession]:
    for run in _agent_runs.values():
        if run.ide_session_id == ide_session_id:
            return run
    return None


async def broadcast_monitor_event(event_type: str, payload: dict) -> None:
    """向所有 Web 监控连接广播事件（fire-and-forget）"""
    global _monitor_connections
    if not _monitor_connections:
        logger.debug(f"[Monitor] broadcast_monitor_event: {event_type}, 无监控连接")
        return
    msg = {
        "type": event_type,
        "ts": time.time(),
        "payload": payload,
    }
    sent_count = 0
    dead: Set[WebSocket] = set()
    for ws in list(_monitor_connections):
        try:
            await ws.send_json(msg)
            sent_count += 1
        except Exception:
            dead.add(ws)
    _monitor_connections -= dead
    if sent_count > 0:
        logger.info(f"[Monitor] broadcast {event_type} → {sent_count} 个监控连接")


def _runtime_settings_payload(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = dict(payload or {})
    if not settings.get("approval_mode"):
        try:
            from zulong.config.approval_config import get_runtime_approval_mode
            settings["approval_mode"] = get_runtime_approval_mode()
        except Exception:
            settings["approval_mode"] = "manual"
    return settings


async def _send_runtime_settings_to_session(session: IDESession, payload: Optional[Dict[str, Any]] = None) -> None:
    await session.send_msg("ide_runtime_settings", _runtime_settings_payload(payload))


async def broadcast_runtime_settings(payload: Optional[Dict[str, Any]] = None) -> int:
    """同步 Web 运行时设置到已连接的 IDE 执行桥。"""
    settings = _runtime_settings_payload(payload)
    sent_count = 0
    for session in list(_sessions.values()):
        if session.client_type not in ("ide_plugin", "unknown"):
            continue
        try:
            await _send_runtime_settings_to_session(session, settings)
            sent_count += 1
        except Exception as exc:
            logger.debug("[ZulongIDE] 运行时设置同步失败 session=%s: %s", session.session_id[:12], exc)
    if sent_count:
        logger.info("[ZulongIDE] 运行时设置已同步到 %s 个 IDE 执行桥: %s", sent_count, settings)
    return sent_count


def _conversation_payload(session: IDESession) -> Dict[str, Any]:
    return {
        "conversation_id": session.conversation_id,
        "turn_id": session.web_turn_id,
        "workspace_path": session.cwd,
        "project_id": session.project_id,
        "task_graph_id": session.task_graph_id,
    }


def _select_ide_bridge(workspace_path: Optional[str] = None) -> Optional[IDEClientConnection]:
    candidates = [
        s for s in _sessions.values()
        if s.client_type in ("ide_plugin", "unknown")
    ]
    if workspace_path:
        target = os.path.normcase(os.path.abspath(workspace_path))
        for session in candidates:
            if session.cwd and os.path.normcase(os.path.abspath(session.cwd)) == target:
                return session
        return None
    return candidates[-1] if candidates else None


def _workspace_matches(session: Optional[IDEClientConnection], workspace_path: Optional[str]) -> bool:
    if not session or not workspace_path:
        return False
    try:
        return os.path.normcase(os.path.abspath(session.cwd or "")) == os.path.normcase(os.path.abspath(workspace_path))
    except Exception:
        return False


def _workspace_key(workspace_path: Optional[str]) -> str:
    try:
        return os.path.normcase(os.path.abspath(workspace_path or ""))
    except Exception:
        return os.path.normcase(str(workspace_path or ""))


def _session_workspace_trusted(session: Optional[IDEClientConnection]) -> bool:
    # Older plugin builds do not report Workspace Trust; keep backward compatibility.
    return session is not None and session.workspace_trusted is not False and not session.trust_required


async def _wait_for_workspace_trust(
    workspace_path: str,
    *,
    reason: str = "",
    timeout: float = 180.0,
) -> Dict[str, Any]:
    """Wait until VS Code reports the workspace as trusted."""
    key = _workspace_key(workspace_path)
    session = _select_ide_bridge(workspace_path)
    if _session_workspace_trusted(session):
        return {"ok": True, "status": "trusted", "session": session}

    loop = asyncio.get_running_loop()
    waiter: asyncio.Future = loop.create_future()
    _workspace_trust_waiters.setdefault(key, []).append(waiter)
    await broadcast_monitor_event("IDE_WORKSPACE_TRUST_REQUIRED", {
        "status": "workspace_trust_required",
        "workspace_path": workspace_path,
        "reason": reason,
        "message": "VS Code 已打开当前任务目录，正在等待你在 VS Code 信任该目录；信任后祖龙会自动继续。",
        "interaction": {
            "pair_id": f"workspace_trust:{workspace_path}",
            "kind": "approval",
            "status": "pending",
            "title": "等待 VS Code 目录信任",
            "detail": "请在 VS Code 弹出的信任窗口中确认当前任务目录。确认后祖龙会自动继续任务。",
            "tool_name": "workspace_trust",
        },
    })
    _notify_web_task_execution_status(
        state="workspace_trust_required",
        phase="workspace_trust_required",
        message="任务已暂停，正在等待你在 VS Code 信任当前任务目录；确认后会自动继续。",
        workspace_path=workspace_path,
        awaiting_approval=True,
        tool_name="workspace_trust",
        approval={
            "approval_id": f"workspace_trust:{workspace_path}",
            "tool_name": "workspace_trust",
            "action_summary": "等待 VS Code 目录信任",
            "workspace_path": workspace_path,
        },
        progress_items=[{
            "label": "等待 VS Code 目录信任",
            "status": "running",
            "source": "heartbeat",
            "detail": workspace_path,
            "timestamp": time.time(),
        }],
    )
    try:
        await asyncio.wait_for(waiter, timeout=max(1.0, timeout))
    except asyncio.TimeoutError:
        _notify_web_task_execution_status(
            state="blocked",
            phase="workspace_trust_timeout",
            message="等待 VS Code 目录信任超时，任务已暂停。",
            workspace_path=workspace_path,
            awaiting_approval=False,
            tool_name="workspace_trust",
            progress_items=[{
                "label": "等待 VS Code 目录信任",
                "status": "blocked",
                "source": "heartbeat",
                "detail": workspace_path,
                "timestamp": time.time(),
            }],
        )
        return {
            "ok": False,
            "status": "workspace_trust_timeout",
            "error": "已打开 VS Code，但当前任务目录未在限定时间内完成信任确认。",
        }
    finally:
        waiters = _workspace_trust_waiters.get(key, [])
        if waiter in waiters:
            waiters.remove(waiter)
        if not waiters:
            _workspace_trust_waiters.pop(key, None)

    session = _select_ide_bridge(workspace_path)
    if _session_workspace_trusted(session):
        _notify_web_task_execution_status(
            state="running",
            phase="workspace_trusted",
            message="VS Code 已信任当前任务目录，任务继续执行。",
            workspace_path=workspace_path,
            awaiting_approval=False,
            tool_name="workspace_trust",
            progress_items=[{
                "label": "VS Code 目录已信任",
                "status": "completed",
                "source": "heartbeat",
                "detail": workspace_path,
                "timestamp": time.time(),
            }],
        )
        return {"ok": True, "status": "trusted_after_grant", "session": session}
    return {
        "ok": False,
        "status": "workspace_untrusted",
        "error": "VS Code 目录信任状态仍未确认，无法继续执行任务。",
    }


def _notify_workspace_trusted(workspace_path: Optional[str]) -> None:
    key = _workspace_key(workspace_path)
    waiters = _workspace_trust_waiters.pop(key, [])
    for waiter in waiters:
        if not waiter.done():
            waiter.set_result(True)


def _notify_web_task_execution_status(**kwargs: Any) -> None:
    """Best-effort bridge to the Web task-status indicator."""
    try:
        from zulong.launcher.web_chat_router import update_task_execution_status

        update_task_execution_status(**kwargs)
    except Exception as exc:
        logger.debug("[ZulongIDE] Web task status notify skipped: %s", exc)


async def ensure_vscode_bridge(
    workspace_path: Optional[str] = None,
    *,
    vscode_command: Optional[str] = None,
    reason: str = "",
    timeout: float = 25.0,
) -> Dict[str, Any]:
    """Ensure a VS Code extension bridge is connected for workspace work.

    This is intentionally an internal state helper, not a new event channel:
    status still goes through broadcast_monitor_event and normal IDE messages.
    """
    requested_workspace = _resolve_active_task_workspace(workspace_path)
    if not requested_workspace:
        return {
            "ok": False,
            "status": "workspace_required",
            "error": "缺少当前任务工作目录，拒绝回退打开祖龙项目目录。",
        }
    session = _select_ide_bridge(requested_workspace)
    if session:
        if not _session_workspace_trusted(session):
            trusted = await _wait_for_workspace_trust(
                requested_workspace,
                reason=reason,
                timeout=max(timeout, 180.0),
            )
            if not trusted.get("ok"):
                return trusted
            session = trusted.get("session")
        return {
            "ok": True,
            "status": "connected",
            "session": session,
            "ide_session_id": session.session_id,
            "workspace_path": session.cwd or requested_workspace,
        }

    launched = _launch_vscode_workspace(requested_workspace, vscode_command)
    if not launched.get("ok"):
        return {
            "ok": False,
            "status": "launch_failed",
            "error": (
                "未检测到 VS Code 后台桥连接，且无法启动 VS Code："
                + launched.get("error", "unknown")
            ),
            **launched,
        }

    await broadcast_monitor_event("IDE_BRIDGE_WAITING", {
        "status": "waiting_for_bridge",
        "workspace_path": launched.get("workspace_path") or requested_workspace,
        "reason": reason,
        "message": "已启动 VS Code，正在等待祖龙插件后台桥连接。",
    })
    _notify_web_task_execution_status(
        state="running",
        phase="waiting_ide_bridge",
        message="已启动 VS Code，正在等待祖龙插件后台桥连接。",
        workspace_path=launched.get("workspace_path") or requested_workspace,
        awaiting_approval=False,
        tool_name="ide_bridge",
        progress_items=[{
            "label": "等待 VS Code 后台桥连接",
            "status": "running",
            "source": "tool",
            "timestamp": time.time(),
        }],
    )

    deadline = time.time() + max(1.0, timeout)
    while time.time() < deadline:
        session = _select_ide_bridge(requested_workspace)
        if session:
            if not _session_workspace_trusted(session):
                trusted = await _wait_for_workspace_trust(
                    requested_workspace,
                    reason=reason,
                    timeout=max(1.0, deadline - time.time() + 180.0),
                )
                if not trusted.get("ok"):
                    return trusted
                session = trusted.get("session")
            return {
                "ok": True,
                "status": "connected_after_launch",
                "session": session,
                "ide_session_id": session.session_id,
                "workspace_path": session.cwd or launched.get("workspace_path") or requested_workspace,
                "launch": launched,
            }
        await asyncio.sleep(0.5)

    return {
        "ok": False,
        "status": "bridge_timeout",
        "error": "已启动 VS Code，但祖龙插件后台桥未在限定时间内连接，无法执行 IDE 工具。",
        "launch": launched,
    }


async def start_agent_run_from_web(run: AgentRunSession) -> Dict[str, Any]:
    """Start a coding run on the active VS Code backend bridge.

    TSD §23.11.3-§23.11.5: 会话窗口绑定图谱。
    恢复优先级: task_graph_id 精确匹配 → InteractionStore 查找 → MemoryGraph BFS 扩散 → 全新任务。
    """
    ensured = await ensure_vscode_bridge(
        run.workspace_path,
        reason="Web 任务需要 VS Code 后台桥执行",
    )
    session = ensured.get("session")
    if not ensured.get("ok") or not session:
        await broadcast_monitor_event("IDE_SESSION_START", {
            **run.to_dict(),
            "status": "waiting_for_ide",
            "task_preview": run.task_text[:200],
            "task_title": run.task_text[:40],
        })
        return {
            "ok": False,
            "error": ensured.get("error") or "未检测到已连接的 VS Code 后台桥。",
            "bridge": {k: v for k, v in ensured.items() if k != "session"},
        }

    run.ide_session_id = session.session_id
    run.status = "running"
    _agent_runs[run.run_id] = run

    # TSD §23.11.5: 多层 task_graph_id 恢复
    try:
        from zulong.tools.task_tools import normalize_task_graph_id
        _effective_graph_id = normalize_task_graph_id(run.task_graph_id)
    except Exception:
        _effective_graph_id = run.task_graph_id or ""

    # Level A: 精确匹配 (前端传入或 Orchestrator 从已有记录补充)
    if _effective_graph_id:
        pass  # 使用已有值
    else:
        # Level B: 从 InteractionStore 查找 (页面刷新后可能存在)
        try:
            from zulong.launcher.interaction_store import get_interaction_store
            _store = get_interaction_store()
            _conv = _store.get_conversation(run.conversation_id)
            if _conv and _conv.get("task_graph_id"):
                try:
                    from zulong.tools.task_tools import normalize_task_graph_id
                    _effective_graph_id = normalize_task_graph_id(_conv["task_graph_id"])
                except Exception:
                    _effective_graph_id = _conv["task_graph_id"]
                logger.info(
                    f"[ZulongIDE] InteractionStore 恢复 graph_id: "
                    f"conv={run.conversation_id}, graph={_effective_graph_id}"
                )
        except Exception as _e:
            logger.debug(f"[ZulongIDE] InteractionStore 查找跳过: {_e}")

    # Level C: MemoryGraph BFS 扩散 (TSD §23.11.3: 窗口即绑定)
    if not _effective_graph_id and run.conversation_id:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            if _mg:
                # 从会话 ID 查找 DIALOGUE/SESSION 节点
                _session_nodes = []
                for _nid, _node in _mg._nodes.items():
                    if (_node.metadata.get("session_id") == run.conversation_id
                            and _node.metadata.get("task_graph_id")):
                        try:
                            from zulong.tools.task_tools import normalize_task_graph_id
                            _tgid = normalize_task_graph_id(_node.metadata["task_graph_id"])
                        except Exception:
                            _tgid = _node.metadata["task_graph_id"]
                        if _tgid and _tgid not in _session_nodes:
                            _session_nodes.append(_tgid)
                if _session_nodes:
                    _effective_graph_id = _session_nodes[0]
                    logger.info(
                        f"[ZulongIDE] MemoryGraph BFS 恢复 graph_id: "
                        f"conv={run.conversation_id}, graph={_effective_graph_id}"
                    )
        except Exception as _e:
            logger.debug(f"[ZulongIDE] MemoryGraph BFS 查找跳过: {_e}")

    payload = {
        "task": run.task_text,
        "cwd": run.workspace_path or session.cwd or ".",
        "project_id": run.project_id or "",
        "graph_id": _effective_graph_id or "",
        "task_graph_id": _effective_graph_id or "",
        "conversation_id": run.conversation_id,
        "turn_id": run.turn_id,
        "request_id": run.turn_id,
        "source": run.source,
    }
    if _effective_graph_id:
        await _handle_session_resume(session, payload)
    else:
        await _handle_session_start(session, payload)
    return {"ok": True, "run": run.to_dict(), "ide_session_id": session.session_id}


async def _retrieve_session_context(conversation_id: str) -> Optional[Dict[str, Any]]:
    """
    TSD 23.11.3: 从 conversation_id 定位 MemoryGraph 根会话节点
    → BFS 扩散 (depth=3, decay=0.5) → 返回上下文 dict

    Returns:
        None = 全新会话，无历史上下文可恢复
        dict = {"session_node_id", "session_label", "recent_rounds",
                "active_tasks", "task_graph_id", "hot_memories"}
    """
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg is None:
            return None

        if hasattr(mg, "get_session_node_id_for_conversation"):
            session_node_id = mg.get_session_node_id_for_conversation(conversation_id)
        else:
            # 推导 session_node_id（确定性规则，与 memory_mirror.py _compact_id 一致）
            safe_id = "".join(
                ch if ch.isalnum() or ch in "-_" else "_"
                for ch in str(conversation_id or "")
            )
            session_node_id = f"dialogue:session_{safe_id}"

        if not session_node_id or not mg.has_node(session_node_id):
            return None

        # BFS 扩散激活
        activations = mg.compute_activations(
            seed_node_ids=[session_node_id],
            max_depth=3,
            decay=0.5,
            min_activation=0.01,
        )

        # 收集上下文
        recent_rounds = []
        active_tasks = []
        task_graph_id = ""

        for node_id, activation in sorted(
            activations.items(), key=lambda x: x[1], reverse=True
        ):
            node = mg.get_node(node_id)
            if node is None:
                continue

            meta = node.metadata or {}
            label = node.label or ""
            ntype = node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type)

            if ntype == "dialogue" and meta.get("sub_type") == "round":
                recent_rounds.append({
                    "node_id": node_id,
                    "label": label,
                    "user_text": meta.get("user_text", ""),
                    "bot_text": meta.get("bot_text", ""),
                    "activation": round(activation, 3),
                })
            elif ntype == "task":
                active_tasks.append({
                    "node_id": node_id,
                    "label": label,
                    "status": meta.get("status", ""),
                    "desc": meta.get("goal", ""),
                    "activation": round(activation, 3),
                })
                if meta.get("task_graph_id") and not task_graph_id:
                    task_graph_id = meta["task_graph_id"]

        session_node = mg.get_node(session_node_id)
        session_label = session_node.label if session_node else ""

        logger.info(
            f"[BFSContext] session={session_node_id}, "
            f"activated_nodes={len(activations)}, "
            f"rounds={len(recent_rounds)}, tasks={len(active_tasks)}"
        )

        return {
            "session_node_id": session_node_id,
            "session_label": session_label,
            "recent_rounds": recent_rounds[:5],
            "active_tasks": active_tasks[:3],
            "task_graph_id": task_graph_id,
        }

    except Exception as e:
        logger.warning(f"[BFSContext] 检索失败 (best-effort): {e}")
        return None


async def request_ide_action(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Forward a non-chat IDE command to the active VS Code backend bridge."""
    task_graph_id = payload.get("task_graph_id") or payload.get("graph_id")
    requested_workspace = payload.get("workspace_path") or payload.get("cwd")
    workspace_path = _resolve_active_task_workspace(
        requested_workspace,
        task_graph_id,
    )
    if workspace_path:
        payload["workspace_path"] = workspace_path
        payload["cwd"] = workspace_path
    vscode_command = payload.get("vscode_command") or payload.get("vscode_path")
    if action in ("ide:execute_tool", "ide_execute_tool"):
        return await _request_ide_tool_execution(payload)
    if action in (MessageType.IDE_APPROVAL_RESULT, "ide_approval_result", "ide:approval_result"):
        return await _forward_ide_approval_result(payload)
    session = _select_ide_bridge(workspace_path)
    if workspace_path and session and not _workspace_matches(session, workspace_path):
        session = None
    logger.info(
        "[ZulongIDE] request_ide_action action=%s requested_workspace=%s resolved_workspace=%s task_graph_id=%s bridge=%s",
        action,
        requested_workspace,
        workspace_path,
        task_graph_id,
        getattr(session, "session_id", "")[:12] if session else "",
    )
    if not session:
        if action in (MessageType.IDE_OPEN_WORKSPACE, "ide_open_workspace", "ide:open_workspace"):
            if not workspace_path:
                return {
                    "ok": False,
                    "status": "workspace_required",
                    "error": "缺少当前任务工作目录，拒绝回退打开祖龙项目目录。",
                }
            launched = _launch_vscode_workspace(
                workspace_path,
                vscode_command,
                active_file=payload.get("active_file") or payload.get("file_path") or payload.get("path"),
                line=payload.get("line") or payload.get("start_line"),
                column=payload.get("column") or payload.get("start_column"),
            )
            if launched.get("ok"):
                return {
                    "ok": True,
                    "action": action,
                    "status": "launched_without_bridge",
                    "message": "未检测到 VS Code 后台桥，已尝试直接打开 VS Code 工作区。",
                    **launched,
                }
            return {
                "ok": False,
                "error": (
                    "未检测到 VS Code 后台桥连接，且无法通过本机 code 命令打开 VS Code："
                    + launched.get("error", "unknown")
                ),
                **launched,
            }
        return {"ok": False, "error": "未检测到 VS Code 后台桥连接"}
    await session.send_unified_msg(action, payload)
    return {"ok": True, "ide_session_id": session.session_id, "action": action}


def _normalize_approval_result_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(payload or {})
    action = str(normalized.get("action") or "").lower()
    if "approved" not in normalized:
        normalized["approved"] = action in {"approve", "approved", "confirm", "confirmed", "yes", "true"}
    approval_id = (
        normalized.get("approval_id")
        or normalized.get("approvalId")
        or normalized.get("interaction_id")
        or normalized.get("pair_id")
        or normalized.get("call_id")
    )
    if approval_id:
        normalized["approval_id"] = approval_id
        normalized.setdefault("pair_id", approval_id)
    normalized["action"] = "approve" if normalized.get("approved") else "reject"
    normalized.setdefault("status", "approved" if normalized.get("approved") else "rejected")
    return normalized


def _apply_approval_whitelist_rule(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not payload.get("approved"):
        return None
    rule = payload.get("add_to_whitelist") or payload.get("whitelist_rule")
    if not rule:
        return None
    try:
        from zulong.config.approval_config import add_approval_whitelist_rule

        result = add_approval_whitelist_rule(str(rule))
        if result:
            logger.info(
                "[ZulongIDE] 审批白名单已更新: kind=%s value=%s changed=%s",
                result.get("kind"),
                result.get("value"),
                result.get("changed"),
            )
        return result
    except Exception as exc:
        logger.warning("[ZulongIDE] 审批白名单更新失败: %s", exc)
        return {"error": str(exc), "rule": str(rule)}


async def _forward_ide_approval_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Forward a Web approval decision to the active VS Code bridge."""
    payload = _normalize_approval_result_payload(payload)
    workspace_path = payload.get("workspace_path") or payload.get("cwd")
    session_id = payload.get("ide_session_id") or payload.get("target_session_id")
    session = _sessions.get(session_id) if session_id else None
    if session is None:
        session = _select_ide_bridge(workspace_path)
    if not session:
        return {"ok": False, "error": "未检测到 VS Code 后台桥连接，无法发送审批结果"}
    _record_runtime_event(
        session,
        "approval_result",
        {
            **payload,
            "message": "用户已允许该操作" if payload.get("approved") else "用户已拒绝该操作",
            "interaction": {
                "interaction_id": f"approval_result:{payload.get('approval_id') or time.time()}",
                "pair_id": payload.get("pair_id") or payload.get("approval_id"),
                "kind": "approval",
                "status": "approved" if payload.get("approved") else "rejected",
                "title": "用户已允许" if payload.get("approved") else "用户已拒绝",
                "detail": payload.get("action_summary") or "",
                "tool_name": payload.get("tool_name") or "",
                "confirmation_state": "confirmed" if payload.get("approved") else "rejected",
            },
        },
        source="web_chat",
    )
    approved = bool(payload.get("approved"))
    _notify_web_task_execution_status(
        state="running" if approved else "blocked",
        phase="approval_resolved",
        message="审批已通过，任务继续执行。" if approved else "审批已拒绝，任务将按拒绝结果停止或重新规划。",
        request_id=session.web_turn_id,
        conversation_id=session.conversation_id,
        session_id=session.conversation_id,
        workspace_path=payload.get("workspace_path") or session.cwd,
        project_id=session.project_id,
        task_graph_id=session.task_graph_id,
        tool_name=payload.get("tool_name") or "",
        awaiting_approval=False,
        approval=payload,
        progress_items=[{
            "label": "审批已通过" if approved else "审批已拒绝",
            "status": "completed" if approved else "blocked",
            "source": "tool",
            "detail": payload.get("action_summary") or "",
            "timestamp": time.time(),
        }],
    )
    await session.send_unified_msg(MessageType.IDE_APPROVAL_RESULT, payload)
    return {
        "ok": True,
        "ide_session_id": session.session_id,
        "action": MessageType.IDE_APPROVAL_RESULT,
        "approval_id": payload.get("approval_id"),
    }


async def _request_ide_tool_execution(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one VS Code-side tool through the connected IDE bridge."""
    workspace_path = _resolve_active_task_workspace(
        payload.get("workspace_path") or payload.get("cwd"),
        payload.get("task_graph_id") or payload.get("graph_id"),
    )
    if workspace_path:
        payload["workspace_path"] = workspace_path
        payload["cwd"] = workspace_path
    ensured = await ensure_vscode_bridge(
        workspace_path,
        vscode_command=payload.get("vscode_command") or payload.get("vscode_path"),
        reason=payload.get("reason") or "L2 请求执行 IDE 工具",
        timeout=float(payload.get("bridge_timeout", 180) or 180),
    )
    session = ensured.get("session")
    if not ensured.get("ok") or not session:
        return {
            "ok": False,
            "error": ensured.get("error") or "未检测到 VS Code 后台桥连接，无法执行 IDE 工具",
            "bridge": {k: v for k, v in ensured.items() if k != "session"},
        }

    tool_name = payload.get("tool_name") or payload.get("name")
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("args") or {}
    if not tool_name:
        return {"ok": False, "error": "tool_name 不能为空"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "arguments 必须是对象"}

    call_id = f"ide_tool_{uuid.uuid4().hex[:12]}"
    await session.send_unified_msg(MessageType.TOOL_REQUEST, {
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }],
        "call_ids": [call_id],
        "tool_names": [tool_name],
        "source": payload.get("source", "web_l2_tool"),
    })

    timeout = float(payload.get("timeout", payload.get("bridge_timeout", 180)) or 180)
    _notify_web_task_execution_status(
        state="running",
        phase="tool_call",
        message=f"正在等待 VS Code 执行工具：{tool_name}",
        request_id=session.web_turn_id,
        conversation_id=session.conversation_id,
        session_id=session.conversation_id,
        workspace_path=session.cwd or workspace_path,
        project_id=session.project_id,
        task_graph_id=session.task_graph_id or payload.get("task_graph_id") or payload.get("graph_id"),
        tool_name=tool_name,
        awaiting_approval=False,
        progress_items=[{
            "label": f"VS Code 执行 {tool_name}",
            "status": "running",
            "source": "tool",
            "timestamp": time.time(),
        }],
    )
    deadline = time.time() + timeout
    skipped = []
    while time.time() < deadline:
        try:
            item = await asyncio.wait_for(
                session.tool_result_queue.get(),
                timeout=max(0.1, min(1.0, deadline - time.time())),
            )
        except asyncio.TimeoutError:
            continue
        if item.get("call_id") == call_id:
            for other in skipped:
                await session.tool_result_queue.put(other)
            is_error = bool(item.get("is_error"))
            _notify_web_task_execution_status(
                state="failed" if is_error else "running",
                phase="tool_result",
                message=f"VS Code 工具返回：{tool_name}" if not is_error else f"VS Code 工具失败：{tool_name}",
                request_id=session.web_turn_id,
                conversation_id=session.conversation_id,
                session_id=session.conversation_id,
                workspace_path=session.cwd or workspace_path,
                project_id=session.project_id,
                task_graph_id=session.task_graph_id or payload.get("task_graph_id") or payload.get("graph_id"),
                tool_name=tool_name,
                awaiting_approval=False,
                progress_items=[{
                    "label": f"VS Code 执行 {tool_name}",
                    "status": "failed" if is_error else "completed",
                    "source": "tool",
                    "timestamp": time.time(),
                }],
            )
            return {
                "ok": not is_error,
                "ide_session_id": session.session_id,
                "tool_name": tool_name,
                "call_id": call_id,
                "result": item.get("result", ""),
                "error": item.get("result", "") if is_error else None,
            }
        skipped.append(item)

    for other in skipped:
        await session.tool_result_queue.put(other)
    _notify_web_task_execution_status(
        state="blocked",
        phase="tool_timeout",
        message=f"等待 VS Code 工具结果超时（{timeout:.0f}s）：{tool_name}",
        request_id=session.web_turn_id,
        conversation_id=session.conversation_id,
        session_id=session.conversation_id,
        workspace_path=session.cwd or workspace_path,
        project_id=session.project_id,
        task_graph_id=session.task_graph_id or payload.get("task_graph_id") or payload.get("graph_id"),
        tool_name=tool_name,
        awaiting_approval=False,
        progress_items=[{
            "label": f"等待 {tool_name} 返回",
            "status": "blocked",
            "source": "tool",
            "detail": f"{timeout:.0f}s 超时",
            "timestamp": time.time(),
        }],
    )
    return {
        "ok": False,
        "ide_session_id": session.session_id,
        "tool_name": tool_name,
        "call_id": call_id,
        "error": f"等待 VS Code 工具结果超时（{timeout:.0f}s）",
    }


def _resolve_vscode_command(override: Optional[str] = None) -> Dict[str, Any]:
    """Resolve the VS Code command without hardcoding an install path."""
    try:
        from zulong.workspace.vscode_launcher import resolve_vscode_command

        return resolve_vscode_command(override)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _launch_vscode_workspace(
    workspace_path: str,
    vscode_command: Optional[str] = None,
    *,
    active_file: Optional[str] = None,
    line: Optional[Any] = None,
    column: Optional[Any] = None,
) -> Dict[str, Any]:
    """Open VS Code from the backend when the extension bridge is not connected."""
    try:
        from zulong.workspace.vscode_launcher import launch_vscode_workspace

        target = os.path.abspath(workspace_path) if workspace_path else ""
        if not target:
            return {
                "ok": False,
                "status": "workspace_required",
                "error": "缺少当前任务工作目录，拒绝回退打开祖龙项目目录。",
            }
        if not os.path.exists(target):
            return {
                "ok": False,
                "status": "workspace_not_found",
                "error": f"工作目录不存在: {target}",
            }

        file_target = ""
        if active_file:
            try:
                candidate = os.path.expanduser(os.path.expandvars(str(active_file)))
                if not os.path.isabs(candidate):
                    candidate = os.path.join(target, candidate)
                candidate = os.path.abspath(candidate)
                if os.path.commonpath([target, candidate]) == target and os.path.isfile(candidate):
                    file_target = candidate
            except Exception:
                file_target = ""

        logger.info(
            "[ZulongIDE] launch_vscode_workspace workspace=%s active_file=%s line=%s column=%s",
            target,
            file_target,
            line,
            column,
        )
        return launch_vscode_workspace(
            target,
            vscode_command,
            file_path=file_target or None,
            line=line,
            column=column,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── 消息处理 ──────────────────────────────────────────

async def _handle_session_start(session: IDESession, payload: Dict) -> None:
    """处理新任务请求，启动 FC 循环"""
    task_text = payload.get("task", "")
    cwd = payload.get("cwd", ".")
    ide_system_prompt = payload.get("ide_system_prompt", "")
    approval_mode = payload.get("approval_mode")
    conversation_id = payload.get("conversation_id") or payload.get("session_id")
    turn_id = payload.get("turn_id") or payload.get("request_id")
    if not task_text:
        await session.send_msg("task_error", {"error": "task 不能为空"})
        return

    if approval_mode:
        try:
            from zulong.config.approval_config import set_runtime_approval_mode

            approval_mode = set_runtime_approval_mode(str(approval_mode))
        except Exception as exc:
            logger.debug("[ZulongIDE] 审批模式更新失败，使用后端默认值: %s", exc)
    logger.info(f"[ZulongIDE] session_start: task={task_text[:100]}, cwd={cwd}, approval_mode={approval_mode or 'default'}")

    # 检测项目模式：从 ProjectRegistry 查找 cwd 绑定。旧版 .zulong/project.json
    # 仅作为兼容提示，不再要求用户工作区存在运行态元数据目录。
    _detected_project_id = None
    _detected_task_graph_id = None
    try:
        from zulong.workspace.project_registry import get_project_registry
        _registry = get_project_registry()
        _proj = _registry.get_project_by_path(cwd)
        if _proj:
            _detected_project_id = _proj.project_id
            _detected_task_graph_id = _proj.task_graph_id
            if _proj.status == "pending_execution":
                _registry.update_project_status(_proj.project_id, "executing")
                logger.info(f"[ZulongIDE] 项目 {_proj.project_id} 状态更新为 executing")
    except Exception as _e:
        logger.debug(f"[ZulongIDE] 项目状态更新跳过: {_e}")

    # 设置会话元数据（供 Web 监控和 REST API 使用）
    session.cwd = cwd
    session.project_id = _detected_project_id
    session.task_graph_id = _detected_task_graph_id
    session.conversation_id = conversation_id
    session.web_turn_id = turn_id

    # 如果有正在运行的 FC 循环，先取消
    if session.fc_task and not session.fc_task.done():
        session.cancel_event.set()
        try:
            await asyncio.wait_for(session.fc_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            session.fc_task.cancel()

    session.cancel_event.clear()

    # 在后台启动 FC 循环
    session.fc_task = asyncio.create_task(
        _run_fc_loop(session, task_text, cwd, ide_system_prompt))

    # Web 监控: IDE 会话启动（清洗 task_text，移除 <task> 标签和系统指令噪声）
    import re as _re
    _task_tag_match = _re.search(r"<task>\s*(.*?)\s*</task>", task_text, _re.DOTALL)
    if _task_tag_match:
        _clean_task = _task_tag_match.group(1).strip()
    else:
        _clean_task = _re.split(
            r"\n#\s*task_progress|<task_progress>|\n====", task_text
        )[0].strip()
    if not _clean_task:
        _clean_task = task_text.strip()

    await broadcast_monitor_event("IDE_SESSION_START", {
        "session_id": session.session_id,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "task_preview": _clean_task[:200],
        "task_title": _clean_task[:40],
        "cwd": cwd,
        "project_id": _detected_project_id,
        "task_graph_id": _detected_task_graph_id,
    })


async def _handle_tool_result(session: IDESession, payload: Dict) -> None:
    """处理插件返回的工具执行结果"""
    call_id = payload.get("call_id", "")
    tool_name = payload.get("tool_name", "")
    result = payload.get("result", "")
    is_error = payload.get("is_error", False)

    logger.info(
        f"[ZulongIDE] tool_result: call_id={call_id}, tool={tool_name}, "
        f"is_error={is_error}, len={len(result)}")

    await session.tool_result_queue.put({
        "call_id": call_id,
        "tool_name": tool_name,
        "result": result,
        "is_error": is_error,
    })


async def _handle_ide_context(session: IDESession, payload: Dict) -> None:
    session.cwd = payload.get("workspace_path") or payload.get("cwd") or session.cwd
    if "workspace_trusted" in payload:
        session.workspace_trusted = bool(payload.get("workspace_trusted"))
        session.trust_required = not session.workspace_trusted
        if session.workspace_trusted:
            _notify_workspace_trusted(session.cwd)
    elif "trust_required" in payload:
        session.trust_required = bool(payload.get("trust_required"))
        session.workspace_trusted = not session.trust_required
    await broadcast_monitor_event("IDE_CONTEXT", {
        "session_id": session.session_id,
        **_conversation_payload(session),
        **payload,
    })


async def _handle_ide_file_changed(session: IDESession, payload: Dict) -> None:
    await broadcast_monitor_event(MessageType.IDE_FILE_CHANGED, {
        "session_id": session.session_id,
        **_conversation_payload(session),
        **payload,
    })
    _record_runtime_event(session, MessageType.IDE_FILE_CHANGED, payload, source="ide_plugin")


async def _handle_ide_terminal_status(session: IDESession, payload: Dict) -> None:
    if "workspace_trusted" in payload:
        session.workspace_trusted = bool(payload.get("workspace_trusted"))
        session.trust_required = not session.workspace_trusted
        if session.workspace_trusted:
            _notify_workspace_trusted(payload.get("workspace_path") or session.cwd)
    elif "trust_required" in payload:
        session.trust_required = bool(payload.get("trust_required"))
        session.workspace_trusted = not session.trust_required
    await broadcast_monitor_event(MessageType.IDE_TERMINAL_STATUS, {
        "session_id": session.session_id,
        **_conversation_payload(session),
        **payload,
    })
    _record_runtime_event(session, MessageType.IDE_TERMINAL_STATUS, payload, source="ide_plugin")


def _approval_tool_args_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    args = payload.get("arguments") or payload.get("tool_args") or payload.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    for src_key, dst_key in (
        ("path", "path"),
        ("file_path", "path"),
        ("workspace_path", "path"),
        ("cwd", "path"),
        ("command", "command"),
    ):
        if payload.get(src_key) and not args.get(dst_key):
            args[dst_key] = payload.get(src_key)
    return args


def _should_auto_approve_ide_payload(payload: Dict[str, Any]) -> bool:
    try:
        from zulong.config.approval_config import should_runtime_auto_approve

        tool_name = (
            payload.get("tool_name")
            or payload.get("tool")
            or payload.get("name")
            or ""
        )
        return should_runtime_auto_approve(
            str(tool_name),
            _approval_tool_args_from_payload(payload),
            risk_level=str(payload.get("risk_level") or payload.get("risk") or ""),
        )
    except Exception as exc:
        logger.debug("[ZulongIDE] 自动审批判断失败: %s", exc)
        return False


async def _handle_ide_approval_status(session: IDESession, payload: Dict) -> None:
    await broadcast_monitor_event(MessageType.IDE_APPROVAL_STATUS, {
        "session_id": session.session_id,
        **_conversation_payload(session),
        **payload,
    })
    status = str(payload.get("status") or payload.get("confirmation_state") or "").lower()
    approved = payload.get("approved")
    is_resolved = status in {"approved", "rejected", "confirmed"} or approved is not None
    is_approved = status in {"approved", "confirmed"} or approved is True
    action_summary = payload.get("action_summary") or payload.get("summary") or payload.get("message") or "需要确认操作"
    if is_resolved:
        _notify_web_task_execution_status(
            state="running" if is_approved else "blocked",
            phase="approval_resolved",
            message="审批已通过，任务继续执行。" if is_approved else "审批已拒绝，任务将按拒绝结果停止或重新规划。",
            request_id=session.web_turn_id,
            conversation_id=session.conversation_id,
            session_id=session.conversation_id,
            workspace_path=payload.get("workspace_path") or session.cwd,
            project_id=session.project_id,
            task_graph_id=session.task_graph_id,
            tool_name=payload.get("tool_name") or "",
            awaiting_approval=False,
            approval=payload,
            progress_items=[{
                "label": "审批已通过" if is_approved else "审批已拒绝",
                "status": "completed" if is_approved else "blocked",
                "source": "tool",
                "detail": action_summary,
                "timestamp": time.time(),
            }],
        )
        _record_runtime_event(session, "approval_result", payload, source="ide_plugin")
        return
    if _should_auto_approve_ide_payload(payload):
        try:
            from zulong.config.approval_config import get_runtime_approval_mode
            approval_mode = get_runtime_approval_mode()
        except Exception:
            approval_mode = "full_auto"
        normalized = _normalize_approval_result_payload({
            **payload,
            "approved": True,
            "status": "approved",
            "workspace_path": payload.get("workspace_path") or session.cwd,
            "cwd": payload.get("cwd") or session.cwd,
            "ide_session_id": session.session_id,
        })
        logger.info(
            "[ZulongIDE] 自动审批通过: tool=%s approval_id=%s workspace=%s",
            normalized.get("tool_name") or payload.get("tool_name"),
            normalized.get("approval_id"),
            normalized.get("workspace_path"),
        )
        await _forward_ide_approval_result(normalized)
        _record_runtime_event(
            session,
            "approval_auto_approved",
            {
                **payload,
                "approved": True,
                "status": "approved",
                "message": "已按 Web 审批模式自动允许该操作",
                "interaction": {
                    "interaction_id": f"approval_auto:{normalized.get('approval_id') or time.time()}",
                    "pair_id": normalized.get("approval_id") or payload.get("approval_id"),
                    "kind": "approval",
                    "status": "approved",
                    "title": "自动审批已允许",
                    "detail": action_summary,
                    "tool_name": payload.get("tool_name") or "",
                    "confirmation_state": "confirmed",
                    "approval_mode": approval_mode,
                },
            },
            source="ide_server",
        )
        return
    _notify_web_task_execution_status(
        state="waiting_approval",
        phase="approval_required",
        message=f"任务已暂停，等待用户审批：{action_summary}",
        request_id=session.web_turn_id,
        conversation_id=session.conversation_id,
        session_id=session.conversation_id,
        workspace_path=payload.get("workspace_path") or session.cwd,
        project_id=session.project_id,
        task_graph_id=session.task_graph_id,
        tool_name=payload.get("tool_name") or "",
        awaiting_approval=True,
        approval=payload,
        progress_items=[{
            "label": "等待用户审批",
            "status": "running",
            "source": "tool",
            "detail": action_summary,
            "timestamp": time.time(),
        }],
    )
    _record_runtime_event(session, "approval_required", payload, source="ide_plugin")


async def _handle_ide_approval_result(session: IDESession, payload: Dict) -> None:
    normalized = _normalize_approval_result_payload(payload)
    normalized.setdefault("workspace_path", session.cwd)
    normalized.setdefault("cwd", session.cwd)
    whitelist_update = _apply_approval_whitelist_rule(normalized)
    if whitelist_update is not None:
        normalized["whitelist_update"] = whitelist_update
    await session.tool_result_queue.put({
        **normalized,
        "type": "approval_result",
        "result": "用户已允许该操作" if normalized.get("approved") else "用户已拒绝该操作",
        "is_error": not bool(normalized.get("approved")),
    })
    await session.send_unified_msg(MessageType.APPROVAL_RESULT, {
        **normalized,
        "message": "用户已允许该操作" if normalized.get("approved") else "用户已拒绝该操作",
        "interaction": {
            "interaction_id": f"approval_result:{normalized.get('approval_id') or time.time()}",
            "pair_id": normalized.get("pair_id") or normalized.get("approval_id"),
            "approval_id": normalized.get("approval_id"),
            "kind": "approval",
            "status": "approved" if normalized.get("approved") else "rejected",
            "title": "审批已通过" if normalized.get("approved") else "审批已拒绝",
            "detail": normalized.get("action_summary") or "",
            "tool_name": normalized.get("tool_name") or "",
            "risk_level": normalized.get("risk_level") or "",
            "confirmation_state": "approved" if normalized.get("approved") else "rejected",
        },
    })
    await _forward_ide_approval_result(normalized)


async def _handle_ide_diff_status(session: IDESession, payload: Dict) -> None:
    await broadcast_monitor_event(MessageType.IDE_DIFF_STATUS, {
        "session_id": session.session_id,
        **_conversation_payload(session),
        **payload,
    })
    _record_runtime_event(session, "diff_ready", payload, source="ide_plugin")


async def _handle_ide_checkpoint_status(session: IDESession, payload: Dict) -> None:
    await broadcast_monitor_event(MessageType.IDE_CHECKPOINT_STATUS, {
        "session_id": session.session_id,
        **_conversation_payload(session),
        **payload,
    })
    _record_runtime_event(session, "checkpoint_created", payload, source="ide_plugin")


def _record_runtime_event(
    session: IDESession,
    event_type: str,
    payload: Dict[str, Any],
    *,
    source: str = "ide_server",
) -> None:
    """Persist user-visible execution events without affecting the hot path."""
    try:
        from zulong.launcher.interaction_store import get_interaction_store
        from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph
        text = payload.get("message") or payload.get("action_summary") or payload.get("summary") or event_type
        conversation_id = session.conversation_id
        if not conversation_id:
            try:
                active = get_interaction_store().find_active_conversation(max_age_seconds=3600)
                conversation_id = active.get("conversation_id") if active else None
            except Exception:
                conversation_id = None
        event_id = get_interaction_store().append_event(
            conversation_id=conversation_id,
            turn_id=session.web_turn_id,
            event_type=event_type,
            role="system",
            source=source,
            text=text,
            payload={
                "session_id": session.session_id,
                **_conversation_payload(session),
                **payload,
            },
            workspace_path=session.cwd,
            project_id=session.project_id,
            task_graph_id=session.task_graph_id,
        )
        mirror_interaction_to_memory_graph(
            conversation_id=conversation_id,
            turn_id=session.web_turn_id,
            role="system",
            text=text,
            event_type=event_type,
            source=source,
            payload={"session_id": session.session_id, "source_event_id": event_id, **payload},
        )
        try:
            from zulong.review.task_execution_extractor import maybe_finalize_task_execution_trace
            maybe_finalize_task_execution_trace(
                conversation_id=conversation_id,
                turn_id=session.web_turn_id,
                task_graph_id=session.task_graph_id,
                event_type=event_type,
            )
        except Exception:
            pass
    except Exception as exc:
        logger.debug(f"[ZulongIDE] runtime event 记录跳过: {exc}")


async def _handle_user_cancel(session: IDESession, _payload: Dict) -> None:
    """处理用户取消"""
    logger.info(f"[ZulongIDE] user_cancel: session={session.session_id[:12]}")
    session.cancel_event.set()
    # 同步设置 engine._interrupt_flag，确保所有取消检查路径都生效
    try:
        engine = _get_engine()
        if engine:
            engine._interrupt_flag = True
            logger.info("[ZulongIDE] user_cancel: engine._interrupt_flag 已设置")
    except Exception as e:
        logger.warning(f"[ZulongIDE] user_cancel: 设置 interrupt_flag 失败: {e}")
    # 广播取消状态到Web端
    await broadcast_monitor_event("IDE_SESSION_CANCEL", {
        "session_id": session.session_id,
        **_conversation_payload(session),
    })


async def _handle_handshake(session: IDESession, payload: Dict) -> None:
    """统一协议握手: 记录客户端类型和 API 版本"""
    client_type = payload.get("client_type", "unknown")
    api_version = payload.get("api_version", "1.0")
    session.client_type = client_type
    session.api_version = api_version
    logger.info(f"[ZulongIDE] 握手完成: client={client_type}, api={api_version}, session={session.session_id[:12]}")
    # 返回握手确认，包含服务端支持的协议版本；保留 legacy handshake_ack，
    # 同时给统一客户端补发 task:ack，便于新版只监听 namespace 类型。
    ack_payload = {
        "server_version": "2.0",
        "supported_types": list(_MESSAGE_HANDLERS.keys()),
        "supported_unified_types": [
            MessageType.TASK_START,
            MessageType.TASK_RESUME,
            MessageType.TASK_CANCEL,
            MessageType.TOOL_RESULT,
            MessageType.PING,
            MessageType.AUDIO_START,
            MessageType.AUDIO_CHUNK,
            MessageType.AUDIO_END,
        ],
    }
    await session.ws.send_json({
        "msg_id": uuid.uuid4().hex[:12],
        "type": "handshake_ack",
        "session_id": session.session_id,
        "ts": time.time(),
        "payload": ack_payload,
    })
    if api_version.startswith("2"):
        await session.ws.send_json(make_unified_message(
            MessageType.TASK_ACK,
            {"session_id": session.session_id, **ack_payload},
            session_id=session.session_id,
        ))
    if client_type == "ide_plugin":
        await _send_runtime_settings_to_session(session)


async def _handle_ping(session: IDESession, _payload: Dict) -> None:
    """处理心跳ping消息，立即返回pong"""
    pong = make_unified_message(
        MessageType.PONG if session.api_version.startswith("2") else "pong",
        {},
        session_id=session.session_id,
    )
    await session.ws.send_json(pong)


async def _handle_audio_start(session: IDESession, _payload: Dict) -> None:
    """处理音频流开始"""
    from zulong.ide.audio_handler import handle_audio_start
    result = await handle_audio_start(session.session_id)
    if result:
        msg = {
            "msg_id": uuid.uuid4().hex[:12],
            "type": "audio_start_ack",
            "session_id": session.session_id,
            "ts": time.time(),
            "payload": result,
        }
        await session.ws.send_json(msg)


async def _handle_audio_chunk(session: IDESession, payload: Dict) -> None:
    """处理音频块,执行实时转录"""
    from zulong.ide.audio_handler import handle_audio_chunk
    result = await handle_audio_chunk(session.session_id, payload)
    if result and result.get("text"):
        msg = {
            "msg_id": uuid.uuid4().hex[:12],
            "type": "audio_transcript",
            "session_id": session.session_id,
            "ts": time.time(),
            "payload": result,
        }
        await session.ws.send_json(msg)


async def _handle_audio_end(session: IDESession, _payload: Dict) -> None:
    """处理音频流结束,返回最终转录结果"""
    from zulong.ide.audio_handler import handle_audio_end
    result = await handle_audio_end(session.session_id)
    if result:
        msg = {
            "msg_id": uuid.uuid4().hex[:12],
            "type": "audio_transcript",
            "session_id": session.session_id,
            "ts": time.time(),
            "payload": result,
        }
        await session.ws.send_json(msg)


def _load_graph_deterministic(graph_id: str, workspace_dir: str = None) -> bool:
    """确定性三级加载 TaskGraph: 内存 → 磁盘 → MemoryGraph

    TSD §23.11.3: 恢复时传入 workspace_dir，确保工作目录一致性。

    Returns: True 表示加载成功并已设置为活跃图
    """
    try:
        from zulong.tools.task_tools import (
            load_task_graph_deterministic,
            normalize_task_graph_id,
        )

        graph_id = normalize_task_graph_id(graph_id)
        return load_task_graph_deterministic(graph_id, workspace_dir=workspace_dir)
    except Exception as e:
        logger.warning(f"[ZulongIDE] 确定性恢复失败: {graph_id}: {e}")
        return False


async def _handle_session_resume(session: IDESession, payload: Dict) -> None:
    """处理会话恢复（插件重连后继续未完成的任务）"""
    task_text = payload.get("task", "")
    cwd = payload.get("cwd", ".")
    ide_system_prompt = payload.get("ide_system_prompt", "")
    try:
        from zulong.tools.task_tools import normalize_task_graph_id
        graph_id = normalize_task_graph_id(payload.get("graph_id") or payload.get("task_graph_id"))
    except Exception:
        graph_id = payload.get("graph_id", "") or payload.get("task_graph_id", "")
    session.conversation_id = payload.get("conversation_id") or payload.get("session_id") or session.conversation_id
    session.web_turn_id = payload.get("turn_id") or payload.get("request_id") or session.web_turn_id

    if not task_text:
        await session.send_msg("task_error", {"error": "resume task 不能为空"})
        return

    logger.info(f"[ZulongIDE] session_resume: task={task_text[:100]}")

    # 通知 Web 前端：任务恢复（在已有会话中继续，而非新建窗口）
    try:
        from zulong.tools.task_tools import _active_graph_id
        await broadcast_monitor_event("IDE_SESSION_START", {
            "session_id": session.session_id,
            "conversation_id": session.conversation_id,
            "turn_id": session.web_turn_id,
            "task_preview": task_text[:200],
            "task_title": ("恢复: " + task_text[:33]) if len(task_text) > 33 else ("恢复: " + task_text),
            "cwd": cwd,
            "project_id": None,
            "task_graph_id": graph_id or _active_graph_id,
            "is_resume": True,
        })
    except Exception:
        pass

    # 如果有正在运行的 FC 循环，先取消
    if session.fc_task and not session.fc_task.done():
        session.cancel_event.set()
        try:
            await asyncio.wait_for(session.fc_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            session.fc_task.cancel()

    session.cancel_event.clear()
    session.cwd = cwd
    session.task_graph_id = graph_id or session.task_graph_id

    # 恢复活跃 TaskGraph (TSD §23.11.3: 传入 workspace_dir 保持工作目录一致)
    if graph_id:
        # 确定性恢复路径
        _load_graph_deterministic(graph_id, workspace_dir=cwd)
    else:
        message = (
            "已收到显式恢复请求，但没有明确的 task_graph_id。"
            "为避免把旧任务误接到当前对话，请先选择要恢复的旧任务图，"
            "或明确回复「删除旧任务并重新创建」。"
        )
        logger.warning("[ZulongIDE] session_resume 拒绝无 graph_id 的自动恢复")
        _notify_web_task_execution_status(
            state="blocked",
            phase="resume_needs_task_graph_id",
            message=message,
            request_id=session.web_turn_id,
            conversation_id=session.conversation_id,
            session_id=session.conversation_id,
            workspace_path=cwd,
            task_graph_id="",
            awaiting_approval=False,
            progress_items=[{
                "label": "恢复需要明确任务图",
                "status": "blocked",
                "source": "task_graph",
                "detail": message,
                "timestamp": time.time(),
            }],
        )
        await session.send_msg("task_error", {"error": message})
        return

    # 恢复模式：通过 force_graph_id 触发任务图复用策略
    resume_text = f"继续之前的任务：{task_text}"
    session.fc_task = asyncio.create_task(
        _run_fc_loop(session, resume_text, cwd, ide_system_prompt,
                     force_graph_id=graph_id))


# 消息路由表
_MESSAGE_HANDLERS = {
    "handshake": _handle_handshake,
    "session_start": _handle_session_start,
    "session_resume": _handle_session_resume,
    "tool_result": _handle_tool_result,
    "ide:context": _handle_ide_context,
    "ide:file_changed": _handle_ide_file_changed,
    "ide:terminal_status": _handle_ide_terminal_status,
    "ide:approval_status": _handle_ide_approval_status,
    MessageType.IDE_APPROVAL_RESULT: _handle_ide_approval_result,
    "ide:diff_status": _handle_ide_diff_status,
    "ide:checkpoint_status": _handle_ide_checkpoint_status,
    "user_cancel": _handle_user_cancel,
    "ping": _handle_ping,
    "audio_start": _handle_audio_start,
    "audio_chunk": _handle_audio_chunk,
    "audio_end": _handle_audio_end,
}


def _normalize_inbound_message(msg: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """将 /ide 新旧消息统一成内部旧 handler 名称。"""
    fmt = _protocol_bridge.detect_format(msg)
    unified = _protocol_bridge.to_unified(msg, fmt)
    msg_type = unified.get("type", "")
    if msg_type in (
        MessageType.IDE_CONTEXT,
        MessageType.IDE_FILE_CHANGED,
            MessageType.IDE_TERMINAL_STATUS,
            MessageType.IDE_APPROVAL_STATUS,
            MessageType.IDE_APPROVAL_RESULT,
            MessageType.IDE_DIFF_STATUS,
            MessageType.IDE_CHECKPOINT_STATUS,
    ):
        return msg_type, unified.get("payload", {}) if isinstance(unified.get("payload", {}), dict) else {}
    legacy_type = _protocol_bridge.unified_to_legacy_type(msg_type, target_format="ide")
    payload = unified.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    return legacy_type, payload


# ── Web 监控: TaskGraph / MemoryGraph 回调注入 ─────────

def _task_graph_change_callback(event_type: str, data: dict) -> None:
    """TaskGraph on_change_callback → 广播完整图谱快照到 Web 监控"""
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg:
            payload = {
                "event": event_type,
                "detail": data,
                "graph": tg.to_frontend_dict(),
            }
            _broadcast_sync("TASK_GRAPH_UPDATE", payload)
    except Exception:
        pass


_main_event_loop: Optional[asyncio.AbstractEventLoop] = None


def _broadcast_sync(event_type: str, payload: dict) -> None:
    """在同步上下文中安排 broadcast_monitor_event（fire-and-forget）

    修复：使用 asyncio.run_coroutine_threadsafe 从工作线程安全地调度到主事件循环。
    _exec_internal 通过 run_in_executor 在工作线程中执行，直接使用
    asyncio.get_event_loop() 会获取到错误的循环，导致 WebSocket 发送失败。
    """
    try:
        loop = _main_event_loop
        if loop is None:
            # 回退：尝试获取当前线程的 loop（仅在主线程有效）
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
        if loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(broadcast_monitor_event(event_type, payload), loop)
    except Exception:
        pass


def _inject_task_graph_monitor_callback() -> None:
    """将 Web 广播回调注入到当前活跃 TaskGraph（如果存在）"""
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg and not tg.on_change_callback:
            tg.on_change_callback = _task_graph_change_callback
            logger.info("[ZulongIDE] TaskGraph Web 监控回调已注入")
    except Exception:
        pass


def _inject_memory_graph_monitor_hook() -> None:
    """为 MemoryGraph 注入 Web 广播钩子（覆盖 _pending_changes 刷新）"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg and not getattr(mg, "_web_monitor_hooked", False):
            mg._web_monitor_hooked = True
            original_mark_dirty = mg._mark_dirty

            def _hooked_mark_dirty():
                original_mark_dirty()
                # 将 pending_changes 广播到 Web 监控
                if mg._pending_changes:
                    changes = list(mg._pending_changes)
                    _broadcast_sync("MEMORY_GRAPH_UPDATE", {
                        "changes": changes,
                        "stats": {
                            "total_nodes": mg._stats.get("total_nodes", 0),
                            "total_edges": mg._stats.get("total_edges", 0),
                        },
                    })

            mg._mark_dirty = _hooked_mark_dirty
            logger.info("[ZulongIDE] MemoryGraph Web 监控钩子已注入")
    except Exception:
        pass


def _inject_code_anchor_monitor_hook() -> None:
    """为 CodeAnchorStore 注入 Web 广播钩子"""
    try:
        from zulong.memory.code_anchor import get_code_anchor_store
        store = get_code_anchor_store()
        if store and not getattr(store, "_web_monitor_hooked", False):
            store._web_monitor_hooked = True
            original_mark_dirty = store._mark_dirty

            def _hooked_mark_dirty():
                original_mark_dirty()
                # 将 pending_changes 广播到 Web 监控
                if store._pending_changes:
                    changes = list(store._pending_changes)
                    _broadcast_sync("CODE_ANCHOR_UPDATE", {
                        "changes": changes,
                        "stats": store.get_stats(),
                    })

            store._mark_dirty = _hooked_mark_dirty
            logger.info("[ZulongIDE] CodeAnchorStore Web 监控钩子已注入")
    except Exception:
        pass


# ── FC 循环（核心） ───────────────────────────────────

async def _run_fc_loop(
    session: IDESession, task_text: str, cwd: str,
    ide_system_prompt: str = "",
    force_graph_id: str = "",
) -> None:
    """在后台运行祖龙 FC 循环

    通过 session.outbound_queue 向插件发送消息,
    通过 session.tool_result_queue 接收工具执行结果。
    """
    try:
        await session.send_msg("status_update", {"turn": 0, "phase": "initializing"})

        # 懒加载引擎（首次调用时初始化）
        engine = _get_engine()
        if engine is None:
            await session.send_msg("task_error", {
                "error": "InferenceEngine 未初始化，请检查 zulong 配置"})
            return

        from zulong.ide.ide_session import AgentSession, IDEFCState
        from zulong.ide.ide_tool_registry import IDEToolRegistry

        # 创建 per-session 实例
        ide_session = AgentSession(
            session_id=session.session_id,
            created_at=time.time(),
            last_accessed=time.time(),
            request_count=1,
        )
        tool_registry = IDEToolRegistry(engine.tool_engine)
        from zulong.ide.ide_fc_runner import IDEFCRunner
        runner = IDEFCRunner(engine, ide_session, tool_registry)
        runner.cwd = cwd  # 保存工作目录，供 CRG 自动锚定读取文件
        runner.force_graph_id = force_graph_id  # 确定性恢复锚点
        runner.ide_session = session  # WS层IDESession引用，供 _notify_session_linked 使用
        session.runner = runner

        # 注入 TaskGraph Web 广播回调
        _inject_task_graph_monitor_callback()

        # 注入 MemoryGraph Web 监控钩子
        _inject_memory_graph_monitor_hook()

        # 注入 CodeAnchorStore Web 监控钩子
        _inject_code_anchor_monitor_hook()

        # 构建初始消息（支持插件传入的 IDE 系统提示词）
        messages = _build_initial_messages(
            engine, task_text, cwd, ide_system_prompt)

        await session.send_msg("status_update", {"turn": 0, "phase": "running"})

        # 启动心跳：每30秒发送 status_update 确保前端显示任务仍在运行
        async def _heartbeat_status():
            while True:
                await asyncio.sleep(30)
                try:
                    await session.send_msg("status_update", {"turn": 0, "phase": "running"})
                except Exception:
                    break

        heartbeat_task = asyncio.create_task(_heartbeat_status())

        try:
            from zulong.core.state_manager import state_manager
            state_manager.touch_activity()
        except Exception:
            pass

        # 运行异步 FC 循环
        result = await runner.run_loop_async(
            messages=messages,
            send_callback=session.send_msg,
            tool_result_queue=session.tool_result_queue,
            cancel_event=session.cancel_event,
        )

        # 取消心跳
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

        # FC 循环完成 — 根据终止原因发送 task_complete 或 task_error
        _reason = getattr(result, "reason", None) or "done"
        _completion_result = (result.text_response if result and result.text_response
                              else "(任务完成，无输出)")

        # CB相关reason不应视为错误（对齐TSD §23.3.6: blocked→progress归一化）
        _cb_reasons = {"cb_pattern_loop", "cb_context_pressure", "cb_no_progress", "cb_consecutive_yellow"}
        _is_error = _reason not in ("done", None) and _reason not in _cb_reasons
        _msg_type = "task_error" if _is_error else "task_complete"
        _msg_payload = ({"error": _completion_result} if _is_error
                        else {"result": _completion_result, "status": "succeeded", "phase": "succeeded"})

        # 发送任务完成状态更新（对齐TSD phase值）
        if _reason in _cb_reasons:
            await session.send_msg("status_update", {"turn": 0, "phase": "blocked"})
        elif _is_error:
            await session.send_msg("status_update", {"turn": 0, "phase": "failed"})
        else:
            await session.send_msg("status_update", {"turn": 0, "phase": "succeeded"})

        # 等待短暂时间确保 display_text 被前端消费，避免 WebSocket 提前断开
        await asyncio.sleep(0.5)

        try:
            await session.send_msg(_msg_type, _msg_payload)
        except Exception:
            pass

        # Web 监控: IDE 会话结束
        _run = _find_run_by_ide_session(session.session_id)
        if _run:
            _run.status = "completed"
        await broadcast_monitor_event("IDE_SESSION_END", {
            "session_id": session.session_id,
            "status": "completed",
            **_conversation_payload(session),
        })

    except asyncio.CancelledError:
        logger.info(f"[ZulongIDE] FC 循环被取消: session={session.session_id[:12]}")
        # 取消心跳
        if 'heartbeat_task' in locals():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        try:
            await session.send_msg("task_error", {"error": "任务已取消"})
        except Exception:
            pass
        try:
            await session.ws.send_json({
                "type": "task_error",
                "session_id": session.session_id,
                "payload": {"error": "任务已取消"},
            })
        except Exception:
            pass
        _run = _find_run_by_ide_session(session.session_id)
        if _run:
            _run.status = "cancelled"
        await broadcast_monitor_event("IDE_SESSION_END", {
            "session_id": session.session_id,
            "status": "cancelled",
            **_conversation_payload(session),
        })
    except Exception as e:
        logger.error(f"[ZulongIDE] FC 循环异常: {e}", exc_info=True)
        error_msg = str(e)[:500]
        try:
            await session.send_msg("task_error", {"error": error_msg})
        except Exception:
            pass
        try:
            await session.ws.send_json({
                "type": "task_error",
                "session_id": session.session_id,
                "payload": {"error": error_msg},
            })
        except Exception:
            pass
        _run = _find_run_by_ide_session(session.session_id)
        if _run:
            _run.status = "error"
        await broadcast_monitor_event("IDE_SESSION_END", {
            "session_id": session.session_id,
            "status": "error",
            "error": error_msg[:200],
            **_conversation_payload(session),
        })


def _build_initial_messages(
    engine, task_text: str, cwd: str,
    ide_system_prompt: str = "",
) -> list:
    """构建 FC 循环初始消息列表

    Args:
        engine: InferenceEngine 实例
        task_text: 用户任务文本
        cwd: 工作目录
        ide_system_prompt: 插件传来的 IDE 系统提示词（包含环境上下文和 XML 工具定义）。
            后端会剥离 XML 工具定义区域，保留环境上下文，注入祖龙增强内容。
    """
    system_prompt = ""

    if ide_system_prompt:
        # 插件传来了完整系统提示词
        # 使用 IDEPromptHandler 剥离 XML 工具定义 + 注入祖龙增强
        try:
            from zulong.ide.ide_prompt_handler import IDEPromptHandler
            handler = IDEPromptHandler()

            # 获取记忆/任务上下文（用于注入增强内容）
            memory_ctx = ""
            task_ctx = ""
            experience_hints = ""
            try:
                mg = None
                try:
                    from zulong.memory.memory_graph import get_memory_graph
                    mg = get_memory_graph()
                except Exception:
                    pass
                if mg:
                    from zulong.memory.graph_adapters import TaskGraphAdapter
                    tga = TaskGraphAdapter()
                    task_ctx = tga.get_active_task_summary(mg) or ""
            except Exception as ctx_err:
                logger.debug(f"[ZulongIDE] 上下文获取跳过: {ctx_err}")

            raw_messages = [
                {"role": "system", "content": ide_system_prompt},
                {"role": "user", "content": task_text},
            ]
            processed = handler.process_system_prompt(
                raw_messages,
                memory_context=memory_ctx,
                task_context=task_ctx,
                experience_hints=experience_hints,
                cwd=cwd,
            )
            logger.info(
                f"[ZulongIDE] 使用插件系统提示词并增强 "
                f"(原始={len(ide_system_prompt)}, "
                f"处理后={len(processed[0].get('content', ''))})")
            return processed

        except Exception as e:
            logger.warning(
                f"[ZulongIDE] IDEPromptHandler 处理失败，直接使用原始: {e}")
            system_prompt = ide_system_prompt
    else:
        # 兜底：后端自行构建（无插件系统提示词时）
        try:
            from zulong.ide.ide_prompt_handler import IDEPromptHandler
            handler = IDEPromptHandler()
            base_prompt = f"你是祖龙智能编程助手。当前工作目录: {cwd}"
            processed = handler.process_system_prompt(
                [{"role": "system", "content": base_prompt}], cwd=cwd)
            system_prompt = processed[0].get("content", base_prompt)
        except Exception as e:
            logger.warning(f"[ZulongIDE] 系统提示词构建失败: {e}")
            system_prompt = "你是祖龙智能编程助手。请帮助用户完成编程任务。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
    ]
    return messages


# ── 引擎单例 ──────────────────────────────────────────

_engine_instance = None
_engine_lock = asyncio.Lock()
_preload_manager = None


def _get_engine():
    """获取 InferenceEngine 单例（同步版本，首次初始化在启动时完成）"""
    global _engine_instance
    return _engine_instance


async def _init_engine():
    """初始化 InferenceEngine"""
    global _engine_instance
    async with _engine_lock:
        if _engine_instance is not None:
            return
        try:
            logger.info("[ZulongIDE] 正在初始化 InferenceEngine...")
            from zulong.l2.inference_engine import InferenceEngine
            _engine_instance = InferenceEngine()
            logger.info("[ZulongIDE] InferenceEngine 初始化完成")
        except Exception as e:
            logger.error(f"[ZulongIDE] InferenceEngine 初始化失败: {e}", exc_info=True)


# ── WebSocket 端点 ────────────────────────────────────

@ide_router.websocket("/ide")
async def websocket_endpoint(ws: WebSocket):
    await _serve_ide_connection(ws)


async def _serve_ide_connection(
    ws: WebSocket,
    *,
    accepted: bool = False,
    initial_msg: Optional[Dict[str, Any]] = None,
) -> None:
    """服务一个 IDE WebSocket 连接。

    /ide 直接调用；统一根入口在读取 handshake 后也复用这里。
    """
    global _main_event_loop
    if _main_event_loop is None:
        _main_event_loop = asyncio.get_running_loop()
    if not accepted:
        await ws.accept()
    session_id = uuid.uuid4().hex
    session = IDESession(session_id, ws)
    _sessions[session_id] = session
    logger.info(f"[ZulongIDE] WebSocket 已连接: session={session_id[:12]}")

    # 发送 session_ack (包含后端模型上下文窗口信息)
    _context_window_size = 131072
    try:
        from zulong.l2.inference_engine import InferenceEngine
        _engine = InferenceEngine.get_instance()
        if _engine and getattr(_engine, "_context_window_size", 0) > 0:
            _context_window_size = _engine._context_window_size
    except Exception:
        pass
    
    # 🔥 检查预加载就绪状态
    if _preload_manager and not _preload_manager.is_ready():
        ack = make_unified_message(
            "session_ack",
            {
                "session_id": session_id,
                "context_window_size": _context_window_size,
                "system_status": "booting",
            },
            session_id=session_id,
        )
        await ws.send_json(ack)
        await ws.send_json(make_unified_message(
            "task_error",
            {"error": "系统正在启动中，请稍候重试"},
            session_id=session_id,
        ))
        _sessions.pop(session_id, None)
        return
    
    ack = make_unified_message(
        "session_ack",
        {
            "session_id": session_id,
            "context_window_size": _context_window_size,
        },
        session_id=session_id,
    )
    await ws.send_json(ack)
    
    # 恢复缓存消息（断线重连场景）
    pending_msgs = PendingMessageCache.pop(session_id)
    for pending_msg in pending_msgs:
        try:
            await ws.send_json(pending_msg)
            logger.info(f"[ZulongIDE] 恢复缓存消息: type={pending_msg.get('type')}")
        except Exception as e:
            logger.warning(f"[ZulongIDE] 恢复消息失败: {e}")
            break

    # 启动出站消息发送协程
    sender_task = asyncio.create_task(_outbound_sender(session))

    try:
        if initial_msg:
            msg_type, payload = _normalize_inbound_message(initial_msg)
            handler = _MESSAGE_HANDLERS.get(msg_type)
            if handler:
                await handler(session, payload)
            else:
                logger.warning(f"[ZulongIDE] 未知初始消息类型: {msg_type}")
        while True:
            raw = await ws.receive_text()
            try:
                from zulong.core.state_manager import state_manager
                state_manager.touch_activity()
            except Exception:
                pass
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"[ZulongIDE] 无效 JSON: {raw[:200]}")
                continue

            msg_type, payload = _normalize_inbound_message(msg)

            handler = _MESSAGE_HANDLERS.get(msg_type)
            if handler:
                await handler(session, payload)
            else:
                logger.warning(f"[ZulongIDE] 未知消息类型: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[ZulongIDE] WebSocket 断开: session={session_id[:12]}")
        # 🔥 关键修复：WS断开时，强制等待FC任务完成并发送task_complete
        if session.fc_task and not session.fc_task.done():
            logger.info(f"[ZulongIDE] 等待FC任务完成（最多5秒）...")
            try:
                await asyncio.wait_for(session.fc_task, timeout=5.0)
                logger.info(f"[ZulongIDE] FC任务已完成")
            except asyncio.TimeoutError:
                logger.warning(f"[ZulongIDE] FC任务超时，强制发送task_complete")
                # 超时后强制发送task_complete，确保前端不卡死
                try:
                    await session.ws.send_json({
                        "type": "task_complete",
                        "session_id": session_id,
                        "payload": {"result": "[任务超时终止]"},
                    })
                    logger.info(f"[ZulongIDE] 已发送超时task_complete")
                except Exception as e:
                    logger.warning(f"[ZulongIDE] 发送超时task_complete失败: {e}")
            except asyncio.CancelledError:
                logger.info(f"[ZulongIDE] FC任务被取消")
    except Exception as e:
        logger.error(f"[ZulongIDE] WebSocket 异常: {e}", exc_info=True)
    finally:
        # 清理
        sender_task.cancel()
        if session.fc_task and not session.fc_task.done():
            session.cancel_event.set()
            session.fc_task.cancel()
        _sessions.pop(session_id, None)
        logger.info(f"[ZulongIDE] 会话清理完成: session={session_id[:12]}")


@ide_router.websocket("/")
async def unified_websocket_endpoint(ws: WebSocket):
    """统一 WebSocket 入口。

    新客户端连接 ws://127.0.0.1:8090/ 后先发送 handshake:
    client_type=ide_plugin 走 IDE FC 通道；client_type=dashboard/monitor
    转交 Web/监控语义。旧 /ide 与 /ws 端点仍保留。
    """
    await ws.accept()
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10.0)
        try:
            first_msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json(make_unified_message(
                MessageType.TASK_ERROR,
                {"error": "首条消息必须是 JSON handshake"},
            ))
            await ws.close()
            return

        fmt = _protocol_bridge.detect_format(first_msg)
        unified = _protocol_bridge.to_unified(first_msg, fmt)
        payload = unified.get("payload", {}) if isinstance(unified, dict) else {}
        client_type = payload.get("client_type", "ide_plugin")
        if unified.get("type") != MessageType.HANDSHAKE:
            await ws.send_json(make_unified_message(
                MessageType.TASK_ERROR,
                {"error": "统一入口首条消息必须是 handshake"},
            ))
            await ws.close()
            return

        if client_type in ("dashboard", "monitor"):
            try:
                from zulong.launcher.web_chat_router import handle_unified_root_ws
                await handle_unified_root_ws(ws, accepted=True, initial_msg=unified)
                return
            except ImportError:
                await ws.send_json(make_unified_message(
                    MessageType.TASK_ERROR,
                    {"error": "Dashboard 路由未加载"},
                ))
                await ws.close()
                return

        await _serve_ide_connection(ws, accepted=True, initial_msg=unified)
    except asyncio.TimeoutError:
        await ws.send_json(make_unified_message(
            MessageType.TASK_ERROR,
            {"error": "等待 handshake 超时"},
        ))
        await ws.close()


async def _outbound_sender(session: IDESession):
    """从 outbound_queue 读取消息并通过 WebSocket 发送"""
    try:
        while True:
            msg = await session.outbound_queue.get()
            send_msg = msg
            if session.api_version.startswith("2"):
                msg_type = msg.get("type", "")
                if ":" not in msg_type and msg_type not in (
                    MessageType.HANDSHAKE,
                    MessageType.PING,
                    MessageType.PONG,
                ):
                    send_msg = _protocol_bridge.to_unified(msg, format_version="ide")
            try:
                await session.ws.send_json(send_msg)
            except Exception as e:
                logger.warning(f"[ZulongIDE] 发送失败: {e}")
                # WS断开时，尝试排空队列中的关键消息（task_error/task_complete）
                # 通过直接发送确保IDE不会卡在"思考中"状态
                remaining = [send_msg]
                while not session.outbound_queue.empty():
                    try:
                        m = session.outbound_queue.get_nowait()
                        if session.api_version.startswith("2"):
                            m_type = m.get("type", "")
                            if ":" not in m_type and m_type not in (
                                MessageType.HANDSHAKE,
                                MessageType.PING,
                                MessageType.PONG,
                            ):
                                m = _protocol_bridge.to_unified(m, format_version="ide")
                        remaining.append(m)
                    except asyncio.QueueEmpty:
                        break
                for m in remaining:
                    if m.get("type") in (
                        "task_error",
                        "task_complete",
                        MessageType.TASK_ERROR,
                        MessageType.TASK_COMPLETE,
                    ):
                        try:
                            await session.ws.send_json(m)
                        except Exception:
                            pass
                break
    except asyncio.CancelledError:
        pass


# ── HTTP 端点（健康检查） ─────────────────────────────

@ide_router.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(_sessions),
        "engine_ready": _engine_instance is not None,
    }


@ide_router.get("/api/ide/sessions")
async def get_active_ide_sessions():
    """查询所有活跃 IDE 会话及其任务/项目关联"""
    now = time.time()
    sessions = []
    for s in _sessions.values():
        info = s.to_info_dict()
        info["uptime_seconds"] = round(now - s.created_at, 1)
        sessions.append(info)
    return {"sessions": sessions, "count": len(sessions)}


# ── Web 监控前端 ──────────────────────────────────────

# 静态前端文件路径
_STATIC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "zulong_web", "static"
)


@app.get("/")
async def root():
    """Web 监控前端入口（独立运行时使用，Launcher 模式下由 LauncherApp 管理）"""
    index_path = os.path.join(_STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Zulong IDE Server running", "ws_ide": "ws://127.0.0.1:8090/ide"}


@ide_router.websocket("/monitor")
async def monitor_websocket(ws: WebSocket):
    """Web 前端监控 WebSocket — 实时推送系统事件"""
    global _main_event_loop
    if _main_event_loop is None:
        _main_event_loop = asyncio.get_running_loop()
    await ws.accept()
    _monitor_connections.add(ws)
    logger.info(f"[ZulongIDE] Web 监控客户端已连接 (total={len(_monitor_connections)})")

    # 收集初始快照
    task_graph_snapshot = None
    memory_graph_stats = None
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg:
            task_graph_snapshot = tg.to_frontend_dict()
    except Exception:
        pass
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg:
            memory_graph_stats = {
                "total_nodes": mg._stats.get("total_nodes", 0),
                "total_edges": mg._stats.get("total_edges", 0),
            }
    except Exception:
        pass

    # 收集代码锚点统计
    code_anchor_stats = None
    try:
        from zulong.memory.code_anchor import get_code_anchor_store
        store = get_code_anchor_store()
        if store:
            code_anchor_stats = store.get_stats()
    except Exception:
        pass

    # 发送欢迎消息和当前状态快照
    await ws.send_json({
        "type": "WELCOME",
        "ts": time.time(),
        "payload": {
            "active_sessions": [s.to_info_dict() for s in _sessions.values()],
            "engine_ready": _engine_instance is not None,
            "task_graph": task_graph_snapshot,
            "memory_graph_stats": memory_graph_stats,
            "code_anchor_stats": code_anchor_stats,
        },
    })
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                msg_type = data.get("type", "")
                if msg_type == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
                elif msg_type == "REQUEST_MEMORY_GRAPH":
                    asyncio.create_task(_push_memory_graph_snapshot(ws))
                elif msg_type == "EXPAND_NODE":
                    node_id = data.get("node_id")
                    if node_id:
                        asyncio.create_task(_handle_expand_node(node_id, ws))
                elif msg_type == "STOP_TASK":
                    # Web 端停止所有活跃 FC 循环
                    stopped_count = 0
                    for sid, sess in _sessions.items():
                        if hasattr(sess, 'cancel_event') and sess.cancel_event:
                            sess.cancel_event.set()
                            stopped_count += 1
                            logger.info(f"[ZulongIDE] Web停止: session={sid[:12]}")
                    # 设置引擎级中断标志（_check 方法会检测）
                    if _engine_instance and hasattr(_engine_instance, '_interrupt_flag'):
                        _engine_instance._interrupt_flag = True
                    await ws.send_json({
                        "type": "STOP_ACK",
                        "ts": time.time(),
                        "payload": {"stopped": True, "sessions": stopped_count},
                    })
                    logger.info(f"[ZulongIDE] Web停止指令: 影响 {stopped_count} 个会话")
            except (json.JSONDecodeError, Exception):
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[ZulongIDE] 监控连接异常: {e}")
    finally:
        _monitor_connections.discard(ws)
        logger.info(f"[ZulongIDE] Web 监控客户端断开 (total={len(_monitor_connections)})")


async def _push_memory_graph_snapshot(ws: WebSocket) -> None:
    """推送记忆图谱快照到指定 WebSocket"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if not mg:
            return
        snapshot = mg.to_frontend_dict() if hasattr(mg, "to_frontend_dict") else None
        if snapshot:
            await ws.send_json({
                "type": "MEMORY_GRAPH_UPDATE",
                "ts": time.time(),
                "payload": snapshot,
            })
    except Exception as e:
        logger.debug(f"[WebChat] 推送记忆图谱失败: {e}")


async def _handle_expand_node(node_id: str, ws: WebSocket) -> None:
    """处理展开节点请求"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if not mg:
            return
        if hasattr(mg, "get_node_children_for_frontend"):
            result = mg.get_node_children_for_frontend(node_id)
            if isinstance(result, dict):
                result.setdefault("node_id", result.get("parent_id", node_id))
        else:
            node = (
                mg.get_node(node_id)
                if hasattr(mg, "get_node")
                else getattr(mg, "_nodes", {}).get(node_id)
            )
            if not node:
                return
            result = {
                "node_id": node_id,
                "neighbors": mg.get_neighbors(node_id) if hasattr(mg, "get_neighbors") else [],
            }
        await ws.send_json({
            "type": "MEMORY_GRAPH_EXPAND_RESULT",
            "ts": time.time(),
            "payload": result,
        })
    except Exception as e:
        logger.debug(f"[WebChat] 展开节点失败: {e}")


# ── LLM 模型配置 API（运行时热切换） ─────────────────

@ide_router.get("/api/llm/config")
async def get_llm_config_api():
    """获取当前 LLM 配置和可用后端列表"""
    from zulong.config.config_manager import get_config_manager
    cm = get_config_manager()

    # 当前活跃配置
    current_backend = cm.get('llm.backend', 'ollama')
    current_config = cm.get_dict(f'llm.{current_backend}', {})

    # 所有可用后端
    backends = {}
    for name in ['ollama', 'siliconflow', 'vllm', 'sglang', 'llamacpp', 'lmstudio', 'openai']:
        cfg = cm.get_dict(f'llm.{name}', {})
        if cfg:
            backends[name] = {
                'base_url': cfg.get('base_url', ''),
                'model_id': cfg.get('model_id', ''),
            }

    return {
        "current_backend": current_backend,
        "current_model_id": current_config.get('model_id', ''),
        "current_base_url": current_config.get('base_url', ''),
        "backends": backends,
    }


@ide_router.post("/api/llm/switch")
async def switch_llm(data: dict):
    """热切换 LLM 后端和/或模型 ID"""
    backend = data.get("backend")
    model_id = data.get("model_id")
    base_url = data.get("base_url")
    api_key = data.get("api_key")

    engine = _get_engine()
    if not engine:
        return {"status": "error", "message": "Engine 未初始化"}

    success, msg = engine.hot_switch_llm(
        backend=backend, model_id=model_id, base_url=base_url, api_key=api_key)
    return {"status": "ok" if success else "error", "message": msg}


# ── 全模型层设置 API ─────────────────────────────────

@ide_router.get("/api/models/layers")
async def get_model_layers():
    """获取所有模型层的配置和状态"""
    from zulong.config.config_manager import get_config_manager
    cm = get_config_manager()

    layers = []

    # L1-A 反射层 (规则驱动，无独立模型)
    layers.append({
        "id": "l1_a",
        "name": "L1-A 反射层",
        "type": "rule",
        "enabled": True,
        "status": "running",
        "config": {},
        "editable_fields": [],
    })

    # L1-B 调度层
    l1b_config = {}
    try:
        from zulong.models.config import MODEL_CONFIGS, ModelID
        l1b_cfg = MODEL_CONFIGS.get(ModelID.L1_SCHEDULER)
        if l1b_cfg:
            l1b_config = {
                "model_path": l1b_cfg.repo_id,
                "device": l1b_cfg.device,
                "use_int4": l1b_cfg.use_int4,
            }
    except Exception:
        pass
    layers.append({
        "id": "l1_b",
        "name": "L1-B 调度层",
        "type": "local",
        "enabled": l1b_cfg.enabled if l1b_cfg else False,
        "status": "running" if l1b_config else "unloaded",
        "config": l1b_config,
        "editable_fields": ["model_path", "device", "use_int4"],
    })

    # L1-C 视觉层
    vision_cfg = cm.get_dict('vision', {})
    yolo_cfg = cm.get_dict('vision.yolo', {})
    layers.append({
        "id": "l1_c",
        "name": "L1-C 视觉层",
        "type": "vision",
        "enabled": cm.get('vision.camera.enabled', False),
        "status": "running" if cm.get('vision.camera.enabled', False) else "stopped",
        "config": {
            "model_path": yolo_cfg.get('model_path', ''),
            "confidence": yolo_cfg.get('confidence_threshold', 0.25),
            "device": yolo_cfg.get('device', 'cuda'),
        },
        "editable_fields": ["model_path", "confidence", "device"],
    })

    # L1-D 音频层
    tts_cfg = cm.get_dict('audio.tts', {})
    layers.append({
        "id": "l1_d",
        "name": "L1-D 音频层",
        "type": "audio",
        "enabled": cm.get('audio.microphone.enabled', False),
        "status": "running" if cm.get('audio.microphone.enabled', False) else "stopped",
        "config": {
            "backend": tts_cfg.get('backend', 'cosyvoice'),
            "model_path": tts_cfg.get('model_path', ''),
            "voice": tts_cfg.get('voice', ''),
            "device": tts_cfg.get('device', 'cuda'),
        },
        "editable_fields": ["backend", "model_path", "voice", "device"],
    })

    # L2 推理核心 (云端 API)
    current_backend = cm.get('llm.backend', 'ollama')
    current_llm_config = cm.get_dict(f'llm.{current_backend}', {})
    api_key_display = current_llm_config.get('api_key', '')
    if api_key_display and len(api_key_display) > 8:
        api_key_display = api_key_display[:4] + '***' + api_key_display[-4:]
    layers.append({
        "id": "l2_core",
        "name": "L2 推理核心",
        "type": "cloud",
        "enabled": True,
        "status": "running",
        "config": {
            "backend": current_backend,
            "model_id": current_llm_config.get('model_id', ''),
            "base_url": current_llm_config.get('base_url', ''),
            "api_key": api_key_display,
            "num_ctx": int(current_llm_config.get('num_ctx', 131072)),
        },
        "editable_fields": ["backend", "model_id", "base_url", "api_key", "num_ctx"],
    })

    # L2 备用
    l2b_config = {}
    try:
        l2b_cfg = MODEL_CONFIGS.get(ModelID.L2_BACKUP)
        if l2b_cfg:
            l2b_config = {
                "model_path": l2b_cfg.repo_id,
                "device": l2b_cfg.device,
                "use_int4": l2b_cfg.use_int4,
            }
    except Exception:
        pass
    layers.append({
        "id": "l2_backup",
        "name": "L2 备用",
        "type": "local",
        "enabled": l2b_cfg.enabled if l2b_cfg else False,
        "status": "running" if l2b_config else "unloaded",
        "config": l2b_config,
        "editable_fields": ["model_path", "device", "use_int4"],
    })

    # L3 专家层
    expert_models = []
    try:
        for mid in [ModelID.EXPERT_NAV, ModelID.EXPERT_MANIPULATION, ModelID.EXPERT_VISION]:
            ecfg = MODEL_CONFIGS.get(mid)
            if ecfg:
                expert_models.append({
                    "id": mid.value,
                    "name": mid.value.replace("_", " ").title(),
                    "model_path": ecfg.repo_id,
                    "device": ecfg.device,
                    "enabled": ecfg.enabled,
                })
    except Exception:
        pass
    layers.append({
        "id": "l3_experts",
        "name": "L3 专家层",
        "type": "expert",
        "enabled": True,
        "status": "running",
        "config": {"experts": expert_models},
        "editable_fields": [],
    })

    # L1 插件扩展
    l1_extensions = cm.get_dict('plugins.l1_extensions', {})
    if isinstance(l1_extensions, dict):
        for ext_id, ext_cfg in l1_extensions.items():
            layers.append({
                "id": f"l1_ext_{ext_id}",
                "name": ext_cfg.get('name', f'L1-{ext_id.upper()}'),
                "type": ext_cfg.get('type', 'local'),
                "enabled": ext_cfg.get('enabled', True),
                "status": "stopped",
                "config": ext_cfg.get('config', {}),
                "editable_fields": list(ext_cfg.get('config', {}).keys()),
            })

    # 可用后端列表
    available_backends = []
    for name in ['ollama', 'siliconflow', 'vllm', 'sglang', 'llamacpp', 'lmstudio', 'openai']:
        cfg = cm.get_dict(f'llm.{name}', {})
        if cfg:
            available_backends.append(name)

    return {
        "layers": layers,
        "available_backends": available_backends,
        "can_add_l1": True,
    }


@ide_router.post("/api/models/layers/{layer_id}/update")
async def update_model_layer(layer_id: str, data: dict):
    """更新指定模型层的配置"""
    config = data.get("config", {})
    if not config:
        return {"status": "error", "message": "config 不能为空"}

    from zulong.config.config_manager import get_config_manager
    cm = get_config_manager()

    try:
        if layer_id == "l2_core":
            # L2 核心使用现有 hot_switch_llm
            engine = _get_engine()
            if not engine:
                return {"status": "error", "message": "Engine 未初始化"}
            # 如果包含 num_ctx，先写入配置（hot_switch_llm 会从配置读取）
            if "num_ctx" in config:
                backend_name = config.get("backend") or cm.get("llm.backend", "ollama")
                cm.config.setdefault("llm", {}).setdefault(backend_name, {})["num_ctx"] = int(config["num_ctx"])
                cm.save()
            success, msg = engine.hot_switch_llm(
                backend=config.get("backend"),
                model_id=config.get("model_id"),
                base_url=config.get("base_url"),
                api_key=config.get("api_key"),
            )
            return {"status": "ok" if success else "error", "message": msg}

        elif layer_id == "l1_c":
            # 视觉层配置更新
            if "model_path" in config:
                cm.config.setdefault('vision', {}).setdefault('yolo', {})['model_path'] = config['model_path']
            if "confidence" in config:
                cm.config['vision']['yolo']['confidence_threshold'] = float(config['confidence'])
            if "device" in config:
                cm.config['vision']['yolo']['device'] = config['device']
            cm.save()
            return {"status": "ok", "message": "L1-C 视觉层配置已更新"}

        elif layer_id == "l1_d":
            # 音频层配置更新
            tts = cm.config.setdefault('audio', {}).setdefault('tts', {})
            for k in ['backend', 'model_path', 'voice', 'device']:
                if k in config:
                    tts[k] = config[k]
            cm.save()
            return {"status": "ok", "message": "L1-D 音频层配置已更新"}

        elif layer_id.startswith("l1_ext_"):
            # L1 插件层配置更新
            ext_id = layer_id[7:]  # 去掉 "l1_ext_" 前缀
            extensions = cm.config.setdefault('plugins', {}).setdefault('l1_extensions', {})
            if ext_id in extensions:
                extensions[ext_id]['config'] = config
                cm.save()
                return {"status": "ok", "message": f"插件层 {ext_id} 配置已更新"}
            return {"status": "error", "message": f"插件层 {ext_id} 不存在"}

        else:
            return {"status": "error", "message": f"层 {layer_id} 不支持运行时更新"}

    except Exception as e:
        logger.error(f"[ModelLayers] 更新层 {layer_id} 失败: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@ide_router.post("/api/models/layers/add")
async def add_model_layer(data: dict):
    """添加新的 L1 插件层"""
    name = data.get("name", "").strip()
    layer_type = data.get("type", "local")
    config = data.get("config", {})

    if not name:
        return {"status": "error", "message": "name 不能为空"}

    from zulong.config.config_manager import get_config_manager
    cm = get_config_manager()

    # 生成 ext_id
    extensions = cm.config.setdefault('plugins', {}).setdefault('l1_extensions', {})
    # 按字母序生成：e, f, g, ...
    existing_keys = set(extensions.keys())
    ext_id = None
    for ch in 'efghijklmnopqrstuvwxyz':
        if ch not in existing_keys:
            ext_id = ch
            break
    if not ext_id:
        return {"status": "error", "message": "已达到 L1 插件层数量上限"}

    extensions[ext_id] = {
        "name": name,
        "type": layer_type,
        "enabled": True,
        "config": config,
    }
    cm.save()

    return {
        "status": "ok",
        "message": f"已添加 L1 插件层: {name}",
        "layer_id": f"l1_ext_{ext_id}",
    }


# ── 聊天会话兼容 API ────────────────────────────────


@ide_router.get("/api/chat/sessions")
async def get_chat_sessions():
    """兼容旧前端接口：从 InteractionStore 返回轻量会话索引，不再读写 chat_sessions.json。"""
    try:
        from zulong.launcher.interaction_store import get_interaction_store
        store = get_interaction_store()
        sessions = []
        active_id = None
        for conv in store.list_conversations(limit=200):
            if conv.get("active") and not active_id:
                active_id = conv.get("conversation_id")
            sessions.append({
                "id": conv.get("conversation_id"),
                "title": conv.get("title") or "对话记录",
                "messages": [],
                "createdAt": int((conv.get("created_at") or 0) * 1000),
                "source": "memory_graph",
                "dialogue_session_id": conv.get("conversation_id"),
                "last_active_at": conv.get("last_active_at") or conv.get("created_at") or 0,
                "workspace_path": conv.get("workspace_path"),
                "project_id": conv.get("project_id"),
                "task_graph_id": conv.get("task_graph_id"),
            })
        return {"activeSessionId": active_id, "sessions": sessions}
    except Exception as e:
        logger.error(f"[ChatSessions] 会话索引读取失败: {e}")
    return {"activeSessionId": None, "sessions": []}


@ide_router.post("/api/chat/sessions")
async def save_chat_sessions(data: dict):
    """兼容旧前端同步调用：不再把聊天保存到单 JSON 文件。"""
    return {"status": "ok", "storage": "memory_graph_only", "persisted": False}


@ide_router.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, cascade: bool = True):
    """删除指定会话，可选级联删除 MemoryGraph 节点。"""
    dialogue_node_id = session_id
    mg_deleted = 0
    if cascade and dialogue_node_id:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg:
                nodes_to_remove = [dialogue_node_id]
                queue = [dialogue_node_id]
                while queue:
                    parent = queue.pop(0)
                    for child in mg.get_children(parent):
                        child_id = getattr(child, "node_id", "")
                        if child_id and child_id not in nodes_to_remove:
                            nodes_to_remove.append(child_id)
                            queue.append(child_id)
                for nid in reversed(nodes_to_remove):
                    if mg.remove_node(nid):
                        mg_deleted += 1
                if hasattr(mg, "save_all"):
                    mg.save_all()
                logger.info(f"[ChatSessions] 级联删除 {mg_deleted} 个 MemoryGraph 节点")
        except Exception as e:
            logger.warning(f"[ChatSessions] MemoryGraph 级联删除失败: {e}")

    return {
        "status": "ok",
        "message": f"会话已删除" + (f"，清理了 {mg_deleted} 个图谱节点" if mg_deleted > 0 else ""),
        "mg_nodes_deleted": mg_deleted,
    }


@ide_router.delete("/api/chat/sessions/{session_id}/messages/{message_id}")
async def delete_chat_message(session_id: str, message_id: str):
    """兼容旧前端接口：消息以 MemoryGraph 为准，不再写 chat_sessions.json。"""
    try:
        removed = 0
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if mg and mg.remove_node(message_id):
            removed = 1
            if hasattr(mg, "save_all"):
                mg.save_all()
        return {"status": "ok", "removed": removed}
    except Exception as e:
        logger.error(f"[ChatSessions] 删除消息失败: {e}")
        return {"status": "error", "message": str(e)}


@ide_router.get("/api/task-graph/active")
async def get_active_task_graph_snapshot(task_graph_id: Optional[str] = None, workspace_path: Optional[str] = None):
    """获取当前活跃任务图谱快照（前端按需拉取/重建用）"""
    try:
        from zulong.tools.task_tools import (
            get_active_task_graph,
            load_task_graph_deterministic,
            normalize_task_graph_id,
            _active_graph_id,
        )

        requested_graph_id = normalize_task_graph_id(task_graph_id)
        if not requested_graph_id:
            try:
                from zulong.launcher.web_chat_router import _task_execution_status

                requested_graph_id = normalize_task_graph_id(
                    _task_execution_status.get("task_graph_id")
                )
                workspace_path = workspace_path or _task_execution_status.get("workspace_path")
            except Exception:
                requested_graph_id = ""

        tg = get_active_task_graph(workspace_dir=workspace_path) if workspace_path else get_active_task_graph()
        if requested_graph_id and (
            not tg or normalize_task_graph_id(getattr(tg, "id", "")) != requested_graph_id
        ):
            if load_task_graph_deterministic(requested_graph_id, workspace_dir=workspace_path):
                tg = get_active_task_graph(workspace_dir=workspace_path) if workspace_path else get_active_task_graph()

        if not tg:
            return {"status": "ok", "graph": None, "graph_id": None}
        graph_data = tg.to_frontend_dict()
        graph_id = normalize_task_graph_id(getattr(tg, "id", "") or _active_graph_id)
        return {
            "status": "ok",
            "graph": graph_data,
            "graph_id": graph_id,
            "node_count": len(tg._nodes),
            "edge_count": len(tg._h_edges) + len(tg._d_edges),
        }
    except Exception as e:
        logger.error(f"[TaskGraph] 获取活跃图谱快照失败: {e}")
        return {"status": "error", "message": str(e), "graph": None}


@ide_router.post("/api/task-graph/bind-file")
async def bind_task_graph_file(body: Dict[str, Any]):
    """空会话入口：按用户指定的 JSON 文件创建/绑定任务图谱。"""
    try:
        file_path_raw = str(
            body.get("file_path")
            or body.get("path")
            or body.get("task_graph_file_path")
            or ""
        ).strip().strip('"').strip("'")
        if not file_path_raw:
            return JSONResponse(
                {"status": "error", "message": "请先填写任务图谱文件地址。"},
                status_code=400,
            )

        file_path = os.path.abspath(
            os.path.normpath(os.path.expanduser(os.path.expandvars(file_path_raw)))
        )
        if os.path.isdir(file_path):
            return JSONResponse(
                {"status": "error", "message": "请填写具体 JSON 文件地址，而不是文件夹。"},
                status_code=400,
            )
        if os.path.splitext(file_path)[1].lower() != ".json":
            return JSONResponse(
                {"status": "error", "message": "任务图谱文件需要使用 .json 后缀。"},
                status_code=400,
            )

        parent_dir = os.path.dirname(file_path)
        if not parent_dir:
            return JSONResponse(
                {"status": "error", "message": "任务图谱文件地址无效。"},
                status_code=400,
            )
        os.makedirs(parent_dir, exist_ok=True)

        from zulong.l2.task_graph import TaskGraph
        from zulong.tools.task_tools import (
            normalize_task_graph_id,
            set_active_task_graph,
            _backup_graph_to_disk,
        )

        created = False
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            tg = TaskGraph.deserialize(raw_data)
            graph_id = normalize_task_graph_id(getattr(tg, "id", "") or raw_data.get("id"))
        else:
            graph_id = f"tg_{int(time.time())}"
            title = str(body.get("title") or "").strip() or os.path.splitext(os.path.basename(file_path))[0] or "新任务图谱"
            tg = TaskGraph(title=title, graph_id=graph_id)
            tg.add_node(
                id="req",
                label=title,
                type="requirement",
                status="pending",
                desc="空会话预建任务图谱，等待用户输入任务。",
            )
            created = True

        if not graph_id:
            graph_id = f"tg_{int(time.time())}"
            tg.id = graph_id
            tg.graph_id = graph_id

        workspace_dir = str(body.get("workspace_path") or body.get("cwd") or "").strip()
        if not workspace_dir:
            workspace_dir = parent_dir
        workspace_dir = os.path.abspath(
            os.path.normpath(os.path.expanduser(os.path.expandvars(workspace_dir)))
        )
        os.makedirs(workspace_dir, exist_ok=True)

        tg.metadata["workspace_dir"] = workspace_dir
        tg.metadata["task_graph_file_path"] = file_path
        tg.metadata["user_seeded_empty_graph"] = bool(
            body.get("empty_graph", True)
            and len(getattr(tg, "_nodes", {}) or {}) <= 1
        )
        tg.metadata["conversation_id"] = body.get("session_id") or body.get("conversation_id") or ""

        set_active_task_graph(tg, graph_id, workspace_dir=workspace_dir)
        _backup_graph_to_disk(tg, graph_id)

        conversation_id = str(body.get("session_id") or body.get("conversation_id") or "").strip()
        if conversation_id:
            try:
                from zulong.launcher.interaction_store import get_interaction_store
                get_interaction_store().upsert_conversation(
                    conversation_id,
                    title=str(body.get("conversation_title") or body.get("title") or tg.title or "新会话"),
                    source="web_chat",
                    workspace_path=workspace_dir,
                    task_graph_id=graph_id,
                    metadata={"task_graph_file_path": file_path},
                    active=True,
                )
            except Exception as exc:
                logger.debug("[TaskGraph] 会话绑定写入跳过: %s", exc)

        graph_data = tg.to_frontend_dict()
        payload = {
            "event": "bind_file",
            "created": created,
            "graph": graph_data,
            "graph_id": graph_id,
            "task_graph_id": graph_id,
            "workspace_path": workspace_dir,
            "task_graph_file_path": file_path,
            "conversation_id": conversation_id,
            "session_id": conversation_id,
        }
        _broadcast_sync("TASK_GRAPH_UPDATE", payload)

        return {
            "status": "ok",
            "created": created,
            "graph": graph_data,
            "graph_id": graph_id,
            "task_graph_id": graph_id,
            "workspace_path": workspace_dir,
            "task_graph_file_path": file_path,
        }
    except Exception as e:
        logger.error(f"[TaskGraph] 绑定图谱文件失败: {e}", exc_info=True)
        return JSONResponse(
            {"status": "error", "message": f"任务图谱文件处理失败: {e}"},
            status_code=500,
        )


@ide_router.delete("/api/task-graph/{graph_id}")
async def delete_task_graph(graph_id: str):
    """删除指定任务图谱（清除活跃图 + 删除磁盘备份）"""
    cleared_active = False
    deleted_backup = False
    try:
        from zulong.tools.task_tools import (
            get_active_task_graph, set_active_task_graph,
            normalize_task_graph_id, _GRAPH_BACKUP_DIR, _active_graph_id
        )
        graph_id = normalize_task_graph_id(graph_id)
        # 如果要删除的是当前活跃图，清除它
        tg = get_active_task_graph()
        if tg and normalize_task_graph_id(_active_graph_id) == graph_id:
            set_active_task_graph(None, None)
            cleared_active = True
            logger.info(f"[TaskGraph] 已清除活跃图: {graph_id}")
        # 删除磁盘备份文件
        backup_path = os.path.join(_GRAPH_BACKUP_DIR, f"{graph_id}.json")
        if os.path.exists(backup_path):
            os.remove(backup_path)
            deleted_backup = True
            logger.info(f"[TaskGraph] 已删除备份: {backup_path}")
    except Exception as e:
        logger.error(f"[TaskGraph] 删除图谱 {graph_id} 失败: {e}")
        return {"status": "error", "message": str(e)}

    # 广播图谱删除事件到 Web 前端
    _broadcast_sync("TASK_GRAPH_DELETED", {
        "graph_id": graph_id,
        "cleared_active": cleared_active,
    })

    return {
        "status": "ok",
        "graph_id": graph_id,
        "cleared_active": cleared_active,
        "deleted_backup": deleted_backup,
    }


# ── 启动 ──────────────────────────────────────────────

# 独立运行时：挂载 IDE 路由到 app
app.include_router(ide_router)

# 挂载静态文件（放在所有路由注册之后，避免覆盖 API 端点）
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.on_event("startup")
async def startup():
    """独立运行时的启动钩子（Launcher 模式下不走此路径）"""
    global _main_event_loop, _preload_manager
    _main_event_loop = asyncio.get_running_loop()
    await _init_engine()
    
    # 🔥 系统启动预加载
    if _engine_instance is not None:
        try:
            from zulong.l2.preload_manager import PreloadManager
            _preload_manager = PreloadManager(_engine_instance)
            await _preload_manager.start_preload()
        except Exception as e:
            logger.error(f"[ZulongIDE] 预加载失败: {e}", exc_info=True)
    
    logger.info(f"[ZulongIDE] 服务启动完成")
    logger.info(f"[ZulongIDE]   IDE WebSocket: ws://127.0.0.1:8090/ide")
    logger.info(f"[ZulongIDE]   Web 监控前端: http://127.0.0.1:8090/")
    logger.info(f"[ZulongIDE]   监控 WebSocket: ws://127.0.0.1:8090/monitor")


@app.on_event("shutdown")
async def shutdown():
    """独立运行时的关闭钩子，优雅关闭线程池"""
    try:
        from zulong.ide.ide_fc_runner import ThreadPoolManager
        logger.info("[ZulongIDE] 服务关闭中...")
        ThreadPoolManager.get_instance().graceful_shutdown()
        logger.info("[ZulongIDE] 服务已关闭")
    except Exception as e:
        logger.warning(f"[ZulongIDE] 关闭时发生异常: {e}")


def main():
    # P2-22: 从配置读取端口号，默认8090
    port = 8090
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "config", "zulong_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            port = (cfg.get("ide") or {}).get("port") or (cfg.get("launcher") or {}).get("port") or 8090
    except Exception:
        pass
    uvicorn.run(
        "zulong.ide.ide_server:app",
        host="127.0.0.1",
        port=port,
        log_level="info",
        ws_ping_interval=None,
        ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
