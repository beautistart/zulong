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
from fastapi.responses import FileResponse
from fastapi.routing import APIRouter
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = logging.getLogger(__name__)

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

    def to_info_dict(self) -> Dict[str, Any]:
        """序列化会话元数据（用于 REST API / WELCOME 消息）"""
        return {
            "session_id": self.session_id,
            "cwd": self.cwd,
            "project_id": self.project_id,
            "task_graph_id": self.task_graph_id,
            "created_at": self.created_at,
            "has_fc_task": self.fc_task is not None and not self.fc_task.done(),
        }

    async def send_msg(self, msg_type: str, payload: Dict[str, Any]) -> None:
        """向插件发送消息"""
        msg = {
            "msg_id": uuid.uuid4().hex[:12],
            "type": msg_type,
            "session_id": self.session_id,
            "ts": time.time(),
            "payload": payload,
        }
        await self.outbound_queue.put(msg)


# 活跃会话
_sessions: Dict[str, IDESession] = {}

# ── Web 监控连接 ──────────────────────────────────────
_monitor_connections: Set[WebSocket] = set()


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


# ── 消息处理 ──────────────────────────────────────────

async def _handle_session_start(session: IDESession, payload: Dict) -> None:
    """处理新任务请求，启动 FC 循环"""
    task_text = payload.get("task", "")
    cwd = payload.get("cwd", ".")
    ide_system_prompt = payload.get("ide_system_prompt", "")
    if not task_text:
        await session.send_msg("task_error", {"error": "task 不能为空"})
        return

    logger.info(f"[ZulongIDE] session_start: task={task_text[:100]}, cwd={cwd}")

    # 检测项目模式：如果 cwd 下存在 .zulong/project.json，更新项目状态为 executing
    _detected_project_id = None
    _detected_task_graph_id = None
    _project_json_path = os.path.join(cwd, ".zulong", "project.json")
    if os.path.isfile(_project_json_path):
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


async def _handle_user_cancel(session: IDESession, _payload: Dict) -> None:
    """处理用户取消"""
    logger.info(f"[ZulongIDE] user_cancel: session={session.session_id[:12]}")
    session.cancel_event.set()
    # 同步设置 engine._interrupt_flag，确保所有取消检查路径都生效
    try:
        from zulong.l2.inference_engine import get_inference_engine
        engine = get_inference_engine()
        if engine:
            engine._interrupt_flag = True
            logger.info("[ZulongIDE] user_cancel: engine._interrupt_flag 已设置")
    except Exception as e:
        logger.warning(f"[ZulongIDE] user_cancel: 设置 interrupt_flag 失败: {e}")
    # 广播取消状态到Web端
    await broadcast_monitor_event("IDE_SESSION_CANCEL", {
        "session_id": session.session_id,
    })


async def _handle_ping(session: IDESession, _payload: Dict) -> None:
    """处理心跳ping消息，立即返回pong"""
    pong = {
        "msg_id": uuid.uuid4().hex[:12],
        "type": "pong",
        "session_id": session.session_id,
        "ts": time.time(),
        "payload": {},
    }
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


def _load_graph_deterministic(graph_id: str) -> bool:
    """确定性三级加载 TaskGraph: 内存 → 磁盘 → MemoryGraph

    Returns: True 表示加载成功并已设置为活跃图
    """
    from zulong.tools.task_tools import (
        get_active_task_graph, set_active_task_graph, load_graph_from_backup,
    )

    # Level 1: 内存匹配
    tg = get_active_task_graph()
    if tg and getattr(tg, 'id', '') == graph_id:
        logger.info(f"[ZulongIDE] 确定性恢复 Level 1 (内存): {graph_id}")
        return True

    # Level 2: 磁盘备份
    tg = load_graph_from_backup(graph_id)
    if tg:
        set_active_task_graph(tg, graph_id)
        logger.info(f"[ZulongIDE] 确定性恢复 Level 2 (磁盘): {graph_id}")
        return True

    # Level 3: MemoryGraph 重建
    try:
        from zulong.memory.memory_graph import get_memory_graph
        from zulong.memory.graph_adapters import rebuild_task_graph_from_memory
        mg = get_memory_graph()
        if mg:
            tg = rebuild_task_graph_from_memory(mg, graph_id)
            if tg:
                set_active_task_graph(tg, graph_id)
                logger.info(
                    f"[ZulongIDE] 确定性恢复 Level 3 (MemoryGraph): {graph_id}")
                return True
    except Exception as e:
        logger.debug(f"[ZulongIDE] Level 3 MemoryGraph 重建失败: {e}")

    logger.warning(f"[ZulongIDE] 确定性恢复失败: 三级加载均未找到 {graph_id}")
    return False


async def _handle_session_resume(session: IDESession, payload: Dict) -> None:
    """处理会话恢复（插件重连后继续未完成的任务）"""
    task_text = payload.get("task", "")
    cwd = payload.get("cwd", ".")
    ide_system_prompt = payload.get("ide_system_prompt", "")

    if not task_text:
        await session.send_msg("task_error", {"error": "resume task 不能为空"})
        return

    logger.info(f"[ZulongIDE] session_resume: task={task_text[:100]}")

    # 通知 Web 前端：任务恢复（在已有会话中继续，而非新建窗口）
    try:
        from zulong.tools.task_tools import _active_graph_id
        await broadcast_monitor_event("IDE_SESSION_START", {
            "session_id": session.session_id,
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

    # 恢复活跃 TaskGraph
    graph_id = payload.get("graph_id", "")

    if graph_id:
        # 确定性恢复路径
        _load_graph_deterministic(graph_id)
    else:
        # 兼容旧逻辑: 从磁盘备份加载最近的图谱
        try:
            from zulong.tools.task_tools import (
                get_active_task_graph, load_latest_backup,
                set_active_task_graph,
            )
            if get_active_task_graph() is None:
                backup_tg, backup_gid = load_latest_backup()
                if backup_tg and backup_gid:
                    set_active_task_graph(backup_tg, backup_gid)
                    logger.info(
                        f"[ZulongIDE] session_resume: 从备份恢复活跃图 {backup_gid}")
        except Exception as e:
            logger.debug(f"[ZulongIDE] session_resume: 备份恢复尝试失败: {e}")

    # 恢复模式：使用 "继续" 前缀触发 RESUME 意图检测
    resume_text = f"继续之前的任务：{task_text}"
    session.fc_task = asyncio.create_task(
        _run_fc_loop(session, resume_text, cwd, ide_system_prompt,
                     force_graph_id=graph_id))


# 消息路由表
_MESSAGE_HANDLERS = {
    "session_start": _handle_session_start,
    "session_resume": _handle_session_resume,
    "tool_result": _handle_tool_result,
    "user_cancel": _handle_user_cancel,
    "ping": _handle_ping,
    "audio_start": _handle_audio_start,
    "audio_chunk": _handle_audio_chunk,
    "audio_end": _handle_audio_end,
}


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

        # FC 循环完成 — 根据终止原因发送 task_complete 或 task_error
        _reason = getattr(result, "reason", None) or "done"
        _completion_result = (result.text_response if result and result.text_response
                              else "(任务完成，无输出)")
        _is_error = _reason not in ("done", None)
        _msg_type = "task_error" if _is_error else "task_complete"
        _msg_payload = ({"error": _completion_result} if _is_error
                        else {"result": _completion_result})
        try:
            await session.send_msg(_msg_type, _msg_payload)
        except Exception:
            pass
        # 兜底：直接通过WS发送，防止队列消费者已退出导致消息丢失
        try:
            await session.ws.send_json({
                "type": _msg_type,
                "session_id": session.session_id,
                "payload": _msg_payload,
            })
            logger.info(f"[ZulongIDE] 兜底WS直发成功: type={_msg_type}, session={session.session_id[:12]}")
        except Exception as e:
            logger.warning(f"[ZulongIDE] 兜底WS直发失败: type={_msg_type}, error={e}")

        # Web 监控: IDE 会话结束
        await broadcast_monitor_event("IDE_SESSION_END", {
            "session_id": session.session_id, "status": "completed"})

    except asyncio.CancelledError:
        logger.info(f"[ZulongIDE] FC 循环被取消: session={session.session_id[:12]}")
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
        await broadcast_monitor_event("IDE_SESSION_END", {
            "session_id": session.session_id, "status": "cancelled"})
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
        await broadcast_monitor_event("IDE_SESSION_END", {
            "session_id": session.session_id, "status": "error",
            "error": error_msg[:200]})


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
                [{"role": "system", "content": base_prompt}])
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
    global _main_event_loop
    if _main_event_loop is None:
        _main_event_loop = asyncio.get_running_loop()
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
    ack = {
        "msg_id": uuid.uuid4().hex[:12],
        "type": "session_ack",
        "session_id": session_id,
        "ts": time.time(),
        "payload": {
            "session_id": session_id,
            "context_window_size": _context_window_size,
        },
    }
    await ws.send_json(ack)

    # 启动出站消息发送协程
    sender_task = asyncio.create_task(_outbound_sender(session))

    try:
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

            msg_type = msg.get("type", "")
            payload = msg.get("payload", {})

            handler = _MESSAGE_HANDLERS.get(msg_type)
            if handler:
                await handler(session, payload)
            else:
                logger.warning(f"[ZulongIDE] 未知消息类型: {msg_type}")

    except WebSocketDisconnect:
        logger.info(f"[ZulongIDE] WebSocket 断开: session={session_id[:12]}")
    except Exception as e:
        logger.error(f"[ZulongIDE] WebSocket 异常: {e}", exc_info=True)
    finally:
        # 清理
        sender_task.cancel()
        if session.fc_task and not session.fc_task.done():
            session.cancel_event.set()
            # 给FC任务一个短窗口完成收尾（如发送task_complete），而非立即取消
            try:
                await asyncio.wait_for(session.fc_task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                session.fc_task.cancel()
        _sessions.pop(session_id, None)
        logger.info(f"[ZulongIDE] 会话清理完成: session={session_id[:12]}")


async def _outbound_sender(session: IDESession):
    """从 outbound_queue 读取消息并通过 WebSocket 发送"""
    try:
        while True:
            msg = await session.outbound_queue.get()
            try:
                await session.ws.send_json(msg)
            except Exception as e:
                logger.warning(f"[ZulongIDE] 发送失败: {e}")
                # WS断开时，尝试排空队列中的关键消息（task_error/task_complete）
                # 通过直接发送确保IDE不会卡在"思考中"状态
                remaining = [msg]
                while not session.outbound_queue.empty():
                    try:
                        remaining.append(session.outbound_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                for m in remaining:
                    if m.get("type") in ("task_error", "task_complete"):
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
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "openclaw_bridge", "web", "static"
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
        node = mg._nodes.get(node_id)
        if node:
            # 获取邻居节点
            neighbors = mg.get_neighbors(node_id) if hasattr(mg, "get_neighbors") else []
            await ws.send_json({
                "type": "MEMORY_GRAPH_EXPAND_RESULT",
                "ts": time.time(),
                "payload": {
                    "node_id": node_id,
                    "neighbors": neighbors,
                },
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


# ── 聊天会话持久化 API ────────────────────────────────

_CHAT_SESSIONS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "chat_sessions.json"
)


@ide_router.get("/api/chat/sessions")
async def get_chat_sessions():
    """读取服务端持久化的聊天会话数据"""
    try:
        if os.path.exists(_CHAT_SESSIONS_FILE):
            with open(_CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.loads(f.read())
    except Exception as e:
        logger.error(f"[ChatSessions] 读取失败: {e}")
    return {"activeSessionId": None, "sessions": []}


@ide_router.post("/api/chat/sessions")
async def save_chat_sessions(data: dict):
    """保存聊天会话数据到服务端"""
    try:
        os.makedirs(os.path.dirname(_CHAT_SESSIONS_FILE), exist_ok=True)
        with open(_CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"[ChatSessions] 保存失败: {e}")
        return {"status": "error", "message": str(e)}


@ide_router.delete("/api/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str, cascade: bool = True):
    """删除指定会话，可选级联删除 MemoryGraph 节点"""
    deleted_from_file = False
    dialogue_node_id = None

    # 1. 从 chat_sessions.json 删除
    try:
        if os.path.exists(_CHAT_SESSIONS_FILE):
            with open(_CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                store = json.loads(f.read())
            sessions = store.get("sessions", [])
            # 找到要删除的会话，记录 dialogue_session_id
            for s in sessions:
                if s.get("id") == session_id:
                    dialogue_node_id = s.get("dialogue_session_id")
                    break
            # 过滤删除
            store["sessions"] = [s for s in sessions if s.get("id") != session_id]
            if store.get("activeSessionId") == session_id:
                store["activeSessionId"] = store["sessions"][0]["id"] if store["sessions"] else None
            with open(_CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(store, ensure_ascii=False, indent=2))
            deleted_from_file = True
    except Exception as e:
        logger.error(f"[ChatSessions] 删除会话 {session_id} 失败: {e}")
        return {"status": "error", "message": f"文件删除失败: {e}"}

    # 2. 级联删除 MemoryGraph 节点
    mg_deleted = 0
    if cascade and dialogue_node_id:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg:
                # 收集子树节点（session → rounds → sub_dialogues）
                nodes_to_remove = [dialogue_node_id]
                # BFS 收集 HIERARCHY 子节点
                queue = [dialogue_node_id]
                while queue:
                    parent = queue.pop(0)
                    for edge in mg.get_edges_from(parent):
                        if edge.edge_type.value == "HIERARCHY":
                            nodes_to_remove.append(edge.target)
                            queue.append(edge.target)
                # 逐个删除
                for nid in reversed(nodes_to_remove):
                    if mg.remove_node(nid):
                        mg_deleted += 1
                mg.save()
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
    """删除指定会话中的单条消息"""
    try:
        if not os.path.exists(_CHAT_SESSIONS_FILE):
            return {"status": "ok", "message": "无数据文件"}
        with open(_CHAT_SESSIONS_FILE, "r", encoding="utf-8") as f:
            store = json.loads(f.read())
        sessions = store.get("sessions", [])
        session = next((s for s in sessions if s.get("id") == session_id), None)
        if not session:
            return {"status": "ok", "message": "会话不存在"}
        messages = session.get("messages", [])
        original_count = len(messages)
        session["messages"] = [m for m in messages if m.get("id") != message_id]
        removed = original_count - len(session["messages"])
        if removed > 0:
            with open(_CHAT_SESSIONS_FILE, "w", encoding="utf-8") as f:
                f.write(json.dumps(store, ensure_ascii=False, indent=2))
        return {"status": "ok", "removed": removed}
    except Exception as e:
        logger.error(f"[ChatSessions] 删除消息失败: {e}")
        return {"status": "error", "message": str(e)}


@ide_router.get("/api/task-graph/active")
async def get_active_task_graph_snapshot():
    """获取当前活跃任务图谱快照（前端按需拉取/重建用）"""
    try:
        from zulong.tools.task_tools import get_active_task_graph, _active_graph_id
        tg = get_active_task_graph()
        if not tg:
            return {"status": "ok", "graph": None, "graph_id": None}
        graph_data = tg.to_frontend_dict()
        return {
            "status": "ok",
            "graph": graph_data,
            "graph_id": _active_graph_id,
            "node_count": len(tg._nodes),
            "edge_count": len(tg._h_edges) + len(tg._d_edges),
        }
    except Exception as e:
        logger.error(f"[TaskGraph] 获取活跃图谱快照失败: {e}")
        return {"status": "error", "message": str(e), "graph": None}


@ide_router.delete("/api/task-graph/{graph_id}")
async def delete_task_graph(graph_id: str):
    """删除指定任务图谱（清除活跃图 + 删除磁盘备份）"""
    cleared_active = False
    deleted_backup = False
    try:
        from zulong.tools.task_tools import (
            get_active_task_graph, set_active_task_graph,
            _GRAPH_BACKUP_DIR, _active_graph_id
        )
        # 如果要删除的是当前活跃图，清除它
        tg = get_active_task_graph()
        if tg and _active_graph_id == graph_id:
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
    global _main_event_loop
    _main_event_loop = asyncio.get_running_loop()
    await _init_engine()
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
