"""
Web 聊天路由器 — 通过 EventBus 路由消息到祖龙系统主链路

Web 前端是主系统界面，聊天消息不经过 IDE Server。

消息流:
  Full 模式: Web /ws → EventBus(USER_TEXT) → L1-B → L2 → EventBus(L2_OUTPUT/STREAM) → Web /ws
  IDE  模式: Web /ws → EventBus(USER_TEXT) → L1-B → L2 → EventBus(L2_OUTPUT/STREAM) → Web /ws
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

from zulong.core.unified_protocol import (
    ProtocolBridge, MessageType, make_unified_message,
)
from zulong.launcher.conversation_orchestrator import (
    RouteDecision,
    get_conversation_orchestrator,
)
from zulong.launcher.interaction_store import get_interaction_store

logger = logging.getLogger(__name__)

router = APIRouter()

_PRE_LLM_CONTEXT_TIMEOUT = 0.25

# ── 状态 ──────────────────────────────────────────────

# WebSocket 连接管理
_ws_clients: Set[WebSocket] = set()

# 协议版本追踪 (ws id → protocol_version: "legacy" | "unified")
_ws_protocols: Dict[int, str] = {}

# 客户端类型追踪 (ws id → client_type: "dashboard" | "ide_plugin" | ...)
_ws_client_types: Dict[int, str] = {}

# 统一协议桥接器
_protocol_bridge = ProtocolBridge()

# 运行模式（由 LauncherApp 在启动后设置）
_launch_mode: Optional[str] = None  # "full" | "ide"

# asyncio 事件循环引用（用于从 EventBus 分发线程安全地调度协程）
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# 活跃聊天取消事件（IDE 模式使用）
_chat_cancels: Dict[str, asyncio.Event] = {}


def _extract_text_from_message(message: dict) -> str:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    for key in ("text", "result", "message", "error"):
        value = message.get(key)
        if value:
            return str(value)
        value = payload.get(key)
        if value:
            return str(value)
    data = message.get("data") if isinstance(message.get("data"), dict) else payload.get("data")
    if isinstance(data, dict):
        interaction = data.get("interaction")
        if isinstance(interaction, dict):
            title = interaction.get("title") or ""
            detail = interaction.get("detail") or ""
            return "\n".join(str(x) for x in (title, detail) if x)
    interaction = message.get("interaction") or payload.get("interaction")
    if isinstance(interaction, dict):
        title = interaction.get("title") or ""
        detail = interaction.get("detail") or ""
        return "\n".join(str(x) for x in (title, detail) if x)
    for key in ("action_summary", "summary", "risk_reason", "tool_name", "phase"):
        value = message.get(key) or payload.get(key)
        if value:
            return str(value)
    return ""


def _resolve_conversation_binding(message: dict) -> Dict[str, Optional[str]]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    turn_id = (
        message.get("turn_id")
        or payload.get("turn_id")
        or message.get("request_id")
        or payload.get("request_id")
    )
    explicit_conversation_id = (
        message.get("conversation_id")
        or message.get("session_id")
        or payload.get("conversation_id")
        or payload.get("session_id")
    )

    store = get_interaction_store()
    binding: Dict[str, Optional[str]] = {
        "conversation_id": explicit_conversation_id,
        "turn_id": turn_id,
        "workspace_path": message.get("workspace_path") or payload.get("workspace_path") or payload.get("cwd"),
        "project_id": message.get("project_id") or payload.get("project_id"),
        "task_graph_id": message.get("task_graph_id") or payload.get("task_graph_id"),
    }

    if turn_id:
        try:
            turn_binding = store.find_conversation_for_turn(turn_id)
        except Exception:
            turn_binding = None
        if turn_binding:
            resolved_conversation_id = turn_binding.get("conversation_id")
            if (
                explicit_conversation_id
                and resolved_conversation_id
                and explicit_conversation_id != resolved_conversation_id
            ):
                logger.debug(
                    "[WebChatRouter] turn 绑定覆盖会话: explicit=%s -> resolved=%s, turn_id=%s",
                    explicit_conversation_id,
                    resolved_conversation_id,
                    turn_id,
                )
            binding["conversation_id"] = turn_binding.get("conversation_id")
            binding["workspace_path"] = binding["workspace_path"] or turn_binding.get("workspace_path")
            binding["project_id"] = binding["project_id"] or turn_binding.get("project_id")
            binding["task_graph_id"] = binding["task_graph_id"] or turn_binding.get("task_graph_id")

    if not binding["conversation_id"]:
        try:
            active = store.find_active_conversation(max_age_seconds=3600)
            binding["conversation_id"] = active.get("conversation_id") if active else None
            if active:
                binding["workspace_path"] = binding["workspace_path"] or active.get("workspace_path")
                binding["project_id"] = binding["project_id"] or active.get("project_id")
                binding["task_graph_id"] = binding["task_graph_id"] or active.get("task_graph_id")
        except Exception:
            binding["conversation_id"] = None

    return binding


def _persist_web_visible_message(message: dict) -> None:
    """Persist user-visible Web messages that were generated outside prepare_turn.

    Web localStorage intentionally no longer stores message bodies.  This small
    bridge keeps assistant outputs and important tool/status cards recoverable
    after refresh without introducing another event channel.
    """
    if message.get("_persisted"):
        return
    original_payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    binding = _resolve_conversation_binding(message)
    conversation_id = binding.get("conversation_id")
    if not conversation_id:
        return
    event_type = str(message.get("type") or "")
    role = _event_role_for_persistence(event_type, message)
    if not role:
        return
    text = _extract_text_from_message(message)
    payload = dict(original_payload) if isinstance(original_payload, dict) and original_payload else dict(message)
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    if event_type == "THINKING_STEP":
        step_type = payload.get("step_type") or message.get("step_type")
        if step_type:
            payload["step_type"] = step_type
            event_type = str(step_type)
        if not data and isinstance(message.get("data"), dict):
            data = message["data"]
            payload["data"] = data
    if conversation_id and not payload.get("conversation_id"):
        payload["conversation_id"] = conversation_id
    if conversation_id and not payload.get("session_id"):
        payload["session_id"] = conversation_id
    if binding.get("turn_id") and not payload.get("turn_id"):
        payload["turn_id"] = binding["turn_id"]
    if binding.get("turn_id") and not payload.get("request_id"):
        payload["request_id"] = binding["turn_id"]
    interaction = payload.get("interaction") if isinstance(payload, dict) else None
    if not interaction and isinstance(data, dict):
        interaction = data.get("interaction")
        if interaction:
            payload["interaction"] = interaction
    if not text and isinstance(interaction, dict):
        text = interaction.get("detail") or interaction.get("title") or ""
    if not text and event_type == "IDE_APPROVAL_STATUS":
        text = (
            payload.get("action_summary")
            or payload.get("summary")
            or payload.get("risk_reason")
            or payload.get("tool_name")
            or "需要确认操作"
        )
    if not text:
        return
    try:
        event_id = get_interaction_store().append_event(
            conversation_id=conversation_id,
            turn_id=binding.get("turn_id"),
            event_type=event_type or "web_event",
            role=role,
            source="web_runtime" if role == "assistant" else "ide_bridge",
            text=text,
            payload=payload,
            workspace_path=binding.get("workspace_path"),
            project_id=binding.get("project_id"),
            task_graph_id=binding.get("task_graph_id"),
        )
        try:
            from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph

            mirror_payload = dict(payload)
            mirror_payload.setdefault("source_event_id", event_id)
            if binding.get("task_graph_id") and not mirror_payload.get("task_graph_id"):
                mirror_payload["task_graph_id"] = binding["task_graph_id"]
            mirror_interaction_to_memory_graph(
                conversation_id=conversation_id,
                turn_id=binding.get("turn_id") or event_id,
                role=role,
                text=text,
                event_type=event_type or "web_event",
                source="web_runtime" if role == "assistant" else "ide_bridge",
                payload=mirror_payload,
            )
        except Exception as mirror_exc:
            logger.debug(f"[WebChatRouter] MemoryGraph 镜像跳过: {event_type}: {mirror_exc}")
        try:
            from zulong.review.task_execution_extractor import maybe_finalize_task_execution_trace

            maybe_finalize_task_execution_trace(
                conversation_id=conversation_id,
                turn_id=binding.get("turn_id"),
                task_graph_id=binding.get("task_graph_id") or payload.get("task_graph_id"),
                event_type=event_type or "web_event",
            )
        except Exception as trace_exc:
            logger.debug(f"[WebChatRouter] TaskExecutionTrace 跳过: {event_type}: {trace_exc}")
    except Exception as exc:
        logger.debug(f"[WebChatRouter] 消息持久化跳过: {event_type}: {exc}")


def _event_role_for_persistence(event_type: str, message: dict) -> Optional[str]:
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    if event_type == "STREAMING_RESPONSE":
        if message.get("complete") is not True and payload.get("complete") is not True:
            return None
        return "assistant"
    if event_type in ("CHAT_RESPONSE", "DISPLAY_TEXT", "TASK_COMPLETE", "FC_DONE"):
        return "assistant"
    if event_type in (
        "IDE_TERMINAL_STATUS",
        "IDE_APPROVAL_STATUS",
        "IDE_DIFF_STATUS",
        "IDE_CHECKPOINT_STATUS",
        "IDE_FILE_CHANGED",
        "IDE_TOOL_REQUEST",
        "IDE_TOOL_RESULT",
        "IDE_TOOL_EXEC",
        "THINKING_STEP",
    ):
        return "tool"
    return None


def _failure_response_text(reason: str) -> str:
    return f"系统当前出问题了，{reason}，因此无法正常回复。"


def _collect_dialogue_sessions(limit: int = 200) -> Dict[str, Any]:
    store = get_conversation_orchestrator().store
    store_sessions = []
    try:
        for conv in store.list_conversations(limit=limit):
            store_sessions.append({
                "id": conv.get("conversation_id"),
                "title": conv.get("title") or "对话记录",
                "created_at": conv.get("created_at") or 0,
                "last_active_at": conv.get("last_active_at") or conv.get("created_at") or 0,
                "preview": "",
                "round_count": 0,
                "source": conv.get("source") or "interaction_store",
                "workspace_path": conv.get("workspace_path"),
                "cwd": conv.get("workspace_path"),
                "project_id": conv.get("project_id"),
                "task_graph_id": conv.get("task_graph_id"),
            })
    except Exception as store_err:
        logger.debug(f"[WebChatRouter] interaction store session list skipped: {store_err}")

    sessions = store_sessions
    try:
        from zulong.memory.graph_adapters import DialogueAdapter

        mg = _get_active_memory_graph()
        if mg:
            graph_sessions = DialogueAdapter.list_sessions(mg)
            known = {s.get("id") for s in graph_sessions}
            sessions = store_sessions + [s for s in graph_sessions if s.get("id") not in known]
    except Exception as mg_err:
        logger.debug(f"[WebChatRouter] memory graph session list skipped: {mg_err}")

    active = None
    try:
        active = store.find_active_conversation(max_age_seconds=86400)
    except Exception:
        active = None

    return {
        "activeSessionId": (
            active.get("conversation_id")
            if isinstance(active, dict)
            else (sessions[0].get("id") if sessions else None)
        ),
        "sessions": sessions,
    }


def _get_active_memory_graph():
    """Return the MemoryGraph instance owned by LauncherApp when available.

    The launcher module manager is the owner of the running memory graph. Some
    lower-level helpers can create fallback graph instances from config, which
    risks split-brain snapshots. Web routes should prefer the active launcher
    context and only fall back when the launcher has not started yet.
    """
    try:
        from zulong.launcher import app as launcher_app_module

        launcher = getattr(launcher_app_module, "_app_instance", None)
        if launcher is not None:
            mg = getattr(launcher, "manager", None)
            if mg is not None:
                graph = mg.context.get("memory_graph")
                if graph is not None:
                    return graph
    except Exception:
        pass
    try:
        from zulong.memory.memory_graph import get_memory_graph

        return get_memory_graph()
    except Exception:
        return None

# EventBus 是否已订阅
_eventbus_subscribed = False


# ── 公共接口 ──────────────────────────────────────────

_heartbeat_task: Optional[asyncio.Task] = None

async def _ws_heartbeat_loop():
    """🔧 定期向所有 /ws 客户端发送 THINKING_STEP 心跳
    防止 Web 前端 120 秒看门狗在长 FC 循环中超时。
    """
    while True:
        await asyncio.sleep(30)
        if _ws_clients:
            try:
                await _broadcast({
                    "type": "THINKING_STEP",
                    "step_type": "heartbeat",
                    "request_id": "",
                    "data": {"message": "正在处理中..."},
                    "timestamp": time.time(),
                    "_persisted": True,  # 跳过持久化
                })
            except Exception:
                pass

def _start_ws_heartbeat():
    """启动 WebSocket 心跳任务（幂等）"""
    global _heartbeat_task
    if _heartbeat_task is None or _heartbeat_task.done():
        try:
            loop = asyncio.get_event_loop()
            _heartbeat_task = loop.create_task(_ws_heartbeat_loop())
            logger.info("[WebChatRouter] ✅ 已启动 WebSocket 心跳 (每30秒)")
        except RuntimeError as e:
            logger.warning(f"[WebChatRouter] 无法启动心跳: {e}")

def set_launch_mode(mode: str) -> None:
    """设置运行模式并初始化对应的事件订阅"""
    global _launch_mode
    _launch_mode = mode
    logger.info(f"[WebChatRouter] 运行模式: {mode}")

    if mode == "full":
        _subscribe_eventbus()
    else:
        # IDE 模式同样走 EventBus 主链，因此也要订阅主会话下行事件。
        _subscribe_eventbus_lite()
    
    # 启动 /ws 心跳，防止 Web 前端 120s 看门狗在长 FC 循环中超时
    _start_ws_heartbeat()


def _is_full_mode() -> bool:
    return _launch_mode == "full"


# ── EventBus 订阅（Full 模式） ────────────────────────

_eventbus_lite_subscribed = False


def _subscribe_eventbus_lite() -> None:
    """IDE 模式精简订阅 — 订阅主会话链下行事件与图谱事件

    为了与 TSD 的 USER -> L1-B -> L2 主链保持一致，IDE 模式下的 /ws
    聊天也必须消费 EventBus 下行事件，而不能直接旁路调用 InferenceEngine。
    """
    global _eventbus_lite_subscribed
    if _eventbus_lite_subscribed:
        return
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType

        event_bus.subscribe(EventType.L2_OUTPUT, _on_l2_output, "WebChatRouter")
        event_bus.subscribe(EventType.L2_OUTPUT_STREAM, _on_l2_output_stream, "WebChatRouter")
        event_bus.subscribe(EventType.L2_THINKING_STEP, _on_l2_thinking_step, "WebChatRouter")
        event_bus.subscribe(EventType.MEMORY_GRAPH_UPDATED, _on_memory_graph_updated, "WebChatRouter")
        event_bus.subscribe(EventType.ACTION_SPEAK, _on_action_speak, "WebChatRouter")
        _eventbus_lite_subscribed = True
        logger.info(
            "[WebChatRouter] IDE 模式: 已订阅 L2_OUTPUT/L2_OUTPUT_STREAM/"
            "L2_THINKING_STEP/MEMORY_GRAPH_UPDATED/ACTION_SPEAK"
        )
    except Exception as e:
        logger.error(f"[WebChatRouter] IDE EventBus 精简订阅失败: {e}")

def _subscribe_eventbus() -> None:
    """订阅 EventBus 下行事件 — 将 L2 输出转发到 /ws 客户端"""
    global _eventbus_subscribed
    if _eventbus_subscribed:
        return
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType

        event_bus.subscribe(EventType.L2_OUTPUT, _on_l2_output, "WebChatRouter")
        event_bus.subscribe(EventType.L2_OUTPUT_STREAM, _on_l2_output_stream, "WebChatRouter")
        event_bus.subscribe(EventType.L2_THINKING_STEP, _on_l2_thinking_step, "WebChatRouter")
        event_bus.subscribe(EventType.MEMORY_GRAPH_UPDATED, _on_memory_graph_updated, "WebChatRouter")
        event_bus.subscribe(EventType.ACTION_SPEAK, _on_action_speak, "WebChatRouter")
        event_bus.subscribe(EventType.PROJECT_CREATED, _on_project_created, "WebChatRouter")
        _eventbus_subscribed = True
        logger.info("[WebChatRouter] 已订阅 EventBus 下行事件")
    except Exception as e:
        logger.error(f"[WebChatRouter] EventBus 订阅失败: {e}")


def _schedule_broadcast(message: dict) -> None:
    """从非 asyncio 线程安全地调度广播到所有 /ws 客户端"""
    loop = _event_loop
    msg_type = message.get("type", "?")
    if loop and loop.is_running():
        logger.info(f"[WebChatRouter] _schedule_broadcast: type={msg_type}, loop_running=True, ws_clients={len(_ws_clients)}")
        future = asyncio.run_coroutine_threadsafe(_broadcast(message), loop)
        # 捕获异步广播的异常
        def _on_done(f):
            exc = f.exception()
            if exc:
                logger.error(f"[WebChatRouter] _broadcast 异常: {exc}")
            else:
                logger.info(f"[WebChatRouter] _broadcast 完成: type={msg_type}")
        future.add_done_callback(_on_done)
    else:
        logger.warning(f"[WebChatRouter] _schedule_broadcast 跳过: type={msg_type}, loop={loop}, loop_running={loop.is_running() if loop else 'N/A'}")


async def _broadcast(message: dict) -> None:
    """向所有 /ws 客户端广播消息，根据各自协议版本自动选择格式"""
    msg_type = message.get("type", "?")
    _persist_web_visible_message(message)
    if not _ws_clients:
        logger.warning(f"[WebChatRouter] _broadcast: type={msg_type}, 无客户端连接")
        return
    logger.info(f"[WebChatRouter] _broadcast: type={msg_type}, 发送给 {len(_ws_clients)} 个客户端")
    dead: Set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await _send_to_ws(ws, message, persist=False)
            logger.info(f"[WebChatRouter] _broadcast: 成功发送 {msg_type}")
        except Exception as e:
            logger.error(f"[WebChatRouter] _broadcast: 发送失败 {msg_type}: {e}")
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── EventBus 回调（从分发线程调用，非 asyncio） ───────

def _on_l2_output(event) -> None:
    payload = event.payload or {}
    text = payload.get("display_text") or payload.get("text", "")
    speech_text = payload.get("speech_text") or ""
    request_id = payload.get("request_id")
    binding = _resolve_conversation_binding({
        "request_id": request_id,
        "turn_id": payload.get("turn_id"),
        "conversation_id": payload.get("conversation_id"),
        "session_id": payload.get("session_id"),
        "workspace_path": payload.get("workspace_path"),
        "project_id": payload.get("project_id"),
        "task_graph_id": payload.get("task_graph_id"),
        "payload": payload,
    })
    conversation_id = binding.get("conversation_id")
    logger.info(f"[WebChatRouter] _on_l2_output 被调用: text_len={len(text)}, request_id={request_id}, ws_clients={len(_ws_clients)}")
    if text:
        try:
            if conversation_id:
                decision = RouteDecision(
                    conversation_id=conversation_id,
                    turn_id=request_id or "",
                    text="",
                    workspace_path=binding.get("workspace_path"),
                    project_id=binding.get("project_id"),
                    task_graph_id=binding.get("task_graph_id"),
                )
                get_conversation_orchestrator().record_assistant_text(
                    decision,
                    text,
                    payload={
                        "raw_markdown": text,
                        "display_text": text,
                        "speech_text": speech_text,
                    },
                )
        except Exception:
            pass
        _schedule_broadcast({
            "type": "CHAT_RESPONSE",
            "text": text,
            "display_text": text,
            "speech_text": speech_text,
            "request_id": request_id,
            "session_id": conversation_id,
            "conversation_id": conversation_id,
            "workspace_path": binding.get("workspace_path"),
            "project_id": binding.get("project_id"),
            "task_graph_id": binding.get("task_graph_id"),
            "_persisted": True,
        })


def _on_l2_output_stream(event) -> None:
    text = event.payload.get("text", "")
    chunk = event.payload.get("chunk", "")
    request_id = event.payload.get("request_id")
    binding = _resolve_conversation_binding({
        "request_id": request_id,
        "turn_id": event.payload.get("turn_id"),
        "conversation_id": event.payload.get("conversation_id"),
        "session_id": event.payload.get("session_id"),
        "workspace_path": event.payload.get("workspace_path"),
        "project_id": event.payload.get("project_id"),
        "task_graph_id": event.payload.get("task_graph_id"),
        "payload": event.payload,
    })
    conversation_id = binding.get("conversation_id")
    if chunk or text:
        _schedule_broadcast({
            "type": "STREAMING_RESPONSE",
            "text": text,
            "chunk": chunk,
            "request_id": request_id,
            "session_id": conversation_id,
            "conversation_id": conversation_id,
            "workspace_path": binding.get("workspace_path"),
            "project_id": binding.get("project_id"),
            "task_graph_id": binding.get("task_graph_id"),
        })


def _on_l2_thinking_step(event) -> None:
    payload = event.payload
    if payload:
        _schedule_broadcast({"type": "THINKING_STEP", **payload})


def _on_memory_graph_updated(event) -> None:
    payload = event.payload
    if payload:
        _schedule_broadcast({"type": "MEMORY_GRAPH_UPDATE", **payload})


def _on_action_speak(event) -> None:
    text = event.payload.get("text", "")
    if text:
        logger.info(
            "[WebChatRouter] ACTION_SPEAK 仅后端记录，不广播到前端: "
            "text_len=%s, voice_mode=%s, preview=%r",
            len(text),
            event.payload.get("voice_mode"),
            text[:120],
        )


def _on_project_created(event) -> None:
    """项目创建事件 → 通知 Web 前端"""
    payload = event.payload
    if payload:
        _schedule_broadcast({
            "type": "PROJECT_CREATED",
            "project_id": payload.get("project_id", ""),
            "name": payload.get("name", ""),
            "path": payload.get("path", ""),
            "task_graph_id": payload.get("task_graph_id", ""),
            "status": payload.get("status", ""),
        })


# ── 消息处理 ──────────────────────────────────────────

async def _handle_chat_message(ws: WebSocket, data: dict) -> None:
    """处理 CHAT_MESSAGE: Web 总入口，按需自动激活编程能力"""
    text = data.get("text", "")
    referenced_nodes = data.get("referenced_nodes", [])

    if not text:
        return

    orchestrator = get_conversation_orchestrator()
    decision = orchestrator.prepare_turn(data, source=data.get("source") or "web_chat")
    data["session_id"] = decision.conversation_id
    data["conversation_id"] = decision.conversation_id
    data["request_id"] = decision.turn_id
    data["turn_id"] = decision.turn_id

    await _send_turn_accepted(ws, decision)

    logger.info(
        "[WebChatRouter] 消息: text=%s, mode=%s",
        text[:80],
        _launch_mode,
    )

    await _chat_via_eventbus(
        ws,
        text,
        decision.turn_id,
        decision.conversation_id,
        referenced_nodes,
        decision=decision,
    )

async def _send_turn_accepted(ws: WebSocket, decision: RouteDecision) -> None:
    payload = decision.to_payload()
    msg = {
        "type": "TURN_ACCEPTED",
        "request_id": decision.turn_id,
        "session_id": decision.conversation_id,
        "conversation_id": decision.conversation_id,
        "payload": payload,
    }
    await _send_to_ws(ws, msg)


async def _chat_via_eventbus(ws, text, request_id, session_id, referenced_nodes, decision=None):
    """Full 模式: 发布 USER_TEXT 到核心 EventBus → L1-B → L2"""
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType, EventPriority, ZulongEvent

        payload = {"text": text, "confidence": 1.0}
        if session_id:
            payload["session_id"] = session_id
        if request_id:
            payload["request_id"] = request_id
        if referenced_nodes:
            payload["referenced_nodes"] = referenced_nodes
        if decision:
            payload.update({
                "conversation_id": decision.conversation_id,
                "turn_id": decision.turn_id,
                "workspace_path": decision.workspace_path,
                "project_id": decision.project_id,
                "task_graph_id": decision.task_graph_id,
                "source": decision.source,
            })

        # TSD 23.11.3: BFS 自恢复 — 注入会话上下文
        if session_id:
            try:
                from zulong.ide.ide_server import _retrieve_session_context
                bfs_ctx = await _retrieve_session_context(session_id)
                if bfs_ctx:
                    payload["bfs_context"] = bfs_ctx
                    if bfs_ctx.get("task_graph_id") and not payload.get("task_graph_id"):
                        payload["task_graph_id"] = bfs_ctx["task_graph_id"]
                    logger.info(
                        f"[WebChatRouter] BFS 上下文注入: "
                        f"session={session_id}, "
                        f"rounds={len(bfs_ctx.get('recent_rounds', []))}, "
                        f"tasks={len(bfs_ctx.get('active_tasks', []))}"
                    )
            except Exception as _e:
                logger.debug(f"[WebChatRouter] BFS 上下文跳过: {_e}")

        event = ZulongEvent(
            type=EventType.USER_TEXT,
            source="launcher/web_ui",
            payload=payload,
            priority=EventPriority.NORMAL,
        )
        # 在线程池中发布事件，避免阻塞 asyncio 事件循环
        # event_bus.publish(USER_TEXT) 会同步调用 L1-B → L2 全链路，
        # 如果在事件循环线程中执行会阻塞 WebSocket ping/pong 和消息广播
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, event_bus.publish, event)
        logger.info("[WebChatRouter] USER_TEXT 已发布到 EventBus (via executor)")
    except Exception as e:
        logger.error(f"[WebChatRouter] EventBus 发布失败: {e}", exc_info=True)
        msg = {
            "type": "CHAT_RESPONSE",
            "text": _failure_response_text(f"消息无法提交到推理链路：{e}"),
            "request_id": request_id,
        }
        try:
            await _send_to_ws(ws, msg)
        except Exception:
            await _broadcast(msg)


async def _chat_via_engine(ws, text, request_id, session_id, referenced_nodes, decision=None):
    """兼容入口：统一转发到 EventBus 主链，禁止绕过 L1-B 直达 L2。"""
    logger.warning("[WebChatRouter] _chat_via_engine 已弃用，转发到 _chat_via_eventbus")
    await _chat_via_eventbus(
        ws,
        text,
        request_id,
        session_id,
        referenced_nodes,
        decision=decision,
    )


async def _retrieve_memory_context_for_prompt(text: str, top_k: int = 3) -> str:
    """为首轮 prompt 获取轻量记忆上下文。

    调用方会用短超时等待；这里独立成 task，避免阻塞 LLM 首字。
    """
    try:
        mg = _get_active_memory_graph()
        if not mg:
            return ""
        context_results = await mg.retrieve_context(text, top_k=top_k)
        memory_lines = []
        for r in context_results or []:
            label = r.get("label", "")
            content = r.get("content", "")
            if label or content:
                memory_lines.append(f"- {label}: {content[:200]}")
        return "\n".join(memory_lines)
    except Exception as e:
        logger.debug(f"[WebChatRouter] retrieve memory context failed: {e}")
        return ""


async def _sync_ide_cancel(cancel_evt: asyncio.Event, session_id: str):
    """监听IDE端cancel_event并同步到Web端cancel_evt"""
    while True:
        await asyncio.sleep(1.0)
        try:
            from zulong.ide.ide_server import _sessions
            # 检查是否有IDE端对应session被取消
            for sid, sess in _sessions.items():
                if hasattr(sess, 'cancel_event') and sess.cancel_event.is_set():
                    cancel_evt.set()
                    logger.info(f"[WebChatRouter] IDE端取消已同步到Web端")
                    return
            # 也检查引擎级中断标志
            from zulong.l2.inference_engine import get_inference_engine
            engine = get_inference_engine()
            if engine and engine._interrupt_flag:
                cancel_evt.set()
                logger.info(f"[WebChatRouter] 引擎中断标志已同步到Web端")
                return
        except Exception:
            pass


async def _handle_stop_generation(data: dict) -> None:
    """处理停止生成请求 — 支持单session停止和全局停止"""
    request_id = data.get("request_id")
    session_id = data.get("session_id")

    # ── 核心：设置 IDE FC Runner 的 cancel_event ──
    stopped_sessions = 0
    try:
        from zulong.ide.ide_server import _sessions, _engine_instance
        if session_id:
            # 停止指定 session
            sess = _sessions.get(session_id)
            if sess and hasattr(sess, 'cancel_event') and sess.cancel_event:
                sess.cancel_event.set()
                stopped_sessions = 1
                logger.info(f"[WebChatRouter] 停止指定 session: {session_id[:12]}")
        else:
            # 无 session_id 时停止所有 session
            for sid, sess in _sessions.items():
                if hasattr(sess, 'cancel_event') and sess.cancel_event:
                    sess.cancel_event.set()
                    stopped_sessions += 1
        if _engine_instance and hasattr(_engine_instance, '_interrupt_flag'):
            _engine_instance._interrupt_flag = True
        if stopped_sessions:
            logger.info(
                f"[WebChatRouter] 停止生成: 已设置 {stopped_sessions} 个 session 的 cancel_event")
    except Exception as e:
        logger.warning(f"[WebChatRouter] 设置 cancel_event 失败: {e}")

    if _is_full_mode():
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, EventPriority, ZulongEvent
            event = ZulongEvent(
                type=EventType.USER_TEXT,
                source="launcher/web_ui",
                payload={"action": "stop_generation", "request_id": request_id},
                priority=EventPriority.HIGH,
            )
            event_bus.publish(event)
        except Exception as e:
            logger.error(f"[WebChatRouter] 停止生成失败: {e}")
    else:
        if request_id and request_id in _chat_cancels:
            _chat_cancels[request_id].set()
            logger.info(f"[WebChatRouter] 已取消: {request_id}")


async def _handle_conversation_switch(data: dict) -> None:
    conversation_id = data.get("conversation_id") or data.get("session_id")
    if not conversation_id:
        return
    try:
        get_conversation_orchestrator().store.set_active_conversation(conversation_id)
    except Exception as e:
        logger.debug(f"[WebChatRouter] conversation switch skipped: {e}")


async def _handle_chat_visible_message(data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
    conversation_id = payload.get("conversation_id") or payload.get("session_id")
    text = (payload.get("text") or "").strip()
    role = payload.get("role") or "assistant"
    if not conversation_id or not text or role not in ("user", "assistant", "tool"):
        return
    try:
        get_interaction_store().upsert_conversation(
            conversation_id,
            title=text[:20] if role == "user" else "",
            source=payload.get("source") or "web_ui",
            workspace_path=payload.get("workspace_path") or payload.get("cwd"),
            project_id=payload.get("project_id"),
            task_graph_id=payload.get("task_graph_id"),
            active=True,
        )
        get_interaction_store().append_event(
            conversation_id=conversation_id,
            turn_id=payload.get("turn_id") or payload.get("request_id"),
            event_type="visible_message",
            role=role,
            source=payload.get("source") or "web_ui",
            text=text,
            payload=payload,
            workspace_path=payload.get("workspace_path") or payload.get("cwd"),
            project_id=payload.get("project_id"),
            task_graph_id=payload.get("task_graph_id"),
        )
    except Exception as exc:
        logger.debug(f"[WebChatRouter] visible message 持久化失败: {exc}")


async def _handle_voice_bind(ws: WebSocket, data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else data
    try:
        result = get_conversation_orchestrator().route_voice_record(payload)
        await _send_to_ws(ws, {
            "type": "voice:record_created",
            "payload": result,
            "voice_event_id": result.get("voice_event_id"),
            "conversation_id": result.get("conversation_id"),
        })
        if result.get("linked"):
            await _send_to_ws(ws, {
                "type": "voice:record_linked",
                "payload": result,
                "voice_event_id": result.get("voice_event_id"),
                "conversation_id": result.get("conversation_id"),
            })
    except Exception as e:
        logger.error(f"[WebChatRouter] voice bind failed: {e}", exc_info=True)
        await _send_to_ws(ws, {
            "type": "voice:record_created",
            "payload": {"linked": False, "error": str(e)},
        })


async def _handle_voice_list(ws: WebSocket, data: dict) -> None:
    """列出语音记录，支持过滤和分页。"""
    store = get_interaction_store()
    limit = data.get("limit", 50)
    offset = data.get("offset", 0)
    linked_only = data.get("linked_only")
    if linked_only is not None:
        linked_only = bool(linked_only)
    records = store.list_voice_records(limit=limit, offset=offset, linked_only=linked_only)
    await _send_to_ws(ws, {
        "type": "voice:list",
        "payload": {
            "records": records,
            "count": len(records),
            "limit": limit,
            "offset": offset,
        },
    })


async def _handle_voice_delete(ws: WebSocket, data: dict) -> None:
    """删除单条语音记录。"""
    voice_event_id = data.get("voice_event_id")
    if not voice_event_id:
        await _send_to_ws(ws, {
            "type": "voice:delete",
            "payload": {"ok": False, "error": "缺少 voice_event_id"},
        })
        return
    store = get_interaction_store()
    ok = store.delete_voice_record(voice_event_id)
    await _send_to_ws(ws, {
        "type": "voice:delete",
        "payload": {"ok": ok, "voice_event_id": voice_event_id},
    })


async def _handle_ide_action(ws: WebSocket, action: str, data: dict) -> None:
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else dict(data)
    payload.setdefault("conversation_id", data.get("conversation_id") or data.get("session_id"))
    payload.setdefault("turn_id", data.get("turn_id") or data.get("request_id"))
    try:
        from zulong.ide.ide_server import request_ide_action
        result = await request_ide_action(action, payload)
        await _send_to_ws(ws, {
            "type": "ide_action_result",
            "action": action,
            "payload": result,
            "conversation_id": payload.get("conversation_id"),
            "request_id": payload.get("turn_id"),
        })
    except Exception as e:
        await _send_to_ws(ws, {
            "type": "ide_action_result",
            "action": action,
            "payload": {"ok": False, "error": str(e)},
            "conversation_id": payload.get("conversation_id"),
            "request_id": payload.get("turn_id"),
        })


_EXECUTION_NODE_TYPES = {"tool_call", "tool_result", "approval"}
_MEMORY_GRAPH_EXECUTION_BACKFILL_DONE = False


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _memory_node_id(node: Any) -> str:
    if isinstance(node, dict):
        return str(node.get("id") or node.get("node_id") or "")
    return str(getattr(node, "node_id", "") or getattr(node, "id", "") or "")


def _memory_node_type(node: Any) -> str:
    if isinstance(node, dict):
        return _enum_value(node.get("type") or node.get("node_type"))
    return _enum_value(getattr(node, "node_type", "") or getattr(node, "type", ""))


def _memory_node_metadata(node: Any) -> Dict[str, Any]:
    if isinstance(node, dict):
        meta = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        return dict(meta)
    meta = getattr(node, "metadata", None)
    return dict(meta or {}) if isinstance(meta, dict) else {}


def _get_memory_node(mg: Any, node_id: str) -> Optional[Any]:
    try:
        if hasattr(mg, "get_node"):
            return mg.get_node(node_id)
    except Exception:
        return None
    return None


def _serialize_memory_node_for_snapshot(mg: Any, node: Any) -> Dict[str, Any]:
    node_id = _memory_node_id(node)
    node_type = _memory_node_type(node)
    metadata = _memory_node_metadata(node)
    content = getattr(node, "content", None)
    if content and "content" not in metadata:
        metadata["content"] = content
    content_summary = getattr(node, "content_summary", None)
    if content_summary and "content_summary" not in metadata:
        metadata["content_summary"] = content_summary
    if node_type in _EXECUTION_NODE_TYPES:
        metadata["is_execution_event"] = True

    label = ""
    activation = 0.0
    if isinstance(node, dict):
        label = str(node.get("label") or node_id)
        try:
            activation = float(node.get("activation") or 0.0)
        except (TypeError, ValueError):
            activation = 0.0
    else:
        label = str(getattr(node, "label", "") or node_id)
        try:
            activation = float(getattr(node, "activation", 0.0) or 0.0)
        except (TypeError, ValueError):
            activation = 0.0

    data = {
        "id": node_id,
        "type": node_type,
        "label": label,
        "activation": round(activation, 3),
        "metadata": metadata,
    }
    for attr in ("backend_ref", "created_at", "last_accessed", "access_count"):
        value = node.get(attr) if isinstance(node, dict) else getattr(node, attr, None)
        if value not in (None, ""):
            data[attr] = value
    try:
        if hasattr(mg, "get_children"):
            data["children_count"] = len(mg.get_children(node_id))
    except Exception:
        pass
    return data


def _serialize_memory_edge_for_snapshot(src: str, dst: str, edge_type: Any, edge_data: Any = None) -> Dict[str, Any]:
    if isinstance(edge_data, dict):
        edge_type = edge_data.get("edge_type") or edge_data.get("type") or edge_type
        weight = edge_data.get("weight", 1.0)
        protected = edge_data.get("protected", False)
        metadata = edge_data.get("metadata", {})
    else:
        edge_type = getattr(edge_data, "edge_type", edge_type)
        weight = getattr(edge_data, "weight", 1.0)
        protected = getattr(edge_data, "protected", False)
        metadata = getattr(edge_data, "metadata", {})
    try:
        weight = round(float(weight), 3)
    except (TypeError, ValueError):
        weight = 1.0
    return {
        "source": src,
        "target": dst,
        "type": _enum_value(edge_type) or "reference",
        "weight": weight,
        "protected": bool(protected),
        "metadata": metadata if isinstance(metadata, dict) else {},
    }


def _collect_execution_edges_networkx(mg: Any, execution_ids: Set[str]) -> List[Dict[str, Any]]:
    graph = getattr(mg, "_graph", None)
    if graph is None or not hasattr(graph, "edges"):
        return []
    edges: List[Dict[str, Any]] = []
    for src, dst, data in graph.edges(data=True):
        if src in execution_ids or dst in execution_ids:
            edges.append(_serialize_memory_edge_for_snapshot(str(src), str(dst), data.get("edge_type"), data))
    return edges


def _collect_execution_edges_sharded(mg: Any, execution_ids: Set[str]) -> List[Dict[str, Any]]:
    if not hasattr(mg, "list_all_shards") or not hasattr(mg, "get_shard"):
        return []
    edges: List[Dict[str, Any]] = []
    for shard_id in mg.list_all_shards():
        try:
            shard = mg.get_shard(shard_id, load_if_missing=True)
        except TypeError:
            shard = mg.get_shard(shard_id)
        except Exception:
            shard = None
        if not shard or not getattr(shard, "topology", None):
            continue
        graph = getattr(shard.topology, "graph", None)
        if graph is None:
            continue
        for edge in graph.es:
            try:
                src = mg._vertex_node_id(graph.vs[edge.source]) if hasattr(mg, "_vertex_node_id") else graph.vs[edge.source]["name"]
                dst = mg._vertex_node_id(graph.vs[edge.target]) if hasattr(mg, "_vertex_node_id") else graph.vs[edge.target]["name"]
                if src not in execution_ids and dst not in execution_ids:
                    continue
                edge_type = edge["type"] if "type" in edge.attributes() else "association"
                edge_props = shard.get_edge(src, dst) if hasattr(shard, "get_edge") else None
                edges.append(_serialize_memory_edge_for_snapshot(str(src), str(dst), edge_type, edge_props))
            except Exception:
                continue
    return edges


def _get_memory_graph_execution_view(mg: Any) -> Dict[str, Any]:
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    execution_ids: Set[str] = set()

    if hasattr(mg, "get_nodes_by_type"):
        for node_type in sorted(_EXECUTION_NODE_TYPES):
            try:
                nodes = mg.get_nodes_by_type(node_type)
            except Exception:
                nodes = []
            for node in nodes or []:
                node_id = _memory_node_id(node)
                if not node_id:
                    continue
                execution_ids.add(node_id)
                nodes_by_id[node_id] = _serialize_memory_node_for_snapshot(mg, node)

    edges = _collect_execution_edges_networkx(mg, execution_ids)
    if not edges:
        edges = _collect_execution_edges_sharded(mg, execution_ids)

    for edge in edges:
        for endpoint in (edge.get("source"), edge.get("target")):
            if endpoint and endpoint not in nodes_by_id:
                node = _get_memory_node(mg, str(endpoint))
                if node:
                    nodes_by_id[str(endpoint)] = _serialize_memory_node_for_snapshot(mg, node)

    nodes = list(nodes_by_id.values())
    nodes.sort(key=lambda item: (item.get("created_at") or 0, item.get("id") or ""))

    type_counts: Dict[str, int] = {node_type: 0 for node_type in sorted(_EXECUTION_NODE_TYPES)}
    for node_id in execution_ids:
        node_type = nodes_by_id.get(node_id, {}).get("type")
        if node_type in type_counts:
            type_counts[node_type] += 1

    return {
        "nodes": nodes,
        "edges": edges,
        "execution_node_ids": sorted(execution_ids),
        "stats": {
            "total_execution_nodes": len(execution_ids),
            "total_execution_edges": len(edges),
            **type_counts,
        },
    }


def _ensure_execution_events_backfilled(mg: Any) -> None:
    global _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE
    if _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE:
        return
    _MEMORY_GRAPH_EXECUTION_BACKFILL_DONE = True
    try:
        from zulong.launcher.memory_mirror import backfill_recent_interactions_to_memory_graph

        count = backfill_recent_interactions_to_memory_graph(limit=20, events_per_conversation=800)
        if count:
            logger.info("[WebChatRouter] MemoryGraph 执行事件回填完成: %s events", count)
    except Exception as exc:
        logger.debug("[WebChatRouter] MemoryGraph 执行事件回填跳过: %s", exc)


def _merge_execution_view_into_payload(payload: Dict[str, Any], execution_view: Dict[str, Any]) -> Dict[str, Any]:
    if not execution_view:
        return payload
    payload = dict(payload or {})
    nodes = list(payload.get("nodes") or [])
    edges = list(payload.get("edges") or [])

    node_ids = {str(node.get("id")) for node in nodes if isinstance(node, dict) and node.get("id")}
    for node in execution_view.get("nodes") or []:
        node_id = str(node.get("id") or "")
        if node_id and node_id not in node_ids:
            nodes.append(node)
            node_ids.add(node_id)

    edge_keys = {
        (
            str(edge.get("source") if not isinstance(edge.get("source"), dict) else edge.get("source", {}).get("id")),
            str(edge.get("target") if not isinstance(edge.get("target"), dict) else edge.get("target", {}).get("id")),
            str(edge.get("type") or ""),
        )
        for edge in edges
        if isinstance(edge, dict)
    }
    for edge in execution_view.get("edges") or []:
        key = (str(edge.get("source") or ""), str(edge.get("target") or ""), str(edge.get("type") or ""))
        if key[0] and key[1] and key not in edge_keys:
            edges.append(edge)
            edge_keys.add(key)

    stats = dict(payload.get("stats") or {})
    execution_stats = dict(execution_view.get("stats") or {})
    stats["execution"] = execution_stats
    stats["execution_nodes"] = execution_stats.get("total_execution_nodes", 0)

    payload["nodes"] = nodes
    payload["edges"] = edges
    payload["stats"] = stats
    payload["execution_view"] = execution_view
    payload["execution_node_ids"] = execution_view.get("execution_node_ids") or []
    return payload


async def _push_memory_graph_snapshot(ws: WebSocket) -> None:
    """推送记忆图谱快照到指定 WebSocket"""
    try:
        payload = _get_memory_graph_snapshot_payload()
        if payload:
            await _send_to_ws(ws, {
                "type": "MEMORY_GRAPH_UPDATE",
                **payload,
            })
    except Exception as e:
        logger.debug(f"[WebChatRouter] 推送记忆图谱失败: {e}")
        try:
            await _send_to_ws(ws, {
                "type": "MEMORY_GRAPH_UPDATE",
                "update_type": "error",
                "ts": time.time(),
                "nodes": [],
                "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0},
                "error": str(e),
            })
        except Exception:
            pass


def _get_memory_graph_snapshot_payload() -> dict:
    try:
        mg = _get_active_memory_graph()
        if not mg:
            return {
                "update_type": "full",
                "ts": time.time(),
                "nodes": [],
                "edges": [],
                "stats": {"total_nodes": 0, "total_edges": 0},
            }
        if hasattr(mg, "to_frontend_dict"):
            payload = mg.to_frontend_dict(depth=0)
        elif hasattr(mg, "get_snapshot_for_frontend"):
            payload = mg.get_snapshot_for_frontend()
        else:
            payload = {}
        _ensure_execution_events_backfilled(mg)
        execution_view = _get_memory_graph_execution_view(mg)
        payload = _merge_execution_view_into_payload(payload or {}, execution_view)
        return {
            "update_type": "full",
            "ts": time.time(),
            **(payload or {}),
        }
    except Exception as exc:
        return {
            "update_type": "error",
            "ts": time.time(),
            "nodes": [],
            "edges": [],
            "stats": {"total_nodes": 0, "total_edges": 0},
            "error": str(exc),
        }


@router.get("/api/memory-graph/snapshot")
async def get_memory_graph_snapshot():
    """Return a fast MemoryGraph frontend snapshot for Web fallback loading."""
    return _get_memory_graph_snapshot_payload()


async def _handle_expand_node(node_id: str, ws: WebSocket) -> None:
    """处理展开节点请求"""
    try:
        mg = _get_active_memory_graph()
        if not mg:
            return
        if hasattr(mg, "get_node_children_for_frontend"):
            result = mg.get_node_children_for_frontend(node_id)
        elif hasattr(mg, "get_neighbors"):
            neighbors = mg.get_neighbors(node_id)
            result = {"node_id": node_id, "neighbors": neighbors}
        else:
            return
        await _send_to_ws(ws, {
            "type": "MEMORY_GRAPH_EXPAND_RESULT",
            "ts": time.time(),
            **result,
        })
    except Exception as e:
        logger.debug(f"[WebChatRouter] 展开节点失败: {e}")


# ── 对话会话管理（Web 前端会话栏重建） ─────────────────

async def _handle_list_dialogue_sessions(ws: WebSocket) -> None:
    """查询 MemoryGraph 中所有对话会话节点，返回会话列表"""
    try:
        session_store = _collect_dialogue_sessions()
        await _send_to_ws(ws, {
            "type": "SESSION_LIST",
            "ts": time.time(),
            "sessions": session_store["sessions"],
            "activeSessionId": session_store.get("activeSessionId"),
        })
        logger.info(f"[WebChatRouter] SESSION_LIST: 返回 {len(session_store['sessions'])} 个会话")
    except Exception as e:
        logger.error(f"[WebChatRouter] LIST_DIALOGUE_SESSIONS 失败: {e}", exc_info=True)
        await _send_to_ws(ws, {
            "type": "SESSION_LIST",
            "ts": time.time(),
            "sessions": [],
            "error": str(e),
        })


async def _handle_get_session_messages(ws: WebSocket, session_id: str) -> None:
    """获取指定会话的完整消息列表"""
    try:
        try:
            store_messages = get_conversation_orchestrator().store.get_messages(session_id)
            if store_messages:
                await _send_to_ws(ws, {
                    "type": "SESSION_MESSAGES",
                    "ts": time.time(),
                    "session_id": session_id,
                    "messages": store_messages,
                })
                return
        except Exception as store_err:
            logger.debug(f"[WebChatRouter] interaction store messages skipped: {store_err}")

        from zulong.memory.graph_adapters import DialogueAdapter

        mg = _get_active_memory_graph()
        if not mg:
            await _send_to_ws(ws, {
                "type": "SESSION_MESSAGES",
                "ts": time.time(),
                "session_id": session_id,
                "messages": [],
            })
            return

        messages = DialogueAdapter.get_session_messages(mg, session_id)
        await _send_to_ws(ws, {
            "type": "SESSION_MESSAGES",
            "ts": time.time(),
            "session_id": session_id,
            "messages": messages,
        })
        logger.info(
            f"[WebChatRouter] SESSION_MESSAGES: {session_id} → {len(messages)} 条消息")
    except Exception as e:
        logger.error(f"[WebChatRouter] GET_SESSION_MESSAGES 失败: {e}", exc_info=True)
        await _send_to_ws(ws, {
            "type": "SESSION_MESSAGES",
            "ts": time.time(),
            "session_id": session_id,
            "messages": [],
            "error": str(e),
        })


async def _handle_delete_dialogue_session(ws: WebSocket, session_id: str) -> None:
    """删除对话会话及其所有子节点（BFS 级联删除），同步清理 TaskGraph/AgentSessionStore/IDESession"""
    try:
        mg = _get_active_memory_graph()
        if not mg or not mg.has_node(session_id):
            await _send_to_ws(ws, {
                "type": "SESSION_DELETED",
                "ts": time.time(),
                "session_id": session_id,
                "nodes_deleted": 0,
                "error": "会话不存在",
            })
            return

        # BFS 收集 HIERARCHY 子节点
        nodes_to_remove = [session_id]
        queue = [session_id]
        while queue:
            parent = queue.pop(0)
            if hasattr(mg, '_graph') and parent in mg._graph:
                for child_id in list(mg._graph.successors(parent)):
                    edge_data = mg._graph[parent].get(child_id, {})
                    if edge_data.get("edge_type") == "hierarchy":
                        nodes_to_remove.append(child_id)
                        queue.append(child_id)

        # 从叶子到根逐个删除 MemoryGraph 节点
        mg_deleted_count = 0
        for nid in reversed(nodes_to_remove):
            if mg.remove_node(nid):
                mg_deleted_count += 1

        if hasattr(mg, "save_all"):
            mg.save_all()

        # 同步清理 TaskGraph 中的任务节点
        tg_deleted_count = 0
        for nid in nodes_to_remove:
            if nid.startswith("task:tg_"):
                try:
                    from zulong.tools.task_tools import get_active_task_graph
                    tg = get_active_task_graph()
                    if tg and tg.has_node(nid):
                        tg.remove_node(nid)
                        tg_deleted_count += 1
                except Exception:
                    pass
            else:
                try:
                    from zulong.tools.task_tools import get_active_task_graph
                    tg = get_active_task_graph()
                    if tg and tg.has_node(nid):
                        tg.remove_node(nid)
                        tg_deleted_count += 1
                except Exception:
                    pass

        # 同步清理 AgentSessionStore
        try:
            from zulong.ide.ide_server import get_session_store
            store = get_session_store()
            store.delete(session_id)
        except Exception:
            pass

        # 同步清理 IDE _sessions（如存在）
        ide_session_cleared = False
        try:
            from zulong.ide.ide_server import _sessions
            if session_id in _sessions:
                sess = _sessions[session_id]
                if sess.fc_task and not sess.fc_task.done():
                    sess.cancel_event.set()
                    sess.fc_task.cancel()
                _sessions.pop(session_id, None)
                ide_session_cleared = True
        except Exception:
            pass

        # 广播删除事件
        try:
            from zulong.ide.ide_server import broadcast_monitor_event
            await broadcast_monitor_event("SESSION_DELETED", {
                "session_id": session_id,
                "mg_nodes_deleted": mg_deleted_count,
                "tg_nodes_deleted": tg_deleted_count,
            })
        except Exception:
            pass

        await _send_to_ws(ws, {
            "type": "SESSION_DELETED",
            "ts": time.time(),
            "session_id": session_id,
            "nodes_deleted": mg_deleted_count,
            "tg_nodes_deleted": tg_deleted_count,
        })
        logger.info(
            f"[WebChatRouter] DELETE_DIALOGUE_SESSION: {session_id} → "
            f"MG删除{mg_deleted_count}个, TG删除{tg_deleted_count}个, "
            f"IDE session={'已清除' if ide_session_cleared else '无'}")
    except Exception as e:
        logger.error(f"[WebChatRouter] DELETE_DIALOGUE_SESSION 失败: {e}", exc_info=True)
        await _send_to_ws(ws, {
            "type": "SESSION_DELETED",
            "ts": time.time(),
            "session_id": session_id,
            "nodes_deleted": 0,
            "error": str(e),
        })


# ── REST API 端点 ────────────────────────────────────

@router.get("/api/chat/sessions")
async def list_sessions_rest():
    return _collect_dialogue_sessions()

@router.delete("/api/chat/sessions/{session_id}")
async def delete_session_rest(session_id: str, cascade: bool = True):
    """REST API 删除会话（级联清理 MemoryGraph/TaskGraph/AgentSessionStore）"""
    mg_nodes_deleted = 0
    tg_nodes_deleted = 0

    if cascade:
        try:
            mg = _get_active_memory_graph()
            if mg and mg.has_node(session_id):
                nodes_to_remove = [session_id]
                queue = [session_id]
                while queue:
                    parent = queue.pop(0)
                    if hasattr(mg, '_graph') and parent in mg._graph:
                        for child_id in list(mg._graph.successors(parent)):
                            edge_data = mg._graph[parent].get(child_id, {})
                            if edge_data.get("edge_type") == "hierarchy":
                                nodes_to_remove.append(child_id)
                                queue.append(child_id)
                for nid in reversed(nodes_to_remove):
                    if mg.remove_node(nid):
                        mg_nodes_deleted += 1
                if hasattr(mg, "save_all"):
                    mg.save_all()
        except Exception as e:
            logger.warning(f"[REST] 级联删除MG失败: {e}")

        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg and tg.has_node(session_id):
                if tg.remove_node(session_id):
                    tg_nodes_deleted += 1
        except Exception:
            pass

        try:
            from zulong.ide.ide_server import get_session_store
            store = get_session_store()
            store.delete(session_id)
        except Exception:
            pass

        try:
            from zulong.ide.ide_server import _sessions
            if session_id in _sessions:
                sess = _sessions[session_id]
                if sess.fc_task and not sess.fc_task.done():
                    sess.cancel_event.set()
                    sess.fc_task.cancel()
                _sessions.pop(session_id, None)
        except Exception:
            pass

    return {
        "session_id": session_id,
        "deleted": True,
        "mg_nodes_deleted": mg_nodes_deleted,
        "tg_nodes_deleted": tg_nodes_deleted,
    }


async def _send_to_ws(ws: WebSocket, message: dict, *, persist: bool = True) -> bool:
    """向 WebSocket 发送消息，根据协议版本自动选择格式

    统一协议客户端收到 `{msg_id, type: ns:action, session_id, ts, payload}` 格式。
    旧客户端收到传统的 `{type: UPPER_CASE, ...fields}` 格式。
    """
    ws_id = id(ws)
    protocol = _ws_protocols.get(ws_id, "legacy")
    client_type = _ws_client_types.get(ws_id, "unknown")
    if persist:
        _persist_web_visible_message(message)

    if protocol == "unified" and ":" not in message.get("type", ""):
        # 旧格式 → 统一格式。只要客户端完成 handshake，就按统一协议下发。
        message = _protocol_bridge.to_unified(message, format_version="web")
    elif protocol == "unified" and ":" in message.get("type", "") and "payload" not in message:
        payload = {k: v for k, v in message.items() if k not in ("type", "request_id", "session_id", "conversation_id")}
        message = make_unified_message(
            message["type"],
            payload,
            session_id=message.get("conversation_id") or message.get("session_id", ""),
        )

    try:
        await ws.send_json(message)
        return True
    except WebSocketDisconnect:
        logger.debug("[WebChatRouter] WebSocket 已断开，跳过发送")
        return False
    except Exception as e:
        if "close message has been sent" in str(e) or "ClientDisconnected" in type(e).__name__:
            logger.debug(f"[WebChatRouter] WebSocket 已关闭，跳过发送: {e}")
            return False
        raise


async def _send_initial_dashboard_state(ws: WebSocket) -> None:
    """发送 Dashboard 建连后的初始状态。"""
    engine_ready = False
    task_graph_snapshot = None
    memory_graph_stats = None
    code_anchor_stats = None
    try:
        from zulong.ide.ide_server import _get_engine
        engine_ready = _get_engine() is not None
    except Exception:
        pass
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg:
            task_graph_snapshot = tg.to_frontend_dict()
    except Exception:
        pass
    try:
        mg = _get_active_memory_graph()
        if mg:
            memory_graph_stats = {
                "total_nodes": mg._stats.get("total_nodes", 0),
                "total_edges": mg._stats.get("total_edges", 0),
            }
    except Exception:
        pass
    try:
        from zulong.memory.code_anchor import get_code_anchor_store
        store = get_code_anchor_store()
        if store:
            code_anchor_stats = store.get_stats()
    except Exception:
        pass

    active_sessions_info = []
    try:
        from zulong.ide.ide_server import _sessions as _ide_sessions
        active_sessions_info = [s.to_info_dict() for s in _ide_sessions.values()]
    except Exception:
        pass

    return await _send_to_ws(ws, {
        "type": "WELCOME",
        "ts": time.time(),
        "payload": {
            "engine_ready": engine_ready,
            "launch_mode": _launch_mode,
            "task_graph": task_graph_snapshot,
            "memory_graph_stats": memory_graph_stats,
            "code_anchor_stats": code_anchor_stats,
            "active_sessions": active_sessions_info,
        },
    })


async def handle_unified_root_ws(
    ws: WebSocket,
    *,
    accepted: bool = False,
    initial_msg: Optional[dict] = None,
) -> None:
    """统一根 WebSocket 入口中的 dashboard/monitor 分支。

    该函数复用 /ws 的处理语义，只是连接已经完成 handshake，因此不再要求
    客户端走旧的 /ws 路径。
    """
    global _event_loop
    if not accepted:
        await ws.accept()
    _ws_clients.add(ws)
    _ws_protocols[id(ws)] = "unified"

    if _event_loop is None:
        _event_loop = asyncio.get_running_loop()

    try:
        from zulong.ide.ide_server import _monitor_connections
        _monitor_connections.add(ws)
        import zulong.ide.ide_server as _ide_srv
        if _ide_srv._main_event_loop is None:
            _ide_srv._main_event_loop = _event_loop
    except Exception:
        pass

    if initial_msg:
        payload = initial_msg.get("payload", {}) if isinstance(initial_msg, dict) else {}
        _ws_client_types[id(ws)] = payload.get("client_type", "dashboard")

    if not await _send_to_ws(ws, make_unified_message(
        MessageType.TASK_ACK,
        {
            "status": "ok",
            "client_type": _ws_client_types.get(id(ws), "dashboard"),
        },
    )):
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        return
    if not await _send_initial_dashboard_state(ws):
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        return
    try:
        await _push_memory_graph_snapshot(ws)
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                fmt = _protocol_bridge.detect_format(data)
                unified = _protocol_bridge.to_unified(data, fmt)
                msg_type = unified.get("type", "")

                if msg_type == MessageType.HANDSHAKE:
                    payload = unified.get("payload", {})
                    _ws_client_types[id(ws)] = payload.get("client_type", "dashboard")
                    await _send_to_ws(ws, make_unified_message(
                        MessageType.TASK_ACK,
                        {"status": "ok", "client_type": _ws_client_types[id(ws)]},
                    ))
                    continue

                if msg_type == MessageType.PING:
                    await _send_to_ws(ws, make_unified_message(MessageType.PONG, {}))
                    continue

                legacy = _protocol_bridge.from_unified(unified, target_format="web", direction="uplink")
                legacy_type = legacy.get("type", "")
                if legacy_type == "CHAT_MESSAGE":
                    asyncio.create_task(_handle_chat_message(ws, legacy))
                elif legacy_type in ("STOP_GENERATION", "STOP_TASK"):
                    asyncio.create_task(_handle_stop_generation(legacy))
                elif msg_type == "conversation:switch":
                    asyncio.create_task(_handle_conversation_switch(unified.get("payload", {})))
                elif msg_type == "chat:visible_message":
                    asyncio.create_task(_handle_chat_visible_message(unified))
                elif msg_type == MessageType.VOICE_BIND:
                    asyncio.create_task(_handle_voice_bind(ws, unified.get("payload", {})))
                elif msg_type in (
                    MessageType.IDE_OPEN_WORKSPACE,
                    MessageType.IDE_OPEN_FILE,
                    MessageType.IDE_OPEN_TERMINAL,
                    MessageType.IDE_SHOW_DIFF,
                    MessageType.IDE_GET_CONTEXT,
                    MessageType.IDE_APPROVAL_RESULT,
                    "ide:execute_tool",
                ):
                    asyncio.create_task(_handle_ide_action(ws, msg_type, unified))
                elif legacy_type == "REQUEST_MEMORY_GRAPH":
                    asyncio.create_task(_push_memory_graph_snapshot(ws))
                elif legacy_type == "EXPAND_NODE":
                    node_id = legacy.get("node_id")
                    if node_id:
                        asyncio.create_task(_handle_expand_node(node_id, ws))
                elif legacy_type == "LIST_DIALOGUE_SESSIONS":
                    asyncio.create_task(_handle_list_dialogue_sessions(ws))
                elif legacy_type == "GET_SESSION_MESSAGES":
                    session_id = legacy.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_get_session_messages(ws, session_id))
                elif legacy_type == "DELETE_DIALOGUE_SESSION":
                    session_id = legacy.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_delete_dialogue_session(ws, session_id))
                elif legacy_type == "audio_start":
                    asyncio.create_task(_handle_audio_start_web(ws, legacy))
                elif legacy_type == "audio_chunk":
                    asyncio.create_task(_handle_audio_chunk_web(ws, legacy))
                elif legacy_type == "audio_end":
                    asyncio.create_task(_handle_audio_end_web(ws, legacy))
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.debug(f"[WebChatRouter] 统一入口消息处理异常: {e}")
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        try:
            from zulong.ide.ide_server import _monitor_connections
            _monitor_connections.discard(ws)
        except Exception:
            pass


# ── WebSocket 端点 ────────────────────────────────────

@router.websocket("/ws")
async def ws_chat_endpoint(ws: WebSocket):
    """Web 聊天 WebSocket — 主系统前端通信端点"""
    global _event_loop
    await ws.accept()
    _ws_clients.add(ws)

    # 捕获事件循环引用（首次）
    if _event_loop is None:
        _event_loop = asyncio.get_running_loop()

    # 同时加入 /monitor 广播集，接收 TASK_GRAPH_UPDATE 等系统事件
    try:
        from zulong.ide.ide_server import _monitor_connections
        _monitor_connections.add(ws)
        # 确保 ide_server 的 _main_event_loop 也被设置（Launcher 模式下 startup 不走）
        import zulong.ide.ide_server as _ide_srv
        if _ide_srv._main_event_loop is None:
            _ide_srv._main_event_loop = _event_loop
    except Exception:
        pass

    logger.info(f"[WebChatRouter] /ws 已连接 (total={len(_ws_clients)})")

    if not await _send_initial_dashboard_state(ws):
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        try:
            from zulong.ide.ide_server import _monitor_connections
            _monitor_connections.discard(ws)
        except Exception:
            pass
        logger.info("[WebChatRouter] /ws 初始状态发送前客户端已断开")
        return

    # 推送记忆图谱快照
    try:
        await _push_memory_graph_snapshot(ws)
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        try:
            from zulong.ide.ide_server import _monitor_connections
            _monitor_connections.discard(ws)
        except Exception:
            pass
        logger.info("[WebChatRouter] /ws 初始图谱发送前客户端已断开")
        return

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                # ── 统一协议: handshake ──
                if msg_type == "handshake":
                    _ws_protocols[id(ws)] = "unified"
                    client_type = data.get("payload", {}).get("client_type", "unknown")
                    _ws_client_types[id(ws)] = client_type
                    logger.info(f"[WebChatRouter] 统一协议握手: client_type={client_type}")
                    await _send_to_ws(ws, make_unified_message(
                        MessageType.TASK_ACK,
                        {"status": "ok", "client_type": client_type},
                    ))
                    continue

                # ── 统一协议: 未显式握手但直接发送 namespace 类型 ──
                if ":" in msg_type and _ws_protocols.get(id(ws)) != "unified":
                    _ws_protocols[id(ws)] = "unified"
                    _ws_client_types.setdefault(id(ws), "dashboard")

                # ── 统一协议: 新格式消息 → 旧格式处理 ──
                if _ws_protocols.get(id(ws)) == "unified" and ":" in msg_type:
                    data = _protocol_bridge.from_unified(data, target_format="web", direction="uplink")
                    msg_type = data.get("type", "")

                logger.info(
                    "[WebChatRouter] /ws 收到消息: type=%s, protocol=%s",
                    msg_type,
                    _ws_protocols.get(id(ws), "legacy"),
                )

                if msg_type == "ping":
                    await _send_to_ws(ws, {"type": "pong", "ts": time.time()})
                elif msg_type == "CHAT_MESSAGE":
                    asyncio.create_task(_handle_chat_message(ws, data))
                elif msg_type == "STOP_GENERATION":
                    asyncio.create_task(_handle_stop_generation(data))
                elif msg_type == "STOP_TASK":
                    asyncio.create_task(_handle_stop_generation(data))
                elif msg_type == "conversation:switch":
                    asyncio.create_task(_handle_conversation_switch(data.get("payload", data)))
                elif msg_type in ("CHAT_VISIBLE_MESSAGE", "chat:visible_message"):
                    asyncio.create_task(_handle_chat_visible_message(data))
                elif msg_type == MessageType.VOICE_BIND or msg_type == "voice_bind":
                    asyncio.create_task(_handle_voice_bind(ws, data.get("payload", data)))
                elif msg_type == MessageType.VOICE_LIST or msg_type == "voice:list":
                    asyncio.create_task(_handle_voice_list(ws, data.get("payload", data)))
                elif msg_type == MessageType.VOICE_DELETE or msg_type == "voice:delete":
                    asyncio.create_task(_handle_voice_delete(ws, data.get("payload", data)))
                elif msg_type in (
                    MessageType.IDE_OPEN_WORKSPACE,
                    MessageType.IDE_OPEN_FILE,
                    MessageType.IDE_OPEN_TERMINAL,
                    MessageType.IDE_SHOW_DIFF,
                    MessageType.IDE_GET_CONTEXT,
                    MessageType.IDE_APPROVAL_RESULT,
                    "ide_open_workspace",
                    "ide_open_file",
                    "ide_open_terminal",
                    "ide_show_diff",
                    "ide_get_context",
                    "ide_approval_result",
                    "ide:open_workspace",
                    "ide:open_file",
                    "ide:open_terminal",
                    "ide:show_diff",
                    "ide:get_context",
                    "ide:approval_result",
                    "ide:execute_tool",
                    "ide_execute_tool",
                ):
                    asyncio.create_task(_handle_ide_action(ws, msg_type, data))
                elif msg_type == "REQUEST_MEMORY_GRAPH":
                    asyncio.create_task(_push_memory_graph_snapshot(ws))
                elif msg_type == "EXPAND_NODE":
                    node_id = data.get("node_id")
                    if node_id:
                        asyncio.create_task(_handle_expand_node(node_id, ws))
                elif msg_type == "LIST_DIALOGUE_SESSIONS":
                    asyncio.create_task(_handle_list_dialogue_sessions(ws))
                elif msg_type == "GET_SESSION_MESSAGES":
                    session_id = data.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_get_session_messages(ws, session_id))
                elif msg_type == "DELETE_DIALOGUE_SESSION":
                    session_id = data.get("session_id")
                    if session_id:
                        asyncio.create_task(_handle_delete_dialogue_session(ws, session_id))
                elif msg_type == "audio_start":
                    asyncio.create_task(_handle_audio_start_web(ws, data))
                elif msg_type == "audio_chunk":
                    asyncio.create_task(_handle_audio_chunk_web(ws, data))
                elif msg_type == "audio_end":
                    asyncio.create_task(_handle_audio_end_web(ws, data))
                else:
                    logger.warning(f"[WebChatRouter] 未处理的 /ws 消息类型: {msg_type}")
            except json.JSONDecodeError:
                logger.warning("[WebChatRouter] /ws 收到无效 JSON")
            except Exception as e:
                logger.exception(f"[WebChatRouter] 消息处理异常: {e}")
                try:
                    await _send_to_ws(ws, {
                        "type": "CHAT_RESPONSE",
                        "text": _failure_response_text(f"WebSocket 消息处理异常：{e}"),
                    })
                except Exception:
                    pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WebChatRouter] 连接异常: {e}")
    finally:
        _ws_clients.discard(ws)
        _ws_protocols.pop(id(ws), None)
        _ws_client_types.pop(id(ws), None)
        try:
            from zulong.ide.ide_server import _monitor_connections
            _monitor_connections.discard(ws)
        except Exception:
            pass
        logger.info(f"[WebChatRouter] /ws 已断开 (total={len(_ws_clients)})")


# ── 音频处理（Web 前端） ────────────────────────────────

async def _handle_audio_start_web(ws: WebSocket, data: dict):
    """处理 Web 前端音频流开始 - 触发麦克风设备录音"""
    try:
        from zulong.l0.devices.microphone_device import MicrophoneDevice
        from zulong.launcher.app import LauncherApp
        
        mic_device = None
        
        # 从 LauncherApp 全局单例获取
        try:
            import zulong.launcher.app as app_module
            if hasattr(app_module, '_app_instance') and app_module._app_instance:
                launcher_app = app_module._app_instance
                mm = launcher_app.manager
                mic_module = mm._modules.get("microphone")
                if mic_module and hasattr(mic_module, '_mic') and mic_module._mic:
                    mic_device = mic_module._mic
                    logger.debug("[WebChatRouter] 获取麦克风设备成功")
        except Exception as e:
            logger.error(f"[WebChatRouter] 获取麦克风设备失败: {e}")
        
        if mic_device and hasattr(mic_device, 'start_manual_recording'):
            mic_device.start_manual_recording()
            logger.info(f"[WebChatRouter] 手动录音已启动")
            await _send_to_ws(ws, {
                "type": "audio_start_ack",
                "ts": time.time(),
                "payload": {"status": "ok"},
            })
        else:
            logger.warning("[WebChatRouter] 麦克风设备不可用")
            await _send_to_ws(ws, {
                "type": "audio_start_ack",
                "ts": time.time(),
                "payload": {"status": "error", "message": "microphone_not_available"},
            })
    except Exception as e:
        logger.error(f"[WebChatRouter] audio_start 失败: {e}")


async def _handle_audio_chunk_web(ws: WebSocket, data: dict):
    """处理 Web 前端音频块 - 已废弃，使用麦克风设备直接采集"""
    pass


async def _handle_audio_end_web(ws: WebSocket, data: dict):
    """处理 Web 前端音频流结束 - 停止麦克风录音并触发 ASR"""
    try:
        from zulong.l0.devices.microphone_device import MicrophoneDevice
        from zulong.launcher.app import LauncherApp
        
        mic_device = None
        
        # 从 LauncherApp 获取
        try:
            import zulong.launcher.app as app_module
            if hasattr(app_module, '_app_instance') and app_module._app_instance:
                launcher_app = app_module._app_instance
                mm = launcher_app.manager
                mic_module = mm._modules.get("microphone")
                if mic_module and hasattr(mic_module, '_mic') and mic_module._mic:
                    mic_device = mic_module._mic
                    logger.debug("[WebChatRouter] 获取麦克风设备成功")
        except Exception as e:
            logger.error(f"[WebChatRouter] 获取麦克风设备失败: {e}")
        
        if mic_device and hasattr(mic_device, 'stop_manual_recording'):
            try:
                audio_data = await mic_device.stop_manual_recording()
                if audio_data is None:
                    audio_data = b""
                logger.info(f"[WebChatRouter] 手动录音结束：{len(audio_data)} bytes")
                transcript = await _transcribe_pcm16_audio(audio_data)
                if transcript.get("status") == "ok" and transcript.get("text"):
                    bind_result = get_conversation_orchestrator().route_voice_record({
                        **transcript,
                        "source": "voice_page",
                        "session_id": data.get("session_id"),
                        "conversation_id": data.get("conversation_id") or data.get("session_id"),
                    })
                    transcript["voice_event_id"] = bind_result.get("voice_event_id")
                    transcript["linked_conversation_id"] = bind_result.get("conversation_id")
                    await _send_to_ws(ws, {
                        "type": "voice:record_created",
                        "payload": bind_result,
                        "voice_event_id": bind_result.get("voice_event_id"),
                        "conversation_id": bind_result.get("conversation_id"),
                    })
                    if bind_result.get("linked"):
                        await _send_to_ws(ws, {
                            "type": "voice:record_linked",
                            "payload": bind_result,
                            "voice_event_id": bind_result.get("voice_event_id"),
                            "conversation_id": bind_result.get("conversation_id"),
                        })
                await _send_to_ws(ws, {
                    "type": "audio_transcript",
                    "ts": time.time(),
                    "payload": transcript,
                })
            except Exception as e:
                logger.error(f"[WebChatRouter] stop_manual_recording 异常: {e}")
                await _send_to_ws(ws, {
                    "type": "audio_transcript",
                    "ts": time.time(),
                    "payload": {"status": "error", "text": "", "message": str(e)},
                })
        else:
            logger.warning("[WebChatRouter] 麦克风设备不可用或方法不存在")
            await _send_to_ws(ws, {
                "type": "audio_transcript",
                "ts": time.time(),
                "payload": {"status": "error", "text": "", "message": "microphone_not_available"},
            })
    except Exception as e:
        logger.error(f"[WebChatRouter] audio_end 失败: {e}", exc_info=True)


async def _transcribe_pcm16_audio(audio_data: bytes) -> dict:
    """转写麦克风手动录音的 PCM16/16k/mono 音频。"""
    if not audio_data:
        return {"status": "error", "text": "", "message": "no_audio_data"}
    try:
        import numpy as np
        from zulong.models.audio_model_container import get_audio_model_container

        container = get_audio_model_container()
        if not getattr(container, "_initialized", False):
            try:
                from zulong.config.config_manager import ConfigManager
                cm = ConfigManager()
                sensevoice_model_path = cm.get(
                    "audio.asr.model_path",
                    "./models/OpenASR/sensevoice-small-onnx",
                )
                asr_device = cm.get("audio.asr.device", "auto")
            except Exception:
                sensevoice_model_path = "./models/OpenASR/sensevoice-small-onnx"
                asr_device = "auto"
            try:
                from zulong.utils.device import resolve_device
                asr_device = resolve_device(asr_device, prefer_gpu=True)
            except Exception:
                pass
            container.initialize(
                enable_yamnet=False,
                enable_sensevoice=True,
                enable_whisper=True,
                sensevoice_device=asr_device,
                sensevoice_model_path=sensevoice_model_path,
            )

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype("float32") / 32768.0
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        result = await loop.run_in_executor(
            None,
            lambda: container.transcribe_speech(audio_np, 16000, "zh"),
        )
        elapsed = time.perf_counter() - start
        text = (getattr(result, "text", "") or "").strip()
        logger.info(
            "[WebChatRouter] ASR转写完成: len=%s, %.2fs, engine=%s",
            len(text),
            elapsed,
            getattr(result, "engine", ""),
        )
        return {
            "status": "ok",
            "text": text,
            "is_final": True,
            "engine": getattr(result, "engine", ""),
            "emotion": getattr(result, "emotion", ""),
            "language": getattr(result, "language", ""),
            "confidence": getattr(result, "confidence", 0.0),
            "elapsed": elapsed,
        }
    except Exception as e:
        logger.error(f"[WebChatRouter] ASR转写失败: {e}", exc_info=True)
        return {"status": "error", "text": "", "message": str(e)}


# ── TSD 23.11.3: BFS 自恢复辅助函数 ──────────────────────────

async def _retrieve_bfs_context(conversation_id: str) -> Optional[Dict]:
    """Web 端 BFS 会话上下文检索（薄 wrapper，避免循环导入）"""
    try:
        from zulong.ide.ide_server import _retrieve_session_context
        return await _retrieve_session_context(conversation_id)
    except Exception:
        return None


def _format_bfs_context_for_prompt(bfs_ctx: Dict) -> str:
    """将 BFS 上下文格式化为 LLM system prompt 注入文本"""
    parts = []

    session_label = bfs_ctx.get("session_label", "")
    if session_label:
        parts.append(f"当前会话主题: {session_label}")

    rounds = bfs_ctx.get("recent_rounds", [])
    if rounds:
        parts.append("近期对话:")
        for r in rounds[-3:]:
            user = (r.get("user_text", "") or r.get("label", ""))[:200]
            bot = (r.get("bot_text", ""))[:200]
            parts.append(f"- 用户: {user}")
            if bot:
                parts.append(f"  助手: {bot}")

    tasks = bfs_ctx.get("active_tasks", [])
    if tasks:
        parts.append("进行中的任务:")
        for t in tasks:
            status = t.get("status", "未知")
            label = t.get("label", "")
            desc = t.get("desc", "")[:200]
            parts.append(f"- [{status}] {label}: {desc}")

    return "\n".join(parts)
