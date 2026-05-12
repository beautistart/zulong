"""
Web 聊天路由器 — 通过 EventBus 路由消息到祖龙系统主链路

Web 前端是主系统界面，聊天消息不经过 IDE Server。

消息流:
  Full 模式: Web /ws → EventBus(USER_TEXT) → L1-B → L2 → EventBus(L2_OUTPUT/STREAM) → Web /ws
  IDE  模式: Web /ws → InferenceEngine(直接调用) → Web /ws
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# ── 状态 ──────────────────────────────────────────────

# WebSocket 连接管理
_ws_clients: Set[WebSocket] = set()

# 运行模式（由 LauncherApp 在启动后设置）
_launch_mode: Optional[str] = None  # "full" | "ide"

# asyncio 事件循环引用（用于从 EventBus 分发线程安全地调度协程）
_event_loop: Optional[asyncio.AbstractEventLoop] = None

# 活跃聊天取消事件（IDE 模式使用）
_chat_cancels: Dict[str, asyncio.Event] = {}

# EventBus 是否已订阅
_eventbus_subscribed = False


# ── 公共接口 ──────────────────────────────────────────

def set_launch_mode(mode: str) -> None:
    """设置运行模式并初始化对应的事件订阅"""
    global _launch_mode
    _launch_mode = mode
    logger.info(f"[WebChatRouter] 运行模式: {mode}")

    if mode == "full":
        _subscribe_eventbus()
    else:
        # IDE 模式也需要订阅 L2_THINKING_STEP 和 MEMORY_GRAPH_UPDATED，
        # 以便将任务图谱/记忆图谱更新推送到 /ws 前端
        _subscribe_eventbus_lite()


def _is_full_mode() -> bool:
    return _launch_mode == "full"


# ── EventBus 订阅（Full 模式） ────────────────────────

_eventbus_lite_subscribed = False


def _subscribe_eventbus_lite() -> None:
    """IDE 模式精简订阅 — 仅订阅图谱相关事件（L2_THINKING_STEP / MEMORY_GRAPH_UPDATED）

    IDE 模式下 LLM 响应通过 /ide WebSocket 直推前端，
    但任务图谱和记忆图谱更新仍需通过 EventBus → /ws 客户端推送。
    """
    global _eventbus_lite_subscribed
    if _eventbus_lite_subscribed:
        return
    try:
        from zulong.core.event_bus import event_bus
        from zulong.core.types import EventType

        event_bus.subscribe(EventType.L2_THINKING_STEP, _on_l2_thinking_step, "WebChatRouter")
        event_bus.subscribe(EventType.MEMORY_GRAPH_UPDATED, _on_memory_graph_updated, "WebChatRouter")
        _eventbus_lite_subscribed = True
        logger.info("[WebChatRouter] IDE 模式: 已订阅 L2_THINKING_STEP + MEMORY_GRAPH_UPDATED")
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
    """向所有 /ws 客户端广播消息"""
    msg_type = message.get("type", "?")
    if not _ws_clients:
        logger.warning(f"[WebChatRouter] _broadcast: type={msg_type}, 无客户端连接")
        return
    logger.info(f"[WebChatRouter] _broadcast: type={msg_type}, 发送给 {len(_ws_clients)} 个客户端")
    dead: Set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(message)
            logger.info(f"[WebChatRouter] _broadcast: 成功发送 {msg_type}")
        except Exception as e:
            logger.error(f"[WebChatRouter] _broadcast: 发送失败 {msg_type}: {e}")
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── EventBus 回调（从分发线程调用，非 asyncio） ───────

def _on_l2_output(event) -> None:
    text = event.payload.get("text", "")
    request_id = event.payload.get("request_id")
    logger.info(f"[WebChatRouter] _on_l2_output 被调用: text_len={len(text)}, request_id={request_id}, ws_clients={len(_ws_clients)}")
    if text:
        _schedule_broadcast({
            "type": "CHAT_RESPONSE",
            "text": text,
            "request_id": request_id,
        })


def _on_l2_output_stream(event) -> None:
    text = event.payload.get("text", "")
    chunk = event.payload.get("chunk", "")
    request_id = event.payload.get("request_id")
    if chunk or text:
        _schedule_broadcast({
            "type": "STREAMING_RESPONSE",
            "text": text,
            "chunk": chunk,
            "request_id": request_id,
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
        _schedule_broadcast({
            "type": "CHAT_RESPONSE",
            "text": text,
        })


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
    """处理 CHAT_MESSAGE: Full 走 EventBus，IDE 走 InferenceEngine"""
    text = data.get("text", "")
    request_id = data.get("request_id")
    session_id = data.get("session_id")
    referenced_nodes = data.get("referenced_nodes", [])

    if not text:
        return

    logger.info(f"[WebChatRouter] 消息: text={text[:80]}, mode={_launch_mode}")

    if _is_full_mode():
        await _chat_via_eventbus(text, request_id, session_id, referenced_nodes)
    else:
        await _chat_via_engine(ws, text, request_id, session_id, referenced_nodes)


async def _chat_via_eventbus(text, request_id, session_id, referenced_nodes):
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
        await loop.run_in_executor(None, event_bus.publish, event)
        logger.info("[WebChatRouter] USER_TEXT 已发布到 EventBus (via executor)")
    except Exception as e:
        logger.error(f"[WebChatRouter] EventBus 发布失败: {e}", exc_info=True)
        await _broadcast({
            "type": "CHAT_RESPONSE",
            "text": f"系统错误: {e}",
            "request_id": request_id,
        })


async def _chat_via_engine(ws, text, request_id, session_id, referenced_nodes):
    """IDE 模式: 直接调用 InferenceEngine 流式生成，同时广播状态与IDE端同步"""
    from zulong.ide.ide_server import _get_engine, broadcast_monitor_event

    # 广播会话开始状态到IDE端
    await broadcast_monitor_event("WEB_SESSION_START", {
        "session_id": session_id,
        "request_id": request_id,
        "task_preview": text[:100],
    })

    engine = _get_engine()
    if engine is None:
        await ws.send_json({
            "type": "CHAT_RESPONSE",
            "text": "推理引擎未初始化，请稍后重试。",
            "request_id": request_id,
        })
        return

    cancel_evt = asyncio.Event()
    if request_id:
        _chat_cancels[request_id] = cancel_evt

    # 同步IDE端的cancel_event到Web端的cancel_evt
    _ide_cancel_sync_task = asyncio.create_task(
        _sync_ide_cancel(cancel_evt, session_id))

    try:
        system_prompt = "你是祖龙智能助手。请用中文简洁友好地回答用户的问题。"

        # 注入记忆上下文
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if mg:
                context_results = await mg.retrieve_context(text, top_k=3)
                if context_results:
                    memory_lines = []
                    for r in context_results:
                        label = r.get("label", "")
                        content = r.get("content", "")
                        if label or content:
                            memory_lines.append(f"- {label}: {content[:200]}")
                    if memory_lines:
                        system_prompt += "\n\n以下是相关记忆上下文：\n" + "\n".join(memory_lines)
        except Exception:
            pass

        # 引用节点
        if referenced_nodes:
            ref_text = "\n".join(
                f"- [{n.get('label', '')}] {n.get('address', '')}"
                for n in referenced_nodes
            )
            text += f"\n\n[引用节点]\n{ref_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        await ws.send_json({
            "type": "THINKING_STEP",
            "request_id": request_id,
            "step_type": "model_call",
            "data": {"message": "正在调用推理引擎..."},
        })

        vllm_client = getattr(engine, "vllm_client", None)
        if vllm_client is None:
            await ws.send_json({
                "type": "CHAT_RESPONSE",
                "text": "LLM 客户端不可用，请检查配置。",
                "request_id": request_id,
            })
            return

        from zulong.models.container import LLM_MODEL_ID
        extra_kwargs = {}
        if hasattr(engine, "_get_llm_extra_kwargs"):
            extra_kwargs = engine._get_llm_extra_kwargs()

        stream = vllm_client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=messages,
            max_tokens=2048,
            temperature=0.3,
            top_p=0.85,
            stream=True,
            **extra_kwargs,
        )

        full_text = ""
        for chunk in stream:
            if cancel_evt.is_set():
                logger.info(f"[WebChatRouter] 生成已取消: {request_id}")
                break
            delta = chunk.choices[0].delta
            if delta and delta.content:
                full_text += delta.content
                try:
                    await ws.send_json({
                        "type": "STREAMING_RESPONSE",
                        "text": full_text,
                        "chunk": delta.content,
                        "request_id": request_id,
                    })
                except Exception:
                    break

        # 广播会话结束状态到IDE端
        end_status = "cancelled" if cancel_evt.is_set() else "completed"
        await broadcast_monitor_event("WEB_SESSION_END", {
            "session_id": session_id,
            "request_id": request_id,
            "status": end_status,
        })

        await ws.send_json({
            "type": "CHAT_RESPONSE",
            "text": full_text or "(无输出)",
            "request_id": request_id,
        })
        logger.info(f"[WebChatRouter] IDE 回复完成: len={len(full_text)}")

    except Exception as e:
        logger.error(f"[WebChatRouter] IDE 模式异常: {e}", exc_info=True)
        try:
            await ws.send_json({
                "type": "CHAT_RESPONSE",
                "text": f"处理出错: {e}",
                "request_id": request_id,
            })
        except Exception:
            pass
    finally:
        _chat_cancels.pop(request_id, None)
        _ide_cancel_sync_task.cancel()


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


async def _push_memory_graph_snapshot(ws: WebSocket) -> None:
    """推送记忆图谱快照到指定 WebSocket"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if not mg:
            return
        if hasattr(mg, "to_frontend_dict"):
            payload = mg.to_frontend_dict(depth=0)
        elif hasattr(mg, "get_snapshot_for_frontend"):
            payload = mg.get_snapshot_for_frontend()
        else:
            return
        if payload:
            await ws.send_json({
                "type": "MEMORY_GRAPH_UPDATE",
                "update_type": "full",
                "ts": time.time(),
                **payload,
            })
    except Exception as e:
        logger.debug(f"[WebChatRouter] 推送记忆图谱失败: {e}")


async def _handle_expand_node(node_id: str, ws: WebSocket) -> None:
    """处理展开节点请求"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if not mg:
            return
        if hasattr(mg, "get_node_children_for_frontend"):
            result = mg.get_node_children_for_frontend(node_id)
        elif hasattr(mg, "get_neighbors"):
            neighbors = mg.get_neighbors(node_id)
            result = {"node_id": node_id, "neighbors": neighbors}
        else:
            return
        await ws.send_json({
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
        from zulong.memory.memory_graph import get_memory_graph
        from zulong.memory.graph_adapters import DialogueAdapter

        mg = get_memory_graph()
        if not mg:
            await ws.send_json({
                "type": "SESSION_LIST",
                "ts": time.time(),
                "sessions": [],
            })
            return

        sessions = DialogueAdapter.list_sessions(mg)
        await ws.send_json({
            "type": "SESSION_LIST",
            "ts": time.time(),
            "sessions": sessions,
        })
        logger.info(f"[WebChatRouter] SESSION_LIST: 返回 {len(sessions)} 个会话")
    except Exception as e:
        logger.error(f"[WebChatRouter] LIST_DIALOGUE_SESSIONS 失败: {e}", exc_info=True)
        await ws.send_json({
            "type": "SESSION_LIST",
            "ts": time.time(),
            "sessions": [],
            "error": str(e),
        })


async def _handle_get_session_messages(ws: WebSocket, session_id: str) -> None:
    """获取指定会话的完整消息列表"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        from zulong.memory.graph_adapters import DialogueAdapter

        mg = get_memory_graph()
        if not mg:
            await ws.send_json({
                "type": "SESSION_MESSAGES",
                "ts": time.time(),
                "session_id": session_id,
                "messages": [],
            })
            return

        messages = DialogueAdapter.get_session_messages(mg, session_id)
        await ws.send_json({
            "type": "SESSION_MESSAGES",
            "ts": time.time(),
            "session_id": session_id,
            "messages": messages,
        })
        logger.info(
            f"[WebChatRouter] SESSION_MESSAGES: {session_id} → {len(messages)} 条消息")
    except Exception as e:
        logger.error(f"[WebChatRouter] GET_SESSION_MESSAGES 失败: {e}", exc_info=True)
        await ws.send_json({
            "type": "SESSION_MESSAGES",
            "ts": time.time(),
            "session_id": session_id,
            "messages": [],
            "error": str(e),
        })


async def _handle_delete_dialogue_session(ws: WebSocket, session_id: str) -> None:
    """删除对话会话及其所有子节点（BFS 级联删除），同步清理 TaskGraph/AgentSessionStore/IDESession"""
    try:
        from zulong.memory.memory_graph import get_memory_graph

        mg = get_memory_graph()
        if not mg or not mg.has_node(session_id):
            await ws.send_json({
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

        mg.save()

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

        await ws.send_json({
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
        await ws.send_json({
            "type": "SESSION_DELETED",
            "ts": time.time(),
            "session_id": session_id,
            "nodes_deleted": 0,
            "error": str(e),
        })


# ── REST API 端点 ────────────────────────────────────

@router.delete("/api/chat/sessions/{session_id}")
async def delete_session_rest(session_id: str, cascade: bool = True):
    """REST API 删除会话（级联清理 MemoryGraph/TaskGraph/AgentSessionStore）"""
    mg_nodes_deleted = 0
    tg_nodes_deleted = 0

    if cascade:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
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
                mg.save()
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

    # 收集初始状态
    engine_ready = False
    task_graph_snapshot = None
    memory_graph_stats = None
    code_anchor_stats = None
    try:
        from zulong.ide.ide_server import _get_engine, _sessions
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
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
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

    # 发送 WELCOME
    # 收集活跃 IDE 会话元数据
    _active_sessions_info = []
    try:
        from zulong.ide.ide_server import _sessions as _ide_sessions
        _active_sessions_info = [s.to_info_dict() for s in _ide_sessions.values()]
    except Exception:
        pass

    await ws.send_json({
        "type": "WELCOME",
        "ts": time.time(),
        "payload": {
            "engine_ready": engine_ready,
            "launch_mode": _launch_mode,
            "task_graph": task_graph_snapshot,
            "memory_graph_stats": memory_graph_stats,
            "code_anchor_stats": code_anchor_stats,
            "active_sessions": _active_sessions_info,
        },
    })

    # 推送记忆图谱快照
    await _push_memory_graph_snapshot(ws)

    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
                elif msg_type == "CHAT_MESSAGE":
                    asyncio.create_task(_handle_chat_message(ws, data))
                elif msg_type == "STOP_GENERATION":
                    asyncio.create_task(_handle_stop_generation(data))
                elif msg_type == "STOP_TASK":
                    asyncio.create_task(_handle_stop_generation(data))
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
            except json.JSONDecodeError:
                pass
            except Exception as e:
                logger.debug(f"[WebChatRouter] 消息处理异常: {e}")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"[WebChatRouter] 连接异常: {e}")
    finally:
        _ws_clients.discard(ws)
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
            await ws.send_json({
                "type": "audio_start_ack",
                "ts": time.time(),
                "payload": {"status": "ok"},
            })
        else:
            logger.warning("[WebChatRouter] 麦克风设备不可用")
            await ws.send_json({
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
            except Exception as e:
                logger.error(f"[WebChatRouter] stop_manual_recording 异常: {e}")
        else:
            logger.warning("[WebChatRouter] 麦克风设备不可用或方法不存在")
    except Exception as e:
        logger.error(f"[WebChatRouter] audio_end 失败: {e}", exc_info=True)
