"""
IDE 模式 FC 循环运行器

继承自 FCRunner (zulong.l2.fc_runner)，复用通用安全网逻辑。
使用 Python while 循环替代 LangGraph StateGraph，支持跨 HTTP 请求的暂停/恢复。

IDE 特有功能：
1. 流式推送 (display_text 逐句推送到前端)
2. 工具调用分流：内部工具直接执行，远程工具暂停返回
3. 状态完全序列化到 IDEFCState
4. 注意力窗口/RuleGuardian/CircuitBreaker per-runner 实例
5. 参数验证/别名映射
6. 429 限流重试 + 流式处理
"""

import asyncio
import concurrent.futures
import atexit
import json as _json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from zulong.l2.fc_runner import FCRunner
from zulong.ide.ide_session import IDEFCState, AgentSession
from zulong.ide.ide_tool_registry import IDEToolRegistry, IDE_REMOTE_TOOLS
from zulong.ide.ide_format_translator import IDEFormatTranslator
from zulong.ide.common.error_handler import ErrorHandler, ErrorCode
from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
from zulong.l2.circuit_breaker import CircuitBreakerState
from zulong.core.unified_protocol import MessageType

if TYPE_CHECKING:
    from zulong.l2.inference_engine import InferenceEngine

_SHARED_INTENT_FILTER = None
_SHARED_INTENT_FILTER_LOCK = threading.Lock()
_EMBEDDING_PREWARM_STARTED = False
_EMBEDDING_PREWARM_LOCK = threading.Lock()


def _get_shared_intent_filter():
    """进程级复用 L1-B IntentFilter，避免每个 IDE 会话重复初始化模型。"""
    global _SHARED_INTENT_FILTER
    if _SHARED_INTENT_FILTER is not None:
        return _SHARED_INTENT_FILTER

    with _SHARED_INTENT_FILTER_LOCK:
        if _SHARED_INTENT_FILTER is not None:
            return _SHARED_INTENT_FILTER
        try:
            from zulong.l1b.intent_filter import IntentFilter
            try:
                from zulong.config.config_manager import get_config
                intent_config = get_config("intent_classification", {})
            except Exception:
                intent_config = {}
            _SHARED_INTENT_FILTER = IntentFilter(config=intent_config)
            return _SHARED_INTENT_FILTER
        except Exception as e:
            logger.warning(f"[IDEFCRunner] IntentFilter 初始化失败: {e}")
            return None


def _ensure_embedding_prewarm_async() -> None:
    """后台预热 Embedding，避免首次 FC 会话同步等待模型加载。"""
    global _EMBEDDING_PREWARM_STARTED
    if _EMBEDDING_PREWARM_STARTED:
        return
    with _EMBEDDING_PREWARM_LOCK:
        if _EMBEDDING_PREWARM_STARTED:
            return
        _EMBEDDING_PREWARM_STARTED = True

    def _prewarm() -> None:
        try:
            from zulong.memory.embedding_manager import get_embedding_manager
            emb_mgr = get_embedding_manager()
            if getattr(emb_mgr, "_model", None) is None:
                logger.info("[IDEFCRunner] 后台预热 Embedding 模型...")
                emb_mgr.encode("预热")
                logger.info("[IDEFCRunner] Embedding 模型预热完成")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] Embedding 模型预热失败: {e}")

    threading.Thread(
        target=_prewarm,
        name="zulong-embedding-prewarm",
        daemon=True,
    ).start()

# Web 监控事件广播（延迟导入避免循环依赖：ide_fc_runner ↔ ide_server）
# 注意：ide_server 在模块顶层导入 ide_fc_runner（用于 FC Runner 实例化），
# 若此处也在顶层导入 ide_server，将形成循环引用。因此 _broadcast_sync 内部
# 使用函数级延迟导入，仅在首次调用时解析 ide_server 模块。
def _broadcast_sync(event_type: str, payload: dict) -> None:
    """在同步上下文中安排广播（fire-and-forget），线程安全"""
    try:
        from zulong.ide.ide_server import broadcast_monitor_event, _main_event_loop
        loop = _main_event_loop if _main_event_loop is not None else asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                broadcast_monitor_event(event_type, payload), loop
            )
    except (RuntimeError, AttributeError, ImportError) as e:
        logger.debug(f"[IDEFCRunner] 广播失败(预期异常): {e}")
    except Exception as e:
        logger.warning(f"[IDEFCRunner] 广播失败(未预期异常): {type(e).__name__}: {e}")

logger = logging.getLogger(__name__)


def _safe_truncate(text: str, max_len: int = 200) -> str:
    """安全截断字符串，避免多字节字符截断导致乱码"""
    if len(text) <= max_len:
        return text
    # 尝试在字符边界截断
    try:
        truncated = text[:max_len]
        # 验证截断后是否为有效UTF-8
        truncated.encode('utf-8').decode('utf-8')
        return truncated + "..."
    except UnicodeDecodeError:
        # 回退到安全截断
        return text.encode('utf-8')[:max_len].decode('utf-8', errors='ignore') + "..."


class ThreadPoolManager:
    """线程池生命周期管理器（单例模式）"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ide_fc_model"
        )
        self._futures: List[concurrent.futures.Future] = []
        self._shutdown = False

    @classmethod
    def get_instance(cls) -> "ThreadPoolManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    atexit.register(cls._instance.graceful_shutdown)
        return cls._instance

    def submit(self, fn, *args, **kwargs) -> concurrent.futures.Future:
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager has been shutdown")
        future = self._executor.submit(fn, *args, **kwargs)
        self._futures.append(future)
        self._cleanup_completed_futures()
        return future

    def _cleanup_completed_futures(self):
        self._futures = [f for f in self._futures if not f.done()]

    def graceful_shutdown(self, timeout: float = 10.0) -> None:
        if self._shutdown:
            return

        logger.info("[ThreadPoolManager] 开始优雅关闭线程池...")
        self._shutdown = True

        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
            logger.info("[ThreadPoolManager] 线程池已正常关闭")
        except Exception as e:
            logger.warning(f"[ThreadPoolManager] 等待关闭超时，强制终止: {e}")
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        pending_count = sum(1 for f in self._futures if not f.done())
        if pending_count > 0:
            logger.warning(f"[ThreadPoolManager] 仍有 {pending_count} 个任务未完成")
        else:
            logger.info("[ThreadPoolManager] 所有任务已完成")

    def __del__(self):
        if not self._shutdown:
            self.graceful_shutdown()


@dataclass
class IDEFCResult:
    """FC 循环执行结果"""
    phase: str
    text_response: Optional[str] = None
    pending_call_ids: Optional[List[str]] = None
    reason: Optional[str] = None


@dataclass
class ExecutionEvent:
    """Internal execution event normalized before fan-out.

    It is not a third transport.  send_callback and broadcast_monitor_event are
    the two outward adapters that consume this single in-runner state source.
    """

    phase: str
    message: str
    turn: int = 0
    event_type: str = "TASK_PROGRESS"
    payload: Dict[str, Any] = field(default_factory=dict)
    interaction: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def callback_payload(self, max_turns: int) -> Dict[str, Any]:
        data = {
            "protocol_version": "2.0",
            "phase": self.phase,
            "message": self.message,
            "current_turn": self.turn,
            "max_turns": max_turns,
            "timestamp": self.created_at,
        }
        data.update(self.payload)
        if self.interaction:
            data["interaction"] = self.interaction
        return data


_FILLER_PATTERNS = [
    "我正在思考", "让我继续", "我来继续", "让我想想", "接下来我",
    "我正在处理", "正在分析", "正在执行", "稍等", "我需要",
    "但我需要", "不过我需要", "还需要进一步", "需要更多信息",
]

_PROGRESS_VERB_PATTERNS = (
    "现在", "正在", "接下来", "开始", "创建", "生成", "编写", "实现",
    "完成", "添加", "修改", "构建", "开发", "设计", "部署", "运行",
    "执行", "处理", "分析", "准备", "优化", "更新", "调整", "修复",
    "将", "会", "要", "需要", "首先", "然后", "最后", "接着",
)

# UncompletedGuard 未完成节点拦截阈值（从0.5降至0.3，覆盖更多未完成场景）
_UNCOMPLETED_THRESHOLD = 0.3

# 工具结果缓冲区最大条目数（防止无限增长）
_TOOL_RESULTS_BUFFER_MAX = 100

# Backfill JSON 密度阈值：response 中 JSON 特征字符占比超过此值则跳过
_JSON_DENSITY_THRESHOLD = 0.12


def _is_filler_content(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 6:
        return True
    if len(stripped) < 80:
        if stripped.rstrip().endswith(("？", "?")):
            return False
        if any(v in stripped for v in _PROGRESS_VERB_PATTERNS):
            return False
        return True
    pattern_count = sum(1 for p in _FILLER_PATTERNS if p in stripped)
    if len(stripped) > 300:
        return pattern_count >= 3
    return pattern_count >= 2


def _has_content_match(response: str, node_label: str) -> bool:
    import re as _re
    if not response or not node_label:
        return False
    if node_label in response:
        return True
    cjk_runs = _re.findall("[一-鿿]{2,}", node_label)
    # 短标签（CJK < 4 字符）的 bigram 太少，只接受精确子串匹配
    total_cjk = sum(len(r) for r in cjk_runs)
    if total_cjk < 4:
        return False
    matched = set()
    for run in cjk_runs:
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if bg in response:
                matched.add(bg)
    # 要求匹配数量 ≥ max(3, 总 bigram 数的 40%)
    total_bigrams = max(1, sum(len(r) - 1 for r in cjk_runs))
    threshold = max(3, int(total_bigrams * 0.4))
    return len(matched) >= threshold


def _extract_node_content(response: str, node_label: str, max_len: int = 500) -> str:
    import re as _re
    if not response or not node_label:
        return response[:max_len] if response else ""
    idx = response.find(node_label)
    if idx >= 0:
        start = idx
        end = min(len(response), start + max_len)
        ns = response.find("\n\n", start + len(node_label))
        if 0 < ns - start <= max_len:
            end = ns
        return response[start:end].strip()
    for kw in _re.findall("[一-鿿]{2,}", node_label):
        idx = response.find(kw)
        if idx >= 0:
            return response[max(0, idx - 20):min(len(response), idx + max_len)].strip()
    return response[:max_len]


class _FunctionProxy:
    __slots__ = ("name", "arguments")
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ToolCallProxy:
    __slots__ = ("id", "type", "function")
    def __init__(self, data: Dict):
        self.id = data["id"]
        self.type = data.get("type", "function")
        fd = data["function"]
        self.function = _FunctionProxy(fd["name"], fd["arguments"])


class IDEFCRunner(FCRunner):
    """IDE 模式 FC 循环运行器

    继承 FCRunner，复用通用安全网逻辑 (_detect_duplicate_tool_loop, _detect_progress_stall)。
    添加 IDE 特有：流式推送、工具分流、暂停/恢复、参数验证。
    per-runner 实例隔离注意力窗口/RuleGuardian/CircuitBreaker，
    避免并发 Session 状态冲突。
    """

    def __init__(self, engine: "InferenceEngine", session: AgentSession,
                 tool_registry: IDEToolRegistry):
        super().__init__(engine)
        self.session = session
        self.tool_registry = tool_registry
        self.translator = IDEFormatTranslator()
        self._max_fc_turns = getattr(engine, "_max_fc_turns", 100)
        self._soft_limit = getattr(engine, "_soft_limit", 50)
        self._hard_limit = getattr(engine, "_hard_limit", 100)
        self._warning_interval = getattr(engine, "_warning_interval", 10)
        self._fc_loop_timeout = getattr(engine, "_fc_loop_timeout", 600)
        self._fc_request_interval = getattr(engine, "_fc_request_interval", 1.0)
        self._remote_tool_timeout = getattr(engine, "_remote_tool_timeout", 600)
        # P2: 进度报告 + 弹性预算
        self._progress_report_interval = getattr(engine, "_progress_report_interval", 5)
        self._auto_continue = getattr(engine, "_auto_continue", True)
        self._max_reports_before_force_stop = getattr(engine, "_max_reports_before_force_stop", 5)
        self._attn_window = None
        self._rule_guardian = None
        self._circuit_breaker = None
        self._drift_detector = None
        # DialogueAdapter 对话轮次记录
        self._dialogue_adapter = None
        self._current_round_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._model_executor = ThreadPoolManager.get_instance()
        # BFS 调度控制
        self._last_bfs_seeds_hash: str = ""
        self._last_bfs_turn: int = 0
        self._last_pressure_tier: str = "green"  # 压力分级跟踪（green/yellow/red）
        self._bfs_min_interval: int = 3  # 最小间隔轮次
        # WS层IDESession引用（由 ide_server._run_fc_loop 注入）
        self.ide_session = None
        # L1-B 细粒度分类器：只作为工具预判的辅助信号
        self._intent_filter = None
        self._execution_events: List[ExecutionEvent] = []
        self._interaction_seq = 0
        self._tool_interaction_pairs: Dict[str, str] = {}
        self._init_intent_filter()

    def _notify_session_linked(self, task_graph_id: str) -> None:
        """通知监控客户端: 会话已关联到任务图谱，同时更新 InteractionStore。

        TSD §23.11: 会话窗口绑定根会话节点，根节点下挂载任务图谱。
        task_graph_id 持久化到 InteractionStore，确保页面刷新后可恢复。
        """
        try:
            ide_sess = self.ide_session
            if ide_sess:
                ide_sess.task_graph_id = task_graph_id
            from zulong.ide.ide_server import _broadcast_sync
            _broadcast_sync("IDE_SESSION_LINKED", {
                "session_id": self.session.session_id,
                "conversation_id": getattr(ide_sess, "conversation_id", None) if ide_sess else None,
                "turn_id": getattr(ide_sess, "web_turn_id", None) if ide_sess else None,
                "workspace_path": getattr(ide_sess, "cwd", None) if ide_sess else None,
                "task_graph_id": task_graph_id,
                "project_id": ide_sess.project_id if ide_sess else None,
            })
            # TSD §23.11.6: 将 task_graph_id 写入 InteractionStore
            # 确保 conversation 表中持久化 task_graph_id，供页面刷新后恢复
            if ide_sess:
                _conv_id = getattr(ide_sess, "conversation_id", None)
                if _conv_id and task_graph_id:
                    try:
                        from zulong.launcher.interaction_store import get_interaction_store
                        _store = get_interaction_store()
                        _store.upsert_conversation(
                            _conv_id,
                            task_graph_id=task_graph_id,
                            workspace_path=getattr(ide_sess, "cwd", None),
                            project_id=getattr(ide_sess, "project_id", None),
                        )
                        logger.debug(
                            f"[IDEFCRunner] InteractionStore 已更新: "
                            f"conv={_conv_id}, graph={task_graph_id}"
                        )
                    except Exception as _ie:
                        logger.debug(
                            f"[IDEFCRunner] InteractionStore 更新跳过: {_ie}"
                        )
        except Exception as e:
            logger.debug(f"[IDEFCRunner] _notify_session_linked 跳过: {e}")

    def run_or_resume(self, new_messages: Optional[List[Dict]] = None,
                      tool_results: Optional[List[Dict]] = None) -> IDEFCResult:
        state = self.session.fc_state
        if tool_results and state and state.phase == "waiting_remote":
            logger.info(f"[IDEFCRunner] 恢复 FC, turn={state.fc_turn}, results={len(tool_results)}")
            self._restore_runner_state()
            self._inject_tool_results(state, tool_results)
            self._maybe_run_bfs(state.fc_turn, "tool_complete")
            state.phase = "running"
        elif new_messages:
            logger.info(f"[IDEFCRunner] 新 FC, messages={len(new_messages)}")
            state = self._init_state(new_messages)
            self.session.fc_state = state
        else:
            return IDEFCResult(phase="done", text_response="")
        return self._run_loop(state)

    async def run_loop_async(
        self,
        messages: List[Dict],
        send_callback,
        tool_result_queue: "asyncio.Queue",
        cancel_event: "asyncio.Event",
    ) -> IDEFCResult:
        """WebSocket 模式异步 FC 循环

        与 run_or_resume + _run_loop 的同步 HTTP 模式不同：
        - 远程工具不暂停返回，而是通过 send_callback 推送 tool_request 后
          等待 tool_result_queue 中的结果，自动注入后继续循环
        - 模型调用和内部工具执行在线程池中运行，避免阻塞事件循环
        - 通过 cancel_event 支持随时取消

        Args:
            messages: 初始消息列表 (system + user)
            send_callback: async callable(msg_type: str, payload: dict) 推送消息到插件
            tool_result_queue: asyncio.Queue 接收插件工具执行结果
            cancel_event: asyncio.Event 取消信号
        """
        # 保存 send_callback 供进度推送使用
        self._ide_send_callback = send_callback
        # 初始化状态（同步，在线程池中运行）
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(None, self._init_state, messages)
        self.session.fc_state = state
        from zulong.ide.ide_server import broadcast_monitor_event

        # 设置 FC 循环运行状态为 True（禁止节点审查提交）
        try:
            from zulong.core.state_manager import state_manager
            state_manager.set_fc_loop_running(True)
        except Exception:
            pass

        await self._emit_execution_event(
            send_callback,
            "started",
            f"开始处理: {(state.user_input_text or '')[:50]}",
            event_type="FC_START",
            payload={
                "max_turns": self._max_fc_turns,
                "task_graph_policy": getattr(state, "task_graph_policy", "none"),
                "user_input": (state.user_input_text or "")[:500],
                "next_step": "我会先判断是否需要记忆、项目上下文或 VS Code 后台桥。",
                "interaction": {
                    "kind": "state",
                    "status": "running",
                    "title": "已接收任务",
                    "detail": f"开始处理: {(state.user_input_text or '')[:120]}",
                    "next_step": "正在准备上下文和执行能力。",
                },
            },
            send_status=True,
        )

        # ── TSD v2.7: 发射 task:plan 和 tool:prediction 消息 ──
        try:
            _tool_pred = getattr(state, "tool_prediction", None)
            if _tool_pred:
                await send_callback(MessageType.TOOL_PREDICTION, {
                    "prediction": _tool_pred,
                    "suggested_tools": _tool_pred.get("suggested_tools", []),
                    "tool_bag": _tool_pred.get("tool_bag", []),
                    "confidence": _tool_pred.get("confidence", 0),
                    "reason": _tool_pred.get("reason", ""),
                    "timestamp": time.time(),
                })
            await send_callback(MessageType.TASK_PLAN, {
                "task": (state.user_input_text or "")[:500],
                "task_graph_policy": getattr(state, "task_graph_policy", "none"),
                "max_turns": self._max_fc_turns,
                "tool_prediction": _tool_pred,
                "interaction": {
                    "kind": "plan",
                    "status": "running",
                    "title": "已接收任务",
                    "detail": f"开始处理: {(state.user_input_text or '')[:120]}",
                    "next_step": "正在准备上下文和执行能力。",
                },
                "timestamp": time.time(),
            })
        except Exception as _e:
            logger.debug(f"[IDEFCRunner] 发射 task:plan/tool:prediction 失败: {_e}")

        while True:
            # 检查取消
            if cancel_event.is_set():
                return await loop.run_in_executor(
                    None, self._finalize, state, "cancelled")

            # 检查轮次限制
            tr = self._check(state)
            if tr:
                return await loop.run_in_executor(
                    None, self._finalize, state, tr)

            try:
                # 每轮推送含图谱进度的统一状态
                _progress_snapshot = self._get_progress_snapshot()
                await self._emit_execution_event(
                    send_callback,
                    "calling_model",
                    f"正在调用模型进行推理... (Turn {state.fc_turn}/{self._max_fc_turns})",
                    turn=state.fc_turn,
                    event_type="CALLING_MODEL",
                    payload={
                        "progress": _progress_snapshot,
                        "model": getattr(state, "vllm_model_id", ""),
                    },
                    send_progress=False,
                    send_status=True,
                )
                
                # 🔥 修复：持续发送reasoning心跳，保持"思考中"状态稳定
                reasoning_msg = f"正在调用模型进行推理... (Turn {state.fc_turn}/{self._max_fc_turns})"
                await send_callback("display_reasoning", {
                    "reasoning": reasoning_msg
                })
                self._emit_execution_event_sync(
                    "waiting_model",
                    reasoning_msg,
                    turn=state.fc_turn,
                    event_type="TASK_HEARTBEAT",
                    payload={"reasoning": reasoning_msg},
                )

                # FC 请求间隔
                if state.fc_turn > 1 and self._fc_request_interval > 0:
                    await asyncio.sleep(self._fc_request_interval)

                # 在线程池中调用模型（可取消：每 2 秒检查 cancel_event）
                model_future = loop.run_in_executor(
                    None, self._call_model, state)
                tc_data, resp_content = None, None
                while True:
                    if cancel_event.is_set():
                        model_future.cancel()
                        return await loop.run_in_executor(
                            None, self._finalize, state, "cancelled")
                    try:
                        tc_data, resp_content = await asyncio.wait_for(
                            asyncio.shield(model_future), timeout=2.0)
                        break
                    except asyncio.TimeoutError:
                        # 模型调用等待中，推送心跳进度防止前端超时
                        await self._emit_execution_event(
                            send_callback,
                            "waiting_model",
                            f"模型推理中... (Turn {state.fc_turn}/{self._max_fc_turns})",
                            turn=state.fc_turn,
                            event_type="TASK_HEARTBEAT",
                            payload={"progress": self._get_progress_snapshot()},
                            send_progress=False,
                            send_status=True,
                            monitor=True,
                        )
                        try:
                            await send_callback("display_reasoning", {
                                "reasoning": f"模型推理中... (Turn {state.fc_turn}/{self._max_fc_turns})"
                            })
                        except Exception:
                            pass
                        continue  # 继续等待，下次循环检查 cancel_event

                if tc_data is None and resp_content is None:
                    if state.api_timeout_count >= 2:
                        return await loop.run_in_executor(
                            None, self._finalize, state, "api_error")
                    await asyncio.sleep(2)  # API 错误后短暂退避
                    continue

                state.loop_error_count = 0

                # 刷新 Gatekeeper 空闲计时器，防止模型调用间隔被误判空闲
                try:
                    from zulong.l1b.scheduler_gatekeeper import gatekeeper
                    if gatekeeper:
                        gatekeeper.touch_idle_timer()
                except Exception:
                    pass

                if tc_data:
                    tool_names = [tc["function"]["name"] for tc in tc_data]
                    await self._emit_execution_event(
                        send_callback,
                        "executing",
                        f"执行工具: {', '.join(tool_names)}",
                        turn=state.fc_turn,
                        event_type="TOOL_CALL",
                        payload={"tools": tool_names, "count": len(tc_data)},
                    )
                    
                    # CB 模式工具调用计数：防止死循环
                    if state.cb_force_no_tools:
                        state.cb_tool_streak += 1
                    else:
                        state.cb_tool_streak = 0
                    # 有工具调用 → 分流处理
                    should_continue = await self._exec_tools_async(
                        state, tc_data, resp_content or "",
                        send_callback, tool_result_queue, cancel_event, loop)
                    if should_continue == "cancelled":
                        return await loop.run_in_executor(
                            None, self._finalize, state, "cancelled")
                    await self._emit_execution_event(
                        send_callback,
                        "turn_complete",
                        "本轮工具执行完成，继续判断下一步。",
                        turn=state.fc_turn,
                        event_type="TURN_COMPLETE",
                        payload={"has_tool_calls": True, "tool_names": tool_names},
                        send_progress=False,
                    )
                    # 继续循环
                    continue

                # 纯文本回复 → 先评估，再决定是否推送给前端
                # 避免安全网返回 "continue" 时重复显示 filler 内容
                final_text = state.last_response_content or resp_content
                verdict = await loop.run_in_executor(
                    None, self._eval_response, state, resp_content or "")

                # 🔥 修复：流式推理已在 _call_model 中实时推送，此处不再重复推送
                # 只发送完成标记即可
                if verdict == "done":
                    # 发送包含task_status的完成标记（无text字段，避免重复）
                    complete_msg = {
                        "text": "", 
                        "turn": state.fc_turn,
                        "streaming": False,
                        "complete": True,
                        "task_status": "completed"
                    }
                    logger.info(f"[IDEFCRunner] 发送display_text完成标记（流式已完成，仅发送完成信号）")
                    await send_callback("display_text", complete_msg)
                    await self._emit_execution_event(
                        send_callback,
                        "text_streamed",
                        "模型文本回复已流式输出完成。",
                        turn=state.fc_turn,
                        event_type="DISPLAY_TEXT",
                        payload={"complete": True, "text_length": len(final_text or "")},
                        send_progress=False,
                        monitor=False,
                    )
                    
                elif verdict == "continue":
                    # continue时不推送，避免中间状态重复显示
                    logger.info(f"[FC] Turn {state.fc_turn} verdict=continue, skip display_text push (len={len(final_text or '')})")
                    # 保存内容供下一轮使用
                    if final_text:
                        state.last_response_content = final_text
                
                if verdict == "done" and final_text:
                    await self._emit_execution_event(
                        send_callback,
                        "model_response",
                        "模型已生成最终文本回复。",
                        turn=state.fc_turn,
                        event_type="MODEL_RESPONSE",
                        payload={
                            "text": (final_text or "")[:5000],
                            "text_preview": (final_text or "")[:200],
                            "text_length": len(final_text or ""),
                        },
                        send_progress=False,
                        monitor=True,
                    )
                    await self._emit_execution_event(
                        send_callback,
                        "display_text",
                        "向 Web 端推送最终文本。",
                        turn=state.fc_turn,
                        event_type="DISPLAY_TEXT",
                        payload={"text": (final_text or "")[:10000]},
                        send_progress=False,
                        monitor=True,
                    )

                if verdict == "done":
                    state.phase = "done"
                    
                    await self._emit_execution_event(
                        send_callback,
                        "completed",
                        f"任务完成 (共{state.fc_turn}轮推理)",
                        turn=state.fc_turn,
                        event_type="TASK_COMPLETE",
                        payload={
                            "summary": {
                                "completed": ["模型已生成最终回复"],
                                "verified": [],
                                "remaining": [],
                                "risk": "",
                                "next_step": "等待用户继续补充或提出新调整。",
                            },
                            "interaction": {
                                "kind": "summary",
                                "status": "succeeded",
                                "title": "任务完成",
                                "detail": f"本轮共进行了 {state.fc_turn} 轮推理。",
                                "progress": 100,
                                "next_step": "你可以继续提问或插入新的调整。",
                            },
                        },
                    )
                    
                    # ── TSD v2.7: 发射 task:summary 和 graph:memory:diff ──
                    try:
                        from zulong.memory.memory_graph import get_memory_graph
                        _mg = get_memory_graph()
                        _mem_changes = {"created": 0, "strengthened": 0, "pruned": 0}
                        if _mg:
                            # 从 ShardedMemoryGraph 获取实际计数
                            if hasattr(_mg, '_last_activated_edges'):
                                _mem_changes["strengthened"] = len(_mg._last_activated_edges)
                                _created = 0
                                for _sid in _mg.list_all_shards():
                                    _shard = _mg.get_shard(_sid)
                                    if _shard:
                                        try:
                                            _created += len(_shard.topology)
                                        except Exception:
                                            pass
                                _mem_changes["created"] = _created
                            elif hasattr(_mg, 'stats'):
                                _mem_changes["created"] = _mg.stats.get("total_nodes", 0)
                        await send_callback(MessageType.TASK_SUMMARY, {
                            "completed_items": ["模型已生成最终回复"],
                            "verified_items": [],
                            "pending_items": [],
                            "risks_summary": "",
                            "memory_changes": _mem_changes,
                            "next_step": "等待用户继续补充或提出新调整。",
                            "interaction": {
                                "kind": "summary",
                                "status": "succeeded",
                                "title": "任务完成",
                                "detail": f"本轮共进行了 {state.fc_turn} 轮推理。",
                                "progress": 100,
                                "memory_changes": _mem_changes,
                            },
                            "timestamp": time.time(),
                        })
                        await send_callback(MessageType.GRAPH_MEMORY_DIFF, {
                            "memory_changes": _mem_changes,
                            "timestamp": time.time(),
                        })
                    except Exception as _e:
                        logger.debug(f"[IDEFCRunner] 发射 task:summary/graph:memory:diff 失败: {_e}")
                    
                    # 清除 FC 循环运行状态（允许节点审查提交）
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    
                    # 立即通知前端任务完成（在后处理之前，防止 WS 断开导致丢失）
                    _done_text = state.last_response_content or ""
                    
                    # 🎯 优化3: 完成消息也使用清洗后的文本
                    if _done_text:
                        from zulong.utils.text_cleaner import clean_text_for_tts
                        _done_text = clean_text_for_tts(_done_text)
                    
                    # 🔥 修复：不在FC循环内发送task_complete，由ide_server统一发送
                    # 避免重复发送导致前端状态混乱
                    # await send_callback("task_complete", {"result": _done_text})
                    
                    await self._emit_execution_event(
                        send_callback,
                        "completed",
                        "FC 循环完成。",
                        turn=state.fc_turn,
                        event_type="FC_DONE",
                        payload={
                            "total_turns": state.fc_turn,
                            "reason": "done",
                            "summary": {
                                "completed": ["FC 循环已完成"],
                                "verified": [],
                                "remaining": [],
                                "risk": "",
                                "next_step": "等待下一轮用户输入。",
                            },
                            "interaction": {
                                "kind": "summary",
                                "status": "succeeded",
                                "title": "执行结束",
                                "detail": "祖龙已完成本轮执行。",
                                "progress": 100,
                            },
                        },
                        send_progress=False,
                        monitor=True,
                    )
                    # 🔥 修复：提取 submit_final_answer 内容，确保后处理和 ide_server 能读到完整答案
                    _final_answer = self._extract_final_answer(state)
                    if _final_answer:
                        state.final_answer = _final_answer
                        logger.info(
                            f"[IDEFCRunner] 提取 submit_final_answer: "
                            f"len={len(_final_answer)}")
                    else:
                        state.final_answer = None

                    # 将 FC 统计写入 TaskGraph 元数据，供 TaskArchive 使用
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph()
                        if _tg and hasattr(_tg, "metadata"):
                            _tg.metadata["total_turns"] = state.fc_turn
                            _tg.metadata["duration"] = time.time() - getattr(_tg, "created_at", time.time())
                    except Exception:
                        pass

                    # 后处理（耗时操作，客户端已收到完成通知）
                    await loop.run_in_executor(
                        None, self._auto_complete_task, state)
                    self._finalize_dialogue_round(state, status="completed")
                    await loop.run_in_executor(
                        None, self._auto_save_session_memory, state)
                    self._save_runner_state()
                    self.session.fc_state = state

                    # 🔥 修复：FC 完成后重新归档，补全 final_answer/duration/turns
                    try:
                        from zulong.tools.task_tools import get_active_task_graph, _auto_archive_completed
                        _tg = get_active_task_graph()
                        if _tg:
                            _auto_archive_completed(_tg)
                    except Exception:
                        pass

                    return IDEFCResult(
                        phase="done",
                        text_response=state.final_answer or state.last_response_content)
                elif verdict == "cb_force":
                    state.cb_force_no_tools = True
                # "continue" → 继续循环

            except asyncio.CancelledError:
                # 清除 FC 循环运行状态
                try:
                    from zulong.core.state_manager import state_manager
                    state_manager.set_fc_loop_running(False)
                except Exception:
                    pass
                return await loop.run_in_executor(
                    None, self._finalize, state, "cancelled")
            except Exception as loop_err:
                logger.error(
                    f"[IDEFCRunner] async 循环异常 turn={state.fc_turn}: "
                    f"{loop_err}", exc_info=True)
                state.loop_error_count += 1
                if state.loop_error_count >= 3:
                    # 清除 FC 循环运行状态
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    return await loop.run_in_executor(
                        None, self._finalize, state, "loop_error")
                continue

    async def _exec_tools_async(
        self,
        state: IDEFCState,
        tool_calls_data: List[Dict],
        response_content: str,
        send_callback,
        tool_result_queue: "asyncio.Queue",
        cancel_event: "asyncio.Event",
        loop,
    ) -> Optional[str]:
        """异步版工具执行 + 分流

        内部工具在线程池中执行；远程工具通过 WebSocket 推送后等待结果。
        返回 None 表示正常继续循环，"cancelled" 表示被取消。
        """
        from zulong.ide.ide_server import broadcast_monitor_event
        fc = state.fc_turn
        msgs = state.messages
        internal, remote = [], []
        for td in tool_calls_data:
            cat = self.tool_registry.classify(td["function"]["name"])
            (remote if cat == "remote" else internal).append(td)

        grp = self._attn_window.new_tool_group() if self._attn_window else None

        # ── 内部工具（线程池执行） ──
        if internal:
            a_msg = {
                "role": "assistant", "content": response_content or "",
                "tool_calls": internal}
            msgs.append(a_msg)
            if self._attn_window:
                self._attn_window.register_message(a_msg, turn=fc, group_id=grp)

            for td in internal:
                if cancel_event.is_set():
                    return await loop.run_in_executor(None, self._finalize, state, "cancelled")
                tn = td["function"]["name"]
                call_id = td.get("id") or self._next_interaction_id("internal_tool")
                self._tool_interaction_pairs[call_id] = call_id
                await self._emit_execution_event(
                    send_callback,
                    "tool_requested",
                    f"准备执行内部工具: {tn}",
                    turn=fc,
                    event_type="TOOL_CALL",
                    payload={
                        "tool_name": tn,
                        "tool_scope": "internal",
                        "call_id": call_id,
                        "pair_id": call_id,
                        "interaction": {
                            "pair_id": call_id,
                            "kind": "action",
                            "status": "running",
                            "title": f"准备执行内部工具: {tn}",
                            "detail": "祖龙将在后端内部执行该工具，并把结果回填给模型。",
                            "tool_name": tn,
                        },
                    },
                    send_progress=False,
                    send_status=True,
                )
                await loop.run_in_executor(
                    None, self._exec_internal, state, td, fc, grp)
                await self._emit_execution_event(
                    send_callback,
                    "tool_finished",
                    f"内部工具执行完成: {tn}",
                    turn=fc,
                    event_type="IDE_TOOL_EXEC",
                    payload={
                        "tool_name": tn,
                        "tool_scope": "internal",
                        "call_id": call_id,
                        "pair_id": call_id,
                        "interaction": {
                            "pair_id": call_id,
                            "kind": "observation",
                            "status": "succeeded",
                            "title": f"内部工具执行完成: {tn}",
                            "detail": "工具结果已写回本轮上下文。",
                            "tool_name": tn,
                        },
                    },
                    send_progress=False,
                    send_status=True,
                )

            # CircuitBreaker 评估
            try:
                if self._circuit_breaker:
                    _aw_ratio = self._attn_window.usage_ratio if self._attn_window else -1.0
                    cb_s, cb_r = self._circuit_breaker.evaluate(fc, msgs, attn_usage_ratio=_aw_ratio)
                    if cb_s == CircuitBreakerState.RED:
                        logger.warning(f"[IDEFCRunner][CB] RED: {cb_r}")
                        state.cb_force_no_tools = True
                        cm = {
                            "role": "user",
                            "content": (
                                f"[Circuit Breaker 强制收敛] {cb_r}\n"
                                f"你必须立刻基于已有信息生成最终回复，"
                                f"不允许再调用任何工具。"),
                        }
                        msgs.append(cm)
                        if self._attn_window:
                            self._attn_window.register_message(cm, turn=fc)
                            try:
                                self._attn_window.on_navigate_attention(direction="single_chain")
                            except Exception:
                                pass
                        remote = []  # 取消远程工具
                    elif cb_s == CircuitBreakerState.YELLOW:
                        logger.warning(f"[IDEFCRunner][CB] YELLOW: {cb_r}")
                        # 如果是 task_add_node 模式重复，检查计划深度
                        cb_msg = f"[Circuit Breaker 警告] {cb_r}\n请尽快总结当前信息并回复用户。"
                        if "task_add_node" in cb_r:
                            try:
                                from zulong.tools.task_tools import get_active_task_graph as _gtg
                                _t = _gtg()
                                if _t:
                                    max_depth = max((_t.get_node_depth(n.id) for n in _t._nodes.values()), default=0)
                                    if max_depth <= 1:
                                        cb_msg = (
                                            "[结构校验] 当前任务计划过浅：所有节点都在第1层。\n"
                                            "【强制要求】请立即为每个阶段添加 2-3 个具体子步骤节点，使用 "
                                            "task_add_node(parent_id='阶段节点ID', label='子步骤名')。\n"
                                            "示例：task_add_node(parent_id='o1', label='分析项目README文档')\n"
                                            "完成子步骤添加后再开始执行。"
                                        )
                            except Exception:
                                pass
                        ch = {"role": "user", "content": cb_msg}
                        msgs.append(ch)
                        if self._attn_window:
                            self._attn_window.register_message(ch, turn=fc)
            except Exception as cb_err:
                logger.warning(
                    f"[IDEFCRunner] CircuitBreaker evaluate 异常: {cb_err}")

            # 上下文压力感知（在 CB 评估之后）
            self._apply_pressure_guidance(state, fc)

            # ── TSD v2.7: 发射 graph:memory:diff（内部工具执行后）──
            await self._emit_memory_diff_async(send_callback, state.fc_turn)

            # ── TSD v2.7: 发射 attention:update ──
            if self._attn_window:
                try:
                    _attn_state = {
                        "mode": self._attn_window.mode.value if self._attn_window.mode else "global",
                        "turn": fc,
                        "focus_node_id": self._attn_window._current_node_id,
                        "budget_usage": round(self._attn_window.usage_ratio * 100, 1) if hasattr(self._attn_window, "usage_ratio") else 0,
                    }
                    await send_callback(MessageType.ATTENTION_UPDATE, _attn_state)
                except Exception as _e:
                    logger.debug(f"[IDEFCRunner] 发射 attention:update 失败: {_e}")

        # ── 远程工具（WebSocket 推送 + 等待） ──
        if remote:
            valid_remote, rejected = self._validate_and_clean_remote_calls(
                remote)
            all_calls = valid_remote + [r[0] for r in rejected]
            ra = {
                "role": "assistant",
                "content": "" if internal else (response_content or ""),
                "tool_calls": all_calls,
            }
            msgs.append(ra)
            if self._attn_window:
                self._attn_window.register_message(
                    ra, turn=fc, group_id=grp)

            # 注入被拒绝调用的错误结果
            for rej_tc, err_msg in rejected:
                err_result = {
                    "role": "tool",
                    "tool_call_id": rej_tc["id"],
                    "content": f"[参数验证失败] {err_msg}",
                }
                msgs.append(err_result)
                if self._attn_window:
                    self._attn_window.register_message(
                        err_result, turn=fc,
                        tool_name=rej_tc["function"]["name"])

            if valid_remote:
                # 设置 pending 状态（累积而非覆盖，支持跨轮次）
                state.pending_remote_calls = valid_remote
                new_call_ids = [tc["id"] for tc in valid_remote]
                
                # 累积到pending_call_turns（而非覆盖pending_call_ids）
                for call_id in new_call_ids:
                    state.pending_call_turns[call_id] = fc
                state.pending_call_ids = list(state.pending_call_turns.keys())

                # ── TSD v2.7: 高风险工具预检与 approval:required 发射 ──
                for tc in valid_remote:
                    tool_name = tc["function"]["name"]
                    risk_level = self._get_tool_risk_level(tool_name, tc.get("function", {}).get("arguments", "{}"))
                    if risk_level in ("HIGH", "CRITICAL"):
                        try:
                            from zulong.config.approval_config import get_approval_whitelist
                            whitelist = get_approval_whitelist()
                            if not whitelist.should_auto_approve(tool_name):
                                await send_callback(MessageType.APPROVAL_REQUIRED, {
                                    "call_id": tc.get("id", ""),
                                    "tool_name": tool_name,
                                    "approval_mode": "popup" if risk_level == "CRITICAL" else "manual",
                                    "risk_level": risk_level,
                                    "reason": f"高风险工具 ({risk_level}): {tool_name}",
                                    "tool_args": tc.get("function", {}).get("arguments", "")[:500],
                                    "interaction": {
                                        "pair_id": tc.get("id", ""),
                                        "kind": "approval",
                                        "status": "awaiting_approval",
                                        "title": f"审批请求: {tool_name}",
                                        "detail": f"工具 {tool_name} 风险等级 {risk_level}，需要审批。",
                                        "tool_name": tool_name,
                                        "risk_level": risk_level,
                                        "approval_mode": "popup" if risk_level == "CRITICAL" else "manual",
                                    },
                                    "timestamp": time.time(),
                                })
                        except Exception as _e:
                            logger.debug(f"[IDEFCRunner] 发射高风险 approval:required 失败: {_e}")

                # 通过 WebSocket 推送 tool_request
                tool_names = [
                    tc["function"]["name"] for tc in valid_remote]
                group_id = f"tool_group:{self.session.session_id}:{fc}:{int(time.time() * 1000)}"
                for tc in valid_remote:
                    self._tool_interaction_pairs[tc.get("id", "")] = tc.get("id", "")
                logger.info(
                    f"[IDEFCRunner] async 远程工具推送: {tool_names}, call_ids={new_call_ids}")

                await self._emit_execution_event(
                    send_callback,
                    "tool_requested",
                    f"等待 VS Code 后台桥执行: {', '.join(tool_names)}",
                    turn=fc,
                    event_type="IDE_TOOL_REQUEST",
                    payload={
                        "tools": [
                            {"name": tc["function"]["name"],
                             "arguments_preview": tc["function"].get("arguments", "")[:300],
                             "call_id": tc.get("id", ""),
                             "pair_id": tc.get("id", "")}
                            for tc in valid_remote
                        ],
                        "call_ids": state.pending_call_ids,
                        "tool_names": tool_names,
                        "group_id": group_id,
                        "parallel": len(valid_remote) > 1,
                        "completed_count": 0,
                        "total_count": len(valid_remote),
                        "interaction": {
                            "pair_id": group_id,
                            "kind": "action",
                            "status": "running",
                            "title": "准备交给 VS Code 后台桥执行",
                            "detail": f"将执行 {len(valid_remote)} 个工具: {', '.join(tool_names)}",
                            "tool_name": ",".join(tool_names),
                            "progress": 0,
                            "next_step": "等待 VS Code 返回工具结果。",
                        },
                    },
                    send_progress=True,
                    send_status=True,
                )

                await send_callback("tool_request", {
                    "tool_calls": valid_remote,
                    "call_ids": state.pending_call_ids,
                    "tool_names": tool_names,
                    "group_id": group_id,
                })

                # 等待所有远程工具结果
                results = []
                for i in range(len(valid_remote)):
                    if cancel_event.is_set():
                        return "cancelled"
                    try:
                        result = await asyncio.wait_for(
                            tool_result_queue.get(), timeout=self._remote_tool_timeout)
                        results.append(result)
                    except asyncio.TimeoutError:
                        tc = valid_remote[i]
                        call_id = tc.get("id", "")
                        await self._emit_execution_event(
                            send_callback,
                            "blocked",
                            f"工具执行超时: {tc['function']['name']}",
                            turn=fc,
                            event_type="TASK_BLOCKED",
                            payload={
                                "tool_name": tc["function"]["name"],
                                "call_id": call_id,
                                "pair_id": call_id,
                                "timeout_seconds": self._remote_tool_timeout,
                                "interaction": {
                                    "pair_id": call_id,
                                    "kind": "state",
                                    "status": "blocked",
                                    "title": f"工具执行超时: {tc['function']['name']}",
                                    "detail": f"等待 VS Code 后台桥超过 {self._remote_tool_timeout}s。",
                                    "tool_name": tc["function"]["name"],
                                    "next_step": "可以重试、取消，或根据当前结果调整任务。",
                                },
                            },
                            send_progress=True,
                            send_status=True,
                        )
                        # ── TSD v2.7: 发射 approval:required ──
                        try:
                            await send_callback(MessageType.APPROVAL_REQUIRED, {
                                "call_id": call_id,
                                "tool_name": tc["function"]["name"],
                                "approval_mode": "manual",
                                "reason": f"工具执行超时: {tc['function']['name']}",
                                "timeout_seconds": self._remote_tool_timeout,
                                "interaction": {
                                    "pair_id": call_id,
                                    "kind": "state",
                                    "status": "blocked",
                                    "title": f"工具执行超时: {tc['function']['name']}",
                                    "detail": f"等待 VS Code 后台桥超过 {self._remote_tool_timeout}s。",
                                    "tool_name": tc["function"]["name"],
                                    "next_step": "可以重试、取消，或根据当前结果调整任务。",
                                },
                                "timestamp": time.time(),
                            })
                        except Exception as _e:
                            logger.debug(f"[IDEFCRunner] 发射 approval:required 失败: {_e}")
                        results.append({
                            "call_id": call_id,
                            "tool_name": tc["function"]["name"],
                            "result": f"[工具执行超时 ({self._remote_tool_timeout}s)]",
                            "is_error": True,
                        })

                # 转换为 _inject_tool_results 期望的格式（保留is_error标记）
                formatted_results = []
                call_id_to_remote = {tc["id"]: tc for tc in valid_remote}
                for r in results:
                    call_id = r.get("call_id", "")
                    is_error = r.get("is_error", False)
                    content = r.get("result", "")
                    if is_error and content and not content.startswith("[错误]"):
                        content = f"[错误] {content}"
                    formatted_results.append({
                        "tool_call_id": call_id,
                        "content": content,
                    })

                self._inject_tool_results(state, formatted_results)
                self._maybe_run_bfs(fc, "tool_complete")

                failed_count = sum(1 for r in results if r.get("is_error", False))
                succeeded_count = len(results) - failed_count
                await self._emit_execution_event(
                    send_callback,
                    "tool_finished",
                    f"VS Code 后台桥返回 {len(results)} 个工具结果。",
                    turn=fc,
                    event_type="IDE_TOOL_RESULT",
                    payload={
                        "results": [
                            {"tool_name": r.get("tool_name", ""),
                             "call_id": r.get("call_id", ""),
                             "pair_id": r.get("call_id", ""),
                             "result_preview": (r.get("result", "") or "")[:500],
                             "is_error": r.get("is_error", False)}
                            for r in results
                        ],
                        "group_id": group_id,
                        "completed_count": len(results),
                        "total_count": len(valid_remote),
                        "failed_count": failed_count,
                        "succeeded_count": succeeded_count,
                        "interaction": {
                            "pair_id": group_id,
                            "kind": "observation",
                            "status": "failed" if failed_count else "succeeded",
                            "title": "VS Code 后台桥返回工具结果",
                            "detail": f"{succeeded_count} 个成功，{failed_count} 个失败。",
                            "progress": 100,
                            "next_step": "祖龙会根据这些结果继续判断下一步。",
                        },
                    },
                    send_progress=False,
                    send_status=True,
                )
                # ── TSD v2.7: 发射 graph:memory:diff（远程工具执行后）──
                await self._emit_memory_diff_async(send_callback, state.fc_turn)
                return None  # 继续循环

        self._maybe_run_bfs(fc, "tool_complete")
        # ── TSD v2.7: 发射 graph:memory:diff（内部工具执行后）──
        await self._emit_memory_diff_async(send_callback, state.fc_turn)
        return None  # 继续循环

    def _restore_runner_state(self) -> None:
        if self.session.attention_window_data:
            try:
                from zulong.l2.attention_window import AttentionWindowManager
                from zulong.tools.task_tools import get_active_task_graph
                from zulong.memory.memory_graph import get_memory_graph
                _restore_tg = get_active_task_graph()
                _restore_mg = get_memory_graph()
                self._attn_window = AttentionWindowManager.from_serialized(
                    self.session.attention_window_data,
                    task_graph=_restore_tg,
                    memory_graph=_restore_mg,
                )
            except Exception as e:
                logger.warning(f"[IDEFCRunner] 注意力窗口恢复失败: {e}")
        self._create_rule_guardian()
        # 恢复 RuleGuardian 状态
        if self._rule_guardian and self.session.rule_guardian_data:
            try:
                self._rule_guardian.deserialize(self.session.rule_guardian_data)
            except Exception as e:
                logger.warning(f"[IDEFCRunner] RuleGuardian 状态恢复失败: {e}")
        self._create_circuit_breaker()
        # 恢复 CircuitBreaker 状态
        if self._circuit_breaker and self.session.circuit_breaker_data:
            try:
                self._circuit_breaker.deserialize(self.session.circuit_breaker_data)
            except Exception as e:
                logger.warning(f"[IDEFCRunner] CircuitBreaker 状态恢复失败: {e}")
        self._create_drift_detector()
        # 恢复对话轮次跟踪状态
        self._current_round_id = self.session.dialogue_round_id
        self._current_session_id = self.session.dialogue_session_id
        self._init_dialogue_adapter()

    def _create_rule_guardian(self) -> None:
        try:
            if hasattr(self.engine, "_rule_guardian") and self.engine._rule_guardian:
                self._rule_guardian = type(self.engine._rule_guardian)()
            else:
                from zulong.l2.rule_guardian import RuleGuardian
                self._rule_guardian = RuleGuardian()
        except Exception as e:
            logger.warning(f"[IDEFCRunner] RuleGuardian 创建失败: {e}")

    def _create_circuit_breaker(self) -> None:
        try:
            if hasattr(self.engine, "_circuit_breaker") and self.engine._circuit_breaker:
                cb = self.engine._circuit_breaker
                cb_cfg = dict(getattr(cb, "_config", {}))
                # 确保 CB 的 context_window_size 与引擎一致
                cb_cfg["context_window_size"] = getattr(
                    self.engine, "_context_window_size", 32768)
                self._circuit_breaker = type(cb)(cb_cfg)
            else:
                from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
                self._circuit_breaker = ToolCallCircuitBreaker({
                    "context_window_size": getattr(
                        self.engine, "_context_window_size", 32768),
                })
        except Exception as e:
            logger.warning(f"[IDEFCRunner] CircuitBreaker 创建失败: {e}")

    def _create_drift_detector(self) -> None:
        """创建语义漂移检测器（轻量，仅在需要时才计算 Embedding）"""
        try:
            from zulong.memory.semantic_drift_detector import get_semantic_drift_detector
            self._drift_detector = get_semantic_drift_detector()
            logger.info("[IDEFCRunner] SemanticDriftDetector 已创建")
            _ensure_embedding_prewarm_async()
        except Exception as e:
            logger.warning(f"[IDEFCRunner] SemanticDriftDetector 创建失败: {e}")

    def _init_intent_filter(self) -> None:
        """初始化L1-B意图分类器（与Web端统一使用ALBERT模型）"""
        self._intent_filter = _get_shared_intent_filter()
        if self._intent_filter:
            if self._intent_filter._albert_enabled:
                logger.info("[IDEFCRunner] L1-B IntentFilter 已启用（ALBERT 15类）")
            else:
                logger.info("[IDEFCRunner] L1-B IntentFilter 已启用（关键词匹配）")

    def _send_message_safe(self, send_callback, msg_type: str, data: dict) -> bool:
        """线程安全的消息发送（修复no running event loop问题）
        
        三层容错机制：
        1. 优先使用全局主事件循环（_main_event_loop）
        2. 回退到当前线程的事件循环
        3. 异常时记录日志并返回False
        
        Args:
            send_callback: 异步发送回调函数
            msg_type: 消息类型（display_text/task_complete等）
            data: 消息数据
            
        Returns:
            发送是否成功
        """
        try:
            # 第一层：优先使用全局主事件循环（解决线程池工作线程问题）
            from zulong.ide.ide_server import _main_event_loop
            
            if _main_event_loop is not None and _main_event_loop.is_running():
                # 使用线程安全调度
                future = asyncio.run_coroutine_threadsafe(
                    send_callback(msg_type, data),
                    _main_event_loop
                )
                # 等待发送完成（带超时，避免阻塞）
                future.result(timeout=2.0)
                logger.debug(f"✅ [FC] {msg_type} 已发送（主循环）")
                return True
            
            # 第二层：回退到当前线程的事件循环（向后兼容）
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(send_callback(msg_type, data))
                logger.debug(f"✅ [FC] {msg_type} 已发送（当前循环）")
                return True
            except RuntimeError:
                pass
            
            # 第三层：降级处理
            logger.warning(f"⚠️ [FC] {msg_type} 发送跳过：无可用事件循环")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ [FC] {msg_type} 发送失败: {e}")
            return False

    async def _emit_execution_event(
        self,
        send_callback,
        phase: str,
        message: str,
        *,
        turn: int = 0,
        event_type: str = "TASK_PROGRESS",
        payload: Optional[Dict[str, Any]] = None,
        send_progress: bool = True,
        send_status: bool = False,
        monitor: bool = True,
    ) -> None:
        """Normalize runner state once, then fan out through existing channels."""
        payload = payload or {}
        interaction = self._build_interaction_payload(phase, message, turn, event_type, payload)
        event = ExecutionEvent(
            phase=phase,
            message=message,
            turn=turn,
            event_type=event_type,
            payload=payload,
            interaction=interaction,
        )
        self._execution_events.append(event)
        self._persist_execution_event(event)
        cb_payload = event.callback_payload(self._max_fc_turns)

        if send_progress:
            try:
                await send_callback("task_progress", cb_payload)
            except Exception:
                pass
        if send_status:
            try:
                await send_callback("status_update", {
                    "protocol_version": "2.0",
                    "turn": turn,
                    "phase": phase,
                    "message": message,
                    **payload,
                    "interaction": interaction,
                })
            except Exception:
                pass
        # ── TSD v2.7: 统一交互事件 emission (Web Dashboard 独立消费) ──
        if interaction:
            try:
                await send_callback(MessageType.INTERACTION_EVENT, {
                    "protocol_version": "2.0",
                    "interaction": interaction,
                    "turn": turn,
                    "event_type": event_type,
                    "phase": phase,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
        if monitor:
            try:
                from zulong.ide.ide_server import broadcast_monitor_event
                await broadcast_monitor_event(event_type, {
                    "session_id": self.session.session_id,
                    "conversation_id": getattr(self.ide_session, "conversation_id", None),
                    "turn_id": getattr(self.ide_session, "web_turn_id", None),
                    "workspace_path": getattr(self.ide_session, "cwd", None),
                    "turn": turn,
                    "phase": phase,
                    "message": message,
                    **payload,
                    "interaction": interaction,
                })
            except Exception:
                pass

    def _emit_execution_event_sync(
        self,
        phase: str,
        message: str,
        *,
        turn: int = 0,
        event_type: str = "TASK_PROGRESS",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = payload or {}
        interaction = self._build_interaction_payload(phase, message, turn, event_type, payload)
        event = ExecutionEvent(
            phase=phase,
            message=message,
            turn=turn,
            event_type=event_type,
            payload=payload,
            interaction=interaction,
        )
        self._execution_events.append(event)
        self._persist_execution_event(event)
        _broadcast_sync(event_type, {
            "session_id": self.session.session_id,
            "conversation_id": getattr(self.ide_session, "conversation_id", None),
            "turn_id": getattr(self.ide_session, "web_turn_id", None),
            "workspace_path": getattr(self.ide_session, "cwd", None),
            "turn": turn,
            "phase": phase,
            "message": message,
            **payload,
            "interaction": interaction,
        })

    async def _emit_memory_diff_async(self, send_callback, turn: int) -> None:
        """发射 graph:memory:diff 事件，获取 MemoryGraph 自上次快照以来的变化。
        
        TSD v1.7 §23.7: 在工具执行后、注意力变更时发射记忆差异。
        """
        try:
            from zulong.memory.memory_graph import get_memory_graph
            _mg = get_memory_graph()
            _mem_changes = {"created": 0, "strengthened": 0, "pruned": 0}
            if _mg:
                if hasattr(_mg, '_last_activated_edges'):
                    _mem_changes["strengthened"] = len(_mg._last_activated_edges)
                    _created = 0
                    for _sid in _mg.list_all_shards():
                        _shard = _mg.get_shard(_sid)
                        if _shard:
                            try:
                                _created += len(_shard.topology)
                            except Exception:
                                pass
                    _mem_changes["created"] = _created
                elif hasattr(_mg, 'stats'):
                    _mem_changes["created"] = _mg.stats.get("total_nodes", 0)
            await send_callback(MessageType.GRAPH_MEMORY_DIFF, {
                "memory_changes": _mem_changes,
                "turn": turn,
                "timestamp": time.time(),
            })
        except Exception as _e:
            logger.debug(f"[IDEFCRunner] 发射 graph:memory:diff 失败: {_e}")

    # ── TSD v2.7 §23.4.2: 工具风险等级分类 ──
    _HIGH_RISK_TOOLS = frozenset({
        "write_to_file", "replace_in_file", "execute_command",
        "delete_files", "create_rule", "delete_files_by_pattern",
        "preview_url", "use_skill", "web_fetch", "web_search",
        "ask_followup_question", "attempt_completion",
    })
    _CRITICAL_RISK_TOOLS = frozenset({
        "execute_command",  # 命令执行可被 CRITICAL
        "delete_files",     # 不可逆删除
    })

    @classmethod
    def _get_tool_risk_level(cls, tool_name: str, tool_args: str = "{}") -> str:
        """返回工具的风险等级: LOW / MEDIUM / HIGH / CRITICAL"""
        import json as _json
        if tool_name in cls._CRITICAL_RISK_TOOLS:
            # 进一步判断: execute_command 若含 rm -rf / sudo 等升级为 CRITICAL
            if tool_name == "execute_command":
                try:
                    args = _json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    command = args.get("command", "").lower()
                    dangerous = ("rm -rf", "sudo ", "chmod 777", "mkfs.", "dd if=",
                                ":(){ :|:& };:", "> /dev/sda", "format ")
                    if any(d in command for d in dangerous):
                        return "CRITICAL"
                except Exception:
                    pass
            return "HIGH"
        if tool_name in cls._HIGH_RISK_TOOLS:
            return "HIGH"
        return "LOW"

    def _next_interaction_id(self, prefix: str = "interaction") -> str:
        self._interaction_seq += 1
        return f"{prefix}:{self.session.session_id}:{int(time.time() * 1000)}:{self._interaction_seq}"

    def _build_interaction_payload(
        self,
        phase: str,
        message: str,
        turn: int,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """OpenHands-style visible interaction semantics layered on old events."""
        existing = payload.get("interaction")
        if isinstance(existing, dict):
            merged = dict(existing)
            merged.setdefault("interaction_id", self._next_interaction_id(phase))
            merged.setdefault("pair_id", merged.get("interaction_id"))
            merged.setdefault("title", message)
            merged.setdefault("detail", message)
            merged.setdefault("turn", turn)
            merged.setdefault("event_type", event_type)
            return merged

        kind = "state"
        status = "running"
        title = message
        detail = message
        pair_id = payload.get("pair_id") or payload.get("call_id") or payload.get("group_id")
        tool_name = payload.get("tool_name")
        risk_level = payload.get("risk_level", "")
        confirmation_state = payload.get("confirmation_state", "")

        if phase in {"tool_requested", "executing"}:
            kind = "action"
            status = "running"
            title = self._tool_action_title(payload, message)
        elif phase == "tool_finished":
            kind = "observation"
            status = "failed" if payload.get("is_error") else "succeeded"
            title = self._tool_observation_title(payload, message)
        elif phase in {"approval_required", "diff_ready", "checkpoint_created"}:
            kind = "approval" if phase == "approval_required" else "observation"
            status = "awaiting_approval" if phase == "approval_required" else payload.get("status", "succeeded")
            title = payload.get("action_summary") or payload.get("summary") or message
            risk_level = risk_level or payload.get("risk", "")
            confirmation_state = confirmation_state or ("awaiting_confirmation" if phase == "approval_required" else "")
        elif phase in {"waiting_model", "calling_model"}:
            kind = "state"
            status = "running"
            title = "正在推理"
        elif phase in {"blocked"}:
            kind = "state"
            status = "blocked"
            title = "执行受阻"
        elif phase in {"completed"}:
            kind = "summary"
            status = "succeeded"
            title = "任务完成"
        elif phase in {"cancelled", "interrupted"}:
            kind = "user_adjustment"
            status = "cancelled"
            title = "任务已被打断"
        elif phase in {"error"}:
            kind = "state"
            status = "failed"
            title = "执行出错"

        if not pair_id:
            pair_id = self._next_interaction_id(phase)

        # 审批模式确定 (TSD 23.4.2)
        approval_mode = payload.get("approval_mode", "")
        if phase == "approval_required" and not approval_mode:
            if risk_level == "CRITICAL":
                approval_mode = "popup"
            elif risk_level == "HIGH":
                approval_mode = "manual"
            elif risk_level == "MEDIUM":
                approval_mode = "manual"
            else:
                approval_mode = "manual"

        # L2 thought (为什么选择这个工具 — TSD 23.2.4)
        thought = payload.get("thought", "")

        return {
            "interaction_id": self._next_interaction_id(phase),
            "pair_id": str(pair_id),
            "kind": kind,
            "status": status,
            "title": title,
            "detail": detail,
            "thought": thought,
            "tool_name": tool_name or "",
            "tool_args": payload.get("tool_args", payload.get("args", None)),
            "risk_level": risk_level,
            "risk_reason": payload.get("risk_reason", ""),
            "approval_mode": approval_mode,
            "confirmation_state": confirmation_state,
            "progress": payload.get("progress"),
            "next_step": payload.get("next_step", ""),
            "turn": turn,
            "event_type": event_type,
            "phase": phase,
            "memory_changes": payload.get("memory_changes") if isinstance(payload.get("memory_changes"), dict) else None,
        }

    def _tool_action_title(self, payload: Dict[str, Any], fallback: str) -> str:
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            names = []
            for item in tools:
                if isinstance(item, dict):
                    names.append(item.get("name") or item.get("tool_name") or "")
                else:
                    names.append(str(item))
            names = [n for n in names if n]
            if names:
                return "准备执行工具: " + ", ".join(names[:4])
        if payload.get("tool_name"):
            return f"准备执行工具: {payload['tool_name']}"
        return fallback

    def _tool_observation_title(self, payload: Dict[str, Any], fallback: str) -> str:
        results = payload.get("results")
        if isinstance(results, list) and results:
            failed = sum(1 for item in results if isinstance(item, dict) and item.get("is_error"))
            if failed:
                return f"工具返回结果: {len(results) - failed} 成功，{failed} 失败"
            return f"工具返回结果: {len(results)} 个完成"
        if payload.get("tool_name"):
            return f"工具完成: {payload['tool_name']}"
        return fallback

    def _persist_execution_event(self, event: ExecutionEvent) -> None:
        """Best-effort event ledger + graph memory sink."""
        try:
            from zulong.launcher.interaction_store import get_interaction_store
            event_id = get_interaction_store().append_event(
                conversation_id=getattr(self.ide_session, "conversation_id", None),
                turn_id=getattr(self.ide_session, "web_turn_id", None),
                event_type=event.phase,
                role="system",
                source="ide_fc_runner",
                text=event.message,
                payload={
                    "event_type": event.event_type,
                    "phase": event.phase,
                    "turn": event.turn,
                    "session_id": self.session.session_id,
                    "source_event_id": "",
                    **event.payload,
                    "interaction": event.interaction,
                },
                workspace_path=getattr(self.ide_session, "cwd", None),
                project_id=getattr(self.ide_session, "project_id", None),
                task_graph_id=getattr(self.ide_session, "task_graph_id", None),
            )
            try:
                from zulong.launcher.memory_mirror import mirror_interaction_to_memory_graph
                from zulong.review.task_execution_extractor import maybe_finalize_task_execution_trace

                payload = {
                    "event_type": event.event_type,
                    "phase": event.phase,
                    "turn": event.turn,
                    "session_id": self.session.session_id,
                    "source_event_id": event_id,
                    **event.payload,
                    "interaction": event.interaction,
                }
                mirror_interaction_to_memory_graph(
                    conversation_id=getattr(self.ide_session, "conversation_id", None),
                    turn_id=getattr(self.ide_session, "web_turn_id", None),
                    role="system",
                    text=event.message,
                    event_type=event.phase,
                    source="ide_fc_runner",
                    payload=payload,
                )
                maybe_finalize_task_execution_trace(
                    conversation_id=getattr(self.ide_session, "conversation_id", None),
                    turn_id=getattr(self.ide_session, "web_turn_id", None),
                    task_graph_id=getattr(self.ide_session, "task_graph_id", None),
                    event_type=event.phase,
                )
            except Exception:
                pass
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] InteractionStore 事件记录跳过: {exc}")

        if event.phase in {
            "started",
            "tool_requested",
            "tool_finished",
            "diff_ready",
            "approval_required",
            "checkpoint_created",
            "blocked",
            "completed",
            "cancelled",
            "error",
        }:
            self._persist_event_to_memory(event)

    def _persist_event_to_memory(self, event: ExecutionEvent) -> None:
        try:
            from zulong.memory.memory_graph import (
                get_memory_graph, GraphNode, NodeType, Importance, EdgeType,
            )
            mg = get_memory_graph()
            if not mg:
                return
            now = time.time()
            node_id = f"exec:{self.session.session_id}:{event.created_at:.6f}:{event.phase}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.EPISODE,
                label=f"{event.phase}: {event.message[:80]}",
                activation=0.6,
                created_at=now,
                last_accessed=now,
                access_count=1,
                backend_ref=f"ide_session:{self.session.session_id}",
                metadata={
                    "event_phase": event.phase,
                    "event_type": event.event_type,
                    "message": event.message,
                    "turn": event.turn,
                    "conversation_id": getattr(self.ide_session, "conversation_id", None),
                    "turn_id": getattr(self.ide_session, "web_turn_id", None),
                    "workspace_path": getattr(self.ide_session, "cwd", None),
                    "payload": event.payload,
                    "interaction": event.interaction,
                    "source": "ide_fc_runner",
                },
            )
            mg.add_node(node)
            mg.set_importance(
                node_id,
                Importance.IMPORTANT if event.phase in {"completed", "blocked", "checkpoint_created"} else Importance.NORMAL,
            )
            try:
                mg.index_summary(node_id, f"{event.phase} {event.message} {_json.dumps(event.payload, ensure_ascii=False)[:500]}")
            except Exception:
                pass
            if self._current_round_id:
                try:
                    mg.add_edge(self._current_round_id, node_id, EdgeType.TEMPORAL, weight=0.4)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug(f"[IDEFCRunner] MemoryGraph 事件记录跳过: {exc}")

    def _init_dialogue_adapter(self) -> None:
        """创建 DialogueAdapter 实例（复用或新建）"""
        try:
            from zulong.memory.graph_adapters import DialogueAdapter
            self._dialogue_adapter = DialogueAdapter()
        except Exception as e:
            logger.warning(f"[IDEFCRunner] DialogueAdapter 创建失败: {e}")

    def _init_dialogue_tracking(self, state: IDEFCState) -> None:
        """新 FC 会话开始时初始化对话轮次跟踪

        在 MemoryGraph 中创建 session 和 round 节点，
        使后续工具执行产生的子对话能正确挂载到图谱中。
        """
        try:
            self._init_dialogue_adapter()
            if not self._dialogue_adapter:
                return
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return

            user_input = state.user_input_text or ""
            task_graph_id = self.session.active_task_graph_id

            # 确定或创建 session 节点
            self._current_session_id = self._dialogue_adapter.ensure_session(
                mg, user_input, task_graph_id=task_graph_id)

            # 创建本次对话轮次节点
            request_id = f"ide_{self.session.session_id[:12]}_{int(time.time())}"
            self._current_round_id = self._dialogue_adapter.add_round(
                mg, request_id=request_id, goal=user_input,
                task_graph_id=task_graph_id,
                session_id=self._current_session_id)

            # 绑定 session → task（使 BFS 遍历可从会话节点发现关联任务）
            if task_graph_id and self._current_session_id:
                self._dialogue_adapter.bind_session_to_task(
                    mg, self._current_session_id, task_graph_id)

            logger.info(
                f"[IDEFCRunner] 对话跟踪初始化: session={self._current_session_id}, "
                f"round={self._current_round_id}")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 对话跟踪初始化失败: {e}")

    def _record_sub_dialogue(self, state: IDEFCState,
                             tool_name: str, result: str) -> None:
        """记录一次工具执行为子对话节点"""
        if not self._dialogue_adapter or not self._current_round_id:
            return
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return
            self._dialogue_adapter.add_sub_dialogue(
                mg, round_id=self._current_round_id,
                turn=state.fc_turn, tool_name=tool_name,
                content=result[:200] if result else "",
                role="tool")
        except Exception as e:
            logger.debug(f"[IDEFCRunner] 子对话记录失败: {e}")

    def _finalize_dialogue_round(self, state: IDEFCState,
                                 status: str = "completed") -> None:
        """完成当前对话轮次，更新元数据并索引到 FAISS"""
        if not self._dialogue_adapter or not self._current_round_id:
            return
        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            if not mg:
                return
            self._dialogue_adapter.finalize_round(
                mg, round_id=self._current_round_id,
                total_turns=state.fc_turn, status=status)
            logger.info(
                f"[IDEFCRunner] 对话轮次完成: {self._current_round_id} "
                f"({state.fc_turn} turns, {status})")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 对话轮次完成记录失败: {e}")

    def _save_runner_state(self) -> None:
        if self._attn_window:
            try:
                self.session.attention_window_data = self._attn_window.serialize()
            except Exception as e:
                logger.warning(f"[IDEFCRunner] 注意力窗口序列化失败: {e}")
        if self._rule_guardian:
            try:
                self.session.rule_guardian_data = self._rule_guardian.serialize()
            except Exception as e:
                logger.warning(f"[IDEFCRunner] RuleGuardian 序列化失败: {e}")
        if self._circuit_breaker:
            try:
                self.session.circuit_breaker_data = self._circuit_breaker.serialize()
            except Exception as e:
                logger.warning(f"[IDEFCRunner] CircuitBreaker 序列化失败: {e}")
        # 对话轮次状态持久化
        self.session.dialogue_round_id = self._current_round_id
        self.session.dialogue_session_id = self._current_session_id
        # P1-8: 持久化session到磁盘
        try:
            if hasattr(self.session, 'session_id'):
                from zulong.ide.ide_server import get_session_store
                store = get_session_store()
                if store:
                    store.save_session(self.session)
        except Exception as e:
            logger.debug(f"[IDEFCRunner] session磁盘持久化跳过: {e}")

    def _init_state(self, messages: List[Dict]) -> IDEFCState:
        # 清除残留的中断标志（防止恢复任务时立即中断）
        if hasattr(self, 'engine') and self.engine:
            if getattr(self.engine, '_interrupt_flag', False):
                logger.info("[IDEFCRunner] 清除残留的中断标志")
                self.engine._interrupt_flag = False
        
        from zulong.models.container import LLM_MODEL_ID
        user_input = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                c = msg.get("content", "")
                if isinstance(c, str):
                    user_input = c
                elif isinstance(c, list):
                    # IDE 多模态格式: [{"type": "text", "text": "..."}, ...]
                    user_input = " ".join(
                        item.get("text", "")
                        for item in c
                        if isinstance(item, dict) and item.get("type") == "text"
                    )
                break

        # ── Layer 1: IDE 任务图策略，仅用于工具边界和任务图复用 ──
        _force_gid = getattr(self, 'force_graph_id', '') or ''
        task_graph_policy = "inspect_or_create"
        has_active_tg = False
        if _force_gid:
            # 确定性恢复: graph_id 已由 ide_server 加载，跳过启发式
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg and getattr(_tg, 'id', '') == _force_gid:
                task_graph_policy = "reuse"
                has_active_tg = True
                self.session.active_task_graph_id = _force_gid
                self._notify_session_linked(_force_gid)
                logger.info(
                    f"[IDEFCRunner] 确定性恢复模式: graph_id={_force_gid}")
            else:
                # 活跃图加载失败(不应发生), 降级到启发式
                logger.warning(
                    f"[IDEFCRunner] 确定性恢复降级: 活跃图不匹配 {_force_gid}")
                task_graph_policy, has_active_tg = self._detect_task_graph_policy(user_input)
        else:
            task_graph_policy, has_active_tg = self._detect_task_graph_policy(user_input)

        # 非确定性路径下，继续已有任务图时关联活跃图到 session
        if not _force_gid and task_graph_policy in {"reuse", "inspect", "continue"} and has_active_tg:
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg and hasattr(_tg, 'id') and not self.session.active_task_graph_id:
                self.session.active_task_graph_id = getattr(_tg, 'id', None)
                self._notify_session_linked(self.session.active_task_graph_id)
                logger.info(
                    f"[IDEFCRunner] 继续任务图策略：关联活跃图 "
                    f"{self.session.active_task_graph_id} 到新 session")

        # ── Layer 2: 根据任务图策略获取工具定义 ───────────
        tool_defs = self.tool_registry.get_combined_tool_definitions_for_policy(task_graph_policy)

        # 继续已有任务图时首轮强制 task_view_overview
        force_first = (task_graph_policy in {"reuse", "inspect", "continue"} and has_active_tg)

        state = IDEFCState(
            messages=list(messages), fc_turn=0, tool_definitions=tool_defs,
            user_input_text=user_input, vllm_model_id=LLM_MODEL_ID or "",
            phase="running", response_max_tokens=8192,
            is_resume=(task_graph_policy in {"reuse", "inspect", "continue"}),
            task_graph_policy=task_graph_policy,
            force_first_tool=force_first,
        )
        # ── TSD v2.7: L1BToolPredictor 工具预判接入 ──
        try:
            from zulong.l1b.tool_predictor import L1BToolPredictor
            _predictor = L1BToolPredictor()
            _conv_history = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            state.tool_prediction = _predictor.predict_tools(
                user_input, _conv_history
            )
            logger.debug(
                f"[IDEFCRunner] L1BToolPredictor: "
                f"turn_shape={state.tool_prediction.get('context_bundle', {}).get('turn_shape')}, "
                f"policy={state.tool_prediction.get('task_graph_policy')}, "
                f"suggested={state.tool_prediction.get('suggested_tools')[:5]}"
            )
        except Exception as _e:
            logger.debug(f"[IDEFCRunner] L1BToolPredictor 失败: {_e}")
            state.tool_prediction = None
        from zulong.l2.attention_window import AttentionWindowManager
        from zulong.tools.task_tools import get_active_task_graph
        from zulong.memory.memory_graph import get_memory_graph
        _init_tg = get_active_task_graph()
        _init_mg = get_memory_graph()
        self._attn_window = AttentionWindowManager(
            context_window_size=getattr(self.engine, "_context_window_size", 32768),
            task_graph=_init_tg,
            memory_graph=_init_mg,
        )
        for msg in messages:
            self._attn_window.register_message(msg, turn=0, pinned=msg.get("role") == "system")
        self._create_rule_guardian()
        self._create_circuit_breaker()
        self._create_drift_detector()
        self._auto_create_task_plan(state)
        self._init_dialogue_tracking(state)
        logger.info(
            f"[IDEFCRunner] task_graph_policy={task_graph_policy}, "
            f"force_first_tool={force_first}, tools={len(tool_defs)}, "
            f"has_active_tg={has_active_tg}"
        )
        return state

    def _detect_task_graph_policy(self, user_input: str) -> tuple:
        """根据引用、活跃任务图和 L1-B 细粒度信号预判任务图策略。"""
        has_active_tg = False
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg is not None:
                has_active_tg = True
        except Exception:
            pass

        ref_graph_id = self._try_activate_from_reference(user_input)
        if ref_graph_id:
            return "reuse", True

        if any(
            kw in (user_input or "").strip().lower()
            for kw in ("继续", "接着做", "接着", "恢复", "上次", "之前的任务",
                       "resume", "continue", "pick up")
        ):
            if has_active_tg:
                return "reuse", has_active_tg
            else:
                try:
                    from zulong.tools.task_tools import (
                        load_latest_backup, set_active_task_graph
                    )
                    backup_tg, backup_gid = load_latest_backup()
                    if backup_tg and backup_gid:
                        set_active_task_graph(backup_tg, backup_gid)
                        return "reuse", True
                except Exception as e:
                    logger.debug(f"[IDEFCRunner] 备份加载尝试失败: {e}")

        if self._intent_filter:
            try:
                intent_result = self._intent_filter.analyze(user_input)
                predicted = (intent_result.get("intent", "UNKNOWN") or "").lower()
                if predicted in ("task_code", "task_analysis", "task_write", "task_execute", "task_search", "task_read"):
                    return "inspect_or_create", has_active_tg
            except Exception as e:
                logger.debug(f"[IDEFCRunner] IntentFilter 检查失败: {e}")

        return "inspect_or_create", has_active_tg

    def _try_activate_from_reference(self, user_input: str) -> Optional[str]:
        """尝试从用户输入中的 @[label#address] 引用激活历史 TaskGraph

        流程:
        1. 正则匹配 @[...#tg:xxx/task:yyy] 格式
        2. 通过 MemoryGraph.resolve_address() 定位节点
        3. 从节点 metadata 提取 graph_id
        4. 调用 rebuild_task_graph_from_memory() 重建
        5. 设置为活跃图

        Returns:
            成功时返回 graph_id，失败返回 None
        """
        import re as _re
        if not user_input:
            return None

        # 匹配 @[任意标签#地址] 格式，地址部分以 tg: 开头
        pattern = r'@\[([^#\]]+)#(tg:[^\]]+)\]'
        match = _re.search(pattern, user_input)
        if not match:
            return None

        label = match.group(1)
        address = match.group(2)
        logger.debug(f"[IDEFCRunner] 检测到节点引用: label={label}, address={address}")

        try:
            from zulong.memory.memory_graph import get_memory_graph
            mg = get_memory_graph()
            node = mg.resolve_address(address)
            if node is None:
                logger.warning(
                    f"[IDEFCRunner] 无法解析地址 {address}，MemoryGraph 中未找到")
                return None

            # 提取 graph_id: 从地址 "tg:{graph_id}/task:{node_id}" 解析
            graph_id = None
            if address.startswith("tg:"):
                parts = address.split("/")
                graph_id = parts[0][3:]  # 去掉 "tg:" 前缀

            if not graph_id:
                graph_id = node.metadata.get("graph_id")

            if not graph_id:
                logger.warning(
                    f"[IDEFCRunner] 无法从节点引用中提取 graph_id: {address}")
                return None

            # 检查是否已经是活跃图
            from zulong.tools.task_tools import (
                get_active_task_graph, set_active_task_graph
            )
            current_tg = get_active_task_graph()
            if current_tg and getattr(current_tg, 'id', '') == graph_id:
                logger.info(
                    f"[IDEFCRunner] 引用的图谱已是活跃图: {graph_id}")
                return graph_id

            # 从 MemoryGraph 重建 TaskGraph
            from zulong.memory.graph_adapters import rebuild_task_graph_from_memory
            rebuilt_tg = rebuild_task_graph_from_memory(mg, graph_id)
            if rebuilt_tg is None:
                logger.warning(
                    f"[IDEFCRunner] 重建 TaskGraph 失败: {graph_id}")
                return None

            # 设置为活跃图
            set_active_task_graph(rebuilt_tg, graph_id)
            logger.info(
                f"[IDEFCRunner] 通过节点引用激活 TaskGraph: "
                f"graph_id={graph_id}, nodes={len(rebuilt_tg._nodes)}")
            return graph_id

        except Exception as e:
            logger.warning(f"[IDEFCRunner] 节点引用激活失败: {e}")
            return None

    def _infer_active_node_id(self) -> Optional[str]:
        """推断当前活跃的 TaskGraph 节点 ID（供远程工具结果关联）"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg:
                ip = tg.get_nodes_by_status("in_progress")
                if ip:
                    return ip[0].id
        except Exception:
            pass
        return None

    def _inject_tool_results(self, state: IDEFCState, tool_results: List[Dict]) -> None:
        """注入远程工具执行结果（提升为一等公民，与 _exec_internal 对等处理）"""
        # ── 安全验证 ──
        # 数量上限：不超过 pending_call_ids 长度的 2 倍
        max_results = max(len(state.pending_call_ids) * 2, 1)
        if len(tool_results) > max_results:
            logger.warning(
                f"[IDEFCRunner] 工具结果数量异常: {len(tool_results)} > "
                f"pending*2={max_results}, 截断")
            tool_results = tool_results[:max_results]

        # call_id 白名单
        valid_ids = set(state.pending_call_ids)

        # 构建 call_id → 原始工具信息映射
        call_id_to_func = {}
        for rc in state.pending_remote_calls:
            call_id_to_func[rc["id"]] = rc.get("function", {})

        active_node_id = self._infer_active_node_id()

        for tr in tool_results:
            call_id = tr["tool_call_id"]

            # 验证 call_id 属于 pending 白名单
            if call_id not in valid_ids:
                logger.warning(
                    f"[IDEFCRunner] 拒绝未知 call_id: {call_id}, "
                    f"valid={list(valid_ids)[:5]}")
                continue
            
            # 从白名单移除，防止重复注入
            valid_ids.discard(call_id)
            # 同时从pending_call_turns移除
            if hasattr(state, 'pending_call_turns'):
                state.pending_call_turns.pop(call_id, None)

            func_info = call_id_to_func.get(call_id, {})
            tool_name = func_info.get("name", "ide_remote")
            content = tr["content"]

            # 截断保护（与 _exec_internal 一致）
            if len(content) > MAX_TOOL_RESULT_CHARS:
                orig_len = len(content)
                content = content[:MAX_TOOL_RESULT_CHARS] + \
                    f"\n...(已截断，原始长度 {orig_len} 字符)"

            tm = {"role": "tool", "tool_call_id": call_id, "content": content}
            state.messages.append(tm)

            if self._attn_window:
                self._attn_window.register_message(
                    tm, turn=state.fc_turn,
                    tool_name=tool_name, node_id=active_node_id)
                # 触发注意力模式切换
                try:
                    args_dict = _json.loads(func_info.get("arguments", "{}"))
                except Exception:
                    args_dict = {}
                self._attn_window.observe_tool_call(tool_name, args_dict)

            # CircuitBreaker 记录（与 _exec_internal 一致）
            if self._circuit_breaker:
                self._circuit_breaker.record_call(tool_name, {}, content[:500])

            # 工具结果缓冲（供 InfoGap / Backfill 使用），限制上限
            if len(state.tool_results_buffer) >= _TOOL_RESULTS_BUFFER_MAX:
                state.tool_results_buffer.pop(0)
            state.tool_results_buffer.append(
                {"tool_name": tool_name, "result": content[:500]})

        # 仅在所有pending结果都已处理后清空
        # pending_call_turns 通过 discard 机制自动管理，无需手动清空
        state.pending_remote_calls = []
        state.pending_call_ids = list(state.pending_call_turns.keys()) if hasattr(state, 'pending_call_turns') else []
        # 推送 TaskGraph 更新到 web 仪表盘（远程工具结果返回后）
        try:
            self.engine._publish_task_graph_event(
                "agent_tool_call", state.fc_turn,
                "ide_tool_results", f"远程工具结果注入: {len(tool_results)} 条")
        except Exception:
            pass

        # ── 混合自动锚定：write_to_file / replace_in_file 后置钩子 ──
        # 检测写文件工具并自动触发 CRG 索引 + TASK→CODE_SYMBOL 锚定
        self._auto_anchor_on_write(tool_results, call_id_to_func, active_node_id)

    # ── 混合自动锚定核心方法 ──────────────────────────────────

    _WRITE_TOOLS = {"write_to_file", "replace_in_file", "create_file", "insert_code_block"}

    def _auto_anchor_on_write(
        self,
        tool_results: List[Dict],
        call_id_to_func: Dict[str, Dict],
        active_node_id: Optional[str],
    ) -> None:
        """write_to_file / replace_in_file 后自动触发 CRG 索引 + TASK→CODE_SYMBOL 锚定

        策略:
        - 仅对代码文件（支持的扩展名）触发
        - 通过 MD5 去重避免重复索引
        - 自动为当前活跃 TASK 节点建立 TASK→CODE_SYMBOL REFERENCE 边
        - 自动创建 CodeAnchor 记录（owner_ref 指向任务节点）
        - 自动建立 TaskGraph d_edge（任务节点→代码符号节点）
        """
        try:
            # 收集本批次中的写文件路径
            written_files = set()
            for tr in tool_results:
                call_id = tr.get("tool_call_id", "")
                func_info = call_id_to_func.get(call_id, {})
                tool_name = func_info.get("name", "")
                if tool_name not in self._WRITE_TOOLS:
                    continue

                # 从参数中提取 file_path
                args_str = func_info.get("arguments", "{}")
                try:
                    args = _json.loads(args_str)
                except Exception:
                    args = {}
                fp = args.get("path") or args.get("file_path") or ""
                if fp:
                    written_files.add(fp.replace("\\", "/"))

            if not written_files:
                return

            # 判断是否为可索引的代码文件
            import os
            try:
                from zulong.code.graph_builder import ext_to_lang
            except ImportError:
                return

            code_files = []
            for fp in written_files:
                ext = os.path.splitext(fp)[1]
                if ext_to_lang(ext):
                    code_files.append(fp)

            if not code_files:
                return

            # 延迟导入所需模块
            from zulong.memory.memory_graph import get_memory_graph, NodeType, EdgeType

            mg = get_memory_graph()
            if mg is None:
                return

            adapter = mg._adapters.get("code_graph")
            if adapter is None:
                try:
                    from zulong.memory.graph_adapters import register_all_adapters
                    register_all_adapters(mg)
                    adapter = mg._adapters.get("code_graph")
                except Exception:
                    pass
            if adapter is None:
                return

            # 获取或创建 IndexCodeFileTool 实例（复用哈希缓存）
            if not hasattr(self, "_index_tool_instance"):
                from zulong.tools.code_tools import IndexCodeFileTool
                self._index_tool_instance = IndexCodeFileTool()
                self._index_tool_instance.initialize()

            index_tool = self._index_tool_instance

            # 推断 MemoryGraph 中的 TASK 节点 ID
            task_mg_id = None
            if active_node_id:
                # active_node_id 来自 TaskGraph（短 ID），转为 MG 中的完整 ID
                from zulong.tools.task_tools import get_active_task_graph
                tg = get_active_task_graph()
                if tg:
                    candidate = f"task:{tg.id}/{active_node_id}"
                    if mg.has_node(candidate):
                        task_mg_id = candidate
                    else:
                        # 退化：搜索含 active_node_id 后缀的 TASK 节点
                        for nid, nd in mg._nodes.items():
                            if nd.node_type == NodeType.TASK and nid.endswith(active_node_id):
                                task_mg_id = nid
                                break

            for fp in code_files:
                self._index_and_anchor_file(fp, mg, adapter, index_tool, task_mg_id)

        except Exception as e:
            logger.debug(f"[IDEFCRunner] _auto_anchor_on_write 异常（不影响主流程）: {e}")

    def _index_and_anchor_file(
        self, file_path: str, mg, adapter, index_tool, task_mg_id: Optional[str]
    ) -> None:
        """对单个文件执行 CRG 索引 + 自动锚定边"""
        import os
        import hashlib
        from pathlib import Path
        from zulong.memory.memory_graph import NodeType, EdgeType

        # 读取文件内容
        source_content = ""
        try:
            candidates = [Path(file_path), Path(".") / file_path]
            # 也尝试工作区路径
            if hasattr(self, 'cwd') and self.cwd:
                candidates.insert(0, Path(self.cwd) / file_path)
            for p in candidates:
                if p.exists():
                    source_content = p.read_text(encoding="utf-8", errors="replace")
                    break
        except Exception:
            pass

        if not source_content:
            return

        # MD5 去重：内容未变则跳过
        content_hash = hashlib.md5(
            source_content.encode("utf-8", errors="replace")).hexdigest()
        if index_tool._indexed_hashes.get(file_path) == content_hash:
            return

        # 执行 Tree-sitter 解析 + 增量同步
        try:
            from zulong.code.ast_parser import ASTParser
            from zulong.code.graph_builder import CodeGraphBuilder, CodeEdge, ext_to_lang

            ext = os.path.splitext(file_path)[1]
            lang = ext_to_lang(ext)
            if not lang:
                return

            parser = ASTParser(lang)
            if not parser.available:
                return

            source_bytes = source_content.encode("utf-8", errors="replace")
            result = parser.parse_source(source_bytes, file_path)
            if result.parse_error:
                return

            for sym in result.symbols:
                sym.file_path = file_path

            # 构建边（含跨文件）
            edges = CodeGraphBuilder._build_edges_for_file(result)

            local_sym_names = {s.name for s in result.symbols}
            local_sym_names.update(s.qualified_name for s in result.symbols)
            local_node_ids = {s.node_id for s in result.symbols}

            global_sym_index = {}
            for nid, node in mg._nodes.items():
                if node.node_type == NodeType.CODE_SYMBOL:
                    global_sym_index[node.label] = nid
                    short = node.label.rsplit(".", 1)[-1]
                    if short not in global_sym_index:
                        global_sym_index[short] = nid

            file_node_id = f"file:{file_path}"
            for imp in result.imports:
                if imp.is_from and imp.names:
                    for name in imp.names:
                        target_id = global_sym_index.get(name)
                        if target_id and target_id not in local_node_ids:
                            edges.append(CodeEdge(
                                source_id=file_node_id,
                                target_id=target_id,
                                edge_type="imports",
                                metadata={"line": imp.line, "module": imp.module},
                            ))

            for call in result.calls:
                if call.callee in local_sym_names:
                    continue
                target_id = global_sym_index.get(call.callee)
                if target_id:
                    caller_id = None
                    for s in result.symbols:
                        if s.qualified_name == call.caller:
                            caller_id = s.node_id
                            break
                    if caller_id:
                        edges.append(CodeEdge(
                            source_id=caller_id,
                            target_id=target_id,
                            edge_type="calls",
                            metadata={"line": call.line, "cross_file": True},
                        ))

            # 增量同步到 MemoryGraph
            adapter.incremental_sync(mg, "file_updated", {
                "file_path": file_path,
                "symbols": result.symbols,
                "edges": edges,
                "content_hash": content_hash,
                "project_root": getattr(self, 'cwd', '') or '',
            })

            # 记录哈希
            index_tool._indexed_hashes[file_path] = content_hash

            # 广播 CRG 索引事件到 WEB 面板（双通道）
            crg_update_payload = {
                "file_path": file_path,
                "symbol_count": len(result.symbols),
                "edge_count": len(edges),
                "content_hash": content_hash,
            }
            _broadcast_sync("CRG_INDEX_UPDATE", crg_update_payload)
            try:
                from zulong.launcher.web_chat_router import _schedule_broadcast
                _schedule_broadcast({
                    "type": "CRG_INDEX_UPDATE",
                    "payload": crg_update_payload,
                })
            except Exception:
                pass

            # ── 自动锚定：TASK → CODE_SYMBOL 边 + CodeAnchor 记录 + TaskGraph d_edge ──
            if task_mg_id and mg.has_node(task_mg_id):
                anchored = 0
                anchor_ids = []
                try:
                    from zulong.memory.code_anchor import CodeAnchor, get_code_anchor_store, compute_content_hash, get_current_commit_sha
                    anchor_store = get_code_anchor_store()
                    commit_sha = get_current_commit_sha()
                except Exception:
                    anchor_store = None
                    commit_sha = None

                for sym in result.symbols:
                    code_node_id = sym.node_id
                    if mg.has_node(code_node_id):
                        mg.add_edge(
                            task_mg_id, code_node_id,
                            edge_type=EdgeType.REFERENCE,
                            weight=0.6,
                            metadata={
                                "relation": "auto_anchored",
                                "anchor_type": "implementation",
                                "source": "write_hook",
                            },
                        )
                        anchored += 1

                        # 创建 CodeAnchor 记录
                        if anchor_store:
                            try:
                                snippet = source_content.split("\n")
                                start = max(0, sym.start_line - 1)
                                end = min(len(snippet), sym.end_line)
                                preview = "\n".join(snippet[start:start + 3])
                                sym_content = "\n".join(snippet[start:end])
                                anchor = CodeAnchor(
                                    id=compute_content_hash(f"{file_path}:{sym.node_id}:{time.time()}")[:12],
                                    file_path=file_path,
                                    symbol=sym.name,
                                    line_start=sym.start_line,
                                    line_end=sym.end_line,
                                    commit_sha=commit_sha,
                                    content_hash=compute_content_hash(sym_content),
                                    anchor_type="implementation",
                                    snippet_preview=preview[:200],
                                    owner_ref=task_mg_id.replace("task:", "tg:", 1) if task_mg_id.startswith("task:") else task_mg_id,
                                )
                                anchor_store.add_anchor(anchor)
                                anchor_ids.append(anchor.id)
                            except Exception:
                                pass

                if anchored:
                    logger.info(
                        f"[IDEFCRunner] 自动锚定: {file_path} → "
                        f"TASK({task_mg_id}) ↔ {anchored} CODE_SYMBOL 节点, "
                        f"{len(anchor_ids)} CodeAnchor 记录"
                    )

                # 关联 TaskGraph 符号节点（d_edge）
                try:
                    from zulong.tools.task_tools import get_active_task_graph
                    tg = get_active_task_graph()
                    if tg and active_node_id:
                        from zulong.code.graph_builder import ext_to_lang as _ext_to_lang
                        proj_name = getattr(tg, '_project_name', '') or ''
                        if not proj_name:
                            import os
                            proj_name = os.path.basename(getattr(self, 'cwd', '') or '.')
                        for sym in result.symbols:
                            sym_tg_id = f"crg_{proj_name}/sym:{sym.node_id}"
                            if tg.get_node(sym_tg_id) and tg.get_node(active_node_id):
                                tg.add_d_edge(active_node_id, sym_tg_id, via=f"implements {sym.name}", cross=True)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"[IDEFCRunner] _index_and_anchor_file({file_path}) 异常: {e}")

    def _run_loop(self, state: IDEFCState) -> IDEFCResult:
        # 设置 FC 循环运行状态为 True（禁止节点审查提交）
        try:
            from zulong.core.state_manager import state_manager
            state_manager.set_fc_loop_running(True)
        except Exception:
            pass
        
        while True:
            tr = self._check(state)
            if tr:
                # 清除 FC 循环运行状态
                try:
                    from zulong.core.state_manager import state_manager
                    state_manager.set_fc_loop_running(False)
                except Exception:
                    pass
                return self._finalize(state, tr)
            try:
                # FC 请求间隔：防止 API 被打满（跳过第一轮）
                if state.fc_turn > 1 and self._fc_request_interval > 0:
                    time.sleep(self._fc_request_interval)
                self._publish_fc_progress(state, "calling_model", f"turn={state.fc_turn}")
                tc_data, resp_content = self._call_model(state)
                if tc_data is None and resp_content is None:
                    if state.api_timeout_count >= 2:
                        self._publish_fc_progress(state, "api_error", "连续API错误终止")
                        # 清除 FC 循环运行状态
                        try:
                            from zulong.core.state_manager import state_manager
                            state_manager.set_fc_loop_running(False)
                        except Exception:
                            pass
                        return self._finalize(state, "api_error")
                    time.sleep(2)  # API 错误后短暂退避
                    continue
                # 成功获得模型响应，重置连续错误计数
                state.loop_error_count = 0
                if tc_data:
                    # 模型调用了工具 → 重置独白计数器
                    state.consecutive_text_only_count = 0
                    self._publish_fc_progress(state, "exec_tools", f"{len(tc_data)} tool calls")
                    remote = self._exec_tools(state, tc_data, resp_content)
                    if remote:
                        self._publish_fc_progress(state, "pause_for_remote", f"{len(remote)} remote tools")
                        # 注意：暂停不清除状态，恢复后会继续运行
                        return self._pause_for_remote(state, remote)
                    continue
                # 模型纯文本回复（无工具调用）→ 递增独白计数器
                state.consecutive_text_only_count += 1
                verdict = self._eval_response(state, resp_content or "")
                if verdict == "done":
                    state.phase = "done"
                    
                    # 清除 FC 循环运行状态（允许节点审查提交）
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    
                    # 🔥 修复：提取 submit_final_answer 内容
                    _final_answer = self._extract_final_answer(state)
                    if _final_answer:
                        state.final_answer = _final_answer

                    # 将 FC 统计写入 TaskGraph 元数据
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph()
                        if _tg and hasattr(_tg, "metadata"):
                            _tg.metadata["total_turns"] = state.fc_turn
                            _tg.metadata["duration"] = time.time() - getattr(_tg, "created_at", time.time())
                    except Exception:
                        pass

                    self._auto_complete_task(state)
                    self._finalize_dialogue_round(state, status="completed")
                    self._auto_save_session_memory(state)
                    self._save_runner_state()
                    self.session.fc_state = state

                    # 🔥 修复：FC 完成后重新归档，补全 final_answer/duration/turns
                    try:
                        from zulong.tools.task_tools import get_active_task_graph, _auto_archive_completed
                        _tg = get_active_task_graph()
                        if _tg:
                            _auto_archive_completed(_tg)
                    except Exception:
                        pass

                    return IDEFCResult(phase="done", text_response=state.final_answer or state.last_response_content)
                elif verdict == "cb_force":
                    state.cb_force_no_tools = True
                # verdict == "continue" → 继续循环
            except Exception as loop_err:
                logger.error(
                    f"[IDEFCRunner] 循环体异常 turn={state.fc_turn}: {loop_err}",
                    exc_info=True)
                state.loop_error_count += 1
                if state.loop_error_count >= 3:
                    logger.error("[IDEFCRunner] 连续 3 次循环异常，终止 FC")
                    # 清除 FC 循环运行状态
                    try:
                        from zulong.core.state_manager import state_manager
                        state_manager.set_fc_loop_running(False)
                    except Exception:
                        pass
                    return self._finalize(state, "loop_error")
                continue

    def _check(self, state: IDEFCState) -> str:
        """迭代守卫：软限制注入进度提示，硬限制触发弹性续期或终止

        返回值:
          ""           — 继续
          "interrupted" — 外部中断
          "checkpoint"  — 安全阀触发，强制终止
        """
        state.fc_turn += 1
        fc = state.fc_turn
        if getattr(self.engine, "_interrupt_flag", False):
            logger.info("[IDEFCRunner] 外部中断")
            return "interrupted"
        if fc % self._warning_interval == 0:
            logger.info(f"[IDEFCRunner] 进度: {fc}/{self._hard_limit}")
        
        # 重复工具调用死循环检测（从 FCRunner 继承并适配 IDEFCState）
        self._detect_duplicate_tool_loop_ide(state)
        
        # 周期性进度广播（每 _progress_report_interval 轮，独立于 hard_limit）
        # 仅推送到 Web 仪表盘，不注入消息、不影响 FC 循环控制流
        if (fc > 1
                and self._progress_report_interval > 0
                and fc % self._progress_report_interval == 0
                and fc < self._hard_limit):
            self._broadcast_periodic_progress(state)
        if fc > self._soft_limit and fc % self._warning_interval == 1:
            # 软限制：注入进度提醒到消息列表，引导 LLM 收敛
            report = self._build_progress_hint(state)
            logger.warning(f"[IDEFCRunner] 超软限制 ({self._soft_limit}), 注入进度提示")
            hint_msg = {"role": "system", "content": report}
            state.messages.append(hint_msg)
            if self._attn_window:
                # 独立 group_id：避免被 None 组膨胀后整体淘汰
                gid = self._attn_window.new_tool_group()
                self._attn_window.register_message(hint_msg, turn=fc, group_id=gid)
        if fc >= self._hard_limit:
            # 生成结构化进度报告
            progress = self._generate_progress_report(state)
            state.progress_reports.append(progress)
            state.last_report_turn = fc
            # 安全阀: 连续报告无进展 → 强制终止
            if self._is_progress_stalled(state):
                logger.warning(
                    f"[IDEFCRunner] 安全阀: 连续 {self._max_reports_before_force_stop} "
                    f"次报告无进展，强制终止"
                )
                self._save_runner_state()
                return "checkpoint"
            # 自动续期（无次数上限，只要有进展就持续续期）
            if not self._auto_continue:
                logger.warning(
                    f"[IDEFCRunner] 到达硬限制 ({self._hard_limit}), "
                    f"auto_continue=off，终止"
                )
                self._save_runner_state()
                return "checkpoint"
            # 弹性预算续期
            state.auto_continue_count += 1
            old_limit = self._hard_limit
            self._hard_limit += self._progress_report_interval
            logger.info(
                f"[IDEFCRunner] 弹性续期 #{state.auto_continue_count}: "
                f"硬限制 {old_limit} → {self._hard_limit}"
            )
            # 注入进度报告到消息列表，让 LLM 知道当前状态
            renewal_msg = {
                "role": "system",
                "content": (
                    f"[进度报告 #{state.auto_continue_count}] "
                    f"已执行 {fc} 步，预算已自动续期至 {self._hard_limit} 步。"
                    f"已完成 {progress.get('completed_count', 0)} 个节点，"
                    f"进行中 {progress.get('in_progress_count', 0)} 个，"
                    f"待处理 {progress.get('pending_count', 0)} 个。"
                    f"请继续推进任务。"
                ),
            }
            state.messages.append(renewal_msg)
            if self._attn_window:
                # 取消上一次续期消息的 pinned（只保留最新一条 pinned）
                self._unpin_old_renewals()
                # 独立 group_id + pinned：续期指令是 FC 继续运转的核心信号
                gid = self._attn_window.new_tool_group()
                self._attn_window.register_message(
                    renewal_msg, turn=fc, group_id=gid, pinned=True)
        return ""

    def _detect_duplicate_tool_loop_ide(self, state: IDEFCState) -> None:
        """重复工具调用死循环检测（IDEFCState 适配版）

        检查最近几轮中是否连续调用相同的工具且参数相同。
        如果检测到死循环，注入 CB 强制收敛信号。
        """
        fc = state.fc_turn
        if len(state.messages) < 6 or fc <= 5:
            return

        last_tool_calls = []
        for msg in reversed(state.messages[-6:]):
            tool_calls = msg.get("tool_calls", [])
            if tool_calls and len(tool_calls) > 0:
                tc = tool_calls[0]["function"]
                last_tool_calls.append({
                    "name": tc["name"],
                    "args": tc.get("arguments", ""),
                })

        if len(last_tool_calls) >= self._DUPLICATE_TOOL_CHECK_TURNS:
            tool_names = [tc["name"] for tc in last_tool_calls]
            tool_args = [tc["args"] for tc in last_tool_calls]

            if len(set(tool_names)) == 1 and len(set(tool_args)) == 1:
                logger.warning(
                    f"[IDEFCRunner] 检测到死循环: "
                    f"连续{len(last_tool_calls)}轮调用 {tool_names[0]} 且参数相同"
                )
                state.cb_force_no_tools = True
                cm = {
                    "role": "user",
                    "content": (
                        f"[系统警告] 检测到重复工具调用循环（{tool_names[0]}），"
                        f"请立即基于已有信息生成最终回复，不允许再调用工具。"
                    ),
                }
                state.messages.append(cm)
                if self._attn_window:
                    self._attn_window.register_message(cm, turn=fc)

    def _build_progress_hint(self, state: IDEFCState) -> str:
        """构建进度提示，注入到 LLM 上下文（中性通报，不催促结束）"""
        fc = state.fc_turn
        hint = f"[系统进度通报] 当前已执行 {fc} 步。"
        # 附加任务图进度（如果有）
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg:
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            done = sum(1 for n in all_nodes if n.status in ("completed", "skipped"))
            wip = sum(1 for n in all_nodes if n.status == "in_progress")
            todo = sum(1 for n in all_nodes if n.status in ("pending", ""))
            hint += f" 任务进度: {done} 已完成, {wip} 进行中, {todo} 待处理。"
        return hint

    def _unpin_old_renewals(self) -> None:
        """取消旧续期消息的 pinned 状态，避免 pinned 累积膨胀

        只保留最新一条续期消息为 pinned，旧的降级为普通消息参与权重淘汰。
        """
        if not self._attn_window:
            return
        _RENEWAL_PREFIX = "[进度报告 #"
        for env in self._attn_window.envelopes:
            if (env.is_pinned
                    and env.msg.get("role") == "system"
                    and isinstance(env.msg.get("content"), str)
                    and env.msg["content"].startswith(_RENEWAL_PREFIX)):
                env.is_pinned = False

    def _get_progress_snapshot(self) -> dict:
        """获取任务图谱进度快照（轻量，用于每轮 status_update）"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if not tg:
                return {}
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            return {
                "total_nodes": len(all_nodes),
                "completed_count": sum(1 for n in all_nodes if n.status in ("completed", "skipped")),
                "in_progress_count": sum(1 for n in all_nodes if n.status == "in_progress"),
                "pending_count": sum(1 for n in all_nodes if n.status in ("pending", "")),
            }
        except Exception:
            return {}

    def _broadcast_periodic_progress(self, state: IDEFCState) -> None:
        """周期性进度广播（不触发续期，仅通知 Web 仪表盘当前状态）"""
        from zulong.tools.task_tools import get_active_task_graph
        # 刷新 Gatekeeper 空闲计时器，防止 FC 循环执行中被误判空闲挂起
        try:
            from zulong.l1b.scheduler_gatekeeper import gatekeeper
            if gatekeeper:
                gatekeeper.touch_idle_timer()
        except Exception:
            pass
        tg = get_active_task_graph()
        report = {"turn": state.fc_turn, "type": "periodic"}
        if tg:
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            report["total_nodes"] = len(all_nodes)
            report["completed_count"] = sum(
                1 for n in all_nodes if n.status in ("completed", "skipped"))
            report["in_progress_count"] = sum(
                1 for n in all_nodes if n.status == "in_progress")
            report["pending_count"] = sum(
                1 for n in all_nodes if n.status in ("pending", ""))
        _broadcast_sync("PROGRESS_REPORT", {
            "session_id": self.session.session_id,
            "turn": state.fc_turn,
            "report": report,
            "type": "periodic",
        })
        # 同时推送到 IDE 端（send_callback）
        try:
            import asyncio
            cb = getattr(self, '_ide_send_callback', None)
            if cb:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(
                    cb("status_update", {
                        "turn": state.fc_turn,
                        "phase": "running",
                        "progress": report,
                    }), loop)
        except Exception:
            pass

    def _generate_progress_report(self, state: IDEFCState) -> Dict:
        """生成结构化进度报告，用于弹性续期决策和 Web 推送"""
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        report = {
            "turn": state.fc_turn,
            "elapsed_turns": state.fc_turn,
            "completed_count": 0,
            "in_progress_count": 0,
            "pending_count": 0,
            "total_nodes": 0,
            "completed": [],
            "in_progress": [],
            "pending": [],
        }
        if tg:
            # 统计全部节点（排除 req 根节点），不限于叶节点
            all_nodes = [n for n in tg.nodes if n.id != "req"]
            report["total_nodes"] = len(all_nodes)
            for node in all_nodes:
                entry = {"id": node.id, "label": node.label}
                if node.status in ("completed", "skipped"):
                    report["completed_count"] += 1
                    if node.result:
                        entry["result_preview"] = node.result[:100]
                    report["completed"].append(entry)
                elif node.status == "in_progress":
                    report["in_progress_count"] += 1
                    report["in_progress"].append(entry)
                else:
                    report["pending_count"] += 1
                    report["pending"].append(entry)
        logger.info(
            f"[IDEFCRunner] 进度报告: turn={state.fc_turn}, "
            f"total={report['total_nodes']}, "
            f"done={report['completed_count']}, "
            f"wip={report['in_progress_count']}, "
            f"todo={report['pending_count']}"
        )
        # 同步推送到 Web（确保每次报告用户都能看到）
        _broadcast_sync("PROGRESS_REPORT", {
            "session_id": self.session.session_id,
            "turn": state.fc_turn,
            "report": report,
            "auto_continue_count": state.auto_continue_count,
        })
        return report

    def _is_progress_stalled(self, state: IDEFCState) -> bool:
        """检查连续进度报告是否停滞

        停滞条件：最近 N 次报告中 completed_count 和 total_nodes 都没有增长
        （即既没完成节点、也没创建新节点 → 真正的死循环）
        
        🔥 P1修复：增强检测逻辑
        - completed_count增加 → 有进展
        - in_progress_count变化 → 有变化
        - 消息长度增加 → 有新信息
        """
        reports = state.progress_reports
        n = self._max_reports_before_force_stop
        if len(reports) < n:
            return False
        recent = reports[-n:]
        
        # 检查1：completed_count是否增加
        completed_counts = [r.get("completed_count", 0) for r in recent]
        if completed_counts[-1] > completed_counts[0]:
            return False  # 有进展
        
        # 检查2：total_nodes是否增加（新节点创建）
        total_counts = [r.get("total_nodes", 0) for r in recent]
        if total_counts[-1] > total_counts[0]:
            return False  # 有新节点
        
        # 检查3：in_progress_count是否变化（节点状态流转）
        wip_counts = [r.get("in_progress_count", 0) for r in recent]
        if len(set(wip_counts)) > 1:
            return False  # 有状态变化
        
        # 检查4：消息长度是否增加（有新信息注入）
        if hasattr(state, 'last_report_msg_count'):
            if len(state.messages) > state.last_report_msg_count:
                state.last_report_msg_count = len(state.messages)
                return False  # 有新消息
        else:
            state.last_report_msg_count = len(state.messages)
        
        # 所有指标都无变化 → 停滞
        return True

    def _call_model(self, state: IDEFCState) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """LLM API 调用。返回 (tool_calls, content)。都为 None 表示超时。"""
        call_start = time.perf_counter()
        fc = state.fc_turn
        msgs = self._attn_window.apply_window() if self._attn_window else state.messages
        extra_kw = self.engine._get_llm_extra_kwargs()
        # Qwen3 系列默认开启思维链（<think>），FC 模式下禁用以避免空 content
        eb = extra_kw.get("extra_body", {})
        eb["enable_thinking"] = False
        extra_kw["extra_body"] = eb
        kw: Dict[str, Any] = {
            "model": state.vllm_model_id, "messages": msgs,
            "max_tokens": state.response_max_tokens, "temperature": 0.3,
            "top_p": 0.85, "stream": True, **extra_kw,  # 启用流式模式
        }
        if state.cb_force_no_tools:
            # CB RED: 保留记忆恢复和最终提交工具，移除其余工具
            # 防止 CB 模式下模型持续调用保留工具导致死循环：
            # 连续调用保留工具超过 3 次后，完全移除所有工具强制纯文本回复
            if state.cb_tool_streak >= 3:
                # 已连续 3 次在 CB 模式下调用工具，强制纯文本
                logger.warning(
                    f"[IDEFCRunner][CB] cb_tool_streak={state.cb_tool_streak}，"
                    f"移除所有工具强制纯文本回复"
                )
            else:
                cb_retained = self._get_cb_retained_tools(state.tool_definitions)
                if cb_retained:
                    kw["tools"] = cb_retained
                    kw["tool_choice"] = "auto"
                    logger.info(f"[IDEFCRunner][CB] 保留 {len(cb_retained)} 个收敛工具 (streak={state.cb_tool_streak})")
                else:
                    logger.info("[IDEFCRunner][CB] 强制文本，移除工具")
        elif state.pressure_force_attention:
            # 压力 RED: 工具列表仅保留注意力工具，强制 LLM 调用
            attn_tools = self._get_attention_only_tools(state.tool_definitions)
            if attn_tools:
                kw["tools"] = attn_tools
                kw["tool_choice"] = "required"
                logger.info(f"[IDEFCRunner][Pressure] 工具列表约束为注意力工具 ({len(attn_tools)}个)")
            else:
                logger.warning("[IDEFCRunner][Pressure] 注意力工具不在 tool_definitions 中，回退正常模式")
                state.pressure_force_attention = False
                if state.tool_definitions:
                    kw["tools"] = state.tool_definitions
                    kw["tool_choice"] = "auto"
        elif state.tool_definitions:
            kw["tools"] = state.tool_definitions
            # ── Layer 3: 继续已有任务图时首轮强制 task_view_overview ──
            if state.force_first_tool and fc == 1:
                kw["tool_choice"] = {"type": "function", "function": {"name": "task_view_overview"}}
                state.force_first_tool = False  # 只强制一次
                logger.info("[IDEFCRunner] 继续任务图策略首轮强制 task_view_overview")
            else:
                kw["tool_choice"] = "auto"
        future = self._model_executor.submit(
            lambda: self.engine.vllm_client.chat.completions.create(**kw))
        try:
            # 流式响应处理
            stream_response = future.result(timeout=self._fc_loop_timeout)
            logger.info(
                f"[IDEFCRunner] LLM流连接建立: turn={state.fc_turn}, "
                f"{(time.perf_counter() - call_start) * 1000:.1f}ms"
            )
            
            # 累积 token 并实时推送清洗后的文本
            full_content = ""
            sentence_buffer = ""
            sent_count = 0
            tool_calls_chunks = []  # 累积工具调用片段
            
            stream_start_time = time.time()
            stream_start_perf = time.perf_counter()
            last_heartbeat = stream_start_time
            heartbeat_interval = 10.0  # 每10秒输出一次心跳日志
            first_token_logged = False
            last_flush_perf = stream_start_perf
            first_phase_flush_chars = 24
            first_phase_flush_seconds = 1.5
            first_phase_flush_interval = 0.12
            
            for chunk in stream_response:
                if chunk.choices:
                    choice = chunk.choices[0]
                    # 累积文本内容
                    if choice.delta.content:
                        token = choice.delta.content
                        now_perf = time.perf_counter()
                        if not first_token_logged:
                            first_token_logged = True
                            logger.info(
                                f"[IDEFCRunner] LLM首token: turn={state.fc_turn}, "
                                f"{(now_perf - call_start) * 1000:.1f}ms"
                            )
                        full_content += token
                        sentence_buffer += token
                        
                        in_first_phase = (
                            sent_count == 0
                            and now_perf - stream_start_perf <= first_phase_flush_seconds
                        )
                        should_flush_fast = (
                            in_first_phase
                            and len(sentence_buffer) >= first_phase_flush_chars
                            and now_perf - last_flush_perf >= first_phase_flush_interval
                        )
                        # 检测句子边界（句号、问号、感叹号、换行）
                        if should_flush_fast or any(token.endswith(p) for p in ['。', '！', '？', '\n']):
                            # 清洗并推送当前句子
                            from zulong.utils.text_cleaner import clean_text_for_tts
                            cleaned = clean_text_for_tts(sentence_buffer)
                            if cleaned:
                                # 异步推送（使用安全发送方法）
                                cb = getattr(self, '_ide_send_callback', None)
                                if cb:
                                    self._send_message_safe(cb, "display_text", {
                                        "text": cleaned,
                                        "turn": state.fc_turn,
                                        "streaming": True,
                                        "sentence_index": sent_count
                                    })
                                sent_count += 1
                                sentence_buffer = ""  # 清空缓冲区
                                last_flush_perf = now_perf
                    
                    # 累积工具调用片段
                    if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                        tool_calls_chunks.extend(choice.delta.tool_calls)
                
                # 心跳日志：防止长时间无输出时看起来像卡死
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    elapsed = int(now - stream_start_time)
                    logger.info(f"💓 [FC] 等待模型流式响应中... 已等待 {elapsed}s, 已接收 {len(full_content)} 字符")
                    last_heartbeat = now
            
            # 处理剩余的文本
            if sentence_buffer.strip():
                from zulong.utils.text_cleaner import clean_text_for_tts
                cleaned = clean_text_for_tts(sentence_buffer)
                if cleaned:
                    cb = getattr(self, '_ide_send_callback', None)
                    if cb:
                        self._send_message_safe(cb, "display_text", {
                            "text": cleaned,
                            "turn": state.fc_turn,
                            "streaming": True,
                            "sentence_index": sent_count
                        })
                    sent_count += 1
            
            # 推送完成标记
            cb = getattr(self, '_ide_send_callback', None)
            if cb:
                self._send_message_safe(cb, "display_text", {
                    "text": "",
                    "turn": state.fc_turn,
                    "streaming": False,
                    "complete": True
                })
            
            if sent_count > 0:
                logger.info(f"🌊 [FC] 流式推送完成：共 {sent_count} 个句子，总长度 {len(full_content)}")
            
            # 组装流式工具调用片段
            tc = None
            if tool_calls_chunks:
                # 按 index 分组并累积
                tc_map = {}  # {index: {id, type, function: {name, arguments}}}
                for tc_chunk in tool_calls_chunks:
                    idx = getattr(tc_chunk, 'index', 0)
                    if idx not in tc_map:
                        tc_map[idx] = {
                            'id': '',
                            'type': 'function',
                            'function': {'name': '', 'arguments': ''}
                        }
                    if hasattr(tc_chunk, 'id') and tc_chunk.id:
                        tc_map[idx]['id'] = tc_chunk.id
                    if hasattr(tc_chunk, 'function'):
                        if hasattr(tc_chunk.function, 'name') and tc_chunk.function.name:
                            tc_map[idx]['function']['name'] = tc_chunk.function.name
                        if hasattr(tc_chunk.function, 'arguments') and tc_chunk.function.arguments:
                            tc_map[idx]['function']['arguments'] += tc_chunk.function.arguments
                
                tc = list(tc_map.values())
                if tc:
                    logger.info(f"[IDEFCRunner] Turn {fc}: {len(tc)} 工具调用 (流式)")
                    for t in tc:
                        logger.info(f"[IDEFCRunner]   FC tool: {t['function']['name']} args={_safe_truncate(t['function']['arguments'])}")
            
            # 返回完整内容供后续处理
            # 注意：task_complete在run_loop_async中统一发送，避免重复
            rc = full_content
            
        except concurrent.futures.TimeoutError:
            future.cancel()
            state.api_timeout_count += 1
            logger.warning(f"[IDEFCRunner] Turn {fc} 超时, count={state.api_timeout_count}")
            return None, None
        except Exception as err:
            logger.error(f"[IDEFCRunner] Turn {fc} API 失败: {err}")
            # 检测 429 Rate Limit 错误：等待更长时间后重试，而非立即放弃
            err_str = str(err)
            is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "TPM" in err_str
            if is_rate_limit:
                # 429 专用处理：等待 20 秒让 TPM 窗口恢复，然后重试一次
                wait_secs = 20
                logger.warning(
                    f"[IDEFCRunner] Turn {fc} 触发 429 限流，等待 {wait_secs}s 后重试...")
                time.sleep(wait_secs)
                try:
                    # 重试也使用流式
                    retry_stream = self.engine.vllm_client.chat.completions.create(**kw)
                    full_content = ""
                    sentence_buffer = ""
                    sent_count = 0
                    retry_tool_calls = []  # 累积工具调用片段
                    
                    for chunk in retry_stream:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            if choice.delta.content:
                                token = choice.delta.content
                                full_content += token
                                sentence_buffer += token
                                
                                if any(token.endswith(p) for p in ['。', '！', '？', '\n']):
                                    from zulong.utils.text_cleaner import clean_text_for_tts
                                    cleaned = clean_text_for_tts(sentence_buffer)
                                    if cleaned:
                                        cb = getattr(self, '_ide_send_callback', None)
                                        if cb:
                                            self._send_message_safe(cb, "display_text", {
                                                "text": cleaned,
                                                "turn": state.fc_turn,
                                                "streaming": True,
                                                "sentence_index": sent_count
                                            })
                                        sent_count += 1
                                        sentence_buffer = ""
                            
                            # 累积工具调用片段
                            if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                                retry_tool_calls.extend(choice.delta.tool_calls)
                    
                    if sentence_buffer.strip():
                        from zulong.utils.text_cleaner import clean_text_for_tts
                        cleaned = clean_text_for_tts(sentence_buffer)
                        if cleaned:
                            cb = getattr(self, '_ide_send_callback', None)
                            if cb:
                                self._send_message_safe(cb, "display_text", {
                                    "text": cleaned,
                                    "turn": state.fc_turn,
                                    "streaming": True,
                                    "sentence_index": sent_count
                                })
                            sent_count += 1
                    
                    state.api_timeout_count = 0
                    rc = full_content
                    
                    # 组装流式工具调用片段
                    tc = None
                    if retry_tool_calls:
                        tc_map = {}
                        for tc_chunk in retry_tool_calls:
                            idx = getattr(tc_chunk, 'index', 0)
                            if idx not in tc_map:
                                tc_map[idx] = {
                                    'id': '',
                                    'type': 'function',
                                    'function': {'name': '', 'arguments': ''}
                                }
                            if hasattr(tc_chunk, 'id') and tc_chunk.id:
                                tc_map[idx]['id'] = tc_chunk.id
                            if hasattr(tc_chunk, 'function'):
                                if hasattr(tc_chunk.function, 'name') and tc_chunk.function.name:
                                    tc_map[idx]['function']['name'] = tc_chunk.function.name
                                if hasattr(tc_chunk.function, 'arguments') and tc_chunk.function.arguments:
                                    tc_map[idx]['function']['arguments'] += tc_chunk.function.arguments
                        
                        tc = list(tc_map.values())
                        if tc:
                            logger.info(f"[IDEFCRunner] Turn {fc}: {len(tc)} 工具调用 (429重试成功)")
                            for t in tc:
                                logger.info(f"[IDEFCRunner]   FC tool: {t['function']['name']} args={_safe_truncate(t['function']['arguments'])}")
                        else:
                            logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)} (429重试成功)")
                    else:
                        logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)} (429重试成功)")
                    return tc, rc
                except Exception as retry_err:
                    logger.warning(f"[IDEFCRunner] Turn {fc} 429重试仍失败: {retry_err}")
                    # 继续走原有的备用模型逻辑
            # 追踪连续 API 错误（含 429 rate limit）
            state.api_timeout_count += 1
            if state.api_timeout_count >= 3:
                logger.error(
                    f"[IDEFCRunner] 连续 {state.api_timeout_count} 次 API 错误，触发退出")
                return None, None
            try:
                from zulong.models.container import LLM_MODEL_ID_BACKUP
                if self.engine.backup_client and LLM_MODEL_ID_BACKUP:
                    br = self.engine.backup_client.chat.completions.create(
                        model=LLM_MODEL_ID_BACKUP, messages=state.messages,
                        max_tokens=state.response_max_tokens, temperature=0.3,
                        stream=False, **self.engine._get_llm_extra_kwargs())
                    c = br.choices[0].message.content or ""
                    state.last_response_content = c
                    return None, c
            except Exception as be:
                logger.warning(f"[IDEFCRunner] 备用也失败: {be}")
            # 主+备均失败：注入 API 错误提示消息，让循环上层处理
            state.api_timeout_count += 1
            logger.error(
                f"[IDEFCRunner] 主+备均失败，连续 {state.api_timeout_count} 次错误，触发退出")
            return None, None
        # 流式响应已在上面处理，rc 和 tc 已经设置
        # 这里只需要处理 XML 工具调用回退
        
        # 检查是否包含 XML 格式的工具调用
        xml_tc = self.translator.parse_xml_tool_calls(rc)
        if xml_tc:
            logger.info(f"[IDEFCRunner] Turn {fc}: {len(xml_tc)} 工具调用 (XML 回退解析)")
            for xt in xml_tc:
                logger.info(f"[IDEFCRunner]   XML tool: {xt['function']['name']} args={_safe_truncate(xt['function']['arguments'])}")
            tc = xml_tc
            # 从内容中移除 XML 工具标签，保留前置文本
            rc = self._strip_xml_tool_tags(rc)
        else:
            # 即使未解析出工具调用，文本中仍可能含有 XML 残留片段
            # （LLM 输出了不完整/非标准的 XML 工具调用）
            rc = self._strip_xml_tool_tags(rc)
            if rc:
                logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)}")
                # 调试：记录文本前 300 字符
                snippet = rc[:300].replace('\n', '\\n')
                logger.info(f"[IDEFCRunner] Turn {fc} 文本预览: {snippet}")
        
        return tc, rc

    @staticmethod
    @staticmethod
    def _extract_final_answer(state: IDEFCState) -> Optional[str]:
        """从消息历史中提取 submit_final_answer 的 answer 内容。

        如果 FC 循环中调用了 submit_final_answer，从最近的 assistant
        tool_call 消息中提取 answer 参数作为最终回答内容。
        返回 None 表示未调用 submit_final_answer。
        """
        for msg in reversed(state.messages):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {}).get("name", "")
                if fn == "submit_final_answer":
                    import json as _json
                    try:
                        args_str = tc.get("function", {}).get("arguments", "{}")
                        args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                        answer = args.get("answer", "")
                        if answer:
                            return answer
                    except Exception:
                        pass
        return None

    @staticmethod
    def _get_cb_retained_tools(tool_definitions: List[Dict]) -> List[Dict]:
        """CB RED 时保留的工具子集

        保留记忆恢复和最终提交类工具，使模型在被限制时仍可：
        1. 通过 recall_memory 恢复被淘汰的上下文
        2. 通过 submit_final_answer 生成最终回复
        3. 通过 task_mark_status 标记当前进度
        """
        _CB_RETAINED_NAMES = {
            "recall_memory", "read_memory_node",
            "submit_final_answer",
            "task_mark_status", "task_view_overview",
        }
        retained = []
        for td in tool_definitions:
            fn = td.get("function", {}).get("name", "")
            if fn in _CB_RETAINED_NAMES:
                retained.append(td)
        return retained

    @staticmethod
    def _strip_xml_tool_tags(text: str) -> str:
        """从文本中移除 XML 工具调用标签，保留前置文本"""
        import re as _re
        # 移除 <thinking> 块
        text = _re.sub(r"<thinking>.*?</thinking>", "", text, flags=_re.DOTALL)
        # 移除有闭合标签的远程工具 XML
        for tool_name in IDE_REMOTE_TOOLS:
            text = _re.sub(
                rf"<{_re.escape(tool_name)}>.*?</{_re.escape(tool_name)}>",
                "", text, flags=_re.DOTALL)
        # 移除无闭合标签的远程工具 XML (<tool_name>... 到下一个工具标签或文本末尾)
        _tool_open_re = "|".join(_re.escape(t) for t in IDE_REMOTE_TOOLS)
        for tool_name in IDE_REMOTE_TOOLS:
            text = _re.sub(
                rf"<{_re.escape(tool_name)}>.*?(?=<(?:{_tool_open_re})>|\Z)",
                "", text, flags=_re.DOTALL)
        # 同时移除内部工具的 XML（LLM 可能也用 XML 调用内部工具）
        _internal_xml_tools = {
            "task_create_plan", "task_add_node", "task_mark_status",
            "task_view_overview", "recall_memory", "save_memory_note",
            "navigate_attention", "search_experience", "search_tools",
            "index_project", "index_code_file", "search_code_symbols",
            "get_symbol_context", "get_impact_analysis", "analyze_module",
            "zulong_code_query", "zulong_task_link_code",
        }
        for tool_name in _internal_xml_tools:
            text = _re.sub(
                rf"<{_re.escape(tool_name)}>.*?</{_re.escape(tool_name)}>",
                "", text, flags=_re.DOTALL)
        # 清理通用 XML 包装标签（LLM 可能用 <tool_call>/<function> 等包裹工具调用）
        _generic_wrappers = [
            "tool_call", "function_call", "tool_use",
            "function", "invoke", "tool",
        ]
        for tag in _generic_wrappers:
            text = _re.sub(
                rf"<{tag}(?:\s[^>]*)?>.*?</{tag}>",
                "", text, flags=_re.DOTALL)
        # 清理残留的孤立闭合标签（如 </parameter> </function> </tool_call>）
        text = _re.sub(
            r"</(?:parameter|function|tool_call|function_call|tool_use|"
            r"invoke|tool|name|arguments|thinking)>",
            "", text)
        # 清理残留的孤立开放标签
        text = _re.sub(
            r"<(?:parameter|function|tool_call|function_call|tool_use|"
            r"invoke|tool|name|arguments|thinking)(?:\s[^>]*)?>",
            "", text)
        # 清理 <parameter=name>value 残留
        text = _re.sub(r"<parameter=\w+>", "", text)
        return text.strip()

    def _exec_tools(self, state: IDEFCState, tool_calls_data: List[Dict],
                    response_content: str = "") -> Optional[List[Dict]]:
        """执行工具 + 分流。混合批次拆分为两个 assistant 消息避免 orphaned tool_call IDs。"""
        fc = state.fc_turn
        msgs = state.messages
        internal, remote = [], []
        for td in tool_calls_data:
            (remote if self.tool_registry.classify(td["function"]["name"]) == "remote" else internal).append(td)
        grp = self._attn_window.new_tool_group() if self._attn_window else None

        if internal:
            a_msg = {"role": "assistant", "content": response_content or "", "tool_calls": internal}
            msgs.append(a_msg)
            if self._attn_window:
                self._attn_window.register_message(a_msg, turn=fc, group_id=grp)
            for td in internal:
                self._exec_internal(state, td, fc, grp)
            try:
                if self._circuit_breaker:
                    _aw_ratio = self._attn_window.usage_ratio if self._attn_window else -1.0
                    cb_s, cb_r = self._circuit_breaker.evaluate(fc, msgs, attn_usage_ratio=_aw_ratio)
                    if cb_s == CircuitBreakerState.RED:
                        logger.warning(f"[IDEFCRunner][CB] RED: {cb_r}")
                        state.cb_force_no_tools = True
                        cm = {"role": "user", "content":
                              f"[Circuit Breaker 强制收敛] {cb_r}\n你必须立刻基于已有信息生成最终回复，不允许再调用任何工具。"}
                        msgs.append(cm)
                        if self._attn_window:
                            self._attn_window.register_message(cm, turn=fc)
                            try:
                                self._attn_window.on_navigate_attention(direction="single_chain")
                            except Exception:
                                pass
                        remote = []
                    elif cb_s == CircuitBreakerState.YELLOW:
                        logger.warning(f"[IDEFCRunner][CB] YELLOW: {cb_r}")
                        ch = {"role": "user", "content":
                              f"[Circuit Breaker 警告] {cb_r}\n请尽快总结当前信息并回复用户，避免继续调用更多工具。"}
                        msgs.append(ch)
                        if self._attn_window:
                            self._attn_window.register_message(ch, turn=fc)
                            try:
                                self._attn_window.on_navigate_attention(direction="focus")
                            except Exception:
                                pass
            except Exception as cb_err:
                logger.warning(f"[IDEFCRunner] CircuitBreaker evaluate 异常: {cb_err}")

            # 上下文压力感知（在 CB 评估之后）
            self._apply_pressure_guidance(state, fc)

        if remote:
            valid_remote, rejected = self._validate_and_clean_remote_calls(remote)
            all_calls = valid_remote + [r[0] for r in rejected]
            ra = {"role": "assistant", "content": "" if internal else (response_content or ""),
                  "tool_calls": all_calls}
            msgs.append(ra)
            if self._attn_window:
                self._attn_window.register_message(ra, turn=fc, group_id=grp)
            # 为被拒绝的调用注入错误结果，让 LLM 重试
            for rej_tc, err_msg in rejected:
                err_result = {"role": "tool", "tool_call_id": rej_tc["id"],
                              "content": f"[参数验证失败] {err_msg}"}
                msgs.append(err_result)
                if self._attn_window:
                    self._attn_window.register_message(
                        err_result, turn=fc, tool_name=rej_tc["function"]["name"])
            if valid_remote:
                self._maybe_run_bfs(fc, "tool_complete")
                return valid_remote

        self._maybe_run_bfs(fc, "tool_complete")
        return None

    def _exec_internal(self, state: IDEFCState, td: Dict, fc: int, grp: Optional[int]) -> None:
        tn = td["function"]["name"]
        ta = {}
        try:
            ta = _json.loads(td["function"]["arguments"] or "{}")
        except Exception:
            pass
        if self._attn_window:
            self._attn_window.observe_tool_call(tn, ta)
            if tn == "navigate_attention":
                self._attn_window.on_navigate_attention(
                    direction=ta.get("direction", ""), target_node_id=ta.get("target_node_id"))
        try:
            # TaskGraph CRUD 工具拦截
            from zulong.ide.graph_crud_tools import CRUD_TOOL_NAMES, dispatch_crud_tool
            if tn in CRUD_TOOL_NAMES:
                task_graph = self._get_current_task_graph()
                ws_sender = getattr(self, '_ws_send_callback', None)
                crud_result = dispatch_crud_tool(
                    tool_name=tn, arguments=ta, task_graph=task_graph,
                    session_id=self.session.session_id, ws_sender=ws_sender,
                )
                rt = _json.dumps(crud_result, ensure_ascii=False, default=str)
            else:
                rt = self.engine._execute_tool_call(_ToolCallProxy(td))
        except Exception as tool_err:
            logger.error(f"[IDEFCRunner] 内部工具 {tn} 执行异常: {tool_err}")
            rt = f"[工具执行异常] {tool_err}"
        # task_mark_status 完成后自动导航注意力窗口
        if self._attn_window and tn == "task_mark_status":
            new_status = ta.get("new_status") or ta.get("status") or ""
            mark_node = ta.get("node_id") or ""
            if new_status and mark_node:
                self._attn_window.auto_navigate_on_status_change(mark_node, new_status)
        # 🔥 任务完成验证：代码生成节点标记完成时，检查是否真正写入了文件
        if tn == "task_mark_status" and (ta.get("new_status") or ta.get("status")) == "completed":
            mark_node_id = ta.get("node_id")
            if mark_node_id:
                try:
                    task_graph = self._get_current_task_graph()
                    if task_graph:
                        node = task_graph.get_node(mark_node_id)
                        if node:
                            label_lower = (node.label or "").lower()
                            desc_lower = (node.desc or "").lower()
                            # 判断是否为代码生成类节点
                            _code_gen_kw = ("写", "编写", "创建", "生成", "代码", "code", "write",
                                            "create", "文件", "file", "html", "css", "js", "实现",
                                            "开发", "构建", "build", "页面", "page", "组件", "component")
                            is_code_gen = any(kw in label_lower or kw in desc_lower
                                              for kw in _code_gen_kw)
                            if is_code_gen:
                                # 检查最近的工具调用中是否有 write_to_file/replace_in_file
                                recent_tools = [r.get("tool_name", "") for r in state.tool_results_buffer[-10:]]
                                has_write = any(t in ("write_to_file", "replace_in_file")
                                                for t in recent_tools)
                                if not has_write:
                                    warn = (
                                        f"\n\n⚠️ [任务完成验证] 该节点({node.label})属于代码生成任务，"
                                        f"但未检测到 write_to_file/replace_in_file 工具调用。"
                                        f"请确认文件已正确创建。如需创建文件，请调用: "
                                        f"write_to_file(path='文件路径', content='文件内容')"
                                    )
                                    rt = rt + warn
                except Exception as val_err:
                    logger.debug(f"[IDEFCRunner] 任务完成验证异常: {val_err}")
        # 注意力工具执行完毕后，恢复正常工具列表
        if tn in ("navigate_attention", "adjust_attention_mode"):
            if state.pressure_force_attention:
                state.pressure_force_attention = False
                logger.info(f"[IDEFCRunner][Pressure] 注意力工具 {tn} 已执行，恢复正常工具列表")
        if self._circuit_breaker:
            self._circuit_breaker.record_call(tn, ta, rt)
            if tn in ("task_create_plan", "start_task_plan", "task_add_node"):
                self._circuit_breaker.escalate_for_planning()
        if len(rt) > MAX_TOOL_RESULT_CHARS:
            ol = len(rt)
            rt = rt[:MAX_TOOL_RESULT_CHARS] + f"\n...(已截断，原始长度 {ol} 字符)"
        tm = {"role": "tool", "tool_call_id": td["id"], "content": rt}
        state.messages.append(tm)
        if self._attn_window:
            self._attn_window.register_message(
                tm, turn=fc, tool_name=tn,
                node_id=ta.get("node_id") or ta.get("target_node_id"), group_id=grp)
        if len(state.tool_results_buffer) >= _TOOL_RESULTS_BUFFER_MAX:
            state.tool_results_buffer.pop(0)
        state.tool_results_buffer.append({"tool_name": tn, "result": rt})
        self.engine._publish_task_graph_event("agent_tool_call", fc, tn, rt)
        # 记录子对话到 MemoryGraph
        self._record_sub_dialogue(state, tool_name=tn, result=rt)
        self._emit_execution_event_sync(
            "tool_finished",
            f"内部工具执行完成: {tn}",
            turn=fc,
            event_type="IDE_TOOL_EXEC",
            payload={
                "tool_name": tn,
                "arguments_preview": _json.dumps(ta, ensure_ascii=False)[:300],
                "result_preview": (rt or "")[:500],
            },
        )

    # 参数默认值：LLM 省略某些参数时自动填充
    # 典型场景：list_files() 不传 path; execute_command() 不传 requires_approval
    _PARAM_DEFAULTS: Dict[str, Dict[str, str]] = {
        "list_files": {"path": "."},
        "search_files": {"path": "."},
        "list_code_definition_names": {"path": "."},
        "execute_command": {"requires_approval": "false"},
    }

    # LLM 常见参数名别名 → schema 标准名映射
    # LLM 有时使用 file_path / filepath 等替代 path，导致参数验证失败
    _PARAM_ALIASES: Dict[str, Dict[str, str]] = {
        "read_file": {
            "file_path": "path", "filepath": "path", "file": "path",
            "filename": "path", "file_name": "path",
        },
        "write_to_file": {
            "file_path": "path", "filepath": "path", "file": "path",
            "filename": "path", "file_name": "path",
            "file_content": "content", "text": "content", "data": "content",
        },
        "replace_in_file": {
            "file_path": "path", "filepath": "path",
            "changes": "diff", "replacement": "diff", "replacements": "diff",
        },
        "execute_command": {
            "cmd": "command", "shell_command": "command", "shell": "command",
        },
        "search_files": {
            "directory": "path", "dir": "path", "folder": "path",
            "pattern": "regex", "search_pattern": "regex", "query": "regex",
        },
        "list_files": {
            "directory": "path", "dir": "path", "folder": "path",
        },
        "list_code_definition_names": {
            "file_path": "path", "filepath": "path",
            "directory": "path", "dir": "path",
        },
    }

    def _validate_and_clean_remote_calls(
        self, remote_calls: List[Dict]
    ) -> Tuple[List[Dict], List[Tuple[Dict, str]]]:
        """验证远程工具必需参数 & 清理非 schema 参数（如 task_progress）。

        修复: 增加参数名别名自动映射、非 dict 类型防御、更清晰的错误提示。

        Returns:
            (valid_calls, rejected): rejected 是 [(tool_call_dict, error_msg), ...]
        """
        from zulong.ide.ide_tool_registry import _IDE_TOOL_SCHEMAS
        schema_map: Dict[str, Dict] = {}
        for s in _IDE_TOOL_SCHEMAS:
            f = s.get("function", {})
            p = f.get("parameters", {})
            schema_map[f.get("name", "")] = {
                "required": p.get("required", []),
                "properties": set(p.get("properties", {}).keys()),
            }

        valid: List[Dict] = []
        rejected: List[Tuple[Dict, str]] = []
        for tc in remote_calls:
            fn = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            try:
                args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
            except Exception:
                args = {}

            # 类型防御：确保 args 是 dict（LLM 可能生成 "null"、纯字符串等非对象 JSON）
            if not isinstance(args, dict):
                logger.warning(
                    f"[IDEFCRunner] 工具 {fn} 参数不是 dict: {type(args).__name__}={str(args)[:100]}")
                args = {}

            info = schema_map.get(fn)
            if not info:
                tc["function"]["arguments"] = _json.dumps(args, ensure_ascii=False)
                valid.append(tc)
                continue

            # ── 参数名别名自动映射（file_path→path 等） ──
            aliases = self._PARAM_ALIASES.get(fn, {})
            remapped = []
            for k in list(args.keys()):
                canonical = aliases.get(k)
                if canonical and canonical not in args:
                    args[canonical] = args.pop(k)
                    remapped.append(f"{k}→{canonical}")
            if remapped:
                logger.info(
                    f"[IDEFCRunner] 参数名自动映射 {fn}: {', '.join(remapped)}")

            # ── 参数默认值填充（LLM 省略有合理默认值的必需参数时自动补全） ──
            defaults = self._PARAM_DEFAULTS.get(fn, {})
            defaulted = []
            for param, default_val in defaults.items():
                if param not in args or not args.get(param):
                    args[param] = default_val
                    defaulted.append(f"{param}='{default_val}'")
            if defaulted:
                logger.info(
                    f"[IDEFCRunner] 参数默认值填充 {fn}: {', '.join(defaulted)}")

            # ── 路径矫正：list_code_definition_names 的 path 必须是目录 ──
            # IDE 扩展将 path 用作子进程 cwd，传入文件路径会报错
            # "The cwd option must be a path to a directory"
            if fn == "list_code_definition_names" and args.get("path"):
                import posixpath as _ppath
                p = args["path"]
                # 检测文件路径特征：包含常见文件扩展名
                if "." in _ppath.basename(p.replace("\\", "/")):
                    parent = _ppath.dirname(p.replace("\\", "/")) or "."
                    logger.info(
                        f"[IDEFCRunner] 路径矫正 {fn}: '{p}' → '{parent}' (文件→目录)")
                    args["path"] = parent

            # 清理非 schema 参数（如 task_progress）
            non_schema = [k for k in list(args.keys()) if k not in info["properties"]]
            for k in non_schema:
                logger.info(f"[IDEFCRunner] 清理非 schema 参数: {fn}.{k}={str(args[k])[:80]}")
                del args[k]

            # 检查必需参数
            missing = [p for p in info["required"] if p not in args or not args.get(p)]
            if missing:
                err = (
                    f"工具 {fn} 缺少必需参数: {missing}。"
                    f"该工具的正确参数为: {sorted(info['properties'])}，"
                    f"其中必需: {info['required']}。请使用正确的参数名重新调用。"
                )
                logger.warning(f"[IDEFCRunner] {err}")
                tc["function"]["arguments"] = _json.dumps(args, ensure_ascii=False)
                rejected.append((tc, err))
            else:
                tc["function"]["arguments"] = _json.dumps(args, ensure_ascii=False)
                valid.append(tc)

        return valid, rejected

    def _eval_response(self, state: IDEFCState, response_content: str) -> str:
        fc = state.fc_turn
        msgs = state.messages
        resp = response_content
        is_resume = state.is_resume

        logger.info(
            f"[IDEFCRunner][EvalChain] turn={fc} resp_len={len(resp.strip()) if resp else 0} "
            f"cb_force={state.cb_force_no_tools} null_count={state.null_response_count}"
        )

        if state.cb_force_no_tools:
            if not resp or len(resp.strip()) < 10:
                resp = self._get_cb_fallback(state)
            self._run_backfill(state, resp, is_cb_path=True)
            state.last_response_content = resp
            state.cb_force_no_tools = False
            state.cb_tool_streak = 0  # 重置 CB 工具连续计数
            # 修复 4: CB 路径检查未完成节点 — 防止大量任务未完成时无条件终止
            # 注意：排除 CRG 自动注入节点（crg_ 前缀），它们不代表用户任务进度
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_cb
                tg_cb = _gtg_cb()
                if tg_cb and state.null_response_count < 6:
                    leaves_cb = tg_cb.get_leaf_nodes()
                    # 只计算 LLM 规划的节点（排除 CRG 自动注入节点）
                    user_leaves = [n for n in leaves_cb if not n.id.startswith("crg_")]
                    unc_cb = [n for n in user_leaves
                              if n.status not in ("completed", "skipped")]
                    if len(unc_cb) > 1 and user_leaves and len(unc_cb) > len(user_leaves) * 0.5:
                        logger.info(
                            f"[IDEFCRunner][CB] 仍有 {len(unc_cb)}/{len(user_leaves)} "
                            f"用户任务未完成，恢复工具调用继续执行"
                        )
                        logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=cb_continue")
                        return "continue"
            except Exception:
                pass
            logger.info(f"[IDEFCRunner][CB] 强制回复, len={len(resp)}")
            logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=cb_force_done")
            return "done"

        # 安全网 0: 语义漂移检测
        # 检查模型回复是否偏离了原始任务目标
        if self._drift_detector and resp and len(resp) > 50:
            try:
                import asyncio as _aio
                try:
                    loop = _aio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    # 已有事件循环（如 FastAPI 上下文），使用线程池
                    import concurrent.futures as _cf
                    with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                        drift_result = _pool.submit(
                            lambda: _aio.run(
                                self._drift_detector.detect_drift(resp)
                            )
                        ).result(timeout=5)
                else:
                    drift_result = _aio.run(
                        self._drift_detector.detect_drift(resp))
                is_drifted, similarity, reason = drift_result
                logger.info(
                    f"[IDEFCRunner][DriftDetector] turn={fc} "
                    f"drift={is_drifted}, sim={similarity:.3f}, {reason}")
                if is_drifted:
                    # 显著漂移 → 注入纠偏提示让模型重新聚焦
                    # 注意：必须用 "user" role，SiliconFlow 等 API 要求 system 消息在最前面
                    drift_hint = {
                        "role": "user",
                        "content": (
                            f"[语义漂移检测] {reason}\n"
                            f"原始任务: {state.user_input_text[:200]}\n"
                            f"你的回复偏离了任务目标，请重新聚焦原始任务，"
                            f"调用 task_view_overview 查看当前进度后继续执行。"
                        ),
                    }
                    msgs.append({"role": "assistant", "content": resp})
                    msgs.append(drift_hint)
                    if self._attn_window:
                        self._attn_window.register_message(
                            {"role": "assistant", "content": resp}, turn=fc)
                        self._attn_window.register_message(drift_hint, turn=fc)
                    state.null_response_count += 1
                    return "cb_force" if state.null_response_count >= 4 else "continue"
                # 非漂移：异步记录对话历史供后续检测使用
                try:
                    if loop and loop.is_running():
                        with _cf.ThreadPoolExecutor(max_workers=1) as _pool2:
                            _pool2.submit(
                                lambda: _aio.run(
                                    self._drift_detector.add_conversation_turn(
                                        state.user_input_text or "", resp)
                                )
                            ).result(timeout=5)
                    else:
                        _aio.run(
                            self._drift_detector.add_conversation_turn(
                                state.user_input_text or "", resp))
                except Exception:
                    pass
            except Exception as drift_err:
                logger.warning(f"[IDEFCRunner][DriftDetector] 检测异常: {drift_err}")

        # 安全网 0.5: 独白检测 (OpenHands-style monologue detection)
        # 检测模型连续纯文本回复而不调用工具的退化模式
        # 当连续3次以上纯文本 + 有未完成任务时，注入强纠正提示
        if (state.consecutive_text_only_count >= 3
                and resp and len(resp.strip()) >= 6):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_mono
                tg_mono = _gtg_mono()
                if tg_mono:
                    leaves_mono = tg_mono.get_leaf_nodes()
                    user_leaves_mono = [n for n in leaves_mono if not n.id.startswith("crg_")]
                    uncompleted_mono = [n for n in user_leaves_mono
                                        if n.status not in ("completed", "skipped")]
                    if not user_leaves_mono:
                        req_node = tg_mono.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uncompleted_mono = [req_node]
                    if uncompleted_mono:
                        current_mono = next(
                            (n for n in uncompleted_mono if n.status == "in_progress"),
                            uncompleted_mono[0],
                        )
                        # 🔥 强纠正提示：明确告诉模型用哪个工具、什么参数
                        # 优先级：1. write_to_file（代码生成节点） 2. read_file（分析节点） 3. execute_command（命令节点）
                        label_lower = (current_mono.label or "").lower()
                        tool_hint = ""
                        if any(kw in label_lower for kw in ("写", "编写", "创建", "生成", "代码", "code", "write", "create", "文件", "file", "html", "css", "js", "实现", "开发")):
                            tool_hint = (
                                f'请立即调用 write_to_file 创建文件。\n'
                                f'示例: write_to_file(path="文件路径", content="文件内容")'
                            )
                        elif any(kw in label_lower for kw in ("分析", "分析", "检查", "review", "查看", "阅读", "读", "read")):
                            tool_hint = "请调用 read_file 或 index_code_file 分析代码。"
                        elif any(kw in label_lower for kw in ("运行", "执行", "测试", "运行", "test", "run", "命令", "command")):
                            tool_hint = "请调用 execute_command 运行命令。"
                        else:
                            tool_hint = (
                                "请调用合适的工具（write_to_file/read_file/execute_command）"
                                "来执行任务。不要只口头叙述！"
                            )
                        nudge_mono = {
                            "role": "user",
                            "content": (
                                f"[独白检测] 你已经连续 {state.consecutive_text_only_count} 次"
                                f"只输出文本而没有调用任何工具！\n"
                                f"当前应执行: {current_mono.id}({current_mono.label})。\n"
                                f"还有 {len(uncompleted_mono)} 个任务未完成。\n"
                                f"{tool_hint}\n"
                                f"⚠️ 仅在文本中说「开始编写」而不调用工具，等于什么都没做。"
                            ),
                        }
                        msgs.append({"role": "assistant", "content": resp})
                        msgs.append(nudge_mono)
                        if self._attn_window:
                            self._attn_window.register_message(
                                {"role": "assistant", "content": resp}, turn=fc)
                            self._attn_window.register_message(nudge_mono, turn=fc)
                        state.null_response_count += 1
                        mono_count = state.consecutive_text_only_count
                        state.consecutive_text_only_count = 0  # 重置，避免重复nudge
                        logger.warning(
                            f"[IDEFCRunner][Monologue] turn={fc} "
                            f"连续{mono_count}次纯文本，"
                            f"未完成={len(uncompleted_mono)}, 当前={current_mono.id}")
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][Monologue] {e}")

        # 安全网 1: RuleGuardian
        blocked = False
        if self._rule_guardian:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg
                blk, br = self._rule_guardian.check_premature_completion(resp, _gtg())
                if blk:
                    cor = {"role": "user", "content":
                           f"[规则守护] {br}\n请调用 task_view_overview 查看任务图，然后继续执行未完成的任务。"}
                    msgs.append({"role": "assistant", "content": resp})
                    msgs.append(cor)
                    if self._attn_window:
                        self._attn_window.register_message({"role": "assistant", "content": resp}, turn=fc)
                        self._attn_window.register_message(cor, turn=fc)
                    blocked = True
            except Exception as e:
                logger.warning(f"[IDEFCRunner][RuleGuardian] {e}")
        if blocked:
            state.null_response_count += 1
            return "cb_force" if state.null_response_count >= 4 else "continue"

        # 安全网 2: InfoGap
        try:
            from zulong.l2.info_gap_detector import InfoGapType
            sc = self._build_subtask_context()
            gt, gd, gc = self.engine._info_gap_detector.detect(
                llm_output=resp, tool_results=state.tool_results_buffer or None,
                subtask_context=sc)
            if gt == InfoGapType.NEED_SUBTASK_RESULT and gc >= 0.6 and state.gap_continue_count < 5:
                gh = {"role": "user", "content":
                      f"[信息缺口] 缺少前置结果: {gd}\n请先用 task_view_overview 查看任务图。"}
                msgs.append({"role": "assistant", "content": resp})
                msgs.append(gh)
                if self._attn_window:
                    self._attn_window.register_message({"role": "assistant", "content": resp}, turn=fc)
                    self._attn_window.register_message(gh, turn=fc)
                state.gap_continue_count += 1
                state.null_response_count += 1
                return "cb_force" if state.null_response_count >= 4 else "continue"
        except Exception as e:
            logger.warning(f"[IDEFCRunner][InfoGap] {e}")

        # 安全网 3: 继续任务图 AutoMark
        if (is_resume and len(resp) > 100 and state.resume_automark_count < 5
                and not resp.rstrip().endswith(("?", "\uff1f")) and not _is_filler_content(resp)):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_am, _save_active_backup
                tg = _gtg_am()
                if tg:
                    leaves = tg.get_leaf_nodes()
                    unc = [n for n in leaves if n.status != "completed"]
                    if unc:
                        tgt = next((n for n in unc if n.status == "in_progress"), unc[0])
                        tg.update_node_status(tgt.id, "completed", result=resp[:500])
                        try: _save_active_backup()
                        except Exception as e:
                            ErrorHandler.handle_exception(e, ErrorCode.CFG_SAVE_FAILED, context={"operation": "save_active_backup"}, log_level=logging.WARNING)
                        rem = [n for n in tg.get_leaf_nodes() if n.status != "completed"]
                        if rem:
                            nn = rem[0]
                            tg.update_node_status(nn.id, "in_progress")
                            try: _save_active_backup()
                            except Exception as e:
                                ErrorHandler.handle_exception(e, ErrorCode.CFG_SAVE_FAILED, context={"operation": "save_active_backup"}, log_level=logging.WARNING)
                            cont = {"role": "user", "content":
                                    f"[自动进度] {tgt.id}({tgt.label})完成。继续 {nn.id}({nn.label})。"}
                            msgs.append({"role": "assistant", "content": resp})
                            msgs.append(cont)
                            if self._attn_window:
                                self._attn_window.register_message(
                                    {"role": "assistant", "content": resp}, turn=fc)
                                self._attn_window.register_message(cont, turn=fc)
                            state.resume_automark_count += 1
                            state.null_response_count += 1
                            return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][AutoMark] {e}")

        # 安全网 4: Backfill
        if (not is_resume and len(resp) > 100
                and not resp.rstrip().endswith(("?", "\uff1f")) and not _is_filler_content(resp)):
            self._run_backfill(state, resp, is_cb_path=False)

        # 安全网 5: 空回复拦截
        if not resp or len(resp.strip()) < 10:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_e
                _tg = _gtg_e()
                if _tg:
                    lv = _tg.get_leaf_nodes()
                    uc = [n for n in lv if n.status not in ("completed", "skipped")]
                    # 回退: get_leaf_nodes 排除了 req 根节点，当只有 req 且未完成时也应拦截
                    if not uc and not lv:
                        req_node = _tg.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uc = [req_node]
                    if uc:
                        nx = next((n for n in uc if n.status == "in_progress"), None)
                        if not nx:
                            nx = uc[0]
                            _tg.update_node_status(nx.id, "in_progress")
                        nudge = {"role": "user", "content":
                                 f"[空回复拦截] 任务图有 {len(uc)} 个未完成节点。"
                                 f"请立即调用工具执行任务: {nx.label}。不要输出空内容。"}
                        msgs.append(nudge)
                        if self._attn_window:
                            self._attn_window.register_message(nudge, turn=fc)
                        state.null_response_count += 1
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][EmptyGuard] {e}")

        if not resp or len(resp.strip()) < 10:
            resp = self._synthesize_from_task_graph() or resp

        # 安全网 6: 未完成任务拦截（不依赖关键词，纯状态判断）
        # 模型返回了有效文本，但 TaskGraph 仍有大量未完成节点
        # 跳过 filler 内容：模型持续输出无实质内容时，再次注入提示只会导致循环
        # 注意：排除 CRG 自动注入节点（crg_ 前缀），只看 LLM 规划的用户任务节点
        if resp and len(resp.strip()) >= 10 and not _is_filler_content(resp):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_uc
                tg_uc = _gtg_uc()
                if tg_uc:
                    leaves_uc = tg_uc.get_leaf_nodes()
                    # 排除 CRG 自动注入节点
                    user_leaves_uc = [n for n in leaves_uc if not n.id.startswith("crg_")]
                    total_uc = len(user_leaves_uc)
                    uncompleted_uc = [n for n in user_leaves_uc
                                      if n.status not in ("completed", "skipped")]
                    # 回退: 无叶子节点时检查 req 根节点（模型尚未创建子节点的情况）
                    if not user_leaves_uc:
                        req_node = tg_uc.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uncompleted_uc = [req_node]
                            total_uc = 1
                    # 超过阈值比例用户任务节点未完成 → 继续执行
                    if len(uncompleted_uc) > 0 and total_uc > 0 and len(uncompleted_uc) >= total_uc * _UNCOMPLETED_THRESHOLD:
                        current_uc = next(
                            (n for n in uncompleted_uc if n.status == "in_progress"),
                            uncompleted_uc[0],
                        )
                        if current_uc.status != "in_progress":
                            tg_uc.update_node_status(current_uc.id, "in_progress")
                            try:
                                from zulong.tools.task_tools import _save_active_backup
                                _save_active_backup()
                            except Exception:
                                pass
                        nudge_uc = {
                            "role": "user",
                            "content": (
                                f"[任务未完成] 仍有 {len(uncompleted_uc)}/{total_uc} 个子任务未完成。"
                                f"当前应执行: {current_uc.id}({current_uc.label})。"
                                f"请调用工具执行任务，不要只输出文本叙述。"
                                f"\n⚠️ 口头说「开始编写」「我将创建」而不调用工具 = 什么都没做！"
                            ),
                        }
                        msgs.append({"role": "assistant", "content": resp})
                        msgs.append(nudge_uc)
                        if self._attn_window:
                            self._attn_window.register_message(
                                {"role": "assistant", "content": resp}, turn=fc)
                            self._attn_window.register_message(nudge_uc, turn=fc)
                        state.null_response_count += 1
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][UncompletedGuard] {e}")

        # 安全网 6.5: 响应提前中断检测
        # LLM返回短文本进度汇报但function_call未生成时，不应判定为done
        # 条件：短文本(<80字符) + 非filler(含进度动词) + 有未完成节点(>=30%)
        if (resp and 6 <= len(resp.strip()) < 80
                and not _is_filler_content(resp)
                and state.null_response_count < 3):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _gtg_ri
                tg_ri = _gtg_ri()
                if tg_ri:
                    leaves_ri = tg_ri.get_leaf_nodes()
                    user_leaves_ri = [n for n in leaves_ri if not n.id.startswith("crg_")]
                    total_ri = len(user_leaves_ri)
                    uncompleted_ri = [n for n in user_leaves_ri
                                      if n.status not in ("completed", "skipped")]
                    if (total_ri > 0
                            and len(uncompleted_ri) > 0
                            and len(uncompleted_ri) >= total_ri * 0.3):
                        current_ri = next(
                            (n for n in uncompleted_ri if n.status == "in_progress"),
                            uncompleted_ri[0],
                        )
                        # 🔥 根据当前节点类型提供具体工具引导
                        label_ri = (current_ri.label or "").lower()
                        tool_ri_hint = ""
                        if any(kw in label_ri for kw in ("写", "编写", "创建", "生成", "代码", "code", "write", "create", "文件", "file", "html", "css", "js", "实现", "开发", "页面")):
                            tool_ri_hint = "\n请调用: write_to_file(path='文件路径', content='文件完整内容')"
                        elif any(kw in label_ri for kw in ("运行", "执行", "测试", "test", "run", "命令")):
                            tool_ri_hint = "\n请调用: execute_command(command='命令')"
                        elif any(kw in label_ri for kw in ("查看", "分析", "阅读", "检查", "read", "review")):
                            tool_ri_hint = "\n请调用: read_file(path='文件路径')"
                        nudge_ri = {
                            "role": "user",
                            "content": (
                                f"[响应提前中断] 你的回复仅{len(resp.strip())}字符，"
                                f"疑似function_call未生成。"
                                f"仍有 {len(uncompleted_ri)}/{total_ri} 个子任务未完成。"
                                f"当前应执行: {current_ri.id}({current_ri.label})。"
                                f"{tool_ri_hint}"
                                f"\n⚠️ 只说「开始编写」而不调用工具 = 什么都没做！"
                            ),
                        }
                        msgs.append({"role": "assistant", "content": resp})
                        msgs.append(nudge_ri)
                        if self._attn_window:
                            self._attn_window.register_message(
                                {"role": "assistant", "content": resp}, turn=fc)
                            self._attn_window.register_message(nudge_ri, turn=fc)
                        state.null_response_count += 1
                        logger.info(
                            f"[IDEFCRunner][ResponseIntegrity] turn={fc} "
                            f"resp_len={len(resp.strip())} "
                            f"uncompleted={len(uncompleted_ri)}/{total_ri} "
                            f"疑似响应提前中断(function_call未生成)"
                        )
                        return "cb_force" if state.null_response_count >= 4 else "continue"
            except Exception as e:
                logger.warning(f"[IDEFCRunner][ResponseIntegrity] {e}")

        # 安全网 7: 首轮无效回复拦截
        # 当工具增强策略首轮模型未调用工具，直接返回短回复/问候语时，
        # 注入纠正提示让模型重新聚焦任务并使用工具
        if (_is_filler_content(resp) and fc <= 1
                and getattr(state, "task_graph_policy", "") in {"inspect_or_create", "create", "extend"}
                and state.null_response_count < 3):
            _GREETING_PATTERNS = (
                "你好", "您好", "有什么我可以帮", "有什么可以帮",
                "我可以帮你", "需要帮助", "hello", "hi", "how can i",
                "what can i", "请问", "请说",
            )
            stripped_lower = (resp or "").strip().lower()
            is_greeting = any(p in stripped_lower for p in _GREETING_PATTERNS)
            is_too_short = len((resp or "").strip()) < 50

            if is_greeting or is_too_short:
                nudge_first = {
                    "role": "user",
                    "content": (
                        f"[首轮回复无效] 你返回了一个无意义的问候/短回复，但用户的任务是:\n"
                        f"「{(state.user_input_text or '')[:300]}」\n"
                        f"请不要打招呼或反问，直接分析任务需求并调用工具开始执行。"
                    ),
                }
                msgs.append({"role": "assistant", "content": resp})
                msgs.append(nudge_first)
                if self._attn_window:
                    self._attn_window.register_message(
                        {"role": "assistant", "content": resp}, turn=fc)
                    self._attn_window.register_message(nudge_first, turn=fc)
                state.null_response_count += 1
                logger.info(
                    f"[IDEFCRunner][FirstTurnGuard] 首轮无效回复被拦截: "
                    f"'{resp[:50]}', greeting={is_greeting}")
                return "continue"

        state.last_response_content = resp
        logger.info(f"[IDEFCRunner][EvalChain] turn={fc} path=default_done")
        return "done"

    def _pause_for_remote(self, state: IDEFCState, remote_calls: List[Dict]) -> IDEFCResult:
        state.pending_remote_calls = remote_calls
        state.pending_call_ids = [tc["id"] for tc in remote_calls]
        state.phase = "waiting_remote"
        self._save_runner_state()
        self.session.fc_state = state
        tool_names = [tc["function"]["name"] for tc in remote_calls]
        logger.info(f"[IDEFCRunner] FC 暂停, {len(remote_calls)} 远程工具: {tool_names}")
        # 推送 TaskGraph 更新到 web 仪表盘（远程工具分发时）
        try:
            self.engine._publish_task_graph_event(
                "agent_tool_call", state.fc_turn,
                ",".join(tool_names), f"远程工具调用: {tool_names}")
        except Exception:
            pass
        return IDEFCResult(phase="waiting_remote",
                             pending_call_ids=state.pending_call_ids)

    def _finalize(self, state: IDEFCState, reason: str) -> IDEFCResult:
        state.phase = "done"
        # 完成对话轮次记录（在保存记忆之前，确保 round 已 finalize）
        finalize_status = "completed" if reason == "done" else reason
        self._finalize_dialogue_round(state, status=finalize_status)
        # 如果非正常结束，将未完成的任务节点标记为 blocked
        if reason != "done":
            self._mark_unfinished_nodes_blocked(reason)
        self._auto_save_session_memory(state)
        self._save_runner_state()
        self.session.fc_state = state
        resp = state.last_response_content
        if not resp:
            for m in reversed(state.messages):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    c = m.get("content", "")
                    if c and len(c) > 10:
                        resp = c
                        break
        if not resp:
            resp = self.engine._get_fallback_response(state.user_input_text)
        logger.info(f"[IDEFCRunner] 终止: {reason}, turns={state.fc_turn}, len={len(resp or '')}")
        # 清理共享线程池
        try:
            self._model_executor.shutdown(wait=False)
        except Exception:
            pass
        # 推送终止事件到 web 仪表盘
        try:
            self.engine._publish_task_graph_event(
                "agent_done", state.fc_turn, "finalize", f"FC终止: {reason}")
        except Exception:
            pass
        # Web 监控: FC 终止（同步上下文 fire-and-forget）
        phase = "completed" if reason == "done" else ("cancelled" if reason in {"cancelled", "interrupted"} else "blocked")
        interaction = self._build_interaction_payload(
            phase,
            f"FC终止: {reason}",
            state.fc_turn,
            "FC_DONE",
            {
                "total_turns": state.fc_turn,
                "reason": reason,
                "summary": {
                    "completed": ["已保存当前执行状态"],
                    "verified": [],
                    "remaining": [] if reason == "done" else ["存在未完成步骤，已标记为可继续/可调整"],
                    "risk": "" if reason == "done" else f"本轮以 {reason} 结束，后续需要基于当前上下文继续。",
                    "next_step": "等待用户继续输入。",
                },
            },
        )
        _broadcast_sync("FC_DONE", {
            "protocol_version": "2.0",
            "session_id": self.session.session_id,
            "total_turns": state.fc_turn,
            "reason": reason,
            "phase": phase,
            "message": f"FC终止: {reason}",
            "interaction": interaction,
        })
        return IDEFCResult(phase="done", text_response=resp, reason=reason)

    def _get_current_task_graph(self):
        """获取当前会话的活跃TaskGraph实例"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            return get_active_task_graph()
        except Exception:
            return None

    def _mark_unfinished_nodes_blocked(self, reason: str) -> None:
        """FC 循环异常终止时，将所有 in_progress/pending 的叶节点标记为 blocked"""
        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if not tg:
                return
            for node in tg.get_leaf_nodes():
                if node.status == "in_progress":
                    node.status = "blocked"
                    node.result = f"FC循环异常终止: {reason}"
                    logger.info(f"[IDEFCRunner] 标记 blocked: {node.id} ({node.label})")
        except Exception:
            pass

    def _get_cb_fallback(self, state: IDEFCState) -> str:
        if state.tool_results_buffer:
            useful = [r["result"][:300] for r in state.tool_results_buffer
                      if r.get("result") and len(r.get("result", "")) > 20
                      and "error" not in r.get("result", "").lower()[:50]
                      and not r.get("result", "").lstrip().startswith(("{", "["))]
            if useful:
                return (
                    "系统当前出问题了，IDE FC 循环触发保护后没有生成有效回复，"
                    "因此无法正常回复。\n已收集到的工具信息如下，仅供排查：\n"
                    + "\n".join(useful[:3])
                )
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_fb
            _tg = _gtg_fb()
            if _tg:
                lv = _tg.get_leaf_nodes()
                cp = [n for n in lv if n.status == "completed"]
                uc = [n for n in lv if n.status not in ("completed", "skipped")]
                fb = (
                    "系统当前出问题了，IDE FC 循环触发保护后没有生成有效回复，"
                    "因此无法正常回复。\n"
                    f"当前任务「{_tg.title}」进度：{len(cp)}/{len(lv)} 完成。"
                )
                if uc:
                    fb += f"\n下一步：{uc[0].label}。"
                return fb
        except Exception:
            pass
        return self.engine._get_fallback_response(state.user_input_text)

    def _run_backfill(self, state: IDEFCState, response: str, is_cb_path: bool) -> None:
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_bf, _save_active_backup
            tg = _gtg_bf()
            if not tg:
                return
            jc = sum(1 for c in response if c in '{}[]":,')
            if (jc / max(len(response), 1)) > _JSON_DENSITY_THRESHOLD:
                # 密度超标时用 json.loads 验证是否真的是 JSON
                try:
                    _json.loads(response.strip())
                    return  # 确认为 JSON 结构，跳过 backfill
                except (ValueError, TypeError):
                    pass  # 非 JSON（如 Markdown 表格），继续 backfill
            leaves = tg.get_leaf_nodes()
            unc = [n for n in leaves if n.status not in ("completed", "skipped")]
            if not unc:
                return
            cnt = 0
            for nd in unc:
                if _has_content_match(response, nd.label):
                    tg.update_node_status(nd.id, "completed",
                                          result=_extract_node_content(response, nd.label, 500))
                    cnt += 1
            if cnt > 0:
                try: _save_active_backup()
                except Exception as e:
                    ErrorHandler.handle_exception(e, ErrorCode.CFG_SAVE_FAILED, context={"operation": "save_active_backup"}, log_level=logging.WARNING)
                logger.info(f"[IDEFCRunner][Backfill] {'CB ' if is_cb_path else ''}回填: {cnt}/{len(unc)}")
                self.engine._publish_task_graph_event(
                    "agent_tool_call", state.fc_turn, "task_backfill",
                    f'{{"backfilled":{cnt},"total_leaf":{len(leaves)}}}')
        except Exception as e:
            logger.warning(f"[IDEFCRunner][Backfill] {e}")

    def _apply_pressure_guidance(self, state: IDEFCState, fc: int) -> None:
        """上下文压力感知 → 注意力引导（两级：yellow 引导 / red 强制选择注意力工具）"""
        if not self._attn_window or state.cb_force_no_tools:
            return  # CB RED 已接管，不重复干预

        ratio = self._attn_window.usage_ratio

        # 分级（仅两级）
        if ratio >= 0.90:
            tier = "red"
        elif ratio >= 0.75:
            tier = "yellow"
        else:
            tier = "green"

        # 仅在跨越阈值时触发（避免每轮重复注入）
        if tier == self._last_pressure_tier:
            return

        self._last_pressure_tier = tier

        if tier == "green":
            return

        msgs = state.messages

        if tier == "yellow":
            # ── Yellow: 注入引导提示 + BFS 推荐焦点 ──
            acts = self._maybe_run_bfs(fc, trigger="pressure_crossing")

            parts = [
                f"[上下文压力 - 注意力引导] 当前上下文使用率已达 {ratio:.0%}。",
                "建议调用注意力工具收窄关注范围：",
                "  - adjust_attention_mode(mode='focus') 聚焦当前子任务",
                "  - navigate_attention(direction='deeper') 深入关键节点",
            ]

            # BFS 推荐节点
            if acts:
                seeds_set = set(self._compute_bfs_seeds())
                candidates = [
                    (nid, score) for nid, score in acts.items()
                    if score > 0.6 and nid not in seeds_set
                ]
                if candidates:
                    top_nid, top_score = max(candidates, key=lambda x: x[1])
                    parts.append(
                        f"  - navigate_attention(direction='jump', target_node_id='{top_nid}') "
                        f"[BFS推荐，激活分={top_score:.2f}]"
                    )

            hint = {"role": "system", "content": "\n".join(parts)}
            msgs.append(hint)
            self._attn_window.register_message(hint, turn=fc)
            logger.info(f"[IDEFCRunner][Pressure] YELLOW ({ratio:.0%}): 注入注意力工具引导")

        elif tier == "red":
            # ── Red: 约束 LLM 只能调用注意力工具 ──
            state.pressure_force_attention = True

            # BFS 推荐焦点
            acts = self._maybe_run_bfs(fc, trigger="pressure_crossing")

            parts = [
                f"[注意力强制切换] 上下文使用率达到 {ratio:.0%}（红色警戒）。",
                "你必须立即调用注意力工具进行焦点切换：",
                "  - adjust_attention_mode(mode='single_chain') 切换为单链推理模式",
                "  - navigate_attention(direction='deeper') 深入当前节点",
            ]

            if acts:
                seeds_set = set(self._compute_bfs_seeds())
                candidates = [
                    (nid, score) for nid, score in acts.items()
                    if score > 0.4 and nid not in seeds_set
                ]
                if candidates:
                    top_nid, top_score = max(candidates, key=lambda x: x[1])
                    parts.append(
                        f"  - navigate_attention(direction='jump', target_node_id='{top_nid}') "
                        f"[推荐焦点，激活分={top_score:.2f}]"
                    )

            hint = {"role": "system", "content": "\n".join(parts)}
            msgs.append(hint)
            self._attn_window.register_message(hint, turn=fc)
            logger.info(f"[IDEFCRunner][Pressure] RED ({ratio:.0%}): 强制注意力工具选择")

    @staticmethod
    def _get_attention_only_tools(tool_definitions: List[Dict]) -> List[Dict]:
        """压力 RED 时仅保留注意力工具，强制 LLM 从中选择"""
        _ATTENTION_TOOL_NAMES = {"navigate_attention", "adjust_attention_mode"}
        return [
            td for td in tool_definitions
            if td.get("function", {}).get("name", "") in _ATTENTION_TOOL_NAMES
        ]

    def _run_bfs_activation(self, fc_turn: int) -> None:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            mg = get_memory_graph()
            tg = get_active_task_graph()
            if not (mg and tg):
                return
            try:
                from zulong.memory.graph_adapters import TaskGraphAdapter
                s = TaskGraphAdapter().sync(mg, tg)
                if s:
                    logger.info(f"[IDEFCRunner] TG->MG sync: {s}")
            except Exception:
                pass
            ip = tg.get_nodes_by_status("in_progress")
            if not ip:
                return
            seeds = [f"task:{tg.id}/{n.id}" for n in ip]
            rs = getattr(mg, "_last_retrieved_node_ids", [])
            if rs:
                seeds.extend(rs)
            # CRG 增强: 最近被懒索引触碰的 CODE_SYMBOL 节点也加入 seed
            # 使 BFS 激活能发现代码节点与任务节点之间的语义关联
            try:
                from zulong.memory.memory_graph import NodeType as _NT
                recent_code = [
                    nid for nid, n in mg._nodes.items()
                    if n.node_type == _NT.CODE_SYMBOL
                    and n.last_accessed
                    and (time.time() - n.last_accessed) < 120  # 2 分钟内触碰的
                ]
                if recent_code:
                    seeds.extend(recent_code[:10])  # 限制数量避免 BFS 爆炸
            except Exception:
                pass
            valid, seen = [], set()
            for s in seeds:
                if s not in seen and mg.has_node(s):
                    valid.append(s)
                    seen.add(s)
            if valid:
                _cws = getattr(self.engine, "_context_window_size", 131072) if self.engine else 131072
                _ur = self._attn_window.usage_ratio if self._attn_window else 0.0
                if hasattr(mg, 'compute_activations_dynamic'):
                    acts = mg.compute_activations_dynamic(
                        valid,
                        context_window_size=_cws,
                        usage_ratio=_ur,
                    )
                else:
                    _min_act = 0.05 if len(valid) > 5 else 0.01
                    acts = mg.compute_activations(valid, max_depth=3, decay=0.5,
                                                  min_activation=_min_act)
                if acts:
                    # 记录 BFS 激活得分分布（方便诊断注意力切换行为）
                    top_acts = sorted(acts.items(), key=lambda x: -x[1])[:5]
                    logger.info(
                        f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(valid)}, "
                        f"activated={len(acts)}, top={top_acts}")
                    fc = mg.get_last_focus_context()
                    cf = fc.get("focused_task_node_id", "") if fc else ""
                    top = max(acts, key=acts.get)
                    if top != cf and acts[top] > 0.6 and top not in valid:
                        logger.info(
                            f"[IDEFCRunner][BFS] 焦点切换: {cf} → {top} "
                            f"(score={acts[top]:.3f})")
                        mg.update_focus_to_node(top)
                        if self._attn_window:
                            self._attn_window.on_navigate_attention(direction="jump", target_node_id=top)
                else:
                    logger.info(
                        f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(valid)}, "
                        f"无激活结果")
        except Exception as e:
            logger.info(f"[IDEFCRunner] BFS skip: {e}")

    def _compute_bfs_seeds(self) -> List[str]:
        """收集 BFS 种子（纯计算，无副作用）"""
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            mg = get_memory_graph()
            tg = get_active_task_graph()
            if not (mg and tg):
                return []
            ip = tg.get_nodes_by_status("in_progress")
            if not ip:
                return []
            seeds = [f"task:{tg.id}/{n.id}" for n in ip]
            rs = getattr(mg, "_last_retrieved_node_ids", [])
            if rs:
                seeds.extend(rs)
            # CRG 增强: 最近被懒索引触碰的 CODE_SYMBOL 节点
            try:
                from zulong.memory.memory_graph import NodeType as _NT
                recent_code = [
                    nid for nid, n in mg._nodes.items()
                    if n.node_type == _NT.CODE_SYMBOL
                    and n.last_accessed
                    and (time.time() - n.last_accessed) < 120
                ]
                if recent_code:
                    seeds.extend(recent_code[:10])
            except Exception:
                pass
            # 去重 + 验证存在性
            valid, seen = [], set()
            for s in seeds:
                if s not in seen and mg.has_node(s):
                    valid.append(s)
                    seen.add(s)
            return valid
        except Exception:
            return []

    def _maybe_run_bfs(self, fc_turn: int, trigger: str = "tool_complete") -> Optional[Dict[str, float]]:
        """条件执行 BFS，返回激活结果或 None

        trigger: "tool_complete" | "pressure_crossing"
        """
        if fc_turn <= 1:
            return None

        seeds = self._compute_bfs_seeds()
        if not seeds:
            return None

        # 变更检测
        import hashlib
        seeds_hash = hashlib.md5("|".join(sorted(seeds)).encode()).hexdigest()[:8]

        if trigger != "pressure_crossing":
            # 非压力触发：检查种子变更 + 最小间隔
            if seeds_hash == self._last_bfs_seeds_hash:
                return None
            if fc_turn - self._last_bfs_turn < self._bfs_min_interval:
                return None

        # TG→MG 同步（仅在 BFS 实际执行时）
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.tools.task_tools import get_active_task_graph
            from zulong.memory.graph_adapters import TaskGraphAdapter
            mg = get_memory_graph()
            tg = get_active_task_graph()
            if mg and tg:
                s = TaskGraphAdapter().sync(mg, tg)
                if s:
                    logger.info(f"[IDEFCRunner] TG->MG sync: {s}")
        except Exception:
            pass

        # 执行 BFS
        from zulong.memory.memory_graph import get_memory_graph
        mg = get_memory_graph()
        if not mg:
            return None
        _min_act = 0.05 if len(seeds) > 5 else 0.01
        _cws = getattr(self.engine, "_context_window_size", 131072) if self.engine else 131072
        _ur = self._attn_window.usage_ratio if self._attn_window else 0.0
        if hasattr(mg, 'compute_activations_dynamic'):
            acts = mg.compute_activations_dynamic(
                seeds,
                context_window_size=_cws,
                usage_ratio=_ur,
            )
        else:
            acts = mg.compute_activations(seeds, max_depth=3, decay=0.5,
                                          min_activation=_min_act)

        self._last_bfs_seeds_hash = seeds_hash
        self._last_bfs_turn = fc_turn

        # 日志
        if acts:
            top_acts = sorted(acts.items(), key=lambda x: -x[1])[:5]
            logger.info(
                f"[IDEFCRunner][BFS] turn={fc_turn} seeds={len(seeds)}, "
                f"activated={len(acts)}, top={top_acts}")

        return acts

    def _build_subtask_context(self) -> Optional[Dict]:
        try:
            from zulong.tools.task_tools import get_active_task_graph
            atg = get_active_task_graph()
            if not atg:
                return None
            ip = atg.get_nodes_by_status("in_progress")
            if not ip:
                return None
            cn = ip[0]
            deps = atg.get_dependencies(cn.id)
            avail = {}
            for did in deps:
                dn = atg.get_node(did)
                if dn and dn.status == "completed":
                    avail[did] = dn.result or ""
            return {"current_subtask": cn.id, "dependencies": deps, "available_results": avail}
        except Exception:
            return None

    def _synthesize_from_task_graph(self) -> Optional[str]:
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_s
            tg = _gtg_s()
            if not tg:
                return None
            lv = tg.get_leaf_nodes()
            cp = [n for n in lv if n.status == "completed"]
            if cp and len(cp) == len(lv):
                parts = [f"## {tg.title}\n"]
                for n in cp:
                    r = getattr(n, "result", "") or ""
                    parts.append(f"### {n.label}\n{r or '（已完成）'}\n")
                return "\n".join(parts)
        except Exception:
            pass
        return None

    # ── IDE 会话自动持久化 ──────────────────────────────────

    def _auto_create_task_plan(self, state: IDEFCState) -> None:
        """IDE 会话开始时自动创建任务计划

        智能判断是否需要创建新任务图：
        - 如果本 session 已有关联的 TG → 复用
        - 如果任务图策略是 reuse/inspect/continue → 复用已完成的旧图（用户想修订/扩展）
        - 如果全局 TG 所有叶节点都已完成 且策略不是复用 → 创建新 TG
        - 如果全局 TG 仍有未完成节点 → 不覆盖（其他会话可能在用）
        """
        try:
            if getattr(state, "task_graph_policy", "none") == "none":
                logger.debug("[IDEFCRunner] 任务图策略为 none，跳过自动任务图")
                return

            from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph
            existing_tg = get_active_task_graph()

            # ── 阶段1: 关联已有TaskGraph（不受 user_input 长度限制） ──
            if existing_tg:
                # 本 session 已关联了这个 TG → 直接复用
                if (self.session.active_task_graph_id
                        and hasattr(existing_tg, 'id')
                        and getattr(existing_tg, 'id', '') == self.session.active_task_graph_id):
                    return

                # 会话未关联 → 自动关联到全局活跃TaskGraph
                if hasattr(existing_tg, 'id') and not self.session.active_task_graph_id:
                    self.session.active_task_graph_id = getattr(existing_tg, 'id', '')
                    self._notify_session_linked(self.session.active_task_graph_id)
                    logger.info(
                        f"[IDEFCRunner] 会话自动关联到已有任务图 "
                        f"(graph_id={self.session.active_task_graph_id})"
                    )

                # 检查是否已全部完成
                leaves = existing_tg.get_leaf_nodes()
                uncompleted = [n for n in leaves
                               if n.status not in ("completed", "skipped")]
                if uncompleted:
                    logger.info(
                        f"[IDEFCRunner] 已有活跃任务图（{len(uncompleted)} 未完成），复用")
                    return

                # 继续已有任务图策略 → 复用旧图
                if state.is_resume:
                    logger.info(
                        f"[IDEFCRunner] 继续任务图策略：复用旧图"
                        f"（graph_id={getattr(existing_tg, 'id', '?')}）")
                    return

                # 所有节点已完成 + 非复用策略 → 允许创建新 TG
                logger.info("[IDEFCRunner] 旧任务图已全部完成，创建新任务图")

            # ── 阶段2: 创建新TaskGraph（受 user_input 长度限制） ──
            user_input = state.user_input_text
            if not user_input or len(user_input.strip()) < 5:
                logger.debug("[IDEFCRunner] user_input 过短，跳过创建新任务图")
                return

            import re as _re
            import time as _time
            from zulong.l2.task_graph import TaskGraph

            # 从 <task>...</task> 标签中提取纯任务文本（Cline IDE 会包裹用户输入）
            _task_tag_match = _re.search(
                r"<task>\s*(.*?)\s*</task>", user_input, _re.DOTALL
            )
            if _task_tag_match:
                _clean_input = _task_tag_match.group(1).strip()
            else:
                # 无 <task> 标签时，截断已知噪声段落
                _clean_input = _re.split(
                    r"\n#\s*task_progress|<task_progress>|\n====", user_input
                )[0].strip()

            if not _clean_input or len(_clean_input) < 3:
                _clean_input = user_input.strip()

            title = _clean_input[:80].strip()
            graph_id = f"tg_{int(_time.time())}"
            tg = TaskGraph(title=title, graph_id=graph_id)
            tg.add_node(
                id="req", label=title, type="requirement",
                status="in_progress", desc=title,
            )

            set_active_task_graph(tg, graph_id, workspace_dir=getattr(self, 'cwd', None))

            # 关联到当前 session，避免后续请求重复创建
            self.session.active_task_graph_id = graph_id
            self._notify_session_linked(graph_id)

            # 同步到 MemoryGraph
            try:
                from zulong.memory.memory_graph import get_memory_graph, GraphNode, NodeType
                mg = get_memory_graph()
                if mg:
                    task_node = GraphNode(
                        node_id=f"task:{graph_id}",
                        node_type=NodeType.TASK,
                        label=title,
                        activation=1.0,
                        created_at=_time.time(),
                        last_accessed=_time.time(),
                        access_count=1,
                        metadata={
                            "graph_id": graph_id, "status": "active",
                            "source": "ide_auto",
                        },
                    )
                    mg.add_node(task_node)
                    mg.index_summary(f"task:{graph_id}", title)
            except Exception as me:
                logger.debug(f"[IDEFCRunner] 任务节点同步到记忆图失败: {me}")

            logger.info(
                f"[IDEFCRunner] 自动创建任务计划: {title} (graph_id={graph_id})"
            )

            # 自动创建任务计划后激活规划模式，放宽 CB 模式检测
            # （模型接下来会大量调用 task_add_node 构建子任务节点）
            if self._circuit_breaker:
                self._circuit_breaker.escalate_for_planning()

            # 推送初始图到 web 仪表盘
            try:
                self.engine._publish_task_graph_event(
                    "pipeline_start", 0, "task_auto_create",
                    f"创建任务图: {title}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 自动创建任务计划失败（不影响FC循环）: {e}")

    def _auto_complete_task(self, state: IDEFCState) -> None:
        """FC 正常完成时自动标记任务节点

        in_progress → completed（当前正在做的确实结束了）
        pending → skipped（从未执行，标记为跳过而非虚假完成）
        排除 CRG 自动注入的 crg_ 节点，它们已由后台线程标记为 completed
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, _save_active_backup
            tg = get_active_task_graph()
            if not tg:
                return
            response = state.last_response_content or ""
            leaves = tg.get_leaf_nodes()
            completed_count = 0
            skipped_count = 0
            for leaf in leaves:
                if leaf.id.startswith("crg_"):
                    continue
                if leaf.status == "in_progress":
                    tg.update_node_status(
                        leaf.id, "completed",
                        result=response[:500] if response else "(IDE 会话已完成)",
                    )
                    completed_count += 1
                elif leaf.status == "pending":
                    tg.update_node_status(
                        leaf.id, "skipped",
                        result="(FC循环终止，任务未执行)",
                    )
                    skipped_count += 1
            total = completed_count + skipped_count
            if total > 0:
                try:
                    _save_active_backup()
                except Exception:
                    pass
                logger.info(
                    f"[IDEFCRunner] 自动标记任务完成: "
                    f"{completed_count} 个in_progress→completed, "
                    f"{skipped_count} 个pending→skipped"
                )
        except Exception as e:
            logger.debug(f"[IDEFCRunner] 自动标记任务完成失败: {e}")

    def _publish_fc_progress(self, state: IDEFCState, stage: str, detail: str = ""):
        """发布 FC 循环进度事件到 EventBus → IDEWebBridge → 仪表盘"""
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, ZulongEvent
            payload = {
                "component": "FCRunner",
                "fc_turn": state.fc_turn,
                "phase": state.phase,
                "stage": stage,
                "detail": detail,
                "task_graph_policy": getattr(state, "task_graph_policy", "none"),
                "cb_force": state.cb_force_no_tools,
                "max_turns": self._max_fc_turns,
                "timestamp": time.time(),
            }
            event = ZulongEvent(type=EventType("SYSTEM_STATUS"), payload=payload)
            event_bus.publish(event)
        except Exception:
            pass

    def _auto_save_session_memory(self, state: IDEFCState) -> None:
        """FC 会话结束后自动保存记忆节点（包含任务摘要和工具使用记录）"""
        try:
            user_input = state.user_input_text
            # 🔥 修复：优先使用 submit_final_answer 的完整内容，确保记忆持久化包含完整答案
            response = getattr(state, "final_answer", None) or state.last_response_content
            if not user_input or len(user_input.strip()) < 10:
                return

            # 收集本次会话使用的远程工具
            remote_tools_used = set()
            for msg in state.messages:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    for tc in (msg.get("tool_calls") or []):
                        fn = tc.get("function", {}).get("name", "")
                        if fn in IDE_REMOTE_TOOLS:
                            remote_tools_used.add(fn)

            import time as _time
            from zulong.memory.memory_graph import (
                get_memory_graph, GraphNode, NodeType, Importance,
            )
            mg = get_memory_graph()
            if not mg:
                return

            # 构建摘要
            tools_str = ", ".join(sorted(remote_tools_used)) if remote_tools_used else "无"
            summary = (
                f"IDE 任务: {user_input[:200]}\n"
                f"使用工具: {tools_str}\n"
                f"结果摘要: {(response or '')[:300]}"
            )

            node_id = f"note:ide_{int(_time.time() * 1000)}"
            node = GraphNode(
                node_id=node_id,
                node_type=NodeType.KNOWLEDGE,
                label=f"IDE任务: {user_input[:50]}",
                activation=0.8,
                created_at=_time.time(),
                last_accessed=_time.time(),
                access_count=1,
                metadata={
                    "content": summary,
                    "importance": Importance.NORMAL.value,
                    "source": "ide_auto_session",
                    "tools_used": list(remote_tools_used),
                },
            )
            mg.add_node(node)
            mg.set_importance(node_id, Importance.NORMAL)
            mg.index_summary(node_id, summary)

            # 关联到任务节点
            try:
                from zulong.tools.task_tools import get_active_task_graph
                from zulong.memory.memory_graph import EdgeType
                tg = get_active_task_graph()
                if tg:
                    task_mg_id = f"task:{tg.graph_id}" if hasattr(tg, "graph_id") else None
                    if task_mg_id and mg.has_node(task_mg_id):
                        mg.add_edge(task_mg_id, node_id, EdgeType.REFERENCE, weight=0.7)
            except Exception:
                pass

            # [P1 修复] 调用 ExperienceGenerator 从对话中提取经验
            try:
                from zulong.memory.experience_generator import ExperienceGenerator
                from zulong.memory.rag_manager import RAGManager
                rag = RAGManager()
                if hasattr(rag, '_initialized') and rag._initialized:
                    eg = ExperienceGenerator(rag_manager=rag)
                    dialogue_history = [
                        m for m in state.messages
                        if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                        and m.get("content")
                    ]
                    if len(dialogue_history) >= 2:
                        stats = eg.process_dialogue_batch(dialogue_history)
                        if stats.get("added", 0) > 0:
                            logger.info(
                                f"[IDEFCRunner] 经验提取: "
                                f"extracted={stats['extracted']}, added={stats['added']}")
            except Exception as exp_err:
                logger.debug(f"[IDEFCRunner] 经验提取跳过: {exp_err}")

            logger.info(f"[IDEFCRunner] 自动保存会话记忆: {node_id}")
        except Exception as e:
            logger.warning(f"[IDEFCRunner] 自动保存会话记忆失败: {e}")
