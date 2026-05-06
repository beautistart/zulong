"""
IDE 模式 FC 循环运行器

使用 Python while 循环替代 LangGraph StateGraph，支持跨 HTTP 请求的暂停/恢复。
完整复用 fc_graph.py 的所有安全网逻辑。

与 fc_graph.py 的区别：
1. while 循环替代 StateGraph（跨请求暂停）
2. 工具调用分流：内部工具直接执行，远程工具暂停返回
3. 状态完全序列化到 IDEFCState
4. 注意力窗口、RuleGuardian、CircuitBreaker 为 per-runner 实例
"""

import asyncio
import concurrent.futures
import json as _json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from zulong.ide.ide_session import IDEFCState, AgentSession
from zulong.ide.ide_tool_registry import IDEToolRegistry, IDE_REMOTE_TOOLS
from zulong.ide.ide_format_translator import IDEFormatTranslator
from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
from zulong.l2.circuit_breaker import CircuitBreakerState

if TYPE_CHECKING:
    from zulong.l2.inference_engine import InferenceEngine

# Web 监控事件广播（延迟导入避免循环依赖）
def _broadcast_sync(event_type: str, payload: dict) -> None:
    """在同步上下文中安排广播（fire-and-forget）"""
    try:
        from zulong.ide.ide_server import broadcast_monitor_event
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(broadcast_monitor_event(event_type, payload))
        else:
            loop.run_until_complete(broadcast_monitor_event(event_type, payload))
    except Exception:
        pass

logger = logging.getLogger(__name__)


@dataclass
class IDEFCResult:
    """FC 循环执行结果"""
    phase: str
    text_response: Optional[str] = None
    pending_call_ids: Optional[List[str]] = None


_FILLER_PATTERNS = [
    "我正在思考", "让我继续", "我来继续", "让我想想", "接下来我",
    "我正在处理", "正在分析", "正在执行", "稍等", "我需要",
    "但我需要", "不过我需要", "还需要进一步", "需要更多信息",
]

# 工具结果缓冲区最大条目数（防止无限增长）
_TOOL_RESULTS_BUFFER_MAX = 100

# Backfill JSON 密度阈值：response 中 JSON 特征字符占比超过此值则跳过
_JSON_DENSITY_THRESHOLD = 0.12


def _is_filler_content(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    pattern_count = sum(1 for p in _FILLER_PATTERNS if p in stripped)
    # 长文本中少量关键词不应被判为 filler
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


class IDEFCRunner:
    """IDE 模式 FC 循环运行器

    per-runner 实例隔离注意力窗口/RuleGuardian/CircuitBreaker，
    避免并发 Session 状态冲突。
    """

    def __init__(self, engine: "InferenceEngine", session: AgentSession,
                 tool_registry: IDEToolRegistry):
        self.engine = engine
        self.session = session
        self.tool_registry = tool_registry
        self.translator = IDEFormatTranslator()
        self._max_fc_turns = getattr(engine, "_max_fc_turns", 100)
        self._soft_limit = getattr(engine, "_soft_limit", 50)
        self._hard_limit = getattr(engine, "_hard_limit", 100)
        self._warning_interval = getattr(engine, "_warning_interval", 10)
        self._fc_loop_timeout = getattr(engine, "_fc_loop_timeout", 600)
        self._fc_request_interval = getattr(engine, "_fc_request_interval", 1.0)
        # P2: 进度报告 + 弹性预算
        self._progress_report_interval = getattr(engine, "_progress_report_interval", 30)
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
        # 共享线程池：整个 runner 生命周期复用，避免每次 _call_model 创建/销毁
        self._model_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ide_fc_model")
        # BFS 调度控制
        self._last_bfs_seeds_hash: str = ""
        self._last_bfs_turn: int = 0
        self._last_pressure_tier: str = "green"  # 压力分级跟踪（green/yellow/red）
        self._bfs_min_interval: int = 3  # 最小间隔轮次

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
        # 初始化状态（同步，在线程池中运行）
        loop = asyncio.get_event_loop()
        state = await loop.run_in_executor(None, self._init_state, messages)
        self.session.fc_state = state

        # Web 监控: FC 循环启动
        from zulong.ide.ide_server import broadcast_monitor_event
        try:
            await broadcast_monitor_event("FC_START", {
                "session_id": self.session.session_id,
                "max_turns": self._max_fc_turns,
                "intent": getattr(state, "ide_intent", ""),
                "user_input": (state.user_input_text or "")[:500],
            })
        except Exception:
            pass

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
                await send_callback("status_update", {
                    "turn": state.fc_turn, "phase": "calling_model"})

                # Web 监控: 调用模型（保护性包裹）
                try:
                    await broadcast_monitor_event("CALLING_MODEL", {
                        "turn": state.fc_turn,
                        "model": getattr(state, "vllm_model_id", ""),
                    })
                except Exception:
                    pass

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
                        continue  # 继续等待，下次循环检查 cancel_event

                if tc_data is None and resp_content is None:
                    if state.api_timeout_count >= 2:
                        return await loop.run_in_executor(
                            None, self._finalize, state, "api_error")
                    await asyncio.sleep(2)  # API 错误后短暂退避
                    continue

                state.loop_error_count = 0

                if tc_data:
                    # Web 监控: 工具调用（保护性包裹）
                    tool_names = [tc["function"]["name"] for tc in tc_data]
                    try:
                        await broadcast_monitor_event("TOOL_CALL", {
                            "turn": state.fc_turn,
                            "tools": tool_names,
                            "count": len(tc_data),
                        })
                    except Exception:
                        pass
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
                    # Web 监控: 工具执行完毕，轮次完成（保护性包裹）
                    try:
                        await broadcast_monitor_event("TURN_COMPLETE", {
                            "turn": state.fc_turn,
                            "has_tool_calls": True,
                            "tool_names": tool_names,
                        })
                    except Exception:
                        pass
                    # 继续循环
                    continue

                # 纯文本回复 → 先评估，再决定是否推送给前端
                # 避免安全网返回 "continue" 时重复显示 filler 内容
                verdict = await loop.run_in_executor(
                    None, self._eval_response, state, resp_content or "")

                # 只有最终确认 "done" 时才推送 display_text
                # cb_force 时不推送，下一轮强制无工具回复后再推送
                final_text = state.last_response_content or resp_content
                if verdict == "done" and final_text:
                    await send_callback("display_text", {
                        "text": final_text, "turn": state.fc_turn})
                    # Web 监控: 模型文本回复（保护性包裹）
                    try:
                        await broadcast_monitor_event("MODEL_RESPONSE", {
                            "session_id": self.session.session_id,
                            "turn": state.fc_turn,
                            "text": (final_text or "")[:5000],
                            "text_preview": (final_text or "")[:200],
                            "text_length": len(final_text or ""),
                        })
                    except Exception:
                        pass

                if verdict == "done":
                    state.phase = "done"
                    # 立即通知前端任务完成（在后处理之前，防止 WS 断开导致丢失）
                    _done_text = state.last_response_content or ""
                    await send_callback("task_complete", {"result": _done_text})
                    # Web 监控: FC 循环完成（保护性包裹）
                    try:
                        await broadcast_monitor_event("FC_DONE", {
                            "session_id": self.session.session_id,
                            "total_turns": state.fc_turn,
                            "reason": "done",
                        })
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
                    return IDEFCResult(
                        phase="done",
                        text_response=state.last_response_content)
                elif verdict == "cb_force":
                    state.cb_force_no_tools = True
                # "continue" → 继续循环

            except asyncio.CancelledError:
                return await loop.run_in_executor(
                    None, self._finalize, state, "cancelled")
            except Exception as loop_err:
                logger.error(
                    f"[IDEFCRunner] async 循环异常 turn={state.fc_turn}: "
                    f"{loop_err}", exc_info=True)
                state.loop_error_count += 1
                if state.loop_error_count >= 3:
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
                tn = td["function"]["name"]
                await send_callback("status_update", {
                    "turn": fc, "phase": "exec_internal_tool",
                    "tool_name": tn})
                await loop.run_in_executor(
                    None, self._exec_internal, state, td, fc, grp)

            # CircuitBreaker 评估
            try:
                if self._circuit_breaker:
                    cb_s, cb_r = self._circuit_breaker.evaluate(fc, msgs)
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
                # 设置 pending 状态（_inject_tool_results 会验证）
                state.pending_remote_calls = valid_remote
                state.pending_call_ids = [
                    tc["id"] for tc in valid_remote]

                # 通过 WebSocket 推送 tool_request
                tool_names = [
                    tc["function"]["name"] for tc in valid_remote]
                logger.info(
                    f"[IDEFCRunner] async 远程工具推送: {tool_names}")

                await send_callback("tool_request", {
                    "tool_calls": valid_remote,
                    "call_ids": state.pending_call_ids,
                    "tool_names": tool_names,
                })

                # Web 监控: 远程工具调用详情（保护性包裹，不影响核心流程）
                try:
                    await broadcast_monitor_event("IDE_TOOL_REQUEST", {
                        "session_id": self.session.session_id,
                        "turn": fc,
                        "tools": [
                            {"name": tc["function"]["name"],
                             "arguments_preview": tc["function"].get("arguments", "")[:300],
                             "call_id": tc.get("id", "")}
                            for tc in valid_remote
                        ],
                    })
                except Exception:
                    pass

                await send_callback("status_update", {
                    "turn": fc, "phase": "waiting_tool_result"})

                # 等待所有远程工具结果
                results = []
                for i in range(len(valid_remote)):
                    if cancel_event.is_set():
                        return "cancelled"
                    try:
                        result = await asyncio.wait_for(
                            tool_result_queue.get(), timeout=300)
                        results.append(result)
                    except asyncio.TimeoutError:
                        tc = valid_remote[i]
                        results.append({
                            "call_id": tc["id"],
                            "tool_name": tc["function"]["name"],
                            "result": "[工具执行超时 (300s)]",
                            "is_error": True,
                        })

                # 转换为 _inject_tool_results 期望的格式
                formatted_results = []
                for r in results:
                    formatted_results.append({
                        "tool_call_id": r.get("call_id", ""),
                        "content": r.get("result", ""),
                    })

                self._inject_tool_results(state, formatted_results)
                self._maybe_run_bfs(fc, "tool_complete")

                # Web 监控: 远程工具执行结果（保护性包裹）
                try:
                    await broadcast_monitor_event("IDE_TOOL_RESULT", {
                        "session_id": self.session.session_id,
                        "turn": fc,
                        "results": [
                            {"tool_name": r.get("tool_name", ""),
                             "call_id": r.get("call_id", ""),
                             "result_preview": (r.get("result", "") or "")[:500],
                             "is_error": r.get("is_error", False)}
                            for r in results
                        ],
                    })
                except Exception:
                    pass

                await send_callback("status_update", {
                    "turn": state.fc_turn, "phase": "running"})
                return None  # 继续循环

        self._maybe_run_bfs(fc, "tool_complete")
        return None  # 继续循环

    def _restore_runner_state(self) -> None:
        if self.session.attention_window_data:
            try:
                from zulong.l2.attention_window import AttentionWindowManager
                self._attn_window = AttentionWindowManager.from_serialized(
                    self.session.attention_window_data)
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
        except Exception as e:
            logger.warning(f"[IDEFCRunner] SemanticDriftDetector 创建失败: {e}")

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

    def _init_state(self, messages: List[Dict]) -> IDEFCState:
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

        # ── Layer 1: 意图检测 ──────────────────────────────────
        _force_gid = getattr(self, 'force_graph_id', '') or ''
        if _force_gid:
            # 确定性恢复: graph_id 已由 ide_server 加载，跳过启发式
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg and getattr(_tg, 'id', '') == _force_gid:
                intent = "resume"
                has_active_tg = True
                self.session.active_task_graph_id = _force_gid
                logger.info(
                    f"[IDEFCRunner] 确定性恢复模式: graph_id={_force_gid}")
            else:
                # 活跃图加载失败(不应发生), 降级到启发式
                logger.warning(
                    f"[IDEFCRunner] 确定性恢复降级: 活跃图不匹配 {_force_gid}")
                intent, has_active_tg = self._detect_ide_intent(user_input)
        else:
            intent, has_active_tg = self._detect_ide_intent(user_input)

        # 非确定性路径下，恢复模式关联活跃图到 session
        if not _force_gid and intent == "resume" and has_active_tg:
            from zulong.tools.task_tools import get_active_task_graph
            _tg = get_active_task_graph()
            if _tg and hasattr(_tg, 'id') and not self.session.active_task_graph_id:
                self.session.active_task_graph_id = getattr(_tg, 'id', None)
                logger.info(
                    f"[IDEFCRunner] 恢复模式：关联活跃图 "
                    f"{self.session.active_task_graph_id} 到新 session")

        # ── Layer 2: 根据意图获取过滤后的工具定义 ─────────────
        tool_defs = self.tool_registry.get_combined_tool_definitions_for_intent(intent)

        # RESUME 首轮强制 task_view_overview
        force_first = (intent == "resume" and has_active_tg)

        state = IDEFCState(
            messages=list(messages), fc_turn=0, tool_definitions=tool_defs,
            user_input_text=user_input, vllm_model_id=LLM_MODEL_ID or "",
            phase="running", intent_max_tokens=8192,
            is_resume=(intent == "resume"),
            ide_intent=intent,
            force_first_tool=force_first,
        )
        from zulong.l2.attention_window import AttentionWindowManager
        self._attn_window = AttentionWindowManager(
            context_window_size=getattr(self.engine, "_context_window_size", 32768))
        for msg in messages:
            self._attn_window.register_message(msg, turn=0, pinned=msg.get("role") == "system")
        self._create_rule_guardian()
        self._create_circuit_breaker()
        self._create_drift_detector()
        self._auto_create_task_plan(state)
        self._init_dialogue_tracking(state)
        logger.info(
            f"[IDEFCRunner] 意图={intent}, force_first_tool={force_first}, "
            f"tools={len(tool_defs)}, has_active_tg={has_active_tg}"
        )
        return state

    def _detect_ide_intent(self, user_input: str) -> tuple:
        """IDE 意图检测启发式（轻量，无需 LLM 调用）

        规则：
        0. Session 上下文：如果上一次 FC 处于 waiting_remote 且有活跃 TG → RESUME
        1. 存在活跃 TaskGraph 且有未完成节点 → RESUME
        2. 用户输入包含恢复关键词 → RESUME
        3. 其余 → COMPLEX（IDE 模式无 CHAT，因为 IDE 只在有任务时才调用）

        Returns:
            (intent: str, has_active_task_graph: bool)
        """
        has_active_tg = False
        has_uncompleted = False

        try:
            from zulong.tools.task_tools import get_active_task_graph
            tg = get_active_task_graph()
            if tg is not None:
                has_active_tg = True
                leaves = tg.get_leaf_nodes()
                uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
                has_uncompleted = bool(uncompleted)
        except Exception:
            pass

        # 规则 0: Session 上下文 — 上一次 FC 正在等待 IDE 工具结果
        if (self.session.fc_state
                and self.session.fc_state.phase == "waiting_remote"
                and has_active_tg):
            logger.info("[IDEFCRunner] 意图检测: RESUME（session 处于 waiting_remote）")
            return "resume", has_active_tg

        # 规则 1: 有活跃任务图且有未完成节点 → RESUME
        if has_active_tg and has_uncompleted:
            logger.info("[IDEFCRunner] 意图检测: RESUME（活跃任务图有未完成节点）")
            return "resume", has_active_tg

        # 规则 1.5: 用户引用了任务节点 @[label#address] → 从 MemoryGraph 重建并 RESUME
        ref_graph_id = self._try_activate_from_reference(user_input)
        if ref_graph_id:
            logger.info(
                f"[IDEFCRunner] 意图检测: RESUME（节点引用触发重建 graph={ref_graph_id}）")
            return "resume", True

        # 规则 2: 恢复关键词
        _resume_keywords = (
            "继续", "接着做", "接着", "恢复", "上次", "之前的任务",
            "resume", "continue", "pick up",
        )
        stripped = (user_input or "").strip().lower()
        if any(kw in stripped for kw in _resume_keywords):
            if has_active_tg:
                logger.info("[IDEFCRunner] 意图检测: RESUME（关键词 + 活跃任务图）")
                return "resume", has_active_tg
            # 无活跃 TG 但有恢复关键词 → 尝试从磁盘备份加载最近图谱（含已完成的）
            try:
                from zulong.tools.task_tools import (
                    load_latest_backup, set_active_task_graph
                )
                backup_tg, backup_gid = load_latest_backup()
                if backup_tg and backup_gid:
                    set_active_task_graph(backup_tg, backup_gid)
                    logger.info(
                        f"[IDEFCRunner] 意图检测: RESUME"
                        f"（关键词触发从备份恢复 graph={backup_gid}）")
                    return "resume", True
            except Exception as e:
                logger.debug(f"[IDEFCRunner] 备份加载尝试失败: {e}")

        # 默认: COMPLEX
        logger.info("[IDEFCRunner] 意图检测: COMPLEX")
        return "complex", has_active_tg

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

        state.pending_remote_calls = []
        state.pending_call_ids = []
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
        - 不创建 CodeAnchor 记录（精确锚定留给 LLM 手动调用 zulong_task_link_code）
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

            # ── 自动锚定：TASK → CODE_SYMBOL 边 ──
            # 仅在有活跃任务时建立（粗粒度：关联所有新符号）
            if task_mg_id and mg.has_node(task_mg_id):
                anchored = 0
                for sym in result.symbols:
                    code_node_id = sym.node_id
                    if mg.has_node(code_node_id):
                        # add_edge 是幂等的（已存在则仅更新 weight/timestamp）
                        mg.add_edge(
                            task_mg_id, code_node_id,
                            edge_type=EdgeType.REFERENCE,
                            weight=0.6,  # 自动锚定权重低于手动锚定(0.9)
                            metadata={
                                "relation": "auto_anchored",
                                "anchor_type": "implementation",
                                "source": "write_hook",
                            },
                        )
                        anchored += 1
                if anchored:
                    logger.info(
                        f"[IDEFCRunner] 自动锚定: {file_path} → "
                        f"TASK({task_mg_id}) ↔ {anchored} CODE_SYMBOL 节点"
                    )

        except Exception as e:
            logger.debug(f"[IDEFCRunner] _index_and_anchor_file({file_path}) 异常: {e}")

    def _run_loop(self, state: IDEFCState) -> IDEFCResult:
        while True:
            tr = self._check(state)
            if tr:
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
                        return self._finalize(state, "api_error")
                    time.sleep(2)  # API 错误后短暂退避
                    continue
                # 成功获得模型响应，重置连续错误计数
                state.loop_error_count = 0
                if tc_data:
                    self._publish_fc_progress(state, "exec_tools", f"{len(tc_data)} tool calls")
                    remote = self._exec_tools(state, tc_data, resp_content)
                    if remote:
                        self._publish_fc_progress(state, "pause_for_remote", f"{len(remote)} remote tools")
                        return self._pause_for_remote(state, remote)
                    continue
                verdict = self._eval_response(state, resp_content or "")
                if verdict == "done":
                    state.phase = "done"
                    self._auto_complete_task(state)
                    self._finalize_dialogue_round(state, status="completed")
                    self._auto_save_session_memory(state)
                    self._save_runner_state()
                    self.session.fc_state = state
                    return IDEFCResult(phase="done", text_response=state.last_response_content)
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

    def _build_progress_hint(self, state: IDEFCState) -> str:
        """构建进度提示，注入到 LLM 上下文（中性通报，不催促结束）"""
        fc = state.fc_turn
        hint = f"[系统进度通报] 当前已执行 {fc} 步。"
        # 附加任务图进度（如果有）
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        if tg:
            all_nodes = [n for n in tg.nodes.values() if n.id != "req"]
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

    def _broadcast_periodic_progress(self, state: IDEFCState) -> None:
        """周期性进度广播（不触发续期，仅通知 Web 仪表盘当前状态）"""
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
        report = {"turn": state.fc_turn, "type": "periodic"}
        if tg:
            all_nodes = [n for n in tg.nodes.values() if n.id != "req"]
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
            all_nodes = [n for n in tg.nodes.values() if n.id != "req"]
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
        """
        reports = state.progress_reports
        n = self._max_reports_before_force_stop
        if len(reports) < n:
            return False
        recent = reports[-n:]
        completed_counts = [r.get("completed_count", 0) for r in recent]
        total_counts = [r.get("total_nodes", 0) for r in recent]
        # 只有完成数和总节点数都没变化才算停滞
        return len(set(completed_counts)) == 1 and len(set(total_counts)) == 1

    def _call_model(self, state: IDEFCState) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """LLM API 调用。返回 (tool_calls, content)。都为 None 表示超时。"""
        fc = state.fc_turn
        msgs = self._attn_window.apply_window() if self._attn_window else state.messages
        extra_kw = self.engine._get_llm_extra_kwargs()
        # Qwen3 系列默认开启思维链（<think>），FC 模式下禁用以避免空 content
        eb = extra_kw.get("extra_body", {})
        eb["enable_thinking"] = False
        extra_kw["extra_body"] = eb
        kw: Dict[str, Any] = {
            "model": state.vllm_model_id, "messages": msgs,
            "max_tokens": state.intent_max_tokens, "temperature": 0.3,
            "top_p": 0.85, "stream": False, **extra_kw,
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
            # ── Layer 3: force_first_tool (RESUME 首轮强制 task_view_overview) ──
            if state.force_first_tool and fc == 1:
                kw["tool_choice"] = {"type": "function", "function": {"name": "task_view_overview"}}
                state.force_first_tool = False  # 只强制一次
                logger.info("[IDEFCRunner] RESUME 首轮强制 task_view_overview")
            else:
                kw["tool_choice"] = "auto"
        future = self._model_executor.submit(
            lambda: self.engine.vllm_client.chat.completions.create(**kw))
        try:
            api_resp = future.result(timeout=self._fc_loop_timeout)
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
                    retry_resp = self.engine.vllm_client.chat.completions.create(**kw)
                    ch = retry_resp.choices[0]
                    m = ch.message
                    state.api_timeout_count = 0
                    tc = None
                    rc = m.content or ""
                    if m.tool_calls:
                        logger.info(f"[IDEFCRunner] Turn {fc}: {len(m.tool_calls)} 工具调用 (429重试成功)")
                        tc = [{"id": t.id, "type": "function",
                               "function": {"name": t.function.name, "arguments": t.function.arguments}}
                              for t in m.tool_calls]
                        for t in tc:
                            logger.info(f"[IDEFCRunner]   FC tool: {t['function']['name']} args={t['function']['arguments'][:200]}")
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
                        max_tokens=state.intent_max_tokens, temperature=0.3,
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
        ch = api_resp.choices[0]
        m = ch.message
        # API 调用成功，重置连续错误计数
        state.api_timeout_count = 0
        tc = None
        rc = m.content or ""
        if m.tool_calls:
            logger.info(f"[IDEFCRunner] Turn {fc}: {len(m.tool_calls)} 工具调用 (function calling)")
            tc = [{"id": t.id, "type": "function",
                   "function": {"name": t.function.name, "arguments": t.function.arguments}}
                  for t in m.tool_calls]
            for t in tc:
                logger.info(f"[IDEFCRunner]   FC tool: {t['function']['name']} args={t['function']['arguments'][:200]}")
        else:
            # 回退：检查内容中是否包含 XML 格式的工具调用
            # LLM 可能跟随 IDE 系统提示词输出 XML 而非使用 function calling
            xml_tc = self.translator.parse_xml_tool_calls(rc)
            if xml_tc:
                logger.info(f"[IDEFCRunner] Turn {fc}: {len(xml_tc)} 工具调用 (XML 回退解析)")
                for xt in xml_tc:
                    logger.info(f"[IDEFCRunner]   XML tool: {xt['function']['name']} args={xt['function']['arguments'][:200]}")
                tc = xml_tc
                # 从内容中移除 XML 工具标签，保留前置文本
                rc = self._strip_xml_tool_tags(rc)
            else:
                # 即使未解析出工具调用，文本中仍可能含有 XML 残留片段
                # （LLM 输出了不完整/非标准的 XML 工具调用）
                rc = self._strip_xml_tool_tags(rc)
                logger.info(f"[IDEFCRunner] Turn {fc}: 文本回复 len={len(rc)}")
                # 调试：记录文本前 300 字符，帮助诊断 LLM 是否输出了 XML 但未被解析
                if rc:
                    snippet = rc[:300].replace('\n', '\\n')
                    logger.info(f"[IDEFCRunner] Turn {fc} 文本预览: {snippet}")
        return tc, rc

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
                    cb_s, cb_r = self._circuit_breaker.evaluate(fc, msgs)
                    if cb_s == CircuitBreakerState.RED:
                        logger.warning(f"[IDEFCRunner][CB] RED: {cb_r}")
                        state.cb_force_no_tools = True
                        cm = {"role": "user", "content":
                              f"[Circuit Breaker 强制收敛] {cb_r}\n你必须立刻基于已有信息生成最终回复，不允许再调用任何工具。"}
                        msgs.append(cm)
                        if self._attn_window:
                            self._attn_window.register_message(cm, turn=fc)
                        remote = []
                    elif cb_s == CircuitBreakerState.YELLOW:
                        logger.warning(f"[IDEFCRunner][CB] YELLOW: {cb_r}")
                        ch = {"role": "user", "content":
                              f"[Circuit Breaker 警告] {cb_r}\n请尽快总结当前信息并回复用户，避免继续调用更多工具。"}
                        msgs.append(ch)
                        if self._attn_window:
                            self._attn_window.register_message(ch, turn=fc)
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
        # Web 监控: 内部工具执行
        _broadcast_sync("IDE_TOOL_EXEC", {
            "session_id": self.session.session_id,
            "turn": fc,
            "tool_name": tn,
            "arguments_preview": _json.dumps(ta, ensure_ascii=False)[:300],
            "result_preview": (rt or "")[:500],
        })

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
                        return "continue"
            except Exception:
                pass
            logger.info(f"[IDEFCRunner][CB] 强制回复, len={len(resp)}")
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

        # 安全网 3: AutoMark RESUME
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
                        except Exception: pass
                        rem = [n for n in tg.get_leaf_nodes() if n.status != "completed"]
                        if rem:
                            nn = rem[0]
                            tg.update_node_status(nn.id, "in_progress")
                            try: _save_active_backup()
                            except Exception: pass
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
                    # 超过半数用户任务节点未完成 → 继续执行
                    if len(uncompleted_uc) > 0 and total_uc > 0 and len(uncompleted_uc) >= total_uc * 0.5:
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
                                f"请继续调用工具执行任务，不要提前生成最终总结。"
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

        # 安全网 7: 首轮无效回复拦截
        # 当 COMPLEX 意图首轮模型未调用工具，直接返回短回复/问候语时，
        # 注入纠正提示让模型重新聚焦任务并使用工具
        if (_is_filler_content(resp) and fc <= 1
                and getattr(state, "ide_intent", "") == "complex"
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
        _broadcast_sync("FC_DONE", {
            "session_id": self.session.session_id,
            "total_turns": state.fc_turn,
            "reason": reason,
        })
        return IDEFCResult(phase="done", text_response=resp)

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
                return "根据已收集的信息：\n" + "\n".join(useful[:3])
        try:
            from zulong.tools.task_tools import get_active_task_graph as _gtg_fb
            _tg = _gtg_fb()
            if _tg:
                lv = _tg.get_leaf_nodes()
                cp = [n for n in lv if n.status == "completed"]
                uc = [n for n in lv if n.status not in ("completed", "skipped")]
                fb = f"任务「{_tg.title}」进度：{len(cp)}/{len(lv)} 完成。"
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
                except Exception: pass
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
                # 当种子过多时提高 min_activation 阈值，抑制 BFS 扩散膨胀
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
        - 如果是 RESUME 意图 → 复用已完成的旧图（用户想修订/扩展）
        - 如果全局 TG 所有叶节点都已完成 且 不是 RESUME → 创建新 TG
        - 如果全局 TG 仍有未完成节点 → 不覆盖（其他会话可能在用）
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph
            existing_tg = get_active_task_graph()
            if existing_tg:
                # 本 session 已关联了这个 TG → 直接复用
                if (self.session.active_task_graph_id
                        and hasattr(existing_tg, 'id')
                        and getattr(existing_tg, 'id', '') == self.session.active_task_graph_id):
                    return
                # 全局 TG 存在 → 检查是否已全部完成
                leaves = existing_tg.get_leaf_nodes()
                uncompleted = [n for n in leaves
                               if n.status not in ("completed", "skipped")]
                if uncompleted:
                    # 仍有未完成节点，不覆盖 → 主动关联到当前 session
                    if hasattr(existing_tg, 'id') and not self.session.active_task_graph_id:
                        self.session.active_task_graph_id = getattr(existing_tg, 'id', '')
                    logger.info(
                        f"[IDEFCRunner] 已有活跃任务图（{len(uncompleted)} 未完成），复用")
                    return
                # 所有叶节点已完成 → 但如果是 RESUME 意图，仍复用旧图
                if state.is_resume:
                    if hasattr(existing_tg, 'id') and not self.session.active_task_graph_id:
                        self.session.active_task_graph_id = getattr(existing_tg, 'id', '')
                    logger.info(
                        f"[IDEFCRunner] RESUME 意图：旧图已完成但用户要求恢复，复用"
                        f"（graph_id={getattr(existing_tg, 'id', '?')}）")
                    return
                # 所有节点已完成 + 非 RESUME → 允许创建新 TG
                logger.info("[IDEFCRunner] 旧任务图已全部完成，创建新任务图")

            user_input = state.user_input_text
            if not user_input or len(user_input.strip()) < 5:
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
            logger.warning(f"[IDEFCRunner] 自动创建任务计划失败: {e}")

    def _auto_complete_task(self, state: IDEFCState) -> None:
        """FC 正常完成时自动标记任务节点为已完成

        包括 in_progress 和 pending 状态的用户任务叶节点
        （排除 CRG 自动注入的 crg_ 节点，它们已由后台线程标记为 completed）
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, _save_active_backup
            tg = get_active_task_graph()
            if not tg:
                return
            response = state.last_response_content or ""
            leaves = tg.get_leaf_nodes()
            marked = 0
            for leaf in leaves:
                # 跳过 CRG 自动注入节点
                if leaf.id.startswith("crg_"):
                    continue
                if leaf.status in ("in_progress", "pending"):
                    tg.update_node_status(
                        leaf.id, "completed",
                        result=response[:500] if response else "(IDE 会话已完成)",
                    )
                    marked += 1
            if marked > 0:
                try:
                    _save_active_backup()
                except Exception:
                    pass
                logger.info(f"[IDEFCRunner] 自动标记 {marked} 个任务节点为已完成")
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
                "intent": getattr(state, "ide_intent", ""),
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
            response = state.last_response_content
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
