"""核心 FC 循环执行器

纯 Python while 循环驱动的 FC 循环引擎，不依赖 LangGraph。
提供可覆盖的钩子方法，供 IDEFCRunner 等子类扩展。

与 fc_nodes.py 的关系:
- fc_nodes.py 提供4个节点工厂函数 (纯函数，无状态)
- fc_runner.py 用 while 循环调度这些工厂函数

用法:
    runner = FCRunner(engine)
    response, fc_turn = runner.run(
        messages, tools, model_id,
        initial_tool_calls_data=tool_calls,
    )

便捷函数:
    response, fc_turn = run_fc_loop(engine, messages, tools, model_id, ...)
"""

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from zulong.l2.inference_engine import InferenceEngine

from zulong.l2.circuit_breaker import CircuitBreakerState
from zulong.l2.attention_pressure_view import build_threshold_pressure_view
from zulong.l2.tool_capabilities import (
    filter_tools_by_capabilities,
    tool_capabilities,
)
from zulong.l2.tool_budget import (
    engine_tool_budget_exhausted,
    get_engine_tool_budget,
    get_engine_tool_calls_used,
    record_engine_tool_calls_used,
    sync_engine_tool_budget,
)

logger = logging.getLogger(__name__)


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


class FCRunner:
    """核心 FC 循环执行器

    子类可覆盖以下钩子方法以实现扩展:
    - _on_before_loop(state): 循环开始前
    - _on_before_check(state): check 前
    - _on_after_check(state, result): check 后
    - _on_before_model_call(state): 调用模型前
    - _on_after_model_call(state, tc_data, resp_content): 调用模型后
    - _on_before_exec_tools(state, tc_data): 执行工具前
    - _on_after_exec_tools(state): 执行工具后
    - _on_before_eval(state, resp_content): 评估前
    - _on_after_eval(state, verdict): 评估后
    - _on_loop_done(state, response, fc_turn): 循环完成
    """

    # 连续 response=None 拦截的安全上限
    _MAX_NULL_RESPONSES = 3
    _MAX_UNCOMPLETED_RETRIES = 2

    # 重复工具死循环检测阈值
    _DUPLICATE_TOOL_CHECK_TURNS = 3  # 连续N轮调用相同工具+相同参数判定为死循环

    # 进度停滞检测：连续N次报告无进展则终止
    _MAX_STALLED_REPORTS = 5

    def __init__(self, engine: "InferenceEngine"):
        self.engine = engine
        self._hard_limit = _safe_int(getattr(engine, "_hard_limit", 100), 100)
        self._soft_limit = _safe_int(getattr(engine, "_soft_limit", 50), 50)
        self._max_fc_turns = _safe_int(getattr(engine, "_max_fc_turns", 100), 100)
        self._step_limits_enabled = bool(getattr(engine, "_step_limits_enabled", True))
        self._warning_interval = max(1, _safe_int(getattr(engine, "_warning_interval", 10), 10))
        self._fc_request_interval = getattr(engine, "_fc_request_interval", 1.0)
        self._setup_nodes()

    def _setup_nodes(self):
        """初始化4个节点工厂函数"""
        from zulong.l2.fc_nodes import (
            _make_check_node,
            _make_call_model_node,
            _make_exec_tools_node,
            _make_eval_response_node,
        )
        self._check_fn = _make_check_node(self.engine)
        self._call_model_fn = _make_call_model_node(self.engine)
        self._exec_tools_fn = _make_exec_tools_node(self.engine)
        self._eval_response_fn = _make_eval_response_node(self.engine)

    # ── 节点执行封装：IDE/测试可复用同一套核心节点 ─────────────

    def check_state(self, state: Dict) -> Dict:
        """执行核心 check 节点并应用通用安全网。"""
        self._on_before_check(state)
        result = self._check_fn(state)
        self._detect_duplicate_tool_loop(state)
        self._on_after_check(state, result)
        return result

    def call_model_state(self, state: Dict) -> Dict:
        """执行核心 call_model 节点。"""
        self._on_before_model_call(state)
        result = self._call_model_fn(state)
        self._on_after_model_call(
            state,
            result.get("tool_calls_data"),
            result.get("response_content"),
        )
        return result

    def exec_tools_state(self, state: Dict) -> Dict:
        """执行核心 exec_tools 节点。"""
        self._on_before_exec_tools(state, state.get("tool_calls_data") or [])
        result = self._exec_tools_fn(state)
        self._on_after_exec_tools(state)
        return result

    def eval_response_state(self, state: Dict) -> Dict:
        """执行核心 eval_response 节点。"""
        self._on_before_eval(state, state.get("response_content", "") or "")
        result = self._eval_response_fn(state)
        self._on_after_eval(state, result.get("should_terminate", ""))
        return result

    def run(
        self,
        messages: List[Dict],
        tool_definitions: List[Dict],
        vllm_model_id: str,
        force_first_tool: bool = False,
        forced_first_tool_name: str = "",
        user_input: str = "",
        is_resume: bool = None,
        response_max_tokens: int = 1024,
        initial_tool_calls_data: Optional[List[Dict]] = None,
        initial_response_content: str = "",
    ) -> Tuple[Optional[str], int]:
        """执行工具调用循环，返回 (response, fc_turn)。

        普通 L2 直答不应进入本循环。主回答链会先做一次 L2 模型
        决策；只有该决策真实返回 tool_calls 时，才将这些调用作为
        initial_tool_calls_data 交给本 runner 执行工具闭环。
        """
        # 重置 RuleGuardian 计数器
        if hasattr(self.engine, '_rule_guardian'):
            self.engine._rule_guardian.reset()
        sync_engine_tool_budget(self.engine, user_input)

        # 组装初始状态
        state: Dict = {
            "messages": messages,
            "_all_tool_definitions": list(tool_definitions or []),
            "fc_turn": 0,
            "response": None,
            "tool_results_buffer": [],
            "cb_force_no_tools": False,
            "gap_continue_count": 0,
            "should_terminate": "",
            "tool_calls_data": None,
            "response_content": None,
            "force_first_tool": force_first_tool,
            "forced_first_tool_name": forced_first_tool_name,
            "forced_next_tool_name": "",
            "vllm_model_id": vllm_model_id,
            "tool_definitions": tool_definitions,
            "user_input_text": user_input,
            "is_resume": (
                is_resume if is_resume is not None else force_first_tool
            ),
            "resume_automark_count": 0,
            "null_response_count": 0,
            "api_timeout_count": 0,
            "response_max_tokens": response_max_tokens,
            "pressure_force_attention": False,
            "pressure_stage": "",
            "pressure_attention_context": {},
            "pressure_recovery_note_saved": False,
            "pressure_recovery_attention_switched": False,
            "pressure_recovery_requires_note": True,
            "pressure_recovery_requires_attention": True,
            "pressure_recovery_start_result_count": 0,
            "_last_pressure_tier": "green",
            # 进度停滞检测跟踪
            "_progress_snapshots": [],
            "_stalled_reports": 0,
        }

        # 钩子: 循环开始前
        self._on_before_loop(state)

        if not initial_tool_calls_data:
            logger.warning(
                "[FCRunner] 未提供真实工具调用，跳过工具循环；"
                "普通 L2 直答应使用单次模型决策路径"
            )
            self.engine._last_fc_terminate_reason = "no_tool_call"
            self._on_loop_done(state, None, 0)
            return None, 0

        initial_tool_calls_data = self._apply_tool_call_budget(
            state,
            initial_tool_calls_data,
        )
        if not initial_tool_calls_data:
            self.engine._last_fc_terminate_reason = "tool_budget_exhausted"
            self._on_loop_done(state, None, 0)
            return None, 0

        max_iterations = None if not self._step_limits_enabled else self._hard_limit + 15

        logger.info(
            f"[FCRunner] 开始工具循环: "
            f"tools={len(tool_definitions)}, model={vllm_model_id}, "
            f"initial_tool_calls={len(initial_tool_calls_data or [])}, "
            f"step_limits_enabled={self._step_limits_enabled}, "
            f"hard_limit={self._hard_limit if self._step_limits_enabled else 'disabled'}"
        )

        try:
            if initial_tool_calls_data:
                state["fc_turn"] = 1
                state["tool_calls_data"] = initial_tool_calls_data
                state["response_content"] = initial_response_content or ""
                logger.info(
                    "[FCRunner] 执行首批工具调用: %s",
                    [
                        tc.get("function", {}).get("name", "")
                        for tc in initial_tool_calls_data
                    ],
                )
                result = self.exec_tools_state(state)
                state.update(result)
                if state.get("should_terminate"):
                    response = state.get("response")
                    fc_turn = state.get("fc_turn", 0)
                    self._on_loop_done(state, response, fc_turn)
                    return response, fc_turn

            loop_iter = 0
            while max_iterations is None or loop_iter < max_iterations:
                loop_iter += 1
                # ── Phase 1: Check ──
                result = self.check_state(state)
                state.update(result)
                if state.get("should_terminate"):
                    break

                # ── Phase 2: Call Model ──
                result = self.call_model_state(state)
                state.update(result)
                if state.get("should_terminate"):
                    break

                # 超时重试（tc_data 和 response_content 都为 None）
                if (state.get("tool_calls_data") is None
                        and state.get("response_content") is None):
                    continue

                # ── Phase 3a: Exec Tools (有工具调用) ──
                if state.get("tool_calls_data"):
                    limited = self._apply_tool_call_budget(
                        state,
                        state.get("tool_calls_data") or [],
                    )
                    state["tool_calls_data"] = limited
                    if not limited:
                        continue
                    clears_uncompleted_retry = self._tool_calls_show_real_progress(limited)
                    result = self.exec_tools_state(state)
                    state.update(result)
                    if state.get("should_terminate"):
                        break
                    if clears_uncompleted_retry:
                        state["null_response_count"] = 0
                        state["_uncompleted_retry_cycles"] = 0
                    continue  # 回到 check

                # ── Phase 3b: Eval Response (纯文本回复) ──
                result = self.eval_response_state(state)
                state.update(result)
                if state.get("should_terminate"):
                    break

                # 回复被拦截（Rule A / InfoGap / AutoMark）
                if state.get("response") is None:
                    null_count = state.get("null_response_count", 0)
                    if null_count >= self._MAX_NULL_RESPONSES:
                        if self._has_uncompleted_task_graph():
                            retries = state.get("_uncompleted_retry_cycles", 0) + 1
                            state["_uncompleted_retry_cycles"] = retries
                            if retries > self._MAX_UNCOMPLETED_RETRIES:
                                response = self._block_uncompleted_task_graph(state)
                                state["response"] = response
                                state["should_terminate"] = "uncompleted_retry_exhausted"
                                logger.warning(
                                    "[FCRunner] 未完成任务图连续重试无进展，标记 blocked 并终止"
                                )
                                break
                            state["null_response_count"] = 0
                            state["cb_force_no_tools"] = False
                            self._inject_continue_uncompleted_task(state)
                            logger.warning(
                                f"[FCRunner] 连续 {null_count} 次拦截，但任务图仍未完成，"
                                "重置拦截计数并继续工具循环"
                            )
                            continue
                        logger.warning(
                            f"[FCRunner] 连续 {null_count} 次拦截，"
                            f"超过安全上限 ({self._MAX_NULL_RESPONSES})，"
                            f"强制终止"
                        )
                        break
                    continue
                # response 不为 None 且 should_terminate 未设置 → 不应到达
                break

        except Exception as e:
            err_name = type(e).__name__
            logger.exception(f"[FCRunner] 循环异常 ({err_name}): {e}")
            self.engine._last_fc_terminate_reason = "exception"
            fallback_turn = state.get("fc_turn", 0)
            # 从 messages 中恢复最后一条 assistant 回复
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and len(content) > 10:
                        logger.info(
                            f"[FCRunner] 从 messages 恢复最后回复，"
                            f"长度={len(content)}"
                        )
                        self._on_loop_done(state, content, fallback_turn)
                        return content, fallback_turn
            fallback = self.engine._get_fallback_response(user_input)
            self._on_loop_done(state, fallback, fallback_turn)
            return fallback, fallback_turn

        response = state.get("response")
        fc_turn = state.get("fc_turn", 0)

        self.engine._last_fc_terminate_reason = state.get("should_terminate", "")
        self._on_loop_done(state, response, fc_turn)

        logger.info(
            f"[FCRunner] 工具循环完成: "
            f"共 {fc_turn} 轮, response={'有' if response else '无'}"
        )
        return response, fc_turn

    # ── 通用安全网 ──────────────────────────────────────────

    def _apply_tool_call_budget(
        self,
        state: Dict,
        tool_calls_data: List[Dict],
    ) -> List[Dict]:
        """Enforce explicit user tool-call budgets before execution."""
        if not tool_calls_data:
            return []
        budget = get_engine_tool_budget(self.engine)
        if budget is None:
            return tool_calls_data
        used = get_engine_tool_calls_used(self.engine)
        remaining = max(0, budget - used)
        if remaining <= 0:
            self._inject_tool_budget_convergence(state, budget, used)
            return []
        allowed = tool_calls_data[:remaining]
        skipped = len(tool_calls_data) - len(allowed)
        record_engine_tool_calls_used(self.engine, len(allowed))
        if skipped > 0 or engine_tool_budget_exhausted(self.engine):
            self._inject_tool_budget_convergence(
                state,
                budget,
                used + len(allowed),
                skipped=skipped,
            )
        return allowed

    @staticmethod
    def _tool_calls_show_real_progress(tool_calls_data: List[Dict]) -> bool:
        """Return True when a tool call is likely to advance task state.

        Read-only probes should not reset the "unfinished task" retry guard.
        This keeps a complex task visibly blocked when the model keeps
        inspecting or marking "in_progress" without producing files/results.
        """
        progress_tools = {
            "ide_write_file",
            "exec_write_file",
            "write_to_file",
            "replace_in_file",
            "delete_file",
            "create_directory",
            "exec_run_command",
            "execute_command",
            "task_create_plan",
            "task_add_node",
            "task_update_node",
            "task_update_content",
            "task_attach_file",
            "task_resume_by_address",
            "task_suspend",
            "save_memory_note",
            "delete_memory_node",
            "delete_memory_edge",
            "set_importance",
            "activate_memory_network",
            "submit_final_answer",
        }
        for tc in tool_calls_data or []:
            function = tc.get("function") or {}
            name = function.get("name") or ""
            if name in progress_tools:
                return True
            if name == "task_mark_status":
                raw_args = function.get("arguments") or "{}"
                try:
                    import json
                    args = json.loads(raw_args)
                except Exception:
                    args = {}
                if str(args.get("status") or "").lower() in {
                    "completed",
                    "blocked",
                    "skipped",
                    "failed",
                }:
                    return True
        return False

    @staticmethod
    def _inject_tool_budget_convergence(
        state: Dict,
        budget: int,
        used: int,
        *,
        skipped: int = 0,
    ) -> None:
        note = (
            f"[工具预算硬控] 用户要求本轮最多调用 {budget} 个工具；"
            f"当前已允许执行 {used} 个。"
        )
        if skipped > 0:
            note += f" 已拦截 {skipped} 个超额工具调用。"
        note += " 请基于已有工具结果和上下文直接总结，不允许继续调用工具。"
        state["cb_force_no_tools"] = True
        state.setdefault("messages", []).append({
            "role": "user",
            "content": note,
        })
        state["tool_calls_data"] = None

    def _detect_duplicate_tool_loop(self, state: Dict) -> None:
        """重复工具调用死循环检测

        检查最近几轮中是否连续调用相同的工具且参数相同。
        如果检测到死循环，注入 CB 强制收敛信号。
        """
        fc = state["fc_turn"]
        messages = state.get("messages", [])
        if len(messages) < 6 or fc <= 5:
            return

        last_tool_calls = []
        for msg in reversed(messages[-6:]):
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
                    f"[FCRunner] 检测到死循环: "
                    f"连续{len(last_tool_calls)}轮调用 {tool_names[0]} 且参数相同"
                )
                state["cb_force_no_tools"] = True
                state["cb_recovery_stage"] = "restricted_recovery"
                state.setdefault("cb_recovery_note_saved", False)
                state.setdefault("cb_recovery_attention_switched", False)
                state["cb_recovery_start_result_count"] = len(state.get("tool_results_buffer") or [])
                cm = {
                    "role": "user",
                    "content": (
                        f"[系统警告] 检测到重复工具调用循环（{tool_names[0]}），"
                        f"请先把当前证据、未完成项、失败原因写入便签并切换注意力；"
                        f"如果仍无法继续，只能输出 partial/blocked summary，不得伪完成。"
                    ),
                }
                state["messages"].append(cm)

    def _detect_progress_stall(self, state: Dict) -> bool:
        """进度停滞检测

        通过采样工具结果缓冲区的大小和消息列表长度来判断是否有进展。
        返回 True 表示检测到停滞。
        """
        _snapshots = state.setdefault("_progress_snapshots", [])
        fc = state["fc_turn"]

        # 每 _warning_interval 轮采样一次
        if fc % self._warning_interval != 0 or fc < self._warning_interval:
            return False

        current_snapshot = {
            "fc_turn": fc,
            "tool_results_count": len(state.get("tool_results_buffer", [])),
            "messages_count": len(state.get("messages", [])),
        }
        _snapshots.append(current_snapshot)

        # 只保留最近 N 次采样
        if len(_snapshots) > self._MAX_STALLED_REPORTS:
            _snapshots = _snapshots[-self._MAX_STALLED_REPORTS:]
            state["_progress_snapshots"] = _snapshots

        if len(_snapshots) < self._MAX_STALLED_REPORTS:
            return False

        recent = _snapshots[-self._MAX_STALLED_REPORTS:]
        # 检查所有指标是否有变化
        tc_counts = [s["tool_results_count"] for s in recent]
        msg_counts = [s["messages_count"] for s in recent]

        if len(set(tc_counts)) == 1 and len(set(msg_counts)) == 1:
            state["_stalled_reports"] = state.get("_stalled_reports", 0) + 1
            logger.warning(
                f"[FCRunner] 进度停滞检测: 最近{self._MAX_STALLED_REPORTS}次采样无进展"
            )
            if state["_stalled_reports"] >= self._MAX_STALLED_REPORTS:
                logger.warning(
                    f"[FCRunner] 连续{state['_stalled_reports']}次停滞报告，"
                    f"强制终止"
                )
                return True
        else:
            state["_stalled_reports"] = 0

        return False

    _CB_RESTRICTED_RECOVERY_CAPABILITIES = {
        "attention_switch",
        "note_anchor",
        "memory_persist",
        "tag_anchor",
    }
    _CB_RESTRICTED_EXCLUDED_CAPABILITIES = {
        "file_write",
        "verification",
    }

    def evaluate_circuit_breaker(
        self,
        state: Dict,
        *,
        attn_usage_ratio: float = -1.0,
    ) -> Tuple[bool, str, str]:
        """统一 CB 评估与提示注入。

        Returns:
            (force_no_tools, state_name, reason)
        """
        cb = getattr(self.engine, "_circuit_breaker", None)
        if cb is None:
            return False, "green", ""
        cb_state, cb_reason = cb.evaluate(
            state.get("fc_turn", 0),
            state.get("messages", []),
            attn_usage_ratio=attn_usage_ratio,
        )
        if cb_state == CircuitBreakerState.RED:
            state["cb_force_no_tools"] = True
            state["cb_recovery_stage"] = "restricted_recovery"
            state["cb_recovery_note_saved"] = False
            state["cb_recovery_attention_switched"] = False
            state["cb_recovery_start_result_count"] = len(state.get("tool_results_buffer") or [])
            convergence = {
                "role": "user",
                "content": (
                    f"[Circuit Breaker RED 受限恢复] {cb_reason}\n"
                    "请先把当前证据、未完成项、失败原因、下一步建议写入便签/标签/记忆并关联当前任务节点，"
                    "再进行一次 GLOBAL / FOCUS / SINGLE_CHAIN 注意力重选；"
                    "本轮只允许使用便签/标签/记忆落盘和注意力切换能力。"
                ),
            }
            state.setdefault("messages", []).append(convergence)
            return True, cb_state.value, cb_reason
        if cb_state == CircuitBreakerState.YELLOW:
            hint = {
                "role": "user",
                "content": (
                    f"[Circuit Breaker 警告] {cb_reason}\n"
                    "请尽快总结当前信息并回复用户，避免继续调用更多工具。"
                ),
            }
            state.setdefault("messages", []).append(hint)
            return False, cb_state.value, cb_reason
        return False, cb_state.value, cb_reason

    # ── 钩子方法（子类可覆盖）─────────────────────────────────

    def _on_before_loop(self, state: Dict) -> None:
        """钩子: 循环开始前"""
        pass

    def _on_before_check(self, state: Dict) -> None:
        """钩子: check 前"""
        pass

    def _on_after_check(self, state: Dict, result: Dict) -> None:
        """钩子: check 后

        通用安全网: 检查进度停滞
        """
        if self._detect_progress_stall(state):
            if self._has_uncompleted_task_graph():
                result["should_terminate"] = ""
                state["cb_force_no_tools"] = False
                state.setdefault("messages", []).append({
                    "role": "user",
                    "content": (
                        "[系统提示] 当前任务图仍有未完成节点。"
                        "请继续调用工具推进任务，不要把阶段性进度当作最终回答。"
                    ),
                })
                logger.warning("[FCRunner] 检测到进度停滞，但任务图未完成，继续工具循环")
                return
            result["should_terminate"] = "progress_stalled"
            state["response"] = self.engine._get_fallback_response(
                state.get("user_input_text", "")
            )

    def _on_before_model_call(self, state: Dict) -> None:
        """钩子: 调用模型前

        通用安全网: CB RED 时保留收敛工具
        """
        self._apply_pressure_guidance(state)
        if self._is_pressure_restricted_recovery(state):
            self._enforce_pressure_restricted_recovery(state)
        if self._is_cb_restricted_recovery(state):
            recovery_tools = self._get_cb_recovery_tools(state.get("tool_definitions") or [])
            if recovery_tools:
                state["tool_definitions"] = recovery_tools
                state["cb_force_no_tools"] = False
                state["_pressure_limited_tools"] = True
                forced = self._next_restricted_recovery_tool(state, recovery_tools)
                state["forced_next_tool_name"] = forced
                state.setdefault("messages", []).append({
                    "role": "user",
                    "content": (
                        "[Circuit Breaker RED 受限恢复工具集]\n"
                        "普通执行/搜索/写文件/命令/验证工具已临时收走；当前仅保留具备"
                        "“便签/标签/记忆落盘”或“注意力切换”能力的工具。请先保存现场"
                        "并锚定当前节点，再完成一次"
                        " GLOBAL / FOCUS / SINGLE_CHAIN 注意力重选。"
                    ),
                })
                logger.warning(
                    "[FCRunner][CB] RED 使用受限恢复工具集: %s",
                    [
                        td.get("function", {}).get("name", "")
                        for td in recovery_tools
                    ],
                )

    def _on_after_model_call(
        self,
        state: Dict,
        tool_calls_data: Optional[List[Dict]],
        response_content: Optional[str],
    ) -> None:
        """钩子: 调用模型后"""
        pass

    def _on_before_exec_tools(
        self, state: Dict, tool_calls_data: List[Dict]
    ) -> None:
        """钩子: 执行工具前"""
        pass

    def _on_after_exec_tools(self, state: Dict) -> None:
        """钩子: 执行工具后"""
        self._update_cb_restricted_recovery_progress(state)
        self._update_pressure_restricted_recovery_progress(state)
        self._restore_full_tool_definitions_if_limited(state)

    def _on_before_eval(self, state: Dict, response_content: str) -> None:
        """钩子: 评估前"""
        pass

    def _on_after_eval(self, state: Dict, verdict: str) -> None:
        """钩子: 评估后"""
        pass

    def _on_loop_done(
        self, state: Dict, response: Optional[str], fc_turn: int
    ) -> None:
        """钩子: 循环完成"""
        pass

    def _apply_pressure_guidance(self, state: Dict) -> None:
        """Web/Core FC pressure policy aligned with the IDE runner.

        Pressure does not compress context. It injects LLM-owned guidance to
        choose GLOBAL / FOCUS / SINGLE_CHAIN.  The first threshold hit always
        injects guidance only; if pressure is still RED on a later pass, the
        runner enters restricted recovery and temporarily narrows available
        tools to note/tag/memory landing plus attention-switch capabilities.
        """
        attn_window = getattr(self.engine, "_attn_window", None)
        if not attn_window or state.get("cb_force_no_tools"):
            return
        if self._is_cb_restricted_recovery(state):
            return
        if self._is_pressure_restricted_recovery(state):
            self._enforce_pressure_restricted_recovery(state)
            return
        if not state.get("_all_tool_definitions"):
            state["_all_tool_definitions"] = list(state.get("tool_definitions") or [])

        try:
            ratio = float(
                getattr(
                    attn_window,
                    "trigger_context_pressure_ratio",
                    getattr(
                        attn_window,
                        "context_pressure_ratio",
                        getattr(attn_window, "usage_ratio", 0.0),
                    ),
                )
                or 0.0
            )
        except Exception:
            ratio = 0.0
        yellow_ratio = 0.90
        red_ratio = 1.0
        threshold_budget_ratio = 0.5
        try:
            attn_cfg = getattr(attn_window, "_llm_config", None)
            if attn_cfg:
                yellow_ratio = float(getattr(attn_cfg, "pressure_threshold_medium", yellow_ratio))
                red_ratio = float(getattr(attn_cfg, "pressure_threshold_high", red_ratio))
                threshold_budget_ratio = float(getattr(attn_cfg, "threshold_budget_ratio", threshold_budget_ratio))
            else:
                cb = getattr(self.engine, "_circuit_breaker", None)
                cb_cfg = getattr(cb, "_config", {}) or {}
                yellow_ratio = float(cb_cfg.get("context_yellow_ratio", yellow_ratio))
                red_ratio = float(cb_cfg.get("context_red_ratio", red_ratio))
                threshold_budget_ratio = float(cb_cfg.get("threshold_budget_ratio", threshold_budget_ratio))
        except Exception:
            pass

        if ratio > red_ratio:
            tier = "red"
        elif ratio > yellow_ratio:
            tier = "yellow"
        else:
            tier = "green"
        pressure_view = build_threshold_pressure_view(
            ratio,
            yellow_ratio,
            red_ratio,
            tier=tier,
        )

        state["_last_pressure_tier"] = tier
        if tier == "green":
            if state.get("pressure_stage") not in {"restricted_recovery"}:
                state["pressure_stage"] = ""
                state["pressure_force_attention"] = False
            return

        fc = int(state.get("fc_turn", 0) or 0)
        # TSD v2.9.22 两段式：即使 yellow/red 阈值配置相同，第一次
        # 达阈值也只能提醒 LLM 自主选择注意力，不得立刻收走普通工具。
        if state.get("pressure_stage") not in {"yellow_guidance", "restricted_recovery"}:
            hint = {
                "role": "system",
                "content": (
                    "[上下文压力 - 注意力引导] "
                    f"当前上下文压力已达 {pressure_view.threshold_relative_percent:.0f}%。"
                    "请由 LLM 自主判断是否需要切换 GLOBAL / FOCUS / SINGLE_CHAIN；"
                    "动态注意力不是压缩上下文，而是重新选择当前必要上下文，暂排/降权无关上下文；"
                    "若需要收窄关注范围，可调用注意力切换能力，例如聚焦当前子任务"
                    "或深入关键节点。普通读写/命令工具结果只作为证据，不直接决定模式。"
                ),
            }
            state.setdefault("messages", []).append(hint)
            try:
                attn_window.register_message(hint, turn=fc)
            except Exception:
                pass
            logger.info(
                "[FCRunner][Pressure] YELLOW %.0f%% of threshold budget: 第一次阈值响应，仅注入动态注意力引导",
                pressure_view.threshold_relative_percent,
            )
            state["pressure_stage"] = "yellow_guidance"
            state["pressure_attention_context"] = {
                "tier": tier,
                "ratio": round(ratio, 4),
                "yellow_ratio": round(float(yellow_ratio), 4),
                "red_ratio": round(float(red_ratio), 4),
                "context_pressure_ratio": round(pressure_view.context_pressure_ratio, 4),
                "context_pressure_percent": round(pressure_view.context_pressure_percent, 1),
                "threshold_budget_ratio": round(float(threshold_budget_ratio), 4),
                "threshold_budget_percent": round(pressure_view.threshold_relative_percent, 1),
                "active_threshold_ratio": round(pressure_view.active_threshold_ratio, 4),
                "budget_reference": "threshold_budget_is_100_percent",
                "fc_turn": fc,
                "decision_owner": "llm",
                "allowed_modes": ["GLOBAL", "FOCUS", "SINGLE_CHAIN"],
                "first_threshold_response": "guidance_only",
            }
            return

        if tier != "red":
            return

        self._enter_pressure_restricted_recovery(
            state,
            ratio=ratio,
            yellow_ratio=yellow_ratio,
            red_ratio=red_ratio,
            threshold_budget_ratio=threshold_budget_ratio,
            pressure_view=pressure_view,
            fc=fc,
            attn_window=attn_window,
        )

    def _enter_pressure_restricted_recovery(
        self,
        state: Dict,
        *,
        ratio: float,
        yellow_ratio: float,
        red_ratio: float,
        threshold_budget_ratio: float,
        pressure_view,
        fc: int,
        attn_window,
    ) -> None:
        """Enter the second pressure response stage.

        This is deliberately capability based.  It never binds policy to one
        concrete tool name; the selected tool names are only the runtime
        providers of the required capabilities.
        """
        recovery_tools = self._get_cb_recovery_tools(
            state.get("_all_tool_definitions") or state.get("tool_definitions") or []
        )
        requires_note = bool(
            self._first_recovery_landing_tool(recovery_tools)
        )
        requires_attention = bool(
            self._first_tool_with_capability(recovery_tools, "attention_switch")
        )
        state["pressure_stage"] = "restricted_recovery" if recovery_tools else "guidance_only"
        state["pressure_force_attention"] = bool(recovery_tools)
        state["pressure_recovery_note_saved"] = False
        state["pressure_recovery_attention_switched"] = False
        state["pressure_recovery_requires_note"] = requires_note
        state["pressure_recovery_requires_attention"] = requires_attention
        state["pressure_recovery_start_result_count"] = len(state.get("tool_results_buffer") or [])
        state["pressure_attention_context"] = {
            "tier": "red",
            "ratio": round(ratio, 4),
            "yellow_ratio": round(float(yellow_ratio), 4),
            "red_ratio": round(float(red_ratio), 4),
            "context_pressure_ratio": round(pressure_view.context_pressure_ratio, 4),
            "context_pressure_percent": round(pressure_view.context_pressure_percent, 1),
            "threshold_budget_ratio": round(float(threshold_budget_ratio), 4),
            "threshold_budget_percent": round(pressure_view.threshold_relative_percent, 1),
            "active_threshold_ratio": round(pressure_view.active_threshold_ratio, 4),
            "budget_reference": "threshold_budget_is_100_percent",
            "fc_turn": fc,
            "decision_owner": "llm",
            "allowed_modes": ["GLOBAL", "FOCUS", "SINGLE_CHAIN"],
            "second_threshold_response": "restricted_recovery" if recovery_tools else "guidance_only",
            "requires_note": requires_note,
            "requires_attention": requires_attention,
        }
        hint = {
            "role": "system",
            "content": (
                f"[动态注意力重选] 当前上下文压力达到 {pressure_view.threshold_relative_percent:.0f}%。\n"
                "你必须基于当前任务、未覆盖节点、证据状态和必要上下文，"
                "自主选择一次注意力动作：\n"
                "- GLOBAL：需要重新注入全局任务/证据上下文时选择；\n"
                "- FOCUS：需要聚焦当前局部节点或局部失败时选择；\n"
                "- SINGLE_CHAIN：需要沿当前执行链连续推进时选择。\n"
                "请说明本次要注入哪些必要上下文、暂排哪些无关上下文、"
                "以及何时回到全局。"
            ),
        }
        state.setdefault("messages", []).append(hint)
        try:
            attn_window.register_message(hint, turn=fc)
        except Exception:
            pass

        if recovery_tools:
            self._enforce_pressure_restricted_recovery(state)
            logger.info(
                "[FCRunner][Pressure] RED %.0f%% of threshold budget: 第二次阈值响应，进入受限恢复，仅保留便签/标签/记忆落盘+注意力切换能力工具 %s",
                pressure_view.threshold_relative_percent,
                [
                    td.get("function", {}).get("name", "")
                    for td in recovery_tools
                ],
            )
        else:
            logger.info(
                "[FCRunner][Pressure] RED %.0f%%: 未找到受限恢复能力工具，仅注入动态注意力重选提醒",
                ratio * 100,
            )

    def _enforce_pressure_restricted_recovery(self, state: Dict) -> None:
        recovery_tools = self._get_cb_recovery_tools(
            state.get("_all_tool_definitions") or state.get("tool_definitions") or []
        )
        if not recovery_tools:
            state["pressure_force_attention"] = False
            state["pressure_stage"] = "guidance_only"
            return
        state["tool_definitions"] = recovery_tools
        state["_pressure_limited_tools"] = True
        state["pressure_force_attention"] = True
        forced_recovery = self._next_pressure_recovery_tool(state, recovery_tools)
        if forced_recovery:
            state["forced_next_tool_name"] = forced_recovery

    @staticmethod
    def _get_attention_only_tools(tool_definitions: List[Dict]) -> List[Dict]:
        return filter_tools_by_capabilities(tool_definitions, {"attention_switch"})

    @classmethod
    def _get_cb_recovery_tools(cls, tool_definitions: List[Dict]) -> List[Dict]:
        retained: List[Dict] = []
        for td in tool_definitions or []:
            caps = tool_capabilities(td)
            if caps & cls._CB_RESTRICTED_RECOVERY_CAPABILITIES:
                if not (caps & cls._CB_RESTRICTED_EXCLUDED_CAPABILITIES):
                    retained.append(td)
        return retained

    @staticmethod
    def _first_tool_with_capability(
        tool_definitions: List[Dict],
        capability: str,
    ) -> str:
        for td in tool_definitions or []:
            if capability in tool_capabilities(td):
                name = str(td.get("function", {}).get("name", "") or "").strip()
                if name:
                    return name
        return ""

    @staticmethod
    def _first_tool_with_any_capability(
        tool_definitions: List[Dict],
        capabilities: set,
    ) -> str:
        wanted = {str(cap or "").strip() for cap in capabilities if str(cap or "").strip()}
        for td in tool_definitions or []:
            if tool_capabilities(td) & wanted:
                name = str(td.get("function", {}).get("name", "") or "").strip()
                if name:
                    return name
        return ""

    @classmethod
    def _first_recovery_landing_tool(cls, tool_definitions: List[Dict]) -> str:
        for capability in ("note_anchor", "memory_persist", "tag_anchor"):
            name = cls._first_tool_with_capability(tool_definitions, capability)
            if name:
                return name
        return ""

    @classmethod
    def _next_restricted_recovery_tool(
        cls,
        state: Dict,
        tool_definitions: List[Dict],
    ) -> str:
        if not state.get("cb_recovery_note_saved"):
            return cls._first_recovery_landing_tool(tool_definitions)
        if not state.get("cb_recovery_attention_switched"):
            return cls._first_tool_with_capability(tool_definitions, "attention_switch")
        return ""

    @classmethod
    def _next_pressure_recovery_tool(
        cls,
        state: Dict,
        tool_definitions: List[Dict],
    ) -> str:
        if (
            state.get("pressure_recovery_requires_note", True)
            and not state.get("pressure_recovery_note_saved")
        ):
            return cls._first_recovery_landing_tool(tool_definitions)
        if (
            state.get("pressure_recovery_requires_attention", True)
            and not state.get("pressure_recovery_attention_switched")
        ):
            return cls._first_tool_with_capability(tool_definitions, "attention_switch")
        return ""

    @staticmethod
    def _is_cb_restricted_recovery(state: Dict) -> bool:
        return state.get("cb_recovery_stage") in {
            "restricted_recovery",
            "note_attention",  # 兼容旧会话状态
        }

    @staticmethod
    def _is_pressure_restricted_recovery(state: Dict) -> bool:
        return state.get("pressure_stage") == "restricted_recovery"

    def _update_cb_restricted_recovery_progress(self, state: Dict) -> None:
        if not self._is_cb_restricted_recovery(state):
            return
        start_index = int(state.get("cb_recovery_start_result_count", 0) or 0)
        tool_names = [
            str(item.get("tool_name") or "").strip()
            for item in (state.get("tool_results_buffer", []) or [])[start_index:]
            if str(item.get("tool_name") or "").strip()
        ]
        if not tool_names:
            return
        caps_by_name = {
            str(td.get("function", {}).get("name", "") or "").strip(): tool_capabilities(td)
            for td in state.get("_all_tool_definitions") or state.get("tool_definitions") or []
        }
        if any(
            caps_by_name.get(name, set()) & {"note_anchor", "memory_persist", "tag_anchor"}
            for name in tool_names
        ):
            state["cb_recovery_note_saved"] = True
        if any(
            "attention_switch" in caps_by_name.get(name, set())
            for name in tool_names
        ):
            state["cb_recovery_attention_switched"] = True
        if state.get("cb_recovery_note_saved") and state.get("cb_recovery_attention_switched"):
            state["cb_recovery_stage"] = ""
            state["cb_force_no_tools"] = False

    def _update_pressure_restricted_recovery_progress(self, state: Dict) -> None:
        if not self._is_pressure_restricted_recovery(state):
            return
        start_index = int(state.get("pressure_recovery_start_result_count", 0) or 0)
        tool_names = [
            str(item.get("tool_name") or "").strip()
            for item in (state.get("tool_results_buffer", []) or [])[start_index:]
            if str(item.get("tool_name") or "").strip()
        ]
        if not tool_names:
            return
        caps_by_name = {
            str(td.get("function", {}).get("name", "") or "").strip(): tool_capabilities(td)
            for td in state.get("_all_tool_definitions") or state.get("tool_definitions") or []
        }
        if any(
            caps_by_name.get(name, set()) & {"note_anchor", "memory_persist", "tag_anchor"}
            for name in tool_names
        ):
            state["pressure_recovery_note_saved"] = True
        if any(
            "attention_switch" in caps_by_name.get(name, set())
            for name in tool_names
        ):
            state["pressure_recovery_attention_switched"] = True

        requires_note = bool(state.get("pressure_recovery_requires_note", True))
        requires_attention = bool(state.get("pressure_recovery_requires_attention", True))
        note_done = state.get("pressure_recovery_note_saved") or not requires_note
        attention_done = state.get("pressure_recovery_attention_switched") or not requires_attention
        if note_done and attention_done:
            state["pressure_stage"] = ""
            state["pressure_force_attention"] = False
            ctx = dict(state.get("pressure_attention_context") or {})
            ctx.update({
                "completed": True,
                "resolved_at_result_count": len(state.get("tool_results_buffer") or []),
            })
            state["pressure_attention_context"] = ctx

    @staticmethod
    def _restore_full_tool_definitions_if_limited(state: Dict) -> None:
        if state.get("cb_recovery_stage") in {"restricted_recovery", "note_attention"}:
            return
        if state.get("pressure_stage") == "restricted_recovery":
            return
        if not state.get("_pressure_limited_tools"):
            return
        full_tools = state.get("_all_tool_definitions") or []
        if full_tools:
            state["tool_definitions"] = full_tools
        state["_pressure_limited_tools"] = False

    @staticmethod
    def _inject_continue_uncompleted_task(state: Dict) -> None:
        next_label = ""
        next_desc = ""
        try:
            from zulong.tools.task_tools import get_active_task_graph, _save_active_backup

            tg = get_active_task_graph()
            if tg:
                leaves = [
                    node for node in tg.get_leaf_nodes()
                    if not getattr(node, "id", "").startswith("crg_")
                ]
                uncompleted = [
                    node for node in leaves
                    if getattr(node, "status", "") not in ("completed", "skipped")
                ]
                if uncompleted:
                    current = next(
                        (
                            node for node in uncompleted
                            if getattr(node, "status", "") == "in_progress"
                        ),
                        uncompleted[0],
                    )
                    if getattr(current, "status", "") != "in_progress":
                        tg.update_node_status(current.id, "in_progress")
                        try:
                            _save_active_backup()
                        except Exception:
                            pass
                    next_label = f"{current.id}({current.label})"
                    next_desc = current.desc or current.label
        except Exception:
            next_label = ""
            next_desc = ""

        detail = (
            f" 当前应执行: {next_label}。{next_desc}"
            if next_label else ""
        )
        state.setdefault("messages", []).append({
            "role": "user",
            "content": (
                "[任务图继续执行] 当前任务图仍有未完成节点。"
                "请继续调用真实工具推进任务，不要直接总结或只输出进度句。"
                "如果当前节点要求文件产出，下一步必须调用 ide_write_file 或相应写入工具真实落盘。"
                f"{detail}"
            ),
        })

    @staticmethod
    def _block_uncompleted_task_graph(state: Dict) -> str:
        next_label = ""
        detail = "模型多轮没有继续调用真实工具，任务被标记为 blocked，等待用户或后续恢复。"
        try:
            from zulong.tools.task_tools import get_active_task_graph, _save_active_backup

            tg = get_active_task_graph()
            if tg:
                leaves = [
                    node for node in tg.get_leaf_nodes()
                    if not getattr(node, "id", "").startswith("crg_")
                ]
                uncompleted = [
                    node for node in leaves
                    if getattr(node, "status", "") not in ("completed", "skipped")
                ]
                current = next(
                    (
                        node for node in uncompleted
                        if getattr(node, "status", "") == "in_progress"
                    ),
                    uncompleted[0] if uncompleted else None,
                )
                if current is not None:
                    next_label = f"{current.id}({current.label})"
                    detail = (
                        f"节点 {next_label} 连续多轮没有产生真实工具调用或文件产出，"
                        "已标记为 blocked。"
                    )
                    try:
                        tg.update_node_status(current.id, "blocked", result=detail)
                        _save_active_backup()
                    except Exception:
                        pass
        except Exception:
            pass

        return (
            "任务已阻断：当前任务图仍有未完成节点，但模型连续多轮没有继续调用真实工具。"
            + (f"\n阻断节点：{next_label}。" if next_label else "")
            + f"\n原因：{detail}"
        )

    @staticmethod
    def _has_uncompleted_task_graph() -> bool:
        try:
            from zulong.tools.task_tools import get_active_task_graph

            tg = get_active_task_graph()
            if not tg:
                return False
            leaves = [
                node for node in tg.get_leaf_nodes()
                if not getattr(node, "id", "").startswith("crg_")
            ]
            if not leaves:
                return False
            return any(
                getattr(node, "status", "") not in ("completed", "skipped")
                for node in leaves
            )
        except Exception:
            return False


def run_fc_loop(
    engine: "InferenceEngine",
    messages: List[Dict],
    tool_definitions: List[Dict],
    vllm_model_id: str,
    force_first_tool: bool = False,
    forced_first_tool_name: str = "",
    user_input: str = "",
    is_resume: bool = None,
    response_max_tokens: int = 1024,
    initial_tool_calls_data: Optional[List[Dict]] = None,
    initial_response_content: str = "",
) -> Tuple[Optional[str], int]:
    """执行工具调用循环，返回 (response, fc_turn)。

    签名与原 fc_graph.run_fc_loop() 完全一致，作为直接替代。
    内部使用 FCRunner while 循环驱动，不依赖 LangGraph。
    新主链应仅在模型已经返回真实 tool_calls 后调用本函数。

    Args:
        engine: InferenceEngine 实例
        messages: OpenAI 格式对话消息列表（可变引用，循环内直接 append）
        tool_definitions: 工具定义列表
        vllm_model_id: 远程模型 ID
        force_first_tool: 继续已有任务图时第一轮是否强制 task_view_overview
        forced_first_tool_name: 指定首轮强制调用的工具名，优先级高于 force_first_tool
        user_input: 用户原始输入（用于降级回复）
        is_resume: 是否为继续已有任务图的执行场景
        response_max_tokens: 最大生成 token 数
        initial_tool_calls_data: 单次 L2 决策已经返回的首批工具调用
        initial_response_content: 首批工具调用对应 assistant content

    Returns:
        (response, fc_turn) -- response 可能为 None
    """
    runner = FCRunner(engine)
    return runner.run(
        messages=messages,
        tool_definitions=tool_definitions,
        vllm_model_id=vllm_model_id,
        force_first_tool=force_first_tool,
        forced_first_tool_name=forced_first_tool_name,
        user_input=user_input,
        is_resume=is_resume,
        response_max_tokens=response_max_tokens,
        initial_tool_calls_data=initial_tool_calls_data,
        initial_response_content=initial_response_content,
    )
