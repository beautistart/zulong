"""统一 FC 循环执行器

将 fc_graph.py 的 LangGraph StateGraph 转换为普通 while 循环，
消除 LangGraph 运行时依赖，保留全部认知能力：
- AttentionWindow 消息裁剪
- CircuitBreaker 6 信号降级
- RuleGuardian 过早完成拦截
- InfoGap 信息缺口重试
- RESUME Auto-Mark 安全网
- COMPLEX Backfill 回填
- MemoryGraph BFS 扩散激活 + 焦点漂移

用法:
    runner = UnifiedFCRunner(engine)
    response, fc_turn = runner.run(messages, tools, model_id, ...)

兼容函数:
    response, fc_turn = run_fc_loop(engine, messages, tools, model_id, ...)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UnifiedFCRunner:
    """统一 FC 循环执行器

    复用 fc_graph.py 的节点工厂函数（经过充分验证的认知逻辑），
    以 while 循环驱动，无需 LangGraph 依赖。
    """

    # 连续 response=None 拦截的安全上限
    _MAX_NULL_RESPONSES = 3

    def __init__(self, engine: "InferenceEngine"):
        self.engine = engine

        # 从 fc_graph.py 导入节点工厂并实例化
        from zulong.l2.fc_graph import (
            _make_check_node,
            _make_call_model_node,
            _make_exec_tools_node,
            _make_eval_response_node,
        )
        self._check_fn = _make_check_node(engine)
        self._call_model_fn = _make_call_model_node(engine)
        self._exec_tools_fn = _make_exec_tools_node(engine)
        self._eval_response_fn = _make_eval_response_node(engine)

    def run(
        self,
        messages: List[Dict],
        tool_definitions: List[Dict],
        vllm_model_id: str,
        force_first_tool: bool = False,
        user_input: str = "",
        is_resume: bool = None,
        intent_max_tokens: int = 1024,
    ) -> Tuple[Optional[str], int]:
        """执行 FC 循环，返回 (response, fc_turn)。

        参数与 fc_graph.run_fc_loop() 完全一致。
        """
        # 重置 RuleGuardian 计数器
        if hasattr(self.engine, '_rule_guardian'):
            self.engine._rule_guardian.reset()

        # 组装初始状态（与 FCLoopState 字段一致）
        state: Dict = {
            "messages": messages,
            "fc_turn": 0,
            "response": None,
            "tool_results_buffer": [],
            "cb_force_no_tools": False,
            "gap_continue_count": 0,
            "should_terminate": "",
            "tool_calls_data": None,
            "response_content": None,
            "force_first_tool": force_first_tool,
            "vllm_model_id": vllm_model_id,
            "tool_definitions": tool_definitions,
            "user_input_text": user_input,
            "is_resume": (
                is_resume if is_resume is not None else force_first_tool
            ),
            "resume_automark_count": 0,
            "null_response_count": 0,
            "api_timeout_count": 0,
            "intent_max_tokens": intent_max_tokens,
        }

        hard_limit = getattr(self.engine, "_hard_limit", 100)
        max_iterations = hard_limit + 15  # 绝对安全上限

        logger.info(
            f"[UnifiedFC] 开始 FC 循环: "
            f"tools={len(tool_definitions)}, model={vllm_model_id}, "
            f"hard_limit={hard_limit}"
        )

        try:
            for _ in range(max_iterations):
                # ── Phase 1: Check ──
                result = self._check_fn(state)
                state.update(result)
                if state.get("should_terminate"):
                    break

                # ── Phase 2: Call Model ──
                result = self._call_model_fn(state)
                state.update(result)
                if state.get("should_terminate"):
                    break

                # 超时重试（tc_data 和 response_content 都为 None）
                if (state.get("tool_calls_data") is None
                        and state.get("response_content") is None):
                    continue

                # ── Phase 3a: Exec Tools (有工具调用) ──
                if state.get("tool_calls_data"):
                    result = self._exec_tools_fn(state)
                    state.update(result)
                    if state.get("should_terminate"):
                        break
                    continue  # 回到 check

                # ── Phase 3b: Eval Response (纯文本回复) ──
                result = self._eval_response_fn(state)
                state.update(result)
                if state.get("should_terminate"):
                    break

                # 回复被拦截（Rule A / InfoGap / AutoMark）
                if state.get("response") is None:
                    null_count = state.get("null_response_count", 0)
                    if null_count >= self._MAX_NULL_RESPONSES:
                        logger.warning(
                            f"[UnifiedFC] 连续 {null_count} 次拦截，"
                            f"超过安全上限 ({self._MAX_NULL_RESPONSES})，"
                            f"强制终止"
                        )
                        break
                    continue
                # response 不为 None 且 should_terminate 未设置 → 不应到达
                break

        except Exception as e:
            err_name = type(e).__name__
            logger.error(f"[UnifiedFC] 循环异常 ({err_name}): {e}")
            # 从 messages 中恢复最后一条 assistant 回复
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and len(content) > 10:
                        logger.info(
                            f"[UnifiedFC] 从 messages 恢复最后回复，"
                            f"长度={len(content)}"
                        )
                        return content, hard_limit
            return (
                self.engine._get_fallback_response(user_input),
                hard_limit,
            )

        response = state.get("response")
        fc_turn = state.get("fc_turn", 0)

        # 记录终止原因到 engine，供 Orchestrator 感知 API 错误
        self.engine._last_fc_terminate_reason = state.get("should_terminate", "")

        logger.info(
            f"[UnifiedFC] FC 循环完成: "
            f"共 {fc_turn} 轮, response={'有' if response else '无'}"
        )
        return response, fc_turn


def run_fc_loop(
    engine: "InferenceEngine",
    messages: List[Dict],
    tool_definitions: List[Dict],
    vllm_model_id: str,
    force_first_tool: bool = False,
    user_input: str = "",
    is_resume: bool = None,
    intent_max_tokens: int = 1024,
) -> Tuple[Optional[str], int]:
    """执行统一 FC 循环，返回 (response, fc_turn)。

    签名与 fc_graph.run_fc_loop() 完全一致，作为直接替代。
    内部使用 while 循环驱动，不依赖 LangGraph。

    Args:
        engine: InferenceEngine 实例
        messages: OpenAI 格式对话消息列表（可变引用，循环内直接 append）
        tool_definitions: 工具定义列表
        vllm_model_id: 远程模型 ID
        force_first_tool: RESUME 场景第一轮是否强制 task_view_overview
        user_input: 用户原始输入（用于降级回复）
        is_resume: 是否为任务恢复场景
        intent_max_tokens: 最大生成 token 数

    Returns:
        (response, fc_turn) -- response 可能为 None
    """
    runner = UnifiedFCRunner(engine)
    return runner.run(
        messages=messages,
        tool_definitions=tool_definitions,
        vllm_model_id=vllm_model_id,
        force_first_tool=force_first_tool,
        user_input=user_input,
        is_resume=is_resume,
        intent_max_tokens=intent_max_tokens,
    )
