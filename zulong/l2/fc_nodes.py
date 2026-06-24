"""
FC Loop 节点工厂函数

从 fc_graph.py 提取，仅保留4个节点工厂函数和辅助函数。
移除了 LangGraph StateGraph 构建代码、路由函数、checkpointer 等已废弃组件。

这4个工厂函数被以下模块复用：
1. fc_runner.py — 统一FC运行器（while循环模式）
2. ide_fc_runner.py — IDE模式FC运行器（继承FCRunner）

复用内容包括：
- _make_check_node, _make_call_model_node, _make_exec_tools_node, _make_eval_response_node
- FCLoopState TypedDict
- 辅助函数: _is_filler_content, _has_content_match, _extract_node_content
"""

import concurrent.futures
import asyncio as _asyncio
import json as _json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from zulong.l2.inference_engine import InferenceEngine

from zulong.l2.circuit_breaker import CircuitBreakerState
from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
from zulong.l2.smart_degradation_handler import TimeoutPhase
from zulong.l2.tool_budget import (
    engine_tool_budget_exhausted,
    get_engine_tool_budget,
    get_engine_tool_calls_used,
)
from zulong.core.message_visibility import (
    internal_control_message,
    strip_llm_message_metadata,
)

logger = logging.getLogger(__name__)


_PROTECTED_MUTATION_TOOLS = {
    "ide_write_file",
    "write_to_file",
    "replace_in_file",
    "delete_file",
    "execute_command",
    "create_directory",
}

_LOCAL_MUTATION_TOOLS = {
    "exec_write_file",
    "exec_run_command",
}

_WRITE_CONTENT_TOOLS = {
    "exec_write_file",
    "ide_write_file",
    "write_to_file",
}

_APPROVAL_DENIAL_MARKERS = (
    "用户未",
    "用户拒绝",
    "审批拒绝",
    "审批超时",
    "审批未通过",
    "审批未完成",
    "未允许",
    "未应用",
    "未真实存在",
    "尚未受信任",
    "workspace_trust",
    "approval_blocked",
)


def _looks_like_approval_denial_text(result_text: str) -> bool:
    raw = str(result_text or "")
    if not raw:
        return False
    lowered = raw.lower()
    return any(marker.lower() in lowered for marker in _APPROVAL_DENIAL_MARKERS)


def _tool_result_failed(result_text: str) -> bool:
    """Return True when a tool result is structurally or textually failed."""
    raw = str(result_text or "").strip()
    if not raw:
        return False
    try:
        parsed = _json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        nested = " ".join(
            str(parsed.get(key) or "")
            for key in ("result", "error", "message", "reason")
        )
        if _looks_like_approval_denial_text(nested):
            return True
        if parsed.get("ok") is True or parsed.get("success") is True:
            return False
        if parsed.get("ok") is False or parsed.get("success") is False:
            return True
        if parsed.get("error"):
            return True
    head = raw[:240].lower()
    return (
        '"error"' in head
        or " error" in head
        or head.startswith("error")
        or "失败" in head
        or "异常" in head
    )


def _is_protected_mutation_block(tool_name: str, result_text: str) -> bool:
    """Detect write/command approval denials.

    TSD requires approval to be an execution gate. A denial must prevent
    bypassing through local mutation tools, but it should not end the whole FC
    loop immediately: the model may retry the same approved channel after the
    user changes approval mode, or ask for approval explicitly.
    """
    if tool_name not in _PROTECTED_MUTATION_TOOLS:
        return False
    raw = str(result_text or "")
    if not raw:
        return False
    return _looks_like_approval_denial_text(raw)


def _approval_block_response(tool_name: str, result_text: str) -> str:
    preview = " ".join(str(result_text or "").split())[:260]
    return (
        "任务正在等待用户审批/应用受保护的写入或命令"
        f"（工具：{tool_name}）。"
        "祖龙不会改用内部写文件或命令工具绕过审批；确认后可继续同一路径执行。"
        + (f"\n结果摘要：{preview}" if preview else "")
    )


def _tool_args_parse_error_result(tool_name: str, parse_error: str) -> str:
    payload = {
        "error": f"工具参数解析失败: {parse_error}",
        "recoverable": True,
        "reason": "tool_arguments_json_invalid",
        "tool_name": tool_name,
    }
    if tool_name in _WRITE_CONTENT_TOOLS:
        payload.update({
            "chunk_policy": "adaptive_file_chunking",
            "recommended_chunk_chars": "800-1200",
            "next_action": (
                "重新调用同一个写入工具；优先完整写入。若模型输出容易被截断，"
                "可第一块 mode='overwrite'，后续块 mode='append'，并按当前模型输出能力自适应分片。"
            ),
        })
    return _json.dumps(payload, ensure_ascii=False)


def _build_write_chunking_correction(tool_name: str, parse_error: str) -> Dict[str, Any]:
    return internal_control_message(
        "[工具参数分片纠偏] 上一次写文件工具调用的 arguments 不是合法 JSON，"
        f"通常是 content 过长导致模型输出被 max_tokens 截断。工具={tool_name}；"
        f"解析错误={parse_error}。"
        "下一轮必须重新调用同一个写入工具；优先完整写入。"
        "若模型输出容易被截断，再采用自适应分片："
        "第一块 mode='overwrite'，后续块 mode='append'，分片大小按当前模型输出能力决定。"
        "不要继续输出大段自然语言，不要 submit_final_answer；全部写入后再运行验证命令或读取文件确认。"
    )


# ---------------------------------------------------------------------------
# 1. FCLoopState — 节点间传递的状态袋
# ---------------------------------------------------------------------------

class FCLoopState(TypedDict, total=False):
    """FC Loop 的图状态（TypedDict 确保 LangGraph 正确合并部分更新）。

    total=False 使所有字段可选，节点只需返回要更新的字段，
    未返回的字段由 LangGraph 自动保留上一轮的值。
    messages 列表通过引用共享，节点内直接 append 即可。
    """
    messages: List[Dict]
    fc_turn: int
    response: Optional[str]
    tool_results_buffer: List[Dict]
    cb_force_no_tools: bool
    gap_continue_count: int
    should_terminate: str
    tool_calls_data: Optional[List[Dict]]
    response_content: Optional[str]
    force_first_tool: bool
    vllm_model_id: str
    tool_definitions: List[Dict]
    user_input_text: str
    is_resume: bool
    forced_first_tool_name: str
    forced_next_tool_name: str
    resume_automark_count: int
    null_response_count: int  # 连续 response=None 的拦截次数
    api_timeout_count: int  # API 连续超时次数
    response_max_tokens: int


# ---------------------------------------------------------------------------
# 2. 节点工厂函数（闭包捕获 engine 实例）
# ---------------------------------------------------------------------------

def _make_check_node(engine: "InferenceEngine"):
    """节点 1: 前置检查与步数递增"""

    def check_node(state: dict) -> dict:
        fc_turn = state["fc_turn"] + 1
        state["fc_turn"] = fc_turn

        step_limits_enabled = bool(getattr(engine, "_step_limits_enabled", True))
        max_turns = getattr(engine, "_max_fc_turns", 0)
        soft_limit = getattr(engine, "_soft_limit", 0)
        hard_limit = getattr(engine, "_hard_limit", 0)

        # 进度监控
        if fc_turn % engine._warning_interval == 0:
            limit_text = f"/{max_turns}" if step_limits_enabled and max_turns else ""
            logger.info(
                f"[FC][Graph] 进度: {fc_turn}{limit_text} 步，"
                f"已执行 {len(state['tool_results_buffer'])} 次工具调用"
            )

        if step_limits_enabled and soft_limit and fc_turn > soft_limit:
            logger.warning(f"[FC][Graph] ⚠️ 已超过软限制 ({soft_limit} 步)，继续执行...")

        # 硬限制检查
        if step_limits_enabled and hard_limit and fc_turn >= hard_limit:
            logger.error(f"[FC][Graph] 🚨 达到硬限制 ({hard_limit} 步)，强制终止")
            return {"fc_turn": fc_turn, "should_terminate": "hard_limit"}

        # 中断信号检查
        is_interrupt_requested = getattr(engine, "_is_interrupt_requested", None)
        if callable(is_interrupt_requested):
            interrupted = bool(is_interrupt_requested())
        else:
            with engine._lock:
                interrupted = engine._interrupt_flag
        if interrupted:
            logger.info(f"[FC][Graph] Turn {fc_turn}: 检测到中断信号，终止 FC 循环")
            # 保留最后一个有效回复而非空字符串
            if not state.get("response"):
                # 从消息历史中提取最后的 assistant 回复
                last_reply = ""
                for msg in reversed(state.get("messages", [])):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content and len(content) > 10:
                            last_reply = content
                            break
                state["response"] = last_reply
            return {"fc_turn": fc_turn, "should_terminate": "interrupt"}

        return {"fc_turn": fc_turn, "should_terminate": ""}

    return check_node


def _make_call_model_node(engine: "InferenceEngine"):
    """节点 2: LLM API 调用"""

    def call_model_node(state: dict) -> dict:
        fc_turn = state["fc_turn"]
        messages = state["messages"]
        cb_force_no_tools = state.get("cb_force_no_tools", False)
        tool_definitions = state["tool_definitions"]
        vllm_model_id = state["vllm_model_id"]
        force_first_tool = state.get("force_first_tool", False)
        forced_first_tool_name = state.get("forced_first_tool_name")
        forced_next_tool_name = state.get("forced_next_tool_name")

        # 构建 API 调用参数（使用注意力窗口裁剪后的消息）
        windowed_messages = (
            engine._attn_window.apply_window()
            if engine._attn_window else strip_llm_message_metadata(messages)
        )
        api_kwargs: Dict[str, Any] = {
            "model": vllm_model_id,
            "messages": windowed_messages,
            "max_tokens": state.get("response_max_tokens", 1024),
            "temperature": 0.3,
            "top_p": 0.85,
            "stream": False,
            **engine._get_llm_extra_kwargs(),
        }

        # 传入工具定义
        tool_budget = get_engine_tool_budget(engine)
        tool_used = get_engine_tool_calls_used(engine)
        if tool_budget is not None and tool_used >= tool_budget:
            logger.info(
                "[FC][Graph] 工具预算已用尽: used=%s budget=%s，移除工具定义",
                tool_used,
                tool_budget,
            )
        elif cb_force_no_tools:
            logger.info("[FC][Graph][CB] Circuit Breaker RED: 强制文本回复，移除所有工具定义")
            try:
                _has_uncompleted, _next_node = _task_graph_uncompleted_context()
                if (
                    _has_uncompleted
                    and not engine_tool_budget_exhausted(engine)
                    and state.get("cb_recovery_stage") not in {"restricted_recovery", "note_attention"}
                ):
                    cb_force_no_tools = False
                    state["cb_force_no_tools"] = False
                    api_kwargs["tools"] = tool_definitions
                    if forced_next_tool_name and _set_forced_tool_choice(
                        api_kwargs, tool_definitions, forced_next_tool_name
                    ):
                        state["forced_next_tool_name"] = ""
                        logger.warning(
                            "[FC][Graph][CB] 任务图未完成，强制恢复工具执行: %s -> %s",
                            getattr(_next_node, "id", ""),
                            forced_next_tool_name,
                        )
                    else:
                        api_kwargs["tool_choice"] = "auto"
                    logger.warning(
                        "[FC][Graph][CB] 任务图仍有未完成节点，恢复工具定义供模型继续执行: %s",
                        getattr(_next_node, "id", ""),
                    )
            except Exception as cb_restore_err:
                logger.debug("[FC][Graph][CB] 恢复工具定义检查失败: %s", cb_restore_err)
        elif tool_definitions:
            api_kwargs["tools"] = tool_definitions
            if forced_next_tool_name and _set_forced_tool_choice(
                api_kwargs, tool_definitions, forced_next_tool_name
            ):
                state["forced_next_tool_name"] = ""
                logger.info("[FC][Graph] 纠偏轮：强制调用 %s", forced_next_tool_name)
            elif forced_first_tool_name and fc_turn == 1:
                api_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": forced_first_tool_name},
                }
                logger.info(
                    "[FC][Graph] 第一轮：强制调用 %s",
                    forced_first_tool_name,
                )
            elif force_first_tool and fc_turn == 1:
                api_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "task_view_overview"},
                }
                logger.info("[FC][Graph] 继续任务图策略第一轮：强制调用 task_view_overview")
            else:
                api_kwargs["tool_choice"] = "auto"

        # API 调用（含超时）
        # FC 请求间隔：防止 API 被打满（跳过第一轮）
        _req_interval = getattr(engine, "_fc_request_interval", 1.0)
        if fc_turn > 1 and _req_interval > 0:
            time.sleep(_req_interval)

        def _call(kwargs=api_kwargs):
            return engine.vllm_client.chat.completions.create(**kwargs)

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(_call)
            wait_with_interrupt = getattr(engine, "_wait_future_with_interrupt", None)
            if callable(wait_with_interrupt):
                api_response = wait_with_interrupt(
                    future,
                    engine._fc_loop_timeout,
                    label=f"[FC][Graph] Turn {fc_turn}",
                )
            else:
                api_response = future.result(timeout=engine._fc_loop_timeout)
        except InterruptedError:
            logger.info(f"[FC][Graph] Turn {fc_turn}: 模型调用被用户停止")
            return {
                "response": state.get("response") or state.get("response_content") or "",
                "should_terminate": "interrupt",
                "tool_calls_data": None,
                "response_content": None,
            }
        except concurrent.futures.TimeoutError:
            logger.warning(
                f"⚠️ [FC][Graph] Turn {fc_turn} 超时 (>{engine._fc_loop_timeout}s)，继续尝试..."
            )
            api_timeout_count = state.get("api_timeout_count", 0) + 1
            _MAX_API_TIMEOUTS = 1  # 超时 1 次即切换备用模型，避免用户长时间等待

            if bool(getattr(engine, "_step_limits_enabled", True)) and fc_turn >= engine._hard_limit:
                logger.error("[FC][Graph] 🚨 超时且达到硬限制，使用降级回复")
                return {
                    "response": engine._get_fallback_response(state.get("user_input_text", "")),
                    "should_terminate": "api_error",
                    "tool_calls_data": None,
                    "response_content": None,
                    "api_timeout_count": api_timeout_count,
                }
            if api_timeout_count >= _MAX_API_TIMEOUTS:
                logger.warning(
                    f"[FC][Graph] 连续 {api_timeout_count} 次超时，尝试备用模型"
                )
                try:
                    from zulong.models.container import LLM_MODEL_ID_BACKUP
                    if engine.backup_client and LLM_MODEL_ID_BACKUP:
                        backup_resp = engine.backup_client.chat.completions.create(
                            model=LLM_MODEL_ID_BACKUP,
                            messages=strip_llm_message_metadata(messages),
                            max_tokens=state.get("response_max_tokens", 1024),
                            temperature=0.3,
                            stream=False,
                            **engine._get_llm_extra_kwargs(),
                        )
                        return {
                            "response": backup_resp.choices[0].message.content or "",
                            "should_terminate": "backup_model",
                            "tool_calls_data": None,
                            "response_content": None,
                            "api_timeout_count": api_timeout_count,
                        }
                except Exception as backup_err:
                    logger.warning(f"[FC][Graph] 备用模型也失败: {backup_err}")
                return {
                    "response": engine._get_fallback_response(state.get("user_input_text", "")),
                    "should_terminate": "api_error",
                    "tool_calls_data": None,
                    "response_content": None,
                    "api_timeout_count": api_timeout_count,
                }
            # 超时重试 → 路由回 check
            return {
                "tool_calls_data": None,
                "response_content": None,
                "should_terminate": "",
                "api_timeout_count": api_timeout_count,
            }
        except Exception as api_err:
            logger.error(f"🚨 [FC][Graph] Turn {fc_turn} API 调用失败: {api_err}")
            # 尝试备用模型
            try:
                from zulong.models.container import LLM_MODEL_ID_BACKUP
                if engine.backup_client and LLM_MODEL_ID_BACKUP:
                    logger.info(f"🔄 [FC][Graph] 切换备用模型: {LLM_MODEL_ID_BACKUP}")
                    backup_resp = engine.backup_client.chat.completions.create(
                        model=LLM_MODEL_ID_BACKUP,
                        messages=strip_llm_message_metadata(messages),
                        max_tokens=1024,
                        temperature=0.3,
                        stream=False,
                        **engine._get_llm_extra_kwargs(),
                    )
                    return {
                        "response": backup_resp.choices[0].message.content or "",
                        "should_terminate": "api_error",
                        "tool_calls_data": None,
                        "response_content": None,
                    }
                else:
                    return {
                        "response": engine._get_fallback_response(state.get("user_input_text", "")),
                        "should_terminate": "api_error",
                        "tool_calls_data": None,
                        "response_content": None,
                    }
            except Exception as backup_err:
                logger.warning(f"🚨 [FC][Graph] 备用模型也失败: {backup_err}")
                return {
                    "response": engine._get_fallback_response(state.get("user_input_text", "")),
                    "should_terminate": "api_error",
                    "tool_calls_data": None,
                    "response_content": None,
                }
        finally:
            executor.shutdown(wait=False)

        choice = api_response.choices[0]
        msg = choice.message

        # 拆解 API 返回（避免将 OpenAI 对象存入 state）
        tool_calls_data = None
        response_content = msg.content or ""
        raw_tool_calls = getattr(msg, "tool_calls", None) or []
        finish_reason = getattr(choice, "finish_reason", None)
        usage_summary = _summarize_llm_usage(getattr(api_response, "usage", None))
        tool_call_names = [
            str(getattr(getattr(tc, "function", None), "name", "") or "")
            for tc in raw_tool_calls
        ]
        logger.info(
            "[FC][Graph][LLMReturn] turn=%s model=%s finish_reason=%r "
            "content_len=%s content_preview=%r tool_calls=%s tool_names=%s usage=%s",
            fc_turn,
            vllm_model_id,
            finish_reason,
            len(response_content),
            _compact_line(response_content, 240),
            len(raw_tool_calls),
            tool_call_names,
            usage_summary,
        )
        if not response_content and not raw_tool_calls:
            logger.warning(
                "[FC][Graph][EmptyLLMReturn] turn=%s model=%s finish_reason=%r "
                "usage=%s messages=%s tools=%s tool_choice=%r",
                fc_turn,
                vllm_model_id,
                finish_reason,
                usage_summary,
                len(windowed_messages),
                len(tool_definitions or []),
                api_kwargs.get("tool_choice"),
            )

        if raw_tool_calls:
            logger.info(
                f"[FC][Graph] Turn {fc_turn}: 模型请求 {len(raw_tool_calls)} 个工具调用"
            )
            tool_calls_data = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_tool_calls
            ]
        else:
            try:
                from zulong.ide.ide_format_translator import (
                    IDEFormatTranslator,
                    strip_xml_tool_call_markup,
                )

                xml_tool_calls = IDEFormatTranslator.parse_xml_tool_calls(response_content)
            except Exception as xml_err:
                xml_tool_calls = []
                logger.debug("[FC][Graph] XML/DSML 工具调用回退解析失败: %s", xml_err)
            if xml_tool_calls:
                tool_calls_data = xml_tool_calls
                response_content = strip_xml_tool_call_markup(response_content)
                logger.warning(
                    "[FC][Graph] Turn %s: 从模型文本中恢复 %s 个 XML/DSML 工具调用: %s",
                    fc_turn,
                    len(xml_tool_calls),
                    [tc.get("function", {}).get("name", "") for tc in xml_tool_calls],
                )
            else:
                logger.info(
                    f"[FC][Graph] Turn {fc_turn}: 模型直接回复，长度 {len(response_content)}"
                )

        return {
            "tool_calls_data": tool_calls_data,
            "response_content": response_content,
            "should_terminate": "",
        }

    return call_model_node


def _make_exec_tools_node(engine: "InferenceEngine"):
    """节点 3: 工具执行"""

    def exec_tools_node(state: dict) -> dict:
        fc_turn = state["fc_turn"]
        messages = state["messages"]
        tool_calls_data = state["tool_calls_data"]
        tool_results_buffer = state["tool_results_buffer"]
        response_content = state.get("response_content", "")

        # 将 assistant 消息（含 tool_calls）追加到上下文
        assistant_msg = {
            "role": "assistant",
            "content": response_content or "",
            "tool_calls": tool_calls_data,
        }
        grp = engine._attn_window.new_tool_group() if engine._attn_window else None
        messages.append(assistant_msg)
        if engine._attn_window:
            engine._attn_window.register_message(
                assistant_msg, turn=fc_turn, group_id=grp,
            )

        if response_content:
            try:
                engine._publish_task_graph_event(
                    "model_progress",
                    fc_turn,
                    "",
                    response_content,
                    f"model-step-{fc_turn}",
                )
            except Exception:
                logger.debug("[FC][Graph] model_progress 生命周期事件发布失败", exc_info=True)

        # 执行每个工具调用
        for tc_data in tool_calls_data:
            # 中断检查
            with engine._lock:
                interrupted = engine._interrupt_flag
            if interrupted:
                logger.info("[FC][Graph] 工具执行中检测到中断，跳过剩余工具")
                break

            tool_name = tc_data["function"]["name"]
            tool_args = {}
            parse_error = ""
            try:
                tool_args = _json.loads(tc_data["function"]["arguments"] or "{}")
            except Exception as exc:
                parse_error = str(exc)

            if parse_error:
                logger.error(
                    "[FC][Graph] 工具 %s 参数解析失败: %s",
                    tool_name,
                    parse_error,
                )
                result_text = _tool_args_parse_error_result(tool_name, parse_error)
                engine._circuit_breaker.record_call(
                    tool_name,
                    {"_parse_error": parse_error},
                    result_text,
                )
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc_data["id"],
                    "content": result_text,
                }
                messages.append(tool_msg)
                if engine._attn_window:
                    engine._attn_window.register_message(
                        tool_msg,
                        turn=fc_turn,
                        tool_name=tool_name,
                        group_id=grp,
                    )
                tool_results_buffer.append({
                    "tool_name": tool_name,
                    "reason": _extract_tool_call_reason(response_content, tool_name),
                    "result": result_text,
                })
                engine._publish_task_graph_event(
                    "agent_tool_call",
                    fc_turn,
                    tool_name,
                    result_text,
                    tc_data.get("id", ""),
                )
                if tool_name in _WRITE_CONTENT_TOOLS:
                    correction = _build_write_chunking_correction(
                        tool_name,
                        parse_error,
                    )
                    messages.append(correction)
                    if engine._attn_window:
                        engine._attn_window.register_message(
                            correction,
                            turn=fc_turn,
                            group_id=grp,
                        )
                    state["forced_next_tool_name"] = tool_name
                continue

            # 注意力窗口：观察工具调用
            if engine._attn_window:
                engine._attn_window.observe_tool_call(tool_name, tool_args)
                if tool_name == "navigate_attention":
                    engine._attn_window.on_navigate_attention(
                        direction=tool_args.get("direction", ""),
                        target_node_id=tool_args.get("target_node_id"),
                    )
                # P2-14: task_mark_status完成后自动导航注意力
                if tool_name == "task_mark_status":
                    new_status = tool_args.get("status", "")
                    node_id = tool_args.get("node_id", "")
                    if new_status == "completed" and node_id:
                        try:
                            engine._attn_window.auto_navigate_on_status_change(
                                node_id, new_status
                            )
                        except Exception:
                            pass

            # 构造类 tool_call 对象供 _execute_tool_call 使用
            class _ToolCallProxy:
                """轻量代理，模拟 OpenAI ChatCompletionMessageToolCall 接口"""
                def __init__(self, data):
                    self.id = data["id"]
                    self.type = data["type"]
                    self.function = type("F", (), {
                        "name": data["function"]["name"],
                        "arguments": data["function"]["arguments"],
                    })()

            tc_proxy = _ToolCallProxy(tc_data)
            engine._publish_task_graph_event(
                "agent_tool_call", fc_turn, tool_name, "", tc_data.get("id", ""),
            )
            result_text = engine._execute_tool_call(tc_proxy)

            if tool_name in ("request_tool_supplement", "search_tools"):
                _merge_discovered_tool_schemas(state, result_text)

            # Circuit Breaker: 记录工具调用
            engine._circuit_breaker.record_call(tool_name, tool_args, result_text)
            if tool_name in ("task_create_plan", "start_task_plan"):
                engine._circuit_breaker.escalate_for_planning()

            # 工具结果截断保护
            if len(result_text) > MAX_TOOL_RESULT_CHARS:
                orig_len = len(result_text)
                result_text = (
                    result_text[:MAX_TOOL_RESULT_CHARS]
                    + f"\n...(已截断，原始长度 {orig_len} 字符)"
                )
                logger.debug(
                    f"[FC][Graph] 工具 {tool_name} 结果截断: {orig_len} → {MAX_TOOL_RESULT_CHARS}"
                )

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": result_text,
            }
            messages.append(tool_msg)
            if engine._attn_window:
                engine._attn_window.register_message(
                    tool_msg, turn=fc_turn,
                    tool_name=tool_name,
                    node_id=tool_args.get("node_id") or tool_args.get("target_node_id"),
                    group_id=grp,
                )

            logger.info(f"[FC][Graph] 工具 {tool_name} 结果: {result_text[:200]}")
            tool_results_buffer.append({
                "tool_name": tool_name,
                "reason": _extract_tool_call_reason(response_content, tool_name),
                "result": result_text,
            })

            # 发布任务图谱更新事件
            engine._publish_task_graph_event(
                "agent_tool_call", fc_turn, tool_name, result_text, tc_data.get("id", ""),
            )

            if _is_protected_mutation_block(tool_name, result_text):
                logger.warning(
                    "[FC][Graph] 受保护变更工具未获应用，禁止本地绕过但保持FC可恢复: %s",
                    tool_name,
                )
                setattr(engine, "_approval_block_active", True)
                messages.append(internal_control_message(
                    "[审批边界] 上一次受保护写入/命令未被用户应用。"
                    "不要改用 exec_write_file 或 exec_run_command 绕过审批；"
                    "如果用户已经开启自动审批，继续使用同一个 IDE/受保护工具重试；"
                    "否则向用户说明需要审批后再继续。"
                ))

        # Circuit Breaker: 本轮所有工具执行完毕，评估状态
        cb_state, cb_reason = engine._circuit_breaker.evaluate(fc_turn, messages)

        cb_force_no_tools = False
        forced_next_tool_name = ""
        if cb_state == CircuitBreakerState.RED:
            logger.warning(f"[FC][Graph][CB] RED 触发 (turn={fc_turn}): {cb_reason}")
            # 记录降级阶段为 CB 触发，而非超时（避免 SmartDegradation 误报"主模型响应超时"）
            engine._last_timeout_phase = TimeoutPhase.CIRCUIT_BREAKER_TRIPPED
            engine._last_timeout_elapsed = 0.0
            _has_uncompleted, _next_node = _task_graph_uncompleted_context()
            if _has_uncompleted and not engine_tool_budget_exhausted(engine):
                cb_convergence, forced_next_tool_name = _build_uncompleted_task_correction(
                    f"Circuit Breaker 触发：{cb_reason}",
                    tag="[Circuit Breaker纠偏]",
                    tool_definitions=state.get("tool_definitions"),
                )
                logger.warning(
                    "[FC][Graph][CB] 任务图未完成，RED 转为工具纠偏: %s",
                    getattr(_next_node, "id", ""),
                )
            else:
                cb_force_no_tools = True
                cb_convergence = internal_control_message(
                    f"[Circuit Breaker RED 受限恢复] {cb_reason}\n"
                    "不要把未完成任务伪装成完成。请先把当前证据、未完成项、失败原因写成便签并关联当前任务节点，"
                    "再进行一次注意力重选；若仍无法继续，只能输出 partial/blocked summary。"
                )
            messages.append(cb_convergence)
            if engine._attn_window:
                engine._attn_window.register_message(cb_convergence, turn=fc_turn)

        elif cb_state == CircuitBreakerState.YELLOW:
            logger.warning(f"[FC][Graph][CB] YELLOW 警告 (turn={fc_turn}): {cb_reason}")
            _has_uncompleted, _next_node = _task_graph_uncompleted_context()
            if _has_uncompleted and not engine_tool_budget_exhausted(engine):
                cb_hint, forced_next_tool_name = _build_uncompleted_task_correction(
                    f"Circuit Breaker 警告：{cb_reason}",
                    tag="[Circuit Breaker纠偏]",
                    tool_definitions=state.get("tool_definitions"),
                )
                logger.warning(
                    "[FC][Graph][CB] 任务图未完成，YELLOW 转为工具纠偏: %s",
                    getattr(_next_node, "id", ""),
                )
            else:
                cb_hint = internal_control_message(
                    f"[Circuit Breaker 警告] {cb_reason}\n"
                    "请尽快总结当前信息并回复用户，避免继续调用更多工具。"
                )
            messages.append(cb_hint)
            if engine._attn_window:
                engine._attn_window.register_message(cb_hint, turn=fc_turn)

        # ── MemoryGraph: BFS 扩散激活（惰性触发）──
        # BFS扩散仅在以下情况执行：
        # 1. 首次执行（_bfs_first_run=False）
        # 2. LLM主动调用记忆检索工具（_bfs_memory_triggered=True）
        # 3. 动态注意力触发（_bfs_attention_triggered=True）
        _bfs_first_run = state.get("_bfs_first_run", False)
        _bfs_memory_triggered = state.get("_bfs_memory_triggered", False)
        _bfs_attention_triggered = state.get("_bfs_attention_triggered", False)
        
        # 检查全局BFS触发标记（来自记忆工具）
        try:
            from zulong.tools.memory_graph_tools import get_bfs_memory_trigger
            if get_bfs_memory_trigger():
                _bfs_memory_triggered = True
                logger.info("[FC][Graph] 检测到记忆工具触发BFS扩散")
        except ImportError:
            pass
        
        _should_run_bfs = (not _bfs_first_run) or _bfs_memory_triggered or _bfs_attention_triggered
        
        if _should_run_bfs:
            try:
                from zulong.memory.memory_graph import get_memory_graph
                from zulong.tools.task_tools import get_active_task_graph
                _mg = get_memory_graph()
                _tg = get_active_task_graph()
                logger.info(
                    f"[FC][Graph] BFS 前置检查: mg={'有' if _mg else '无'}, "
                    f"tg={'有' if _tg else '无'}, trigger={'首次' if not _bfs_first_run else '记忆触发' if _bfs_memory_triggered else '注意力触发'}"
                )
                if _mg and _tg:
                    # 确保 TaskGraph 节点已投射到 MemoryGraph（首次创建时未同步）
                    try:
                        from zulong.memory.graph_adapters import TaskGraphAdapter
                        _tga = TaskGraphAdapter()
                        _synced = _tga.sync(_mg, _tg)
                        if _synced:
                            logger.info(f"[FC][Graph] TaskGraph→MemoryGraph 同步: {_synced} 个节点")
                    except Exception as _sync_err:
                        logger.info(f"[FC][Graph] TaskGraph 同步跳过: {_sync_err}")

                    _in_progress = _tg.get_nodes_by_status("in_progress")
                    if _in_progress:
                        _seed_ids = [f"task:{_tg.id}/{n.id}" for n in _in_progress]
                        # P1-3: 将最近 retrieve_context 命中的 top-3 节点也加入种子
                        _retrieved_seeds = getattr(_mg, '_last_retrieved_node_ids', [])
                        if _retrieved_seeds:
                            _seed_ids.extend(_retrieved_seeds)
                        # 过滤掉 MemoryGraph 中不存在的种子 ID（去重）
                        _seen_seeds = set()
                        _valid_seeds = []
                        for s in _seed_ids:
                            if s not in _seen_seeds and _mg.has_node(s):
                                _valid_seeds.append(s)
                                _seen_seeds.add(s)
                        logger.info(
                            f"[FC][Graph] BFS 种子: seed_ids={_seed_ids}, "
                            f"valid={len(_valid_seeds)}/{len(_seed_ids)}"
                        )
                        if _valid_seeds:
                            _activations = _mg.compute_activations(
                                _valid_seeds, max_depth=3, decay=0.5,
                            )
                            logger.info(
                                f"[FC][Graph] BFS 激活扩散完成: "
                                f"seeds={[n.id for n in _in_progress]}"
                            )

                            # ── 思维导航: 自动焦点漂移 ──
                            # 如果最高激活节点不在当前种子中，自动迁移焦点
                            if _activations:
                                _focus_ctx = _mg.get_last_focus_context()
                                _current_focus = (
                                    _focus_ctx.get("focused_task_node_id", "")
                                    if _focus_ctx else ""
                                )
                                _top_node = max(
                                    _activations, key=_activations.get,
                                )
                                # 仅当最高激活节点不是当前焦点且激活值 > 0.6 时漂移
                                if (_top_node != _current_focus
                                        and _activations[_top_node] > 0.6
                                        and _top_node not in _valid_seeds):
                                    _mg.update_focus_to_node(_top_node)
                                    # 联动 AttentionWindow 模式切换
                                    if engine._attn_window:
                                        engine._attn_window.on_navigate_attention(
                                            direction="jump",
                                            target_node_id=_top_node,
                                        )
                                    logger.info(
                                        f"[FC][Graph] 焦点自动漂移: "
                                        f"{_current_focus} → {_top_node} "
                                        f"(activation={_activations[_top_node]:.2f})"
                                    )
            except Exception as _mg_err:
                logger.info(f"[FC][Graph] MemoryGraph 激活扩散跳过: {_mg_err}")
        else:
            logger.debug(f"[FC][Graph] BFS 扩散跳过（非触发条件）: first_run={_bfs_first_run}, memory={_bfs_memory_triggered}, attention={_bfs_attention_triggered}")

        result = {
            "cb_force_no_tools": cb_force_no_tools,
            "tool_calls_data": None,
            "response_content": None,
            "should_terminate": "",
            "_bfs_first_run": True,  # 标记首次执行已完成
            "_bfs_memory_triggered": False,  # 重置记忆触发标记
            "_bfs_attention_triggered": False,  # 重置注意力触发标记
        }
        if forced_next_tool_name:
            result["forced_next_tool_name"] = forced_next_tool_name
        return result

    return exec_tools_node


def _make_eval_response_node(engine: "InferenceEngine"):
    """节点 4: 文本回复评估"""

    def eval_response_node(state: dict) -> dict:
        fc_turn = state["fc_turn"]
        messages = state["messages"]
        response_content = state.get("response_content", "") or ""
        cb_force_no_tools = state.get("cb_force_no_tools", False)
        tool_results_buffer = state["tool_results_buffer"]
        gap_continue_count = state.get("gap_continue_count", 0)

        response = response_content

        # ── 语义漂移检测 ───────────────────────────────────────
        # IDE 路径已经有这层防护；这里补进核心 FC 节点，使 L2 原生
        # run_fc_loop 与 IDE runner 共享同一类“跑题拦截”语义。
        drift_detector = getattr(engine, "_semantic_drift_detector", None)
        if drift_detector is None and getattr(engine, "_enable_semantic_drift_guard", True):
            try:
                from zulong.memory.semantic_drift_detector import get_semantic_drift_detector
                drift_detector = get_semantic_drift_detector()
                setattr(engine, "_semantic_drift_detector", drift_detector)
            except Exception as drift_init_err:
                setattr(engine, "_semantic_drift_detector", False)
                logger.debug(f"[FC][Graph][DriftDetector] 初始化跳过: {drift_init_err}")

        if drift_detector and response and len(response) > 50:
            try:
                drift_result = _run_async_blocking(
                    drift_detector.detect_drift(response), timeout=5.0,
                )
                is_drifted, similarity, drift_reason = drift_result
                logger.info(
                    f"[FC][Graph][DriftDetector] turn={fc_turn} "
                    f"drift={is_drifted}, sim={similarity:.3f}, {drift_reason}"
                )
                if is_drifted:
                    drift_hint = internal_control_message(
                        f"[语义漂移拦截] 当前回复疑似偏离用户原始任务，"
                        f"相似度 {similarity:.3f}。原因：{drift_reason}\n"
                        f"请重新聚焦任务：「{state.get('user_input_text', '')[:300]}」"
                    )
                    rejected_reply = internal_control_message(response, role="assistant")
                    messages.append(rejected_reply)
                    messages.append(drift_hint)
                    if engine._attn_window:
                        engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                        engine._attn_window.register_message(drift_hint, turn=fc_turn)
                    new_null_count = state.get("null_response_count", 0) + 1
                    result = {
                        "response": None,
                        "should_terminate": "",
                        "null_response_count": new_null_count,
                    }
                    if new_null_count >= 2:
                        result["cb_force_no_tools"] = True
                    return result

                try:
                    _run_async_blocking(
                        drift_detector.add_conversation_turn(
                            state.get("user_input_text", ""), response,
                        ),
                        timeout=5.0,
                    )
                except Exception:
                    pass
            except Exception as drift_err:
                logger.warning(f"[FC][Graph][DriftDetector] 检测异常，跳过: {drift_err}")

        # CB RED 受限恢复：在完成“便签/标签/记忆落盘 + 注意力切换”之前，
        # 不允许模型用纯文本提前结束。
        if state.get("cb_recovery_stage") in {"restricted_recovery", "note_attention"} and not (
            state.get("cb_recovery_note_saved")
            and state.get("cb_recovery_attention_switched")
        ):
            recovery_hint = internal_control_message(
                "[Circuit Breaker RED 受限恢复未完成]\n"
                "请不要直接输出最终总结。当前只允许完成："
                "1) 写入并锚定便签/标签/记忆；2) 执行一次 GLOBAL / FOCUS / SINGLE_CHAIN 注意力切换。"
            )
            if response:
                messages.append(internal_control_message(response, role="assistant"))
            messages.append(recovery_hint)
            if engine._attn_window:
                if response:
                    engine._attn_window.register_message(
                        internal_control_message(response, role="assistant"),
                        turn=fc_turn,
                    )
                engine._attn_window.register_message(recovery_hint, turn=fc_turn)
            return {
                "response": None,
                "should_terminate": "",
                "cb_force_no_tools": True,
                "null_response_count": state.get("null_response_count", 0) + 1,
            }

        # 上下文压力 RED 受限恢复：与 CB RED 使用同一完成口径。
        # 未完成“便签/标签/记忆落盘 + 注意力切换”前，不允许 LLM
        # 通过纯文本绕开第二阶段工具拦截。
        if state.get("pressure_stage") == "restricted_recovery":
            requires_note = bool(state.get("pressure_recovery_requires_note", True))
            requires_attention = bool(state.get("pressure_recovery_requires_attention", True))
            note_done = bool(state.get("pressure_recovery_note_saved")) or not requires_note
            attention_done = bool(state.get("pressure_recovery_attention_switched")) or not requires_attention
            if not (note_done and attention_done):
                recovery_hint = internal_control_message(
                    "[上下文压力 RED 受限恢复未完成]\n"
                    "请不要直接输出最终总结。当前只允许完成："
                    "1) 写入并锚定便签/标签/记忆；2) 执行一次 GLOBAL / FOCUS / SINGLE_CHAIN 注意力切换。"
                )
                if response:
                    messages.append(internal_control_message(response, role="assistant"))
                messages.append(recovery_hint)
                if engine._attn_window:
                    if response:
                        engine._attn_window.register_message(
                            internal_control_message(response, role="assistant"),
                            turn=fc_turn,
                        )
                    engine._attn_window.register_message(recovery_hint, turn=fc_turn)
                return {
                    "response": None,
                    "should_terminate": "",
                    "cb_force_no_tools": False,
                    "null_response_count": state.get("null_response_count", 0) + 1,
                }

        # CB 强制收敛：接受回复，跳过 Rule A 和 InfoGap
        if cb_force_no_tools:
            synthesized = _synthesize_cleanup_report_from_tool_results(
                tool_results_buffer,
                state.get("user_input_text", ""),
            )
            if response and _looks_like_incomplete_result(response) and synthesized:
                logger.info(
                    "[FC][Graph][CB] 用已完成工具结果替换失败式收敛回复，长度=%s",
                    len(synthesized),
                )
                response = synthesized
            # 空回复保护：CB RED 后模型可能返回空字符串，此时使用降级回复
            if not response or len(response.strip()) < 10:
                logger.warning(
                    f"[FC][Graph][CB] Circuit Breaker 强制回复为空(len={len(response)})，生成降级回复"
                )
                if synthesized:
                    response = synthesized
                    logger.info(
                        "[FC][Graph][CB] 已根据工具结果生成收敛汇报，长度=%s",
                        len(response),
                    )
                else:
                    # 优先利用工具结果缓冲区中的内容生成回复
                    fallback = ""
                    if tool_results_buffer:
                        useful_results = [
                            r["result"][:300] for r in tool_results_buffer
                            if r.get("result") and len(r.get("result", "")) > 20
                            and "error" not in r.get("result", "").lower()[:50]
                            # 过滤 JSON 结构化工具输出（task_view_overview 等返回的 JSON）
                            and not r.get("result", "").lstrip().startswith(("{", "["))
                        ]
                        if useful_results:
                            fallback = (
                                "已停止继续调用工具。我根据已经拿到的工具结果整理如下：\n"
                                + "\n".join(useful_results[:3])
                            )
                    # 其次利用任务图生成进度报告
                    if not fallback:
                        try:
                            from zulong.tools.task_tools import get_active_task_graph as _get_tg_fb
                            _fb_tg = _get_tg_fb()
                            if _fb_tg:
                                _fb_title = _fb_tg.title or "当前任务"
                                _fb_leaves = _fb_tg.get_leaf_nodes()
                                _fb_completed = [n for n in _fb_leaves if n.status == "completed"]
                                _fb_uncompleted = [n for n in _fb_leaves if n.status not in ("completed", "skipped")]
                                fallback = (
                                    "已停止继续调用工具。我根据当前任务图整理进度：\n"
                                    f"当前任务「{_fb_title}」进度：{len(_fb_completed)}/{len(_fb_leaves)} 完成。"
                                )
                                if _fb_uncompleted:
                                    fallback += f"\n下一步需要执行：{_fb_uncompleted[0].label}。"
                        except Exception:
                            pass
                    # 最后使用引擎降级
                    if not fallback:
                        fallback = engine._get_fallback_response(
                            state.get("user_input_text", "")
                        )
                    response = fallback
            # CB 路径下也记录 Backfill 候选：只写入 metadata，不自动完成节点。
            # 节点完成只能来自显式 task_mark_status 与统一完成质量门证据。
            # 质量检查：如果回复主要是 JSON/结构化工具输出，跳过 Backfill 防止数据污染
            _json_chars = sum(1 for c in response if c in '{}[]":,')
            _is_structured = (_json_chars / max(len(response), 1)) > 0.12
            if (
                response
                and len(response) > 100
                and not _is_structured
                and not _looks_like_incomplete_result(response)
            ):
                try:
                    from zulong.tools.task_tools import get_active_task_graph as _get_tg_cb
                    from zulong.tools.task_tools import _save_active_backup
                    _cb_tg = _get_tg_cb()
                    if _cb_tg:
                        _cb_leaves = _cb_tg.get_leaf_nodes()
                        _cb_uncompleted = [
                            n for n in _cb_leaves
                            if n.status not in ("completed", "skipped")
                        ]
                        if _cb_uncompleted:
                            _cb_filled = 0
                            for _cb_node in _cb_uncompleted:
                                if _has_content_match(response, _cb_node.label):
                                    _cb_content = _extract_node_content(
                                        response, _cb_node.label, max_len=500,
                                    )
                                    if _looks_like_incomplete_result(_cb_content):
                                        logger.info(
                                            "[FC][Graph][CB][Backfill] 跳过未完成片段: "
                                            f"{_cb_node.id}({_cb_node.label})"
                                        )
                                        continue
                                    if hasattr(_cb_node, "metadata"):
                                        _cb_node.metadata["backfill_candidate_result"] = _cb_content
                                        _cb_node.metadata["backfill_candidate_at_turn"] = fc_turn
                                        _cb_node.metadata["backfill_candidate_cb_path"] = True
                                    _cb_filled += 1
                            if _cb_filled > 0:
                                try:
                                    _save_active_backup()
                                except Exception:
                                    pass
                                logger.info(
                                    f"[FC][Graph][CB][Backfill] CB 路径记录候选: "
                                    f"{_cb_filled}/{len(_cb_uncompleted)}，不自动改 completed"
                                )
                                engine._publish_task_graph_event(
                                    "agent_tool_call", fc_turn, "task_backfill",
                                    _json.dumps({
                                        "candidate_backfill": _cb_filled,
                                        "total_leaf": len(_cb_leaves),
                                        "auto_completed": 0,
                                        "cb_path": True,
                                    }, ensure_ascii=False),
                                )
                except Exception as cb_bf_err:
                    logger.warning(f"[FC][Graph][CB][Backfill] 异常: {cb_bf_err}")

            logger.info(
                f"[FC][Graph][CB] Circuit Breaker 强制文本回复已接受，长度={len(response)}"
            )
            return {
                "response": response,
                "cb_force_no_tools": False,
                "should_terminate": "done",
            }

        # Rule A: 过早完成声明拦截（使用 engine 级别持久化实例，保留 retry_count）
        should_block = False
        try:
            rule_guardian = engine._rule_guardian
            from zulong.tools.task_tools import get_active_task_graph as _get_tg
            block, block_reason = rule_guardian.check_premature_completion(
                response, _get_tg()
            )
            if block:
                correction, forced_tool = _build_uncompleted_task_correction(
                    block_reason,
                    previous_reply=response,
                    tag="[规则守护]",
                )
                rejected_reply = internal_control_message(response, role="assistant")
                messages.append(rejected_reply)
                messages.append(correction)
                if forced_tool:
                    state["forced_next_tool_name"] = forced_tool
                if engine._attn_window:
                    engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                    engine._attn_window.register_message(correction, turn=fc_turn)
                should_block = True
        except Exception as guard_err:
            logger.warning(f"[FC][Graph][RuleGuardian] 检查异常: {guard_err}")

        if should_block:
            new_null_count = state.get("null_response_count", 0) + 1
            result = {
                "response": None,
                "should_terminate": "",
                "null_response_count": new_null_count,
                "cb_force_no_tools": False,
                "forced_next_tool_name": state.get("forced_next_tool_name", ""),
            }
            # 拦截次数达到阈值时，继续做工具纠偏，而不是收敛成最终回复。
            if new_null_count >= 2:
                correction_msg, forced_tool = _build_uncompleted_task_correction(
                    "多次拦截检测到任务图有未完成节点，但模型持续尝试直接回复。",
                    previous_reply=response,
                    tag="[强制纠偏]",
                )
                messages.append(correction_msg)
                if forced_tool:
                    state["forced_next_tool_name"] = forced_tool
                    result["forced_next_tool_name"] = forced_tool
                if engine._attn_window:
                    engine._attn_window.register_message(correction_msg, turn=fc_turn)
                logger.info(
                    f"[FC][Graph] 拦截次数达 {new_null_count}，注入工具纠偏"
                )
            return result

        # 构建子任务上下文（供 InfoGapDetector 结构化依赖检查）
        subtask_ctx = None
        try:
            from zulong.tools.task_tools import get_active_task_graph
            active_tg = get_active_task_graph()
            if active_tg:
                in_progress = active_tg.get_nodes_by_status("in_progress")
                if in_progress:
                    cur_node = in_progress[0]
                    deps = active_tg.get_dependencies(cur_node.id)
                    available = {}
                    for dep_id in deps:
                        dep_node = active_tg.get_node(dep_id)
                        if dep_node and dep_node.status == "completed":
                            available[dep_id] = dep_node.result or ""
                    subtask_ctx = {
                        "current_subtask": cur_node.id,
                        "dependencies": deps,
                        "available_results": available,
                    }
        except Exception:
            pass

        # 信息缺口检测
        _MAX_GAP_CONTINUES = 5
        should_continue = False
        try:
            from zulong.l2.info_gap_detector import InfoGapType
            gap_type, gap_desc, gap_conf = engine._info_gap_detector.detect(
                llm_output=response,
                tool_results=tool_results_buffer if tool_results_buffer else None,
                subtask_context=subtask_ctx,
            )
            if gap_type == InfoGapType.NEED_USER_INPUT and gap_conf >= 0.6:
                logger.info(
                    f"[FC][Graph][InfoGap] 需要用户输入: {gap_desc} (置信度={gap_conf:.2f})"
                )
                # 模型已在回复中向用户提问，直接接受
            elif gap_type == InfoGapType.NEED_SUBTASK_RESULT and gap_conf >= 0.6:
                logger.info(
                    f"[FC][Graph][InfoGap] 需要子任务结果: {gap_desc} "
                    f"(置信度={gap_conf:.2f}), 重试={gap_continue_count}/{_MAX_GAP_CONTINUES}"
                )
                # 继续已有任务图时，模型正在为子任务产生内容，不应被 InfoGap 拦截
                _is_resume = state.get("is_resume", False)
                if _is_resume and len(response) > 100:
                    logger.info(
                        "[FC][Graph][InfoGap] 继续任务图策略且回复充实，跳过子任务结果拦截"
                    )
                elif gap_continue_count >= _MAX_GAP_CONTINUES:
                    logger.warning(
                        f"[FC][Graph][InfoGap] 闭环已达重试上限 ({_MAX_GAP_CONTINUES})，"
                        f"标记任务为 blocked 并放行"
                    )
                    # 修复：达到上限后标记当前节点为 blocked，而非静默放行
                    try:
                        from zulong.tools.task_tools import get_active_task_graph
                        _tg = get_active_task_graph()
                        if _tg:
                            in_prog = _tg.get_nodes_by_status("in_progress")
                            for _n in in_prog:
                                _tg.update_node_status(
                                    _n.id, "blocked",
                                    result=f"信息缺口: {gap_desc}"
                                )
                                logger.info(
                                    f"[FC][Graph][InfoGap] 节点 {_n.id} 标记为 blocked"
                                )
                    except Exception:
                        pass
                else:
                    gap_hint = internal_control_message(
                        f"[信息缺口提示] 当前子任务缺少前置结果: {gap_desc}\n"
                        "请先用 task_view_overview 查看任务图，找到并执行未完成的前置子任务，"
                        "或用 task_mark_status 更新进度后继续。"
                    )
                    rejected_reply = internal_control_message(response, role="assistant")
                    messages.append(rejected_reply)
                    messages.append(gap_hint)
                    if engine._attn_window:
                        engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                        engine._attn_window.register_message(gap_hint, turn=fc_turn)
                    should_continue = True
                    new_null_count = state.get("null_response_count", 0) + 1
                    result = {
                        "response": None,
                        "gap_continue_count": gap_continue_count + 1,
                        "should_terminate": "",
                        "null_response_count": new_null_count,
                    }
                    if new_null_count >= 2:
                        result["cb_force_no_tools"] = True
                        logger.info(
                            f"[FC][Graph][InfoGap] 拦截次数达 {new_null_count}，注入 CB 强制收敛"
                        )
                    return result
            else:
                logger.debug(
                    f"[FC][Graph][InfoGap] 信息充足 (type={gap_type.value}, conf={gap_conf:.2f})"
                )
        except Exception as e:
            logger.warning(f"[FC][Graph][InfoGap] 检测异常，跳过: {e}")

        if should_continue:
            return {
                "response": None,
                "should_terminate": "",
            }

        # ── 继续任务图 Auto-Mark 安全网 ──────────────────────────────
        # 小模型在继续任务图流程中常常生成实质内容但忘记调用 task_mark_status。
        # 这里不再自动把节点标记为 completed，只记录“进度候选”，再要求
        # LLM 结合真实工具/验证证据显式调用 task_mark_status，避免绕过
        # TaskSpec Coverage Gate。
        _MAX_CONTINUE_AUTOMARKS = 5
        is_resume = state.get("is_resume", False)
        resume_automark_count = state.get("resume_automark_count", 0)

        if (
            is_resume
            and len(response) > 100
            and resume_automark_count < _MAX_CONTINUE_AUTOMARKS
            and not response.rstrip().endswith(("?", "\uff1f"))
            and not _is_filler_content(response)
            and not _looks_like_incomplete_result(response)
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_am
                from zulong.tools.task_tools import _save_active_backup
                tg = _get_tg_am()
                if tg:
                    leaf_nodes = tg.get_leaf_nodes()
                    uncompleted = [n for n in leaf_nodes if n.status != "completed"]
                    if uncompleted:
                        # 优先选 in_progress 节点，其次 pending
                        target = None
                        for n in uncompleted:
                            if n.status == "in_progress":
                                target = n
                                break
                        if not target:
                            target = uncompleted[0]

                        # 只记录候选进度，不自动标记完成。
                        result_text = response[:500]
                        if _looks_like_incomplete_result(result_text):
                            return {
                                "response": response,
                                "should_terminate": "done",
                            }
                        if hasattr(target, "metadata"):
                            target.metadata["auto_progress_candidate"] = result_text
                            target.metadata["auto_progress_candidate_at_turn"] = fc_turn
                            target.metadata["auto_progress_candidate_source"] = "resume_text_review"
                        try:
                            _save_active_backup()
                        except Exception:
                            pass
                        logger.info(
                            f"[FC][ContinueTaskGraph][AutoMark] 记录候选进度 {target.id}"
                            f" ({target.label}), result_len={len(result_text)}"
                        )
                        engine._publish_task_graph_event(
                            "agent_tool_call", fc_turn, "task_mark_status",
                            _json.dumps({
                                "node_id": target.id,
                                "auto_progress_candidate": True,
                                "auto_completed": False,
                            }, ensure_ascii=False),
                        )

                        # 检查剩余未完成节点
                        remaining = [
                            n for n in tg.get_leaf_nodes()
                            if n.status not in ("completed", "skipped")
                        ]
                        if remaining:
                            next_node = next(
                                (n for n in remaining if n.status == "in_progress"),
                                remaining[0],
                            )
                            if next_node.status in ("pending", "needs_adjust", "waiting_input"):
                                tg.update_node_status(next_node.id, "in_progress")
                            try:
                                _save_active_backup()
                            except Exception:
                                pass
                            logger.info(
                                f"[FC][ContinueTaskGraph][AutoMark] 下一节点: {next_node.id}"
                                f" ({next_node.label}), 标记 in_progress"
                            )
                            continuation = internal_control_message(
                                f"[自动进度候选] 检测到节点 {target.id}（{target.label}）"
                                "可能已有文本产出，但系统不会自动标记完成。\n"
                                "请基于真实工具/验证证据显式调用 task_mark_status；"
                                f"当前继续处理节点 {next_node.id}（{next_node.label}）"
                                f"：{next_node.desc or next_node.label}。"
                            )
                            rejected_reply = internal_control_message(response, role="assistant")
                            messages.append(rejected_reply)
                            messages.append(continuation)
                            if engine._attn_window:
                                engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                                engine._attn_window.register_message(continuation, turn=fc_turn)
                            new_null_count = state.get("null_response_count", 0) + 1
                            result = {
                                "response": None,
                                "resume_automark_count": resume_automark_count + 1,
                                "should_terminate": "",
                                "null_response_count": new_null_count,
                                "cb_force_no_tools": False,
                            }
                            return result
                        else:
                            logger.info(
                                "[FC][ContinueTaskGraph][AutoMark] 仅记录候选进度，等待显式完成确认"
                            )
            except Exception as am_err:
                logger.warning(f"[FC][ContinueTaskGraph][AutoMark] 异常: {am_err}")
        # ── END ContinueTaskGraph Auto-Mark ─────────────────────────────────

        # ── 文件操作真实性守卫 ──────────────────────────────────
        # 用户要求创建/写入/删除文件时，不能只靠自然语言声称完成。
        # 必须至少有一次相关工具成功结果，否则继续要求模型调用工具；
        # 多次失败后返回明确故障态，避免“假成功”。
        file_op_guard = _check_file_operation_truth(
            response,
            state.get("user_input_text", ""),
            tool_results_buffer,
        )
        if file_op_guard == "retry":
            rejected_reply = internal_control_message(response, role="assistant")
            file_guard_hint = internal_control_message(
                    "[文件操作真实性校验] 用户要求创建、写入、修改或删除文件，"
                    "但目前没有看到任何成功的文件操作工具结果。"
                    "请立刻调用可用的写入工具完成真实落盘："
                    "如果用户给出宿主机绝对路径，优先调用 ide_write_file；"
                    "否则调用 exec_write_file。"
                    "完成前不要声称文件已创建或已写入。"
            )
            messages.append(rejected_reply)
            messages.append(file_guard_hint)
            if engine._attn_window:
                engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                engine._attn_window.register_message(file_guard_hint, turn=fc_turn)
            return {
                "response": None,
                "should_terminate": "",
                "null_response_count": state.get("null_response_count", 0) + 1,
            }
        if file_op_guard == "fail":
            return {
                "response": (
                    "系统当前出问题了，文件操作没有返回真实成功结果，"
                    "因此无法确认文件已创建或写入。"
                ),
                "should_terminate": "done",
            }

        # ── 工具增强首次执行：任务图节点候选内容记录 ──────────────────
        # 小模型在任务图首次执行时常见行为：
        #   1. 调用 task_create_plan + task_add_node 创建任务图骨架
        #   2. 直接生成完整回复内容，跳过逐节点 task_mark_status
        # 导致任务图节点全部为空但回复内容完整。
        # 此安全网只把候选内容写入节点 metadata，供质量门和后续 LLM 复核；
        # 不自动把节点改成 completed，避免“文本相似”绕过显式执行和验证证据。
        if (
            not is_resume
            and len(response) > 100
            and not response.rstrip().endswith(("?", "\uff1f"))
            and not _is_filler_content(response)
            and not _looks_like_incomplete_result(response)
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_bf
                from zulong.tools.task_tools import _save_active_backup
                tg = _get_tg_bf()
                if tg:
                    leaf_nodes = tg.get_leaf_nodes()
                    uncompleted = [
                        n for n in leaf_nodes
                        if n.status not in ("completed", "skipped")
                    ]
                    if uncompleted:
                        backfill_count = 0
                        skipped_count = 0
                        for node in uncompleted:
                            # 尝试从回复中提取与节点相关的内容片段
                            node_content = _extract_node_content(
                                response, node.label, max_len=500,
                            )
                            # 只有在回复中确实找到匹配内容时才记录候选。
                            if (
                                _has_content_match(response, node.label)
                                and not _looks_like_incomplete_result(node_content)
                            ):
                                if hasattr(node, "metadata"):
                                    node.metadata["backfill_candidate_result"] = node_content
                                    node.metadata["backfill_candidate_at_turn"] = fc_turn
                                    node.metadata["backfill_candidate_cb_path"] = False
                                backfill_count += 1
                            else:
                                skipped_count += 1
                                logger.debug(
                                    f"[FC][Backfill] 跳过节点 {node.id}（{node.label}）：回复中未找到匹配内容"
                                )
                        try:
                            _save_active_backup()
                        except Exception:
                            pass
                        logger.info(
                            f"[FC][Backfill] 工具增强首次执行记录候选: "
                            f"候选 {backfill_count} 个节点，跳过 {skipped_count} 个无匹配节点；不自动改 completed"
                        )
                        engine._publish_task_graph_event(
                            "agent_tool_call", fc_turn, "task_backfill",
                            _json.dumps({
                                "candidate_backfill": backfill_count,
                                "skipped": skipped_count,
                                "total_leaf": len(leaf_nodes),
                                "auto_completed": 0,
                            }, ensure_ascii=False),
                        )
            except Exception as bf_err:
                logger.warning(f"[FC][Backfill] 异常: {bf_err}")
        # ── END Backfill ─────────────────────────────────

        # ── 空/短回复 + 未完成任务图 安全网 ──────────────────────
        # 4B 模型常在创建任务图骨架后返回空字符串就停止，
        # 此时任务图有未完成节点，不应接受空回复。
        if not response or len(response.strip()) < 10:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_empty
                _tg_empty = _get_tg_empty()
                if _tg_empty:
                    _leaves = _tg_empty.get_leaf_nodes()
                    _uncompleted = [
                        n for n in _leaves
                        if n.status not in ("completed", "skipped")
                    ]
                    if _uncompleted:
                        new_null_count = state.get("null_response_count", 0) + 1
                        # 找到下一个需要执行的节点
                        _next = None
                        for _n in _uncompleted:
                            if _n.status == "in_progress":
                                _next = _n
                                break
                        if not _next:
                            _next = _uncompleted[0]
                            _tg_empty.update_node_status(_next.id, "in_progress")

                        # 构建提示：告知当前任务图状态和下一步
                        _completed_count = len([
                            n for n in _leaves if n.status == "completed"
                        ])
                        nudge = internal_control_message(
                            f"[空回复拦截] 你的回复为空，但任务图还有 "
                            f"{len(_uncompleted)}/{len(_leaves)} 个未完成节点。\n"
                            f"当前进度：{_completed_count}/{len(_leaves)} 完成。\n"
                            f"请立即开始执行节点 {_next.id}（{_next.label}）"
                            f"：{_next.desc or _next.label}\n"
                            f"生成该节点的详细内容，完成后调用 "
                            f"task_mark_status(node_id='{_next.id}', "
                            f"status='completed', result='你的结果')。"
                        )
                        messages.append(nudge)
                        if engine._attn_window:
                            engine._attn_window.register_message(nudge, turn=fc_turn)
                        logger.info(
                            f"[FC][Graph][EmptyGuard] 空回复拦截，"
                            f"未完成节点 {len(_uncompleted)}/{len(_leaves)}，"
                            f"提示执行 {_next.id}（{_next.label}），"
                            f"null_count={new_null_count}"
                        )
                        result = {
                            "response": None,
                            "should_terminate": "",
                            "null_response_count": new_null_count,
                        }
                        # 多次空回复后强制收敛，避免无限循环
                        if new_null_count >= 2:
                            logger.info(
                                f"[FC][Graph][EmptyGuard] 空回复拦截次数达 "
                                f"{new_null_count}，任务图未完成，继续保留工具执行能力"
                            )
                        return result
            except Exception as eg_err:
                logger.warning(f"[FC][Graph][EmptyGuard] 异常: {eg_err}")
        # ── END 空/短回复安全网 ───────────────────────────────────

        # ── 任务全完成但回复为空 → 从 TaskGraph 合成摘要 ─────────
        if not response or len(response.strip()) < 10:
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_synth
                _tg_synth = _get_tg_synth()
                if _tg_synth:
                    _synth_leaves = _tg_synth.get_leaf_nodes()
                    _synth_completed = [
                        n for n in _synth_leaves if n.status == "completed"
                    ]
                    if _synth_completed and len(_synth_completed) == len(_synth_leaves):
                        # 所有任务已完成 → 从节点 result 合成摘要回复
                        parts = [f"## {_tg_synth.title}\n"]
                        for node in _synth_completed:
                            result_text = getattr(node, 'result', '') or ''
                            if result_text:
                                parts.append(f"### {node.label}\n{result_text}\n")
                            else:
                                parts.append(f"### {node.label}\n（已完成）\n")
                        response = "\n".join(parts)
                        logger.info(
                            f"[FC][Graph][Synthesize] 任务全部完成但模型返回空响应，"
                            f"从 {len(_synth_completed)} 个已完成节点合成摘要回复"
                        )
            except Exception as synth_err:
                logger.warning(f"[FC][Graph][Synthesize] 合成异常: {synth_err}")
        # ── END 任务全完成合成 ──────────────────────────────────

        # ── 未完成任务拦截 ─────────────────────────────────────
        # IDE runner 已有这层防护。补入核心节点后，普通 L2 FC 循环在
        # 叶子任务仍未完成时不会过早接受最终文本。短进度句/填充句也
        # 必须拦截，避免 “继续编写...” 被当成最终回复。
        if (
            response
            and len(response.strip()) >= 6
            and not _looks_like_incomplete_result(response)
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_uc
                tg_uc = _get_tg_uc()
                if tg_uc:
                    leaves_uc = tg_uc.get_leaf_nodes()
                    user_leaves_uc = [
                        n for n in leaves_uc
                        if not getattr(n, "id", "").startswith("crg_")
                    ]
                    total_uc = len(user_leaves_uc)
                    uncompleted_uc = [
                        n for n in user_leaves_uc
                        if n.status not in ("completed", "skipped")
                    ]
                    if not user_leaves_uc:
                        req_node = tg_uc.get_node("req")
                        if req_node and req_node.status not in ("completed", "skipped"):
                            uncompleted_uc = [req_node]
                            total_uc = 1
                    if total_uc > 0 and len(uncompleted_uc) > 0:
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
                        nudge_uc = internal_control_message(
                            f"[任务未完成] 仍有 {len(uncompleted_uc)}/{total_uc} 个子任务未完成。"
                            f"当前应执行: {current_uc.id}({current_uc.label})。"
                            f"请继续调用工具执行任务，不要提前生成最终总结。"
                        )
                        rejected_reply = internal_control_message(response, role="assistant")
                        messages.append(rejected_reply)
                        messages.append(nudge_uc)
                        if engine._attn_window:
                            engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                            engine._attn_window.register_message(nudge_uc, turn=fc_turn)
                        new_null_count = state.get("null_response_count", 0) + 1
                        result = {
                            "response": None,
                            "should_terminate": "",
                            "null_response_count": new_null_count,
                            "cb_force_no_tools": False,
                        }
                        return result
            except Exception as uc_err:
                logger.warning(f"[FC][Graph][UncompletedGuard] 异常: {uc_err}")

        # ── 响应提前中断检测 ───────────────────────────────────
        # 短进度句很容易是 function_call 生成被截断后的残留文本。
        if (
            response
            and 6 <= len(response.strip()) < 80
            and not _is_filler_content(response)
            and state.get("null_response_count", 0) < 3
        ):
            try:
                from zulong.tools.task_tools import get_active_task_graph as _get_tg_ri
                tg_ri = _get_tg_ri()
                if tg_ri:
                    leaves_ri = [
                        n for n in tg_ri.get_leaf_nodes()
                        if not getattr(n, "id", "").startswith("crg_")
                    ]
                    uncompleted_ri = [
                        n for n in leaves_ri
                        if n.status not in ("completed", "skipped")
                    ]
                    if leaves_ri and len(uncompleted_ri) >= len(leaves_ri) * 0.3:
                        current_ri = next(
                            (n for n in uncompleted_ri if n.status == "in_progress"),
                            uncompleted_ri[0],
                        )
                        nudge_ri = internal_control_message(
                            f"[响应提前中断] 回复仅 {len(response.strip())} 字符，"
                            f"疑似工具调用未完整生成。仍有 "
                            f"{len(uncompleted_ri)}/{len(leaves_ri)} 个子任务未完成。"
                            f"当前应执行: {current_ri.id}({current_ri.label})。"
                            f"请继续调用工具执行任务。"
                        )
                        rejected_reply = internal_control_message(response, role="assistant")
                        messages.append(rejected_reply)
                        messages.append(nudge_ri)
                        if engine._attn_window:
                            engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                            engine._attn_window.register_message(nudge_ri, turn=fc_turn)
                        new_null_count = state.get("null_response_count", 0) + 1
                        return {
                            "response": None,
                            "should_terminate": "",
                            "null_response_count": new_null_count,
                        }
            except Exception as ri_err:
                logger.warning(f"[FC][Graph][ResponseIntegrity] 异常: {ri_err}")

        # ── 首轮无效回复拦截 ───────────────────────────────────
        if (
            _is_filler_content(response)
            and fc_turn <= 1
            and not state.get("is_resume", False)
            and state.get("null_response_count", 0) < 3
        ):
            stripped_lower = (response or "").strip().lower()
            greeting_patterns = (
                "你好", "您好", "有什么我可以帮", "有什么可以帮",
                "我可以帮你", "需要帮助", "hello", "hi", "how can i",
                "what can i", "请问", "请说",
            )
            if any(p in stripped_lower for p in greeting_patterns) or len(stripped_lower) < 50:
                first_hint = internal_control_message(
                    f"[首轮回复无效] 你返回了问候或过短回复，但用户任务是：\n"
                    f"「{state.get('user_input_text', '')[:300]}」\n"
                    f"请不要打招呼或反问，直接分析任务需求并调用工具开始执行。"
                )
                rejected_reply = internal_control_message(response, role="assistant")
                messages.append(rejected_reply)
                messages.append(first_hint)
                if engine._attn_window:
                    engine._attn_window.register_message(rejected_reply, turn=fc_turn)
                    engine._attn_window.register_message(first_hint, turn=fc_turn)
                return {
                    "response": None,
                    "should_terminate": "",
                    "null_response_count": state.get("null_response_count", 0) + 1,
                }

        # 所有检查通过，接受回复
        return {
            "response": response,
            "should_terminate": "done",
        }

    return eval_response_node


# ---------------------------------------------------------------------------
# 3. 辅助函数
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    "我正在思考", "让我继续", "我来继续", "让我想想", "接下来我",
    "我正在处理", "正在分析", "正在执行", "稍等", "我需要",
    "但我需要", "不过我需要", "还需要进一步", "需要更多信息",
]


def _is_filler_content(text: str) -> bool:
    """检测回复是否为填充性内容（无实质任务成果）"""
    stripped = text.strip()
    if len(stripped) < 50:
        return True
    filler_count = sum(1 for p in _FILLER_PATTERNS if p in stripped)
    if filler_count >= 2:
        return True
    return False


def _looks_like_audit_or_cleanup_report(text: str) -> bool:
    stripped = str(text or "").strip()
    if len(stripped) < 40:
        return False
    report_shape = ("汇报", "报告", "清单", "列表", "总览", "如下")
    cleanup_context = (
        "记忆",
        "节点",
        "挂起任务",
        "历史任务",
        "未完成任务",
        "经验",
        "清理",
        "删除",
        "候选",
        "blocked",
    )
    return (
        any(marker in stripped for marker in report_shape)
        and sum(1 for marker in cleanup_context if marker.lower() in stripped.lower()) >= 2
    )


def _looks_like_incomplete_result(text: str) -> bool:
    """Detect blockage/status prose that must not be backfilled as completed."""
    raw = str(text or "")
    if not raw:
        return False
    lowered = raw.lower()
    hard_system_markers = (
        "任务执行中断",
        "强制收敛",
        "无法正常回复",
        "系统当前出问题",
        "安全防护触发",
        "触发循环保护",
    )
    if any(marker.lower() in lowered for marker in hard_system_markers):
        return True
    if _looks_like_audit_or_cleanup_report(raw):
        return False
    hard_blockers = (
        "阻塞",
        "被阻塞",
        "无法运行测试",
        "无法验证",
        "无法完成",
        "不能完成",
        "审批拒绝",
        "审批超时",
        "用户未应用",
        "用户拒绝",
        "未应用写入",
        "未真实存在",
        "workspace_trust",
        "approval_blocked",
        "blocked",
        "timed out",
        "timeout",
        "interrupted",
    )
    if any(marker.lower() in lowered for marker in hard_blockers):
        return True

    success_markers = (
        "全部通过",
        "测试通过",
        "全部测试通过",
        "ran ",
        "\nok",
        " ok",
        "无失败",
        "0 failed",
        '"failed_count": 0',
        "'failed_count': 0",
    )
    has_success = any(marker.lower() in lowered for marker in success_markers)
    soft_failures = (
        "工具执行失败",
        "命令执行失败",
        "测试失败",
        "语法错误",
        "returncode=1",
        "returncode\": 1",
        "syntaxerror",
        "traceback",
        "failed (",
        "failed=",
        "error:",
        " errors=",
    )
    if not has_success and any(marker.lower() in lowered for marker in soft_failures):
        return True

    positive_markers = (
        "已创建",
        "已生成",
        "已写入",
        "已完成",
        "已实现",
        "创建成功",
        "写入成功",
        "生成成功",
    )
    if any(marker in raw for marker in positive_markers):
        return False
    markers = (
        "任务执行中断",
        "强制收敛",
        "未产出",
        "进行中但未产出",
        "尚未完成",
        "还未完成",
        "未完成节点",
        "未生成",
        "未创建",
        "没有产出",
        "触发循环保护",
        "系统当前出问题",
        "stalled",
    )
    return any(marker.lower() in lowered for marker in markers)


def _synthesize_cleanup_report_from_tool_results(
    tool_results_buffer: List[Dict],
    user_input: str = "",
) -> str:
    """Build a user-facing cleanup summary from completed structured tools."""
    if not tool_results_buffer:
        return ""

    deleted_memory_count = 0
    protected_memory_count = 0
    not_found_memory_count = 0
    preview_count = 0
    preview_keywords: List[str] = []
    deleted_samples: List[str] = []
    task_list_count: Optional[int] = None
    task_samples: List[str] = []
    removed_task_count = 0
    removed_task_samples: List[str] = []
    deleted_edge_count = 0
    errors: List[str] = []
    reasons: List[str] = []

    for item in tool_results_buffer or []:
        tool_name = str(item.get("tool_name") or "")
        reason = str(item.get("reason") or "").strip()
        if reason:
            reasons.append(_compact_line(reason, 96))
        result_text = str(item.get("result") or "").strip()
        if not tool_name or not result_text:
            continue
        payload: Any
        try:
            payload = _json.loads(result_text)
        except Exception:
            payload = {"message": result_text}
        if not isinstance(payload, dict):
            payload = {"data": payload}

        if payload.get("error"):
            errors.append(f"{tool_name}: {str(payload.get('error'))[:120]}")
            continue

        if tool_name == "delete_memory_node":
            if payload.get("action") == "preview":
                candidates = payload.get("candidates") or []
                preview_count += len(candidates) if isinstance(candidates, list) else 0
                keyword = payload.get("keyword")
                if keyword:
                    preview_keywords.append(str(keyword))
                continue
            deleted = payload.get("deleted") or []
            deleted_memory_count += int(payload.get("deleted_count") or 0)
            protected_memory_count += int(payload.get("protected_count") or 0)
            not_found = payload.get("not_found") or []
            if isinstance(not_found, list):
                not_found_memory_count += len(not_found)
            if isinstance(deleted, list):
                for node in deleted[:5]:
                    if isinstance(node, dict):
                        label = str(node.get("label") or node.get("node_id") or "")
                        if label:
                            deleted_samples.append(_compact_line(label, 48))

        elif tool_name == "delete_memory_edge":
            deleted_edge_count += int(payload.get("deleted_count") or 0)

        elif tool_name == "task_list_suspended":
            if "count" in payload:
                task_list_count = int(payload.get("count") or 0)
            tasks = payload.get("tasks") or []
            if isinstance(tasks, list):
                task_list_count = len(tasks) if task_list_count is None else task_list_count
                for task in tasks[:5]:
                    if isinstance(task, dict):
                        label = (
                            task.get("description")
                            or task.get("title")
                            or task.get("task_id")
                        )
                        if label:
                            task_samples.append(_compact_line(str(label), 48))

        elif tool_name == "task_remove_node":
            removed = payload.get("removed_ids") or []
            removed_count = int(payload.get("removed_count") or 0)
            removed_task_count += removed_count
            if isinstance(removed, list):
                removed_task_samples.extend(str(x) for x in removed[:5])

    has_cleanup_signal = any((
        deleted_memory_count,
        protected_memory_count,
        not_found_memory_count,
        preview_count,
        task_list_count is not None,
        removed_task_count,
        deleted_edge_count,
        errors,
    ))
    if not has_cleanup_signal:
        return ""

    lines = ["我已停止继续调用工具，并根据已经返回的结果整理清理汇报："]
    if reasons:
        lines.append(f"- 执行意图：{'；'.join(dict.fromkeys(reasons[:3]))}。")
    if deleted_memory_count:
        lines.append(f"- 已删除记忆节点：{deleted_memory_count} 个。")
    if deleted_edge_count:
        lines.append(f"- 已删除记忆边：{deleted_edge_count} 条。")
    if removed_task_count:
        lines.append(f"- 已从任务图移除节点：{removed_task_count} 个。")
    if task_list_count is not None:
        lines.append(f"- 已查询挂起任务：{task_list_count} 个。")
    if preview_count:
        kws = "、".join(dict.fromkeys(preview_keywords[:6]))
        suffix = f"（关键词：{kws}）" if kws else ""
        lines.append(f"- 已预览待删除候选：{preview_count} 个{suffix}，尚未确认删除。")
    if protected_memory_count:
        lines.append(f"- 受保护未删除：{protected_memory_count} 个核心记忆。")
    if not_found_memory_count:
        lines.append(f"- 未找到：{not_found_memory_count} 个节点。")
    if deleted_samples:
        samples = "；".join(dict.fromkeys(deleted_samples[:5]))
        lines.append(f"- 删除样例：{samples}。")
    if task_samples:
        samples = "；".join(dict.fromkeys(task_samples[:5]))
        lines.append(f"- 挂起任务样例：{samples}。")
    if removed_task_samples:
        samples = "、".join(dict.fromkeys(removed_task_samples[:8]))
        lines.append(f"- 移除节点样例：{samples}。")
    if errors:
        lines.append("- 工具错误：" + "；".join(errors[:3]) + "。")
    if "不要删除经验" in str(user_input or ""):
        lines.append("- 已按要求保留经验类记忆；未把经验清理计入删除目标。")
    if not (deleted_memory_count or deleted_edge_count or removed_task_count):
        lines.append("- 当前只拿到了查询/预览结果，未看到确认删除已经完成。")
    lines.append("后续若继续清理，应基于上述候选分批确认，避免误删。")
    return "\n".join(lines)


def _compact_line(text: str, max_len: int = 64) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3].rstrip() + "..."


def _summarize_llm_usage(usage: Any) -> Dict[str, Any]:
    if not usage:
        return {}
    keys = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    )
    summary: Dict[str, Any] = {}
    for key in keys:
        value = None
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if value is not None:
            summary[key] = value
    return summary


def _extract_tool_call_reason(text: str, tool_name: str, max_len: int = 160) -> str:
    """Extract the model's own short explanation for a tool call when present."""
    raw = " ".join(str(text or "").split())
    if not raw:
        return ""
    tool = str(tool_name or "")
    candidates: List[str] = []
    for marker in [m for m in (tool, "调用", "工具", "为了", "因为", "用于", "需要") if m]:
        idx = raw.find(marker)
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(raw), idx + max_len)
            candidates.append(raw[start:end].strip(" ，。；;:："))
    if not candidates:
        return ""
    best = min(candidates, key=len)
    if len(best) <= max_len:
        return best
    return best[: max_len - 3].rstrip() + "..."


def _task_graph_uncompleted_context():
    """Return (has_uncompleted, next_node) for the active user task graph."""
    try:
        from zulong.tools.task_tools import get_active_task_graph
        tg = get_active_task_graph()
    except Exception:
        return False, None
    if not tg:
        return False, None
    try:
        leaves = [
            n for n in tg.get_leaf_nodes()
            if not getattr(n, "id", "").startswith("crg_")
        ]
    except Exception:
        leaves = []
    if not leaves:
        req_node = tg.get_node("req") if hasattr(tg, "get_node") else None
        if req_node and getattr(req_node, "status", "") not in ("completed", "skipped"):
            return True, req_node
        return False, None
    uncompleted = [
        n for n in leaves
        if getattr(n, "status", "") not in ("completed", "skipped")
    ]
    if not uncompleted:
        return False, None
    current = next(
        (n for n in uncompleted if getattr(n, "status", "") == "in_progress"),
        uncompleted[0],
    )
    return True, current


def _set_forced_tool_choice(
    api_kwargs: Dict[str, Any],
    tool_definitions: List[Dict],
    tool_name: str,
) -> bool:
    """Force one tool call when the current API/tool bundle exposes it."""
    if not tool_name:
        return False
    available = {
        str((schema.get("function") or {}).get("name") or "")
        for schema in (tool_definitions or [])
        if isinstance(schema, dict)
    }
    if tool_name not in available:
        return False
    api_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": tool_name},
    }
    return True


def _build_uncompleted_task_correction(
    reason: str,
    *,
    previous_reply: str = "",
    tag: str = "[规则守护]",
    tool_definitions: Optional[List[Dict]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Build a correction turn that steers an unfinished task back to tools."""
    has_uncompleted, current = _task_graph_uncompleted_context()
    node_id = getattr(current, "id", "") if current else ""
    label = getattr(current, "label", "") if current else ""
    desc = getattr(current, "desc", "") if current else ""
    node_text = " ".join(part for part in (label, desc) if part)
    expected_files = _extract_expected_file_names(node_text)

    available_tools = {
        str((schema.get("function") or {}).get("name") or "")
        for schema in (tool_definitions or [])
        if isinstance(schema, dict)
    }
    forced_tool = ""
    if expected_files:
        if not available_tools or "exec_write_file" in available_tools:
            forced_tool = "exec_write_file"
        elif "ide_write_file" in available_tools:
            forced_tool = "ide_write_file"
    current_line = (
        f"当前应执行节点: {node_id}({label})。" if node_id or label
        else "当前任务图仍有未完成节点。"
    )
    desc_line = f"节点要求: {desc}" if desc else ""
    files_line = (
        "预期产物: " + ", ".join(expected_files[:3])
        if expected_files else ""
    )
    previous_line = (
        f"上一轮只输出了进度句: {previous_reply.strip()[:120]}"
        if previous_reply else ""
    )

    if expected_files:
        if forced_tool == "exec_write_file":
            action_line = (
                f"下一轮必须优先调用 exec_write_file 写入 {expected_files[0]}。"
                "如果内容较长，先用 mode='overwrite' 写第一段，再用 mode='append' "
                "分多轮追加，每轮 content 控制在 800-1200 字符；写入成功并验证后再调用 "
                f"task_mark_status(node_id='{node_id}', status='completed')。"
            )
        elif forced_tool == "ide_write_file":
            action_line = (
                f"下一轮必须优先调用 ide_write_file 写入 {expected_files[0]}。"
                "如果内容较长，先用 mode='overwrite' 写第一段，再用 mode='append' "
                "通过同一个受保护通道分段追加；每轮 content 控制在 800-1200 字符，"
                "不要继续输出超长工具参数。"
                "写入成功并验证后再调用 "
                f"task_mark_status(node_id='{node_id}', status='completed')。"
            )
        else:
            action_line = (
                f"下一轮必须调用可用写入工具真实落盘 {expected_files[0]}；"
                "如果内容较长，采用分段或脚本化方式，避免超长工具参数被截断。"
            )
    else:
        action_line = (
            "下一轮必须调用能产生真实进展的工具，例如写文件、执行命令或更新当前节点；"
            "不要只查看任务图、不要只输出进度句、不要提交最终回答。"
        )

    content = "\n".join(
        line for line in (
            f"{tag} {reason}",
            previous_line,
            current_line,
            desc_line,
            files_line,
            action_line,
            "在当前节点完成前，禁止自然语言回复用户，禁止 submit_final_answer。",
        )
        if line
    )
    if not has_uncompleted:
        forced_tool = ""
    return internal_control_message(content), forced_tool


def _extract_expected_file_names(text: str) -> List[str]:
    """Extract likely file outputs from a task node description."""
    if not text:
        return []
    import re as _re

    pattern = (
        r"(?<![\w.-])"
        r"([A-Za-z0-9_\-./\\]+"
        r"\.(?:js|mjs|cjs|ts|tsx|jsx|py|html|css|json|md|txt|yaml|yml|vue|svelte))"
    )
    seen = set()
    files: List[str] = []
    for match in _re.findall(pattern, text, flags=_re.IGNORECASE):
        name = match.strip("`'\"，。；;:：、()（）[]【】")
        if not name or name in seen:
            continue
        seen.add(name)
        files.append(name)
    return files


def _check_file_operation_truth(
    response: str,
    user_input: str,
    tool_results_buffer: List[Dict],
) -> str:
    """Validate that file-operation success claims are backed by tool results.

    Returns:
        "ok"    - no guard needed, or a successful file tool result exists.
        "retry" - ask the model to call a real file tool.
        "fail"  - repeated attempts failed; return an explicit failure.
    """
    text = (user_input or "").lower()
    if not text:
        return "ok"
    if any(k in text for k in (
        "不要修改", "不修改", "无需修改", "不要创建", "不要写",
        "只读", "仅分析", "只分析",
    )):
        return "ok"
    requested = any(k in text for k in (
        "文件", "文件夹", "目录", "写入", "创建", "新建", "命名为",
        "命名成", "生成", "修改", "删除", "保存到",
        "file", "folder", "directory", "write", "create", "delete",
    ))
    success_claim = any(k in (response or "") for k in (
        "已创建", "创建成功", "已写入", "写入成功", "已保存",
        "已删除", "删除成功", "已应用文件变更", "已创建文件夹",
        "created", "written", "saved", "deleted",
    ))
    if not requested or not success_claim:
        return "ok"

    file_tools = {
        "exec_write_file",
        "ide_write_file",
        "write_to_file",
        "replace_in_file",
        "delete_file",
        "create_directory",
    }
    saw_file_tool = False
    for item in tool_results_buffer or []:
        tool_name = item.get("tool_name")
        result_text = str(item.get("result") or "")
        if tool_name not in file_tools:
            continue
        saw_file_tool = True
        result_lower = result_text.lower()
        if any(k in result_text for k in (
            "文件已写入", "已应用文件变更", "已应用文件替换",
            "已删除文件", "已创建文件夹",
        )) or (
            '"error"' not in result_lower
            and "error" not in result_lower[:120]
            and "未允许" not in result_text
            and "未应用" not in result_text
            and "失败" not in result_text[:120]
        ):
            return "ok"

    return "fail" if saw_file_tool else "retry"


def _merge_discovered_tool_schemas(state: dict, result_text: str) -> None:
    """Merge tool schemas returned by tool-supplement meta tools into FC state."""
    if not result_text:
        return
    try:
        payload = _json.loads(result_text)
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    schemas = (
        payload.get("supplemented_tool_schemas")
        or payload.get("_discovered_schemas")
        or []
    )
    if not isinstance(schemas, list) or not schemas:
        return

    existing = set()
    for schema in state.get("tool_definitions", []) or []:
        try:
            name = schema.get("function", {}).get("name")
            if name:
                existing.add(name)
        except Exception:
            continue

    added = []
    for schema in schemas:
        if not isinstance(schema, dict):
            continue
        name = schema.get("function", {}).get("name")
        if not name or name in existing:
            continue
        state.setdefault("tool_definitions", []).append(schema)
        existing.add(name)
        added.append(name)

    if added:
        logger.info("[FC][ToolSupplement] 已动态补充工具: %s", added)


def _has_content_match(response: str, node_label: str) -> bool:
    """检查回复中是否包含与节点标签匹配的内容。

    用于 Backfill 前置判断：只有回复中确实存在与节点相关的内容时，
    才能将该节点标记为 completed。避免将兜底截取的内容误判为完成。

    策略：
    1. 精确匹配节点标签
    2. 节点标签中的中文关键词（>=2字）至少匹配到一个
    """
    if not response or not node_label:
        return False

    # 策略1: 精确匹配
    if node_label in response:
        return True

    # 策略2: 2字符滑动窗口匹配（中文无空格分词，取连续2字符子串）
    # 要求至少命中 2 个不同 bigram，减少短文本单个常用词误匹配
    import re as _re
    cjk_runs = _re.findall(r"[\u4e00-\u9fff]{2,}", node_label)
    matched_bigrams = set()
    for run in cjk_runs:
        for i in range(len(run) - 1):
            bigram = run[i:i+2]
            if bigram in response:
                matched_bigrams.add(bigram)
    if len(matched_bigrams) >= 2:
        return True

    return False


def _extract_node_content(response: str, node_label: str, max_len: int = 500) -> str:
    """从回复中提取与节点标签相关的内容片段。

    策略：
    1. 在回复中搜索节点标签（或关键词），提取标签后的 max_len 字符
    2. 如果找不到标签，用节点标签中的核心关键词模糊匹配
    3. 兜底：返回回复前 max_len 字符
    """
    if not response or not node_label:
        return response[:max_len] if response else ""

    # 策略1: 精确匹配节点标签
    idx = response.find(node_label)
    if idx >= 0:
        start = idx
        end = min(len(response), start + max_len)
        # 向后扩展到段落边界（双换行）
        next_section = response.find("\n\n", start + len(node_label))
        if 0 < next_section - start <= max_len:
            end = next_section
        return response[start:end].strip()

    # 策略2: 提取节点标签中的核心关键词进行模糊匹配
    import re as _re
    keywords = _re.findall(r"[\u4e00-\u9fff]{2,}", node_label)
    for kw in keywords:
        idx = response.find(kw)
        if idx >= 0:
            start = max(0, idx - 20)
            end = min(len(response), idx + max_len)
            return response[start:end].strip()

    # 策略3: 兜底 — 返回回复前 max_len 字符
    return response[:max_len]


def _run_async_blocking(coro, timeout: float = 5.0):
    """在同步 FC 节点中运行短异步检测。

    如果当前线程已经有事件循环，就临时放进独立线程，避免 nested event loop。
    """
    try:
        loop = _asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: _asyncio.run(coro)).result(timeout=timeout)

    return _asyncio.run(coro)
