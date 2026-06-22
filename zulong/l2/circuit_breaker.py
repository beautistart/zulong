"""
Circuit Breaker - 自适应迭代控制器

取消硬编码的 max_tool_iterations = 10，改为多信号智能死循环检测。
模型正常情况下会自己停止（返回文本不调工具），Circuit Breaker 只是防异常的安全网。

4 个活跃信号（v8 — 删除信号1/3重复调用限制）:
2. 模式循环（同一工具被反复调用，搜索查询相似度过高）
4. 上下文窗口压力（messages 总 token 接近模型上下文窗口上限）
5. 经过时间（已禁用，仅依赖步数控制收敛）
6. 无进度空转（连续调用信息检索工具而无行动工具）

已删除:
1. 相同调用重复 — 不同参数但相同失误被漏检，过于机械
3. 信息增益递减 — 纯哈希比对，错误消息含变量时失效
"""

import hashlib
import json
import logging
import re
import time
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _cfg_float(value: Any, default: float) -> float:
    """Return a numeric config value even when YAML/env provided a string."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        logger.warning(
            "[CircuitBreaker] Invalid float config %r, fallback=%s",
            value,
            default,
        )
        return float(default)


def _cfg_int(value: Any, default: int) -> int:
    """Return an integer config value even when YAML/env provided a string."""
    try:
        if value is None:
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        logger.warning(
            "[CircuitBreaker] Invalid int config %r, fallback=%s",
            value,
            default,
        )
        return int(default)


def _cfg_optional_int(value: Any, default: Optional[int]) -> Optional[int]:
    """Parse optional integer limits; common disabled sentinels become None."""
    disabled = {None, 0, -1, "0", "-1", "none", "disabled", "off", "unlimited"}
    if isinstance(value, str):
        value = value.strip().lower()
    if value in disabled:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class CircuitBreakerState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class ToolCallRecord:
    """单次工具调用记录"""

    def __init__(self, function_name: str, params_hash: str, result_hash: str,
                 result_len: int, timestamp: float, query: str = "",
                 result_preview: str = "", normalized_result_hash: str = ""):
        self.function_name = function_name
        self.params_hash = params_hash
        self.result_hash = result_hash
        self.result_len = result_len
        self.timestamp = timestamp
        self.query = query
        self.result_preview = result_preview
        self.normalized_result_hash = normalized_result_hash or result_hash


class ToolCallCircuitBreaker:
    """多信号智能死循环检测器 (v8 — 删除机械重复)"""

    SEARCH_TOOL_NAMES = {
        "web_search", "search", "searxng_search",
        "search_web", "google_search", "bing_search"
    }

    SEARCH_QUERY_KEYS = {"query", "q", "search_query", "keyword", "keywords"}

    PLANNING_TOOL_NAMES = {
        "plan_add_node", "plan_mark_status", "plan_add_dependency",
        "task_add_node", "task_mark_status", "task_create_plan",
        "view_graph_overview", "task_view_overview",
        "exec_write_file", "exec_run_command",
        "submit_final_answer", "start_task_plan"
    }

    INFO_RETRIEVAL_TOOLS = {
        "recall_memory", "search_experience", "read_memory_node",
        "search_memory", "search_tools", "task_view_overview",
        "web_search", "search", "searxng_search",
    }

    ACTION_TOOLS = {
        "exec_run_command", "exec_write_file", "task_mark_status",
        "task_add_node", "task_create_plan", "submit_final_answer",
        "navigate_attention", "save_memory_note", "delete_memory_node",
        "delete_memory_edge", "set_importance",
    }

    BATCH_ACTION_TOOLS = {
        "delete_memory_node", "delete_memory_edge", "set_importance",
        "save_memory_note", "task_mark_status", "task_add_node",
        "task_update_node", "task_update_content", "task_attach_file",
        "task_remove_node",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._config = cfg
        self.enabled = cfg.get("enabled", True)
        self._safety_hard_cap = _cfg_optional_int(cfg.get("safety_hard_cap", 100), 100)

        # --- 信号 2: 模式循环 ---
        self._pattern_window = _cfg_int(cfg.get("pattern_window", 10), 10)
        self._pattern_yellow_count = _cfg_int(cfg.get("pattern_yellow_count", 8), 8)
        self._pattern_red_count = _cfg_int(cfg.get("pattern_red_count", 10), 10)
        self._query_similarity_threshold = _cfg_float(cfg.get("query_similarity_threshold", 0.85), 0.85)

        # --- 信号 4: 上下文窗口压力 ---
        self._context_window_size = _cfg_int(cfg.get("context_window_size", 131072), 131072)
        self._context_yellow_ratio = _cfg_float(cfg.get("context_yellow_ratio", 0.50), 0.50)
        self._context_red_ratio = _cfg_float(cfg.get("context_red_ratio", 0.60), 0.60)

        # --- 信号 5: 经过时间（已禁用） ---
        self._time_yellow_seconds = _cfg_float(cfg.get("time_yellow_seconds", 60), 60)
        self._time_red_seconds = _cfg_float(cfg.get("time_red_seconds", 120), 120)

        # --- 信号 6: 无进度空转 ---
        self._no_progress_yellow = _cfg_int(cfg.get("no_progress_yellow", 5), 5)
        self._no_progress_red = _cfg_int(cfg.get("no_progress_red", 8), 8)

        self._max_yellow_before_red = _cfg_int(cfg.get("max_yellow_before_red", 5), 5)

        self._call_history: List[ToolCallRecord] = []
        self._start_time: float = 0.0
        self._consecutive_yellow_count: int = 0
        self._planning_mode: bool = False

        if not self.enabled:
            self._safety_hard_cap = 10

    @property
    def safety_hard_cap(self) -> Optional[int]:
        return self._safety_hard_cap

    def reset(self):
        self._call_history.clear()
        self._start_time = time.time()
        self._consecutive_yellow_count = 0

    def escalate_for_planning(self):
        self._planning_mode = True
        self._pattern_window = 20
        self._pattern_yellow_count = 15
        self._pattern_red_count = 20
        self._max_yellow_before_red = 5
        logger.info(
            f"[CircuitBreaker] 已切换到规划模式: "
            f"hard_cap={self._safety_hard_cap}"
        )

    def escalate_for_resume(self):
        self._planning_mode = True
        self._pattern_window = 20
        self._pattern_yellow_count = 15
        self._pattern_red_count = 20
        self._max_yellow_before_red = 5
        logger.info(
            f"[CircuitBreaker] 已切换到恢复模式: "
            f"hard_cap={self._safety_hard_cap}"
        )

    def reset_to_default(self):
        self._planning_mode = False
        cfg = {}
        try:
            from zulong.config.config_manager import ConfigManager
            cfg = ConfigManager().get("l2_inference.circuit_breaker", {})
        except Exception:
            pass
        self._safety_hard_cap = _cfg_optional_int(cfg.get("safety_hard_cap", 100), 100)
        self._pattern_window = _cfg_int(cfg.get("pattern_window", 10), 10)
        self._pattern_yellow_count = _cfg_int(cfg.get("pattern_yellow_count", 8), 8)
        self._pattern_red_count = _cfg_int(cfg.get("pattern_red_count", 10), 10)
        self._query_similarity_threshold = _cfg_float(cfg.get("query_similarity_threshold", 0.85), 0.85)
        self._context_window_size = _cfg_int(cfg.get("context_window_size", 131072), 131072)
        self._context_yellow_ratio = _cfg_float(cfg.get("context_yellow_ratio", 0.50), 0.50)
        self._context_red_ratio = _cfg_float(cfg.get("context_red_ratio", 0.60), 0.60)
        self._time_yellow_seconds = _cfg_float(cfg.get("time_yellow_seconds", 60), 60)
        self._time_red_seconds = _cfg_float(cfg.get("time_red_seconds", 120), 120)
        self._no_progress_yellow = _cfg_int(cfg.get("no_progress_yellow", 5), 5)
        self._no_progress_red = _cfg_int(cfg.get("no_progress_red", 8), 8)
        self._max_yellow_before_red = _cfg_int(cfg.get("max_yellow_before_red", 5), 5)
        logger.info(f"[CircuitBreaker] 已重置: hard_cap={self._safety_hard_cap}")

    def record_call(self, function_name: str, params_dict: Dict, result_content: str):
        params_hash = self._hash_dict(params_dict)
        result_hash = self._hash_text(result_content)
        result_preview = str(result_content or "")[:200]
        normalized_result_hash = self._hash_text(
            self._normalize_result_text(result_content)
        )
        query = ""
        if function_name.lower() in self.SEARCH_TOOL_NAMES or "search" in function_name.lower():
            for key in self.SEARCH_QUERY_KEYS:
                if key in params_dict:
                    query = str(params_dict[key])
                    break
        record = ToolCallRecord(
            function_name=function_name,
            params_hash=params_hash,
            result_hash=result_hash,
            result_len=len(result_content),
            timestamp=time.time(),
            query=query,
            result_preview=result_preview,
            normalized_result_hash=normalized_result_hash,
        )
        self._call_history.append(record)

    def evaluate(self, iteration: int, messages: List[Dict],
                 attn_usage_ratio: float = -1.0) -> Tuple[CircuitBreakerState, str]:
        if not self.enabled:
            return CircuitBreakerState.GREEN, ""

        signals: List[Tuple[CircuitBreakerState, str]] = [
            self._signal_pattern_loop(),
            self._signal_context_pressure(messages, attn_usage_ratio=attn_usage_ratio),
            self._signal_elapsed_time(),
            self._signal_no_progress(),
            self._signal_repeating_result(),
            self._signal_alternating_results(),
            self._signal_consecutive_errors(),
        ]

        reds = [(s, r) for s, r in signals if s == CircuitBreakerState.RED]
        yellows = [(s, r) for s, r in signals if s == CircuitBreakerState.YELLOW]

        if reds:
            reasons = "; ".join(r for _, r in reds)
            self._consecutive_yellow_count = 0
            # 增强日志：包含调用历史摘要和各信号得分
            call_summary = self._get_call_history_summary()
            signal_scores = self._get_signal_scores(signals)
            logger.warning(
                f"[CircuitBreaker] RED (iter={iteration}): {reasons} | "
                f"signals={signal_scores} | history={call_summary}"
            )
            self._publish_state_change(CircuitBreakerState.RED, reasons, iteration)
            return CircuitBreakerState.RED, reasons

        if yellows:
            self._consecutive_yellow_count += 1
            reasons = "; ".join(r for _, r in yellows)
            if self._consecutive_yellow_count >= self._max_yellow_before_red:
                upgrade = f"连续 {self._consecutive_yellow_count} 次 YELLOW → RED: {reasons}"
                logger.warning(f"[CircuitBreaker] {upgrade}")
                self._consecutive_yellow_count = 0
                self._publish_state_change(CircuitBreakerState.RED, upgrade, iteration)
                return CircuitBreakerState.RED, upgrade
            logger.info(f"[CircuitBreaker] YELLOW (iter={iteration}, ×{self._consecutive_yellow_count}): {reasons}")
            self._publish_state_change(CircuitBreakerState.YELLOW, reasons, iteration)
            return CircuitBreakerState.YELLOW, reasons

        self._consecutive_yellow_count = 0
        return CircuitBreakerState.GREEN, ""

    def _publish_state_change(self, state: CircuitBreakerState, reason: str, iteration: int):
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, ZulongEvent
            event_bus.publish(ZulongEvent(
                type=EventType("SYSTEM_STATUS"),
                payload={
                    "component": "CircuitBreaker",
                    "state": state.value,
                    "reason": reason,
                    "iteration": iteration,
                    "call_history_len": len(self._call_history),
                    "planning_mode": self._planning_mode,
                    "timestamp": time.time(),
                },
            ))
        except Exception:
            pass

    # ==================== 信号 2: 模式循环 ====================

    def _signal_pattern_loop(self) -> Tuple[CircuitBreakerState, str]:
        if len(self._call_history) < self._pattern_window:
            return CircuitBreakerState.GREEN, ""

        window = self._call_history[-self._pattern_window:]
        tool_counts = Counter(r.function_name for r in window)
        for tool_name, count in tool_counts.items():
            if tool_name in self.PLANNING_TOOL_NAMES:
                continue
            tool_records = [r for r in window if r.function_name == tool_name]
            if self._is_search_tool(tool_name):
                state, reason = self._signal_search_pattern(tool_name, tool_records)
                if state != CircuitBreakerState.GREEN:
                    return state, reason
                continue
            if (
                tool_name in self.BATCH_ACTION_TOOLS
                and self._is_diverse_action_batch(tool_records)
            ):
                continue
            if count >= self._pattern_red_count:
                return CircuitBreakerState.RED, (
                    f"模式循环: {tool_name} "
                    f"在最近 {self._pattern_window} 次调用中出现 {count} 次"
                )
            if count >= self._pattern_yellow_count:
                return CircuitBreakerState.YELLOW, (
                    f"模式警告: {tool_name} "
                    f"在最近 {self._pattern_window} 次中出现 {count} 次"
                )
        return CircuitBreakerState.GREEN, ""

    def _signal_search_pattern(
        self,
        tool_name: str,
        records: List[ToolCallRecord],
    ) -> Tuple[CircuitBreakerState, str]:
        count = len(records)
        if count < self._pattern_yellow_count:
            return CircuitBreakerState.GREEN, ""
        queries = [r.query for r in records if r.query]
        if len(queries) < 3:
            return CircuitBreakerState.GREEN, ""
        high = sum(
            1
            for i in range(len(queries) - 1)
            if self._query_jaccard(queries[i], queries[i + 1])
            > self._query_similarity_threshold
        )
        if high >= 2:
            return CircuitBreakerState.RED, (
                f"搜索循环: {tool_name} "
                f"最近查询高度相似 (>{self._query_similarity_threshold})"
            )
        return CircuitBreakerState.GREEN, ""

    @staticmethod
    def _is_diverse_action_batch(records: List[ToolCallRecord]) -> bool:
        """Allow legitimate batch operations with varied targets/results.

        Repeated identical action calls are still caught by the pattern signal;
        varied delete/update batches should continue so cleanup tasks can finish.
        """
        count = len(records)
        if count < 4:
            return False
        param_counts = Counter(r.params_hash for r in records)
        result_counts = Counter(r.result_hash for r in records)
        most_common_param = param_counts.most_common(1)[0][1] / count
        most_common_result = result_counts.most_common(1)[0][1] / count
        if most_common_param >= 0.80:
            return False
        if most_common_param >= 0.60 and most_common_result >= 0.60:
            return False
        diversity_floor = max(3, count // 2)
        return (
            len(param_counts) >= diversity_floor
            or len(result_counts) >= diversity_floor
        )

    # ==================== 信号 4: 上下文窗口压力 ====================

    def _signal_context_pressure(self, messages: List[Dict],
                                  attn_usage_ratio: float = -1.0) -> Tuple[CircuitBreakerState, str]:
        if attn_usage_ratio >= 0:
            ratio = attn_usage_ratio
            source = "AW.usage_ratio"
        else:
            total_tokens = self._estimate_messages_tokens(messages)
            ratio = total_tokens / self._context_window_size if self._context_window_size > 0 else 0
            source = f"独立估算({total_tokens}t/{self._context_window_size})"
        if ratio >= self._context_red_ratio:
            return CircuitBreakerState.RED, (
                f"上下文压力过高: {ratio:.0%} (≥{self._context_red_ratio:.0%}) [{source}]"
            )
        if ratio >= self._context_yellow_ratio:
            return CircuitBreakerState.YELLOW, (
                f"上下文压力警告: {ratio:.0%} (≥{self._context_yellow_ratio:.0%}) [{source}]"
            )
        return CircuitBreakerState.GREEN, ""

    # ==================== 信号 5: 经过时间（已禁用） ====================

    def _signal_elapsed_time(self) -> Tuple[CircuitBreakerState, str]:
        return CircuitBreakerState.GREEN, ""

    # ==================== 信号 6: 无进度空转 ====================

    def _signal_no_progress(self) -> Tuple[CircuitBreakerState, str]:
        if len(self._call_history) < self._no_progress_yellow:
            return CircuitBreakerState.GREEN, ""
        # 宽松条件：若最近一次调用了行动工具则重置计数（模型在推进任务）
        if self._call_history and self._call_history[-1].function_name in self.ACTION_TOOLS:
            return CircuitBreakerState.GREEN, ""
        tail = self._call_history[-self._no_progress_red:]
        consecutive_info = 0
        for record in reversed(tail):
            if record.function_name in self.ACTION_TOOLS:
                break
            if record.function_name in self.INFO_RETRIEVAL_TOOLS:
                consecutive_info += 1
            else:
                break
        if consecutive_info >= self._no_progress_red:
            return CircuitBreakerState.RED, (
                f"无进度空转: 连续 {consecutive_info} 次调用信息检索工具，未执行任何行动工具"
            )
        if consecutive_info >= self._no_progress_yellow:
            return CircuitBreakerState.YELLOW, (
                f"无进度警告: 连续 {consecutive_info} 次调用信息检索工具，请执行实际任务"
            )
        return CircuitBreakerState.GREEN, ""

    # ==================== 信号 7-9: 语义级结果模式 ====================

    def _signal_repeating_result(self) -> Tuple[CircuitBreakerState, str]:
        if len(self._call_history) < 4:
            return CircuitBreakerState.GREEN, ""
        window = self._call_history[-6:]
        counts = Counter(
            (r.function_name, r.normalized_result_hash)
            for r in window
            if r.normalized_result_hash
        )
        if not counts:
            return CircuitBreakerState.GREEN, ""
        (tool_name, _), count = counts.most_common(1)[0]
        if count >= 6:
            return CircuitBreakerState.RED, (
                f"结果重复: {tool_name} 最近 6 次返回语义等价结果"
            )
        if count >= 4:
            return CircuitBreakerState.YELLOW, (
                f"结果重复警告: {tool_name} 最近 6 次中 {count} 次返回语义等价结果"
            )
        return CircuitBreakerState.GREEN, ""

    def _signal_alternating_results(self) -> Tuple[CircuitBreakerState, str]:
        if len(self._call_history) < 6:
            return CircuitBreakerState.GREEN, ""

        def signature(record: ToolCallRecord) -> Tuple[str, str]:
            return (record.function_name, record.normalized_result_hash)

        for size, is_red in ((8, True), (6, False)):
            if len(self._call_history) < size:
                continue
            window = self._call_history[-size:]
            even = [signature(r) for r in window[0::2]]
            odd = [signature(r) for r in window[1::2]]
            if len(set(even)) == 1 and len(set(odd)) == 1 and even[0] != odd[0]:
                pairs = size // 2
                state = CircuitBreakerState.RED if is_red else CircuitBreakerState.YELLOW
                return state, (
                    f"交替循环: 最近 {size} 次工具结果呈 A/B 交替模式 ({pairs} 对)"
                )
        return CircuitBreakerState.GREEN, ""

    def _signal_consecutive_errors(self) -> Tuple[CircuitBreakerState, str]:
        if len(self._call_history) < 3:
            return CircuitBreakerState.GREEN, ""
        count = 0
        last_tool = ""
        for record in reversed(self._call_history[-6:]):
            if not self._is_error_result(record.result_preview):
                break
            count += 1
            last_tool = record.function_name
        if count >= 4:
            return CircuitBreakerState.RED, (
                f"连续错误: 最近 {count} 次工具调用均失败，最后工具={last_tool}"
            )
        if count >= 3:
            return CircuitBreakerState.YELLOW, (
                f"连续错误警告: 最近 {count} 次工具调用均失败，请先分析原因"
            )
        return CircuitBreakerState.GREEN, ""

    # ==================== 辅助诊断方法 ====================

    def _get_call_history_summary(self) -> str:
        """生成调用历史摘要用于诊断日志"""
        if not self._call_history:
            return "empty"
        tool_counts = Counter(r.function_name for r in self._call_history)
        top_tools = tool_counts.most_common(5)
        summary_parts = [f"{name}×{count}" for name, count in top_tools]
        total_info = sum(1 for r in self._call_history if r.function_name in self.INFO_RETRIEVAL_TOOLS)
        total_action = sum(1 for r in self._call_history if r.function_name in self.ACTION_TOOLS)
        return (
            f"total={len(self._call_history)}, info={total_info}, action={total_action}, "
            f"top=[{', '.join(summary_parts)}]"
        )

    def _get_signal_scores(self, signals: List[Tuple[CircuitBreakerState, str]]) -> str:
        """生成各信号得分摘要"""
        signal_names = [
            "pattern_loop", "context_pressure", "elapsed_time", "no_progress",
            "repeating_result", "alternating_results", "consecutive_errors",
        ]
        parts = []
        for i, (state, _) in enumerate(signals):
            name = signal_names[i] if i < len(signal_names) else f"sig{i}"
            parts.append(f"{name}={state.value}")
        return ", ".join(parts)

    # ==================== 序列化 ====================

    def serialize(self) -> Dict[str, Any]:
        return {
            "call_history": [
                {
                    "function_name": r.function_name,
                    "params_hash": r.params_hash,
                    "result_hash": r.result_hash,
                    "result_len": r.result_len,
                    "timestamp": r.timestamp,
                    "query": r.query,
                    "result_preview": r.result_preview,
                    "normalized_result_hash": r.normalized_result_hash,
                }
                for r in self._call_history
            ],
            "elapsed_at_suspend": time.time() - self._start_time,
            "consecutive_yellow_count": self._consecutive_yellow_count,
            "planning_mode": self._planning_mode,
        }

    def deserialize(self, state: Dict[str, Any]):
        self._call_history = [
            ToolCallRecord(**record) for record in state.get("call_history", [])
        ]
        self._start_time = time.time()
        self._consecutive_yellow_count = 0
        if state.get("planning_mode"):
            self.escalate_for_planning()
        logger.info(
            f"[CircuitBreaker] 状态已恢复: "
            f"call_history={len(self._call_history)} 条, "
            f"planning_mode={self._planning_mode}"
        )

    # ==================== 工具方法 ====================

    @staticmethod
    def _hash_dict(d: Dict) -> str:
        try:
            s = json.dumps(d, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(d)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _normalize_result_text(cls, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return "empty"

        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                status = data.get("status") or data.get("state") or ""
                success = data.get("success", data.get("ok", None))
                error = data.get("error") or data.get("message") or data.get("detail") or ""
                changed = data.get("changed", data.get("modified", None))
                if success is False or status in {"error", "failed", "failure"}:
                    return "error:" + cls._scrub_result_noise(str(error or status or "failed"))[:300]
                if changed is False:
                    return "no_change:" + cls._scrub_result_noise(str(status or "unchanged"))[:200]
                if success is True or status in {"ok", "success", "succeeded"}:
                    keys = sorted(str(k) for k in data.keys())[:12]
                    return "success:" + ",".join(keys)
        except Exception:
            pass

        scrubbed = cls._scrub_result_noise(raw.lower())
        if cls._contains_error_terms(scrubbed):
            return "error:" + scrubbed[:500]
        if any(term in scrubbed for term in ("no change", "unchanged", "未改变", "无需修改", "没有变化")):
            return "no_change:" + scrubbed[:300]
        if any(term in scrubbed for term in ("success", "succeeded", "ok", "完成", "成功")):
            return "success:" + scrubbed[:500]
        if len(scrubbed) < 30:
            return "short:" + scrubbed
        return "content:" + scrubbed[:500]

    @staticmethod
    def _scrub_result_noise(text: str) -> str:
        scrubbed = text
        scrubbed = re.sub(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", "<uuid>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\b[0-9a-f]{24,64}\b", "<hex>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\b\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b", "<timestamp>", scrubbed)
        scrubbed = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds|秒|毫秒)\b", "<duration>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"(?:line|行|lineno)[:= ]+\d+", "line:<n>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"(?:col|column|列)[:= ]+\d+", "col:<n>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"[a-z]:[\\/][^\s\"']+", "<path>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"/(?:tmp|var|private|users|home)/[^\s\"']+", "<path>", scrubbed, flags=re.I)
        scrubbed = re.sub(r"\b\d{5,}\b", "<num>", scrubbed)
        scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
        return scrubbed

    @classmethod
    def _is_error_result(cls, text: str) -> bool:
        return cls._contains_error_terms(str(text or "").lower())

    @staticmethod
    def _contains_error_terms(text: str) -> bool:
        return any(term in text for term in (
            "error", "failed", "failure", "exception", "traceback",
            "not found", "permission denied", "timeout", "timed out",
            "失败", "错误", "异常", "不存在", "未找到", "拒绝", "超时",
        ))

    @staticmethod
    def _query_jaccard(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        union = words_a | words_b
        if not union:
            return 0.0
        return len(words_a & words_b) / len(union)

    def _is_search_tool(self, name: str) -> bool:
        return name.lower() in self.SEARCH_TOOL_NAMES or "search" in name.lower()

    def _estimate_messages_tokens(self, messages: List[Dict]) -> int:
        total = 0
        for msg in messages:
            content = ""
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
            elif hasattr(msg, "content") and msg.content:
                content = str(msg.content)
            total += self._estimate_tokens(content)
        return total

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
        en_words = len(en_text.split())
        return int(cn_chars * 1.5 + en_words * 0.75)
