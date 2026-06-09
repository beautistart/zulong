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


class CircuitBreakerState(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class ToolCallRecord:
    """单次工具调用记录"""

    def __init__(self, function_name: str, params_hash: str, result_hash: str,
                 result_len: int, timestamp: float, query: str = ""):
        self.function_name = function_name
        self.params_hash = params_hash
        self.result_hash = result_hash
        self.result_len = result_len
        self.timestamp = timestamp
        self.query = query


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
        self._safety_hard_cap = cfg.get("safety_hard_cap", 100)

        # --- 信号 2: 模式循环 ---
        self._pattern_window = cfg.get("pattern_window", 10)
        self._pattern_yellow_count = cfg.get("pattern_yellow_count", 8)
        self._pattern_red_count = cfg.get("pattern_red_count", 10)
        self._query_similarity_threshold = cfg.get("query_similarity_threshold", 0.85)

        # --- 信号 4: 上下文窗口压力 ---
        self._context_window_size = cfg.get("context_window_size", 131072)
        self._context_yellow_ratio = cfg.get("context_yellow_ratio", 0.50)
        self._context_red_ratio = cfg.get("context_red_ratio", 0.60)

        # --- 信号 5: 经过时间（已禁用） ---
        self._time_yellow_seconds = cfg.get("time_yellow_seconds", 60)
        self._time_red_seconds = cfg.get("time_red_seconds", 120)

        # --- 信号 6: 无进度空转 ---
        self._no_progress_yellow = cfg.get("no_progress_yellow", 5)
        self._no_progress_red = cfg.get("no_progress_red", 8)

        self._max_yellow_before_red = cfg.get("max_yellow_before_red", 4)

        self._call_history: List[ToolCallRecord] = []
        self._start_time: float = 0.0
        self._consecutive_yellow_count: int = 0
        self._planning_mode: bool = False

        if not self.enabled:
            self._safety_hard_cap = 10

    @property
    def safety_hard_cap(self) -> int:
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
        self._safety_hard_cap = cfg.get("safety_hard_cap", 100)
        self._pattern_window = cfg.get("pattern_window", 10)
        self._pattern_yellow_count = cfg.get("pattern_yellow_count", 8)
        self._pattern_red_count = cfg.get("pattern_red_count", 10)
        self._max_yellow_before_red = cfg.get("max_yellow_before_red", 4)
        logger.info(f"[CircuitBreaker] 已重置: hard_cap={self._safety_hard_cap}")

    def record_call(self, function_name: str, params_dict: Dict, result_content: str):
        params_hash = self._hash_dict(params_dict)
        result_hash = self._hash_text(result_content)
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
        signal_names = ["pattern_loop", "context_pressure", "elapsed_time", "no_progress"]
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
