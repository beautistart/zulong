"""
Circuit Breaker - 自适应迭代控制器

取消硬编码的 max_tool_iterations = 10，改为多信号智能死循环检测。
模型正常情况下会自己停止（返回文本不调工具），Circuit Breaker 只是防异常的安全网。

5 个信号：
1. 相同调用重复（function_name + params 完全一致）
2. 模式循环（同一工具被反复调用，搜索查询相似度过高）
3. 信息增益递减（工具返回结果内容重叠率过高）
4. 上下文窗口压力（messages 总 token 接近模型上下文窗口上限）
5. 经过时间（本次推理的墙钟时间）
"""

import hashlib
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    GREEN = "green"    # 正常，继续
    YELLOW = "yellow"  # 警告，注入提示消息
    RED = "red"        # 强制停止，生成最终答案


class ToolCallRecord:
    """单次工具调用记录"""

    def __init__(self, function_name: str, params_hash: str, result_hash: str,
                 result_len: int, timestamp: float, query: str = ""):
        self.function_name = function_name
        self.params_hash = params_hash
        self.result_hash = result_hash
        self.result_len = result_len
        self.timestamp = timestamp
        self.query = query  # 搜索类工具的查询字符串（用于相似度计算）


class ToolCallCircuitBreaker:
    """多信号智能死循环检测器"""

    # 搜索类工具名称（用于信号 2 的查询相似度检测）
    SEARCH_TOOL_NAMES = {
        "web_search", "search", "searxng_search",
        "search_web", "google_search", "bing_search"
    }

    # 搜索查询的参数 key（用于提取查询字符串）
    SEARCH_QUERY_KEYS = {"query", "q", "search_query", "keyword", "keywords"}

    # 规划构建类工具（规划模式下豁免模式循环检测）
    # 包含编排层 (plan_*) 和 IDE 层 (task_*) 两套工具名
    PLANNING_TOOL_NAMES = {
        "plan_add_node", "plan_mark_status", "plan_add_dependency",
        "task_add_node", "task_mark_status", "task_create_plan",
        "view_graph_overview", "task_view_overview",
        "exec_write_file", "exec_run_command",
        "submit_final_answer", "start_task_plan"
    }

    # 信息检索类工具（只读，不推进任务进度）
    INFO_RETRIEVAL_TOOLS = {
        "recall_memory", "search_experience", "read_memory_node",
        "search_memory", "search_tools", "task_view_overview",
        "web_search", "search", "searxng_search",
    }

    # 行动类工具（实际推进任务进度）
    ACTION_TOOLS = {
        "exec_run_command", "exec_write_file", "task_mark_status",
        "task_add_node", "task_create_plan", "submit_final_answer",
        "navigate_attention", "save_memory_note", "delete_memory_node",
        "delete_memory_edge", "set_importance",
    }

    # 终结类工具（重复调用豁免 _signal_repetition 检测）
    # 模型重试这些工具通常是因为安全网拦截后被迫继续，不是死循环
    TERMINAL_TOOLS = {
        "submit_final_answer", "attempt_completion",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self._config = cfg  # 保存原始配置，供克隆时复用
        self.enabled = cfg.get("enabled", True)

        # 安全硬顶：绝对上限
        self._safety_hard_cap = cfg.get("safety_hard_cap", 100)

        # --- 信号 1: 相同调用重复 ---
        self._repetition_window = cfg.get("repetition_window", 3)

        # --- 信号 2: 模式循环 ---
        self._pattern_window = cfg.get("pattern_window", 6)
        self._pattern_yellow_count = cfg.get("pattern_yellow_count", 5)
        self._pattern_red_count = cfg.get("pattern_red_count", 7)
        self._query_similarity_threshold = cfg.get("query_similarity_threshold", 0.7)

        # --- 信号 3: 信息增益递减 ---
        self._info_gain_window = cfg.get("info_gain_window", 3)
        self._content_overlap_threshold = cfg.get("content_overlap_threshold", 0.8)

        # --- 信号 4: 上下文窗口压力 ---
        self._context_window_size = cfg.get("context_window_size", 65536)
        self._context_yellow_ratio = cfg.get("context_yellow_ratio", 0.75)
        self._context_red_ratio = cfg.get("context_red_ratio", 0.90)

        # --- 信号 5: 经过时间 ---
        self._time_yellow_seconds = cfg.get("time_yellow_seconds", 60)
        self._time_red_seconds = cfg.get("time_red_seconds", 120)

        # --- 信号 6: 无进度空转（连续调用信息检索工具而无行动工具） ---
        self._no_progress_yellow = cfg.get("no_progress_yellow", 4)
        self._no_progress_red = cfg.get("no_progress_red", 6)

        # --- 升级策略 ---
        self._max_yellow_before_red = cfg.get("max_yellow_before_red", 2)

        # 运行时状态
        self._call_history: List[ToolCallRecord] = []
        self._start_time: float = 0.0
        self._consecutive_yellow_count: int = 0
        self._planning_mode: bool = False  # 规划模式标志

        # 如果禁用，退化为旧版行为
        if not self.enabled:
            self._safety_hard_cap = 10

    @property
    def safety_hard_cap(self) -> int:
        return self._safety_hard_cap

    def reset(self):
        """每次推理开始前调用，清空状态"""
        self._call_history.clear()
        self._start_time = time.time()
        self._consecutive_yellow_count = 0

    def escalate_for_planning(self):
        """规划模式：放宽模式检测（由 COMPLEX 意图或 start_task_plan 工具触发）"""
        self._planning_mode = True
        # 放宽模式检测阈值：规划模式下允许更多连续调用
        self._pattern_window = 20
        self._pattern_yellow_count = 15
        self._pattern_red_count = 20
        self._max_yellow_before_red = 5
        logger.info(
            f"[CircuitBreaker] 已切换到规划模式: "
            f"hard_cap={self._safety_hard_cap}, "
            f"pattern_window={self._pattern_window}, "
            f"pattern_yellow={self._pattern_yellow_count}"
        )

    def escalate_for_resume(self):
        """恢复模式：4B 模型需要逐节点处理，放宽模式检测"""
        self._planning_mode = True
        # 恢复模式下 task_mark_status 会重复调用，放宽模式检测
        self._pattern_window = 20
        self._pattern_yellow_count = 15
        self._pattern_red_count = 20
        self._max_yellow_before_red = 5
        logger.info(
            f"[CircuitBreaker] 已切换到恢复模式: "
            f"hard_cap={self._safety_hard_cap}, "
            f"pattern_yellow={self._pattern_yellow_count}"
        )

    def reset_to_default(self):
        """重置为默认预算（规划完成或异常退出时调用）"""
        self._planning_mode = False
        cfg = {}
        try:
            from zulong.config.config_manager import ConfigManager
            cfg = ConfigManager().get("l2_inference.circuit_breaker", {})
        except Exception:
            pass
        self._safety_hard_cap = cfg.get("safety_hard_cap", 100)
        self._pattern_window = cfg.get("pattern_window", 6)
        self._pattern_yellow_count = cfg.get("pattern_yellow_count", 5)
        self._pattern_red_count = cfg.get("pattern_red_count", 7)
        self._max_yellow_before_red = cfg.get("max_yellow_before_red", 2)
        logger.info(
            f"[CircuitBreaker] 已重置为默认模式: "
            f"hard_cap={self._safety_hard_cap}"
        )

    def record_call(self, function_name: str, params_dict: Dict, result_content: str):
        """记录一次工具调用（在 evaluate 前调用）"""
        params_hash = self._hash_dict(params_dict)
        result_hash = self._hash_text(result_content)

        # 提取搜索查询字符串
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
            query=query
        )
        self._call_history.append(record)

    def evaluate(self, iteration: int, messages: List[Dict],
                 attn_usage_ratio: float = -1.0) -> Tuple[CircuitBreakerState, str]:
        """综合 6 个信号，返回状态和原因

        Args:
            attn_usage_ratio: AttentionWindow.usage_ratio，若>=0则作为上下文压力信号源
        """
        if not self.enabled:
            return CircuitBreakerState.GREEN, ""

        signals: List[Tuple[CircuitBreakerState, str]] = [
            self._signal_repetition(),
            self._signal_pattern_loop(),
            self._signal_info_gain(),
            self._signal_context_pressure(messages, attn_usage_ratio=attn_usage_ratio),
            self._signal_elapsed_time(),
            self._signal_no_progress(),
        ]

        # 收集所有非 GREEN 信号
        reds = [(s, r) for s, r in signals if s == CircuitBreakerState.RED]
        yellows = [(s, r) for s, r in signals if s == CircuitBreakerState.YELLOW]

        # 任一 RED → 返回 RED
        if reds:
            reasons = "; ".join(r for _, r in reds)
            self._consecutive_yellow_count = 0
            logger.warning(f"[CircuitBreaker] RED 触发 (iter={iteration}): {reasons}")
            self._publish_state_change(CircuitBreakerState.RED, reasons, iteration)
            return CircuitBreakerState.RED, reasons

        # 任一 YELLOW → 累计 yellow，连续 N 次 YELLOW 升级为 RED
        if yellows:
            self._consecutive_yellow_count += 1
            reasons = "; ".join(r for _, r in yellows)

            if self._consecutive_yellow_count >= self._max_yellow_before_red:
                upgrade_reason = f"连续 {self._consecutive_yellow_count} 次 YELLOW 警告，升级为 RED: {reasons}"
                logger.warning(f"[CircuitBreaker] YELLOW→RED 升级 (iter={iteration}): {upgrade_reason}")
                self._consecutive_yellow_count = 0
                self._publish_state_change(CircuitBreakerState.RED, upgrade_reason, iteration)
                return CircuitBreakerState.RED, upgrade_reason

            logger.info(f"[CircuitBreaker] YELLOW (iter={iteration}, 连续第{self._consecutive_yellow_count}次): {reasons}")
            self._publish_state_change(CircuitBreakerState.YELLOW, reasons, iteration)
            return CircuitBreakerState.YELLOW, reasons

        # 全部 GREEN
        self._consecutive_yellow_count = 0
        return CircuitBreakerState.GREEN, ""

    def _publish_state_change(self, state: CircuitBreakerState, reason: str, iteration: int):
        """发布 CB 状态变更事件到 EventBus → WebBridge → 仪表盘"""
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, ZulongEvent
            payload = {
                "component": "CircuitBreaker",
                "state": state.value,
                "reason": reason,
                "iteration": iteration,
                "call_history_len": len(self._call_history),
                "planning_mode": self._planning_mode,
                "timestamp": time.time(),
            }
            event = ZulongEvent(type=EventType("SYSTEM_STATUS"), payload=payload)
            event_bus.publish(event)
        except Exception:
            pass

    # ==================== 信号评估器 ====================

    def _signal_repetition(self) -> Tuple[CircuitBreakerState, str]:
        """信号 1: 相同调用重复（function_name + params 完全一致）"""
        if len(self._call_history) < 2:
            return CircuitBreakerState.GREEN, ""

        window = self._call_history[-self._repetition_window:]
        if len(window) < 2:
            return CircuitBreakerState.GREEN, ""

        # 终结类工具 + 规划类工具豁免
        if window[-1].function_name in self.TERMINAL_TOOLS | self.PLANNING_TOOL_NAMES:
            return CircuitBreakerState.GREEN, ""

        # 检查最近 N 次是否 function_name + params_hash 完全一致
        signatures = [(r.function_name, r.params_hash) for r in window]

        if len(window) >= 3 and len(set(signatures)) == 1:
            return CircuitBreakerState.RED, f"连续 {len(window)} 次完全相同的工具调用: {window[-1].function_name}"

        if len(window) >= 2 and signatures[-1] == signatures[-2]:
            return CircuitBreakerState.YELLOW, f"连续 2 次相同调用: {window[-1].function_name}"

        return CircuitBreakerState.GREEN, ""

    def _signal_pattern_loop(self) -> Tuple[CircuitBreakerState, str]:
        """信号 2: 模式循环（同一工具被反复调用，或搜索查询过于相似）"""
        if len(self._call_history) < self._pattern_window:
            return CircuitBreakerState.GREEN, ""

        window = self._call_history[-self._pattern_window:]

        # 2a: 同一工具在窗口内出现次数过多
        from collections import Counter
        tool_counts = Counter(r.function_name for r in window)
        for tool_name, count in tool_counts.items():
            # 规划/行动类工具完全豁免模式循环检测
            # 超复杂长程任务中，task_add_node 等可能调用上万次
            if tool_name in self.PLANNING_TOOL_NAMES:
                continue

            if count >= self._pattern_red_count:
                return CircuitBreakerState.RED, f"模式循环: {tool_name} 在最近 {self._pattern_window} 次调用中出现 {count} 次"
            if count >= self._pattern_yellow_count:
                # 2b: 搜索类工具进一步检查查询相似度
                if self._is_search_tool(tool_name):
                    recent_queries = [r.query for r in window if r.function_name == tool_name and r.query]
                    if len(recent_queries) >= 3:
                        # 检查最近 3 个查询的两两相似度
                        high_sim_count = 0
                        for i in range(len(recent_queries) - 1):
                            sim = self._query_jaccard(recent_queries[i], recent_queries[i + 1])
                            if sim > self._query_similarity_threshold:
                                high_sim_count += 1
                        if high_sim_count >= 2:
                            return CircuitBreakerState.RED, f"搜索循环: {tool_name} 最近查询高度相似 (相似度>{self._query_similarity_threshold})"

                return CircuitBreakerState.YELLOW, f"模式警告: {tool_name} 在最近 {self._pattern_window} 次中出现 {count} 次"

        return CircuitBreakerState.GREEN, ""

    def _signal_info_gain(self) -> Tuple[CircuitBreakerState, str]:
        """信号 3: 信息增益递减（工具返回结果内容重叠率过高或完全相同）"""
        if len(self._call_history) < self._info_gain_window:
            return CircuitBreakerState.GREEN, ""

        window = self._call_history[-self._info_gain_window:]

        # 3a: 最近 N 次结果 hash 完全相同 → RED
        result_hashes = [r.result_hash for r in window]
        if len(set(result_hashes)) == 1 and all(r.result_len > 0 for r in window):
            return CircuitBreakerState.RED, f"信息增益为零: 最近 {self._info_gain_window} 次工具返回内容完全相同"

        # 3b: 结果全部为空或极短 → YELLOW
        if all(r.result_len < 10 for r in window):
            return CircuitBreakerState.YELLOW, f"信息增益极低: 最近 {self._info_gain_window} 次工具返回内容均为空或极短"

        return CircuitBreakerState.GREEN, ""

    def _signal_context_pressure(self, messages: List[Dict], attn_usage_ratio: float = -1.0) -> Tuple[CircuitBreakerState, str]:
        """信号 4: 上下文窗口压力

        优先使用 AttentionWindow.usage_ratio 作为统一压力源，
        避免与 AttentionWindow 判断不一致。
        回退: 若 attn_usage_ratio < 0，则独立估算 tokens 比率。
        """
        if attn_usage_ratio >= 0:
            ratio = attn_usage_ratio
            source = "AW.usage_ratio"
        else:
            total_tokens = self._estimate_messages_tokens(messages)
            ratio = total_tokens / self._context_window_size if self._context_window_size > 0 else 0
            source = f"独立估算({total_tokens}t/{self._context_window_size})"

        if ratio >= self._context_red_ratio:
            return CircuitBreakerState.RED, f"上下文窗口压力过高: {ratio:.0%} (≥{self._context_red_ratio:.0%}) [{source}]"
        if ratio >= self._context_yellow_ratio:
            return CircuitBreakerState.YELLOW, f"上下文窗口压力警告: {ratio:.0%} (≥{self._context_yellow_ratio:.0%}) [{source}]"

        return CircuitBreakerState.GREEN, ""

    def _signal_elapsed_time(self) -> Tuple[CircuitBreakerState, str]:
        """信号 5: 经过时间 — 已禁用，仅依赖步数控制收敛"""
        return CircuitBreakerState.GREEN, ""

    def _signal_no_progress(self) -> Tuple[CircuitBreakerState, str]:
        """信号 6: 无进度空转（连续调用信息检索工具而无任何行动工具）"""
        if len(self._call_history) < self._no_progress_yellow:
            return CircuitBreakerState.GREEN, ""

        # 从最近的调用记录中反向查找最后一次行动类工具调用
        tail = self._call_history[-self._no_progress_red:]
        consecutive_info = 0
        for record in reversed(tail):
            if record.function_name in self.ACTION_TOOLS:
                break
            if record.function_name in self.INFO_RETRIEVAL_TOOLS:
                consecutive_info += 1
            else:
                # 未知工具，不计入信息检索
                break

        if consecutive_info >= self._no_progress_red:
            return CircuitBreakerState.RED, (
                f"无进度空转: 连续 {consecutive_info} 次调用信息检索工具，"
                f"未执行任何行动工具"
            )
        if consecutive_info >= self._no_progress_yellow:
            return CircuitBreakerState.YELLOW, (
                f"无进度警告: 连续 {consecutive_info} 次调用信息检索工具，"
                f"请执行实际任务工作"
            )

        return CircuitBreakerState.GREEN, ""

    # ==================== 序列化支持（Phase 2 挂起用） ====================

    def serialize(self) -> Dict[str, Any]:
        """导出内部状态（供任务挂起时保存）"""
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
        """恢复内部状态（供任务恢复时加载）

        关键修复：恢复时重置 _start_time 为当前时间，避免挂起期间的
        等待时间被计入经过时间信号，导致恢复后立即触发 RED。
        同时重置连续 YELLOW 计数器，新推理周期不应继承旧的累计值。
        """
        self._call_history = [
            ToolCallRecord(**record)
            for record in state.get("call_history", [])
        ]
        # 重置为当前时间 —— 新的推理周期开始
        self._start_time = time.time()
        # 重置黄警计数器 —— 新周期不继承旧的累计值
        self._consecutive_yellow_count = 0
        # 恢复规划模式
        if state.get("planning_mode"):
            self.escalate_for_planning()
        logger.info(
            f"[CircuitBreaker] 状态已恢复: "
            f"call_history={len(self._call_history)} 条, "
            f"planning_mode={self._planning_mode}, "
            f"start_time=now (已重置)"
        )

    # ==================== 工具方法 ====================

    @staticmethod
    def _hash_dict(d: Dict) -> str:
        """对字典计算稳定 hash"""
        try:
            s = json.dumps(d, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            s = str(d)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _hash_text(text: str) -> str:
        """对文本计算 hash"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _query_jaccard(a: str, b: str) -> float:
        """Jaccard 相似度（用于搜索查询比较）"""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        union = words_a | words_b
        if not union:
            return 0.0
        return len(words_a & words_b) / len(union)

    def _is_search_tool(self, name: str) -> bool:
        """判断是否为搜索类工具"""
        name_lower = name.lower()
        return name_lower in self.SEARCH_TOOL_NAMES or "search" in name_lower

    def _estimate_messages_tokens(self, messages: List[Dict]) -> int:
        """估算 messages 的 token 数（复用项目中已有的公式）"""
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
        """估算文本的 token 数：中文字符 x 1.5 + 英文单词 x 0.75"""
        if not text:
            return 0
        # 中文字符数
        cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        # 英文单词数（去掉中文后按空格分割）
        en_text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
        en_words = len(en_text.split())
        return int(cn_chars * 1.5 + en_words * 0.75)
