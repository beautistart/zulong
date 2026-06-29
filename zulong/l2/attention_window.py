# File: zulong/l2/attention_window.py
"""
动态注意力窗口 (Dynamic Attention Window)

将 Agent 主循环的消息历史控制在模型上下文窗口的安全范围内。

三种注意力模式：
  - GLOBAL: 全局视角，关注大纲和整体结构，深层节点权重递减
  - FOCUS: 聚焦某个节点的细节，提高关联上下文权重
  - SINGLE_CHAIN: 单链推理，优先注入当前执行链路的高权重信息，暂排无关上下文

模式切换不再由普通工具名驱动。上下文压力阈值监控会触发 LLM 自主选择
GLOBAL / FOCUS / SINGLE_CHAIN；LLM 也可以通过显式注意力控制能力
（如 navigate_attention / adjust_attention_mode 或等价工具）切换模式。
普通读写、命令、检索工具只进入工具账本、压力观测和质量证据。
"""

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Set

from zulong.core.message_visibility import strip_llm_message_metadata

# ── LLM自主注意力选择模块导入 ──
from .attention_types import (
    PressureMetrics, DecisionRequest, DecisionResponse,
    SwitchRecord, OscillationState, TriggerType,
    PressureTrend, OscillationLevel,
)
from .attention_config import AttentionConfig
from .pressure_detector import PressureDetector
from .attention_mode_selector import AttentionModeSelector
from .mode_switch_controller import (
    ModeSwitchController, CooldownManager, OscillationDetector
)

logger = logging.getLogger(__name__)


class AttentionMode(Enum):
    """注意力模式"""
    GLOBAL = "global"             # 全局：关注大纲和整体进度
    FOCUS = "focus"               # 聚焦：关注某节点的细节和关联
    SINGLE_CHAIN = "single_chain" # 单链：深度推理，暂排不相关上下文


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数

    复用项目既有公式：中文字符 × 1.5 + 英文单词 × 0.75
    """
    if not text:
        return 0
    cn_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    en_text = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    en_words = len(en_text.split())
    return int(cn_chars * 1.5 + en_words * 0.75)


def _estimate_message_tokens(msg: Dict) -> int:
    """估算单条 OpenAI 格式消息的 token 数"""
    total = 0
    content = msg.get("content", "")
    if content:
        total += estimate_tokens(str(content))
    # tool_calls 中的 arguments 也占 token
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        total += estimate_tokens(fn.get("name", ""))
        total += estimate_tokens(str(fn.get("arguments", "")))
    return max(total, 1)  # 至少 1 token


@dataclass
class MessageEnvelope:
    """消息信封：包装原始消息并附加评分元数据"""
    msg: Dict              # 原始 OpenAI 格式消息
    seq: int               # 全局序号（用于保持时间顺序）
    turn: int              # 所属交互轮次
    tool_name: Optional[str] = None   # 关联工具名
    node_id: Optional[str] = None     # 关联节点 ID
    tokens: int = 0        # 估算 token 数
    is_pinned: bool = False  # 是否钉住（system prompt、goal 等）
    weight: float = 1.0    # 当前权重分数
    # tool_call 原子组标识：同一组的 assistant(tool_calls) + tool(results) 共享
    group_id: Optional[int] = None


# 工具结果最大字符数（超出截断）
MAX_TOOL_RESULT_CHARS = 10000


class AttentionWindowManager:
    """动态注意力窗口管理器

    核心职责：
    1. 接收所有消息（register_message）
    2. 记录工具调用的节点/证据关联；普通工具调用不得直接切换模式
    3. 在 LLM 调用前裁剪消息（apply_window）
    4. 执行 LLM 显式注意力选择或压力触发后的模式切换
    """

    def __init__(
        self,
        context_window_size: int,
        task_graph=None,
        memory_graph=None,
        reserved_tokens: int = 7096,
    ):
        """
        Args:
            context_window_size: 模型上下文窗口大小（tokens）
            task_graph: TaskGraph 实例（用于节点关系查询）
            memory_graph: MemoryGraph 实例（用于激活值融合）
            reserved_tokens: 预留 tokens（system prompt + tools schema + output buffer）
        """
        self.context_window_size = context_window_size
        self.task_graph = task_graph
        self.memory_graph = memory_graph
        self.reserved_tokens = reserved_tokens

        # 可用预算 = (上下文 - 预留) × 90%
        #
        # 注意：这是早期“接近原始 LLM 窗口”的兜底预算。动态注意力启用后，
        # 实际注入给 LLM 的上下文预算以 Web 配置的完整上下文窗口为准；
        # threshold_budget_ratio 仅保留为 1.0 兼容遥测。
        # 否则压力监控按“阈值预算”触发了 GLOBAL/FOCUS/SINGLE_CHAIN 切换，
        # 但 apply_window 仍按原始窗口放行，注意力只会改变 UI 状态，不能真正
        # 暂排非必要上下文。
        self._base_budget = max(
            int((context_window_size - reserved_tokens) * 0.90),
            1024,
        )
        self.budget = self._base_budget

        # P2-13: 动态budget调整因子
        self._budget_multiplier: float = 1.0  # 由任务图节点数动态调整
        self._threshold_budget_base: int = self._base_budget
        self._last_window_tokens: int = 0
        self._last_window_message_count: int = 0
        self._last_evicted_message_count: int = 0
        self._last_window_budget: int = self.budget
        # Per-call dynamic active-context retrieval injection (not persisted/pinned).
        self._last_active_context_extra_tokens: int = 0
        self._last_provider_prompt_tokens: int = 0

        self.mode: AttentionMode = AttentionMode.GLOBAL
        self.envelopes: List[MessageEnvelope] = []
        self._current_turn: int = 0
        self._current_node_id: Optional[str] = None
        self._last_visible_node_ids: Set[str] = set()
        self._seq_counter: int = 0
        self._group_counter: int = 0

        logger.info(
            f"[AttentionWindow] 初始化: context={context_window_size}, "
            f"reserved={reserved_tokens}, budget={self.budget}"
        )

        # ── LLM自主注意力选择组件初始化 ──
        self._llm_config: Optional[AttentionConfig] = None
        self._pressure_detector: Optional[PressureDetector] = None
        self._mode_selector: Optional[AttentionModeSelector] = None
        self._cooldown_manager: Optional[CooldownManager] = None
        self._oscillation_detector: Optional[OscillationDetector] = None
        self._mode_controller: Optional[ModeSwitchController] = None
        self._llm_client: Optional[Any] = None
        self._last_llm_selection_time: Optional[float] = None
        self._last_pressure_metrics: Optional[Any] = None
        self._last_threshold_result: Optional[Any] = None
        self._last_llm_decision: Optional[Any] = None
        self._last_selection_event: Optional[Dict[str, Any]] = None
        self._selection_attempt_count: int = 0
        self._selection_inflight: bool = False
        self._selection_lock = threading.Lock()

        try:
            _cfg = AttentionConfig.load_from_yaml()
            _cfg.validate()

            if _cfg.enabled:
                self._llm_config = _cfg
                self._pressure_detector = PressureDetector(self, self._llm_config)
                self._mode_selector = AttentionModeSelector(self._llm_config)
                self._cooldown_manager = CooldownManager(self._llm_config)
                self._oscillation_detector = OscillationDetector(self._llm_config)
                self._mode_controller = ModeSwitchController(
                    self._llm_config,
                    self._cooldown_manager,
                    self._oscillation_detector,
                )
                logger.info(
                    f"[AttentionWindow] LLM自主注意力选择已启用: "
                    f"pressure_threshold={self._llm_config.pressure_threshold_high}"
                )
            else:
                logger.info("[AttentionWindow] LLM自主注意力选择已禁用")
        except Exception as e:
            logger.warning(f"[AttentionWindow] LLM注意力选择初始化失败(非致命): {e}")

    # ── 消息注册 ──

    def register_message(
        self,
        msg: Dict,
        turn: int,
        tool_name: Optional[str] = None,
        node_id: Optional[str] = None,
        pinned: bool = False,
        group_id: Optional[int] = None,
    ):
        """注册一条消息到窗口

        Args:
            msg: OpenAI 格式消息
            turn: 当前轮次
            tool_name: 关联的工具名（tool result 消息时传入）
            node_id: 关联的节点 ID
            pinned: 是否钉住（永不淘汰）
            group_id: 工具调用组 ID（assistant+tool 同组）
        """
        self._current_turn = max(self._current_turn, turn)

        # 防御性截断：tool result 消息内容超长时截断
        if msg.get("role") == "tool" and msg.get("content"):
            _content = str(msg["content"])
            if len(_content) > MAX_TOOL_RESULT_CHARS:
                msg = dict(msg)  # 浅拷贝，不修改原始对象
                msg["content"] = _content[:MAX_TOOL_RESULT_CHARS] + "\n...(内容已截断)"

        tokens = _estimate_message_tokens(msg)

        envelope = MessageEnvelope(
            msg=msg,
            seq=self._seq_counter,
            turn=turn,
            tool_name=tool_name,
            node_id=node_id,
            tokens=tokens,
            is_pinned=pinned,
            group_id=group_id,
        )
        self._seq_counter += 1
        self.envelopes.append(envelope)

    def _remember_visible_node_ids(self, envs: List[MessageEnvelope]) -> None:
        """Cache the latest visible node ids from a windowing pass."""
        self._last_visible_node_ids = {
            str(env.node_id)
            for env in envs
            if getattr(env, "node_id", None)
        }

    def new_tool_group(self) -> int:
        """分配一个新的工具调用组 ID"""
        self._group_counter += 1
        return self._group_counter

    # ── 模式切换 ──

    def observe_tool_call(
        self,
        tool_name: str,
        tool_args: Dict,
    ) -> Optional[AttentionMode]:
        """观察工具调用并处理显式注意力控制。

        TSD v2.9.22 要求：普通工具名不能作为 GLOBAL/FOCUS/SINGLE_CHAIN
        的主规则绑定。该入口只做两件事：
        1. 记录当前工具参数里的节点锚点，供后续压力/LLM 选择使用；
        2. 当工具本身就是显式注意力控制能力时执行模式切换。

        Args:
            tool_name: 工具名
            tool_args: 工具参数

        Returns:
            如果模式发生切换，返回新模式；否则 None
        """
        # 提取 node_id（多个参数名兼容）
        node_id = (
            tool_args.get("node_id")
            or tool_args.get("outline_id")
            or tool_args.get("target_node_id")
        )
        if node_id:
            self._current_node_id = node_id

        old_mode = self.mode
        new_mode = self._compute_transition(tool_name, tool_args)

        if new_mode and new_mode != old_mode:
            self.mode = new_mode
            logger.info(
                f"[AttentionWindow] 显式注意力切换: {old_mode.value} → {new_mode.value} "
                f"(control={tool_name}, node={self._current_node_id})"
            )
            self._publish_mode_change(old_mode, new_mode, f"attention_control:{tool_name}")
            # 聚焦切换时，自动注入目标节点的已有知识
            if new_mode in (AttentionMode.FOCUS, AttentionMode.SINGLE_CHAIN):
                self._inject_node_knowledge(self._current_node_id)
            return new_mode

        return None

    def on_navigate_attention(self, direction: str, target_node_id: Optional[str] = None):
        """navigate_attention 工具调用时的联动回调

        根据导航方向调整注意力模式：
        - deeper: 当前是 GLOBAL 则切换到 FOCUS；已是 FOCUS 则切换到 SINGLE_CHAIN
        - broader: 当前是 SINGLE_CHAIN 则切换到 FOCUS；已是 FOCUS 则切换到 GLOBAL
        - jump: 根据目标节点深度自动选择模式

        Args:
            direction: "deeper" | "broader" | "jump"
            target_node_id: 跳转目标节点 ID（jump 时传入）
        """
        old_mode = self.mode

        if direction == "deeper":
            if self.mode == AttentionMode.GLOBAL:
                self.mode = AttentionMode.FOCUS
            elif self.mode == AttentionMode.FOCUS:
                self.mode = AttentionMode.SINGLE_CHAIN
            # SINGLE_CHAIN 已是最深，保持不变

        elif direction == "broader":
            if self.mode == AttentionMode.SINGLE_CHAIN:
                self.mode = AttentionMode.FOCUS
            elif self.mode == AttentionMode.FOCUS:
                self.mode = AttentionMode.GLOBAL
            # GLOBAL 已是最浅，保持不变

        elif direction == "jump" and target_node_id and self.task_graph:
            # jump 根据目标节点深度自动选择模式
            depth = self.task_graph.get_node_depth(target_node_id)
            if depth is not None:
                if depth <= 1:
                    self.mode = AttentionMode.GLOBAL
                elif depth == 2:
                    self.mode = AttentionMode.FOCUS
                else:
                    self.mode = AttentionMode.SINGLE_CHAIN

        if target_node_id:
            self._current_node_id = target_node_id

        if self.mode != old_mode:
            logger.info(
                f"[AttentionWindow] navigate_attention 联动: "
                f"{old_mode.value} → {self.mode.value} "
                f"(direction={direction}, target={target_node_id})"
            )
            self._publish_mode_change(old_mode, self.mode, f"navigate:{direction}")
            # 聚焦切换时，自动注入目标节点的已有知识
            if self.mode in (AttentionMode.FOCUS, AttentionMode.SINGLE_CHAIN):
                self._inject_node_knowledge(self._current_node_id)

    def auto_navigate_on_status_change(self, node_id: str, new_status: str):
        """兼容旧调用：任务状态变化只更新焦点锚点，不自动切换模式。

        v2.9.22 后，状态变化属于 LLM 注意力选择的输入证据；是否从
        GLOBAL 下探到 FOCUS/SINGLE_CHAIN、或从局部回到 GLOBAL，必须由
        上下文压力触发后的 LLM 决策或显式注意力工具完成，不能由
        task_mark_status 这类普通任务工具自动决定。
        """
        if node_id:
            self._current_node_id = node_id
        logger.debug(
            "[AttentionWindow] 状态变化已记录为注意力候选输入，不自动切换: "
            "node=%s status=%s mode=%s",
            node_id,
            new_status,
            self.mode.value,
        )

    def _publish_mode_change(self, old_mode: AttentionMode, new_mode: AttentionMode, trigger: str):
        """发布注意力模式变更事件到 EventBus → WebBridge → 仪表盘"""
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, ZulongEvent
            import time as _time
            payload = {
                "component": "AttentionWindow",
                "old_mode": old_mode.value,
                "new_mode": new_mode.value,
                "trigger": trigger,
                "current_node_id": self._current_node_id,
                "usage_ratio": self.usage_ratio,
                "total_messages": len(self.envelopes),
                "timestamp": _time.time(),
            }
            event = ZulongEvent(type=EventType("SYSTEM_STATUS"), payload=payload)
            event_bus.publish(event)
        except Exception:
            pass

    def _compute_transition(
        self,
        tool_name: str,
        tool_args: Dict,
    ) -> Optional[AttentionMode]:
        """根据显式注意力控制计算目标模式。

        TSD v2.9.22: 动态注意力不再由普通工具名硬触发。读写、命令、
        检索等工具调用只作为上下文压力和 LLM 决策的观测输入；真正切换
        注意力必须来自 LLM 显式注意力选择或压力控制消息。
        """
        # navigate_attention 由 on_navigate_attention 单独处理
        if tool_name == "navigate_attention":
            return None

        # P2-1: adjust_attention_mode 直接指定目标模式
        if tool_name == "adjust_attention_mode":
            mode_str = tool_args.get("mode", "")
            _mode_map = {
                "global": AttentionMode.GLOBAL,
                "focus": AttentionMode.FOCUS,
                "single_chain": AttentionMode.SINGLE_CHAIN,
            }
            return _mode_map.get(mode_str)

        return None

    # ── 窗口裁剪 ──

    def apply_window(self) -> List[Dict]:
        """对所有消息评分、裁剪，返回适合 LLM 的消息列表

        Returns:
            经过窗口过滤的消息列表（保持时间顺序）
        """
        # ── LLM自主注意力选择检查 ──
        if self._should_try_llm_selection():
            self._dispatch_llm_mode_selection()

        if not self.envelopes:
            self._last_visible_node_ids = set()
            return []

        # P2-13: 动态调整budget（基于任务图节点数和当前模式）
        self._adjust_budget()
        window_budget = self._effective_window_budget()

        # 1. 对所有消息评分
        for env in self.envelopes:
            env.weight = self._score_message(env)

        # 2. 分离 pinned 和 non-pinned
        pinned = [e for e in self.envelopes if e.is_pinned]
        unpinned = [e for e in self.envelopes if not e.is_pinned]

        pinned_tokens = sum(e.tokens for e in pinned)
        remaining_budget = window_budget - pinned_tokens

        if remaining_budget <= 0:
            logger.warning(
                f"[AttentionWindow] pinned 消息超预算: "
                f"{pinned_tokens} > {window_budget}"
            )
            # 渐进式降级：保留首尾 pinned，其余降级参与权重排序
            sorted_pinned = sorted(pinned, key=lambda e: e.seq)
            if len(sorted_pinned) > 2:
                essential = [sorted_pinned[0], sorted_pinned[-1]]
                demoted = sorted_pinned[1:-1]
                essential_tokens = sum(e.tokens for e in essential)
                demoted_budget = window_budget - essential_tokens
                if demoted_budget > 0:
                    all_candidates = demoted + unpinned
                    for env in all_candidates:
                        env.weight = self._score_message(env)
                    all_candidates.sort(
                        key=lambda e: e.weight, reverse=True
                    )
                    kept = []
                    used = 0
                    for env in all_candidates:
                        if used + env.tokens <= demoted_budget:
                            kept.append(env)
                            used += env.tokens
                    kept.sort(key=lambda e: e.seq)
                    result = [e.msg for e in essential]
                    result.extend(e.msg for e in kept)
                    self._remember_visible_node_ids(essential + kept)
                    self._remember_last_window(
                        essential + kept,
                        window_budget=window_budget,
                        evicted_count=len(all_candidates) - len(kept),
                    )
                    return strip_llm_message_metadata(
                        self._normalize_system_prefix(result)
                    )
            # 兜底：仍然只返回 pinned
            self._remember_visible_node_ids(sorted_pinned)
            self._remember_last_window(
                sorted_pinned,
                window_budget=window_budget,
                evicted_count=len(unpinned),
            )
            return strip_llm_message_metadata([e.msg for e in sorted_pinned])

        # 3. 按组处理：计算每个组的最高权重和总 tokens
        group_info: Dict[Optional[int], Dict] = {}
        for env in unpinned:
            # Only explicit tool-call groups are atomic.  Plain messages often
            # have ``group_id=None``; treating all of them as one group makes
            # FOCUS/SINGLE_CHAIN unable to keep the current-node message while
            # evicting unrelated history.
            gid = env.group_id if env.group_id is not None else ("msg", env.seq)
            if gid not in group_info:
                group_info[gid] = {
                    "max_weight": 0.0,
                    "total_tokens": 0,
                    "envelopes": [],
                }
            info = group_info[gid]
            info["max_weight"] = max(info["max_weight"], env.weight)
            info["total_tokens"] += env.tokens
            info["envelopes"].append(env)

        # 4. 按组最高权重排序
        sorted_groups = sorted(
            group_info.values(),
            key=lambda g: g["max_weight"],
            reverse=True,
        )

        # 5. 贪心选择：从高权重到低权重累加，直到用尽预算。
        # 若已经确定会淘汰消息，预留少量预算给“淘汰摘要”。这样动态
        # 注意力不是把摘要额外塞回去导致再次超预算，而是在同一个阈值
        # 预算内决定“当前注入什么、暂排什么、用摘要承接什么”。
        kept_envs: List[MessageEnvelope] = []
        evicted_envs: List[MessageEnvelope] = []
        used_tokens = 0
        total_unpinned_tokens = sum(e.tokens for e in unpinned)
        summary_reserve = (
            max(0, int(window_budget * 0.10))
            if total_unpinned_tokens > remaining_budget
            else 0
        )
        selection_budget = max(0, remaining_budget - summary_reserve)

        for group in sorted_groups:
            if used_tokens + group["total_tokens"] <= selection_budget:
                kept_envs.extend(group["envelopes"])
                used_tokens += group["total_tokens"]
            else:
                evicted_envs.extend(group["envelopes"])

        # 6. 生成淘汰摘要
        summary_msg = None
        if evicted_envs:
            summary_text = self._build_summary(evicted_envs)
            if summary_text:
                summary_msg = {
                    "role": "system",
                    "content": summary_text,
                }

            # 将淘汰内容的语义摘要写回 MemoryGraph（闭合淘汰-恢复环路）
            self._persist_evicted_to_memory(evicted_envs)

            logger.info(
                f"[AttentionWindow] 淘汰 {len(evicted_envs)} 条消息, "
                f"保留 {len(kept_envs)} 条, 模式={self.mode.value}, "
                f"预算={window_budget}, 已用={used_tokens}, "
                f"阈值预算={self._threshold_budget_base}, "
                f"原始预算={self._base_budget}"
            )

        # 7. 按原始时间顺序排列
        kept_envs.sort(key=lambda e: e.seq)
        result = [e.msg for e in sorted(pinned, key=lambda e: e.seq)]

        # 在 pinned 消息后插入摘要
        if summary_msg:
            result.append(summary_msg)

        result.extend(e.msg for e in kept_envs)
        visible_envs = list(pinned) + kept_envs
        if summary_msg:
            summary_tokens = _estimate_message_tokens(summary_msg)
        else:
            summary_tokens = 0
        self._remember_visible_node_ids(visible_envs)
        self._remember_last_window(
            visible_envs,
            window_budget=window_budget,
            evicted_count=len(evicted_envs),
            extra_tokens=summary_tokens,
            extra_messages=1 if summary_msg else 0,
        )
        return strip_llm_message_metadata(self._normalize_system_prefix(result))

    def _normalize_system_prefix(self, messages: List[Dict]) -> List[Dict]:
        """确保所有 role=system 消息位于数组开头（API 兼容性要求）

        SiliconFlow/Qwen 等 API 要求 system message 必须在 messages 数组起始位置，
        Circuit Breaker / 淘汰摘要等组件可能注入 role=system 到对话中间，需要前置。
        """
        if not messages:
            return messages

        # 快速路径：无 system 消息或只有第一条是 system
        first_non_sys_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") != "system"),
            len(messages),
        )
        # 检查 first_non_sys_idx 之后是否还有 system 消息
        has_scattered = any(
            m.get("role") == "system" for m in messages[first_non_sys_idx:]
        )
        if not has_scattered:
            return messages  # 已经是合法顺序

        # 需要重排：所有 system 前置，其余保持相对顺序
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system_msgs = [m for m in messages if m.get("role") != "system"]
        return system_msgs + non_system_msgs

    # ── 评分逻辑 ──

    def _score_message(self, env: MessageEnvelope) -> float:
        """根据当前模式计算消息权重

        评分公式: base × time_decay × mode_multiplier × memory_boost
        """
        base = 1.0

        # 时效衰减：每过一轮衰减 5%
        age = max(0, self._current_turn - env.turn)
        time_decay = 0.95 ** age

        # 模式加权
        mode_mult = self._mode_multiplier(env)

        score = base * time_decay * mode_mult

        # MemoryGraph 激活值融合：高激活节点获得额外权重提升
        if self.memory_graph and env.node_id:
            try:
                _mem_node = self.memory_graph.get_node(env.node_id)
                if _mem_node and hasattr(_mem_node, 'activation') and _mem_node.activation > 0:
                    # boost 范围: 1.0 ~ 1.5 (activation 0→0 → 1.0→1.5)
                    score *= (1.0 + 0.5 * _mem_node.activation)
            except Exception:
                pass

        return score

    def _mode_multiplier(self, env: MessageEnvelope) -> float:
        """按模式计算权重乘数"""
        if self.mode == AttentionMode.GLOBAL:
            return self._mult_global(env)
        elif self.mode == AttentionMode.FOCUS:
            return self._mult_focus(env)
        else:
            return self._mult_single_chain(env)

    def _mult_global(self, env: MessageEnvelope) -> float:
        """全局模式：大纲和概览权重高，深层节点递减"""
        if env.tool_name == "task_view_overview":
            return 1.5
        if env.tool_name in ("task_add_node", "task_update_node", "task_remove_node"):
            return 1.3
        if env.tool_name == "submit_final_answer":
            return 2.0

        # 按节点深度递减（如果有 task_graph）
        if env.node_id and self.task_graph:
            depth = self.task_graph.get_node_depth(env.node_id)
            if depth is not None:
                # 深度 0-1: ×1.2, 深度 2: ×1.0, 深度 3: ×0.8, ...
                return max(0.3, 1.2 - depth * 0.2)

        return 1.0

    def _mult_focus(self, env: MessageEnvelope) -> float:
        """聚焦模式：当前节点和关联节点权重高"""
        if not self._current_node_id:
            return 1.0

        if env.node_id == self._current_node_id:
            return 3.0

        # 检查是否是当前节点的祖先或依赖
        if env.node_id and self.task_graph:
            # 祖先链
            ancestors = self.task_graph.get_ancestor_chain(
                self._current_node_id
            )
            ancestor_ids = {a.id for a in ancestors} if ancestors else set()
            if env.node_id in ancestor_ids:
                return 2.0

            # 依赖
            deps = self.task_graph.get_dependencies(self._current_node_id)
            if env.node_id in deps:
                return 2.0

            # 兄弟节点
            parent_id = self.task_graph.get_parent(self._current_node_id)
            if parent_id:
                siblings = self.task_graph.get_children(parent_id)
                sibling_ids = {s.id for s in siblings}
                if env.node_id in sibling_ids:
                    return 1.5

            # 不相关
            return 0.5

        return 1.0

    def _mult_single_chain(self, env: MessageEnvelope) -> float:
        """单链模式：优先保留当前执行链路，暂排当前阶段无关上下文"""
        if not self._current_node_id:
            return 1.0

        if env.node_id == self._current_node_id:
            return 5.0

        if env.node_id and self.task_graph:
            # 直接祖先链
            ancestors = self.task_graph.get_ancestor_chain(
                self._current_node_id
            )
            ancestor_ids = {a.id for a in ancestors} if ancestors else set()
            if env.node_id in ancestor_ids:
                return 3.0

            # 直接依赖
            deps = self.task_graph.get_dependencies(self._current_node_id)
            if env.node_id in deps:
                return 2.5

            # 不相关 → 大幅降权
            return 0.2

        # 无 node_id 的消息（纯文本等）保持中等权重
        return 0.8

    # ── 淘汰摘要 ──

    def _build_summary(self, evicted: List[MessageEnvelope]) -> str:
        """为被淘汰的消息生成语义级摘要

        与旧版仅记录元数据（工具名×次数）不同，新版提取每组工具结果的
        关键内容片段，保留结论性信息而非仅记录操作记录。
        """
        if not evicted:
            return ""

        # 收集被淘汰的工具调用和节点
        tool_counts: Dict[str, int] = {}
        node_ids: Set[str] = set()
        # 收集工具结果中的关键内容片段
        content_snippets: List[str] = []

        for env in evicted:
            if env.tool_name:
                tool_counts[env.tool_name] = (
                    tool_counts.get(env.tool_name, 0) + 1
                )
            if env.node_id:
                node_ids.add(env.node_id)
            # 从 tool result 消息中提取内容摘要
            if env.msg.get("role") == "tool" and env.msg.get("content"):
                snippet = self._extract_content_snippet(
                    str(env.msg["content"]), env.tool_name)
                if snippet:
                    content_snippets.append(snippet)

        parts = [f"[上下文窗口管理] 已淘汰 {len(evicted)} 条历史消息。"]

        if tool_counts:
            tool_summary = ", ".join(
                f"{name}×{count}" for name, count in
                sorted(tool_counts.items(), key=lambda x: -x[1])[:5]
            )
            parts.append(f"涉及工具: {tool_summary}")

        if node_ids:
            nodes_str = ", ".join(sorted(node_ids)[:8])
            if len(node_ids) > 8:
                nodes_str += f" 等共 {len(node_ids)} 个节点"
            parts.append(f"涉及节点: {nodes_str}")

        # 插入内容摘要（关键改进：保留语义信息）
        if content_snippets:
            merged = " | ".join(content_snippets[:5])
            parts.append(f"关键发现: {merged}")

        parts.append("如需回顾已淘汰的内容，请使用 recall_memory 或 read_memory_node 工具重新查询。")

        summary = " ".join(parts)

        # 摘要本身不超过预算的 10%
        max_summary_tokens = int(self.budget * 0.10)
        summary_tokens = estimate_tokens(summary)
        if summary_tokens > max_summary_tokens:
            ratio = max_summary_tokens / max(summary_tokens, 1)
            summary = summary[:int(len(summary) * ratio)] + "..."

        return summary

    @staticmethod
    def _extract_content_snippet(content: str, tool_name: Optional[str],
                                 max_len: int = 200) -> str:
        """从工具结果中提取关键内容片段

        规则：
        - read_file 类结果：提取文件路径和结构性关键词
        - JSON 格式结果：提取 message/result/data 字段
        - 其他：取首段非空文本
        """
        if not content or len(content) < 10:
            return ""

        # JSON 格式结果 — 提取 message/data 字段
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                import json
                obj = json.loads(stripped)
                for key in ("message", "result", "data", "summary"):
                    val = obj.get(key)
                    if val and isinstance(val, str) and len(val) > 10:
                        return val[:max_len]
                    if val and isinstance(val, dict):
                        # data 是字典时，取其 message 字段
                        msg = val.get("message", "")
                        if msg:
                            return str(msg)[:max_len]
            except (json.JSONDecodeError, ValueError):
                pass

        # 纯文本 — 取首段有意义内容（跳过空行和分隔线）
        lines = content.split("\n")
        meaningful = []
        total = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("---") or line.startswith("==="):
                continue
            meaningful.append(line)
            total += len(line)
            if total >= max_len:
                break

        return " ".join(meaningful)[:max_len] if meaningful else ""

    def _persist_evicted_to_memory(self, evicted: List[MessageEnvelope]) -> None:
        """将淘汰消息的语义摘要持久化到 MemoryGraph 和 TaskGraph

        按 node_id 分组收集淘汰的工具结果内容，生成每个节点的语义摘要，
        写入 MemoryGraph 的 eviction_summary metadata 字段，使后续
        recall_memory 检索能找回关键信息。
        同时将摘要追加到 TaskNode 的 analysis_content，形成知识积累。
        """
        if not self.memory_graph and not self.task_graph:
            return

        # 按 node_id 分组收集内容片段
        node_snippets: Dict[str, List[str]] = {}
        for env in evicted:
            nid = env.node_id
            if not nid:
                continue
            if env.msg.get("role") == "tool" and env.msg.get("content"):
                snippet = self._extract_content_snippet(
                    str(env.msg["content"]), env.tool_name, max_len=300)
                if snippet:
                    if nid not in node_snippets:
                        node_snippets[nid] = []
                    node_snippets[nid].append(snippet)

        if not node_snippets:
            return

        for nid, snippets in node_snippets.items():
            new_summary = " | ".join(snippets[:3])

            # 写入 MemoryGraph
            if self.memory_graph:
                try:
                    mg_node = self.memory_graph.get_node(nid)
                    if mg_node is not None:
                        existing = mg_node.metadata.get("eviction_summary", "")
                        if existing:
                            combined = f"{existing} | {new_summary}"
                            if len(combined) > 1500:
                                combined = combined[-1500:]
                            mg_node.metadata["eviction_summary"] = combined
                        else:
                            mg_node.metadata["eviction_summary"] = new_summary[:1500]
                        mg_node.metadata["eviction_turn"] = self._current_turn
                except Exception as e:
                    logger.info(
                        f"[AttentionWindow] 淘汰内容持久化到 MG 失败 (node={nid}): {e}")

            # 写入 TaskGraph（追加到 analysis_content，形成知识积累）
            if self.task_graph:
                try:
                    task_node = self.task_graph.get_node(nid)
                    if task_node is not None:
                        eviction_note = f"\n[淘汰恢复 turn={self._current_turn}] {new_summary}"
                        if task_node.analysis_content:
                            task_node.analysis_content += eviction_note
                        else:
                            task_node.analysis_content = eviction_note.strip()
                        task_node.content_version += 1
                except Exception as e:
                    logger.info(
                        f"[AttentionWindow] 淘汰内容持久化到 TG 失败 (node={nid}): {e}")

    def _inject_node_knowledge(self, node_id: Optional[str]) -> None:
        """聚焦切换时，从 TaskNode 加载已有分析内容注入上下文

        当 AttentionWindow 切换到 FOCUS/SINGLE_CHAIN 模式聚焦某节点时，
        自动将该节点的 semantic_summary 注入为 system 消息，帮助模型
        快速恢复对该节点的知识上下文。
        """
        if not node_id or not self.task_graph:
            return

        try:
            node = self.task_graph.get_node(node_id)
            if node is None:
                return

            # 优先使用 semantic_summary（简洁，适合上下文注入）
            knowledge = node.semantic_summary
            if not knowledge and node.analysis_content:
                # 没有摘要时，取 analysis_content 的前 500 字符
                knowledge = node.analysis_content[:500]
            if not knowledge:
                return

            # 构造知识回顾消息
            recall_msg = {
                "role": "system",
                "content": (
                    f"[节点知识回顾] {node.label} (v{node.content_version}):\n"
                    f"{knowledge}"
                ),
            }
            self.register_message(
                recall_msg,
                turn=self._current_turn,
                node_id=node_id,
                pinned=False,
            )
            logger.info(
                f"[AttentionWindow] 注入节点知识: {node_id} "
                f"({len(knowledge)} chars, v{node.content_version})"
            )
        except Exception as e:
            logger.info(f"[AttentionWindow] 节点知识注入失败: {e}")

    # ── 编排器阶段感知 ──

    def set_phase(self, phase: str, subtask_id: str = None):
        """编排器阶段切换时调整注意力模式

        不新增额外乘数系统，只利用已有的三种 AttentionMode：
        - plan   → GLOBAL（全局视角，关注大纲和整体结构）
        - execute → FOCUS（聚焦当前子任务，依赖产出权重提升）
        - reflect → GLOBAL（回到全局视野评估质量）
        - synthesize → GLOBAL（汇总需要全局概览）

        Args:
            phase: 编排器阶段名 ("plan" / "execute" / "reflect" / "synthesize")
            subtask_id: 当前子任务节点 ID（execute 阶段必传）
        """
        old_mode = self.mode

        if phase == "plan":
            self.mode = AttentionMode.GLOBAL
        elif phase == "execute":
            self.mode = AttentionMode.FOCUS
            if subtask_id:
                self._current_node_id = subtask_id
        elif phase in ("reflect", "synthesize"):
            self.mode = AttentionMode.GLOBAL
        else:
            logger.warning(f"[AttentionWindow] 未知阶段: {phase}，保持当前模式")
            return

        if self.mode != old_mode:
            logger.info(
                f"[AttentionWindow] 阶段切换: {old_mode.value} → {self.mode.value} "
                f"(phase={phase}, subtask={subtask_id})"
            )

    # ── 容量查询 ──

    @property
    def remaining_budget(self) -> int:
        """返回剩余可用 token 预算"""
        used = sum(e.tokens for e in self.envelopes)
        return max(0, self.budget - used)

    @property
    def usage_ratio(self) -> float:
        """返回当前使用比率 (0.0 ~ 1.0)"""
        used = sum(e.tokens for e in self.envelopes)
        return min(1.0, used / max(self.budget, 1))

    def _threshold_budget_ratio(self) -> float:
        """Return 1.0 — threshold budget now equals the full context window size.

        Previously this returned a configurable ratio (e.g. 0.5) that multiplied
        the context window to get a smaller pressure budget. That mechanism is
        removed: threshold_budget_tokens == context_window_size directly.
        The ratio config field is kept for backward compatibility but always 1.0.
        """
        return 1.0

    @property
    def threshold_budget_tokens(self) -> int:
        """LLM threshold budget in tokens — equals the full context window size."""
        return max(1, int(self.context_window_size))

    @property
    def window_injection_budget_tokens(self) -> int:
        """Token budget used by ``apply_window`` for injectable message context.

        Tools schema, system scaffolding, and output buffer consume part of the
        LLM window even though they are not always present in ``envelopes``.
        Therefore the message-selection budget is bounded by the threshold
        budget minus reserved tokens.  GLOBAL/FOCUS/SINGLE_CHAIN can only
        narrow this budget further; they must not expand it past the threshold
        budget, otherwise attention switching cannot actually control context.
        """
        return max(1024, self.threshold_budget_tokens - max(0, int(self.reserved_tokens or 0)))

    @property
    def active_context_tokens(self) -> int:
        """Tokens currently visible to the LLM for pressure display/triggering.

        ``_last_window_tokens`` records the message envelopes selected by
        ``apply_window``.  The real LLM call also needs reserved/system/tool
        scaffolding tokens, so the user-visible pressure must include the
        reserved portion once a windowing pass has happened.  Before the first
        pass this stays at 0 and callers should fall back to the registered
        message-pool pressure for bootstrap.
        """
        provider_prompt = max(0, int(getattr(self, "_last_provider_prompt_tokens", 0) or 0))
        if provider_prompt:
            return provider_prompt
        visible = max(0, int(self._last_window_tokens or 0))
        if self._last_window_message_count:
            visible += max(0, int(self.reserved_tokens or 0))
        visible += max(0, int(getattr(self, "_last_active_context_extra_tokens", 0) or 0))
        return visible

    @property
    def active_context_pressure_ratio(self) -> float:
        """Pressure of the last LLM-visible window.

        ``context_pressure_ratio`` (a.k.a. registered_context_pressure) describes
        the registered message pool and is useful for deciding when to ask the
        LLM to re-select attention.  This property describes what was actually
        injected after windowing.
        """
        return max(0.0, self.active_context_tokens / max(self.threshold_budget_tokens, 1))

    @property
    def trigger_context_pressure_ratio(self) -> float:
        """Pressure ratio used for live trigger decisions.

        This matches the Web-visible pressure after at least one windowing pass.
        Before that pass there is no visible-window telemetry yet, so bootstrap
        detection still uses the registered message-pool pressure to allow the
        first attention switch before provider-visible pressure telemetry exists.
        """
        if self._last_window_message_count:
            return self.active_context_pressure_ratio
        return self.context_pressure_ratio

    @property
    def context_pressure_ratio(self) -> float:
        """Return actual occupied context divided by configured threshold budget."""
        used = sum(e.tokens for e in self.envelopes)
        return max(0.0, used / max(self.threshold_budget_tokens, 1))

    @property
    def stats(self) -> Dict[str, Any]:
        """返回窗口统计信息（供 prompt 注入使用）"""
        total_tokens = sum(e.tokens for e in self.envelopes)
        pinned_tokens = sum(
            e.tokens for e in self.envelopes if e.is_pinned
        )
        return {
            "mode": self.mode.value,
            "budget": self.budget,
            "threshold_budget_tokens": self.threshold_budget_tokens,
            "window_injection_budget_tokens": self.window_injection_budget_tokens,
            "total_messages": len(self.envelopes),
            "total_tokens": total_tokens,
            "pinned_tokens": pinned_tokens,
            "remaining_tokens": max(0, self.budget - total_tokens),
            "usage_ratio": self.usage_ratio,
            "context_pressure_ratio": self.context_pressure_ratio,
            "active_context_pressure_ratio": self.active_context_pressure_ratio,
            "trigger_context_pressure_ratio": self.trigger_context_pressure_ratio,
            "active_context_tokens": self.active_context_tokens,
            "last_window_tokens": self._last_window_tokens,
            "last_window_message_count": self._last_window_message_count,
            "last_evicted_message_count": self._last_evicted_message_count,
            "current_node_id": self._current_node_id,
            "context_window_size": self.context_window_size,
        }

    # ── Provider usage 回填 (TSD §26.1.5) ──

    def record_provider_usage(self, prompt_tokens: int) -> None:
        """回填 provider 返回的真实 prompt_tokens，供 active_context_tokens 优先使用。

        TSD §26.1.5 要求压力口径优先使用 provider usage，无 usage 时才退回本地估算。
        此方法在每次 provider 调用返回后由调用方调用。
        """
        try:
            self._last_provider_prompt_tokens = max(0, int(prompt_tokens or 0))
        except (TypeError, ValueError):
            self._last_provider_prompt_tokens = 0

    @property
    def provider_context_pressure_percent(self) -> Optional[float]:
        """本轮实际注入压力百分比，优先 provider usage；无 usage 返回 None。"""
        if self._last_provider_prompt_tokens > 0:
            return round(
                self._last_provider_prompt_tokens
                / max(self.threshold_budget_tokens, 1)
                * 100.0,
                2,
            )
        return None

    @property
    def registered_context_pressure(self) -> float:
        """AttentionWindow 已登记消息池相对阈值预算的压力（诊断字段）。

        TSD §26.1.5：registered 只表示已登记消息池压力，不等于本轮实际发给 LLM 的
        active context。等价于旧 context_pressure_ratio，重命名以对齐 TSD 字段裁决。
        """
        return self.context_pressure_ratio

    # ── LLM自主注意力选择辅助方法 ──

    def set_llm_client(self, llm_client: Any) -> None:
        """设置LLM客户端实例，启用LLM自主注意力选择

        Args:
            llm_client: LLM客户端实例（需支持chat或generate方法）
        """
        if self._llm_config and self._llm_config.enabled:
            self._llm_client = llm_client
            if self._mode_selector:
                self._mode_selector.set_llm_client(llm_client)
            logger.info("[AttentionWindow] LLM客户端已设置，LLM自主注意力选择已就绪")

    def _should_try_llm_selection(self) -> bool:
        """检查是否应尝试LLM自主选择

        条件：
        1. 功能已启用且配置有效
        2. LLM客户端已设置
        3. 不在冷却期内

        Returns:
            是否应尝试
        """
        if not self._llm_config or not self._llm_config.enabled:
            return False
        if not self._llm_client:
            return False
        if not self._mode_controller:
            return False
        cooldown_result = self._mode_controller.can_switch()
        if not cooldown_result.is_allowed:
            return False
        return True

    def _dispatch_llm_mode_selection(self) -> None:
        """线程安全触发 LLM 自主注意力选择。

        apply_window() 会同时出现在 FastAPI/WebSocket 事件循环线程与
        L2-FC-Worker 等普通工作线程中。普通工作线程没有默认
        asyncio event loop，不能使用 asyncio.get_event_loop()。这里按
        当前运行环境选择调度方式：

        - 已在运行中的事件循环：创建后台任务，避免阻塞调用链；
        - 无运行中事件循环：使用 asyncio.run() 在当前线程同步完成。

        同时用轻量锁避免同一窗口在压力持续期间重入并发触发多个
        LLM 决策。该方法不做模式规则判断，仍完全交由压力阈值和
        LLM 自主选择决定 GLOBAL / FOCUS / SINGLE_CHAIN。
        """
        with self._selection_lock:
            if self._selection_inflight:
                logger.debug("[AttentionWindow] LLM注意力选择已有进行中任务，跳过重复触发")
                return
            self._selection_inflight = True

        async def _run_and_clear() -> None:
            try:
                await self._try_llm_mode_selection_async()
            finally:
                with self._selection_lock:
                    self._selection_inflight = False

        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                loop.create_task(_run_and_clear())
            else:
                asyncio.run(_run_and_clear())
        except Exception as e:
            with self._selection_lock:
                self._selection_inflight = False
            logger.warning(f"[AttentionWindow] LLM注意力选择执行失败: {e}")

    async def _try_llm_mode_selection_async(self) -> None:
        """异步执行LLM自主注意力模式选择

        完整流程：
        1. 计算当前压力指标
        2. 检查是否超过阈值
        3. 构建决策请求（含任务上下文）
        4. 调用LLM决策
        5. 应用决策结果（含震荡检测和冷却管理）
        """
        if not self._pressure_detector or not self._mode_selector or not self._mode_controller:
            return

        try:
            # 1. 计算压力
            metrics = self._pressure_detector.calculate_pressure()
            self._last_pressure_metrics = metrics

            # 2. 检查阈值
            threshold_result = self._pressure_detector.check_threshold(metrics)
            self._last_threshold_result = threshold_result
            if not threshold_result.should_trigger:
                self._last_selection_event = {
                    "attempted": False,
                    "triggered": False,
                    "switched": False,
                    "pressure": metrics.to_dict() if hasattr(metrics, "to_dict") else {},
                    "threshold": threshold_result.to_dict() if hasattr(threshold_result, "to_dict") else {},
                    "reason": threshold_result.message,
                }
                logger.debug(
                    f"[AttentionWindow] 压力未超阈值: "
                    f"threshold_budget={metrics.threshold_relative_pressure * 100:.1f}%, "
                    f"raw_pressure={metrics.current_pressure:.3f}, "
                    f"threshold={threshold_result.message}"
                )
                return

            logger.info(
                f"[AttentionWindow] 触发LLM注意力选择: "
                f"threshold_budget={metrics.threshold_relative_pressure * 100:.1f}%, "
                f"raw_pressure={metrics.current_pressure:.3f}, "
                f"reference={metrics.threshold_reference:.3f}, "
                f"trend={metrics.pressure_trend.value}, "
                f"trigger={threshold_result.trigger_type}"
            )
            self._selection_attempt_count += 1

            # 3. 构建决策请求
            task_context = self._get_task_context_summary()
            current_mode = self.mode.name
            switch_history = self._mode_controller.get_switch_history(5)

            request = self._mode_selector.build_decision_request(
                pressure_metrics=metrics,
                current_mode=current_mode,
                task_context=task_context,
                switch_history=switch_history,
            )

            # 4. 调用LLM决策
            decision = await self._mode_selector.call_llm_decision(request)
            self._last_llm_decision = decision

            logger.info(
                f"[AttentionWindow] LLM决策结果: mode={decision.mode}, "
                f"confidence={decision.confidence:.2f}, "
                f"fallback={decision.is_fallback}, "
                f"reason={decision.reason[:60]}"
            )

            # 5. 应用决策
            switched = False
            if decision.mode != current_mode:
                result = self._mode_controller.apply_llm_decision(
                    current_mode=current_mode,
                    decision=decision,
                    pressure_value=metrics.current_pressure,
                )

                if result.switched:
                    switched = True
                    try:
                        new_mode = AttentionMode[decision.mode]
                        old_mode = self.mode
                        self.mode = new_mode
                        self._last_llm_selection_time = __import__('time').time()
                        logger.info(
                            f"[AttentionWindow] LLM自主切换: "
                            f"{old_mode.value} → {new_mode.value} "
                            f"(confidence={decision.confidence:.2f})"
                        )
                        self._publish_mode_change(
                            old_mode, new_mode,
                            f"llm_autonomous:{decision.reason[:30]}"
                        )
                        # 聚焦切换时注入节点知识
                        if new_mode in (AttentionMode.FOCUS, AttentionMode.SINGLE_CHAIN):
                            self._inject_node_knowledge(self._current_node_id)
                    except KeyError:
                        logger.error(
                            f"[AttentionWindow] LLM返回无效模式名: {decision.mode}"
                        )
            else:
                logger.info(
                    f"[AttentionWindow] LLM选择保持当前模式: {current_mode} "
                    f"(confidence={decision.confidence:.2f})"
                )
            self._last_selection_event = {
                "attempted": True,
                "triggered": True,
                "switched": switched,
                "pressure": metrics.to_dict() if hasattr(metrics, "to_dict") else {},
                "threshold": threshold_result.to_dict() if hasattr(threshold_result, "to_dict") else {},
                "decision": decision.to_dict() if hasattr(decision, "to_dict") else {},
            }

        except Exception as e:
            logger.error(f"[AttentionWindow] LLM注意力选择流程异常: {e}")
            self._last_selection_event = {
                "attempted": True,
                "triggered": True,
                "switched": False,
                "error": str(e),
            }

    def _get_task_context_summary(self) -> str:
        """获取任务上下文摘要（供LLM决策参考）

        收集任务图状态、消息统计、近期工具调用等信息，
        帮助LLM做出更准确的注意力模式选择。

        Returns:
            任务上下文摘要文本（≤500字符）
        """
        summary_parts = []

        if self.task_graph:
            try:
                node_count = (
                    self.task_graph.node_count
                    if hasattr(self.task_graph, 'node_count')
                    else len(self.task_graph._nodes)
                    if hasattr(self.task_graph, '_nodes')
                    else 0
                )
                summary_parts.append(f"任务图节点数: {node_count}")
                if self._current_node_id:
                    summary_parts.append(f"当前节点: {self._current_node_id}")
            except Exception:
                pass

        message_count = len(self.envelopes)
        total_tokens = sum(e.tokens for e in self.envelopes)
        summary_parts.append(f"消息数量: {message_count}")
        summary_parts.append(f"总tokens: {total_tokens}")

        # 近期工具调用
        recent_tools = []
        for env in self.envelopes[-5:]:
            if env.tool_name:
                recent_tools.append(env.tool_name)
        if recent_tools:
            summary_parts.append(f"近期工具: {', '.join(recent_tools)}")

        summary = " | ".join(summary_parts)
        return summary[:500]

    def get_llm_selection_stats(self) -> Dict[str, Any]:
        """获取LLM注意力选择的统计信息

        Returns:
            统计信息字典（含启用状态、当前模式、切换次数等）
        """
        if not self._llm_config:
            return {"enabled": False}

        history = self._mode_controller.get_switch_history(10) if self._mode_controller else []
        llm_switches = [
            r for r in history
            if hasattr(r, 'trigger_type') and r.trigger_type == TriggerType.LLM_AUTONOMOUS
        ]
        fallback_switches = [
            r for r in history
            if hasattr(r, 'trigger_type') and r.trigger_type == TriggerType.FALLBACK
        ]

        return {
            "enabled": self._llm_config.enabled,
            "current_mode": self.mode.name,
            "total_switches": len(history),
            "llm_autonomous_switches": len(llm_switches),
            "fallback_switches": len(fallback_switches),
            "selection_attempt_count": self._selection_attempt_count,
            "last_pressure_metrics": (
                self._last_pressure_metrics.to_dict()
                if hasattr(self._last_pressure_metrics, "to_dict") else None
            ),
            "last_threshold_result": (
                self._last_threshold_result.to_dict()
                if hasattr(self._last_threshold_result, "to_dict") else None
            ),
            "last_llm_decision": (
                self._last_llm_decision.to_dict()
                if hasattr(self._last_llm_decision, "to_dict") else None
            ),
            "last_selection_event": self._last_selection_event,
            "cooldown_seconds": (
                self._cooldown_manager.get_current_cooldown_seconds()
                if self._cooldown_manager else 0
            ),
            "switch_history": [r.to_dict() for r in history[-5:]],
        }

    # ── 序列化/反序列化（IDE 跨请求 Session 持久化）──

    # ── P2-13: 动态budget调整 ──

    def _adjust_budget(self) -> None:
        """根据任务复杂度动态调整budget

        规则：
        - 任务图节点数 > 20: budget × 1.3（复杂任务需要更多上下文）
        - 任务图节点数 > 50: budget × 1.5
        - FOCUS模式: budget × 0.8（按需注入局部必要上下文）
        - SINGLE_CHAIN模式: budget × 0.6（按需注入当前链路上下文）
        """
        multiplier = 1.0
        if self.task_graph is not None:
            try:
                node_count = len(self.task_graph._nodes) if hasattr(self.task_graph, '_nodes') else 0
                if node_count > 50:
                    multiplier *= 1.5
                elif node_count > 20:
                    multiplier *= 1.3
            except Exception:
                pass

        if self.mode == AttentionMode.FOCUS:
            multiplier *= 0.8
        elif self.mode == AttentionMode.SINGLE_CHAIN:
            multiplier *= 0.6

        self._budget_multiplier = multiplier
        # Dynamic attention should not wait for the raw model window to fill.
        # The configured threshold budget is the upper bound for context
        # injection; mode-specific multipliers decide which necessary context is
        # injected inside that threshold.  This is not lossy compression: evicted
        # envelopes stay registered in the message pool and can be re-injected by
        # switching attention back to GLOBAL or recalling memory.
        self._threshold_budget_base = min(
            self._base_budget,
            self.window_injection_budget_tokens,
        )
        self.budget = max(int(self._threshold_budget_base * multiplier), 1024)

    def _effective_window_budget(self) -> int:
        """Return the current message-selection budget for ``apply_window``."""
        return max(1024, int(self.budget or self._threshold_budget_base or self._base_budget))

    def _remember_last_window(
        self,
        visible_envs: List[MessageEnvelope],
        *,
        window_budget: int,
        evicted_count: int,
        extra_tokens: int = 0,
        extra_messages: int = 0,
    ) -> None:
        """Record telemetry for the last LLM-visible window."""
        visible_tokens = sum(max(0, int(getattr(e, "tokens", 0) or 0)) for e in visible_envs)
        visible_tokens += max(0, int(extra_tokens or 0))
        self._last_window_tokens = visible_tokens
        self._last_window_message_count = len(visible_envs) + max(0, int(extra_messages or 0))
        self._last_evicted_message_count = max(0, int(evicted_count or 0))
        self._last_window_budget = max(1, int(window_budget or 1))

    def serialize(self) -> Dict[str, Any]:
        """序列化当前状态，用于 AgentSession 跨请求持久化"""
        return {
            "mode": self.mode.value,
            "current_node_id": self._current_node_id,
            "seq_counter": self._seq_counter,
            "group_counter": self._group_counter,
            "current_turn": self._current_turn,
            "budget": self.budget,
            "context_window_size": self.context_window_size,
            "reserved_tokens": self.reserved_tokens,
            "envelopes": [
                {
                    "seq": e.seq,
                    "turn": e.turn,
                    "tool_name": e.tool_name,
                    "node_id": e.node_id,
                    "tokens": e.tokens,
                    "is_pinned": e.is_pinned,
                    "weight": e.weight,
                    "group_id": e.group_id,
                    "msg": e.msg,
                }
                for e in self.envelopes
            ],
        }

    @classmethod
    def from_serialized(
        cls,
        data: Dict[str, Any],
        task_graph=None,
        memory_graph=None,
    ) -> "AttentionWindowManager":
        """从序列化数据恢复 AttentionWindowManager 实例"""
        instance = cls(
            context_window_size=data["context_window_size"],
            task_graph=task_graph,
            memory_graph=memory_graph,
            reserved_tokens=data["reserved_tokens"],
        )
        instance.mode = AttentionMode(data["mode"])
        instance.budget = data.get("budget", instance.budget)
        instance._current_node_id = data.get("current_node_id")
        instance._seq_counter = data.get("seq_counter", 0)
        instance._group_counter = data.get("group_counter", 0)
        instance._current_turn = data.get("current_turn", 0)
        for e_data in data.get("envelopes", []):
            env = MessageEnvelope(
                msg=e_data["msg"],
                seq=e_data["seq"],
                turn=e_data["turn"],
                tool_name=e_data.get("tool_name"),
                node_id=e_data.get("node_id"),
                tokens=e_data.get("tokens", 0),
                is_pinned=e_data.get("is_pinned", False),
                weight=e_data.get("weight", 1.0),
                group_id=e_data.get("group_id"),
            )
            instance.envelopes.append(env)
        logger.info(
            f"[AttentionWindow] 从序列化恢复: mode={instance.mode.value}, "
            f"envelopes={len(instance.envelopes)}, "
            f"seq={instance._seq_counter}"
        )
        return instance
