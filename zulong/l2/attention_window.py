# File: zulong/l2/attention_window.py
"""
动态注意力窗口 (Dynamic Attention Window)

将 Agent 主循环的消息历史控制在模型上下文窗口的安全范围内。

三种注意力模式：
  - GLOBAL: 全局视角，关注大纲和整体结构，深层节点权重递减
  - FOCUS: 聚焦某个节点的细节，提高关联上下文权重
  - SINGLE_CHAIN: 单链推理，只保留当前执行链路的高权重信息

模式切换由工具调用驱动（状态机），无需额外 LLM 判断。
navigate_attention 工具可直接控制模式和 BFS 种子位置。
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Set

logger = logging.getLogger(__name__)


class AttentionMode(Enum):
    """注意力模式"""
    GLOBAL = "global"             # 全局：关注大纲和整体进度
    FOCUS = "focus"               # 聚焦：关注某节点的细节和关联
    SINGLE_CHAIN = "single_chain" # 单链：深度推理，淘汰不相关内容


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


# ── 模式切换触发器（适配当前工具名）──

# 这些工具触发 GLOBAL → FOCUS
_FOCUS_TRIGGERS: Set[str] = {
    "recall_memory", "read_memory_node", "discover_related",
    "search_experience",
}

# 这些工具触发 FOCUS → SINGLE_CHAIN
_SINGLE_CHAIN_TRIGGERS: Set[str] = {
    "exec_write_file", "exec_run_command",
}

# 这些工具强制回到 GLOBAL
_GLOBAL_FORCE_TRIGGERS: Set[str] = {
    "task_view_overview", "submit_final_answer",
    "task_list_suspended",
}

# 工具结果最大字符数（超出截断）
MAX_TOOL_RESULT_CHARS = 2000


class AttentionWindowManager:
    """动态注意力窗口管理器

    核心职责：
    1. 接收所有消息（register_message）
    2. 根据工具调用驱动模式切换（observe_tool_call）
    3. 在 LLM 调用前裁剪消息（apply_window）
    4. 联动 navigate_attention 工具的模式切换
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
        self.budget = max(
            int((context_window_size - reserved_tokens) * 0.90),
            1024,
        )

        self.mode: AttentionMode = AttentionMode.GLOBAL
        self.envelopes: List[MessageEnvelope] = []
        self._current_turn: int = 0
        self._current_node_id: Optional[str] = None
        self._seq_counter: int = 0
        self._group_counter: int = 0

        logger.info(
            f"[AttentionWindow] 初始化: context={context_window_size}, "
            f"reserved={reserved_tokens}, budget={self.budget}"
        )

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
        """观察工具调用，驱动模式状态机

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
                f"[AttentionWindow] 模式切换: {old_mode.value} → {new_mode.value} "
                f"(触发: {tool_name}, node={self._current_node_id})"
            )
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

    def _compute_transition(
        self,
        tool_name: str,
        tool_args: Dict,
    ) -> Optional[AttentionMode]:
        """根据工具名和参数计算目标模式"""
        # navigate_attention 由 on_navigate_attention 单独处理
        if tool_name == "navigate_attention":
            return None

        # 强制回全局
        if tool_name in _GLOBAL_FORCE_TRIGGERS:
            return AttentionMode.GLOBAL

        # task_mark_status 根据 status 参数决定
        if tool_name == "task_mark_status":
            status = tool_args.get("status", "")
            if status in ("completed", "skipped"):
                return AttentionMode.GLOBAL
            if status == "in_progress":
                if self.mode == AttentionMode.FOCUS:
                    return AttentionMode.SINGLE_CHAIN
                return None

        # GLOBAL → FOCUS
        if self.mode == AttentionMode.GLOBAL:
            if tool_name in _FOCUS_TRIGGERS:
                return AttentionMode.FOCUS

        # FOCUS → SINGLE_CHAIN
        if self.mode == AttentionMode.FOCUS:
            if tool_name in _SINGLE_CHAIN_TRIGGERS:
                return AttentionMode.SINGLE_CHAIN

        return None

    # ── 窗口裁剪 ──

    def apply_window(self) -> List[Dict]:
        """对所有消息评分、裁剪，返回适合 LLM 的消息列表

        Returns:
            经过窗口过滤的消息列表（保持时间顺序）
        """
        if not self.envelopes:
            return []

        # 1. 对所有消息评分
        for env in self.envelopes:
            env.weight = self._score_message(env)

        # 2. 分离 pinned 和 non-pinned
        pinned = [e for e in self.envelopes if e.is_pinned]
        unpinned = [e for e in self.envelopes if not e.is_pinned]

        pinned_tokens = sum(e.tokens for e in pinned)
        remaining_budget = self.budget - pinned_tokens

        if remaining_budget <= 0:
            logger.warning(
                f"[AttentionWindow] pinned 消息超预算: "
                f"{pinned_tokens} > {self.budget}"
            )
            # 渐进式降级：保留首尾 pinned，其余降级参与权重排序
            sorted_pinned = sorted(pinned, key=lambda e: e.seq)
            if len(sorted_pinned) > 2:
                essential = [sorted_pinned[0], sorted_pinned[-1]]
                demoted = sorted_pinned[1:-1]
                essential_tokens = sum(e.tokens for e in essential)
                demoted_budget = self.budget - essential_tokens
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
                    return result
            # 兜底：仍然只返回 pinned
            return [e.msg for e in sorted_pinned]

        # 3. 按组处理：计算每个组的最高权重和总 tokens
        group_info: Dict[Optional[int], Dict] = {}
        for env in unpinned:
            gid = env.group_id
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

        # 5. 贪心选择：从高权重到低权重累加，直到用尽预算
        kept_envs: List[MessageEnvelope] = []
        evicted_envs: List[MessageEnvelope] = []
        used_tokens = 0

        for group in sorted_groups:
            if used_tokens + group["total_tokens"] <= remaining_budget:
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

            logger.debug(
                f"[AttentionWindow] 淘汰 {len(evicted_envs)} 条消息, "
                f"保留 {len(kept_envs)} 条, 模式={self.mode.value}"
            )

        # 7. 按原始时间顺序排列
        kept_envs.sort(key=lambda e: e.seq)
        result = [e.msg for e in sorted(pinned, key=lambda e: e.seq)]

        # 在 pinned 消息后插入摘要
        if summary_msg:
            result.append(summary_msg)

        result.extend(e.msg for e in kept_envs)
        return result

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
        """单链模式：只保留当前执行链路，大幅淘汰不相关内容"""
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
        """为被淘汰的消息生成简短摘要"""
        if not evicted:
            return ""

        # 收集被淘汰的工具调用和节点
        tool_counts: Dict[str, int] = {}
        node_ids: Set[str] = set()

        for env in evicted:
            if env.tool_name:
                tool_counts[env.tool_name] = (
                    tool_counts.get(env.tool_name, 0) + 1
                )
            if env.node_id:
                node_ids.add(env.node_id)

        parts = [f"[上下文窗口管理] 已淘汰 {len(evicted)} 条历史消息以控制上下文长度。"]

        if tool_counts:
            tool_summary = ", ".join(
                f"{name}×{count}" for name, count in
                sorted(tool_counts.items(), key=lambda x: -x[1])[:5]
            )
            parts.append(f"涉及工具调用: {tool_summary}")

        if node_ids:
            nodes_str = ", ".join(sorted(node_ids)[:8])
            if len(node_ids) > 8:
                nodes_str += f" 等共 {len(node_ids)} 个节点"
            parts.append(f"涉及节点: {nodes_str}")

        parts.append("如需回顾已淘汰的内容，请使用 recall_memory 或 read_memory_node 工具重新查询。")

        summary = " ".join(parts)

        # 摘要本身不超过预算的 10%
        max_summary_tokens = int(self.budget * 0.10)
        summary_tokens = estimate_tokens(summary)
        if summary_tokens > max_summary_tokens:
            ratio = max_summary_tokens / max(summary_tokens, 1)
            summary = summary[:int(len(summary) * ratio)] + "..."

        return summary

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
            "total_messages": len(self.envelopes),
            "total_tokens": total_tokens,
            "pinned_tokens": pinned_tokens,
            "remaining_tokens": max(0, self.budget - total_tokens),
            "usage_ratio": self.usage_ratio,
            "current_node_id": self._current_node_id,
            "context_window_size": self.context_window_size,
        }
