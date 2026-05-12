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
    "search_experience", "task_get_detail",
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
MAX_TOOL_RESULT_CHARS = 10000


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
        self._base_budget = max(
            int((context_window_size - reserved_tokens) * 0.90),
            1024,
        )
        self.budget = self._base_budget

        # P2-13: 动态budget调整因子
        self._budget_multiplier: float = 1.0  # 由任务图节点数动态调整

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
            self._publish_mode_change(old_mode, new_mode, f"tool:{tool_name}")
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
        """task_mark_status 后自动导航注意力窗口

        - completed/skipped: 寻找下一个待处理的兄弟节点并跳转；若无，则回退到父级
        - in_progress: 深入聚焦当前节点
        """
        if not self.task_graph or not node_id:
            return

        if new_status in ("completed", "skipped"):
            # 找父节点 → 找下一个 pending 兄弟
            parent_id = self.task_graph.get_parent(node_id)
            if parent_id:
                siblings = self.task_graph.get_children(parent_id)
                next_pending = None
                for sib in siblings:
                    if sib.id != node_id and sib.status in ("pending", "blocked"):
                        next_pending = sib
                        break
                if next_pending:
                    logger.info(
                        f"[AttentionWindow] 自动导航: 节点 {node_id} 完成 → "
                        f"跳转到兄弟 {next_pending.id}"
                    )
                    self.on_navigate_attention("jump", target_node_id=next_pending.id)
                else:
                    # 同级全部完成，回退到父级视角
                    logger.info(
                        f"[AttentionWindow] 自动导航: 节点 {node_id} 完成, "
                        f"同级已全部完成 → 回退到父级 {parent_id}"
                    )
                    self.on_navigate_attention("broader")

        elif new_status == "in_progress":
            # 深入聚焦当前节点
            depth = self.task_graph.get_node_depth(node_id)
            if depth is not None and depth >= 2:
                logger.info(
                    f"[AttentionWindow] 自动导航: 节点 {node_id} 开始执行 (depth={depth}) → deeper"
                )
                self.on_navigate_attention("jump", target_node_id=node_id)

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
        """根据工具名和参数计算目标模式"""
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

        # P2-13: 动态调整budget（基于任务图节点数和当前模式）
        self._adjust_budget()

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
                    return self._normalize_system_prefix(result)
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

            # 将淘汰内容的语义摘要写回 MemoryGraph（闭合淘汰-恢复环路）
            self._persist_evicted_to_memory(evicted_envs)

            logger.info(
                f"[AttentionWindow] 淘汰 {len(evicted_envs)} 条消息, "
                f"保留 {len(kept_envs)} 条, 模式={self.mode.value}, "
                f"预算={self.budget}, 已用={used_tokens}"
            )

        # 7. 按原始时间顺序排列
        kept_envs.sort(key=lambda e: e.seq)
        result = [e.msg for e in sorted(pinned, key=lambda e: e.seq)]

        # 在 pinned 消息后插入摘要
        if summary_msg:
            result.append(summary_msg)

        result.extend(e.msg for e in kept_envs)
        return self._normalize_system_prefix(result)

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

    # ── 序列化/反序列化（IDE 跨请求 Session 持久化）──

    # ── P2-13: 动态budget调整 ──

    def _adjust_budget(self) -> None:
        """根据任务复杂度动态调整budget

        规则：
        - 任务图节点数 > 20: budget × 1.3（复杂任务需要更多上下文）
        - 任务图节点数 > 50: budget × 1.5
        - FOCUS模式: budget × 0.8（聚焦模式精简上下文）
        - SINGLE_CHAIN模式: budget × 0.6
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
        self.budget = max(int(self._base_budget * multiplier), 1024)

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
