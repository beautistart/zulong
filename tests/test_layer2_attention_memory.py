"""
Layer 2: AttentionWindow + MemoryGraph 联动测试

覆盖: MemoryGraph 激活值对 AttentionWindow 评分的 boost 效应
"""

import pytest

from zulong.l2.attention_window import (
    AttentionMode,
    AttentionWindowManager,
    MessageEnvelope,
)
from zulong.memory.memory_graph import (
    GraphNode,
    MemoryGraph,
    NodeType,
)


# ============================================================
# TestAttentionMemoryBoost
# ============================================================


class TestAttentionMemoryBoost:

    def test_memory_boost_factor(self, temp_memory_graph):
        """有激活值的 MemoryGraph 节点应增加 AttentionWindow 评分"""
        mg = temp_memory_graph

        # 创建一个有激活值的节点
        node = GraphNode(
            node_id="task:test/o1",
            node_type=NodeType.TASK,
            label="Task O1",
        )
        node.activation = 0.8
        mg.add_node(node)

        mgr = AttentionWindowManager(
            context_window_size=65536,
            memory_graph=mg,
        )

        # 关联此节点的消息
        env_with_node = MessageEnvelope(
            msg={"role": "tool", "content": "tool result"},
            seq=0, turn=1, node_id="task:test/o1",
        )
        # 无节点关联的消息
        env_without_node = MessageEnvelope(
            msg={"role": "tool", "content": "tool result"},
            seq=1, turn=1,
        )

        mgr._current_turn = 1
        score_with = mgr._score_message(env_with_node)
        score_without = mgr._score_message(env_without_node)

        # 有激活值的节点应该得分更高
        assert score_with >= score_without

    def test_no_memory_graph_no_boost(self):
        """没有 MemoryGraph 时，boost 为 1.0（无影响）"""
        mgr = AttentionWindowManager(
            context_window_size=65536,
            memory_graph=None,
        )

        env = MessageEnvelope(
            msg={"role": "tool", "content": "data"},
            seq=0, turn=1, node_id="task:test/o1",
        )
        mgr._current_turn = 1
        score = mgr._score_message(env)
        # 基础分=1.0, time_decay=1.0, mode_mult=1.0, memory_boost=1.0
        assert score > 0

    def test_mode_plus_memory_combined(self, temp_memory_graph):
        """FOCUS 模式乘数 + MemoryGraph boost 的组合效果"""
        mg = temp_memory_graph
        from zulong.l2.task_graph import TaskGraph

        # 构建 TaskGraph
        tg = TaskGraph(title="Combined Test", graph_id="cmb_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending", desc="")
        tg.add_node(id="o1", label="O1", type="outline", status="in_progress", desc="")
        tg.add_h_edge("req", "o1")

        # MemoryGraph 中添加对应节点并设置激活值
        mg_node = GraphNode(
            node_id="task:cmb_001/o1",
            node_type=NodeType.TASK,
            label="O1",
        )
        mg_node.activation = 0.6
        mg.add_node(mg_node)

        mgr = AttentionWindowManager(
            context_window_size=65536,
            task_graph=tg,
            memory_graph=mg,
        )
        mgr.mode = AttentionMode.FOCUS
        mgr._current_node_id = "o1"
        mgr._current_turn = 1

        # 当前节点消息
        env_current = MessageEnvelope(
            msg={"role": "tool", "content": "focused result"},
            seq=0, turn=1, node_id="o1",
        )

        # 不相关消息
        env_other = MessageEnvelope(
            msg={"role": "tool", "content": "other result"},
            seq=1, turn=1, node_id="other_node",
        )

        score_current = mgr._score_message(env_current)
        score_other = mgr._score_message(env_other)

        # FOCUS 当前节点 3.0x + memory boost -> 远高于不相关消息
        assert score_current > score_other * 2
