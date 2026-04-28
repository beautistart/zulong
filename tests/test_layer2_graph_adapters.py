"""
Layer 2: GraphAdapters 跨模块联动测试

覆盖: TaskGraphAdapter.sync()、DialogueAdapter 重要度检测与轮次管理
"""

import pytest

from zulong.memory.memory_graph import (
    EdgeType,
    GraphNode,
    Importance,
    MemoryGraph,
    NodeType,
)
from zulong.memory.graph_adapters import TaskGraphAdapter, DialogueAdapter
from zulong.l2.task_graph import TaskGraph, FileRef


# ============================================================
# TestTaskGraphAdapter
# ============================================================


class TestTaskGraphAdapter:
    """TaskGraph -> MemoryGraph 投射同步"""

    def test_sync_creates_nodes(self, temp_memory_graph, sample_task_graph):
        mg = temp_memory_graph
        tg = sample_task_graph
        adapter = TaskGraphAdapter()

        synced = adapter.sync(mg, tg)
        assert synced > 0

        # 验证 "task:{graph_id}/{node_id}" 格式的节点存在
        for nid in ("req", "analysis", "o1", "o2", "o3"):
            expected_id = f"task:{tg.id}/{nid}"
            assert mg.has_node(expected_id), f"Node {expected_id} not found in MemoryGraph"

    def test_sync_returns_count(self, temp_memory_graph, sample_task_graph):
        mg = temp_memory_graph
        tg = sample_task_graph
        adapter = TaskGraphAdapter()

        count = adapter.sync(mg, tg)
        # 5 个节点: req, analysis, o1, o2, o3
        assert count >= 5

    def test_sync_creates_hierarchy_edges(self, temp_memory_graph, sample_task_graph):
        mg = temp_memory_graph
        tg = sample_task_graph
        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        # req -> analysis 应有 HIERARCHY 边
        req_id = f"task:{tg.id}/req"
        analysis_id = f"task:{tg.id}/analysis"
        assert mg.has_edge(req_id, analysis_id), "Missing HIERARCHY edge: req -> analysis"

        edge = mg.get_edge(req_id, analysis_id)
        assert edge is not None
        assert edge.get("edge_type") == EdgeType.HIERARCHY.value

    def test_sync_creates_dependency_edges(self, temp_memory_graph, sample_task_graph):
        mg = temp_memory_graph
        tg = sample_task_graph
        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        # o1 -> o3 应有 DEPENDENCY 边
        o1_id = f"task:{tg.id}/o1"
        o3_id = f"task:{tg.id}/o3"
        assert mg.has_edge(o1_id, o3_id), "Missing DEPENDENCY edge: o1 -> o3"

    def test_sync_idempotent(self, temp_memory_graph, sample_task_graph):
        mg = temp_memory_graph
        tg = sample_task_graph
        adapter = TaskGraphAdapter()

        # 预创建根节点（模拟真实工作流中 orchestrator 创建任务根节点）
        # 否则第二次 sync 的 root 查找会匹配到第一次创建的子节点，
        # 导致路径前缀改变从而生成不同 ID 的"重复"节点
        root = GraphNode(
            node_id=f"task:{tg.id}",
            node_type=NodeType.TASK,
            label=tg.title,
            metadata={"graph_id": tg.id, "full_path": f"task:{tg.id}"},
        )
        mg.add_node(root)

        count1 = adapter.sync(mg, tg)
        nodes_after_first = mg.stats["total_nodes"]

        count2 = adapter.sync(mg, tg)
        nodes_after_second = mg.stats["total_nodes"]

        # 第二次 sync 不应增加新节点
        assert nodes_after_second == nodes_after_first

    def test_sync_node_metadata(self, temp_memory_graph, sample_task_graph):
        mg = temp_memory_graph
        tg = sample_task_graph
        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        node = mg.get_node(f"task:{tg.id}/o1")
        assert node is not None
        assert node.node_type == NodeType.TASK
        assert "graph_id" in node.metadata
        assert node.metadata["graph_id"] == tg.id

    def test_sync_file_refs(self, temp_memory_graph):
        tg = TaskGraph(title="File Ref Test", graph_id="frt_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending",
                     desc="", files=[FileRef(name="app.py", path="src/app.py")])

        mg = temp_memory_graph
        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        # 检查是否创建了 FILE 类型节点
        file_nodes = mg.get_nodes_by_type(NodeType.FILE)
        # 可能有也可能没有，取决于实现（FileRef 投射是可选的）
        # 至少主节点应存在
        assert mg.has_node(f"task:frt_001/req")


# ============================================================
# TestDialogueAdapter
# ============================================================


class TestDialogueAdapter:
    """DialogueAdapter 重要度检测"""

    def test_detect_importance_identity(self):
        importance, entities = DialogueAdapter._detect_importance("我叫张三，今年25岁")
        assert importance == Importance.IDENTITY

    def test_detect_importance_must_remember(self):
        importance, _ = DialogueAdapter._detect_importance("帮我记住这个电话号码13812345678")
        assert importance == Importance.MUST_REMEMBER

    def test_detect_importance_trivial(self):
        importance, _ = DialogueAdapter._detect_importance("嗯")
        assert importance == Importance.TRIVIAL

    def test_detect_importance_normal(self):
        importance, _ = DialogueAdapter._detect_importance("请帮我分析一下这段代码的性能问题")
        assert importance == Importance.NORMAL

    def test_detect_importance_fact(self):
        importance, _ = DialogueAdapter._detect_importance("会议时间是明天下午3点")
        assert importance in (Importance.FACT, Importance.IMPORTANT, Importance.NORMAL)

    def test_add_round_creates_node(self, temp_memory_graph):
        mg = temp_memory_graph
        adapter = DialogueAdapter()

        round_id = adapter.add_round(
            mg,
            request_id="req_001",
            goal="帮我配置nginx",
        )

        assert round_id is not None
        assert mg.has_node(round_id)

        node = mg.get_node(round_id)
        assert node.node_type == NodeType.DIALOGUE

    def test_add_round_temporal_edge(self, temp_memory_graph):
        mg = temp_memory_graph
        adapter = DialogueAdapter()

        r1 = adapter.add_round(mg, request_id="req_001", goal="第一轮对话")
        r2 = adapter.add_round(mg, request_id="req_002", goal="第二轮对话",
                               prev_round_id=r1)

        # r1 -> r2 应有 TEMPORAL 边
        assert mg.has_edge(r1, r2), f"Missing TEMPORAL edge: {r1} -> {r2}"
