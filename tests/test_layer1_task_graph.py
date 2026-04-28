"""
Layer 1: TaskGraph 独立模块测试

覆盖: 初始化、节点/边操作、树导航、序列化反序列化
"""

import pytest

from zulong.l2.task_graph import TaskGraph, TaskNode, FileRef, DependencyEdge


# ============================================================
# TestTaskGraphBasics
# ============================================================


class TestTaskGraphBasics:

    def test_init_with_title(self):
        tg = TaskGraph(title="测试任务")
        assert tg.title == "测试任务"
        assert len(tg.id) > 0

    def test_init_with_custom_id(self):
        tg = TaskGraph(title="自定义ID", graph_id="custom_123")
        assert tg.id == "custom_123"

    def test_address_format(self):
        tg = TaskGraph(title="地址测试", graph_id="addr_001")
        assert tg.address == "tg:addr_001"


# ============================================================
# TestNodeOperations
# ============================================================


class TestNodeOperations:

    def test_add_node(self):
        tg = TaskGraph(title="节点测试", graph_id="nt_001")
        node = tg.add_node(id="o1", label="任务A", type="task",
                           status="pending", desc="第一个任务")
        assert isinstance(node, TaskNode)
        assert tg.get_node("o1") is not None
        assert tg.get_node("o1").label == "任务A"

    def test_add_node_with_files(self):
        tg = TaskGraph(title="文件测试", graph_id="ft_001")
        files = [FileRef(name="main.py", path="src/main.py")]
        node = tg.add_node(id="f1", label="带文件", type="task",
                           status="pending", desc="", files=files)
        assert len(node.files) == 1
        assert node.files[0].name == "main.py"

    def test_get_nodes_by_status(self, sample_task_graph):
        tg = sample_task_graph
        pending = tg.get_nodes_by_status("pending")
        in_progress = tg.get_nodes_by_status("in_progress")

        pending_ids = {n.id for n in pending}
        assert "o1" in pending_ids
        assert "o2" in pending_ids
        assert "o3" in pending_ids

        ip_ids = {n.id for n in in_progress}
        assert "req" in ip_ids
        assert "analysis" in ip_ids

    def test_get_nodes_by_type(self, sample_task_graph):
        tg = sample_task_graph
        outlines = tg.get_nodes_by_type("outline")
        assert len(outlines) == 3

    def test_update_node_status(self):
        tg = TaskGraph(title="状态测试", graph_id="st_001")
        tg.add_node(id="s1", label="S1", type="task", status="pending", desc="")

        result = tg.update_node_status("s1", "in_progress")
        assert result is True
        assert tg.get_node("s1").status == "in_progress"

        result = tg.update_node_status("s1", "completed", result="任务完成")
        assert result is True
        assert tg.get_node("s1").status == "completed"
        assert tg.get_node("s1").result == "任务完成"

    def test_update_nonexistent_node(self):
        tg = TaskGraph(title="不存在", graph_id="ne_001")
        result = tg.update_node_status("ghost", "completed")
        assert result is False

    def test_remove_node_cascades(self):
        tg = TaskGraph(title="级联删除", graph_id="cd_001")
        tg.add_node(id="req", label="根", type="requirement", status="pending", desc="")
        tg.add_node(id="a", label="A", type="analysis", status="pending", desc="")
        tg.add_node(id="a1", label="A1", type="outline", status="pending", desc="")
        tg.add_node(id="a2", label="A2", type="outline", status="pending", desc="")
        tg.add_h_edge("req", "a")
        tg.add_h_edge("a", "a1")
        tg.add_h_edge("a", "a2")

        removed = tg.remove_node("a")
        # 应该移除 a, a1, a2
        assert "a" in removed
        assert "a1" in removed
        assert "a2" in removed
        assert tg.get_node("a") is None
        assert tg.get_node("a1") is None

    def test_remove_req_blocked(self, sample_task_graph):
        tg = sample_task_graph
        removed = tg.remove_node("req")
        # req 和 analysis 根节点不能删
        assert tg.get_node("req") is not None


# ============================================================
# TestEdgeOperations
# ============================================================


class TestEdgeOperations:

    def test_add_h_edge(self, sample_task_graph):
        tg = sample_task_graph
        assert ("req", "analysis") in tg._h_edges
        assert ("analysis", "o1") in tg._h_edges

    def test_add_d_edge(self, sample_task_graph):
        tg = sample_task_graph
        d_edges = [(e.s, e.t) for e in tg._d_edges]
        assert ("o1", "o3") in d_edges
        assert ("o2", "o3") in d_edges

        # 检查 via 字段
        o1_o3 = [e for e in tg._d_edges if e.s == "o1" and e.t == "o3"][0]
        assert o1_o3.via == "模块A输出"

    def test_get_dependencies(self, sample_task_graph):
        tg = sample_task_graph
        deps = tg.get_dependencies("o3")
        assert "o1" in deps
        assert "o2" in deps

    def test_get_dependents(self, sample_task_graph):
        tg = sample_task_graph
        dependents = tg.get_dependents("o1")
        assert "o3" in dependents


# ============================================================
# TestTreeNavigation
# ============================================================


class TestTreeNavigation:

    def test_get_children(self, sample_task_graph):
        tg = sample_task_graph
        children = tg.get_children("analysis")
        child_ids = {c.id for c in children}
        assert "o1" in child_ids
        assert "o2" in child_ids
        assert "o3" in child_ids

    def test_get_parent(self, sample_task_graph):
        tg = sample_task_graph
        parent = tg.get_parent("o1")
        assert parent == "analysis"

    def test_get_parent_root(self, sample_task_graph):
        tg = sample_task_graph
        parent = tg.get_parent("req")
        assert parent is None

    def test_get_ancestor_chain(self, sample_task_graph):
        tg = sample_task_graph
        chain = tg.get_ancestor_chain("o1")
        chain_ids = [n.id if isinstance(n, TaskNode) else n for n in chain]
        assert "req" in chain_ids
        assert "analysis" in chain_ids

    def test_get_all_descendants(self, sample_task_graph):
        tg = sample_task_graph
        desc = tg.get_all_descendants("req")
        assert "analysis" in desc
        assert "o1" in desc
        assert "o2" in desc
        assert "o3" in desc

    def test_get_leaf_nodes(self, sample_task_graph):
        tg = sample_task_graph
        leaves = tg.get_leaf_nodes()
        leaf_ids = {n.id for n in leaves}
        assert "o1" in leaf_ids
        assert "o2" in leaf_ids
        assert "o3" in leaf_ids
        assert "req" not in leaf_ids

    def test_depth_to_type(self):
        assert TaskGraph.depth_to_type(0) == "requirement"
        assert TaskGraph.depth_to_type(1) == "analysis"
        assert TaskGraph.depth_to_type(2) == "outline"
        assert TaskGraph.depth_to_type(3) == "task"
        assert TaskGraph.depth_to_type(4) == "subtask"
        assert TaskGraph.depth_to_type(10) == "subtask"


# ============================================================
# TestSerialization
# ============================================================


class TestSerialization:

    def test_serialize_deserialize_roundtrip(self, sample_task_graph):
        tg = sample_task_graph
        data = tg.serialize()

        tg2 = TaskGraph.deserialize(data)
        assert tg2.id == tg.id
        assert tg2.title == tg.title
        assert len(tg2._nodes) == len(tg._nodes)
        assert len(tg2._h_edges) == len(tg._h_edges)
        assert len(tg2._d_edges) == len(tg._d_edges)

        # 验证节点内容
        for nid, node in tg._nodes.items():
            node2 = tg2.get_node(nid)
            assert node2 is not None
            assert node2.label == node.label
            assert node2.status == node.status
            assert node2.type == node.type

    def test_deserialize_handles_bad_edges(self):
        data = {
            "id": "bad_edge_test",
            "title": "坏边测试",
            "created_at": 0,
            "nodes": {
                "req": {"id": "req", "label": "R", "type": "requirement",
                        "status": "pending", "desc": ""},
            },
            "h_edges": [["req", "ghost"]],  # ghost 不存在
            "d_edges": [],
            "parallel_groups": [],
            "metadata": {},
        }
        tg = TaskGraph.deserialize(data)
        assert tg.get_node("req") is not None
        # 坏边应被跳过
        bad = [e for e in tg._h_edges if "ghost" in e]
        assert len(bad) == 0

    def test_parallel_groups_preserved(self, sample_task_graph):
        tg = sample_task_graph
        tg.parallel_groups = [["o1", "o2"]]

        data = tg.serialize()
        tg2 = TaskGraph.deserialize(data)
        assert tg2.parallel_groups == [["o1", "o2"]]

    def test_file_ref_roundtrip(self):
        tg = TaskGraph(title="文件序列化", graph_id="fr_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending", desc="",
                     files=[FileRef(name="app.py", path="src/app.py")])

        data = tg.serialize()
        tg2 = TaskGraph.deserialize(data)
        node = tg2.get_node("req")
        assert len(node.files) == 1
        assert node.files[0].name == "app.py"
        assert node.files[0].path == "src/app.py"
