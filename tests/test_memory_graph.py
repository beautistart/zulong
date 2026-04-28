# tests/test_memory_graph.py
"""
记忆图谱 (MemoryGraph) 单元测试

测试内容:
1. 节点 CRUD
2. 边 CRUD
3. BFS 扩散激活算法
4. 赫布学习增强
5. 突触修剪
6. 语义边发现
7. 适配器同步
8. 持久化 (save/load)
9. 注意力窗口图激活加成
10. 降级行为 (MemoryGraph=None)
"""

import os
import sys
import time
import json
import math
import shutil
import unittest
import tempfile

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zulong.memory.memory_graph import (
    MemoryGraph, GraphNode, NodeType, EdgeType, get_memory_graph,
)


class TestMemoryGraphCRUD(unittest.TestCase):
    """节点和边的 CRUD 操作"""

    def setUp(self):
        # 重置单例
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_get_node(self):
        node = GraphNode(
            node_id="task:o1",
            node_type=NodeType.TASK,
            label="需求分析",
            backend_ref="task_graph:o1",
        )
        result = self.graph.add_node(node)
        self.assertEqual(result, "task:o1")

        retrieved = self.graph.get_node("task:o1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.label, "需求分析")
        self.assertEqual(retrieved.node_type, NodeType.TASK)

    def test_update_existing_node(self):
        node1 = GraphNode(
            node_id="task:o1", node_type=NodeType.TASK,
            label="旧标签", metadata={"key1": "val1"},
        )
        self.graph.add_node(node1)

        node2 = GraphNode(
            node_id="task:o1", node_type=NodeType.TASK,
            label="新标签", metadata={"key2": "val2"},
        )
        self.graph.add_node(node2)

        retrieved = self.graph.get_node("task:o1")
        self.assertEqual(retrieved.label, "新标签")
        self.assertIn("key1", retrieved.metadata)
        self.assertIn("key2", retrieved.metadata)

    def test_remove_node(self):
        self.graph.add_node(GraphNode(
            node_id="task:o1", node_type=NodeType.TASK, label="test",
        ))
        self.assertTrue(self.graph.has_node("task:o1"))
        self.assertTrue(self.graph.remove_node("task:o1"))
        self.assertFalse(self.graph.has_node("task:o1"))
        self.assertIsNone(self.graph.get_node("task:o1"))

    def test_remove_nonexistent_node(self):
        self.assertFalse(self.graph.remove_node("nonexistent"))

    def test_get_nodes_by_type(self):
        self.graph.add_node(GraphNode(
            node_id="task:1", node_type=NodeType.TASK, label="t1",
        ))
        self.graph.add_node(GraphNode(
            node_id="task:2", node_type=NodeType.TASK, label="t2",
        ))
        self.graph.add_node(GraphNode(
            node_id="kg:p1", node_type=NodeType.PERSON, label="张三",
        ))

        tasks = self.graph.get_nodes_by_type(NodeType.TASK)
        self.assertEqual(len(tasks), 2)
        persons = self.graph.get_nodes_by_type(NodeType.PERSON)
        self.assertEqual(len(persons), 1)

    def test_add_and_get_edge(self):
        self.graph.add_node(GraphNode(
            node_id="task:o1", node_type=NodeType.TASK, label="parent",
        ))
        self.graph.add_node(GraphNode(
            node_id="task:o1_1", node_type=NodeType.TASK, label="child",
        ))

        result = self.graph.add_edge(
            "task:o1", "task:o1_1",
            EdgeType.HIERARCHY, weight=1.0, protected=True,
        )
        self.assertTrue(result)

        edge = self.graph.get_edge("task:o1", "task:o1_1")
        self.assertIsNotNone(edge)
        self.assertEqual(edge["edge_type"], "hierarchy")
        self.assertEqual(edge["weight"], 1.0)
        self.assertTrue(edge["protected"])

    def test_add_edge_missing_nodes(self):
        """边的源/目标节点不存在时应返回 False"""
        self.graph.add_node(GraphNode(
            node_id="task:o1", node_type=NodeType.TASK, label="exists",
        ))
        result = self.graph.add_edge(
            "task:o1", "task:nonexistent",
            EdgeType.HIERARCHY, weight=1.0,
        )
        self.assertFalse(result)

    def test_remove_edge(self):
        self.graph.add_node(GraphNode(
            node_id="a", node_type=NodeType.TASK, label="a",
        ))
        self.graph.add_node(GraphNode(
            node_id="b", node_type=NodeType.TASK, label="b",
        ))
        self.graph.add_edge("a", "b", EdgeType.REFERENCE, weight=0.5)
        self.assertTrue(self.graph.has_edge("a", "b"))
        self.assertTrue(self.graph.remove_edge("a", "b"))
        self.assertFalse(self.graph.has_edge("a", "b"))


class TestBFSActivation(unittest.TestCase):
    """BFS 扩散激活算法测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

        # 构建测试图: A -> B -> C -> D (链式)
        for nid in ["A", "B", "C", "D"]:
            self.graph.add_node(GraphNode(
                node_id=nid, node_type=NodeType.TASK, label=nid,
            ))

        self.graph.add_edge("A", "B", EdgeType.HIERARCHY, weight=1.0)
        self.graph.add_edge("B", "C", EdgeType.DEPENDENCY, weight=0.8)
        self.graph.add_edge("C", "D", EdgeType.REFERENCE, weight=0.6)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_seed_node_activation(self):
        """种子节点激活值应为 1.0"""
        activations = self.graph.compute_activations(["A"])
        self.assertAlmostEqual(activations["A"], 1.0)

    def test_one_hop_activation(self):
        """一跳邻居: activation = 1.0 * 1.0 (weight) * 0.5 (decay) = 0.5"""
        activations = self.graph.compute_activations(["A"], decay=0.5)
        self.assertAlmostEqual(activations["B"], 0.5)

    def test_two_hop_activation(self):
        """两跳邻居: 0.5 * 0.8 * 0.5 = 0.2"""
        activations = self.graph.compute_activations(["A"], decay=0.5)
        self.assertAlmostEqual(activations["C"], 0.2)

    def test_three_hop_activation(self):
        """三跳邻居: 0.2 * 0.6 * 0.5 = 0.06"""
        activations = self.graph.compute_activations(["A"], decay=0.5)
        self.assertAlmostEqual(activations["D"], 0.06)

    def test_max_depth_cutoff(self):
        """max_depth=2 时不应到达 D (3跳)"""
        activations = self.graph.compute_activations(["A"], max_depth=2, decay=0.5)
        self.assertIn("A", activations)
        self.assertIn("B", activations)
        self.assertIn("C", activations)
        # D 在3跳，max_depth=2 不应出现
        self.assertNotIn("D", activations)

    def test_min_activation_cutoff(self):
        """低于 min_activation 的传播应被剪枝"""
        activations = self.graph.compute_activations(
            ["A"], decay=0.5, min_activation=0.1,
        )
        # D 的激活值 0.06 < 0.1，应被剪枝
        self.assertNotIn("D", activations)

    def test_multiple_seeds(self):
        """多种子节点"""
        activations = self.graph.compute_activations(["A", "D"], decay=0.5)
        self.assertAlmostEqual(activations["A"], 1.0)
        self.assertAlmostEqual(activations["D"], 1.0)

    def test_empty_seeds(self):
        """空种子应返回空结果"""
        activations = self.graph.compute_activations([])
        self.assertEqual(len(activations), 0)

    def test_nonexistent_seed(self):
        """不存在的种子节点应被忽略"""
        activations = self.graph.compute_activations(["nonexistent"])
        self.assertEqual(len(activations), 0)


class TestHebbianLearning(unittest.TestCase):
    """赫布学习测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

        self.graph.add_node(GraphNode(
            node_id="A", node_type=NodeType.TASK, label="A",
        ))
        self.graph.add_node(GraphNode(
            node_id="B", node_type=NodeType.TASK, label="B",
        ))
        self.graph.add_edge("A", "B", EdgeType.REFERENCE, weight=0.5)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_edge_strengthening(self):
        """共激活后边权应增加"""
        old_edge = self.graph.get_edge("A", "B")
        old_weight = old_edge["weight"]

        # 触发激活
        self.graph.compute_activations(["A"])
        # 执行赫布增强
        self.graph.hebbian_strengthen()

        new_edge = self.graph.get_edge("A", "B")
        self.assertGreater(new_edge["weight"], old_weight)

    def test_weight_asymptotic(self):
        """赫布增强不应超过 1.0"""
        self.graph._graph.edges["A", "B"]["weight"] = 0.99
        self.graph.compute_activations(["A"])
        self.graph.hebbian_strengthen()

        edge = self.graph.get_edge("A", "B")
        self.assertLessEqual(edge["weight"], 1.0)

    def test_protected_edge_not_strengthened(self):
        """protected 边不应被赫布增强"""
        self.graph._graph.edges["A", "B"]["protected"] = True
        old_weight = self.graph._graph.edges["A", "B"]["weight"]

        self.graph.compute_activations(["A"])
        self.graph.hebbian_strengthen()

        new_weight = self.graph._graph.edges["A", "B"]["weight"]
        self.assertEqual(new_weight, old_weight)


class TestPruning(unittest.TestCase):
    """突触修剪测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_weak_edge_pruned(self):
        """弱边应被修剪"""
        self.graph.add_node(GraphNode(
            node_id="A", node_type=NodeType.DIALOGUE, label="A",
        ))
        self.graph.add_node(GraphNode(
            node_id="B", node_type=NodeType.DIALOGUE, label="B",
        ))
        self.graph.add_edge("A", "B", EdgeType.ASSOCIATION, weight=0.04)
        # 设置 last_activated 为 48 小时前
        self.graph._graph.edges["A", "B"]["last_activated"] = time.time() - 48 * 3600

        self.graph.decay_and_prune()

        self.assertFalse(self.graph.has_edge("A", "B"))

    def test_protected_edge_not_pruned(self):
        """protected 边不应被修剪"""
        self.graph.add_node(GraphNode(
            node_id="A", node_type=NodeType.TASK, label="A",
        ))
        self.graph.add_node(GraphNode(
            node_id="B", node_type=NodeType.TASK, label="B",
        ))
        self.graph.add_edge("A", "B", EdgeType.HIERARCHY, weight=0.01, protected=True)
        self.graph._graph.edges["A", "B"]["last_activated"] = time.time() - 72 * 3600

        self.graph.decay_and_prune()

        # protected 边即使很弱也不被修剪
        self.assertTrue(self.graph.has_edge("A", "B"))

    def test_orphan_node_pruned(self):
        """孤立且超过24小时的非TASK节点应被移除"""
        self.graph.add_node(GraphNode(
            node_id="orphan", node_type=NodeType.DIALOGUE, label="orphan",
        ))
        self.graph._nodes["orphan"].last_accessed = time.time() - 25 * 3600

        self.graph.decay_and_prune()

        self.assertFalse(self.graph.has_node("orphan"))

    def test_task_node_never_pruned(self):
        """TASK 类型节点即使孤立也不被移除"""
        self.graph.add_node(GraphNode(
            node_id="task:old", node_type=NodeType.TASK, label="old task",
        ))
        self.graph._nodes["task:old"].last_accessed = time.time() - 72 * 3600

        self.graph.decay_and_prune()

        self.assertTrue(self.graph.has_node("task:old"))


class TestPersistence(unittest.TestCase):
    """持久化测试 (save/load)"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        """保存后加载，图结构应一致"""
        graph1 = MemoryGraph(persist_path=self.tmpdir)
        graph1.add_node(GraphNode(
            node_id="task:o1", node_type=NodeType.TASK, label="任务1",
            metadata={"status": "completed"},
        ))
        graph1.add_node(GraphNode(
            node_id="kg:person1", node_type=NodeType.PERSON, label="张三",
        ))
        graph1.add_edge("task:o1", "kg:person1", EdgeType.REFERENCE, weight=0.7)
        graph1.save()

        # 重置单例，创建新实例
        MemoryGraph._instance = None
        graph2 = MemoryGraph(persist_path=self.tmpdir)

        # 验证节点
        self.assertTrue(graph2.has_node("task:o1"))
        self.assertTrue(graph2.has_node("kg:person1"))
        node = graph2.get_node("task:o1")
        self.assertEqual(node.label, "任务1")
        self.assertEqual(node.metadata["status"], "completed")

        # 验证边
        edge = graph2.get_edge("task:o1", "kg:person1")
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge["weight"], 0.7)

    def test_load_empty(self):
        """无数据文件时应正常初始化"""
        graph = MemoryGraph(persist_path=self.tmpdir)
        self.assertEqual(graph.stats["total_nodes"], 0)


class TestTaskGraphAdapter(unittest.TestCase):
    """TaskGraph 适配器测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_sync_task_graph(self):
        """TaskGraph 同步后应在 MemoryGraph 中生成对应节点和边"""
        from zulong.l2.task_graph import TaskGraph

        tg = TaskGraph(title="测试任务")
        graph_id = tg.id  # 获取自动生成的 graph_id
        tg.add_node("req", label="用户需求", type="requirement",
                     status="completed", desc="测试")
        tg.add_node("analysis", label="需求分析", type="analysis",
                     status="pending", desc="")
        tg.add_node("o1", label="大纲1", type="outline",
                     status="pending", desc="第一个大纲")
        tg.add_h_edge("req", "analysis")
        tg.add_h_edge("analysis", "o1")
        tg.add_d_edge("req", "o1", via="需求数据")

        from zulong.memory.graph_adapters import TaskGraphAdapter
        adapter = TaskGraphAdapter()
        count = adapter.sync(self.graph, tg)

        self.assertGreaterEqual(count, 3)
        # sync 使用 "task:{graph_id}/{node_id}" 格式
        self.assertTrue(self.graph.has_node(f"task:{graph_id}/req"))
        self.assertTrue(self.graph.has_node(f"task:{graph_id}/analysis"))
        self.assertTrue(self.graph.has_node(f"task:{graph_id}/o1"))

        # 验证层级边
        hedge = self.graph.get_edge(f"task:{graph_id}/req", f"task:{graph_id}/analysis")
        self.assertIsNotNone(hedge)
        self.assertEqual(hedge["edge_type"], "hierarchy")
        self.assertTrue(hedge["protected"])

        # 验证依赖边
        dedge = self.graph.get_edge(f"task:{graph_id}/req", f"task:{graph_id}/o1")
        self.assertIsNotNone(dedge)
        self.assertEqual(dedge["edge_type"], "dependency")

    def test_incremental_sync(self):
        """增量同步应正确处理事件"""
        from zulong.memory.graph_adapters import TaskGraphAdapter
        adapter = TaskGraphAdapter()

        # 模拟新增节点事件
        adapter.incremental_sync(self.graph, "node_add", {
            "node_id": "o2",
            "label": "新大纲",
            "type": "outline",
            "status": "pending",
            "desc": "描述",
        })

        self.assertTrue(self.graph.has_node("task:o2"))
        node = self.graph.get_node("task:o2")
        self.assertEqual(node.label, "新大纲")


class TestAttentionWindowIntegration(unittest.TestCase):
    """注意力窗口 + 图激活加成测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.memory_graph = MemoryGraph(persist_path=self.tmpdir)

        # 构建测试图
        self.memory_graph.add_node(GraphNode(
            node_id="task:o1", node_type=NodeType.TASK, label="大纲1",
        ))
        self.memory_graph.add_node(GraphNode(
            node_id="task:o1_1", node_type=NodeType.TASK, label="任务1",
        ))
        self.memory_graph.add_node(GraphNode(
            node_id="task:o2", node_type=NodeType.TASK, label="大纲2",
        ))
        self.memory_graph.add_edge(
            "task:o1", "task:o1_1", EdgeType.HIERARCHY, weight=1.0,
        )

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_graph_attention_disabled(self):
        """memory_graph=None 时行为与原版一致（无图激活加成）"""
        from zulong.l2.attention_window import AttentionWindowManager

        attn = AttentionWindowManager(
            context_window_size=65536,
            memory_graph=None,
        )

        attn.register_message(
            {"role": "user", "content": "hello"}, turn=0,
        )
        attn.register_message(
            {"role": "assistant", "content": "hi"}, turn=0, node_id="task:o1",
        )

        result = attn.apply_window()
        self.assertEqual(len(result), 2)

    def test_graph_attention_enabled_boosts_related(self):
        """启用图注意力时，高激活节点应获得更高评分"""
        from zulong.l2.attention_window import AttentionWindowManager

        # 设置 o1_1 高激活值，o2 低激活值
        self.memory_graph.update_node_activation("task:o1_1", 0.9)
        self.memory_graph.update_node_activation("task:o2", 0.0)

        attn = AttentionWindowManager(
            context_window_size=65536,
            memory_graph=self.memory_graph,
        )

        # 注册消息
        attn.register_message(
            {"role": "user", "content": "start"}, turn=0,
        )
        attn.register_message(
            {"role": "assistant", "content": "related msg"},
            turn=1, node_id="task:o1_1",  # 高激活值节点
        )
        attn.register_message(
            {"role": "assistant", "content": "unrelated msg"},
            turn=1, node_id="task:o2",  # 零激活值节点
        )

        # 计算评分：高激活节点应获得 memory_boost 加成
        env_related = attn.envelopes[1]   # task:o1_1
        env_unrelated = attn.envelopes[2]  # task:o2

        score_related = attn._score_message(env_related)
        score_unrelated = attn._score_message(env_unrelated)

        # task:o1_1 (activation=0.9) 的评分应高于 task:o2 (activation=0.0)
        self.assertGreater(score_related, score_unrelated)

    def test_graph_attention_none_graph(self):
        """memory_graph=None 时不应报错"""
        from zulong.l2.attention_window import AttentionWindowManager

        attn = AttentionWindowManager(
            context_window_size=65536,
            memory_graph=None,
        )

        attn.register_message(
            {"role": "user", "content": "test"}, turn=0,
        )

        # 不应抛出异常
        result = attn.apply_window()
        self.assertEqual(len(result), 1)


class TestSemanticNeighbors(unittest.TestCase):
    """语义边发现测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_semantic_edge_creation(self):
        """高相似度节点应自动创建 SEMANTIC 边"""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy 未安装")

        self.graph.add_node(GraphNode(
            node_id="a", node_type=NodeType.DIALOGUE, label="对话A",
        ))
        self.graph.add_node(GraphNode(
            node_id="b", node_type=NodeType.DIALOGUE, label="对话B",
        ))
        self.graph.add_node(GraphNode(
            node_id="c", node_type=NodeType.DIALOGUE, label="对话C",
        ))

        # 设置 embedding: a 和 b 相似, c 不相似
        rng = np.random.RandomState(42)
        vec_a = rng.randn(512).astype(np.float32)
        vec_a /= np.linalg.norm(vec_a)
        vec_b = vec_a + rng.randn(512).astype(np.float32) * 0.01
        vec_b /= np.linalg.norm(vec_b)
        vec_c = rng.randn(512).astype(np.float32)
        vec_c /= np.linalg.norm(vec_c)

        self.graph.set_embedding("a", vec_a)
        self.graph.set_embedding("b", vec_b)
        self.graph.set_embedding("c", vec_c)

        results = self.graph.discover_semantic_neighbors("a", top_k=2, threshold=0.7)

        # a 和 b 应该很相似
        neighbor_ids = [r[0] for r in results]
        self.assertIn("b", neighbor_ids)


class TestGraphStats(unittest.TestCase):
    """统计信息测试"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stats(self):
        self.graph.add_node(GraphNode(
            node_id="task:1", node_type=NodeType.TASK, label="t1",
        ))
        self.graph.add_node(GraphNode(
            node_id="kg:1", node_type=NodeType.KNOWLEDGE, label="k1",
        ))
        self.graph.add_edge("task:1", "kg:1", EdgeType.REFERENCE, weight=0.5)

        stats = self.graph.stats
        self.assertEqual(stats["total_nodes"], 2)
        self.assertEqual(stats["total_edges"], 1)
        self.assertIn("task", stats["node_types"])
        self.assertIn("knowledge", stats["node_types"])
        self.assertIn("reference", stats["edge_types"])


# ============================================================
# 对话适配器增强方法测试
# ============================================================

class TestDialogueAdapterEnhanced(unittest.TestCase):
    """测试 DialogueAdapter 的增量同步方法"""

    def setUp(self):
        MemoryGraph._instance = None
        self.mg = MemoryGraph(persist_path=self.tmp_dir())
        from zulong.memory.graph_adapters import DialogueAdapter
        self.adapter = DialogueAdapter()

    def tmp_dir(self):
        import tempfile
        d = tempfile.mkdtemp()
        return d

    def tearDown(self):
        MemoryGraph._instance = None

    def test_add_round_creates_dialogue_node(self):
        """add_round 应创建 DIALOGUE 类型节点"""
        round_id = self.adapter.add_round(self.mg, "req_001", "你好世界")
        self.assertEqual(round_id, "dialogue:round_req_001")
        node = self.mg.get_node(round_id)
        self.assertIsNotNone(node)
        self.assertEqual(node.node_type, NodeType.DIALOGUE)
        self.assertEqual(node.metadata["sub_type"], "round")
        self.assertEqual(node.metadata["request_id"], "req_001")

    def test_add_round_temporal_edge(self):
        """连续两轮应通过 TEMPORAL 边连接"""
        r1 = self.adapter.add_round(self.mg, "req_001", "第一轮")
        r2 = self.adapter.add_round(self.mg, "req_002", "第二轮", prev_round_id=r1)
        self.assertTrue(self.mg.has_edge(r1, r2))
        edge = self.mg.get_edge(r1, r2)
        self.assertEqual(edge["edge_type"], EdgeType.TEMPORAL.value)

    def test_add_sub_dialogue_hierarchy(self):
        """子对话应通过 HIERARCHY 边连接到父轮次"""
        r1 = self.adapter.add_round(self.mg, "req_001", "父轮次")
        sub_id = self.adapter.add_sub_dialogue(self.mg, r1, turn=3, tool_name="plan_add_node")
        self.assertIsNotNone(self.mg.get_node(sub_id))
        self.assertTrue(self.mg.has_edge(r1, sub_id))
        edge = self.mg.get_edge(r1, sub_id)
        self.assertEqual(edge["edge_type"], EdgeType.HIERARCHY.value)

    def test_add_sub_dialogue_task_reference(self):
        """子对话应通过 REFERENCE 边连接到关联任务节点"""
        r1 = self.adapter.add_round(self.mg, "req_001", "任务关联")
        # 先创建一个 task 节点
        task_node = GraphNode(node_id="task:t1", node_type=NodeType.TASK, label="测试任务")
        self.mg.add_node(task_node)
        sub_id = self.adapter.add_sub_dialogue(self.mg, r1, turn=1, task_node_id="t1")
        self.assertTrue(self.mg.has_edge(sub_id, "task:t1"))

    def test_finalize_round_metadata(self):
        """finalize_round 应更新完成元数据"""
        r1 = self.adapter.add_round(self.mg, "req_001", "待完成")
        self.adapter.finalize_round(self.mg, r1, total_turns=5, status="completed")
        node = self.mg.get_node(r1)
        self.assertEqual(node.metadata["total_turns"], 5)
        self.assertEqual(node.metadata["status"], "completed")
        self.assertIn("completed_at", node.metadata)


# ============================================================
# 前端序列化测试
# ============================================================

class TestFrontendSerialization(unittest.TestCase):
    """测试 to_frontend_dict 和 flush_changes"""

    def setUp(self):
        MemoryGraph._instance = None
        import tempfile
        self.mg = MemoryGraph(persist_path=tempfile.mkdtemp())

    def tearDown(self):
        MemoryGraph._instance = None

    def test_to_frontend_dict_structure(self):
        """to_frontend_dict 应返回 nodes/edges/stats 结构"""
        n1 = GraphNode(node_id="t:1", node_type=NodeType.TASK, label="任务1")
        n2 = GraphNode(node_id="t:2", node_type=NodeType.TASK, label="任务2")
        self.mg.add_node(n1)
        self.mg.add_node(n2)
        self.mg.add_edge("t:1", "t:2", EdgeType.HIERARCHY)

        result = self.mg.to_frontend_dict()
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertIn("stats", result)
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)

        # 检查节点格式
        node_dict = result["nodes"][0]
        self.assertIn("id", node_dict)
        self.assertIn("type", node_dict)
        self.assertIn("label", node_dict)
        self.assertIn("activation", node_dict)

        # 检查边格式
        edge_dict = result["edges"][0]
        self.assertIn("source", edge_dict)
        self.assertIn("target", edge_dict)
        self.assertIn("type", edge_dict)
        self.assertIn("weight", edge_dict)

    def test_flush_changes_returns_delta(self):
        """flush_changes 应返回增量变更并清空缓冲"""
        n1 = GraphNode(node_id="t:1", node_type=NodeType.TASK, label="任务1")
        self.mg.add_node(n1)

        result = self.mg.flush_changes()
        self.assertEqual(result["type"], "delta")
        self.assertGreater(len(result["changes"]), 0)

        # 第二次 flush 应为空
        result2 = self.mg.flush_changes()
        self.assertEqual(len(result2["changes"]), 0)

    def test_changes_tracked_for_all_operations(self):
        """add_node/add_edge/remove_edge/remove_node 都应记录变更"""
        n1 = GraphNode(node_id="t:1", node_type=NodeType.TASK, label="A")
        n2 = GraphNode(node_id="t:2", node_type=NodeType.TASK, label="B")
        self.mg.add_node(n1)
        self.mg.add_node(n2)
        self.mg.add_edge("t:1", "t:2", EdgeType.REFERENCE)
        self.mg.remove_edge("t:1", "t:2")
        self.mg.remove_node("t:2")

        result = self.mg.flush_changes()
        actions = [c["action"] for c in result["changes"]]
        self.assertIn("add_node", actions)
        self.assertIn("add_edge", actions)
        self.assertIn("remove_edge", actions)
        self.assertIn("remove_node", actions)


class TestDialogueTaskHierarchy(unittest.TestCase):
    """测试对话节点与任务图谱的层级关系（HIERARCHY 边）"""

    def setUp(self):
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.graph = MemoryGraph(persist_path=self.tmpdir)
        from zulong.memory.graph_adapters import DialogueAdapter
        self.DialogueAdapter = DialogueAdapter

    def tearDown(self):
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dialogue_hierarchy_to_task(self):
        """对话节点通过 HIERARCHY 边连接任务根节点"""
        adapter = self.DialogueAdapter()
        round_id = adapter.add_round(self.graph, "test_req", "测试任务")
        # 模拟任务节点
        task_node = GraphNode("task:req", NodeType.TASK, "用户需求")
        self.graph.add_node(task_node)
        # 建立 HIERARCHY 边（对话→任务）
        self.graph.add_edge(round_id, "task:req", EdgeType.HIERARCHY, weight=1.0, protected=True)
        # 验证：task:req 有来自对话节点的 HIERARCHY 入边
        in_edges = list(self.graph._graph.in_edges("task:req", data=True))
        hierarchy_edges = [e for e in in_edges if e[2].get("edge_type") == "hierarchy"]
        self.assertEqual(len(hierarchy_edges), 1)
        self.assertEqual(hierarchy_edges[0][0], round_id)

    def test_simple_dialogue_no_task(self):
        """简单对话只创建对话节点，无任务节点"""
        adapter = self.DialogueAdapter()
        adapter.add_round(self.graph, "simple_req", "你好祖龙")
        task_nodes = self.graph.get_nodes_by_type(NodeType.TASK)
        self.assertEqual(len(task_nodes), 0)
        dialogue_nodes = self.graph.get_nodes_by_type(NodeType.DIALOGUE)
        self.assertEqual(len(dialogue_nodes), 1)

    def test_hierarchy_edge_is_protected(self):
        """HIERARCHY 边标记为 protected，不会被修剪"""
        adapter = self.DialogueAdapter()
        round_id = adapter.add_round(self.graph, "req1", "写一个游戏")
        task_node = GraphNode("task:req", NodeType.TASK, "用户需求")
        self.graph.add_node(task_node)
        self.graph.add_edge(round_id, "task:req", EdgeType.HIERARCHY, weight=1.0, protected=True)
        # 验证 protected 属性
        edge_data = self.graph._graph.edges[round_id, "task:req"]
        self.assertTrue(edge_data.get("protected", False))

    # test_complexity_classifier_simple_cases 已移除
    # （complexity_classifier 已废弃，模型自主路由架构替代硬编码分类）

    def test_reuse_external_dialogue_node(self):
        """Orchestrator 复用外部创建的对话节点"""
        adapter = self.DialogueAdapter()
        # 模拟 Gatekeeper 预创建的对话节点
        ext_round_id = adapter.add_round(self.graph, "ext_req", "帮我写一个游戏")
        self.assertTrue(self.graph.has_node(ext_round_id))
        # Orchestrator 检查到节点存在 → 复用
        self.assertEqual(ext_round_id, "dialogue:round_ext_req")
        node = self.graph.get_node(ext_round_id)
        self.assertEqual(node.node_type, NodeType.DIALOGUE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
