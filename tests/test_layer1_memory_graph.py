"""
Layer 1: MemoryGraph 独立模块测试

覆盖: CRUD、多维标签、BFS 扩散激活、Hebbian 学习、衰减修剪、持久化
"""

import math
import time

import pytest

from zulong.memory.memory_graph import (
    EdgeType,
    GraphNode,
    Importance,
    MemoryGraph,
    NodeType,
    Temperature,
)


# ============================================================
# TestGraphNodeBasics - 不需要 MemoryGraph 实例
# ============================================================


class TestGraphNodeBasics:
    """GraphNode 数据类基本操作"""

    def test_create_node(self):
        node = GraphNode(
            node_id="test:n1",
            node_type=NodeType.KNOWLEDGE,
            label="测试知识节点",
            metadata={"content": "一些知识内容"},
        )
        assert node.node_id == "test:n1"
        assert node.node_type == NodeType.KNOWLEDGE
        assert node.label == "测试知识节点"
        assert node.activation == 0.0
        assert node.access_count == 0
        assert node.metadata["content"] == "一些知识内容"

    def test_node_serialize_roundtrip(self):
        node = GraphNode(
            node_id="test:rt1",
            node_type=NodeType.DIALOGUE,
            label="对话轮次",
            activation=0.75,
            access_count=5,
            backend_ref="stm:turn_42",
            metadata={"sub_type": "round", "goal": "帮我分析代码"},
        )
        data = node.to_dict()
        restored = GraphNode.from_dict(data)

        assert restored.node_id == node.node_id
        assert restored.node_type == node.node_type
        assert restored.label == node.label
        assert restored.activation == node.activation
        assert restored.access_count == node.access_count
        assert restored.backend_ref == node.backend_ref
        assert restored.metadata["goal"] == "帮我分析代码"

    def test_enum_values(self):
        # NodeType: 9 种
        assert len(NodeType) == 9
        expected_node_types = {
            "task", "dialogue", "knowledge", "experience",
            "episode", "file", "concept", "person", "document",
        }
        actual = {nt.value for nt in NodeType}
        assert actual == expected_node_types

        # EdgeType: 7 种
        assert len(EdgeType) == 7
        expected_edge_types = {
            "hierarchy", "dependency", "reference", "temporal",
            "semantic", "causal", "association",
        }
        actual = {et.value for et in EdgeType}
        assert actual == expected_edge_types

        # Importance: 6 种
        assert len(Importance) == 6

        # Temperature: 3 种
        assert len(Temperature) == 3


# ============================================================
# TestMemoryGraphCRUD - 节点和边的增删改查
# ============================================================


class TestMemoryGraphCRUD:
    """MemoryGraph 节点/边 CRUD 操作"""

    def test_add_and_get_node(self, temp_memory_graph):
        mg = temp_memory_graph
        node = GraphNode(
            node_id="crud:n1",
            node_type=NodeType.KNOWLEDGE,
            label="CRUD测试节点",
        )
        node_id = mg.add_node(node)
        assert node_id == "crud:n1"

        retrieved = mg.get_node("crud:n1")
        assert retrieved is not None
        assert retrieved.label == "CRUD测试节点"
        assert retrieved.node_type == NodeType.KNOWLEDGE

    def test_has_node(self, temp_memory_graph):
        mg = temp_memory_graph
        assert not mg.has_node("nonexistent:x")

        mg.add_node(GraphNode(
            node_id="crud:exists",
            node_type=NodeType.CONCEPT,
            label="存在的节点",
        ))
        assert mg.has_node("crud:exists")

    def test_remove_node(self, temp_memory_graph):
        mg = temp_memory_graph
        n1 = GraphNode(node_id="rm:a", node_type=NodeType.KNOWLEDGE, label="A")
        n2 = GraphNode(node_id="rm:b", node_type=NodeType.KNOWLEDGE, label="B")
        mg.add_node(n1)
        mg.add_node(n2)
        mg.add_edge("rm:a", "rm:b", EdgeType.SEMANTIC, weight=0.5)

        assert mg.has_node("rm:a")
        assert mg.has_edge("rm:a", "rm:b")

        mg.remove_node("rm:a")
        assert not mg.has_node("rm:a")
        assert not mg.has_edge("rm:a", "rm:b")

    def test_get_nodes_by_type(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="t:k1", node_type=NodeType.KNOWLEDGE, label="K1"))
        mg.add_node(GraphNode(node_id="t:k2", node_type=NodeType.KNOWLEDGE, label="K2"))
        mg.add_node(GraphNode(node_id="t:d1", node_type=NodeType.DIALOGUE, label="D1"))

        knowledge_nodes = mg.get_nodes_by_type(NodeType.KNOWLEDGE)
        assert len(knowledge_nodes) >= 2
        assert all(n.node_type == NodeType.KNOWLEDGE for n in knowledge_nodes)

    def test_add_edge(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="e:a", node_type=NodeType.KNOWLEDGE, label="A"))
        mg.add_node(GraphNode(node_id="e:b", node_type=NodeType.KNOWLEDGE, label="B"))

        result = mg.add_edge("e:a", "e:b", EdgeType.SEMANTIC, weight=0.8)
        assert result is True
        assert mg.has_edge("e:a", "e:b")

    def test_add_edge_missing_node(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="e:only", node_type=NodeType.KNOWLEDGE, label="Only"))
        result = mg.add_edge("e:only", "e:ghost", EdgeType.SEMANTIC)
        assert result is False

    def test_edge_weight_max(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="w:a", node_type=NodeType.KNOWLEDGE, label="A"))
        mg.add_node(GraphNode(node_id="w:b", node_type=NodeType.KNOWLEDGE, label="B"))

        mg.add_edge("w:a", "w:b", EdgeType.SEMANTIC, weight=0.3)
        edge1 = mg.get_edge("w:a", "w:b")
        assert edge1 is not None
        w1 = edge1.get("weight", 0)

        # 用更高权重重新添加
        mg.add_edge("w:a", "w:b", EdgeType.SEMANTIC, weight=0.9)
        edge2 = mg.get_edge("w:a", "w:b")
        w2 = edge2.get("weight", 0)
        assert w2 >= w1

    def test_remove_edge(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="re:a", node_type=NodeType.KNOWLEDGE, label="A"))
        mg.add_node(GraphNode(node_id="re:b", node_type=NodeType.KNOWLEDGE, label="B"))
        mg.add_edge("re:a", "re:b", EdgeType.SEMANTIC, weight=0.5)

        assert mg.has_edge("re:a", "re:b")
        mg.remove_edge("re:a", "re:b")
        assert not mg.has_edge("re:a", "re:b")


# ============================================================
# TestMultiDimensionalLabels - 温度、重要度、时效性标签
# ============================================================


class TestMultiDimensionalLabels:
    """多维标签系统"""

    def test_temperature_hot(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(
            node_id="temp:hot", node_type=NodeType.KNOWLEDGE, label="Hot",
        ))
        temp = mg.get_temperature("temp:hot")
        assert temp == Temperature.HOT

    def test_temperature_cold(self, temp_memory_graph):
        mg = temp_memory_graph
        node = GraphNode(
            node_id="temp:cold", node_type=NodeType.KNOWLEDGE, label="Cold",
        )
        # 设置 last_accessed 为 25 小时前
        node.last_accessed = time.time() - 90000
        mg.add_node(node, touch=False)

        temp = mg.get_temperature("temp:cold")
        assert temp == Temperature.COLD

    def test_importance_default_normal(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(
            node_id="imp:default", node_type=NodeType.KNOWLEDGE, label="Default",
        ))
        imp = mg.get_importance("imp:default")
        assert imp == Importance.NORMAL

    def test_set_importance(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(
            node_id="imp:set", node_type=NodeType.KNOWLEDGE, label="Set",
        ))
        mg.set_importance("imp:set", Importance.IMPORTANT)
        assert mg.get_importance("imp:set") == Importance.IMPORTANT

    def test_promote_importance_only_up(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(
            node_id="imp:promo", node_type=NodeType.KNOWLEDGE, label="Promo",
        ))
        mg.set_importance("imp:promo", Importance.IMPORTANT)

        # 尝试降级到 NORMAL - 应该不生效
        result = mg.promote_importance("imp:promo", Importance.NORMAL)
        assert result is False
        assert mg.get_importance("imp:promo") == Importance.IMPORTANT

        # 升级到 MUST_REMEMBER - 应该生效
        result = mg.promote_importance("imp:promo", Importance.MUST_REMEMBER)
        assert result is True
        assert mg.get_importance("imp:promo") == Importance.MUST_REMEMBER

    def test_is_recent(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(
            node_id="rec:fresh", node_type=NodeType.KNOWLEDGE, label="Fresh",
        ))
        assert mg.is_recent("rec:fresh") is True

        # 设置过期节点
        node = GraphNode(
            node_id="rec:old", node_type=NodeType.KNOWLEDGE, label="Old",
        )
        node.last_accessed = time.time() - 7200  # 2 小时前
        mg.add_node(node, touch=False)
        assert mg.is_recent("rec:old") is False


# ============================================================
# TestBFSActivation - BFS 扩散激活
# ============================================================


class TestBFSActivation:
    """BFS 加权扩散激活算法"""

    def _build_chain(self, mg, count=4, weight=0.8):
        """构建链式图: n0 -> n1 -> n2 -> ..."""
        for i in range(count):
            mg.add_node(GraphNode(
                node_id=f"bfs:n{i}",
                node_type=NodeType.KNOWLEDGE,
                label=f"Node{i}",
            ))
        for i in range(count - 1):
            mg.add_edge(f"bfs:n{i}", f"bfs:n{i+1}", EdgeType.SEMANTIC, weight=weight)

    def test_seed_activation_1(self, temp_memory_graph):
        mg = temp_memory_graph
        self._build_chain(mg, count=3)

        activations = mg.compute_activations(
            seed_node_ids=["bfs:n0"], max_depth=2, decay=0.5,
        )
        assert activations.get("bfs:n0", 0) == 1.0

    def test_propagation_decay(self, temp_memory_graph):
        mg = temp_memory_graph
        self._build_chain(mg, count=3, weight=0.8)

        activations = mg.compute_activations(
            seed_node_ids=["bfs:n0"], max_depth=2, decay=0.5,
        )
        # n1: 1.0 * 0.8 * 0.5 = 0.4
        n1_act = activations.get("bfs:n1", 0)
        assert 0.2 < n1_act < 0.6, f"n1 activation={n1_act}, expected ~0.4"

        # n2: 比 n1 更小
        n2_act = activations.get("bfs:n2", 0)
        assert n2_act < n1_act

    def test_max_depth_limit(self, temp_memory_graph):
        mg = temp_memory_graph
        self._build_chain(mg, count=5, weight=0.9)

        activations = mg.compute_activations(
            seed_node_ids=["bfs:n0"], max_depth=1, decay=0.5,
        )
        # max_depth=1 只传播一跳
        assert activations.get("bfs:n0", 0) == 1.0
        assert "bfs:n1" in activations
        # n2 应该不在结果中或激活为 0
        assert activations.get("bfs:n2", 0) == 0

    def test_min_activation_cutoff(self, temp_memory_graph):
        mg = temp_memory_graph
        self._build_chain(mg, count=4, weight=0.05)  # 非常弱的边

        activations = mg.compute_activations(
            seed_node_ids=["bfs:n0"], max_depth=3, decay=0.5,
            min_activation=0.01,
        )
        # 弱边传播: 1.0 * 0.05 * 0.5 = 0.025, 再传播更小
        # n3 应该低于 min_activation
        assert activations.get("bfs:n3", 0) < 0.01

    def test_multi_seed(self, temp_memory_graph):
        mg = temp_memory_graph
        # 三角形: A--B--C, A--C
        for nid in ("ms:a", "ms:b", "ms:c"):
            mg.add_node(GraphNode(
                node_id=nid, node_type=NodeType.KNOWLEDGE, label=nid,
            ))
        mg.add_edge("ms:a", "ms:b", EdgeType.SEMANTIC, weight=0.8)
        mg.add_edge("ms:b", "ms:c", EdgeType.SEMANTIC, weight=0.8)
        mg.add_edge("ms:a", "ms:c", EdgeType.SEMANTIC, weight=0.5)

        activations = mg.compute_activations(
            seed_node_ids=["ms:a", "ms:c"], max_depth=2, decay=0.5,
        )
        # 两个种子都应为 1.0
        assert activations.get("ms:a", 0) == 1.0
        assert activations.get("ms:c", 0) == 1.0
        # B 从两边接收激活，应该比单源更高
        b_act = activations.get("ms:b", 0)
        assert b_act > 0.2


# ============================================================
# TestHebbianLearning - 赫布学习
# ============================================================


class TestHebbianLearning:
    """Hebbian 学习: 共激活边权增强"""

    def test_hebbian_increases_weight(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="hb:a", node_type=NodeType.KNOWLEDGE, label="A"))
        mg.add_node(GraphNode(node_id="hb:b", node_type=NodeType.KNOWLEDGE, label="B"))
        mg.add_edge("hb:a", "hb:b", EdgeType.SEMANTIC, weight=0.5)

        # BFS 激活
        mg.compute_activations(["hb:a"], max_depth=1, decay=0.5)

        old_edge = mg.get_edge("hb:a", "hb:b")
        old_weight = old_edge["weight"]

        # 赫布增强
        mg.hebbian_strengthen()

        new_edge = mg.get_edge("hb:a", "hb:b")
        new_weight = new_edge["weight"]
        assert new_weight > old_weight, f"Expected weight increase: {old_weight} -> {new_weight}"

    def test_hebbian_asymptotic(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="ha:a", node_type=NodeType.KNOWLEDGE, label="A"))
        mg.add_node(GraphNode(node_id="ha:b", node_type=NodeType.KNOWLEDGE, label="B"))
        mg.add_edge("ha:a", "ha:b", EdgeType.SEMANTIC, weight=0.95)

        # 多轮 BFS + Hebbian
        for _ in range(10):
            mg.compute_activations(["ha:a"], max_depth=1, decay=0.5)
            mg.hebbian_strengthen()

        edge = mg.get_edge("ha:a", "ha:b")
        assert edge["weight"] <= 1.0, f"Weight exceeded 1.0: {edge['weight']}"

    def test_hebbian_skips_protected(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="hp:a", node_type=NodeType.TASK, label="A"))
        mg.add_node(GraphNode(node_id="hp:b", node_type=NodeType.TASK, label="B"))
        mg.add_edge("hp:a", "hp:b", EdgeType.HIERARCHY, weight=0.5, protected=True)

        initial_edge = mg.get_edge("hp:a", "hp:b")
        initial_weight = initial_edge["weight"]

        mg.compute_activations(["hp:a"], max_depth=1, decay=0.5)
        mg.hebbian_strengthen()

        final_edge = mg.get_edge("hp:a", "hp:b")
        # Protected 边不应被 Hebbian 修改
        assert final_edge["weight"] == initial_weight


# ============================================================
# TestDecayAndPrune - Ebbinghaus 衰减修剪
# ============================================================


class TestDecayAndPrune:
    """突触修剪: 基于 Ebbinghaus 遗忘曲线的边权衰减"""

    def test_trivial_edge_decays_fast(self, temp_memory_graph):
        mg = temp_memory_graph
        # 创建两个 TRIVIAL 节点
        n1 = GraphNode(node_id="dc:a", node_type=NodeType.DIALOGUE, label="A")
        n2 = GraphNode(node_id="dc:b", node_type=NodeType.DIALOGUE, label="B")
        mg.add_node(n1)
        mg.add_node(n2)
        mg.set_importance("dc:a", Importance.TRIVIAL)
        mg.set_importance("dc:b", Importance.TRIVIAL)

        mg.add_edge("dc:a", "dc:b", EdgeType.ASSOCIATION, weight=0.5)

        # 模拟边的 last_activated 为 12 小时前 (半衰期=6h 的 TRIVIAL)
        edge_data = mg._graph.edges["dc:a", "dc:b"]
        edge_data["last_activated"] = time.time() - 43200  # 12h ago

        mg.decay_and_prune()

        edge_after = mg.get_edge("dc:a", "dc:b")
        if edge_after is not None:
            # 12h = 2 个半衰期, 衰减后约 0.5 * 0.25 = 0.125
            assert edge_after["weight"] < 0.5

    def test_must_remember_never_decays(self, temp_memory_graph):
        mg = temp_memory_graph
        mg.add_node(GraphNode(node_id="mr:a", node_type=NodeType.KNOWLEDGE, label="A"))
        mg.add_node(GraphNode(node_id="mr:b", node_type=NodeType.KNOWLEDGE, label="B"))
        mg.set_importance("mr:a", Importance.MUST_REMEMBER)
        mg.set_importance("mr:b", Importance.MUST_REMEMBER)

        mg.add_edge("mr:a", "mr:b", EdgeType.SEMANTIC, weight=0.8)
        edge_data = mg._graph.edges["mr:a", "mr:b"]
        edge_data["last_activated"] = time.time() - 864000  # 10 天前

        mg.decay_and_prune()

        edge_after = mg.get_edge("mr:a", "mr:b")
        assert edge_after is not None
        # MUST_REMEMBER 半衰期为无穷大，不应衰减
        assert edge_after["weight"] >= 0.79


# ============================================================
# TestPersistence - 持久化与加载
# ============================================================


class TestPersistence:
    """save() / _load() 往返持久化"""

    def test_save_and_reload(self, temp_dir):
        # 创建并填充
        from zulong.memory.memory_graph import MemoryGraph as MG
        mg = MG(persist_path=temp_dir)
        mg.add_node(GraphNode(node_id="ps:a", node_type=NodeType.KNOWLEDGE, label="PersistA"))
        mg.add_node(GraphNode(node_id="ps:b", node_type=NodeType.KNOWLEDGE, label="PersistB"))
        mg.add_edge("ps:a", "ps:b", EdgeType.SEMANTIC, weight=0.7)
        mg.save()

        # 重置单例
        MG._instance = None
        if hasattr(MG, '_initialized'):
            del MG._initialized

        # 重新加载
        mg2 = MG(persist_path=temp_dir)
        assert mg2.has_node("ps:a")
        assert mg2.has_node("ps:b")
        assert mg2.has_edge("ps:a", "ps:b")

        edge = mg2.get_edge("ps:a", "ps:b")
        assert abs(edge["weight"] - 0.7) < 0.01
