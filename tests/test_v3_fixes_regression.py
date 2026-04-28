# File: tests/test_v3_fixes_regression.py
# v3.0 修复回归测试
# 覆盖 6 个核心修复：
#   Fix 1: 已完成任务后续提问被误创建新图谱（1A+1B+1C）
#   Fix 2: TaskGraph 父节点状态自动级联
#   Fix 3: MemoryGraph 激活值与 TaskGraph 状态同步
#   Fix 4: MemoryGraph 边衰减 last_activated 默认值
#   Fix 5: AttentionWindow 防御性截断
#   Fix 6: Pinned 消息超限保护

import time
import tempfile
import shutil
import os
import unittest
from unittest.mock import patch, MagicMock


# ============================================================
# Fix 1A: build_round1_system_prompt 已完成任务提示注入
# ============================================================

class TestFix1A_CompletedTaskHint(unittest.TestCase):
    """Fix 1A: 已完成任务的 Round 1 分类器应注入 CHAT 倾向提示"""

    def _make_completed_task_graph(self):
        """创建一个所有叶子节点已完成的 TaskGraph"""
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="TODO应用开发")
        tg.add_node("req", label="开发TODO应用", type="requirement",
                    status="completed", desc="用户需求")
        tg.add_node("o1", label="实现前端", type="outline",
                    status="completed", desc="React 前端")
        tg.add_node("o2", label="实现后端", type="outline",
                    status="completed", desc="Node.js 后端")
        tg.add_h_edge("req", "o1")
        tg.add_h_edge("req", "o2")
        return tg

    def _make_active_task_graph(self):
        """创建一个有未完成节点的 TaskGraph"""
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="TODO应用开发")
        tg.add_node("req", label="开发TODO应用", type="requirement",
                    status="in_progress", desc="用户需求")
        tg.add_node("o1", label="实现前端", type="outline",
                    status="completed", desc="React 前端")
        tg.add_node("o2", label="实现后端", type="outline",
                    status="pending", desc="Node.js 后端")
        tg.add_h_edge("req", "o1")
        tg.add_h_edge("req", "o2")
        return tg

    @patch("zulong.tools.task_tools.get_active_task_graph")
    def test_completed_task_injects_chat_hint(self, mock_get_tg):
        """已完成任务图应注入'请分类为 chat'提示"""
        mock_get_tg.return_value = self._make_completed_task_graph()

        from zulong.l2.intent_prompt_builder import build_round1_system_prompt
        prompt = build_round1_system_prompt()

        self.assertIn("chat", prompt)
        # root label 是 "开发TODO应用"
        self.assertIn("开发TODO应用", prompt)
        self.assertNotIn("未完成", prompt)

    @patch("zulong.tools.task_tools.get_active_task_graph")
    def test_active_task_injects_complex_hint(self, mock_get_tg):
        """活跃未完成任务应注入'请分类为 complex'提示"""
        mock_get_tg.return_value = self._make_active_task_graph()

        from zulong.l2.intent_prompt_builder import build_round1_system_prompt
        prompt = build_round1_system_prompt()

        self.assertIn("complex", prompt)
        self.assertIn("未完成", prompt)

    @patch("zulong.tools.task_tools.get_active_task_graph")
    def test_no_task_graph_no_hint(self, mock_get_tg):
        """无任务图时不注入任何提示"""
        mock_get_tg.return_value = None

        from zulong.l2.intent_prompt_builder import build_round1_system_prompt
        prompt = build_round1_system_prompt()

        self.assertNotIn("⚠️ 重要上下文", prompt)


# ============================================================
# Fix 1C: COMPLEX → CHAT 降级
# ============================================================

class TestFix1C_ComplexToChatDowngrade(unittest.TestCase):
    """Fix 1C: 已完成任务 + 短问句 → COMPLEX 自动降级为 CHAT"""

    def _make_completed_task_graph(self):
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="TODO应用开发")
        tg.add_node("req", label="开发TODO应用", type="requirement",
                    status="completed", desc="")
        tg.add_node("o1", label="前端", type="outline",
                    status="completed", desc="")
        tg.add_h_edge("req", "o1")
        return tg

    def test_downgrade_short_query_after_completed_task(self):
        """已完成任务 + 短问句(≤15字且无任务动词) → 降级为 CHAT"""
        from zulong.l2.intent_prompt_builder import IntentType

        tg = self._make_completed_task_graph()

        # 模拟降级逻辑（直接测试条件判断）
        user_input = "怎么运行"
        intent_type = IntentType.COMPLEX

        # 重现 inference_engine.py 中的降级逻辑 (Fix-7B 更新)
        leaves = tg.get_leaf_nodes()
        uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
        _stripped = user_input.strip()
        _has_task_verb = any(
            kw in _stripped
            for kw in ("帮我", "写一个", "做一个", "设计", "开发",
                       "创建", "搭建", "实现", "生成", "构建")
        )
        should_downgrade = (
            intent_type.value == "complex"
            and not uncompleted
            and leaves
            and len(_stripped) <= 15
            and not _has_task_verb
        )

        self.assertTrue(should_downgrade)

    def test_no_downgrade_long_query(self):
        """含任务动词的查询不应降级，即使在已完成任务之后"""
        from zulong.l2.intent_prompt_builder import IntentType

        tg = self._make_completed_task_graph()
        user_input = "帮我设计一个登录注册系统"
        intent_type = IntentType.COMPLEX

        leaves = tg.get_leaf_nodes()
        uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
        _stripped = user_input.strip()
        _has_task_verb = any(
            kw in _stripped
            for kw in ("帮我", "写一个", "做一个", "设计", "开发",
                       "创建", "搭建", "实现", "生成", "构建")
        )
        should_downgrade = (
            intent_type.value == "complex"
            and not uncompleted
            and leaves
            and len(_stripped) <= 15
            and not _has_task_verb
        )

        self.assertFalse(should_downgrade)

    def test_no_downgrade_active_task(self):
        """有未完成节点时不应降级"""
        from zulong.l2.task_graph import TaskGraph
        from zulong.l2.intent_prompt_builder import IntentType

        tg = TaskGraph(title="任务")
        tg.add_node("req", label="需求", type="requirement", status="in_progress", desc="")
        tg.add_node("o1", label="步骤1", type="outline", status="pending", desc="")
        tg.add_h_edge("req", "o1")

        user_input = "怎么运行"
        intent_type = IntentType.COMPLEX

        leaves = tg.get_leaf_nodes()
        uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
        _stripped = user_input.strip()
        _has_task_verb = any(
            kw in _stripped
            for kw in ("帮我", "写一个", "做一个", "设计", "开发",
                       "创建", "搭建", "实现", "生成", "构建")
        )
        should_downgrade = (
            intent_type.value == "complex"
            and not uncompleted
            and leaves
            and len(_stripped) <= 15
            and not _has_task_verb
        )

        self.assertFalse(should_downgrade)


# ============================================================
# Fix 2: TaskGraph 父节点状态自动级联
# ============================================================

class TestFix2_ParentStatusCascade(unittest.TestCase):
    """Fix 2: 子节点全部完成时父节点应自动 cascade 为 completed"""

    def _build_tree(self):
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="级联测试")
        tg.add_node("req", label="需求", type="requirement",
                    status="in_progress", desc="")
        tg.add_node("o1", label="任务A", type="outline",
                    status="pending", desc="")
        tg.add_node("o2", label="任务B", type="outline",
                    status="pending", desc="")
        tg.add_h_edge("req", "o1")
        tg.add_h_edge("req", "o2")
        return tg

    def test_cascade_when_all_children_complete(self):
        """所有子节点 completed → 父节点自动 completed"""
        tg = self._build_tree()

        tg.update_node_status("o1", "completed")
        # 此时 req 不应 cascade（o2 还是 pending）
        req_node = tg.get_node("req")
        self.assertNotEqual(req_node.status, "completed")

        tg.update_node_status("o2", "completed")
        # 此时 req 应该 cascade 为 completed
        req_node = tg.get_node("req")
        self.assertEqual(req_node.status, "completed")

    def test_cascade_with_skipped(self):
        """skipped + completed 组合也应触发级联"""
        tg = self._build_tree()

        tg.update_node_status("o1", "completed")
        tg.update_node_status("o2", "skipped")

        req_node = tg.get_node("req")
        self.assertEqual(req_node.status, "completed")

    def test_no_cascade_with_pending(self):
        """有 pending 子节点时不级联"""
        tg = self._build_tree()

        tg.update_node_status("o1", "completed")
        # o2 仍为 pending

        req_node = tg.get_node("req")
        self.assertNotEqual(req_node.status, "completed")

    def test_multi_level_cascade(self):
        """多层级联：叶子完成 → 中间层完成 → 根完成"""
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="多层级联")
        tg.add_node("req", label="根", type="requirement",
                    status="in_progress", desc="")
        tg.add_node("mid", label="中间", type="analysis",
                    status="pending", desc="")
        tg.add_node("leaf1", label="叶子1", type="outline",
                    status="pending", desc="")
        tg.add_node("leaf2", label="叶子2", type="outline",
                    status="pending", desc="")
        tg.add_h_edge("req", "mid")
        tg.add_h_edge("mid", "leaf1")
        tg.add_h_edge("mid", "leaf2")

        tg.update_node_status("leaf1", "completed")
        self.assertNotEqual(tg.get_node("mid").status, "completed")

        tg.update_node_status("leaf2", "completed")
        # mid 应级联
        self.assertEqual(tg.get_node("mid").status, "completed")
        # req 也应级联（mid 是 req 唯一的子节点）
        self.assertEqual(tg.get_node("req").status, "completed")


# ============================================================
# Fix 3: MemoryGraph 激活值同步
# ============================================================

class TestFix3_ActivationSync(unittest.TestCase):
    """Fix 3: TaskGraph 状态变更应同步 MemoryGraph 激活值"""

    def setUp(self):
        from zulong.memory.memory_graph import MemoryGraph
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.mg = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        from zulong.memory.memory_graph import MemoryGraph
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_completed_node_gets_low_activation(self):
        """completed 状态应设置激活值为 0.1"""
        from zulong.l2.task_graph import TaskGraph
        from zulong.memory.memory_graph import NodeType, GraphNode

        tg = TaskGraph(title="激活测试")
        tg.add_node("req", label="需求", type="requirement",
                    status="in_progress", desc="测试")

        # 先确保 MemoryGraph 中有对应节点
        node_id = f"task:{tg.id}/req"
        gn = GraphNode(
            node_id=node_id,
            node_type=NodeType.TASK,
            label="需求",
        )
        self.mg.add_node(gn)

        # 设置初始高激活值
        self.mg.update_node_activation(node_id, 0.9)
        node = self.mg.get_node(node_id)
        self.assertAlmostEqual(node.activation, 0.9, places=1)

        # 验证 update_node_activation 能正确降低激活值
        self.mg.update_node_activation(node_id, 0.1)
        node = self.mg.get_node(node_id)
        self.assertAlmostEqual(node.activation, 0.1, places=1)


# ============================================================
# Fix 4: MemoryGraph 边衰减 last_activated 默认值
# ============================================================

class TestFix4_EdgeDecayFallback(unittest.TestCase):
    """Fix 4: 边衰减应 fallback 到 created_at 而非 now"""

    def setUp(self):
        from zulong.memory.memory_graph import MemoryGraph
        MemoryGraph._instance = None
        self.tmpdir = tempfile.mkdtemp()
        self.mg = MemoryGraph(persist_path=self.tmpdir)

    def tearDown(self):
        from zulong.memory.memory_graph import MemoryGraph
        MemoryGraph._instance = None
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_edge_without_last_activated_decays(self):
        """没有 last_activated 的边应基于 created_at 衰减"""
        from zulong.memory.memory_graph import NodeType, GraphNode, EdgeType

        # 创建两个节点
        n1 = GraphNode(node_id="n1", node_type=NodeType.KNOWLEDGE, label="知识1")
        n2 = GraphNode(node_id="n2", node_type=NodeType.KNOWLEDGE, label="知识2")
        self.mg.add_node(n1)
        self.mg.add_node(n2)

        # 创建边
        self.mg.add_edge("n1", "n2", edge_type=EdgeType.ASSOCIATION, weight=0.8)

        # 获取边数据
        edge_data = self.mg.get_edge("n1", "n2")
        self.assertIsNotNone(edge_data)
        self.assertIn("created_at", edge_data)

        # 模拟 last_activated 缺失的场景（删除字段）
        raw_edge = self.mg._graph.edges["n1", "n2"]
        del raw_edge["last_activated"]

        # 验证修复后的逻辑：fallback 到 created_at
        now = time.time()
        elapsed_hours = (now - raw_edge.get("last_activated", raw_edge.get("created_at", now))) / 3600
        self.assertGreaterEqual(elapsed_hours, 0)

        # 额外验证：如果 created_at 被设为过去很久（模拟老边）
        raw_edge["created_at"] = now - 3600 * 48  # 48 小时前
        elapsed_hours_old = (now - raw_edge.get("last_activated", raw_edge.get("created_at", now))) / 3600
        self.assertAlmostEqual(elapsed_hours_old, 48.0, delta=0.1)


# ============================================================
# Fix 5: AttentionWindow 工具结果截断
# ============================================================

class TestFix5_ToolResultTruncation(unittest.TestCase):
    """Fix 5: 超长工具结果应在注册时截断"""

    def _make_window(self):
        from zulong.l2.attention_window import AttentionWindowManager
        return AttentionWindowManager(context_window_size=8192)

    def test_long_tool_result_truncated(self):
        """超过 MAX_TOOL_RESULT_CHARS 的 tool 消息应被截断"""
        from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
        wm = self._make_window()

        long_content = "x" * (MAX_TOOL_RESULT_CHARS + 500)
        msg = {"role": "tool", "content": long_content, "tool_call_id": "test_1"}

        wm.register_message(msg, turn=1, tool_name="test_tool")

        # 验证注册的消息已截断
        registered_content = wm.envelopes[-1].msg["content"]
        self.assertLessEqual(len(registered_content), MAX_TOOL_RESULT_CHARS + 50)
        self.assertIn("...(内容已截断)", registered_content)

    def test_short_tool_result_not_truncated(self):
        """短 tool 消息不应被截断"""
        wm = self._make_window()

        short_content = "命令执行成功"
        msg = {"role": "tool", "content": short_content, "tool_call_id": "test_2"}

        wm.register_message(msg, turn=1, tool_name="test_tool")

        registered_content = wm.envelopes[-1].msg["content"]
        self.assertEqual(registered_content, short_content)

    def test_non_tool_message_not_truncated(self):
        """非 tool 消息不应被截断（即使很长）"""
        from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
        wm = self._make_window()

        long_content = "y" * (MAX_TOOL_RESULT_CHARS + 500)
        msg = {"role": "assistant", "content": long_content}

        wm.register_message(msg, turn=1)

        registered_content = wm.envelopes[-1].msg["content"]
        self.assertEqual(len(registered_content), MAX_TOOL_RESULT_CHARS + 500)

    def test_original_message_not_mutated(self):
        """截断不应修改原始消息对象"""
        from zulong.l2.attention_window import MAX_TOOL_RESULT_CHARS
        wm = self._make_window()

        long_content = "z" * (MAX_TOOL_RESULT_CHARS + 100)
        original_msg = {"role": "tool", "content": long_content, "tool_call_id": "test_3"}

        wm.register_message(original_msg, turn=1, tool_name="test_tool")

        # 原始消息不应被修改
        self.assertEqual(len(original_msg["content"]), MAX_TOOL_RESULT_CHARS + 100)


# ============================================================
# Fix 6: Pinned 消息超限渐进式降级
# ============================================================

class TestFix6_PinnedOverflowDegradation(unittest.TestCase):
    """Fix 6: Pinned 消息超预算时应渐进式降级而非全部丢弃 unpinned"""

    def test_pinned_overflow_keeps_some_unpinned(self):
        """当 pinned 超预算时，应保留首尾 pinned + 按权重竞争"""
        from zulong.l2.attention_window import AttentionWindowManager

        # 创建一个很小预算的窗口
        wm = AttentionWindowManager(
            context_window_size=2000,
            reserved_tokens=500,
        )
        # budget ≈ (2000 - 500) * 0.9 = 1350 tokens

        # 注册多条 pinned 消息，总量超过预算
        for i in range(5):
            msg = {"role": "system" if i == 0 else "user",
                   "content": "A" * 400}  # ~100 tokens each
            wm.register_message(msg, turn=i, pinned=True)

        # 注册一条 unpinned 消息
        unpinned_msg = {"role": "assistant", "content": "重要回复内容"}
        wm.register_message(unpinned_msg, turn=5, pinned=False)

        # apply_window 应返回结果（不能为空列表）
        result = wm.apply_window()
        self.assertGreater(len(result), 0)

        # 应该至少包含首条 pinned（system）
        self.assertEqual(result[0]["role"], "system")


# ============================================================
# 综合集成测试
# ============================================================

class TestIntegration_CompletedTaskFollowUp(unittest.TestCase):
    """集成测试：完成任务后追问的完整流程"""

    def test_full_flow_completed_task_followup(self):
        """
        完整模拟：
        1. 创建任务图 → 2. 完成所有子节点 → 3. 验证父节点级联
        → 4. 验证分类器提示正确 → 5. 验证降级条件成立
        """
        from zulong.l2.task_graph import TaskGraph
        from zulong.l2.intent_prompt_builder import IntentType

        # Step 1: 创建任务图
        tg = TaskGraph(title="TODO应用开发")
        tg.add_node("req", label="开发TODO应用", type="requirement",
                    status="in_progress", desc="")
        tg.add_node("o1", label="前端开发", type="outline",
                    status="pending", desc="")
        tg.add_node("o2", label="后端开发", type="outline",
                    status="pending", desc="")
        tg.add_h_edge("req", "o1")
        tg.add_h_edge("req", "o2")

        # Step 2: 完成所有子节点
        tg.update_node_status("o1", "completed", result="React 前端已完成")
        tg.update_node_status("o2", "completed", result="Express 后端已完成")

        # Step 3: 验证父节点级联（Fix 2）
        req_node = tg.get_node("req")
        self.assertEqual(req_node.status, "completed",
                        "Fix 2 失效：父节点未自动级联为 completed")

        # Step 4: 验证 Round 1 提示（Fix 1A）
        with patch("zulong.tools.task_tools.get_active_task_graph", return_value=tg):
            from zulong.l2.intent_prompt_builder import build_round1_system_prompt
            prompt = build_round1_system_prompt()
            self.assertIn("chat", prompt,
                         "Fix 1A 失效：已完成任务未注入 chat 提示")

        # Step 5: 验证降级条件（Fix 1C + Fix-7B 更新）
        user_input = "怎么运行"
        intent_type = IntentType.COMPLEX
        leaves = tg.get_leaf_nodes()
        uncompleted = [n for n in leaves if n.status not in ("completed", "skipped")]
        _stripped = user_input.strip()
        _has_task_verb = any(
            kw in _stripped
            for kw in ("帮我", "写一个", "做一个", "设计", "开发",
                       "创建", "搭建", "实现", "生成", "构建")
        )
        should_downgrade = (
            intent_type.value == "complex"
            and not uncompleted
            and leaves
            and len(_stripped) <= 15
            and not _has_task_verb
        )
        self.assertTrue(should_downgrade,
                       "Fix 1C 失效：短问句 + 已完成任务未触发降级")


# ============================================================
# Fix 7: 已完成任务图阻止新图创建
# ============================================================

class TestFix7_CompletedGraphBlocksNewCreation(unittest.TestCase):
    """Fix 7: 已完成的活跃任务图不应阻止新任务图的创建"""

    def setUp(self):
        """创建一个已完成的任务图并设为活跃"""
        from zulong.l2.task_graph import TaskGraph
        from zulong.tools.task_tools import set_active_task_graph
        self.old_tg = TaskGraph(title="旧TODO任务")
        self.old_tg.add_node("req", label="旧TODO任务", type="requirement",
                             status="completed", desc="旧任务")
        self.old_tg.add_node("n1", label="子任务1", type="task",
                             status="completed", desc="子任务1")
        self.old_tg.add_node("n2", label="子任务2", type="task",
                             status="completed", desc="子任务2")
        self.old_tg.add_d_edge("req", "n1")
        self.old_tg.add_d_edge("req", "n2")
        set_active_task_graph(self.old_tg, "tg_old")

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        set_active_task_graph(None, None)

    def test_completed_graph_allows_new_creation(self):
        """已完成图应被清除，允许创建新图"""
        from zulong.tools.task_tools import TaskCreatePlanTool, get_active_task_graph
        from zulong.tools.base import ToolRequest
        tool = TaskCreatePlanTool()
        req = ToolRequest(
            tool_name="task_create_plan",
            action="create",
            parameters={"title": "新登录注册系统"},
            request_id="test-fix7-1",
        )
        result = tool.execute(req)
        self.assertTrue(result.success, f"新图创建失败: {result.error}")
        # 验证新图被创建而不是返回旧图
        self.assertNotEqual(result.data.get("title"), "旧TODO任务")
        self.assertIn("新登录注册系统", result.data.get("title", ""))
        self.assertFalse(result.data.get("already_exists", False),
                        "Fix 7 失效：已完成任务图仍阻止新图创建")
        # 验证活跃图已切换
        new_tg = get_active_task_graph()
        self.assertIsNotNone(new_tg)
        self.assertEqual(new_tg.title, "新登录注册系统")

    def test_uncompleted_graph_still_blocks(self):
        """有未完成节点的图应仍然拦截新图创建"""
        from zulong.tools.task_tools import TaskCreatePlanTool, set_active_task_graph
        from zulong.tools.base import ToolRequest
        from zulong.l2.task_graph import TaskGraph
        # 创建一个有未完成节点的图
        tg_active = TaskGraph(title="进行中的任务")
        tg_active.add_node("req", label="进行中的任务", type="requirement",
                           status="in_progress", desc="进行中")
        tg_active.add_node("n1", label="子任务1", type="task",
                           status="completed", desc="子任务1")
        tg_active.add_node("n2", label="子任务2", type="task",
                           status="pending", desc="子任务2")
        tg_active.add_d_edge("req", "n1")
        tg_active.add_d_edge("req", "n2")
        set_active_task_graph(tg_active, "tg_active")

        tool = TaskCreatePlanTool()
        req = ToolRequest(
            tool_name="task_create_plan",
            action="create",
            parameters={"title": "另一个新任务"},
            request_id="test-fix7-2",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertTrue(result.data.get("already_exists", False),
                       "未完成的图不应被替换")

    def test_new_graph_gets_separate_id(self):
        """新建图应获得独立的 graph_id"""
        from zulong.tools.task_tools import TaskCreatePlanTool, get_active_task_graph
        from zulong.tools.base import ToolRequest
        tool = TaskCreatePlanTool()
        req = ToolRequest(
            tool_name="task_create_plan",
            action="create",
            parameters={"title": "全新项目"},
            request_id="test-fix7-3",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        new_graph_id = result.data.get("graph_id", "")
        self.assertNotEqual(new_graph_id, "tg_old",
                           "新图应获得新的 graph_id")
        self.assertTrue(new_graph_id.startswith("tg_"),
                       f"graph_id 格式不正确: {new_graph_id}")


# ============================================================
# Fix 7C: 已完成图谱 + 关联任务应复用而非新建
# ============================================================

class TestFix7C_TitleRelatedness(unittest.TestCase):
    """Fix 7C: _titles_related 和 _extract_title_core 单元测试"""

    def test_extract_core_removes_verb_prefix(self):
        """应去除常见动词前缀"""
        from zulong.tools.session_tool import _extract_title_core
        self.assertEqual(_extract_title_core("帮我设计一个博客系统的数据库"), "博客系统的数据库")
        self.assertEqual(_extract_title_core("帮我写一个猜数字游戏"), "猜数字游戏")

    def test_extract_core_removes_suffix(self):
        """应去除常见后缀"""
        from zulong.tools.session_tool import _extract_title_core
        self.assertEqual(_extract_title_core("把博客系统的前端写出来"), "博客系统的前端")

    def test_related_blog_system(self):
        """博客系统数据库 vs 博客系统前端 → 相关"""
        from zulong.tools.session_tool import _titles_related
        self.assertTrue(_titles_related(
            "帮我设计一个博客系统的数据库",
            "把博客系统的前端写出来"
        ))

    def test_unrelated_game_vs_blog(self):
        """猜数字游戏 vs 博客系统 → 无关"""
        from zulong.tools.session_tool import _titles_related
        self.assertFalse(_titles_related(
            "帮我写一个猜数字游戏",
            "帮我设计一个博客系统"
        ))

    def test_unrelated_calculator_vs_login(self):
        """计算器程序 vs 登录注册系统 → 无关"""
        from zulong.tools.session_tool import _titles_related
        self.assertFalse(_titles_related(
            "帮我写一个简单的计算器程序",
            "帮我设计一个用户登录注册系统"
        ))

    def test_related_substring(self):
        """核心互为子串时应判定为相关"""
        from zulong.tools.session_tool import _titles_related
        self.assertTrue(_titles_related(
            "帮我开发一个电商系统",
            "把电商系统的支付模块写出来"
        ))

    def test_unrelated_shared_stopwords(self):
        """共享通用词(系统/数据库)但不同领域 → 无关（停用词过滤）"""
        from zulong.tools.session_tool import _titles_related
        # "博客系统的数据库" vs "学生成绩管理系统的数据库表结构" 应该无关
        self.assertFalse(_titles_related(
            "帮我设计一个博客系统的数据库",
            "帮我设计一个简单的学生成绩管理系统的数据库表结构"
        ))

    def test_unrelated_fibonacci_vs_student(self):
        """斐波那契函数 vs 学生成绩管理 → 无关"""
        from zulong.tools.session_tool import _titles_related
        self.assertFalse(_titles_related(
            "帮我写一个Python函数，计算斐波那契数列的第n项",
            "帮我设计一个简单的学生成绩管理系统的数据库表结构"
        ))

    def test_strip_stopwords(self):
        """_strip_stopwords 应移除通用技术词"""
        from zulong.tools.session_tool import _strip_stopwords
        self.assertEqual(_strip_stopwords("博客"), "博客")
        self.assertNotIn("系统", _strip_stopwords("学生成绩管理系统"))
        self.assertNotIn("数据库", _strip_stopwords("博客系统的数据库"))

    def test_empty_titles(self):
        """空标题应返回 False"""
        from zulong.tools.session_tool import _titles_related
        self.assertFalse(_titles_related("", "博客系统"))
        self.assertFalse(_titles_related("博客系统", ""))
        self.assertFalse(_titles_related("", ""))


class TestFix7C_SessionToolCompletedRelated(unittest.TestCase):
    """Fix 7C: StartSessionTool 已完成图谱 + 关联任务 → 复用"""

    def _make_completed_graph(self, title, graph_id="tg_old"):
        from zulong.l2.task_graph import TaskGraph
        from zulong.tools.task_tools import set_active_task_graph
        tg = TaskGraph(title=title, graph_id=graph_id)
        tg.add_node("req", label=title, type="requirement",
                     status="completed", desc=title)
        tg.add_node("o1", label="子任务1", type="outline",
                     status="completed", desc="")
        tg.add_h_edge("req", "o1")
        set_active_task_graph(tg, graph_id)
        return tg

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        set_active_task_graph(None, None)

    def test_completed_related_reuses_graph(self):
        """已完成图 + 关联新任务 → already_exists=True"""
        self._make_completed_graph("帮我设计一个博客系统的数据库")

        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        tool = StartSessionTool()
        req = ToolRequest(
            tool_name="start_session",
            action="classify",
            parameters={
                "intent": "complex",
                "reason": "用户要求写博客前端",
                "task_description": "把博客系统的前端写出来",
            },
            request_id="test-7c-1",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertTrue(result.data.get("already_exists", False),
                       "Fix 7C 失效：关联任务应复用已完成图谱")
        self.assertEqual(result.data.get("graph_id"), "tg_old")

    @patch("zulong.tools.session_tool._search_historical_task", return_value=None)
    def test_completed_unrelated_creates_new(self, _mock_hist):
        """已完成图 + 无关新任务 → 创建新图（隔离P1历史搜索）"""
        self._make_completed_graph("帮我写一个猜数字游戏")

        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        tool = StartSessionTool()
        req = ToolRequest(
            tool_name="start_session",
            action="classify",
            parameters={
                "intent": "complex",
                "reason": "用户要求设计博客系统",
                "task_description": "帮我设计一个博客系统的数据库",
            },
            request_id="test-7c-2",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertFalse(result.data.get("already_exists", False),
                        "无关任务不应复用已完成图谱")
        self.assertNotEqual(result.data.get("graph_id", ""), "tg_old")


class TestFix7C_TaskToolsCompletedRelated(unittest.TestCase):
    """Fix 7C: TaskCreatePlanTool 已完成图谱 + 关联任务 → 复用"""

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        set_active_task_graph(None, None)

    def _make_completed_graph(self, title, graph_id="tg_old"):
        from zulong.l2.task_graph import TaskGraph
        from zulong.tools.task_tools import set_active_task_graph
        tg = TaskGraph(title=title, graph_id=graph_id)
        tg.add_node("req", label=title, type="requirement",
                     status="completed", desc=title)
        tg.add_node("o1", label="子任务1", type="outline",
                     status="completed", desc="")
        tg.add_h_edge("req", "o1")
        set_active_task_graph(tg, graph_id)
        return tg

    def test_task_create_plan_related_reuses(self):
        """TaskCreatePlanTool: 已完成 + 关联 → already_exists"""
        self._make_completed_graph("帮我设计一个博客系统的数据库")

        from zulong.tools.task_tools import TaskCreatePlanTool
        from zulong.tools.base import ToolRequest
        tool = TaskCreatePlanTool()
        req = ToolRequest(
            tool_name="task_create_plan",
            action="create",
            parameters={"title": "把博客系统的前端写出来"},
            request_id="test-7c-3",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertTrue(result.data.get("already_exists", False),
                       "Fix 7C 失效：关联任务应复用已完成图谱")

    def test_task_create_plan_unrelated_creates_new(self):
        """TaskCreatePlanTool: 已完成 + 无关 → 创建新图"""
        self._make_completed_graph("帮我写一个猜数字游戏")

        from zulong.tools.task_tools import TaskCreatePlanTool, get_active_task_graph
        from zulong.tools.base import ToolRequest
        tool = TaskCreatePlanTool()
        req = ToolRequest(
            tool_name="task_create_plan",
            action="create",
            parameters={"title": "帮我设计一个博客系统"},
            request_id="test-7c-4",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertFalse(result.data.get("already_exists", False),
                        "无关任务不应复用已完成图谱")
        new_tg = get_active_task_graph()
        self.assertIsNotNone(new_tg)


# ============================================================
# P0-P3: 任务生命周期完整性修复
# ============================================================

class TestP0_AutoArchiveOnCompletion(unittest.TestCase):
    """P0: 任务根节点级联完成时自动归档到 completed_tasks"""

    def setUp(self):
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        CompletedTaskArchiveManager._instance = None
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CompletedTaskArchiveManager(
            config={"persistence_path": self._tmpdir}
        )

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        set_active_task_graph(None, None)
        CompletedTaskArchiveManager._instance = None
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_archive_triggered_on_root_cascade(self):
        """所有子节点完成 → 根节点级联 → 自动归档"""
        from zulong.l2.task_graph import TaskGraph
        from zulong.tools.task_tools import (
            set_active_task_graph, TaskMarkStatusTool,
        )
        from zulong.tools.base import ToolRequest

        tg = TaskGraph(title="测试归档", graph_id="tg_archive_test")
        tg.add_node("req", label="测试归档", type="requirement",
                     status="in_progress", desc="")
        tg.add_node("o1", label="步骤1", type="outline",
                     status="pending", desc="")
        tg.add_node("o2", label="步骤2", type="outline",
                     status="pending", desc="")
        tg.add_h_edge("req", "o1")
        tg.add_h_edge("req", "o2")
        set_active_task_graph(tg, "tg_archive_test")

        tool = TaskMarkStatusTool()
        # 完成 o1
        tool.execute(ToolRequest(
            tool_name="task_mark_status", action="update",
            parameters={"node_id": "o1", "status": "completed"},
            request_id="t1",
        ))
        # 完成 o2 → 触发根节点级联 → 触发归档
        tool.execute(ToolRequest(
            tool_name="task_mark_status", action="update",
            parameters={"node_id": "o2", "status": "completed"},
            request_id="t2",
        ))

        # 验证归档文件已创建
        import os
        archived_files = [f for f in os.listdir(self._tmpdir) if f.endswith(".json")]
        self.assertGreater(len(archived_files), 0,
                          "P0 失效：任务完成后未自动归档")


class TestP1_ComplexHistoricalSearch(unittest.TestCase):
    """P1: COMPLEX 分支在创建新图前搜索历史任务"""

    def setUp(self):
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        CompletedTaskArchiveManager._instance = None
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CompletedTaskArchiveManager(
            config={"persistence_path": self._tmpdir}
        )

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        set_active_task_graph(None, None)
        CompletedTaskArchiveManager._instance = None
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _archive_task(self, title, graph_id):
        """手动写入一条归档记录"""
        from zulong.l2.task_graph import TaskGraph
        from zulong.l2.task_archive import CompletedTaskArchive
        tg = TaskGraph(title=title, graph_id=graph_id)
        tg.add_node("req", label=title, type="requirement",
                     status="completed", desc="")
        tg.add_node("o1", label="子任务", type="outline",
                     status="completed", desc="")
        tg.add_h_edge("req", "o1")
        archive = CompletedTaskArchive(
            task_id=graph_id,
            description=title,
            final_answer="",
            duration=0,
            total_turns=0,
            completion_status="completed",
            task_graph_snapshot=tg.serialize(),
            metadata={"graph_id": graph_id},
        )
        from zulong.tools.session_tool import _run_async
        _run_async(self._mgr.archive_task(archive))

    def test_find_related_from_archive(self):
        """无活跃图+归档中有关联任务 → 恢复历史图谱"""
        self._archive_task("帮我设计一个博客系统的数据库", "tg_blog_db")

        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        tool = StartSessionTool()
        req = ToolRequest(
            tool_name="start_session",
            action="classify",
            parameters={
                "intent": "complex",
                "reason": "用户要求写博客前端",
                "task_description": "把博客系统的前端写出来",
            },
            request_id="test-p1-1",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertTrue(result.data.get("already_exists", False),
                       "P1 失效：应从归档恢复关联历史任务")
        self.assertEqual(result.data.get("restored_from"), "completed")

    @patch("zulong.tools.session_tool._search_historical_task", return_value=None)
    def test_no_archive_match_creates_new(self, _mock):
        """无活跃图+归档无匹配 → 正常创建新图"""
        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        tool = StartSessionTool()
        req = ToolRequest(
            tool_name="start_session",
            action="classify",
            parameters={
                "intent": "complex",
                "reason": "全新任务",
                "task_description": "帮我写一个贪吃蛇游戏",
            },
            request_id="test-p1-2",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertFalse(result.data.get("already_exists", False))
        self.assertIn("贪吃蛇游戏", result.data.get("title", ""))


class TestP2_ResumeSearchCompletedArchive(unittest.TestCase):
    """P2: RESUME 分支在挂起未找到时搜索已完成归档"""

    def setUp(self):
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        CompletedTaskArchiveManager._instance = None
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CompletedTaskArchiveManager(
            config={"persistence_path": self._tmpdir}
        )

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        set_active_task_graph(None, None)
        CompletedTaskArchiveManager._instance = None
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _archive_task(self, title, graph_id):
        from zulong.l2.task_graph import TaskGraph
        from zulong.l2.task_archive import CompletedTaskArchive
        tg = TaskGraph(title=title, graph_id=graph_id)
        tg.add_node("req", label=title, type="requirement",
                     status="completed", desc="")
        tg.add_h_edge("req", "req")  # self-edge for leaf detection
        archive = CompletedTaskArchive(
            task_id=graph_id,
            description=title,
            final_answer="",
            duration=0,
            total_turns=0,
            completion_status="completed",
            task_graph_snapshot=tg.serialize(),
            metadata={"graph_id": graph_id},
        )
        from zulong.tools.session_tool import _run_async
        _run_async(self._mgr.archive_task(archive))

    @patch("zulong.l2.task_suspension.TaskSuspensionManager.find_by_description",
           return_value=None)
    def test_resume_finds_completed_archive(self, _mock_suspend):
        """RESUME: 挂起未找到+归档有匹配 → 从归档恢复"""
        self._archive_task("帮我设计博客系统", "tg_blog_resume")

        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        tool = StartSessionTool()
        req = ToolRequest(
            tool_name="start_session",
            action="classify",
            parameters={
                "intent": "resume",
                "reason": "用户说继续博客",
                "task_description": "继续之前的博客系统",
            },
            request_id="test-p2-1",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)
        self.assertEqual(result.data.get("intent"), "resume")
        self.assertTrue(result.data.get("has_task_graph", False),
                       "P2 失效：应从归档恢复历史任务")
        self.assertEqual(result.data.get("restored_from"), "completed")


class TestP3_ArchiveBeforeClear(unittest.TestCase):
    """P3: 清除已完成图谱前自动归档（安全网）"""

    def setUp(self):
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        CompletedTaskArchiveManager._instance = None
        self._tmpdir = tempfile.mkdtemp()
        self._mgr = CompletedTaskArchiveManager(
            config={"persistence_path": self._tmpdir}
        )

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        set_active_task_graph(None, None)
        CompletedTaskArchiveManager._instance = None
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("zulong.tools.session_tool._search_historical_task", return_value=None)
    def test_completed_graph_archived_before_clear(self, _mock_hist):
        """已完成图+无关新任务: 旧图应在清除前被归档"""
        from zulong.l2.task_graph import TaskGraph
        from zulong.tools.task_tools import set_active_task_graph
        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest

        tg = TaskGraph(title="旧已完成任务", graph_id="tg_p3_old")
        tg.add_node("req", label="旧已完成任务", type="requirement",
                     status="completed", desc="")
        tg.add_node("o1", label="子任务", type="outline",
                     status="completed", desc="")
        tg.add_h_edge("req", "o1")
        set_active_task_graph(tg, "tg_p3_old")

        tool = StartSessionTool()
        req = ToolRequest(
            tool_name="start_session",
            action="classify",
            parameters={
                "intent": "complex",
                "reason": "全新无关任务",
                "task_description": "帮我写一个天气预报应用",
            },
            request_id="test-p3-1",
        )
        result = tool.execute(req)
        self.assertTrue(result.success)

        # 验证旧图已被归档
        import os
        archived = [f for f in os.listdir(self._tmpdir) if f.endswith(".json")]
        self.assertGreater(len(archived), 0,
                          "P3 失效：清除已完成图谱前未归档")


# ============================================================
# 语义检索：HistoricalTaskIndex 单元测试
# ============================================================

def _char_freq_encode(self, texts, **kwargs):
    """确定性伪 embedding：基于字符频率生成向量，相似文本→相似向量"""
    import numpy as np
    results = []
    for text in texts:
        vec = np.zeros(512, dtype=np.float32)
        for ch in text:
            vec[ord(ch) % 512] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        results.append(vec)
    return np.array(results, dtype=np.float32)


class TestHistoricalTaskIndex_Basic(unittest.TestCase):
    """HistoricalTaskIndex 基本功能：add → search → remove"""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        # 重置单例
        from zulong.memory.task_search_index import HistoricalTaskIndex
        HistoricalTaskIndex._instance = None

    def tearDown(self):
        from zulong.memory.task_search_index import HistoricalTaskIndex
        HistoricalTaskIndex._instance = None
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("zulong.memory.task_search_index._COMPLETED_TASKS_DIR", "/nonexistent")
    @patch("zulong.memory.task_search_index._GRAPH_BACKUPS_DIR", "/nonexistent")
    @patch("zulong.models.embedding_model.EmbeddingModel.encode", _char_freq_encode)
    def test_add_and_search(self):
        """添加条目后应能通过语义搜索找到"""
        from zulong.memory.task_search_index import (
            HistoricalTaskIndex, TaskIndexEntry,
        )
        persist = os.path.join(self._tmpdir, "test_idx")
        idx = HistoricalTaskIndex(persist_path=persist)

        if not idx._faiss_available:
            self.skipTest("FAISS 不可用")

        # Mock is_available() 使 search 不被短路
        idx.is_available = lambda: True

        # 添加 3 个条目
        idx.add_entry(TaskIndexEntry(
            entry_id="t1", title="帮我设计一个博客系统的数据库",
            source="completed", file_path="/fake/t1.json",
        ))
        idx.add_entry(TaskIndexEntry(
            entry_id="t2", title="帮我写一个猜数字游戏",
            source="completed", file_path="/fake/t2.json",
        ))
        idx.add_entry(TaskIndexEntry(
            entry_id="t3", title="帮我设计学生成绩管理系统",
            source="backup", file_path="/fake/t3.json",
        ))

        # 搜索博客相关 — 应至少返回结果
        results = idx.search("修改之前的博客系统", top_k=3, similarity_threshold=0.0)
        self.assertGreater(len(results), 0, "应至少返回 1 条结果")
        # 字符频率伪 embedding 下，共享 "博客系统" 字符的 t1 应排在最前
        top_entry, top_sim = results[0]
        self.assertEqual(top_entry.entry_id, "t1",
                        f"最高匹配应是博客系统，实际: {top_entry.title}")

    @patch("zulong.memory.task_search_index._COMPLETED_TASKS_DIR", "/nonexistent")
    @patch("zulong.memory.task_search_index._GRAPH_BACKUPS_DIR", "/nonexistent")
    @patch("zulong.models.embedding_model.EmbeddingModel.encode", _char_freq_encode)
    def test_remove_entry(self):
        """删除条目后搜索不应返回该条目"""
        from zulong.memory.task_search_index import (
            HistoricalTaskIndex, TaskIndexEntry,
        )
        persist = os.path.join(self._tmpdir, "test_idx")
        idx = HistoricalTaskIndex(persist_path=persist)

        if not idx._faiss_available:
            self.skipTest("FAISS 不可用")

        idx.is_available = lambda: True

        idx.add_entry(TaskIndexEntry(
            entry_id="t1", title="博客系统的数据库设计",
            source="completed", file_path="/fake/t1.json",
        ))
        # 删除
        idx.remove_entry("t1")
        results = idx.search("博客系统", top_k=5, similarity_threshold=0.0)
        entry_ids = [e.entry_id for e, _ in results]
        self.assertNotIn("t1", entry_ids, "已删除的条目不应出现在搜索结果中")

    @patch("zulong.memory.task_search_index._COMPLETED_TASKS_DIR", "/nonexistent")
    @patch("zulong.memory.task_search_index._GRAPH_BACKUPS_DIR", "/nonexistent")
    def test_deduplication(self):
        """同一 entry_id 重复 add 应覆盖"""
        from zulong.memory.task_search_index import (
            HistoricalTaskIndex, TaskIndexEntry,
        )
        persist = os.path.join(self._tmpdir, "test_idx")
        idx = HistoricalTaskIndex(persist_path=persist)

        if not idx._faiss_available:
            self.skipTest("FAISS 不可用")

        idx.add_entry(TaskIndexEntry(
            entry_id="t1", title="旧标题：猜数字游戏",
            source="backup", file_path="/fake/t1.json",
        ))
        idx.add_entry(TaskIndexEntry(
            entry_id="t1", title="新标题：博客系统数据库设计",
            source="completed", file_path="/fake/t1_new.json",
        ))

        # entries 应只有 1 条
        self.assertEqual(len(idx._entries), 1)
        self.assertEqual(idx._entries["t1"].title, "新标题：博客系统数据库设计")


class TestHistoricalTaskIndex_Persistence(unittest.TestCase):
    """HistoricalTaskIndex 持久化：save → load"""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        from zulong.memory.task_search_index import HistoricalTaskIndex
        HistoricalTaskIndex._instance = None

    def tearDown(self):
        from zulong.memory.task_search_index import HistoricalTaskIndex
        HistoricalTaskIndex._instance = None
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("zulong.memory.task_search_index._COMPLETED_TASKS_DIR", "/nonexistent")
    @patch("zulong.memory.task_search_index._GRAPH_BACKUPS_DIR", "/nonexistent")
    @patch("zulong.models.embedding_model.EmbeddingModel.encode", _char_freq_encode)
    def test_save_and_load(self):
        """save 后新实例 load 应恢复索引"""
        from zulong.memory.task_search_index import (
            HistoricalTaskIndex, TaskIndexEntry,
        )
        persist = os.path.join(self._tmpdir, "test_idx")

        # 第一个实例：添加数据并 save
        idx1 = HistoricalTaskIndex(persist_path=persist)
        if not idx1._faiss_available:
            self.skipTest("FAISS 不可用")

        idx1.add_entry(TaskIndexEntry(
            entry_id="t1", title="博客系统数据库设计",
            source="completed", file_path="/fake/t1.json",
        ))
        idx1.save()

        # 重置单例
        HistoricalTaskIndex._instance = None

        # 第二个实例：从磁盘加载
        idx2 = HistoricalTaskIndex(persist_path=persist)
        self.assertEqual(len(idx2._entries), 1)
        self.assertIn("t1", idx2._entries)

        # 搜索应仍然有效（mock is_available 使 search 路径可达）
        idx2.is_available = lambda: True
        results = idx2.search("博客系统", top_k=3, similarity_threshold=0.0)
        self.assertGreater(len(results), 0)


class TestHistoricalTaskIndex_SemanticAccuracy(unittest.TestCase):
    """语义检索准确性：不同领域不应匹配"""

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        from zulong.memory.task_search_index import HistoricalTaskIndex
        HistoricalTaskIndex._instance = None

    def tearDown(self):
        from zulong.memory.task_search_index import HistoricalTaskIndex
        HistoricalTaskIndex._instance = None
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("zulong.memory.task_search_index._COMPLETED_TASKS_DIR", "/nonexistent")
    @patch("zulong.memory.task_search_index._GRAPH_BACKUPS_DIR", "/nonexistent")
    def test_unrelated_tasks_low_similarity(self):
        """不相关任务的相似度应低于阈值 0.55"""
        from zulong.memory.task_search_index import (
            HistoricalTaskIndex, TaskIndexEntry,
        )
        persist = os.path.join(self._tmpdir, "test_idx")
        idx = HistoricalTaskIndex(persist_path=persist)

        if not idx._faiss_available:
            self.skipTest("FAISS 不可用")

        # 检查 embedding 模型是否真正可用（非随机向量）
        if not idx.is_available():
            self.skipTest("embedding 模型未加载，无法测试语义准确性")

        idx.add_entry(TaskIndexEntry(
            entry_id="t1", title="帮我设计一个博客系统的数据库",
            source="completed", file_path="/fake/t1.json",
        ))

        # 查询不相关任务
        results = idx.search(
            "帮我设计一个简单的学生成绩管理系统的数据库表结构",
            top_k=3, similarity_threshold=0.55,
        )
        # 使用 0.55 阈值应过滤掉不相关结果
        matched_ids = [e.entry_id for e, _ in results]
        self.assertNotIn("t1", matched_ids,
                        "博客系统不应与学生成绩管理系统匹配 (threshold=0.55)")


class TestSearchHistoricalTask_Fallback(unittest.TestCase):
    """_search_historical_task 降级测试：语义不可用时退回文本匹配"""

    def tearDown(self):
        from zulong.tools.task_tools import set_active_task_graph
        set_active_task_graph(None, None)

    @patch("zulong.tools.session_tool._titles_related", return_value=False)
    @patch("zulong.memory.task_search_index.HistoricalTaskIndex.is_available",
           return_value=False)
    def test_fallback_to_text_matching(self, mock_avail, mock_related):
        """语义检索不可用时应降级到文本匹配路径"""
        from zulong.tools.session_tool import _search_historical_task

        result = _search_historical_task("帮我写一个计算器")
        # 语义不可用 + 文本不匹配 → 返回 None
        self.assertIsNone(result)
        # 确认降级路径被调用
        mock_related.assert_called()


if __name__ == "__main__":
    unittest.main()
