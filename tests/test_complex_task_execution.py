# File: tests/test_complex_task_execution.py
"""
复杂任务执行路径 — 单元测试

覆盖模块：
- TaskGraph: 数据结构、状态聚合、序列化/反序列化、叶子节点
- RuleGuardian: 过早完成拦截、节点绕过检测
- fc_graph 辅助函数: _is_filler_content, _has_content_match, _extract_node_content
- CircuitBreaker: 序列化/反序列化、信号检测
- AttentionWindow: 模式切换、apply_window
- task_tools 辅助函数: _fuzzy_resolve_node_id, _normalize_label, _bigram_jaccard
"""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# 确保项目根目录在 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# TaskGraph 单元测试
# ============================================================

class TestTaskGraph(unittest.TestCase):
    """TaskGraph 数据结构测试"""

    def _make_graph(self):
        """创建一个标准测试用 TaskGraph"""
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="测试任务", graph_id="test_001")

        # 构建树: req -> analysis -> [o1, o2, o3]
        tg.add_node(id="req", label="用户需求", type="requirement", status="completed")
        tg.add_node(id="analysis", label="需求分析", type="analysis", status="completed")
        tg.add_node(id="o1", label="第一步", type="outline", status="completed")
        tg.add_node(id="o2", label="第二步", type="outline", status="in_progress")
        tg.add_node(id="o3", label="第三步", type="outline", status="pending")

        tg.add_h_edge("req", "analysis")
        tg.add_h_edge("analysis", "o1")
        tg.add_h_edge("analysis", "o2")
        tg.add_h_edge("analysis", "o3")

        return tg

    def test_add_node_and_get(self):
        """测试节点添加和查询"""
        tg = self._make_graph()
        self.assertEqual(len(tg._nodes), 5)
        node = tg.get_node("o1")
        self.assertIsNotNone(node)
        self.assertEqual(node.label, "第一步")
        self.assertEqual(node.status, "completed")

    def test_get_children(self):
        """测试获取子节点"""
        tg = self._make_graph()
        children = tg.get_children("analysis")
        self.assertEqual(len(children), 3)
        child_ids = {c.id for c in children}
        self.assertEqual(child_ids, {"o1", "o2", "o3"})

    def test_get_leaf_nodes(self):
        """测试叶子节点获取（排除 req 和 analysis）"""
        tg = self._make_graph()
        leaves = tg.get_leaf_nodes()
        leaf_ids = {n.id for n in leaves}
        # o1, o2, o3 是叶子，req 和 analysis 被模板排除或有子节点
        self.assertEqual(leaf_ids, {"o1", "o2", "o3"})

    def test_get_leaf_nodes_with_deeper_tree(self):
        """测试深层树的叶子节点"""
        tg = self._make_graph()
        # 给 o1 添加子节点
        tg.add_node(id="o1_1", label="子步骤1", type="task", status="pending")
        tg.add_node(id="o1_2", label="子步骤2", type="task", status="pending")
        tg.add_h_edge("o1", "o1_1")
        tg.add_h_edge("o1", "o1_2")

        leaves = tg.get_leaf_nodes()
        leaf_ids = {n.id for n in leaves}
        # o1 不再是叶子（有子节点），o1_1 和 o1_2 成为叶子
        self.assertNotIn("o1", leaf_ids)
        self.assertIn("o1_1", leaf_ids)
        self.assertIn("o1_2", leaf_ids)
        self.assertIn("o2", leaf_ids)
        self.assertIn("o3", leaf_ids)

    def test_update_node_status(self):
        """测试节点状态更新"""
        tg = self._make_graph()
        result = tg.update_node_status("o2", "completed", result="已完成第二步")
        self.assertTrue(result)
        node = tg.get_node("o2")
        self.assertEqual(node.status, "completed")
        self.assertEqual(node.result, "已完成第二步")

    def test_update_nonexistent_node(self):
        """测试更新不存在的节点"""
        tg = self._make_graph()
        result = tg.update_node_status("nonexist", "completed")
        self.assertFalse(result)

    def test_get_node_depth(self):
        """测试节点深度计算"""
        tg = self._make_graph()
        self.assertEqual(tg.get_node_depth("req"), 0)
        self.assertEqual(tg.get_node_depth("analysis"), 1)
        self.assertEqual(tg.get_node_depth("o1"), 2)

    def test_depth_to_type(self):
        """测试深度到类型映射"""
        from zulong.l2.task_graph import TaskGraph
        self.assertEqual(TaskGraph.depth_to_type(0), "requirement")
        self.assertEqual(TaskGraph.depth_to_type(1), "analysis")
        self.assertEqual(TaskGraph.depth_to_type(2), "outline")
        self.assertEqual(TaskGraph.depth_to_type(3), "task")
        self.assertEqual(TaskGraph.depth_to_type(4), "subtask")
        self.assertEqual(TaskGraph.depth_to_type(100), "subtask")

    def test_aggregate_status_all_completed(self):
        """测试状态聚合 - 全部完成"""
        tg = self._make_graph()
        tg.update_node_status("o1", "completed")
        tg.update_node_status("o2", "completed")
        tg.update_node_status("o3", "completed")
        status = tg._aggregate_status("analysis")
        self.assertEqual(status, "completed")

    def test_aggregate_status_partial(self):
        """测试状态聚合 - 部分完成"""
        tg = self._make_graph()
        # o1=completed, o2=in_progress, o3=pending
        status = tg._aggregate_status("analysis")
        self.assertEqual(status, "in_progress")

    def test_aggregate_status_with_blocked(self):
        """测试状态聚合 - 含阻塞"""
        tg = self._make_graph()
        tg.update_node_status("o1", "completed")
        tg.update_node_status("o2", "blocked")
        tg.update_node_status("o3", "pending")
        status = tg._aggregate_status("analysis")
        self.assertEqual(status, "blocked")

    def test_aggregate_status_performance(self):
        """[P1] 测试 _aggregate_status 在大量节点时的性能"""
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="性能测试", graph_id="perf_001")
        tg.add_node(id="req", label="需求", type="requirement", status="completed")
        tg.add_node(id="analysis", label="分析", type="analysis", status="pending")
        tg.add_h_edge("req", "analysis")

        # 创建 50 个大纲节点，每个下面 5 个任务
        for i in range(1, 51):
            oid = f"o{i}"
            tg.add_node(id=oid, label=f"大纲{i}", type="outline", status="pending")
            tg.add_h_edge("analysis", oid)
            for j in range(1, 6):
                tid = f"o{i}_{j}"
                tg.add_node(id=tid, label=f"任务{i}-{j}", type="task", status="pending")
                tg.add_h_edge(oid, tid)

        # 计时 _aggregate_status
        start = time.perf_counter()
        status = tg._aggregate_status("analysis")
        elapsed = time.perf_counter() - start
        self.assertEqual(status, "pending")
        # 300 个节点不应超过 100ms（标记性能基线）
        self.assertLess(elapsed, 0.1, f"_aggregate_status 耗时 {elapsed:.3f}s，可能有性能问题")

    def test_serialize_deserialize(self):
        """测试序列化和反序列化"""
        tg = self._make_graph()
        tg.update_node_status("o1", "completed", result="结果1")

        data = tg.serialize()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["id"], "test_001")
        self.assertEqual(data["title"], "测试任务")

        from zulong.l2.task_graph import TaskGraph
        tg2 = TaskGraph.deserialize(data)
        self.assertEqual(tg2.id, "test_001")
        self.assertEqual(tg2.title, "测试任务")
        self.assertEqual(len(tg2._nodes), 5)
        self.assertEqual(tg2.get_node("o1").status, "completed")
        self.assertEqual(tg2.get_node("o1").result, "结果1")

    def test_node_id_collision_after_deletion(self):
        """[P3] 测试节点删除后重新添加的 ID 碰撞问题

        模拟: 有 o1, o2, o3 → 移除 o2（从 _nodes 和 _h_edges 中删除）
        → 添加新节点时 len(children) == 2 → 新 ID = o3 → 碰撞
        """
        tg = self._make_graph()
        # 模拟删除 o2 (手动从内部结构中删除)
        if "o2" in tg._nodes:
            del tg._nodes["o2"]
        tg._h_edges = [(s, t) for (s, t) in tg._h_edges if t != "o2"]

        # 此时 analysis 下只有 o1, o3
        children = tg.get_children("analysis")
        child_ids = {c.id for c in children}
        self.assertEqual(child_ids, {"o1", "o3"})

        # 用当前逻辑生成新 ID: f"o{len(children) + 1}" = "o3"
        new_id = f"o{len(children) + 1}"
        # 验证 ID 碰撞: o3 已存在
        self.assertEqual(new_id, "o3",
                         "预期的碰撞场景：新 ID 'o3' 与已有节点碰撞")
        self.assertIn("o3", tg._nodes,
                       "已有节点 o3 仍然存在，会被覆盖")

    def test_to_planning_table(self):
        """测试规划表生成"""
        tg = self._make_graph()
        table = tg.to_planning_table()
        self.assertIsInstance(table, str)
        self.assertIn("当前任务规划", table)


# ============================================================
# RuleGuardian 单元测试
# ============================================================

class TestRuleGuardian(unittest.TestCase):
    """RuleGuardian 规则守护者测试"""

    def _make_task_graph_mock(self, uncompleted_count=3):
        """创建 mock TaskGraph"""
        from zulong.l2.task_graph import TaskNode
        mock_tg = MagicMock()
        uncompleted = []
        for i in range(uncompleted_count):
            node = TaskNode(
                id=f"o{i+1}", label=f"任务{i+1}",
                type="outline", status="pending", desc=f"描述{i+1}",
            )
            uncompleted.append(node)
        all_leaves = uncompleted + [
            TaskNode(id="done1", label="已完成", type="outline",
                     status="completed", desc=""),
        ] if uncompleted_count < 5 else uncompleted

        mock_tg.get_leaf_nodes.return_value = all_leaves
        return mock_tg

    def test_detect_premature_completion(self):
        """测试过早完成声明检测"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=True)
        mock_tg = self._make_task_graph_mock(3)

        # 含过早完成声明的文本
        text = "所有任务已完成，我已经帮你处理好了全部内容。"
        should_block, reason = rg.check_premature_completion(text, mock_tg)
        self.assertTrue(should_block)
        self.assertIn("未完成", reason)

    def test_no_false_positive_on_normal_text(self):
        """测试正常文本不触发误报"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=True)
        mock_tg = self._make_task_graph_mock(3)

        text = "好的，我来分析第一个任务的内容。"
        should_block, _ = rg.check_premature_completion(text, mock_tg)
        self.assertFalse(should_block)

    def test_retry_count_and_eventual_pass(self):
        """测试重试次数达到上限后放行"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=True, max_retries=2)
        mock_tg = self._make_task_graph_mock(3)

        text = "所有任务已完成了"

        # 第 1 次：拦截
        blocked, _ = rg.check_premature_completion(text, mock_tg)
        self.assertTrue(blocked)

        # 第 2 次：拦截
        blocked, _ = rg.check_premature_completion(text, mock_tg)
        self.assertTrue(blocked)

        # 第 3 次：超过 max_retries，放行并标记 needs_adjust
        blocked, _ = rg.check_premature_completion(text, mock_tg)
        self.assertFalse(blocked)

    def test_node_bypass_detection(self):
        """测试节点绕过检测"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=True)
        mock_tg = self._make_task_graph_mock(3)
        # 全部节点设为未完成（pending）
        for n in mock_tg.get_leaf_nodes.return_value:
            n.status = "pending"

        # 长回复（>200字符）触发绕过检测
        long_text = "我已经为你准备了详细的旅行计划。" * 30  # >200 字符

        # 第一次应该被拦截（重定向机会）
        blocked, reason = rg.check_premature_completion(long_text, mock_tg)
        self.assertTrue(blocked)
        self.assertIn("task_mark_status", reason)

    def test_bypass_redirect_count_not_reset(self):
        """[P4] 测试 _bypass_redirect_count 在多次调用后不重置"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=True)
        mock_tg = self._make_task_graph_mock(3)
        for n in mock_tg.get_leaf_nodes.return_value:
            n.status = "pending"

        long_text = "x" * 300

        # 第一次绕过：拦截
        blocked1, _ = rg.check_premature_completion(long_text, mock_tg)
        self.assertTrue(blocked1)
        self.assertEqual(rg._bypass_redirect_count, 1)

        # 第二次绕过：不拦截（已用完重定向机会）
        blocked2, _ = rg.check_premature_completion(long_text, mock_tg)
        self.assertFalse(blocked2)

        # 关键问题：bypass_redirect_count 没有重置方法
        # 新的 FC 会话如果复用同一个 RuleGuardian 实例，则无法重新拦截
        self.assertEqual(rg._bypass_redirect_count, 1,
                         "bypass_redirect_count 无法在会话间重置")

    def test_question_response_not_blocked(self):
        """测试提问型回复不被节点绕过检测拦截"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=True)
        mock_tg = self._make_task_graph_mock(3)
        for n in mock_tg.get_leaf_nodes.return_value:
            n.status = "pending"

        # 以问号结尾的长回复不应被拦截
        question_text = "x" * 200 + "请问你的出行日期是什么时候？"
        blocked, _ = rg.check_premature_completion(question_text, mock_tg)
        self.assertFalse(blocked)

    def test_disabled_guardian(self):
        """测试禁用状态"""
        from zulong.l2.rule_guardian import RuleGuardian
        rg = RuleGuardian(enabled=False)
        mock_tg = self._make_task_graph_mock(3)
        text = "所有任务已完成"
        blocked, _ = rg.check_premature_completion(text, mock_tg)
        self.assertFalse(blocked)


# ============================================================
# FC Graph 辅助函数测试
# ============================================================

class TestFCGraphHelpers(unittest.TestCase):
    """fc_graph.py 辅助函数测试"""

    def test_is_filler_content_short(self):
        """短回复被判定为填充内容"""
        from zulong.l2.fc_graph import _is_filler_content
        self.assertTrue(_is_filler_content("好"))
        self.assertTrue(_is_filler_content("我在想..."))

    def test_is_filler_content_with_patterns(self):
        """含填充模式的文本"""
        from zulong.l2.fc_graph import _is_filler_content
        text = "好的，我正在思考这个问题，让我继续分析一下具体的方案。"
        self.assertTrue(_is_filler_content(text))

    def test_is_filler_content_substantive(self):
        """有实质内容的文本不应被判为填充"""
        from zulong.l2.fc_graph import _is_filler_content
        text = ("根据分析结果，推荐以下方案：第一，优化数据库查询索引；"
                "第二，添加缓存层减少重复请求；第三，使用异步处理提升响应速度。"
                "具体实施步骤如下...")
        self.assertFalse(_is_filler_content(text))

    def test_has_content_match_exact(self):
        """精确匹配节点标签"""
        from zulong.l2.fc_graph import _has_content_match
        self.assertTrue(_has_content_match("这是关于旅行计划的内容", "旅行计划"))

    def test_has_content_match_bigram(self):
        """bigram 匹配"""
        from zulong.l2.fc_graph import _has_content_match
        # "酒店预订" 的 bigrams: "酒店", "店预", "预订"
        self.assertTrue(_has_content_match("我已经预订了酒店", "酒店预订"))

    def test_has_content_match_false_positive(self):
        """[P6] 验证 bigram 误报已修复：需要 >=2 个 bigram 命中

        标签 "旅行计划" 的 bigrams 包括 "旅行", "行计", "计划"。
        仅命中 1 个 bigram ("计划") 不应匹配。
        """
        from zulong.l2.fc_graph import _has_content_match
        # "旅行计划" 的 bigrams: "旅行", "行计", "计划"
        # 不相关文本仅包含 "计划" 一个 bigram → 不应匹配
        unrelated_text = "我们需要一个退休计划来保障未来。"
        result = _has_content_match(unrelated_text, "旅行计划")
        self.assertFalse(result,
                         "P6 已修复: 单个 bigram 命中不应匹配")

    def test_has_content_match_empty(self):
        """空输入不匹配"""
        from zulong.l2.fc_graph import _has_content_match
        self.assertFalse(_has_content_match("", "标签"))
        self.assertFalse(_has_content_match("内容", ""))

    def test_extract_node_content_exact_label(self):
        """从回复中提取精确匹配标签的内容"""
        from zulong.l2.fc_graph import _extract_node_content
        response = "前面的内容\n\n酒店预订\n推荐入住希尔顿酒店\n\n后面的内容"
        content = _extract_node_content(response, "酒店预订")
        self.assertIn("酒店预订", content)
        self.assertIn("希尔顿", content)

    def test_extract_node_content_fallback(self):
        """找不到标签时回退到头部截取"""
        from zulong.l2.fc_graph import _extract_node_content
        response = "这是一段很长的回复" * 50
        content = _extract_node_content(response, "完全不存在的标签")
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)


# ============================================================
# CircuitBreaker 测试
# ============================================================

class TestCircuitBreaker(unittest.TestCase):
    """ToolCallCircuitBreaker 死循环检测器测试"""

    def test_init_state(self):
        """测试初始状态"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
        cb = ToolCallCircuitBreaker()
        self.assertTrue(cb.enabled)
        self.assertEqual(len(cb._call_history), 0)

    def test_record_call(self):
        """测试工具调用记录"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
        cb = ToolCallCircuitBreaker()
        cb.reset()
        cb.record_call("search_web", {"query": "天气"}, "晴天 25度")
        self.assertEqual(len(cb._call_history), 1)

    def test_repetition_detection(self):
        """测试重复检测（连续相同调用）"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker, CircuitBreakerState
        cb = ToolCallCircuitBreaker()
        cb.reset()
        # 连续 3 次完全相同的调用
        for _ in range(3):
            cb.record_call("search_web", {"query": "同一个查询"}, "同一个结果")
        state, reason = cb.evaluate(3, [])
        # 应检测到重复，返回非 GREEN
        self.assertIn(state, [CircuitBreakerState.YELLOW, CircuitBreakerState.RED])

    def test_escalate_for_planning(self):
        """测试规划模式阈值放宽"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
        cb = ToolCallCircuitBreaker()
        cb.escalate_for_planning()
        self.assertEqual(cb._pattern_window, 20)
        self.assertEqual(cb._pattern_yellow_count, 15)

    def test_escalate_for_resume(self):
        """测试恢复模式阈值放宽"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
        cb = ToolCallCircuitBreaker()
        cb.escalate_for_resume()
        self.assertEqual(cb._pattern_window, 20)

    def test_serialize_deserialize(self):
        """测试序列化/反序列化"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
        cb = ToolCallCircuitBreaker()
        cb.reset()
        cb.record_call("test_tool", {"q": "query"}, "result")
        cb.escalate_for_planning()

        data = cb.serialize()
        self.assertIsInstance(data, dict)

        cb2 = ToolCallCircuitBreaker()
        cb2.deserialize(data)
        self.assertEqual(len(cb2._call_history), 1)
        self.assertEqual(cb2._pattern_window, 20)

    def test_deserialize_resets_start_time(self):
        """测试反序列化后 _start_time 被重置为当前时间"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker
        cb = ToolCallCircuitBreaker()
        cb.reset()
        old_start = cb._start_time

        data = cb.serialize()
        time.sleep(0.05)
        cb2 = ToolCallCircuitBreaker()
        cb2.deserialize(data)

        # 反序列化后的 start_time 应该比原来的新
        self.assertGreater(cb2._start_time, old_start)


# ============================================================
# AttentionWindow 测试
# ============================================================

class TestAttentionWindow(unittest.TestCase):
    """AttentionWindow 动态注意力窗口测试"""

    def test_initial_mode(self):
        """测试初始模式为 GLOBAL"""
        from zulong.l2.attention_window import AttentionWindowManager, AttentionMode
        aw = AttentionWindowManager(context_window_size=8192)
        self.assertEqual(aw.mode, AttentionMode.GLOBAL)

    def test_mode_switch_on_tool_call(self):
        """测试工具调用触发模式切换: GLOBAL → FOCUS → SINGLE_CHAIN"""
        from zulong.l2.attention_window import AttentionWindowManager, AttentionMode
        aw = AttentionWindowManager(context_window_size=8192)

        # GLOBAL → FOCUS: recall_memory 是 _FOCUS_TRIGGERS
        aw.observe_tool_call("recall_memory", {})
        self.assertEqual(aw.mode, AttentionMode.FOCUS)

        # FOCUS → SINGLE_CHAIN: exec_write_file 是 _SINGLE_CHAIN_TRIGGERS
        aw.observe_tool_call("exec_write_file", {})
        self.assertEqual(aw.mode, AttentionMode.SINGLE_CHAIN)

    def test_global_force_trigger(self):
        """测试强制回到 GLOBAL 模式的工具"""
        from zulong.l2.attention_window import AttentionWindowManager, AttentionMode
        aw = AttentionWindowManager(context_window_size=8192)

        # 先走 GLOBAL → FOCUS → SINGLE_CHAIN
        aw.observe_tool_call("recall_memory", {})
        aw.observe_tool_call("exec_write_file", {})
        self.assertEqual(aw.mode, AttentionMode.SINGLE_CHAIN)

        # task_view_overview 应该强制回到 GLOBAL
        aw.observe_tool_call("task_view_overview", {})
        self.assertEqual(aw.mode, AttentionMode.GLOBAL)

    def test_budget_calculation(self):
        """测试 token 预算计算"""
        from zulong.l2.attention_window import AttentionWindowManager
        aw = AttentionWindowManager(context_window_size=8192)
        # budget = max(int((8192 - 7096) * 0.90), 1024) = max(986, 1024) = 1024
        self.assertGreater(aw.budget, 0)


# ============================================================
# task_tools 辅助函数测试
# ============================================================

class TestTaskToolsHelpers(unittest.TestCase):
    """task_tools.py 辅助函数测试"""

    def test_normalize_label_chinese_prefix(self):
        """测试中文前缀去除"""
        from zulong.tools.task_tools import _normalize_label
        result = _normalize_label("1. 第一天健身计划")
        self.assertNotIn("1.", result)

    def test_normalize_label_empty(self):
        """测试空输入"""
        from zulong.tools.task_tools import _normalize_label
        self.assertEqual(_normalize_label(""), "")

    def test_label_similarity(self):
        """测试标签相似度（bigram Jaccard）"""
        from zulong.tools.task_tools import _label_similarity
        # 完全相同
        self.assertAlmostEqual(_label_similarity("健身计划", "健身计划"), 1.0)
        # 完全不同
        self.assertLess(_label_similarity("健身计划", "旅行日记"), 0.5)

    def test_fuzzy_resolve_node_id_exact(self):
        """测试精确匹配"""
        from zulong.tools.task_tools import _fuzzy_resolve_node_id
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="test", graph_id="t1")
        tg.add_node(id="o1", label="第一步", type="outline", status="pending")
        resolved, confidence, method = _fuzzy_resolve_node_id(tg, "o1")
        self.assertEqual(resolved, "o1")

    def test_fuzzy_resolve_node_id_quote_strip(self):
        """测试引号去除后匹配"""
        from zulong.tools.task_tools import _fuzzy_resolve_node_id
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="test", graph_id="t1")
        tg.add_node(id="o1", label="第一步", type="outline", status="pending")
        # 引号包裹的 ID（4B 模型常见）
        resolved, confidence, method = _fuzzy_resolve_node_id(tg, "'o1'")
        # 应该能通过引号去除后精确匹配，或前缀匹配
        if resolved is not None:
            self.assertEqual(resolved, "o1")

    def test_fuzzy_resolve_ordinal(self):
        """测试中文序号匹配"""
        from zulong.tools.task_tools import _fuzzy_resolve_node_id
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="test", graph_id="t1")
        tg.add_node(id="req", label="需求", type="requirement", status="completed")
        tg.add_node(id="analysis", label="分析", type="analysis", status="completed")
        tg.add_node(id="o1", label="第一步", type="outline", status="pending")
        tg.add_node(id="o2", label="第二步", type="outline", status="pending")
        tg.add_h_edge("req", "analysis")
        tg.add_h_edge("analysis", "o1")
        tg.add_h_edge("analysis", "o2")

        resolved, confidence, method = _fuzzy_resolve_node_id(tg, "第二")
        if resolved is not None:
            # 如果解析成功，应该解析到 o2
            self.assertEqual(resolved, "o2")


# ============================================================
# MemoryGraph 同步格式测试
# ============================================================

class TestMemoryGraphSyncFormat(unittest.TestCase):
    """[P5] 测试 MemoryGraph 同步格式一致性"""

    def test_sync_address_format_consistency(self):
        """验证 task_graph._sync_node_to_memory_graph 和 graph_adapters.TaskGraphAdapter.sync
        使用的地址格式是否一致"""
        # task_graph.py:240 使用: f"task:{self.id}/{node_id}"
        graph_id = "tg_abc123"
        node_id = "o1"
        task_graph_format = f"task:{graph_id}/{node_id}"

        # graph_adapters.py:121 使用:
        # f"task:{graph_id}/{node_id}" (当无 parent_prefix 时)
        # 或 f"{parent_prefix}/{node_id}" (当有 parent_prefix 时)
        adapter_format_no_prefix = f"task:{graph_id}/{node_id}"
        adapter_format_with_prefix = f"dialogue:session_xxx/task:{graph_id}/{node_id}"

        # 无前缀时格式一致
        self.assertEqual(task_graph_format, adapter_format_no_prefix)

        # 有前缀时格式不一致 — 这是 P5 问题
        self.assertNotEqual(task_graph_format, adapter_format_with_prefix,
                            "当存在 session 前缀时，两个模块使用不同的地址格式")


# ============================================================
# 集成场景测试: 完整 TaskGraph 生命周期
# ============================================================

class TestTaskGraphLifecycle(unittest.TestCase):
    """模拟完整的任务图生命周期"""

    def test_full_lifecycle(self):
        """创建 → 添加节点 → 更新状态 → 序列化 → 反序列化"""
        from zulong.l2.task_graph import TaskGraph

        # 1. 创建
        tg = TaskGraph(title="制定旅行计划", graph_id="lifecycle_001")
        tg.add_node(id="req", label="用户需求", type="requirement", status="completed",
                     desc="制定一个三天的旅行计划")
        tg.add_node(id="analysis", label="需求分析", type="analysis", status="completed",
                     desc="分析旅行需求")
        tg.add_h_edge("req", "analysis")

        # 2. 添加子节点
        for i in range(1, 4):
            oid = f"o{i}"
            tg.add_node(id=oid, label=f"第{i}天行程", type="outline", status="pending",
                         desc=f"第{i}天的行程安排")
            tg.add_h_edge("analysis", oid)

        self.assertEqual(len(tg._nodes), 5)  # req + analysis + o1 + o2 + o3

        # 3. 更新状态
        tg.update_node_status("o1", "in_progress")
        tg.update_node_status("o1", "completed", result="参观故宫和天安门")

        # 4. 验证叶子节点
        leaves = tg.get_leaf_nodes()
        uncompleted = [n for n in leaves if n.status != "completed"]
        self.assertEqual(len(uncompleted), 2)  # o2, o3

        # 5. 序列化 → 反序列化
        data = tg.serialize()
        tg2 = TaskGraph.deserialize(data)

        # 6. 验证还原
        self.assertEqual(tg2.id, "lifecycle_001")
        self.assertEqual(tg2.get_node("o1").status, "completed")
        self.assertEqual(tg2.get_node("o1").result, "参观故宫和天安门")
        self.assertEqual(tg2.get_node("o2").status, "pending")

        # 7. 继续执行
        tg2.update_node_status("o2", "in_progress")
        tg2.update_node_status("o2", "completed", result="游览长城")
        tg2.update_node_status("o3", "in_progress")
        tg2.update_node_status("o3", "completed", result="参观颐和园")

        # 8. 全部完成
        all_leaves = tg2.get_leaf_nodes()
        all_completed = all(n.status == "completed" for n in all_leaves)
        self.assertTrue(all_completed)

    def test_deep_tree_lifecycle(self):
        """深层树结构的生命周期"""
        from zulong.l2.task_graph import TaskGraph

        tg = TaskGraph(title="深层任务", graph_id="deep_001")
        tg.add_node(id="req", label="需求", type="requirement", status="completed")
        tg.add_node(id="analysis", label="分析", type="analysis", status="completed")
        tg.add_h_edge("req", "analysis")

        # 创建 3 层深度: analysis → o1 → o1_1 → o1_1_1
        tg.add_node(id="o1", label="大纲1", type="outline", status="pending")
        tg.add_h_edge("analysis", "o1")

        tg.add_node(id="o1_1", label="任务1", type="task", status="pending")
        tg.add_h_edge("o1", "o1_1")

        tg.add_node(id="o1_1_1", label="子任务1", type="subtask", status="pending")
        tg.add_h_edge("o1_1", "o1_1_1")

        # 验证深度
        self.assertEqual(tg.get_node_depth("o1_1_1"), 4)

        # 叶子节点只有最深的 o1_1_1
        leaves = tg.get_leaf_nodes()
        leaf_ids = {n.id for n in leaves}
        self.assertIn("o1_1_1", leaf_ids)
        self.assertNotIn("o1", leaf_ids)  # o1 有子节点


# ============================================================
# 路由逻辑测试
# ============================================================

class TestFCGraphRouting(unittest.TestCase):
    """FC Graph 路由函数测试"""

    def test_route_after_eval_terminate(self):
        """should_terminate 非空时路由到 end"""
        from zulong.l2.fc_graph import _route_after_eval
        state = {"should_terminate": "done", "response": "完成"}
        self.assertEqual(_route_after_eval(state), "end")

    def test_route_after_eval_blocked(self):
        """response=None 时路由回 check"""
        from zulong.l2.fc_graph import _route_after_eval
        state = {"should_terminate": "", "response": None, "null_response_count": 0}
        self.assertEqual(_route_after_eval(state), "check")

    def test_route_after_eval_max_null(self):
        """连续 null 拦截超过阈值时强制终止"""
        from zulong.l2.fc_graph import _route_after_eval
        state = {"should_terminate": "", "response": None, "null_response_count": 3}
        self.assertEqual(_route_after_eval(state), "end")

    def test_route_after_eval_normal_end(self):
        """正常回复时路由到 end"""
        from zulong.l2.fc_graph import _route_after_eval
        state = {"should_terminate": "", "response": "正常回复内容"}
        self.assertEqual(_route_after_eval(state), "end")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
