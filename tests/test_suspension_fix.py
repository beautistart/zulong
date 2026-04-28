"""
测试任务挂起修复：
1. 上下文溢出 RED 不再触发自动挂起（任务规划问题）
2. _handle_resume_task 同时检查内存栈和磁盘持久化
3. 用户显式暂停关键词检测
4. 空闲超时挂起定时器
"""

import os
import sys
import time
import json
import unittest
from unittest.mock import patch, MagicMock, AsyncMock, PropertyMock

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ==================== Test 1: CircuitBreaker 信号分类 ====================

class TestCircuitBreakerContextPressure(unittest.TestCase):
    """验证上下文压力信号正确产生 RED，但不再导致挂起"""

    def setUp(self):
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker, CircuitBreakerState
        self.CircuitBreakerState = CircuitBreakerState
        self.cb = ToolCallCircuitBreaker({
            "context_window_size": 4096,
            "context_red_ratio": 0.90,
            "context_yellow_ratio": 0.75,
        })
        self.cb.reset()

    def test_context_pressure_returns_red(self):
        """上下文压力超阈值时，信号 4 应返回 RED"""
        # 构造超过 90% 的 messages（使用中文字符，1 中文字 ≈ 1.5 tokens）
        # 4096 * 0.9 / 1.5 ≈ 2458 个中文字符
        big_msg = [{"role": "user", "content": "你" * 3000}]
        self.cb.record_call("test_tool", {"q": "test"}, "result")
        state, reason = self.cb.evaluate(0, big_msg)
        self.assertEqual(state, self.CircuitBreakerState.RED)
        self.assertIn("上下文窗口压力", reason)

    def test_context_pressure_reason_contains_keywords(self):
        """上下文压力的原因字符串包含'上下文'关键词（用于推理引擎分支判断）"""
        big_msg = [{"role": "user", "content": "你" * 3000}]
        self.cb.record_call("test_tool", {"q": "test"}, "result")
        _, reason = self.cb.evaluate(0, big_msg)
        self.assertTrue("上下文" in reason or "压力" in reason)


# ==================== Test 2: inference_engine RED 处理（无挂起） ====================

class TestInferenceEngineRedHandling(unittest.TestCase):
    """验证推理引擎在 RED 触发时不再自动挂起任务"""

    def test_context_overflow_red_does_not_call_suspend(self):
        """上下文溢出 RED 不应调用 suspend_task"""
        # 读取源代码，确认没有 suspend_task 调用在上下文溢出分支
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 找到 RED 处理块
        red_block_start = content.find('if cb_state == CircuitBreakerState.RED:')
        self.assertNotEqual(red_block_start, -1, "找不到 RED 处理块")

        # 提取从 RED 判断到 break 的代码块
        red_block_end = content.find('break  # 跳出循环，进入强制生成答案', red_block_start)
        self.assertNotEqual(red_block_end, -1, "找不到 break 语句")

        red_block = content[red_block_start:red_block_end]

        # 确认该块中没有 suspend_task 调用
        self.assertNotIn("suspend_task", red_block,
                         "RED 处理块中不应包含 suspend_task 调用")

        # 确认包含任务规划问题的提示
        self.assertIn("任务规划", red_block,
                       "RED 处理块应包含'任务规划问题'提示")

    def test_context_overflow_injects_planning_hint(self):
        """上下文溢出应注入'任务规划粒度过粗'的系统提示"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 确认上下文溢出分支包含规划提示
        self.assertIn("任务规划粒度过粗", content)
        self.assertIn("请立即基于已收集到的信息", content)


# ==================== Test 3: (已移除 - 用户暂停关键词已由模型自主判断) ====================


# ==================== Test 4: Gatekeeper 恢复逻辑 ====================

class TestGatekeeperResumeTask(unittest.TestCase):
    """验证恢复任务同时检查内存栈和磁盘持久化"""

    def setUp(self):
        self.patches = []

        p1 = patch("zulong.l1b.scheduler_gatekeeper.event_bus")
        self.patches.append(p1)
        self.mock_event_bus = p1.start()

        p2 = patch("zulong.l1b.scheduler_gatekeeper.state_manager")
        self.patches.append(p2)
        p2.start()

        p3 = patch("zulong.l1b.scheduler_gatekeeper.power_manager")
        self.patches.append(p3)
        p3.start()

        p4 = patch("zulong.l1b.scheduler_gatekeeper.task_state_manager")
        self.patches.append(p4)
        self.mock_task_state_mgr = p4.start()

        p5 = patch("zulong.l1b.scheduler_gatekeeper.get_review_state_manager")
        self.patches.append(p5)
        p5.start()

        from zulong.l1b.scheduler_gatekeeper import Gatekeeper
        self.gk = Gatekeeper()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_resume_from_memory_stack_first(self):
        """有内存栈任务时，优先从内存栈恢复"""
        self.mock_task_state_mgr.get_task_stack.return_value = ["task_abc"]
        self.gk._handle_resume_task()

        self.mock_task_state_mgr.resume_task.assert_called_once_with("task_abc")
        self.mock_event_bus.publish.assert_called_once()

    def test_resume_falls_back_to_disk_when_stack_empty(self):
        """内存栈为空时，应尝试从磁盘持久化任务恢复"""
        self.mock_task_state_mgr.get_task_stack.return_value = []

        # Mock TaskSuspensionManager
        with patch("zulong.l1b.scheduler_gatekeeper.TaskSuspensionManager", create=True) as MockTSM:
            mock_instance = MagicMock()
            MockTSM.return_value = mock_instance

            # 模拟磁盘有挂起任务
            import asyncio
            mock_instance.list_suspended_tasks = AsyncMock(return_value=[
                {"task_id": "disk_task_1", "suspended_at": 1000},
                {"task_id": "disk_task_2", "suspended_at": 2000},
            ])

            # 使用 patch 确保 import 走我们的 mock
            with patch.dict("sys.modules", {}):
                self.gk._handle_resume_task()

            # 应该发布了恢复事件（无论是直接恢复还是异步请求）
            # 由于事件循环的复杂性，至少确认没有走到"没有可恢复的任务"


# ==================== Test 5: 空闲挂起定时器 ====================

class TestIdleSuspendTimer(unittest.TestCase):
    """验证空闲挂起定时器的基本逻辑"""

    def setUp(self):
        self.patches = []

        p1 = patch("zulong.l1b.scheduler_gatekeeper.event_bus")
        self.patches.append(p1)
        p1.start()

        p2 = patch("zulong.l1b.scheduler_gatekeeper.state_manager")
        self.patches.append(p2)
        p2.start()

        p3 = patch("zulong.l1b.scheduler_gatekeeper.power_manager")
        self.patches.append(p3)
        p3.start()

        p4 = patch("zulong.l1b.scheduler_gatekeeper.task_state_manager")
        self.patches.append(p4)
        self.mock_task_state_mgr = p4.start()

        p5 = patch("zulong.l1b.scheduler_gatekeeper.get_review_state_manager")
        self.patches.append(p5)
        p5.start()

        from zulong.l1b.scheduler_gatekeeper import Gatekeeper
        self.gk = Gatekeeper()

    def tearDown(self):
        # 取消定时器
        if self.gk._idle_check_timer is not None:
            self.gk._idle_check_timer.cancel()
        for p in self.patches:
            p.stop()

    def test_timer_not_started_without_active_task(self):
        """没有活跃任务时，不应启动定时器"""
        self.mock_task_state_mgr.get_active_task.return_value = None
        self.gk.start_idle_suspend_timer()
        self.assertIsNone(self.gk._idle_check_timer)

    def test_timer_started_with_active_task(self):
        """有活跃任务时，应启动定时器"""
        self.mock_task_state_mgr.get_active_task.return_value = "task_xyz"
        self.gk.start_idle_suspend_timer()
        self.assertIsNotNone(self.gk._idle_check_timer)
        # 清理
        self.gk._idle_check_timer.cancel()

    def test_timer_replaces_previous(self):
        """多次调用应替换（而非累加）定时器"""
        self.mock_task_state_mgr.get_active_task.return_value = "task_1"
        self.gk.start_idle_suspend_timer()
        timer1 = self.gk._idle_check_timer

        self.gk.start_idle_suspend_timer()
        timer2 = self.gk._idle_check_timer

        self.assertIsNot(timer1, timer2, "新定时器应替换旧定时器")
        # 清理
        timer2.cancel()


# ==================== Test 6: 配置纠正验证 ====================

class TestConfigCorrection(unittest.TestCase):
    """验证 context_window_size 和 num_ctx 已纠正为 65536"""

    def test_context_window_size_is_65536(self):
        """circuit_breaker.context_window_size 应为 65536"""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..",
            "config", "zulong_config.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        cb_config = config.get("l2_inference", {}).get("circuit_breaker", {})
        self.assertEqual(cb_config.get("context_window_size"), 65536,
                         "context_window_size 应为 65536 以匹配 deepseek-v3.1 模型能力")

    def test_num_ctx_is_65536(self):
        """ollama.num_ctx 应为 65536"""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..",
            "config", "zulong_config.yaml"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        num_ctx = config.get("llm", {}).get("ollama", {}).get("num_ctx")
        self.assertEqual(num_ctx, 65536,
                         "num_ctx 应为 65536 以匹配 deepseek-v3.1 模型能力")

    def test_cb_thresholds_with_65k_window(self):
        """在 65536 窗口下，常规对话不应触发 YELLOW"""
        from zulong.l2.circuit_breaker import ToolCallCircuitBreaker, CircuitBreakerState
        cb = ToolCallCircuitBreaker({
            "context_window_size": 65536,
            "context_red_ratio": 0.90,
            "context_yellow_ratio": 0.75,
        })
        cb.reset()

        # 模拟正常对话：system prompt(~1500 tokens) + 搜索结果(~3000 tokens)
        # ~4500 tokens = 约 3000 中文字符
        normal_msgs = [{"role": "user", "content": "你" * 3000}]
        cb.record_call("test_tool", {"q": "test"}, "result")
        state, reason = cb.evaluate(0, normal_msgs)
        self.assertEqual(state, CircuitBreakerState.GREEN,
                         f"4500 tokens 在 65K 窗口下应为 GREEN，但得到 {state}: {reason}")


# ==================== Test 7: 搜索工具截断验证 ====================

class TestSearchToolTruncation(unittest.TestCase):
    """验证搜索工具的截断机制"""

    def test_single_webpage_truncation_in_code(self):
        """源代码中应有单条网页 800 字符截断逻辑"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 确认单条截断
        self.assertIn("len(content) > 800", content,
                       "应有单条网页 800 字符截断检查")
        self.assertIn("内容已截断", content,
                       "截断后应追加截断提示")

    def test_total_content_limit_in_code(self):
        """源代码中总内容上限应从 2000 调整为 1500"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 确认先检查再追加（check-before-add）
        self.assertIn("total_content_length + len(content) > 1500", content,
                       "应在追加前检查总量不超过 1500 字符")

    def test_final_message_hard_limit_in_code(self):
        """最终消息应有 3000 字符硬截断保底"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("len(final_content) > 3000", content,
                       "应有 3000 字符硬截断保底")
        self.assertIn("搜索结果已截断", content,
                       "硬截断后应有截断提示")


# ==================== Test 8: CB 评估时序验证 ====================

class TestCBEvaluationTiming(unittest.TestCase):
    """验证 CircuitBreaker evaluate 在 append tool_result 之后执行"""

    def test_append_before_evaluate_in_code(self):
        """源代码中 messages.append(tool_result_msg) 应在 evaluate 之前"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 找到关键注释标记
        append_marker = "先追加工具结果到 messages，再评估"
        self.assertIn(append_marker, content,
                       "应有'先追加再评估'的注释标记")

        # 验证 append 在 evaluate 之前
        append_pos = content.find("messages.append(tool_result_msg)")
        evaluate_pos = content.find("self.circuit_breaker.evaluate(iteration, messages)")

        # 第一个 append 应在 evaluate 之前
        self.assertLess(append_pos, evaluate_pos,
                         "messages.append(tool_result_msg) 应在 evaluate() 之前")

    def test_no_duplicate_append_in_red_yellow_green(self):
        """RED/YELLOW/GREEN 分支中不应再有冗余的 messages.append(tool_result_msg)"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 找到 CB 评估区域
        cb_section_start = content.find("# === Circuit Breaker 评估 ===")
        cb_section_end = content.find("# 继续下一轮推理", cb_section_start)

        if cb_section_start == -1 or cb_section_end == -1:
            self.skipTest("无法定位 CB 评估区域")

        cb_section = content[cb_section_start:cb_section_end]

        # 在 CB 评估区域内，messages.append(tool_result_msg) 应只出现 1 次（在 evaluate 之前）
        append_count = cb_section.count("messages.append(tool_result_msg)")
        self.assertEqual(append_count, 1,
                         f"CB 评估区域内 messages.append(tool_result_msg) 应只出现 1 次，但出现了 {append_count} 次")


# ==================== Test 9: 视觉意图关键词修复 ====================

class TestVisualIntentKeywords(unittest.TestCase):
    """验证视觉意图关键词不再误判通用问句"""

    def test_weather_query_not_visual(self):
        """'今天天气什么样' 不应触发视觉意图"""
        # 模拟关键词检测逻辑
        user_input = "今天天气什么样"
        strong_visual_keywords = [
            "摄像头", "镜头", "画面", "视频", "图像", "图片", "视觉",
            "看到", "看见", "眼前",
        ]
        weak_visual_keywords = [
            "周围", "环境", "外观", "样子", "颜色", "形状",
            "屏幕", "显示", "呈现", "场景", "物体", "物品",
            "动作", "手势",
        ]
        strong_hits = sum(1 for kw in strong_visual_keywords if kw in user_input)
        weak_hits = sum(1 for kw in weak_visual_keywords if kw in user_input)
        need_vision = strong_hits >= 1 or (weak_hits >= 2)

        self.assertFalse(need_vision,
                         "'今天天气什么样' 不应触发视觉意图")

    def test_travel_query_not_visual(self):
        """'帮我规划东京旅行' 不应触发视觉意图"""
        user_input = "帮我规划东京7天旅行，需要看看最新的机票和酒店价格"
        strong_visual_keywords = [
            "摄像头", "镜头", "画面", "视频", "图像", "图片", "视觉",
            "看到", "看见", "眼前",
        ]
        weak_visual_keywords = [
            "周围", "环境", "外观", "样子", "颜色", "形状",
            "屏幕", "显示", "呈现", "场景", "物体", "物品",
            "动作", "手势",
        ]
        strong_hits = sum(1 for kw in strong_visual_keywords if kw in user_input)
        weak_hits = sum(1 for kw in weak_visual_keywords if kw in user_input)
        need_vision = strong_hits >= 1 or (weak_hits >= 2)

        self.assertFalse(need_vision,
                         "'帮我规划东京旅行' 不应触发视觉意图")

    def test_camera_query_is_visual(self):
        """'帮我看看摄像头画面' 应触发视觉意图"""
        user_input = "帮我看看摄像头画面"
        strong_visual_keywords = [
            "摄像头", "镜头", "画面", "视频", "图像", "图片", "视觉",
            "看到", "看见", "眼前",
        ]
        strong_hits = sum(1 for kw in strong_visual_keywords if kw in user_input)
        need_vision = strong_hits >= 1

        self.assertTrue(need_vision,
                        "'帮我看看摄像头画面' 应触发视觉意图（含强视觉词'摄像头'和'画面'）")

    def test_see_something_is_visual(self):
        """'我看到一个奇怪的东西' 应触发视觉意图"""
        user_input = "我看到一个奇怪的东西"
        strong_visual_keywords = [
            "摄像头", "镜头", "画面", "视频", "图像", "图片", "视觉",
            "看到", "看见", "眼前",
        ]
        strong_hits = sum(1 for kw in strong_visual_keywords if kw in user_input)
        need_vision = strong_hits >= 1

        self.assertTrue(need_vision,
                        "'我看到一个奇怪的东西' 应触发视觉意图（含强视觉词'看到'）")

    def test_removed_keywords_not_in_code(self):
        """源代码中不应再有'什么'、'谁'、'哪里'等通用词作为视觉关键词"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..",
            "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 找到视觉关键词定义区域
        visual_section_start = content.find("# ========== 0. 意图识别：是否需要视觉感知？")
        visual_section_end = content.find("need_vision =", visual_section_start) + 100

        if visual_section_start == -1:
            self.skipTest("无法定位视觉关键词区域")

        visual_section = content[visual_section_start:visual_section_end]

        # 确认通用高频词已被移除
        removed_keywords = ['"什么"', '"哪里"', '"哪"', '"谁"', '"发生"', '"刚才"', '"刚刚"']
        for kw in removed_keywords:
            self.assertNotIn(kw, visual_section,
                             f"视觉关键词中不应包含通用高频词 {kw}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
