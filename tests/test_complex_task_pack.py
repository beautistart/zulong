# File: tests/test_complex_task_pack.py
"""
ComplexTaskPack 单元测试

覆盖：
1. KVCacheSlotManager 基础操作
2. DeepReasoningEngine 规则降级模式
3. DeepReasoningEngine LLM 模式（mock）
4. ComplexTaskPack 安装与能力执行
5. PlanAndReason 联合能力
6. KV Cache Slot 过期清理
"""

import sys
import os
import time
import unittest
from unittest.mock import MagicMock, patch

# 确保项目根目录在 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestKVCacheSlotManager(unittest.TestCase):
    """KV Cache Slot Manager 测试"""

    def setUp(self):
        from zulong.skill_packs.packs.complex_task.kv_cache_slot import KVCacheSlotManager
        self.manager = KVCacheSlotManager(default_ttl=5, max_slots_per_session=3)

    def test_save_and_restore(self):
        """测试保存和恢复 slot"""
        self.manager.save_slot("sess1", "macro", {"key": "macro_data"})
        self.manager.save_slot("sess1", "detail", {"key": "detail_data"})

        # 恢复
        data = self.manager.restore_slot("sess1", "macro")
        self.assertEqual(data, {"key": "macro_data"})

        data2 = self.manager.restore_slot("sess1", "detail")
        self.assertEqual(data2, {"key": "detail_data"})

    def test_restore_nonexistent(self):
        """测试恢复不存在的 slot"""
        data = self.manager.restore_slot("sess_x", "nonexist")
        self.assertIsNone(data)

    def test_slot_eviction(self):
        """测试超出最大 slot 数量时淘汰最旧 slot"""
        self.manager.save_slot("sess1", "slot_a", "a")
        time.sleep(0.01)
        self.manager.save_slot("sess1", "slot_b", "b")
        time.sleep(0.01)
        self.manager.save_slot("sess1", "slot_c", "c")
        time.sleep(0.01)
        # 第 4 个 slot 应该淘汰 slot_a
        self.manager.save_slot("sess1", "slot_d", "d")

        self.assertIsNone(self.manager.restore_slot("sess1", "slot_a"))
        self.assertEqual(self.manager.restore_slot("sess1", "slot_d"), "d")

    def test_clear_session(self):
        """测试清除 session 所有 slot"""
        self.manager.save_slot("sess1", "s1", "1")
        self.manager.save_slot("sess1", "s2", "2")
        count = self.manager.clear_session("sess1")
        self.assertEqual(count, 2)
        self.assertIsNone(self.manager.restore_slot("sess1", "s1"))

    def test_expired_slot(self):
        """测试过期 slot 自动清理"""
        # TTL=5s，但我们用一个非常短的 TTL 来测试
        from zulong.skill_packs.packs.complex_task.kv_cache_slot import KVCacheSlotManager
        mgr = KVCacheSlotManager(default_ttl=0)  # 0 秒 TTL
        mgr.save_slot("sess1", "expired", "data")
        time.sleep(0.01)
        # 应该已过期
        self.assertIsNone(mgr.restore_slot("sess1", "expired"))

    def test_list_slots(self):
        """测试列出 slot"""
        self.manager.save_slot("sess1", "a", 1)
        self.manager.save_slot("sess1", "b", 2)
        slots = self.manager.list_slots("sess1")
        self.assertIn("a", slots)
        self.assertIn("b", slots)

    def test_get_stats(self):
        """测试统计信息"""
        self.manager.save_slot("s1", "a", 1)
        self.manager.save_slot("s2", "b", 2)
        stats = self.manager.get_stats()
        self.assertEqual(stats["total_sessions"], 2)
        self.assertEqual(stats["total_slots"], 2)


class TestDeepReasoningEngine(unittest.TestCase):
    """DeepReasoningEngine 测试"""

    def test_rules_fallback_mode(self):
        """测试规则降级模式（无 LLM）"""
        from zulong.skill_packs.packs.complex_task.reasoning_engine import DeepReasoningEngine
        engine = DeepReasoningEngine(llm_client=None, model_id="default")

        result = engine.deep_reason(
            problem="设计一个高并发消息队列系统",
            context="需要支持百万级消息吞吐",
            reasoning_depth=3,
            max_hypotheses=5,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "rules")
        self.assertIn("reasoning_chain", result)
        self.assertIn("conclusion", result)
        self.assertGreater(result["confidence"], 0)
        self.assertGreater(len(result["reasoning_chain"]), 3)  # 至少有分析+假设+结论

    def test_llm_mode_with_mock(self):
        """测试 LLM 模式（Mock 客户端）"""
        from zulong.skill_packs.packs.complex_task.reasoning_engine import DeepReasoningEngine
        import json

        # Mock LLM 客户端
        mock_client = MagicMock()

        # 为四步推理分别设置 mock response
        analysis_resp = MagicMock()
        analysis_resp.choices = [MagicMock()]
        analysis_resp.choices[0].message.content = json.dumps({
            "key_elements": ["高并发", "消息队列"],
            "constraints": ["百万级吞吐"],
            "goal": "设计高并发消息队列",
            "complexity": "high"
        })

        hypothesis_resp = MagicMock()
        hypothesis_resp.choices = [MagicMock()]
        hypothesis_resp.choices[0].message.content = json.dumps({
            "hypotheses": [
                {
                    "name": "Kafka架构",
                    "approach": "使用分区+副本机制",
                    "expected_effect": "百万级吞吐",
                    "risk": "运维复杂度高",
                    "confidence": 0.9
                },
                {
                    "name": "自研队列",
                    "approach": "基于共享内存的无锁队列",
                    "expected_effect": "极低延迟",
                    "risk": "开发周期长",
                    "confidence": 0.6
                }
            ]
        })

        verify_resp = MagicMock()
        verify_resp.choices = [MagicMock()]
        verify_resp.choices[0].message.content = json.dumps({
            "verified": [
                {
                    "name": "Kafka架构",
                    "feasibility": 0.95,
                    "strengths": ["成熟稳定", "社区支持"],
                    "weaknesses": ["运维成本"],
                    "verdict": "accept"
                },
                {
                    "name": "自研队列",
                    "feasibility": 0.5,
                    "strengths": ["极低延迟"],
                    "weaknesses": ["开发周期", "维护成本"],
                    "verdict": "conditional"
                }
            ]
        })

        conclusion_resp = MagicMock()
        conclusion_resp.choices = [MagicMock()]
        conclusion_resp.choices[0].message.content = json.dumps({
            "best_approach": "Kafka架构",
            "reason": "成熟稳定，满足百万级吞吐需求",
            "steps": ["环境搭建", "分区设计", "消费者组配置"],
            "risks": ["运维复杂度"],
            "alternatives": ["自研队列"],
            "confidence": 0.88
        })

        # 按调用顺序返回
        mock_client.chat.completions.create.side_effect = [
            analysis_resp, hypothesis_resp, verify_resp, conclusion_resp
        ]

        engine = DeepReasoningEngine(
            llm_client=mock_client,
            model_id="test-model",
        )

        result = engine.deep_reason(
            problem="设计一个高并发消息队列系统",
            context="需要支持百万级消息吞吐",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["mode"], "llm")
        self.assertIn("Kafka", result["conclusion"])
        self.assertEqual(mock_client.chat.completions.create.call_count, 4)

    def test_llm_fallback_on_error(self):
        """测试 LLM 调用失败时降级到规则"""
        from zulong.skill_packs.packs.complex_task.reasoning_engine import DeepReasoningEngine

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        engine = DeepReasoningEngine(
            llm_client=mock_client,
            model_id="test-model",
        )

        result = engine.deep_reason(problem="测试降级")
        # 应该成功（降级到规则模式）
        self.assertTrue(result["success"])

    def test_empty_problem(self):
        """测试空问题"""
        from zulong.skill_packs.packs.complex_task.reasoning_engine import DeepReasoningEngine
        engine = DeepReasoningEngine()
        result = engine.deep_reason(problem="")
        # 空问题也应能处理（走规则模式）
        self.assertTrue(result["success"])


class TestComplexTaskPack(unittest.TestCase):
    """ComplexTaskPack 集成测试"""

    def _create_mock_registry(self):
        registry = MagicMock()
        registry.register = MagicMock(return_value=True)
        return registry

    def test_install_without_llm(self):
        """测试无 LLM 安装"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()

        success = pack.install(registry, {})
        self.assertTrue(success)
        self.assertEqual(len(pack.get_tools()), 5)
        self.assertEqual(registry.register.call_count, 5)

    def test_install_with_llm(self):
        """测试带 LLM 安装"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()

        mock_llm = MagicMock()
        success = pack.install(registry, {
            "llm_client": mock_llm,
            "model_id": "test-model",
        })
        self.assertTrue(success)
        self.assertEqual(len(pack.get_tools()), 5)

    def test_manifest(self):
        """测试 manifest"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        manifest = pack.get_manifest()

        self.assertEqual(manifest.pack_id, "complex_task")
        self.assertIn("task_decompose", manifest.capabilities)
        self.assertIn("deep_reasoning", manifest.capabilities)
        self.assertIn("plan_and_reason", manifest.capabilities)
        self.assertEqual(len(manifest.capabilities), 5)

    def test_execute_task_decompose(self):
        """测试任务拆解能力"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()
        pack.install(registry, {})

        result = pack.execute("task_decompose", {
            "goal": "搜索最新AI论文，分析趋势，写总结报告"
        })
        self.assertTrue(result["success"])
        self.assertGreater(result["subtask_count"], 0)

    def test_execute_deep_reasoning(self):
        """测试深度推理能力（规则模式）"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()
        pack.install(registry, {})

        result = pack.execute("deep_reasoning", {
            "problem": "评估微服务架构与单体架构的适用场景"
        })
        self.assertTrue(result["success"])
        self.assertIn("conclusion", result)

    def test_execute_plan_and_reason(self):
        """测试联合 plan_and_reason 能力"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()
        pack.install(registry, {})

        result = pack.execute("plan_and_reason", {
            "goal": "分析系统性能瓶颈并优化"
        })
        self.assertTrue(result["success"])
        self.assertIn("subtasks", result)
        self.assertIn("reasoned_count", result)

    def test_uninstall(self):
        """测试卸载"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()
        pack.install(registry, {})

        self.assertTrue(pack.uninstall())
        self.assertEqual(len(pack.get_tools()), 0)

    def test_kv_cache_manager_accessible(self):
        """测试 KV Cache Manager 可访问"""
        from zulong.skill_packs.packs.complex_task import ComplexTaskPack
        pack = ComplexTaskPack()
        registry = self._create_mock_registry()
        pack.install(registry, {"kv_cache_ttl": 600, "max_slots_per_session": 4})

        mgr = pack.kv_cache_manager
        self.assertIsNotNone(mgr)
        self.assertEqual(mgr.default_ttl, 600)
        self.assertEqual(mgr.max_slots_per_session, 4)


if __name__ == '__main__':
    unittest.main()
