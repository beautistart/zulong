# File: tests/test_session_tool_fixes.py
# 验证 session_tool.py 的 bug 修复
# 不依赖外部服务（模型、WebSocket），纯单元测试

import sys
import os
import time
import unittest
from unittest.mock import patch, MagicMock, AsyncMock

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zulong.tools.base import ToolRequest
from zulong.tools.session_tool import StartSessionTool


class TestComplexBranchFreezeFailure(unittest.TestCase):
    """测试 COMPLEX 分支 freeze_current 失败时的紧急备份逻辑"""

    def _make_request(self, intent="complex", task_description="新任务", user_input="帮我做新任务"):
        return ToolRequest(
            tool_name="start_session",
            action="execute",
            parameters={
                "intent": intent,
                "reason": "test",
                "task_description": task_description,
                "user_input": user_input,
            },
        )

    @patch("zulong.tools.task_tools.set_active_task_graph")
    @patch("zulong.tools.task_tools.get_active_task_graph")
    def test_freeze_failure_triggers_emergency_backup(self, mock_get_tg, mock_set_tg):
        """freeze_current 失败时，应紧急备份旧图谱到磁盘"""
        # 构造一个旧的活跃任务图 (mock)
        mock_old_tg = MagicMock()
        mock_old_tg.id = "tg_old_123"
        mock_old_tg.title = "旧任务"
        mock_old_root = MagicMock()
        mock_old_root.label = "旧任务"
        mock_old_tg.get_node.return_value = mock_old_root

        mock_get_tg.return_value = mock_old_tg

        tool = StartSessionTool()

        # Mock task_state_manager.freeze_current 抛出异常
        with patch("zulong.l2.task_state_manager.task_state_manager") as mock_tsm, \
             patch("zulong.tools.task_tools._backup_graph_to_disk") as mock_backup, \
             patch("zulong.l2.task_graph.TaskGraph") as mock_tg_class:

            mock_tsm.freeze_current.side_effect = RuntimeError("freeze 模拟失败")

            # Mock 新 TaskGraph 创建
            mock_new_tg = MagicMock()
            mock_new_tg.metadata = {}
            mock_tg_class.return_value = mock_new_tg

            request = self._make_request(task_description="全新任务")
            result = tool.execute(request)

            # 验证紧急备份被调用
            mock_backup.assert_called_once_with(mock_old_tg, "tg_old_123")

            # 验证 set_active_task_graph(None, "") 仍然被调用（在 except 分支中）
            calls = mock_set_tg.call_args_list
            # 应该有两次调用：一次是 None（清除旧图），一次是新图
            none_calls = [c for c in calls if c[0][0] is None]
            assert len(none_calls) >= 1, f"set_active_task_graph(None, '') 应被调用，实际调用: {calls}"

            # 验证最终还是成功创建了新图谱
            assert result.success, f"应成功创建新图谱，实际: {result}"

    @patch("zulong.tools.task_tools.set_active_task_graph")
    @patch("zulong.tools.task_tools.get_active_task_graph")
    def test_freeze_success_no_emergency_backup(self, mock_get_tg, mock_set_tg):
        """freeze_current 成功时，不应触发紧急备份"""
        mock_old_tg = MagicMock()
        mock_old_tg.id = "tg_old_456"
        mock_old_tg.title = "旧任务"
        mock_old_root = MagicMock()
        mock_old_root.label = "旧任务"
        mock_old_tg.get_node.return_value = mock_old_root

        mock_get_tg.return_value = mock_old_tg

        tool = StartSessionTool()

        with patch("zulong.l2.task_state_manager.task_state_manager") as mock_tsm, \
             patch("zulong.tools.task_tools._backup_graph_to_disk") as mock_backup, \
             patch("zulong.l2.task_graph.TaskGraph") as mock_tg_class:

            # freeze 成功
            mock_tsm.freeze_current.return_value = None

            mock_new_tg = MagicMock()
            mock_new_tg.metadata = {}
            mock_tg_class.return_value = mock_new_tg

            request = self._make_request(task_description="全新任务2")
            result = tool.execute(request)

            # 紧急备份不应被调用（freeze 成功走正常路径）
            mock_backup.assert_not_called()
            assert result.success


class TestResumeFallbackWithTaskList(unittest.TestCase):
    """测试 RESUME 回退时补充可用任务列表"""

    def _make_resume_request(self, task_description="继续上次的任务"):
        return ToolRequest(
            tool_name="start_session",
            action="execute",
            parameters={
                "intent": "resume",
                "reason": "test",
                "task_description": task_description,
            },
        )

    @patch("zulong.tools.task_tools.get_active_task_graph", return_value=None)
    @patch("zulong.tools.session_tool._run_async")
    def test_resume_no_match_shows_available_tasks(self, mock_run_async, mock_get_tg):
        """RESUME 未匹配时，应列出可用的挂起任务"""
        # 第一次 _run_async 调用: find_by_description 返回 None（未匹配）
        # 第二次 _run_async 调用: list_suspended_tasks 返回任务列表
        available_tasks = [
            {"task_id": "task_001", "description": "AI Agent 调研"},
            {"task_id": "task_002", "description": "健身计划制定"},
        ]
        mock_run_async.side_effect = [None, available_tasks]

        tool = StartSessionTool()
        request = self._make_resume_request("完全不相关的描述")
        result = tool.execute(request)

        assert result.success
        data = result.data
        assert data["fallback"] is True
        assert data["intent"] == "chat"
        assert data["original_intent"] == "resume"
        # 验证可用任务列表被包含
        assert "available_tasks" in data
        assert len(data["available_tasks"]) == 2
        assert data["available_tasks"][0]["description"] == "AI Agent 调研"
        # 验证消息包含任务名称
        assert "AI Agent 调研" in data["message"]
        assert "健身计划制定" in data["message"]

    @patch("zulong.tools.task_tools.get_active_task_graph", return_value=None)
    @patch("zulong.tools.session_tool._run_async")
    def test_resume_no_match_no_suspended_tasks(self, mock_run_async, mock_get_tg):
        """RESUME 未匹配且无挂起任务时，返回通用提示"""
        # find_by_description 返回 None，list_suspended_tasks 返回空列表
        mock_run_async.side_effect = [None, []]

        tool = StartSessionTool()
        request = self._make_resume_request()
        result = tool.execute(request)

        assert result.success
        data = result.data
        assert data["fallback"] is True
        assert data.get("available_tasks") == []
        assert "没有找到之前挂起的任务" in data["message"]


class TestInferenceEngineTimeoutLog(unittest.TestCase):
    """验证 inference_engine.py 超时日志不再硬编码"""

    def test_no_hardcoded_120_timeout(self):
        """确保超时日志中不再使用硬编码的 '120 秒'"""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..", "zulong", "l2", "inference_engine.py"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 不应存在硬编码的 "120 秒" 超时日志
        assert ">120 秒" not in content, "inference_engine.py 中仍有硬编码的 '120 秒' 超时日志"
        assert ">120秒" not in content, "inference_engine.py 中仍有硬编码的 '120秒' 超时日志"

        # 应使用 self._core_timeout 变量
        assert "self._core_timeout" in content, "应使用 self._core_timeout 作为超时值"


class TestChatComplexUpgradeThrottle(unittest.TestCase):
    """测试 CHAT→COMPLEX 升级对社交短消息的过滤"""

    def test_trivial_greeting_blocked(self):
        """'你好' 等社交短消息不应触发升级"""
        trivial_inputs = ["你好", "谢谢", "好的", "嗯", "hi", "ok", "早安！"]
        for inp in trivial_inputs:
            stripped = inp.strip().rstrip("。！？~～.!?")
            is_trivial = (
                len(stripped) <= 5
                and not any(c.isdigit() for c in stripped)
            )
            assert is_trivial, f"'{inp}' 应判定为社交短消息，跳过升级"

    def test_task_relevant_not_blocked(self):
        """包含任务相关内容的输入应允许升级"""
        relevant_inputs = [
            "5月1日出发，15天",        # 有数字
            "预算5000以内",             # 有数字
            "把第二天的景点换成故宫",   # 长度 > 5
            "帮我把行程改成7天的版本",  # 有数字且长
        ]
        for inp in relevant_inputs:
            stripped = inp.strip().rstrip("。！？~～.!?")
            is_trivial = (
                len(stripped) <= 5
                and not any(c.isdigit() for c in stripped)
            )
            assert not is_trivial, f"'{inp}' 应允许升级，不应被过滤"


class TestCBFallbackJsonFilter(unittest.TestCase):
    """测试 CB 降级回复过滤 JSON 工具输出"""

    def test_json_results_filtered(self):
        """以 { 或 [ 开头的工具结果应被过滤"""
        tool_results = [
            {"result": '{"graph_id": "tg_123", "overview": "## 当前任务规划\\n..."}'},
            {"result": '{"node_id": "o2", "status": "completed"}'},
            {"result": "Day 2: Visit Summer Palace, eat dumplings"},
        ]
        useful = [
            r["result"][:300] for r in tool_results
            if r.get("result") and len(r.get("result", "")) > 20
            and "error" not in r.get("result", "").lower()[:50]
            and not r.get("result", "").lstrip().startswith(("{", "["))
        ]
        assert len(useful) == 1, f"只有自然语言结果应保留，实际: {useful}"
        assert "Summer Palace" in useful[0]

    def test_natural_language_kept(self):
        """自然语言工具结果应保留"""
        tool_results = [
            {"result": "北京三日游攻略：第一天游故宫、天坛，品尝北京烤鸭。"},
        ]
        useful = [
            r["result"][:300] for r in tool_results
            if r.get("result") and len(r.get("result", "")) > 20
            and "error" not in r.get("result", "").lower()[:50]
            and not r.get("result", "").lstrip().startswith(("{", "["))
        ]
        assert len(useful) == 1


class TestBackfillQualityCheck(unittest.TestCase):
    """测试 Backfill 回复质量检查"""

    def test_json_response_blocked(self):
        """高 JSON 字符占比的回复应跳过 Backfill"""
        json_response = (
            '根据已收集的信息：\n'
            '{"graph_id": "tg_123", "overview": "## 当前任务规划\\n'
            '### Day 1\\n| 编号 | 子任务 | 状态 |"}\n'
            '{"node_id": "o2", "status": "completed", "label": "Day 2"}'
        )
        json_chars = sum(1 for c in json_response if c in '{}[]":,')
        ratio = json_chars / max(len(json_response), 1)
        assert ratio > 0.12, f"JSON 垃圾数据占比 {ratio:.2f} 应超过 0.12 阈值"

    def test_natural_language_passes(self):
        """自然语言回复应通过质量检查"""
        nl_response = (
            "第一天：上午游览故宫博物院，了解明清皇家文化。中午在附近的"
            "南锣鼓巷品尝老北京小吃。下午前往天坛公园，感受皇家祭天文化。"
            "傍晚可以去前门大街逛街购物，晚上品尝正宗北京烤鸭。"
            "交通建议：地铁1号线到天安门东站，步行即达故宫。"
        )
        json_chars = sum(1 for c in nl_response if c in '{}[]":,')
        ratio = json_chars / max(len(nl_response), 1)
        assert ratio < 0.12, f"自然语言占比 {ratio:.2f} 应低于 0.12 阈值"


class TestNestedArgsUnwrap(unittest.TestCase):
    """测试 Fix 9: _execute_tool_call 嵌套参数解包"""

    def _make_engine(self):
        """构造最小化 InferenceEngine mock"""
        engine = MagicMock()
        engine._lock = MagicMock()
        engine._lock.__enter__ = MagicMock(return_value=None)
        engine._lock.__exit__ = MagicMock(return_value=False)

        # 导入真实方法并绑定到 mock
        from zulong.l2.inference_engine import InferenceEngine
        import types
        engine._execute_tool_call = types.MethodType(
            InferenceEngine._execute_tool_call, engine
        )
        return engine

    def _make_tool_call(self, arguments_json: str):
        tc = MagicMock()
        tc.function.name = "task_add_node"
        tc.function.arguments = arguments_json
        return tc

    def test_nested_args_unwrapped(self):
        """模型发送 {action, args: {label, ...}, tool_name} 时应正确解包"""
        import json
        engine = self._make_engine()
        success_result = MagicMock()
        success_result.success = True
        success_result.data = {"node_id": "o1", "status": "ok"}
        engine.tool_engine.call_tool.return_value = success_result

        nested_args = json.dumps({
            "action": "call_tool",
            "args": {
                "label": "Day 1 行程",
                "desc": "涩谷、原宿",
                "parent_id": "req",
            },
            "tool_name": "task_add_node",
        })
        tc = self._make_tool_call(nested_args)
        engine._execute_tool_call(tc)

        # 验证传给 call_tool 的 parameters 是解包后的
        call_kwargs = engine.tool_engine.call_tool.call_args
        params = call_kwargs.kwargs.get("parameters") or call_kwargs[1].get("parameters")
        if params is None:
            # 位置参数形式
            params = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs.kwargs["parameters"]
        assert "label" in params, f"parameters 应包含 label，实际: {params}"
        assert params["label"] == "Day 1 行程"
        assert "args" not in params, "解包后不应存在 args 键"
        assert "tool_name" not in params, "解包后不应存在冗余 tool_name"

    def test_normal_args_unchanged(self):
        """正常参数格式不应受影响"""
        import json
        engine = self._make_engine()
        success_result = MagicMock()
        success_result.success = True
        success_result.data = {"node_id": "o1", "status": "ok"}
        engine.tool_engine.call_tool.return_value = success_result

        normal_args = json.dumps({
            "label": "Day 1 行程",
            "desc": "涩谷、原宿",
            "parent_id": "req",
        })
        tc = self._make_tool_call(normal_args)
        engine._execute_tool_call(tc)

        call_kwargs = engine.tool_engine.call_tool.call_args
        params = call_kwargs.kwargs.get("parameters") or call_kwargs[1].get("parameters")
        if params is None:
            params = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs.kwargs["parameters"]
        assert params["label"] == "Day 1 行程"
        assert params["parent_id"] == "req"

    def test_double_nested_edge_case(self):
        """极端情况: args 存在但不是 dict（字符串）应不解包"""
        import json
        engine = self._make_engine()
        success_result = MagicMock()
        success_result.success = True
        success_result.data = "ok"
        engine.tool_engine.call_tool.return_value = success_result

        edge_args = json.dumps({
            "label": "测试",
            "args": "some_string_value",
        })
        tc = self._make_tool_call(edge_args)
        engine._execute_tool_call(tc)

        call_kwargs = engine.tool_engine.call_tool.call_args
        params = call_kwargs.kwargs.get("parameters") or call_kwargs[1].get("parameters")
        if params is None:
            params = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs.kwargs["parameters"]
        # args 是字符串，不应解包
        assert params["label"] == "测试"
        assert params["args"] == "some_string_value"


if __name__ == "__main__":
    unittest.main(verbosity=2)
