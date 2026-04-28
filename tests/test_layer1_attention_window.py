"""
Layer 1: AttentionWindow 独立模块测试

覆盖: 枚举、token 估算、消息注册、模式切换、评分、窗口裁剪
"""

import pytest

from zulong.l2.attention_window import (
    AttentionMode,
    AttentionWindowManager,
    MessageEnvelope,
    estimate_tokens,
    _estimate_message_tokens,
)


def _make_manager(ctx_size=65536, **kwargs) -> AttentionWindowManager:
    return AttentionWindowManager(context_window_size=ctx_size, **kwargs)


# ============================================================
# TestAttentionModeEnum
# ============================================================


class TestAttentionModeEnum:

    def test_mode_values(self):
        assert AttentionMode.GLOBAL.value == "global"
        assert AttentionMode.FOCUS.value == "focus"
        assert AttentionMode.SINGLE_CHAIN.value == "single_chain"


# ============================================================
# TestTokenEstimation
# ============================================================


class TestTokenEstimation:

    def test_estimate_tokens_chinese(self):
        text = "这是一段中文测试"  # 8 个中文字
        tokens = estimate_tokens(text)
        assert tokens == int(8 * 1.5)  # 12

    def test_estimate_tokens_english(self):
        text = "this is a test sentence"  # 5 words
        tokens = estimate_tokens(text)
        assert tokens == int(5 * 0.75)  # 3

    def test_estimate_tokens_mixed(self):
        text = "这是test混合content"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_estimate_message_tokens(self):
        msg = {"role": "user", "content": "这是测试内容"}
        tokens = _estimate_message_tokens(msg)
        assert tokens > 0

    def test_estimate_message_tokens_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "test query"}',
                    }
                }
            ],
        }
        tokens = _estimate_message_tokens(msg)
        assert tokens > 0


# ============================================================
# TestMessageRegistration
# ============================================================


class TestMessageRegistration:

    def test_register_basic_message(self):
        mgr = _make_manager()
        mgr.register_message(
            msg={"role": "user", "content": "hello"},
            turn=1,
        )
        assert len(mgr.envelopes) == 1
        env = mgr.envelopes[0]
        assert env.turn == 1
        assert env.seq == 0
        assert env.is_pinned is False

    def test_register_pinned_message(self):
        mgr = _make_manager()
        mgr.register_message(
            msg={"role": "system", "content": "You are a helpful assistant"},
            turn=0,
            pinned=True,
        )
        assert mgr.envelopes[0].is_pinned is True

    def test_register_tool_message(self):
        mgr = _make_manager()
        group_id = mgr.new_tool_group()
        mgr.register_message(
            msg={"role": "tool", "content": "search result"},
            turn=2,
            tool_name="web_search",
            node_id="o1",
            group_id=group_id,
        )
        env = mgr.envelopes[0]
        assert env.tool_name == "web_search"
        assert env.node_id == "o1"
        assert env.group_id == group_id

    def test_new_tool_group(self):
        mgr = _make_manager()
        g1 = mgr.new_tool_group()
        g2 = mgr.new_tool_group()
        assert g2 == g1 + 1


# ============================================================
# TestModeTransitions
# ============================================================


class TestModeTransitions:

    def test_observe_focus_trigger(self):
        mgr = _make_manager()
        assert mgr.mode == AttentionMode.GLOBAL

        new_mode = mgr.observe_tool_call("recall_memory", {"query": "test"})
        assert new_mode == AttentionMode.FOCUS
        assert mgr.mode == AttentionMode.FOCUS

    def test_observe_single_chain_trigger(self):
        mgr = _make_manager()
        mgr.mode = AttentionMode.FOCUS

        new_mode = mgr.observe_tool_call("exec_write_file", {"path": "a.py"})
        assert new_mode == AttentionMode.SINGLE_CHAIN
        assert mgr.mode == AttentionMode.SINGLE_CHAIN

    def test_observe_global_force(self):
        mgr = _make_manager()
        mgr.mode = AttentionMode.SINGLE_CHAIN

        new_mode = mgr.observe_tool_call("submit_final_answer", {"answer": "done"})
        assert new_mode == AttentionMode.GLOBAL
        assert mgr.mode == AttentionMode.GLOBAL

    def test_no_transition_for_unrelated_tool(self):
        mgr = _make_manager()
        assert mgr.mode == AttentionMode.GLOBAL

        new_mode = mgr.observe_tool_call("task_add_node", {"id": "o1"})
        assert new_mode is None
        assert mgr.mode == AttentionMode.GLOBAL

    def test_navigate_deeper(self):
        mgr = _make_manager()
        assert mgr.mode == AttentionMode.GLOBAL

        mgr.on_navigate_attention("deeper")
        assert mgr.mode == AttentionMode.FOCUS

        mgr.on_navigate_attention("deeper")
        assert mgr.mode == AttentionMode.SINGLE_CHAIN

        # 最深了，保持不变
        mgr.on_navigate_attention("deeper")
        assert mgr.mode == AttentionMode.SINGLE_CHAIN

    def test_navigate_broader(self):
        mgr = _make_manager()
        mgr.mode = AttentionMode.SINGLE_CHAIN

        mgr.on_navigate_attention("broader")
        assert mgr.mode == AttentionMode.FOCUS

        mgr.on_navigate_attention("broader")
        assert mgr.mode == AttentionMode.GLOBAL

        # 最浅了，保持不变
        mgr.on_navigate_attention("broader")
        assert mgr.mode == AttentionMode.GLOBAL

    def test_navigate_jump(self):
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="Jump Test", graph_id="jt_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending", desc="")
        tg.add_node(id="a", label="A", type="analysis", status="pending", desc="")
        tg.add_node(id="o1", label="O1", type="outline", status="pending", desc="")
        tg.add_node(id="t1", label="T1", type="task", status="pending", desc="")
        tg.add_h_edge("req", "a")
        tg.add_h_edge("a", "o1")
        tg.add_h_edge("o1", "t1")

        mgr = _make_manager(task_graph=tg)
        # depth=0 -> GLOBAL
        mgr.on_navigate_attention("jump", target_node_id="req")
        assert mgr.mode == AttentionMode.GLOBAL

        # depth=2 -> FOCUS
        mgr.on_navigate_attention("jump", target_node_id="o1")
        assert mgr.mode == AttentionMode.FOCUS

        # depth=3 -> SINGLE_CHAIN
        mgr.on_navigate_attention("jump", target_node_id="t1")
        assert mgr.mode == AttentionMode.SINGLE_CHAIN


# ============================================================
# TestScoring
# ============================================================


class TestScoring:

    def test_time_decay(self):
        mgr = _make_manager()
        # 第 1 轮消息
        mgr.register_message({"role": "user", "content": "早期消息"}, turn=1)
        # 第 10 轮消息
        mgr.register_message({"role": "user", "content": "晚期消息"}, turn=10)
        mgr._current_turn = 10

        old_env = mgr.envelopes[0]
        new_env = mgr.envelopes[1]

        old_score = mgr._score_message(old_env)
        new_score = mgr._score_message(new_env)

        # 老消息分数更低
        assert old_score < new_score

    def test_mode_multiplier_global(self):
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="Mult Test", graph_id="mt_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending", desc="")

        mgr = _make_manager(task_graph=tg)
        mgr.mode = AttentionMode.GLOBAL

        # task_view_overview 在 GLOBAL 模式下应有高乘数
        env = MessageEnvelope(
            msg={"role": "tool", "content": "overview data"},
            seq=0, turn=1, tool_name="task_view_overview",
        )
        mgr._current_turn = 1
        mult = mgr._mode_multiplier(env)
        assert mult >= 1.3

    def test_mode_multiplier_focus_current_node(self):
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph(title="Focus Test", graph_id="fo_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending", desc="")
        tg.add_node(id="o1", label="O1", type="outline", status="in_progress", desc="")
        tg.add_h_edge("req", "o1")

        mgr = _make_manager(task_graph=tg)
        mgr.mode = AttentionMode.FOCUS
        mgr._current_node_id = "o1"

        # 当前聚焦节点消息
        env = MessageEnvelope(
            msg={"role": "tool", "content": "focused data"},
            seq=0, turn=1, node_id="o1",
        )
        mult = mgr._mode_multiplier(env)
        assert mult >= 2.5  # FOCUS 模式当前节点 3.0x


# ============================================================
# TestApplyWindow
# ============================================================


class TestApplyWindow:

    def test_empty_envelopes(self):
        mgr = _make_manager()
        result = mgr.apply_window()
        assert result == []

    def test_budget_respected(self):
        # 小预算窗口
        mgr = _make_manager(ctx_size=200, reserved_tokens=50)
        # budget = max(int((200-50)*0.90), 1024) = max(135, 1024) = 1024
        # 为了真正测试裁剪，需要更小的 budget
        mgr.budget = 50  # 手动设置小预算

        mgr.register_message(
            {"role": "system", "content": "sys"}, turn=0, pinned=True,
        )
        for i in range(20):
            mgr.register_message(
                {"role": "user", "content": f"这是第{i}条很长的消息，包含大量内容" * 5},
                turn=i + 1,
            )

        result = mgr.apply_window()
        # 消息应该被裁剪
        assert len(result) < 21

    def test_pinned_always_included(self):
        mgr = _make_manager()
        mgr.budget = 50

        mgr.register_message(
            {"role": "system", "content": "pinned system prompt"},
            turn=0, pinned=True,
        )
        for i in range(10):
            mgr.register_message(
                {"role": "user", "content": "filler message " * 20},
                turn=i + 1,
            )

        result = mgr.apply_window()
        # 第一条消息是 pinned，必须包含
        assert any(m.get("content") == "pinned system prompt" for m in result)

    def test_group_atomicity(self):
        mgr = _make_manager()
        mgr.budget = 5000

        # 注册一个工具调用组
        gid = mgr.new_tool_group()
        mgr.register_message(
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "search", "arguments": "{}"}}
            ]},
            turn=1, tool_name="search", group_id=gid,
        )
        mgr.register_message(
            {"role": "tool", "content": "search result data"},
            turn=1, tool_name="search", group_id=gid,
        )

        result = mgr.apply_window()
        # 组内消息应一起保留或一起淘汰
        has_assistant = any("tool_calls" in m for m in result)
        has_tool = any(m.get("role") == "tool" for m in result)
        # 如果有 assistant，也应有对应 tool（或者两者都没有）
        assert has_assistant == has_tool
