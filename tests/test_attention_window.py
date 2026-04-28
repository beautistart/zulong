# File: tests/test_attention_window.py
"""
测试动态注意力窗口 (AttentionWindowManager)

验证内容：
1. 三种模式的状态机切换
2. token 预算控制和消息淘汰
3. pinned 消息永不淘汰
4. 模式权重评分
5. 工具调用组原子性
6. 淘汰摘要生成
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.attention_window import (
    AttentionWindowManager,
    AttentionMode,
    MessageEnvelope,
    estimate_tokens,
    _estimate_message_tokens,
)
from pipeline.task_graph import TaskGraph


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def create_test_graph() -> TaskGraph:
    """创建测试用任务图谱"""
    g = TaskGraph(title="test")
    g.add_node("req", label="需求", type="requirement", status="completed", desc="测试需求")
    g.add_node("analysis", label="分析", type="analysis", status="completed", desc="需求分析")
    g.add_h_edge("req", "analysis")
    g.add_node("o1", label="大纲1", type="outline", status="pending", desc="模块1")
    g.add_h_edge("analysis", "o1")
    g.add_node("t1", label="任务1", type="task", status="pending", desc="任务1")
    g.add_h_edge("o1", "t1")
    g.add_node("t2", label="任务2", type="task", status="pending", desc="任务2")
    g.add_h_edge("o1", "t2")
    g.add_node("o2", label="大纲2", type="outline", status="pending", desc="模块2")
    g.add_h_edge("analysis", "o2")
    g.add_node("t3", label="任务3", type="task", status="pending", desc="任务3")
    g.add_h_edge("o2", "t3")
    return g


def make_msg(role, content, tool_calls=None):
    """构造消息"""
    msg = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


# ═══════════════════════════════════════════════════════════
# 测试: estimate_tokens
# ═══════════════════════════════════════════════════════════

def test_estimate_tokens():
    """测试 token 估算"""
    # 空字符串
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0

    # 纯英文
    tokens = estimate_tokens("hello world")
    assert tokens > 0

    # 纯中文
    tokens_cn = estimate_tokens("你好世界")
    assert tokens_cn > 0

    # 混合
    tokens_mix = estimate_tokens("hello 你好 world 世界")
    assert tokens_mix > 0

    return True


# ═══════════════════════════════════════════════════════════
# 测试: 模式切换状态机
# ═══════════════════════════════════════════════════════════

def test_mode_transitions():
    """测试工具名驱动的模式切换"""
    graph = create_test_graph()
    mgr = AttentionWindowManager(
        context_window_size=4096,
        task_graph=graph,
    )

    # 初始模式 = GLOBAL
    assert mgr.mode == AttentionMode.GLOBAL

    # GLOBAL → FOCUS: view_node_detail
    new_mode = mgr.observe_tool_call("view_node_detail", {"node_id": "t1"})
    assert new_mode == AttentionMode.FOCUS
    assert mgr.mode == AttentionMode.FOCUS

    # FOCUS → SINGLE_CHAIN: exec_write_file
    new_mode = mgr.observe_tool_call("exec_write_file", {"file_path": "test.py", "content": "pass"})
    assert new_mode == AttentionMode.SINGLE_CHAIN
    assert mgr.mode == AttentionMode.SINGLE_CHAIN

    # SINGLE_CHAIN → GLOBAL: plan_mark_status(completed)
    new_mode = mgr.observe_tool_call("plan_mark_status", {"node_id": "t1", "status": "completed", "result": "done"})
    assert new_mode == AttentionMode.GLOBAL
    assert mgr.mode == AttentionMode.GLOBAL

    return True


def test_mode_transitions_backtrace():
    """测试 backtrace 工具触发聚焦"""
    graph = create_test_graph()
    mgr = AttentionWindowManager(context_window_size=4096, task_graph=graph)

    # GLOBAL → FOCUS: backtrace_dependency_chain
    new_mode = mgr.observe_tool_call("backtrace_dependency_chain", {"node_id": "t1"})
    assert new_mode == AttentionMode.FOCUS

    return True


def test_mode_force_global():
    """测试 view_graph_overview 强制回全局"""
    graph = create_test_graph()
    mgr = AttentionWindowManager(context_window_size=4096, task_graph=graph)

    # 先进入 FOCUS
    mgr.observe_tool_call("view_node_detail", {"node_id": "t1"})
    assert mgr.mode == AttentionMode.FOCUS

    # view_graph_overview 强制回 GLOBAL
    new_mode = mgr.observe_tool_call("view_graph_overview", {})
    assert new_mode == AttentionMode.GLOBAL
    assert mgr.mode == AttentionMode.GLOBAL

    return True


def test_mode_no_transition():
    """测试不触发切换的工具"""
    graph = create_test_graph()
    mgr = AttentionWindowManager(context_window_size=4096, task_graph=graph)

    # plan_add_node 在 GLOBAL 模式不触发切换
    new_mode = mgr.observe_tool_call("plan_add_node", {"parent_id": "analysis", "label": "test"})
    assert new_mode is None
    assert mgr.mode == AttentionMode.GLOBAL

    return True


def test_focus_to_single_chain_via_mark_in_progress():
    """测试 FOCUS → SINGLE_CHAIN: plan_mark_status(in_progress)"""
    graph = create_test_graph()
    mgr = AttentionWindowManager(context_window_size=4096, task_graph=graph)

    # 先进入 FOCUS
    mgr.observe_tool_call("view_focused_context", {"node_id": "t1"})
    assert mgr.mode == AttentionMode.FOCUS

    # plan_mark_status(in_progress) 触发 FOCUS → SINGLE_CHAIN
    new_mode = mgr.observe_tool_call("plan_mark_status", {"node_id": "t1", "status": "in_progress"})
    assert new_mode == AttentionMode.SINGLE_CHAIN

    return True


# ═══════════════════════════════════════════════════════════
# 测试: 消息注册和窗口裁剪
# ═══════════════════════════════════════════════════════════

def test_pinned_messages():
    """测试 pinned 消息永不淘汰"""
    mgr = AttentionWindowManager(
        context_window_size=500,  # 很小的窗口
        reserved_tokens=100,
    )

    # 注册 pinned 消息
    mgr.register_message(
        make_msg("system", "你是 Agent" * 10), turn=0, pinned=True,
    )
    mgr.register_message(
        make_msg("user", "完成任务" * 10), turn=0, pinned=True,
    )

    # 注册大量非 pinned 消息
    for i in range(20):
        mgr.register_message(
            make_msg("assistant", f"回复内容 {i} " * 5),
            turn=i + 1,
        )

    result = mgr.apply_window()

    # pinned 消息必须在结果中
    pinned_contents = [
        m["content"] for m in result
        if m.get("role") == "system" and "Agent" in m.get("content", "")
    ]
    assert len(pinned_contents) >= 1, "pinned system 消息被淘汰了"

    user_contents = [
        m for m in result if m.get("role") == "user" and "完成任务" in m.get("content", "")
    ]
    assert len(user_contents) >= 1, "pinned user 消息被淘汰了"

    return True


def test_window_budget():
    """测试消息总量控制在预算内"""
    mgr = AttentionWindowManager(
        context_window_size=2000,
        reserved_tokens=500,
    )
    # budget = (2000 - 500) * 0.90 = 1350

    # 注册少量 pinned
    mgr.register_message(make_msg("system", "短提示"), turn=0, pinned=True)

    # 注册大量消息
    for i in range(50):
        mgr.register_message(
            make_msg("assistant", f"这是一段相当长的回复内容，需要占用较多 token {i}"),
            turn=i + 1,
        )

    result = mgr.apply_window()

    # 结果应该少于总消息数（有淘汰）
    total_registered = len(mgr.envelopes)
    assert len(result) < total_registered, (
        f"期望淘汰一些消息，但 result={len(result)} >= total={total_registered}"
    )

    return True


def test_tool_call_group_atomicity():
    """测试工具调用组的原子性（assistant + tool result 同保同弃）"""
    mgr = AttentionWindowManager(
        context_window_size=1500,
        reserved_tokens=200,
    )

    mgr.register_message(make_msg("system", "sys"), turn=0, pinned=True)

    # 模拟一轮工具调用
    group_id = mgr.new_tool_group()
    assistant_msg = make_msg("assistant", "我来调用工具", tool_calls=[{
        "id": "call_1", "type": "function",
        "function": {"name": "plan_add_node", "arguments": "{}"},
    }])
    mgr.register_message(assistant_msg, turn=1, group_id=group_id)

    tool_msg = make_msg("tool", "节点已添加")
    tool_msg["tool_call_id"] = "call_1"
    mgr.register_message(
        tool_msg, turn=1, tool_name="plan_add_node", group_id=group_id,
    )

    result = mgr.apply_window()

    # 在不超预算时，两条消息应都在
    roles_in_result = [m["role"] for m in result]
    if "assistant" in roles_in_result:
        assert "tool" in roles_in_result, "assistant 在结果中但 tool 不在"

    return True


def test_eviction_summary():
    """测试淘汰摘要生成"""
    mgr = AttentionWindowManager(
        context_window_size=800,
        reserved_tokens=200,
    )

    mgr.register_message(make_msg("system", "sys"), turn=0, pinned=True)

    # 注册大量消息以触发淘汰
    for i in range(30):
        mgr.register_message(
            make_msg("assistant", f"长回复 {i} " * 20),
            turn=i + 1,
            tool_name="plan_add_node",
            node_id=f"n{i}",
        )

    result = mgr.apply_window()

    # 检查是否有摘要消息
    summary_msgs = [
        m for m in result
        if m.get("role") == "system" and "淘汰" in m.get("content", "")
    ]
    assert len(summary_msgs) >= 1, "应该生成淘汰摘要"

    return True


# ═══════════════════════════════════════════════════════════
# 测试: 评分逻辑
# ═══════════════════════════════════════════════════════════

def test_score_by_mode():
    """测试三种模式下的权重差异"""
    graph = create_test_graph()
    mgr = AttentionWindowManager(
        context_window_size=8000,
        task_graph=graph,
    )

    # 注册一些消息
    env_overview = MessageEnvelope(
        msg=make_msg("tool", "overview"),
        seq=0, turn=0, tool_name="view_graph_overview",
    )
    env_node_detail = MessageEnvelope(
        msg=make_msg("tool", "detail of t1"),
        seq=1, turn=1, tool_name="view_node_detail", node_id="t1",
    )
    env_unrelated = MessageEnvelope(
        msg=make_msg("tool", "detail of t3"),
        seq=2, turn=1, tool_name="view_node_detail", node_id="t3",
    )

    # GLOBAL 模式：overview 应该权重高
    mgr.mode = AttentionMode.GLOBAL
    mgr._current_turn = 1
    score_overview = mgr._score_message(env_overview)
    score_detail = mgr._score_message(env_node_detail)
    assert score_overview > score_detail, "GLOBAL 模式下 overview 权重应更高"

    # FOCUS 模式：当前节点权重最高
    mgr.mode = AttentionMode.FOCUS
    mgr._current_node_id = "t1"
    score_current = mgr._score_message(env_node_detail)
    score_other = mgr._score_message(env_unrelated)
    assert score_current > score_other, "FOCUS 模式下当前节点权重应更高"

    # SINGLE_CHAIN 模式：不相关节点权重极低
    mgr.mode = AttentionMode.SINGLE_CHAIN
    mgr._current_node_id = "t1"
    score_chain = mgr._score_message(env_node_detail)
    score_unrel = mgr._score_message(env_unrelated)
    assert score_chain > score_unrel * 5, "SINGLE_CHAIN 模式下不相关节点应被大幅降权"

    return True


# ═══════════════════════════════════════════════════════════
# 测试: stats 属性
# ═══════════════════════════════════════════════════════════

def test_stats():
    """测试统计信息"""
    mgr = AttentionWindowManager(context_window_size=4096)
    assert mgr.stats["mode"] == "global"
    assert mgr.stats["total_messages"] == 0

    mgr.register_message(make_msg("system", "hello"), turn=0, pinned=True)
    assert mgr.stats["total_messages"] == 1
    assert mgr.stats["pinned_tokens"] > 0

    return True


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

def main():
    tests = [
        test_estimate_tokens,
        test_mode_transitions,
        test_mode_transitions_backtrace,
        test_mode_force_global,
        test_mode_no_transition,
        test_focus_to_single_chain_via_mark_in_progress,
        test_pinned_messages,
        test_window_budget,
        test_tool_call_group_atomicity,
        test_eviction_summary,
        test_score_by_mode,
        test_stats,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        name = test_fn.__name__
        try:
            result = test_fn()
            if result:
                print(f"  PASS: {name}")
                passed += 1
            else:
                print(f"  FAIL: {name} (returned False)")
                failed += 1
        except Exception as e:
            print(f"  FAIL: {name} -> {e}")
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
