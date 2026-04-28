# File: tests/test_history_persistence.py
"""
测试 Agent 执行历史持久化 (HistoryPersistenceManager)

验证内容：
1. 各种事件记录
2. 事件序列化/反序列化
3. flush 到 SharedMemoryPool (mock)
4. 从历史重建消息（断点续传）
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.history_persistence import (
    AgentHistoryEvent,
    HistoryPersistenceManager,
)


# ═══════════════════════════════════════════════════════════
# 测试: AgentHistoryEvent
# ═══════════════════════════════════════════════════════════

def test_event_serialization():
    """测试事件序列化/反序列化"""
    event = AgentHistoryEvent(
        event_type="tool_call",
        request_id="req-001",
        turn=5,
        timestamp=time.time(),
        attention_mode="focus",
        data={
            "tool_name": "plan_add_node",
            "tool_args_keys": ["parent_id", "label"],
            "result_snippet": "节点已添加",
        },
    )

    d = event.to_dict()
    assert d["event_type"] == "tool_call"
    assert d["request_id"] == "req-001"
    assert d["turn"] == 5
    assert d["attention_mode"] == "focus"
    assert d["data"]["tool_name"] == "plan_add_node"

    # 反序列化
    restored = AgentHistoryEvent.from_dict(d)
    assert restored.event_type == event.event_type
    assert restored.request_id == event.request_id
    assert restored.turn == event.turn
    assert restored.attention_mode == event.attention_mode
    assert restored.data == event.data

    return True


# ═══════════════════════════════════════════════════════════
# 测试: HistoryPersistenceManager 事件记录
# ═══════════════════════════════════════════════════════════

def test_record_session_start():
    """测试会话开始记录"""
    mgr = HistoryPersistenceManager(request_id="test-req")
    mgr.record_session_start(
        turn=0,
        goal="完成一个 Web 应用",
        context_window_size=4096,
        budget=3000,
    )

    assert mgr.pending_count == 1
    event = mgr._events[0]
    assert event.event_type == "session_start"
    assert event.data["goal"] == "完成一个 Web 应用"
    assert event.data["context_window_size"] == 4096

    return True


def test_record_tool_call():
    """测试工具调用记录"""
    mgr = HistoryPersistenceManager(request_id="test-req")
    mgr.record_tool_call(
        turn=3,
        tool_name="plan_add_node",
        tool_args={"parent_id": "analysis", "label": "大纲1"},
        result_snippet="成功" * 200,  # 超长结果
        mode="global",
    )

    assert mgr.pending_count == 1
    event = mgr._events[0]
    assert event.event_type == "tool_call"
    assert event.data["tool_name"] == "plan_add_node"
    # result_snippet 应被截断到 200 字符
    assert len(event.data["result_snippet"]) <= 200

    return True


def test_record_mode_change():
    """测试模式切换记录"""
    mgr = HistoryPersistenceManager(request_id="test-req")
    mgr.record_mode_change(
        turn=5,
        old_mode="global",
        new_mode="focus",
        trigger_tool="view_node_detail",
    )

    event = mgr._events[0]
    assert event.event_type == "mode_change"
    assert event.data["old_mode"] == "global"
    assert event.data["new_mode"] == "focus"
    assert event.data["trigger_tool"] == "view_node_detail"

    return True


def test_record_node_complete():
    """测试节点完成记录"""
    mgr = HistoryPersistenceManager(request_id="test-req")
    mgr.record_node_complete(
        turn=10,
        node_id="t1",
        node_label="任务1",
        status="completed",
        mode="single_chain",
    )

    event = mgr._events[0]
    assert event.event_type == "node_complete"
    assert event.data["node_id"] == "t1"
    assert event.data["status"] == "completed"

    return True


def test_record_session_end():
    """测试会话结束记录"""
    mgr = HistoryPersistenceManager(request_id="test-req")
    mgr.record_session_end(
        turn=50,
        final_answer_snippet="项目已完成",
        total_duration=120.5,
        mode="global",
    )

    event = mgr._events[0]
    assert event.event_type == "session_end"
    assert event.data["total_duration"] == 120.5

    return True


# ═══════════════════════════════════════════════════════════
# 测试: should_flush
# ═══════════════════════════════════════════════════════════

def test_should_flush():
    """测试 flush 阈值"""
    mgr = HistoryPersistenceManager(request_id="test-req", flush_interval=5)

    for i in range(4):
        mgr.record_tool_call(
            turn=i, tool_name="plan_add_node",
            tool_args={}, result_snippet="ok", mode="global",
        )
    assert not mgr.should_flush, "4 个事件不应触发 flush"

    mgr.record_tool_call(
        turn=4, tool_name="plan_add_node",
        tool_args={}, result_snippet="ok", mode="global",
    )
    assert mgr.should_flush, "5 个事件应触发 flush"

    return True


# ═══════════════════════════════════════════════════════════
# 测试: resume_messages
# ═══════════════════════════════════════════════════════════

def test_resume_messages():
    """测试从历史事件重建消息"""
    events = [
        AgentHistoryEvent(
            event_type="session_start",
            request_id="r1",
            turn=0,
            timestamp=time.time(),
            attention_mode="global",
            data={"goal": "构建 API"},
        ),
        AgentHistoryEvent(
            event_type="tool_call",
            request_id="r1",
            turn=1,
            timestamp=time.time(),
            attention_mode="global",
            data={
                "tool_name": "plan_add_node",
                "tool_args_keys": ["parent_id"],
                "result_snippet": "节点已添加",
            },
        ),
        AgentHistoryEvent(
            event_type="node_complete",
            request_id="r1",
            turn=5,
            timestamp=time.time(),
            attention_mode="focus",
            data={
                "node_id": "t1",
                "node_label": "任务1",
                "status": "completed",
            },
        ),
    ]

    messages = HistoryPersistenceManager.resume_messages(events)

    assert len(messages) >= 3, f"期望至少 3 条消息，实际 {len(messages)}"

    # 第一条应是 user（来自 session_start 的 goal）
    assert messages[0]["role"] == "user"
    assert "构建 API" in messages[0]["content"]

    # 第二条应是 assistant（历史回放）
    assert messages[1]["role"] == "assistant"
    assert "plan_add_node" in messages[1]["content"]

    # 第三条应是 tool result
    assert messages[2]["role"] == "tool"

    # 第四条应是 system（节点完成）
    assert messages[3]["role"] == "system"
    assert "t1" in messages[3]["content"]

    return True


def test_resume_empty():
    """测试空事件列表"""
    messages = HistoryPersistenceManager.resume_messages([])
    assert messages == []

    return True


# ═══════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════

def main():
    tests = [
        test_event_serialization,
        test_record_session_start,
        test_record_tool_call,
        test_record_mode_change,
        test_record_node_complete,
        test_record_session_end,
        test_should_flush,
        test_resume_messages,
        test_resume_empty,
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
