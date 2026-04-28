"""验证 AttentionWindowManager.bulk_restore_history() 的正确性"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.attention_window import AttentionWindowManager


def test_basic_restore():
    """基本恢复：system + user + assistant + tool 消息"""
    awm = AttentionWindowManager(context_window_size=65536)

    # 模拟保存的历史消息（跳过前2条 system+user，因为 orchestrator 会重建）
    saved_messages = [
        {"role": "assistant", "content": "我来分析一下", "tool_calls": [
            {"id": "tc_1", "type": "function", "function": {"name": "view_graph_overview", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc_1", "content": "图谱概览..."},
        {"role": "assistant", "content": "继续执行", "tool_calls": [
            {"id": "tc_2", "type": "function", "function": {"name": "plan_add_node", "arguments": '{"label":"测试"}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc_2", "content": "节点已添加"},
        {"role": "assistant", "content": "纯文本输出，没有工具调用"},
        {"role": "user", "content": "[Supervisor 实时反馈] 请加快进度"},
        {"role": "assistant", "content": "好的", "tool_calls": [
            {"id": "tc_3", "type": "function", "function": {"name": "exec_write_file", "arguments": '{"path":"test.py"}'}}
        ]},
        {"role": "tool", "tool_call_id": "tc_3", "content": "文件已写入"},
    ]

    awm.bulk_restore_history(saved_messages, base_turn=1)

    # 验证消息数量
    assert len(awm.envelopes) == len(saved_messages), \
        f"Expected {len(saved_messages)} envelopes, got {len(awm.envelopes)}"

    # 验证 tool_call 分组
    # tc_1: assistant[0] + tool[1] 应该同组
    assert awm.envelopes[0].group_id is not None
    assert awm.envelopes[0].group_id == awm.envelopes[1].group_id
    # tc_2: assistant[2] + tool[3] 应该同组，但与 tc_1 不同组
    assert awm.envelopes[2].group_id is not None
    assert awm.envelopes[2].group_id == awm.envelopes[3].group_id
    assert awm.envelopes[0].group_id != awm.envelopes[2].group_id
    # 纯文本 assistant[4] 没有 group
    assert awm.envelopes[4].group_id is None
    # tc_3: assistant[6] + tool[7] 同组
    assert awm.envelopes[6].group_id is not None
    assert awm.envelopes[6].group_id == awm.envelopes[7].group_id

    # 验证工具名提取
    assert awm.envelopes[1].tool_name == "view_graph_overview"
    assert awm.envelopes[3].tool_name == "plan_add_node"
    assert awm.envelopes[7].tool_name == "exec_write_file"

    # 验证轮次推断（每个 assistant 消息 turn += 1）
    assert awm.envelopes[0].turn == 2  # base_turn=1, first assistant → 2
    assert awm.envelopes[1].turn == 2  # same turn as its assistant
    assert awm.envelopes[2].turn == 3  # second assistant
    assert awm.envelopes[4].turn == 4  # third assistant (no tools)
    assert awm.envelopes[5].turn == 4  # user msg stays at same turn
    assert awm.envelopes[6].turn == 5  # fourth assistant

    # 验证 _current_turn 更新
    assert awm._current_turn == 5

    # 验证没有 pinned 消息
    for env in awm.envelopes:
        assert not env.is_pinned, f"Restored message should not be pinned: {env.msg.get('role')}"

    print("test_basic_restore PASSED")


def test_empty_restore():
    """空消息列表"""
    awm = AttentionWindowManager(context_window_size=65536)
    awm.bulk_restore_history([], base_turn=0)
    assert len(awm.envelopes) == 0
    print("test_empty_restore PASSED")


def test_apply_window_after_restore():
    """恢复后 apply_window 正常工作"""
    awm = AttentionWindowManager(context_window_size=65536)

    # 先注册 pinned 消息（模拟 orchestrator 的 system+user）
    awm.register_message(
        {"role": "system", "content": "你是助手"}, turn=0, pinned=True
    )
    awm.register_message(
        {"role": "user", "content": "恢复任务..."}, turn=0, pinned=True
    )

    # 注入恢复历史
    history = [
        {"role": "assistant", "content": "OK " * 100},
        {"role": "assistant", "content": "继续 " * 100, "tool_calls": [
            {"id": "tc_1", "type": "function", "function": {"name": "test_tool", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "tc_1", "content": "结果 " * 100},
    ]
    awm.bulk_restore_history(history, base_turn=1)

    # apply_window 应该能正常返回
    windowed = awm.apply_window()
    assert len(windowed) > 0, "apply_window should return messages"
    # pinned 消息应该在最前面
    assert windowed[0]["role"] == "system"
    assert windowed[0]["content"] == "你是助手"

    print("test_apply_window_after_restore PASSED")


if __name__ == "__main__":
    test_basic_restore()
    test_empty_restore()
    test_apply_window_after_restore()
    print("\nAll tests PASSED!")
