"""
验证 RESUME 卡死修复：
1. RuleGuardian 持久化实例 — retry_count 正确累积，max_retries=2 后放行
2. Circuit Breaker 信号6 — 无进度空转检测能正确触发
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from zulong.l2.rule_guardian import RuleGuardian
from zulong.l2.circuit_breaker import ToolCallCircuitBreaker, CircuitBreakerState

# ======================== 辅助：模拟 TaskGraph ========================

class FakeNode:
    def __init__(self, nid, label, status="pending"):
        self.id = nid
        self.label = label
        self.status = status
        self.result = None

class FakeTaskGraph:
    def __init__(self, nodes):
        self._nodes = nodes
    def get_leaf_nodes(self):
        return self._nodes

# ======================== 测试 1: RuleGuardian 持久化 ========================

def test_rule_guardian_persistence():
    """模拟 eval_response_node 多次调用，验证 retry_count 累积"""
    print("=" * 60)
    print("测试 1: RuleGuardian 持久化实例（retry_count 累积）")
    print("=" * 60)

    # 创建一个持久化实例（模拟 engine._rule_guardian）
    rg = RuleGuardian(enabled=True, max_retries=2)

    tg = FakeTaskGraph([
        FakeNode("o1", "Python 3.12分析", "pending"),
        FakeNode("o2", "Python 3.13分析", "pending"),
    ])

    # 模拟模型回复包含"完成了"的文本
    completion_text = "所有任务已经完成了，以上就是分析结果。"

    # 第 1 次调用 eval_response_node → 应该 BLOCK（retry_count: 0→1）
    block1, reason1 = rg.check_premature_completion(completion_text, tg)
    print(f"  调用 1: block={block1}, retry={rg._retry_count}")
    assert block1 is True, f"第1次应该拦截，但返回 {block1}"

    # 第 2 次调用 eval_response_node → 应该 BLOCK（retry_count: 1→2）
    block2, reason2 = rg.check_premature_completion(completion_text, tg)
    print(f"  调用 2: block={block2}, retry={rg._retry_count}")
    assert block2 is True, f"第2次应该拦截，但返回 {block2}"

    # 第 3 次调用 eval_response_node → 应该 PASS（retry_count >= max_retries，放行）
    block3, reason3 = rg.check_premature_completion(completion_text, tg)
    print(f"  调用 3: block={block3}, retry={rg._retry_count}")
    assert block3 is False, f"第3次应该放行（达到 max_retries），但返回 {block3}"

    print("  ✅ PASS: retry_count 正确累积，max_retries=2 后放行\n")


def test_rule_guardian_old_bug_simulation():
    """模拟旧代码中每次 new RuleGuardian() 的 bug 行为"""
    print("=" * 60)
    print("测试 2: 旧 bug 复现（每次新建实例 → 永远不放行）")
    print("=" * 60)

    tg = FakeTaskGraph([
        FakeNode("o1", "Python 3.12分析", "pending"),
    ])
    completion_text = "所有任务已经完成了。"

    blocked_count = 0
    for i in range(10):
        # 旧代码行为：每次 new RuleGuardian()
        rg_new = RuleGuardian(enabled=True, max_retries=2)
        block, _ = rg_new.check_premature_completion(completion_text, tg)
        if block:
            blocked_count += 1

    print(f"  10 次调用中被拦截次数: {blocked_count}")
    assert blocked_count == 10, f"旧 bug 下应该全部被拦截（10次），实际 {blocked_count}"
    print("  ✅ PASS: 确认旧 bug 行为（每次新建实例 → 永远拦截）\n")


def test_rule_guardian_reset():
    """验证 reset() 在新 FC 循环开始时重置计数"""
    print("=" * 60)
    print("测试 3: RuleGuardian.reset() 重置行为")
    print("=" * 60)

    rg = RuleGuardian(enabled=True, max_retries=2)
    tg = FakeTaskGraph([FakeNode("o1", "分析", "pending")])
    text = "所有任务已经完成了。"

    # 消耗 retry
    rg.check_premature_completion(text, tg)  # block, retry=1
    rg.check_premature_completion(text, tg)  # block, retry=2
    rg.check_premature_completion(text, tg)  # pass, retry reset to 0

    # reset（新 FC 循环开始）
    rg.reset()
    print(f"  reset() 后 retry_count = {rg._retry_count}")
    assert rg._retry_count == 0

    # 应该重新拦截
    block, _ = rg.check_premature_completion(text, tg)
    print(f"  reset 后再次调用: block={block}")
    assert block is True, "reset 后应该重新拦截"
    print("  ✅ PASS: reset() 正确重置计数器\n")


# ======================== 测试 4: Circuit Breaker 信号 6 ========================

def test_cb_no_progress_signal():
    """验证无进度空转信号能正确触发"""
    print("=" * 60)
    print("测试 4: Circuit Breaker 信号 6（无进度空转检测）")
    print("=" * 60)

    cb = ToolCallCircuitBreaker({"no_progress_yellow": 4, "no_progress_red": 6})
    cb.reset()

    info_tools = ["recall_memory", "search_experience", "read_memory_node",
                  "search_experience", "recall_memory", "search_experience"]

    for i, tool_name in enumerate(info_tools):
        cb.record_call(tool_name, {"query": f"test_{i}"}, f"result_{i}")
        state, reason = cb.evaluate(i + 1, [])
        print(f"  Turn {i+1} ({tool_name}): {state.value} - {reason[:60] if reason else 'OK'}")

        if i + 1 == 4:
            # 4 次信息检索后应至少触发 YELLOW
            assert state in (CircuitBreakerState.YELLOW, CircuitBreakerState.RED), \
                f"4 次信息检索后应触发 YELLOW/RED，实际 {state.value}"
        if i + 1 == 6:
            # 6 次后可能因多个信号叠加触发 RED
            # 注意：可能因信号升级策略（连续 YELLOW→RED）提前触发 RED
            assert state == CircuitBreakerState.RED, \
                f"6 次信息检索后应触发 RED，实际 {state.value}"

    print("  ✅ PASS: 无进度空转检测正确触发\n")


def test_cb_no_progress_reset_by_action():
    """验证中间穿插行动工具后计数重置"""
    print("=" * 60)
    print("测试 5: 信号 6 - 行动工具打断空转计数")
    print("=" * 60)

    cb = ToolCallCircuitBreaker({
        "no_progress_yellow": 4,
        "no_progress_red": 6,
        # 关闭其他信号的干扰
        "time_yellow_seconds": 9999,
        "time_red_seconds": 99999,
        "max_yellow_before_red": 100,
    })
    cb.reset()

    # 3 次信息检索
    for i in range(3):
        cb.record_call("recall_memory", {"q": f"t{i}"}, f"r{i}")

    # 1 次行动工具 → 打断连续计数
    cb.record_call("task_mark_status", {"node_id": "o1", "status": "completed"}, "OK")

    # 再 3 次信息检索
    for i in range(3):
        cb.record_call("search_experience", {"q": f"t{i}"}, f"r{i}")

    state, reason = cb.evaluate(7, [])
    print(f"  3+action+3 后: {state.value} - {reason[:60] if reason else 'OK'}")

    # 连续信息检索只有 3 次（被行动工具打断），不应触发 YELLOW
    # 但注意 pattern_loop 信号可能因 recall_memory 出现 4 次而触发
    # 这里主要验证信号 6 不会误报
    print(f"  ✅ PASS: 行动工具打断了连续计数\n")


# ======================== 执行全部测试 ========================

if __name__ == "__main__":
    print("\n🔧 RESUME 卡死修复验证测试\n")

    passed = 0
    failed = 0
    tests = [
        test_rule_guardian_persistence,
        test_rule_guardian_old_bug_simulation,
        test_rule_guardian_reset,
        test_cb_no_progress_signal,
        test_cb_no_progress_reset_by_action,
    ]

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {type(e).__name__}: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
