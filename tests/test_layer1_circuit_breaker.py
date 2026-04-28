"""
Layer 1: CircuitBreaker 独立模块测试

覆盖: 初始化、6个信号、YELLOW->RED 升级、模式切换、序列化
"""

import time

import pytest

from zulong.l2.circuit_breaker import (
    ToolCallCircuitBreaker,
    CircuitBreakerState,
    ToolCallRecord,
)


def _make_breaker(**overrides) -> ToolCallCircuitBreaker:
    """快捷创建 CircuitBreaker（可覆盖配置）"""
    cfg = {"enabled": True}
    cfg.update(overrides)
    return ToolCallCircuitBreaker(config=cfg)


def _record(breaker, name="web_search", params=None, result="ok"):
    """快捷记录一次调用"""
    breaker.record_call(name, params or {}, result)


# ============================================================
# TestCircuitBreakerInit
# ============================================================


class TestCircuitBreakerInit:

    def test_default_config(self):
        cb = ToolCallCircuitBreaker()
        assert cb.enabled is True
        assert cb.safety_hard_cap == 100
        assert cb._repetition_window == 3
        assert cb._pattern_window == 6
        assert cb._pattern_yellow_count == 5
        assert cb._pattern_red_count == 7

    def test_custom_config(self):
        cb = _make_breaker(safety_hard_cap=50, pattern_window=10)
        assert cb.safety_hard_cap == 50
        assert cb._pattern_window == 10

    def test_disabled_mode(self):
        cb = _make_breaker(enabled=False)
        assert cb.enabled is False
        assert cb.safety_hard_cap == 10

        # 禁用时 evaluate 总是返回 GREEN
        state, reason = cb.evaluate(1, [])
        assert state == CircuitBreakerState.GREEN


# ============================================================
# TestSignalRepetition (Signal 1)
# ============================================================


class TestSignalRepetition:

    def test_3_identical_calls_red(self):
        cb = _make_breaker(repetition_window=3)
        cb.reset()
        for _ in range(3):
            cb.record_call("web_search", {"query": "same"}, "result")

        state, reason = cb.evaluate(3, [])
        assert state == CircuitBreakerState.RED
        assert "相同" in reason or "重复" in reason or "repetition" in reason.lower()

    def test_2_identical_calls_yellow(self):
        cb = _make_breaker(repetition_window=3)
        cb.reset()
        cb.record_call("web_search", {"query": "same"}, "result")
        cb.record_call("web_search", {"query": "same"}, "result")

        state, reason = cb.evaluate(2, [])
        assert state == CircuitBreakerState.YELLOW

    def test_different_calls_green(self):
        cb = _make_breaker(repetition_window=3)
        cb.reset()
        cb.record_call("web_search", {"query": "a"}, "result_a")
        cb.record_call("exec_run_command", {"cmd": "ls"}, "result_b")
        cb.record_call("web_search", {"query": "c"}, "result_c")

        state, reason = cb.evaluate(3, [])
        # 三次不同调用，信号1 应该是 GREEN
        signal_state, _ = cb._signal_repetition()
        assert signal_state == CircuitBreakerState.GREEN


# ============================================================
# TestSignalPatternLoop (Signal 2)
# ============================================================


class TestSignalPatternLoop:

    def test_same_tool_exceeds_red_count(self):
        cb = _make_breaker(pattern_window=8, pattern_red_count=7)
        cb.reset()
        for i in range(8):
            cb.record_call("web_search", {"query": f"q{i}"}, f"r{i}")

        state, _ = cb._signal_pattern_loop()
        assert state == CircuitBreakerState.RED

    def test_same_tool_exceeds_yellow_count(self):
        cb = _make_breaker(pattern_window=6, pattern_yellow_count=5, pattern_red_count=7)
        cb.reset()
        for i in range(6):
            cb.record_call("web_search", {"query": f"q{i}"}, f"r{i}")

        state, _ = cb._signal_pattern_loop()
        assert state in (CircuitBreakerState.YELLOW, CircuitBreakerState.RED)

    def test_planning_tool_exempt(self):
        cb = _make_breaker(pattern_window=8, pattern_red_count=7)
        cb.reset()
        cb.escalate_for_planning()

        # 规划工具在规划模式下应被豁免
        for i in range(8):
            cb.record_call("plan_add_node", {"id": f"o{i}"}, f"ok{i}")

        state, _ = cb._signal_pattern_loop()
        assert state == CircuitBreakerState.GREEN


# ============================================================
# TestSignalInfoGain (Signal 3)
# ============================================================


class TestSignalInfoGain:

    def test_identical_results_red(self):
        cb = _make_breaker(info_gain_window=3)
        cb.reset()
        for _ in range(3):
            cb.record_call("web_search", {"query": "test"}, "same result content")

        state, _ = cb._signal_info_gain()
        assert state == CircuitBreakerState.RED

    def test_empty_results_yellow(self):
        cb = _make_breaker(info_gain_window=3)
        cb.reset()
        # 使用不同的极短结果（不同 hash 避免触发信号3a RED）
        cb.record_call("web_search", {"query": "a"}, "x")
        cb.record_call("web_search", {"query": "b"}, "y")
        cb.record_call("web_search", {"query": "c"}, "z")

        state, _ = cb._signal_info_gain()
        assert state == CircuitBreakerState.YELLOW

    def test_diverse_results_green(self):
        cb = _make_breaker(info_gain_window=3)
        cb.reset()
        cb.record_call("web_search", {"query": "a"}, "result about topic A with details")
        cb.record_call("web_search", {"query": "b"}, "result about topic B with details")
        cb.record_call("web_search", {"query": "c"}, "result about topic C with details")

        state, _ = cb._signal_info_gain()
        assert state == CircuitBreakerState.GREEN


# ============================================================
# TestSignalContextPressure (Signal 4)
# ============================================================


class TestSignalContextPressure:

    def _make_messages(self, total_tokens_approx):
        """生成约指定 token 数的消息列表

        CB 的 token 估算: 中文 * 1.5 + 英文单词 * 0.75
        用空格分隔的英文单词最可控: 每个单词约 0.75 token
        """
        word_count = int(total_tokens_approx / 0.75) + 1
        text = " ".join(["word"] * word_count)
        return [{"role": "user", "content": text}]

    def test_red_at_90_percent(self):
        cb = _make_breaker(context_window_size=1000,
                           context_red_ratio=0.90, context_yellow_ratio=0.75)
        cb.reset()
        # 需要 >= 900 tokens
        msgs = self._make_messages(950)
        state, _ = cb._signal_context_pressure(msgs)
        assert state == CircuitBreakerState.RED

    def test_yellow_at_75_percent(self):
        cb = _make_breaker(context_window_size=1000,
                           context_red_ratio=0.90, context_yellow_ratio=0.75)
        cb.reset()
        # 需要 >= 750 但 < 900 tokens
        msgs = self._make_messages(800)
        state, _ = cb._signal_context_pressure(msgs)
        assert state in (CircuitBreakerState.YELLOW, CircuitBreakerState.RED)

    def test_green_below_75(self):
        cb = _make_breaker(context_window_size=10000,
                           context_red_ratio=0.90, context_yellow_ratio=0.75)
        cb.reset()
        msgs = self._make_messages(100)
        state, _ = cb._signal_context_pressure(msgs)
        assert state == CircuitBreakerState.GREEN


# ============================================================
# TestSignalNoProgress (Signal 6)
# ============================================================


class TestSignalNoProgress:

    def test_consecutive_info_retrieval_red(self):
        cb = _make_breaker(no_progress_yellow=4, no_progress_red=6)
        cb.reset()
        for i in range(7):
            cb.record_call("recall_memory", {"query": f"q{i}"}, f"mem{i}")

        state, _ = cb._signal_no_progress()
        assert state == CircuitBreakerState.RED

    def test_action_tool_resets_counter(self):
        cb = _make_breaker(no_progress_yellow=4, no_progress_red=6)
        cb.reset()
        # 3 次检索
        for i in range(3):
            cb.record_call("recall_memory", {"query": f"q{i}"}, f"mem{i}")
        # 1 次行动 -> 重置
        cb.record_call("exec_write_file", {"path": "a.py"}, "ok")
        # 2 次检索
        for i in range(2):
            cb.record_call("search_experience", {"query": f"q{i}"}, f"exp{i}")

        state, _ = cb._signal_no_progress()
        # 只有 2 次连续检索，不应触发
        assert state == CircuitBreakerState.GREEN


# ============================================================
# TestEscalation
# ============================================================


class TestEscalation:

    def test_yellow_to_red_upgrade(self):
        cb = _make_breaker(max_yellow_before_red=2)
        cb.reset()

        # 制造两个连续 YELLOW
        cb.record_call("web_search", {"query": "same"}, "result1")
        cb.record_call("web_search", {"query": "same"}, "result2")
        state1, _ = cb.evaluate(1, [])
        assert state1 == CircuitBreakerState.YELLOW

        cb.record_call("web_search", {"query": "same"}, "result3")
        cb.record_call("web_search", {"query": "same"}, "result4")
        state2, _ = cb.evaluate(2, [])
        # 连续 2 次 YELLOW -> RED
        assert state2 == CircuitBreakerState.RED

    def test_reset_clears_state(self):
        cb = _make_breaker()
        cb.record_call("test", {}, "r")
        cb._consecutive_yellow_count = 5

        cb.reset()
        assert len(cb._call_history) == 0
        assert cb._consecutive_yellow_count == 0

    def test_escalate_for_planning(self):
        cb = _make_breaker()
        cb.escalate_for_planning()

        assert cb._planning_mode is True
        assert cb._pattern_window == 20
        assert cb._pattern_yellow_count == 15
        assert cb._pattern_red_count == 20
        assert cb._max_yellow_before_red == 5

    def test_reset_to_default(self):
        cb = _make_breaker()
        cb.escalate_for_planning()
        cb.reset_to_default()

        assert cb._planning_mode is False
        # 应恢复到默认或配置值
        assert cb._pattern_window <= 10


# ============================================================
# TestSerializationCB
# ============================================================


class TestSerializationCB:

    def test_serialize_deserialize(self):
        cb = _make_breaker()
        cb.reset()
        cb.record_call("web_search", {"query": "test"}, "result data")
        cb.record_call("exec_run_command", {"cmd": "ls"}, "files")

        serialized = cb.serialize()
        assert len(serialized["call_history"]) == 2
        assert "elapsed_at_suspend" in serialized
        assert "planning_mode" in serialized

        # 恢复
        cb2 = _make_breaker()
        cb2.deserialize(serialized)
        assert len(cb2._call_history) == 2
        assert cb2._call_history[0].function_name == "web_search"
        assert cb2._call_history[1].function_name == "exec_run_command"
