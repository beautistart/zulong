"""
FC Graph 模块 [已废弃，保留向后兼容]

本文件的内容已迁移至:
- zulong.l2.fc_nodes — 节点工厂函数 (4个工厂函数 + 辅助函数 + FCLoopState)
- zulong.l2.fc_runner — FC循环运行器 (FCRunner + run_fc_loop)

新代码请使用:
    from zulong.l2.fc_nodes import (
        _make_check_node, _make_call_model_node,
        _make_exec_tools_node, _make_eval_response_node,
        FCLoopState, _is_filler_content, _has_content_match, _extract_node_content,
    )
    from zulong.l2.fc_runner import FCRunner, run_fc_loop
    from zulong.ide.ide_fc_runner import IDEFCRunner
"""

import warnings

warnings.warn(
    "zulong.l2.fc_graph is deprecated, use zulong.l2.fc_nodes + zulong.l2.fc_runner instead",
    DeprecationWarning,
    stacklevel=2,
)

# 重新导出所有公共符号以保持向后兼容
from zulong.l2.fc_nodes import (
    FCLoopState,
    _make_check_node,
    _make_call_model_node,
    _make_exec_tools_node,
    _make_eval_response_node,
    _is_filler_content,
    _has_content_match,
    _extract_node_content,
)

from zulong.l2.fc_runner import FCRunner, run_fc_loop

# 保留 LangGraph 图构建函数（已废弃，仅用于向后兼容）
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


def _route_after_check(state: dict) -> str:
    """[废弃] check 节点后的路由"""
    if state.get("should_terminate"):
        return "end"
    return "call_model"


def _route_after_call(state: dict) -> str:
    """[废弃] call_model 节点后的路由"""
    if state.get("should_terminate"):
        return "end"
    if state.get("tool_calls_data") is None and state.get("response_content") is None:
        return "check"
    if state.get("tool_calls_data"):
        return "exec_tools"
    return "eval_response"


def _route_after_eval(state: dict) -> str:
    """[废弃] eval_response 节点后的路由"""
    if state.get("should_terminate"):
        return "end"
    _MAX_NULL_RESPONSES = 3
    null_count = state.get("null_response_count", 0)
    if null_count >= _MAX_NULL_RESPONSES:
        return "end"
    return "check"


def build_fc_graph(engine) -> object:
    """[废弃] 构建 LangGraph StateGraph"""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph is not available")

    graph = StateGraph(FCLoopState)

    graph.add_node("check", _make_check_node(engine))
    graph.add_node("call_model", _make_call_model_node(engine))
    graph.add_node("exec_tools", _make_exec_tools_node(engine))
    graph.add_node("eval_response", _make_eval_response_node(engine))

    graph.set_entry_point("check")

    graph.add_conditional_edges(
        "check", _route_after_check,
        {"call_model": "call_model", "end": END},
    )
    graph.add_conditional_edges(
        "call_model", _route_after_call,
        {"end": END, "check": "check", "exec_tools": "exec_tools", "eval_response": "eval_response"},
    )
    graph.add_edge("exec_tools", "check")
    graph.add_conditional_edges(
        "eval_response", _route_after_eval,
        {"end": END, "check": "check"},
    )

    return graph.compile()
