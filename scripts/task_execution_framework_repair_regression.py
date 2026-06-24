"""Regression checks for the task execution framework repair.

These checks are intentionally local and model-free.  They validate framework
invariants that must not depend on a specific pressure-test project, directory,
task id, or test runner.
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zulong.ide.ide_fc_runner import IDEFCRunner
from zulong.ide.ide_session import IDEFCState
from zulong.l2.attention_window import AttentionMode, AttentionWindowManager
from zulong.l2.task_graph import TaskGraph
from zulong.l2.fc_nodes import _make_eval_response_node
from zulong.tools import task_tools
from zulong.tools.base import ToolRequest
from zulong.tools.task_tools import SubmitFinalAnswerTool


def _make_tool_def(name: str, description: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


class _NoPersistence:
    def __enter__(self):
        self._orig_backup = task_tools._backup_graph_to_disk
        self._orig_sync = task_tools._sync_task_graph_to_memory
        task_tools._backup_graph_to_disk = lambda *args, **kwargs: None
        task_tools._sync_task_graph_to_memory = lambda *args, **kwargs: None
        return self

    def __exit__(self, exc_type, exc, tb):
        task_tools._backup_graph_to_disk = self._orig_backup
        task_tools._sync_task_graph_to_memory = self._orig_sync
        return False


def _reset_active_graph() -> None:
    with _NoPersistence():
        task_tools.set_active_task_graph(None, None)


def assert_no_keyword_workspace_inference() -> None:
    text = "项目根目录：D:/one/root，参考这个仓库 D:/two/ref，导出到 D:/three/out"
    assert task_tools.infer_project_workspace_hint(text) == ("", "")


def assert_capability_based_red3_progress() -> None:
    state = IDEFCState(cb_recovery_stage="note_attention", cb_force_no_tools=True)
    tools = [
        _make_tool_def(
            "persist_current_state",
            "持久化当前 note/memory anchor 并关联当前节点",
            {"content": {"type": "string"}},
        ),
        _make_tool_def(
            "select_focus_window",
            "选择 attention window: GLOBAL / FOCUS / SINGLE_CHAIN",
            {"mode": {"type": "string"}},
        ),
    ]
    IDEFCRunner._update_cb_recovery_progress(
        state,
        ["persist_current_state", "select_focus_window"],
        tools,
    )
    assert state.cb_recovery_note_saved is True
    assert state.cb_recovery_attention_switched is True
    assert state.cb_force_no_tools is False
    assert state.cb_recovery_stage == ""


def assert_taskspec_coverage_uses_capabilities() -> None:
    tg = TaskGraph("coverage")
    tg.add_node("req", "Root", type="requirement", status="in_progress", desc="root")
    node = tg.add_node(
        "n1",
        "Create report",
        status="completed",
        desc="Create a report file and verify it",
        result="Report file was created and verified with local evidence.",
    )
    tg.add_h_edge("req", node.id)
    tool_defs = [
        _make_tool_def(
            "persist_file_asset",
            "persist generated asset",
            {"path": {"type": "string"}, "content": {"type": "string"}},
        ),
        _make_tool_def(
            "run_validation",
            "return validation result",
            {"command": {"type": "string"}},
        ),
    ]
    tool_results = [
        {
            "tool_name": "persist_file_asset",
            "arguments": {"path": "report.md", "content": "..."},
            "result": {"ok": True, "resolved_path": "report.md"},
            "success": True,
        },
        {
            "tool_name": "run_validation",
            "arguments": {"command": "check report"},
            "result": {"ok": True, "output": "passed"},
            "success": True,
        },
    ]
    coverage = IDEFCRunner._collect_taskspec_coverage(
        tg,
        "Create and verify a report",
        tool_results,
        tool_defs,
    )
    assert coverage["checked"] is True
    assert coverage["missing_required_evidence"] == []
    assert coverage["write_evidence_count"] == 1
    assert coverage["verify_evidence_count"] == 1


def assert_attention_not_tool_name_bound() -> None:
    """Ordinary execution tool names must not drive L2 attention transitions."""
    mgr = AttentionWindowManager(context_window_size=8192)
    mgr.mode = AttentionMode.GLOBAL
    assert mgr._compute_transition("exec_run_command", {"command": "pytest"}) is None
    assert mgr.mode == AttentionMode.GLOBAL
    mgr.mode = AttentionMode.FOCUS
    assert mgr._compute_transition("exec_write_file", {"path": "x", "content": "y"}) is None
    assert mgr.mode == AttentionMode.FOCUS
    assert mgr._compute_transition("task_mark_status", {"node_id": "n1", "status": "completed"}) is None
    assert mgr.mode == AttentionMode.FOCUS
    mgr.auto_navigate_on_status_change("n1", "completed")
    assert mgr.mode == AttentionMode.FOCUS
    assert mgr._compute_transition("adjust_attention_mode", {"mode": "single_chain"}) == AttentionMode.SINGLE_CHAIN


def assert_backfill_does_not_complete_nodes() -> None:
    class DummyRunner:
        engine = type("Engine", (), {"_publish_task_graph_event": lambda *args, **kwargs: None})()

    tg = TaskGraph("backfill")
    tg.add_node("req", "Root", type="requirement", status="in_progress", desc="root")
    node = tg.add_node("n1", "Build module", status="pending", desc="Build module")
    tg.add_h_edge("req", node.id)
    with tempfile.TemporaryDirectory() as tmp:
        with _NoPersistence():
            task_tools.set_active_task_graph(tg, "tg_regression", workspace_dir=tmp)
        state = IDEFCState(fc_turn=7)
        with _NoPersistence():
            IDEFCRunner._run_backfill(
                DummyRunner(),
                state,
                "Build module\nThe module implementation candidate is ready for review.",
                is_cb_path=False,
            )
        assert tg.get_node("n1").status == "pending"
        assert "backfill_candidate_result" in tg.get_node("n1").metadata
    _reset_active_graph()


def assert_core_fc_nodes_do_not_auto_complete() -> None:
    """Core L2 FC text guards may record candidates, never complete nodes."""
    class DummyEngine:
        _attn_window = None
        _enable_semantic_drift_guard = False
        _semantic_drift_detector = False

        def _publish_task_graph_event(self, *args, **kwargs):
            pass

    response = (
        "Build module\n"
        "This module implementation candidate is ready for review with enough detail "
        "to look substantial, but it must not be auto-marked completed by framework text matching."
    )

    # Resume AutoMark path: should write auto_progress_candidate only.
    tg = TaskGraph("core-automark")
    tg.add_node("req", "Root", type="requirement", status="in_progress", desc="root")
    node = tg.add_node("n1", "Build module", status="in_progress", desc="Build module")
    tg.add_h_edge("req", node.id)
    with tempfile.TemporaryDirectory() as tmp:
        with _NoPersistence():
            task_tools.set_active_task_graph(tg, "tg_core_automark", workspace_dir=tmp)
        with _NoPersistence():
            result = _make_eval_response_node(DummyEngine())({
                "fc_turn": 3,
                "messages": [],
                "response_content": response,
                "cb_force_no_tools": False,
                "tool_results_buffer": [],
                "gap_continue_count": 0,
                "is_resume": True,
                "resume_automark_count": 0,
                "null_response_count": 0,
                "user_input_text": "Build module",
            })
        refreshed = tg.get_node("n1")
        assert refreshed.status == "in_progress"
        assert refreshed.metadata.get("auto_progress_candidate")
        assert result.get("response") is None
    _reset_active_graph()

    # First-run Backfill path: should write backfill_candidate_result only.
    tg = TaskGraph("core-backfill")
    tg.add_node("req", "Root", type="requirement", status="in_progress", desc="root")
    node = tg.add_node("n1", "Build module", status="pending", desc="Build module")
    tg.add_h_edge("req", node.id)
    with tempfile.TemporaryDirectory() as tmp:
        with _NoPersistence():
            task_tools.set_active_task_graph(tg, "tg_core_backfill", workspace_dir=tmp)
        with _NoPersistence():
            result = _make_eval_response_node(DummyEngine())({
                "fc_turn": 4,
                "messages": [],
                "response_content": response,
                "cb_force_no_tools": False,
                "tool_results_buffer": [],
                "gap_continue_count": 0,
                "is_resume": False,
                "resume_automark_count": 0,
                "null_response_count": 0,
                "user_input_text": "Build module",
            })
        refreshed = tg.get_node("n1")
        assert refreshed.status == "in_progress"  # guard may select current node, but not complete it
        assert refreshed.metadata.get("backfill_candidate_result")
        assert result.get("response") is None
    _reset_active_graph()


def assert_no_core_l2_final_answer_bypass() -> None:
    """Generic L2 completion must not import the root-completing helper."""
    text = (ROOT / "zulong" / "l2" / "inference_engine.py").read_text(encoding="utf-8")
    assert "_write_final_answer_to_task_graph as _write_final_graph" not in text
    assert "source=\"inference_engine\"" not in text
    assert "inference_engine_non_quality_path" in text


def assert_no_legacy_write_chunk_limit_in_core_fc() -> None:
    text = (ROOT / "zulong" / "l2" / "fc_nodes.py").read_text(encoding="utf-8")
    assert "_MAX_WRITE_CHUNK_CHARS" not in text
    assert "单块绝不能超过" not in text


def assert_submit_final_answer_is_candidate_only() -> None:
    tg = TaskGraph("submit")
    tg.add_node("req", "Root", type="requirement", status="in_progress", desc="root")
    with tempfile.TemporaryDirectory() as tmp:
        with _NoPersistence():
            task_tools.set_active_task_graph(tg, "tg_submit", workspace_dir=tmp)
        with _NoPersistence():
            result = SubmitFinalAnswerTool().execute(
                ToolRequest(
                    tool_name="submit_final_answer",
                    action="execute",
                    parameters={"answer": "候选最终答案，不应直接完成根节点。"},
                    request_id="regression-submit",
                )
            )
        root = tg.get_node("req")
        assert result.success is True
        assert root.status == "in_progress"
        assert root.metadata.get("candidate_final_answer")
    _reset_active_graph()


def main() -> None:
    assert_no_keyword_workspace_inference()
    assert_capability_based_red3_progress()
    assert_taskspec_coverage_uses_capabilities()
    assert_attention_not_tool_name_bound()
    assert_backfill_does_not_complete_nodes()
    assert_core_fc_nodes_do_not_auto_complete()
    assert_no_core_l2_final_answer_bypass()
    assert_no_legacy_write_chunk_limit_in_core_fc()
    assert_submit_final_answer_is_candidate_only()
    print("task_execution_framework_repair_regression: ok")


if __name__ == "__main__":
    main()
