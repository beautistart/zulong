"""Phase 10 FC write-diagnostics regression checks.

This script is intentionally lightweight: it does not start the Web UI, model
server, or VS Code bridge. It verifies the deterministic safety edges that made
Phase 10 fragile: append chunking must preserve full file content, redacted
model-output evidence, root-cause classification, and bridge success that did
not actually apply.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zulong.ide.ide_fc_runner import IDEFCRunner
from zulong.ide.ide_session import IDEFCState
from zulong.tools.base import ToolRequest
from zulong.tools.ide_bridge_tools import IdeWriteFileTool


def assert_no_oversized_legacy_write_splitting() -> None:
    targets = [
        ROOT / "zulong" / "ide" / "ide_fc_runner.py",
        ROOT / "zulong" / "tools" / "ide_bridge_tools.py",
        ROOT / "zulong" / "tools" / "exec_tools.py",
        ROOT / "zulong-ide" / "src" / "hosts" / "vscode" / "VscodeExecutionBridge.ts",
    ]
    forbidden = ("1800",)
    for target in targets:
        text = target.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            assert marker not in text, f"legacy write splitting marker found: {marker} in {target}"


def assert_model_output_summary_is_redacted() -> None:
    runner = IDEFCRunner.__new__(IDEFCRunner)
    state = IDEFCState(fc_turn=1)
    tool_call = {
        "id": "call_1",
        "function": {
            "name": "write_to_file",
            "arguments": json.dumps(
                {
                    "path": "game/index.html",
                    "content": "secret body",
                    "api_key": "sk-secret",
                },
                ensure_ascii=False,
            ),
        },
    }
    summary = runner._record_model_raw_output(
        state,
        1,
        raw_content="about to write",
        final_content="about to write",
        tool_calls=[tool_call],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        source="phase10_regression",
    )
    call_summary = summary["tool_calls"][0]
    assert "content" in call_summary["redacted_keys"]
    assert "api_key" in call_summary["redacted_keys"]
    assert call_summary["redacted_argument_lengths"]["content"] == len("secret body")
    assert state.last_model_raw_summary["tool_call_count"] == 1


def assert_bridge_not_applied_is_classified() -> None:
    runner = IDEFCRunner.__new__(IDEFCRunner)
    runner._attn_window = None
    state = IDEFCState(fc_turn=2)
    state.tool_results_buffer.append(
        {
            "tool_name": "write_to_file",
            "result": "IDE tool returned ok but target path was not verified",
            "success": False,
        }
    )
    cause = runner._log_fc_decision_path(
        state,
        path="tool_result_review",
        tool_calls=[{"function": {"name": "write_to_file", "arguments": "{}"}}],
    )
    assert cause == "bridge_not_applied"


def assert_bridge_ok_without_file_fails(tmp_dir: Path) -> None:
    import zulong.tools.ide_bridge_tools as bridge_tools

    original = bridge_tools._run_async_request

    def fake_request(action, payload):
        return {"ok": True, "workspace_path": str(tmp_dir), "result": "applied"}

    bridge_tools._run_async_request = fake_request
    try:
        target = tmp_dir / "missing.html"
        result = IdeWriteFileTool().execute(
            ToolRequest(
                tool_name="ide_write_file",
                action="execute",
                parameters={
                    "file_path": str(target),
                    "workspace_path": str(tmp_dir),
                    "content": "<html></html>",
                },
            )
        )
    finally:
        bridge_tools._run_async_request = original

    assert result.success is False
    assert result.data["verified"] is False
    assert result.data["applied"] is False


def assert_append_mode_preserves_existing_content(tmp_dir: Path) -> None:
    import zulong.tools.ide_bridge_tools as bridge_tools
    import zulong.tools.task_tools as task_tools

    original_request = bridge_tools._run_async_request
    original_workspace = task_tools.get_active_workspace_dir

    def fake_request(action, payload):
        args = payload["arguments"]
        target = Path(args["path"])
        if not target.is_absolute():
            target = Path(payload["workspace_path"]) / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(args["content"], encoding="utf-8")
        return {
            "ok": True,
            "workspace_path": payload["workspace_path"],
            "result": f"已应用文件变更: {target}",
        }

    bridge_tools._run_async_request = fake_request
    task_tools.get_active_workspace_dir = lambda: str(tmp_dir)
    try:
        tool = IdeWriteFileTool()
        first = tool.execute(
            ToolRequest(
                tool_name="ide_write_file",
                action="execute",
                parameters={
                    "file_path": "append-check.txt",
                    "content": "alpha\n",
                    "mode": "overwrite",
                },
            )
        )
        second = tool.execute(
            ToolRequest(
                tool_name="ide_write_file",
                action="execute",
                parameters={
                    "file_path": "append-check.txt",
                    "content": "beta\n",
                    "mode": "append",
                },
            )
        )
    finally:
        bridge_tools._run_async_request = original_request
        task_tools.get_active_workspace_dir = original_workspace

    assert first.success is True
    assert second.success is True
    assert (tmp_dir / "append-check.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"


def main() -> None:
    tmp_dir = ROOT / "tmp" / "phase10_fc_repair_regression"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    assert_no_oversized_legacy_write_splitting()
    assert_model_output_summary_is_redacted()
    assert_bridge_not_applied_is_classified()
    assert_bridge_ok_without_file_fails(tmp_dir)
    assert_append_mode_preserves_existing_content(tmp_dir)
    print("phase10_fc_repair_regression: ok")


if __name__ == "__main__":
    main()
