"""Web full-chain regression entry for task lifecycle feedback.

Modes:
- auto (default): try ws://127.0.0.1:8090/ws, fall back to offline checks.
- offline: deterministic checks without starting Web/model services.
- online: require a running Web service and check non-chat /ws messages.

The online path is intentionally non-invasive: it verifies the WebSocket
endpoint and visibility filtering without sending real chat prompts that could
pollute the currently open Web UI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from zulong.core.message_visibility import is_public_payload
from task_lifecycle_feedback_regression import (
    assert_background_memory_tool_hidden,
    assert_completion_evidence_constraints,
    assert_preference_memory_reference_edge,
    assert_task_graph_binding_keeps_check_turn,
    assert_visibility_rules,
)


DEFAULT_WS_URL = "ws://127.0.0.1:8090/ws"


def _print_result(name: str, status: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def run_offline_checks() -> None:
    tmp_dir = ROOT / "tmp" / "web_full_chain_regression"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    checks = [
        ("visibility rules", assert_visibility_rules),
        ("background memory tool hidden", assert_background_memory_tool_hidden),
        ("TaskGraph check turn keeps binding", lambda: assert_task_graph_binding_keeps_check_turn(tmp_dir)),
        ("preference REFERENCE edge", assert_preference_memory_reference_edge),
        ("completion evidence gate", assert_completion_evidence_constraints),
    ]
    for name, check in checks:
        check()
        _print_result(name, "PASS")


async def _recv_for(ws: Any, seconds: float) -> List[Dict[str, Any]]:
    deadline = time.monotonic() + seconds
    messages: List[Dict[str, Any]] = []
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        except asyncio.TimeoutError:
            break
        try:
            data = json.loads(raw)
        except Exception:
            continue
        messages.append(data)
    return messages


def _visible_interaction_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    visible = []
    for message in messages:
        interaction = message.get("interaction")
        payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
        data = message.get("data") if isinstance(message.get("data"), dict) else {}
        if not interaction and isinstance(payload.get("interaction"), dict):
            interaction = payload["interaction"]
        if not interaction and isinstance(data.get("interaction"), dict):
            interaction = data["interaction"]
        if isinstance(interaction, dict) and is_public_payload({"interaction": interaction}):
            visible.append(interaction)
    return visible


def _assert_no_visible_background_memory(messages: List[Dict[str, Any]]) -> None:
    forbidden = {"recall_memory", "read_memory_node", "discover_related", "ide_get_context"}
    for interaction in _visible_interaction_messages(messages):
        tool_name = str(interaction.get("tool_name") or interaction.get("raw_details", {}).get("tool_name") or "")
        if any(name in tool_name for name in forbidden):
            raise AssertionError(f"background memory tool leaked to Web UI: {tool_name}")


async def run_online_smoke(ws_url: str, timeout: float) -> None:
    try:
        import websockets
    except Exception as exc:
        raise RuntimeError(f"websockets package unavailable: {exc}") from exc

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
        all_messages: List[Dict[str, Any]] = []
        initial = await _recv_for(ws, 1.0)
        all_messages.extend(initial)

        await ws.send(json.dumps({"type": "ping"}, ensure_ascii=False))
        ping_messages = await _recv_for(ws, min(timeout, 2.0))
        all_messages.extend(ping_messages)
        if not any(msg.get("type") == "pong" for msg in ping_messages):
            raise AssertionError("Web /ws did not respond to ping")
        _print_result("online ping", "PASS", "pong received")

        await ws.send(json.dumps({"type": "LIST_DIALOGUE_SESSIONS"}, ensure_ascii=False))
        session_messages = await _recv_for(ws, min(timeout, 3.0))
        all_messages.extend(session_messages)
        if not any(msg.get("type") in {"SESSION_LIST", "session:list"} for msg in session_messages):
            raise AssertionError("Web /ws did not return session list")
        _print_result("online session list", "PASS", "SESSION_LIST received")

        _assert_no_visible_background_memory(all_messages)
        _print_result("online hidden background tools", "PASS")


async def _can_connect(ws_url: str) -> bool:
    try:
        import websockets

        async with websockets.connect(ws_url, open_timeout=2, ping_interval=None) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            await _recv_for(ws, 0.5)
            return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "offline", "online"], default="auto")
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--timeout", type=float, default=6.0)
    args = parser.parse_args()

    if args.mode == "offline":
        run_offline_checks()
        print("web_full_chain_regression: offline ok")
        return

    if args.mode == "online":
        asyncio.run(run_online_smoke(args.ws_url, args.timeout))
        print("web_full_chain_regression: online ok")
        return

    if asyncio.run(_can_connect(args.ws_url)):
        asyncio.run(run_online_smoke(args.ws_url, args.timeout))
        print("web_full_chain_regression: online ok")
    else:
        _print_result("online WebSocket", "SKIP", f"{args.ws_url} is not reachable")
        run_offline_checks()
        print("web_full_chain_regression: offline ok")


if __name__ == "__main__":
    main()
