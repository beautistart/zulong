"""启动测试系统后端服务 + 自动执行所有测试用例

同时提供:
- WebSocket 推送服务 (ws://127.0.0.1:8091)
- REST API (http://127.0.0.1:8092/api/test/*)
- 静态文件服务 (http://127.0.0.1:8092/ → webui/dist)
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_system.definition.loader import TestCaseLoader
from test_system.definition.enums import TestStatus
from test_system.engine.executor import TestExecutor
from test_system.engine.models import TestRun, StepResult, ProgressReport
from test_system.engine.progress_tracker import ProgressTracker, StagnationDetector
from test_system.engine.snapshot_manager import SnapshotManager
from test_system.engine.attention_simulator import AttentionSimulator
from test_system.integration.event_bridge import EventBridge, TestEvent
from test_system.storage.result_store import TestResultStore
from test_system.storage.log_store import TestLogStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
WEBUI_DIR = BASE_DIR / "webui" / "dist"


class MockBackendProxy:
    def __init__(self):
        self._connected = False

    async def connect(self):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    async def start_session(self, config: dict) -> dict:
        return {"type": "session_ack", "session_id": f"mock_{uuid.uuid4().hex[:6]}"}

    async def execute_tool(self, tool_name: str, tool_input: dict, timeout: int = 300) -> dict:
        await asyncio.sleep(0.5)
        if tool_name == "switch_attention":
            return {"restore_status": "restored", "context": {"task": tool_input.get("to_task")}}
        if tool_name == "verify_context":
            return {"context_intact": True, "verified_fields": ["attention_window_data"]}
        if tool_name == "verify_result":
            return {"resumed": True, "steps_verified": tool_input.get("expected_steps", [])}
        if tool_name == "monitor_progress":
            return {"periodic_reports_received": True, "report_count": 5}
        if tool_name == "verify_completion":
            return {"completed": True, "status": "success"}
        return {"status": "success", "output": f"mock_{tool_name}", "files_count": 10, "summary": "发现 3 个问题"}

    async def cancel(self):
        pass

    async def resume(self, snapshot: dict) -> dict:
        return {"type": "session_ack", "resumed": True}

    @property
    def connected(self):
        return self._connected


app = FastAPI(title="祖龙测试系统")

executor: TestExecutor | None = None
bridge: EventBridge | None = None
log_store: TestLogStore | None = None
store: TestResultStore | None = None
ws_clients: list[WebSocket] = []


async def broadcast_to_ws(message: dict):
    raw = json.dumps(message, ensure_ascii=False, default=str)
    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_text(raw)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_clients.remove(ws)


async def on_test_event(event: TestEvent, data: dict):
    msg = {"type": event.value, "payload": data, "timestamp": datetime.now().isoformat()}
    await broadcast_to_ws(msg)


@app.on_event("startup")
async def startup():
    global executor, bridge, log_store, store

    cases_dir = BASE_DIR / "test_cases"
    db_path = str(BASE_DIR / "test_system_live.db")

    store = TestResultStore(db_path)
    await store.initialize()
    log_store = TestLogStore()

    bridge = EventBridge()
    bridge.subscribe(on_test_event)

    mock_proxy = MockBackendProxy()
    await mock_proxy.connect()

    progress = ProgressTracker(bridge, StagnationDetector())
    snapshot_mgr = SnapshotManager(store, mock_proxy)

    executor = TestExecutor(
        store=store,
        log_store=log_store,
        backend_proxy=mock_proxy,
        event_bridge=bridge,
        progress_tracker=progress,
        snapshot_manager=snapshot_mgr,
    )

    loader = TestCaseLoader(cases_dir)
    test_cases = loader.load_all()
    for tc in test_cases:
        await store.save_test_case(tc)
    logger.info("已加载 %d 个测试用例", len(test_cases))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    logger.info("WebSocket客户端连接, 当前连接数: %d", len(ws_clients))

    welcome_cases = []
    if store:
        cases = await store.list_test_cases(page=1, size=100)
        for c in cases:
            welcome_cases.append(c)

    active = []
    if executor:
        for run in executor.get_active_runs():
            active.append(run.to_dict())

    await ws.send_text(json.dumps({
        "type": "TEST_WELCOME",
        "payload": {"test_cases": welcome_cases, "active_executions": active},
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False, default=str))

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        logger.info("WebSocket客户端断开, 当前连接数: %d", len(ws_clients))


@app.post("/api/test/execute")
async def execute_test(test_case_id: str):
    if not executor or not store:
        return JSONResponse({"error": "系统未就绪"}, status_code=503)
    tc = await store.get_test_case(test_case_id)
    if not tc:
        return JSONResponse({"error": f"用例不存在: {test_case_id}"}, status_code=404)
    run = await executor.start_test(tc)
    return run.to_dict()


@app.post("/api/test/execute/{execution_id}/stop")
async def stop_test(execution_id: str):
    if not executor:
        return JSONResponse({"error": "系统未就绪"}, status_code=503)
    interrupt = await executor.stop_test(execution_id)
    if not interrupt:
        return JSONResponse({"error": f"未找到运行中的测试: {execution_id}"}, status_code=404)
    return {"execution_id": execution_id, "interrupt_point": interrupt.to_dict()}


@app.post("/api/test/execute/{execution_id}/resume")
async def resume_test(execution_id: str):
    if not executor:
        return JSONResponse({"error": "系统未就绪"}, status_code=503)
    try:
        run = await executor.resume_test(execution_id)
        return run.to_dict()
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/test/cases")
async def list_cases():
    if not store:
        return {"items": []}
    cases = await store.list_test_cases(page=1, size=100)
    return {"items": cases}


@app.get("/api/test/execute/{execution_id}")
async def get_execution(execution_id: str):
    if not executor:
        return JSONResponse({"error": "系统未就绪"}, status_code=503)
    run = executor.get_run(execution_id)
    if run:
        return run.to_dict()
    return JSONResponse({"error": "未找到"}, status_code=404)


@app.get("/api/test/history")
async def get_history(days: int = 30):
    if not store:
        return {"items": []}
    history = await store.get_history(days=days)
    return {"items": history}


@app.get("/api/test/logs")
async def get_logs(execution_id: str = None, level: str = None):
    if not log_store:
        return {"items": []}
    logs = await log_store.query(execution_id=execution_id, level=level, page=1, size=200)
    return {"items": logs}


@app.post("/api/test/run-all")
async def run_all_tests():
    if not executor or not store:
        return JSONResponse({"error": "系统未就绪"}, status_code=503)
    cases = await store.list_test_cases(page=1, size=100)
    results = []
    for c in cases:
        tc = await store.get_test_case(c["test_case_id"])
        if tc:
            run = await executor.start_test(tc)
            results.append({"test_case_id": tc.test_case_id, "execution_id": run.execution_id, "status": "started"})
    return {"results": results}


if WEBUI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEBUI_DIR), html=True), name="webui")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8092, log_level="info")
