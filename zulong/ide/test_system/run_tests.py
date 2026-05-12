"""独立启动脚本：直接运行祖龙测试系统并进行自动化测试验证

无需依赖 ide_server，使用模拟后端代理运行完整测试流程。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


class MockBackendProxy:
    """模拟后端代理，无需真实 ide_server"""

    def __init__(self):
        self._connected = False
        self._call_log: list[dict] = []

    async def connect(self):
        self._connected = True
        logger.info("MockBackendProxy: 已连接")

    async def disconnect(self):
        self._connected = False

    async def start_session(self, config: dict) -> dict:
        self._call_log.append({"method": "start_session", "config": config})
        return {"type": "session_ack", "session_id": f"mock_sess_{uuid.uuid4().hex[:6]}"}

    async def execute_tool(self, tool_name: str, tool_input: dict, timeout: int = 300) -> dict:
        self._call_log.append({"method": "execute_tool", "tool": tool_name, "input": tool_input})
        await asyncio.sleep(0.3)
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
        return {"status": "success", "output": f"mock_output_{tool_name}", "files_count": 10, "summary": "发现 3 个问题"}

    async def cancel(self):
        self._call_log.append({"method": "cancel"})

    async def resume(self, snapshot: dict) -> dict:
        self._call_log.append({"method": "resume", "snapshot_id": snapshot.get("snapshot_id")})
        return {"type": "session_ack", "resumed": True}

    @property
    def connected(self):
        return self._connected


async def run_test_system():
    from test_system.app import TestSystemApp
    from test_system.definition.loader import TestCaseLoader
    from test_system.definition.enums import TestStatus
    from test_system.engine.executor import TestExecutor
    from test_system.engine.attention_simulator import AttentionSimulator
    from test_system.integration.event_bridge import EventBridge, TestEvent

    logger.info("=" * 60)
    logger.info("祖龙复杂任务测试系统 - 自动化验证启动")
    logger.info("=" * 60)

    cases_dir = BASE_DIR / "test_cases"
    db_path = str(BASE_DIR / "test_system_test.db")

    store = None
    try:
        from test_system.storage.result_store import TestResultStore
        from test_system.storage.log_store import TestLogStore
        from test_system.engine.progress_tracker import ProgressTracker, StagnationDetector
        from test_system.engine.snapshot_manager import SnapshotManager

        store = TestResultStore(db_path)
        await store.initialize()
        log_store = TestLogStore()

        bridge = EventBridge()

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

        attention = AttentionSimulator(mock_proxy, bridge, log_store)

        loader = TestCaseLoader(cases_dir)
        test_cases = loader.load_all()
        logger.info("加载了 %d 个测试用例", len(test_cases))

        for tc in test_cases:
            await store.save_test_case(tc)
        logger.info("测试用例已存入数据库")

        results = {}

        # ===== 测试1: 复杂任务执行 =====
        logger.info("-" * 40)
        logger.info("测试1: 复杂任务执行")
        tc_complex = next((tc for tc in test_cases if tc.test_case_id == "test_complex_001"), None)
        if tc_complex:
            run = await executor.start_test(tc_complex)
            await asyncio.sleep(2)
            final_run = executor.get_run(run.execution_id)
            if final_run:
                results["complex_task"] = {
                    "status": final_run.status.value,
                    "steps_completed": sum(1 for s in final_run.steps if s.status.value == "completed"),
                    "steps_total": len(final_run.steps),
                    "progress_reports": len(final_run.progress_reports),
                }
                logger.info("复杂任务结果: %s", results["complex_task"])

        # ===== 测试2: 中断与恢复 =====
        logger.info("-" * 40)
        logger.info("测试2: 中断与恢复")
        tc_interrupt = next((tc for tc in test_cases if tc.test_case_id == "test_interrupt_001"), None)
        if tc_interrupt:
            run = await executor.start_test(tc_interrupt)
            await asyncio.sleep(0.8)
            interrupt = await executor.stop_test(run.execution_id)
            logger.info("中断点: step=%s, reason=%s, snapshot=%s",
                        interrupt.step_id if interrupt else "N/A",
                        interrupt.reason if interrupt else "N/A",
                        interrupt.snapshot_id if interrupt else "N/A")
            await asyncio.sleep(0.5)

            resumed_run = await executor.resume_test(run.execution_id)
            await asyncio.sleep(2)
            final_run = executor.get_run(run.execution_id)
            if final_run:
                results["interrupt_resume"] = {
                    "status": final_run.status.value,
                    "was_interrupted": final_run.interrupt_point is not None,
                    "snapshot_id": final_run.interrupt_point.snapshot_id if final_run.interrupt_point else None,
                    "steps_completed": sum(1 for s in final_run.steps if s.status.value == "completed"),
                }
                logger.info("中断恢复结果: %s", results["interrupt_resume"])

        # ===== 测试3: 长任务进度汇报 =====
        logger.info("-" * 40)
        logger.info("测试3: 长任务进度汇报")
        tc_long = next((tc for tc in test_cases if tc.test_case_id == "test_longtask_001"), None)
        if tc_long:
            run = await executor.start_test(tc_long)
            await asyncio.sleep(2)
            final_run = executor.get_run(run.execution_id)
            if final_run:
                results["long_task_report"] = {
                    "status": final_run.status.value,
                    "progress_reports_count": len(final_run.progress_reports),
                    "last_progress": final_run.progress_reports[-1].to_dict() if final_run.progress_reports else None,
                    "steps_completed": sum(1 for s in final_run.steps if s.status.value == "completed"),
                }
                logger.info("长任务汇报结果: %s", results["long_task_report"])

        # ===== 测试4: 注意力切换 =====
        logger.info("-" * 40)
        logger.info("测试4: 注意力切换")
        session_ids = await attention.start_parallel_tasks([
            {"name": "task_a", "attention_window_data": {"scope": "src/"}},
            {"name": "task_b", "attention_window_data": {"scope": "tests/"}},
        ])
        logger.info("并行任务启动: %s", session_ids)

        if len(session_ids) >= 2:
            switch_event = await attention.switch_attention(session_ids[0], session_ids[1], "exec_attention")
            results["attention_switch"] = {
                "from": switch_event.from_session_id,
                "to": switch_event.to_session_id,
                "restore_status": switch_event.restore_status.value,
                "elapsed_ms": switch_event.elapsed_ms,
            }
            logger.info("注意力切换结果: %s", results["attention_switch"])

            switch_back = await attention.switch_attention(session_ids[1], session_ids[0], "exec_attention")
            results["attention_switch_back"] = {
                "restore_status": switch_back.restore_status.value,
                "elapsed_ms": switch_back.elapsed_ms,
            }
            logger.info("注意力切回结果: %s", results["attention_switch_back"])

        # ===== 测试5: 状态快照完整性 =====
        logger.info("-" * 40)
        logger.info("测试5: 状态快照完整性")
        snap = await snapshot_mgr.capture("test_exec", "step_3", {"phase": "running", "fc_turn": 15})
        loaded_snap = await snapshot_mgr.load(snap.snapshot_id)
        results["snapshot_integrity"] = {
            "snapshot_id": snap.snapshot_id,
            "hash_match": snap.integrity_hash == loaded_snap.integrity_hash,
            "verified": loaded_snap.verify_integrity(),
        }
        logger.info("快照完整性结果: %s", results["snapshot_integrity"])

        # ===== 汇总 =====
        logger.info("=" * 60)
        logger.info("测试验证汇总")
        logger.info("=" * 60)
        all_passed = True
        for name, result in results.items():
            status = "PASS" if result else "FAIL"
            if name == "complex_task" and result.get("steps_completed", 0) < 1:
                status = "FAIL"
                all_passed = False
            if name == "interrupt_resume" and not result.get("was_interrupted"):
                status = "FAIL"
                all_passed = False
            if name == "snapshot_integrity" and not result.get("verified"):
                status = "FAIL"
                all_passed = False
            logger.info("  [%s] %s: %s", status, name, json.dumps(result, ensure_ascii=False, default=str))

        logger.info("=" * 60)
        logger.info("总体结果: %s", "ALL PASSED" if all_passed else "SOME FAILED")
        logger.info("=" * 60)

        # 日志查看
        recent_logs = await log_store.query(page=1, size=5)
        logger.info("最近5条日志:")
        for log_entry in recent_logs:
            logger.info("  [%s] %s: %s", log_entry["level"], log_entry["source"], log_entry["message"][:80])

        # 历史记录查询
        history = await store.get_history(days=1)
        logger.info("历史执行记录: %d 条", len(history))

        return all_passed

    finally:
        if store:
            await store.close()


async def start_webui():
    """启动前端开发服务器"""
    import subprocess
    webui_dir = BASE_DIR / "webui"
    if (webui_dir / "node_modules").exists():
        logger.info("启动前端开发服务器: http://127.0.0.1:5180")
        proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", "5180"],
            cwd=str(webui_dir / "dist"),
        )
        return proc
    return None


async def main():
    passed = await run_test_system()

    logger.info("")
    logger.info("提示: 启动Web前端查看实时效果:")
    logger.info("  cd zulong/ide/test_system/webui && npm run dev")
    logger.info("  访问 http://127.0.0.1:5180")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
