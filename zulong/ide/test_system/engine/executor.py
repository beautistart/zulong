from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from test_system.definition.enums import AssertionType, StepStatus, TestStatus
from test_system.definition.models import StepDefinition, TestCaseDefinition
from test_system.definition.resolver import StepDependencyResolver
from test_system.engine.models import (
    AssertionResult,
    InterruptPoint,
    ProgressReport,
    StateSnapshot,
    StepResult,
    TestRun,
)
from test_system.engine.progress_tracker import ProgressTracker
from test_system.engine.snapshot_manager import SnapshotManager
from test_system.integration.backend_proxy import BackendProxy
from test_system.integration.event_bridge import EventBridge, TestEvent
from test_system.storage.log_store import TestLogStore
from test_system.storage.result_store import TestResultStore

logger = logging.getLogger(__name__)


class StepRunner:
    def __init__(self, backend_proxy: BackendProxy, log_store: TestLogStore):
        self._proxy = backend_proxy
        self._log = log_store

    async def execute(
        self,
        step: StepDefinition,
        step_input: dict,
        execution_id: str,
    ) -> StepResult:
        started_at = datetime.now().isoformat()
        start_ms = int(time.time() * 1000)
        self._log.log(execution_id, "INFO", "StepRunner", f"步骤开始: {step.step_id} ({step.name})")

        step_result = StepResult(
            step_id=step.step_id,
            name=step.name,
            status=StepStatus.RUNNING,
            started_at=started_at,
        )

        try:
            output = await asyncio.wait_for(
                self._proxy.execute_tool(step.tool, step_input),
                timeout=step.timeout_seconds,
            )
            step_result.output = output
            step_result.status = StepStatus.COMPLETED

            step_result.assertion_results = self._run_assertions(step.assertions, output)

            if any(not ar.passed for ar in step_result.assertion_results):
                step_result.status = StepStatus.FAILED
                self._log.log(execution_id, "WARN", "StepRunner", f"步骤断言失败: {step.step_id}")

        except asyncio.TimeoutError:
            step_result.status = StepStatus.FAILED
            step_result.error = f"步骤超时({step.timeout_seconds}s)"
            self._log.log(execution_id, "ERROR", "StepRunner", f"步骤超时: {step.step_id}")
        except asyncio.CancelledError:
            step_result.status = StepStatus.INTERRUPTED
            step_result.error = "步骤被中断"
            self._log.log(execution_id, "INFO", "StepRunner", f"步骤中断: {step.step_id}")
        except Exception as e:
            step_result.status = StepStatus.FAILED
            step_result.error = str(e)
            self._log.log(execution_id, "ERROR", "StepRunner", f"步骤异常: {step.step_id} - {e}")

        step_result.completed_at = datetime.now().isoformat()
        step_result.duration_ms = int(time.time() * 1000) - start_ms
        return step_result

    @staticmethod
    def _run_assertions(assertions, output: Any) -> list[AssertionResult]:
        results = []
        for a in assertions:
            actual = output
            if isinstance(output, dict) and a.field_path:
                for part in a.field_path.split("."):
                    actual = actual.get(part, None) if isinstance(actual, dict) else None

            passed = False
            if a.assertion_type == AssertionType.VALUE_MATCH:
                passed = actual == a.expected
            elif a.assertion_type == AssertionType.STATUS_MATCH:
                passed = str(actual) == str(a.expected)
            elif a.assertion_type == AssertionType.PATTERN_MATCH:
                passed = bool(re.search(str(a.expected), str(actual))) if actual is not None else False

            results.append(AssertionResult(
                assertion_type=a.assertion_type,
                field_path=a.field_path,
                expected=a.expected,
                actual=actual,
                passed=passed,
                message=a.message,
            ))
        return results


class TestExecutor:
    def __init__(
        self,
        store: TestResultStore,
        log_store: TestLogStore,
        backend_proxy: BackendProxy,
        event_bridge: EventBridge,
        progress_tracker: ProgressTracker,
        snapshot_manager: SnapshotManager,
    ):
        self._store = store
        self._log = log_store
        self._proxy = backend_proxy
        self._bridge = event_bridge
        self._progress = progress_tracker
        self._snapshot_mgr = snapshot_manager
        self._step_runner = StepRunner(backend_proxy, log_store)
        self._runs: dict[str, TestRun] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._resume_locks: dict[str, asyncio.Lock] = {}

    async def start_test(self, test_case: TestCaseDefinition, config_overrides: Optional[dict] = None) -> TestRun:
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        resolver = StepDependencyResolver(test_case.steps)
        execution_order = resolver.get_execution_order()

        run = TestRun(
            execution_id=execution_id,
            test_case_id=test_case.test_case_id,
            name=test_case.name,
            type=test_case.type,
            status=TestStatus.INITIALIZING,
            started_at=datetime.now().isoformat(),
        )
        self._runs[execution_id] = run
        self._cancel_events[execution_id] = asyncio.Event()
        await self._store.save_execution(run)

        await self._bridge.emit(TestEvent.TEST_STARTED, {
            "execution_id": execution_id,
            "test_case_id": test_case.test_case_id,
            "name": test_case.name,
            "type": test_case.type.value,
        })

        run.status = TestStatus.RUNNING
        await self._store.update_execution_status(execution_id, TestStatus.RUNNING.value)

        asyncio.create_task(self._execute_steps(execution_id, test_case, execution_order, resolver, run))
        return run

    async def _execute_steps(self, execution_id: str, test_case: TestCaseDefinition, execution_order: list[StepDefinition], resolver: StepDependencyResolver, run: TestRun) -> None:
        step_outputs: dict[str, Any] = {}
        try:
            for i, step in enumerate(execution_order):
                if self._cancel_events.get(execution_id, asyncio.Event()).is_set():
                    await self._trigger_interrupt(execution_id, run, step.step_id, "用户中断")
                    return

                if step.condition:
                    try:
                        if not resolver.evaluate_condition(step.condition, step_outputs):
                            skipped = StepResult(step_id=step.step_id, name=step.name, status=StepStatus.SKIPPED)
                            run.steps.append(skipped)
                            continue
                    except Exception as e:
                        self._log.log(execution_id, "WARN", "TestExecutor", f"条件求值失败: {e}")

                resolved_input = resolver.resolve_step_input(step.tool_input, step_outputs)

                await self._bridge.emit(TestEvent.STEP_STARTED, {
                    "execution_id": execution_id,
                    "step_id": step.step_id,
                    "name": step.name,
                    "step_index": i + 1,
                    "total_steps": len(execution_order),
                })

                step_result = await self._step_runner.execute(step, resolved_input, execution_id)
                run.steps.append(step_result)
                await self._store.save_step_result(execution_id, step_result)
                step_outputs[step.step_id] = step_result.output

                if step_result.status == StepStatus.COMPLETED:
                    await self._bridge.emit(TestEvent.STEP_COMPLETED, step_result.to_dict())
                elif step_result.status == StepStatus.FAILED:
                    await self._bridge.emit(TestEvent.STEP_FAILED, step_result.to_dict())
                    if test_case.config.on_step_fail == "abort":
                        run.status = TestStatus.FAILED
                        run.error = f"步骤 {step.step_id} 失败，中止执行"
                        run.completed_at = datetime.now().isoformat()
                        await self._store.update_execution_status(execution_id, TestStatus.FAILED.value, run.completed_at, run.error)
                        await self._bridge.emit(TestEvent.TEST_COMPLETED, run.to_dict())
                        return

                report = self._progress.compute_progress(
                    execution_id=execution_id,
                    completed_steps=sum(1 for s in run.steps if s.status == StepStatus.COMPLETED),
                    total_steps=len(execution_order),
                    stage_description=f"步骤 {step.step_id} 完成",
                )
                run.progress_reports.append(report)
                await self._progress.emit_progress(report)
                await self._progress.check_stagnation(report.completed_steps, execution_id)

            run.status = TestStatus.COMPLETED
            run.completed_at = datetime.now().isoformat()
            await self._store.update_execution_status(execution_id, TestStatus.COMPLETED.value, run.completed_at)
            await self._bridge.emit(TestEvent.TEST_COMPLETED, run.to_dict())

        except Exception as e:
            run.status = TestStatus.FAILED
            run.error = str(e)
            run.completed_at = datetime.now().isoformat()
            await self._store.update_execution_status(execution_id, TestStatus.FAILED.value, run.completed_at, str(e))
            await self._bridge.emit(TestEvent.TEST_COMPLETED, run.to_dict())
            self._log.log(execution_id, "ERROR", "TestExecutor", f"执行异常: {e}")

    async def stop_test(self, execution_id: str) -> Optional[InterruptPoint]:
        cancel_event = self._cancel_events.get(execution_id)
        if cancel_event:
            cancel_event.set()
        run = self._runs.get(execution_id)
        if run and run.status == TestStatus.RUNNING:
            current_step = next((s.step_id for s in reversed(run.steps) if s.status == StepStatus.RUNNING), "unknown")
            return await self._trigger_interrupt(execution_id, run, current_step, "用户中断")
        return None

    async def _trigger_interrupt(self, execution_id: str, run: TestRun, step_id: str, reason: str) -> InterruptPoint:
        await self._proxy.cancel()

        snapshot = None
        if run.interrupt_point is None:
            snapshot = await self._snapshot_mgr.capture(execution_id, step_id, {"status": "interrupted", "steps_completed": [s.step_id for s in run.steps if s.status == StepStatus.COMPLETED]})

        interrupt = InterruptPoint(step_id=step_id, reason=reason, snapshot_id=snapshot.snapshot_id if snapshot else None)
        run.interrupt_point = interrupt
        run.status = TestStatus.INTERRUPTED
        await self._store.update_execution_status(execution_id, TestStatus.INTERRUPTED.value)

        await self._bridge.emit(TestEvent.INTERRUPT_TRIGGERED, interrupt.to_dict())
        self._log.log(execution_id, "INFO", "TestExecutor", f"测试中断: {reason} (step={step_id})")
        return interrupt

    async def resume_test(self, execution_id: str, snapshot_id: Optional[str] = None) -> TestRun:
        if execution_id not in self._resume_locks:
            self._resume_locks[execution_id] = asyncio.Lock()

        async with self._resume_locks[execution_id]:
            run = self._runs.get(execution_id)
            if not run or run.status != TestStatus.INTERRUPTED:
                raise ValueError(f"无法恢复: 执行 {execution_id} 状态非INTERRUPTED")

            if run.interrupt_point and run.interrupt_point.snapshot_id:
                try:
                    snapshot = await self._snapshot_mgr.load(run.interrupt_point.snapshot_id)
                    await self._proxy.resume(snapshot.state_data)
                except Exception as e:
                    self._log.log(execution_id, "ERROR", "TestExecutor", f"快照恢复失败: {e}")

            run.status = TestStatus.RUNNING
            await self._store.update_execution_status(execution_id, TestStatus.RUNNING.value)
            await self._bridge.emit(TestEvent.RESUME_STARTED, {"execution_id": execution_id, "snapshot_id": snapshot_id})

            self._cancel_events[execution_id] = asyncio.Event()
            self._log.log(execution_id, "INFO", "TestExecutor", f"测试恢复: {execution_id}")
            await self._bridge.emit(TestEvent.RESUME_COMPLETED, {"execution_id": execution_id})
            return run

    def get_run(self, execution_id: str) -> Optional[TestRun]:
        return self._runs.get(execution_id)

    def get_active_runs(self) -> list[TestRun]:
        return [r for r in self._runs.values() if r.status in (TestStatus.RUNNING, TestStatus.INITIALIZING)]
