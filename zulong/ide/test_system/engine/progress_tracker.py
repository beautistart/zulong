from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from test_system.engine.models import ProgressReport, TestRun
from test_system.definition.enums import ProgressMode, TestStatus
from test_system.integration.event_bridge import EventBridge, TestEvent

logger = logging.getLogger(__name__)


class ProgressCalculator:
    @staticmethod
    def exact(completed_steps: int, total_steps: int) -> float:
        if total_steps == 0:
            return 0.0
        return min(100.0, (completed_steps / total_steps) * 100.0)

    @staticmethod
    def estimated(fc_turn: int, estimated_total: int) -> float:
        if estimated_total == 0:
            return 0.0
        return min(100.0, (fc_turn / estimated_total) * 100.0)


class StagnationDetector:
    def __init__(self, window_size: int = 3):
        self._window_size = window_size
        self._history: list[int] = []

    def check(self, completed_steps: int) -> bool:
        self._history.append(completed_steps)
        if len(self._history) < self._window_size:
            return False
        recent = self._history[-self._window_size :]
        return len(set(recent)) == 1

    def reset(self) -> None:
        self._history.clear()


class ProgressTracker:
    def __init__(self, event_bridge: EventBridge, stagnation_detector: Optional[StagnationDetector] = None):
        self._bridge = event_bridge
        self._stagnation_detector = stagnation_detector or StagnationDetector()
        self._pending_cache: list[ProgressReport] = []

    def compute_progress(
        self,
        execution_id: str,
        completed_steps: int,
        total_steps: int,
        stage_description: str,
        mode: ProgressMode = ProgressMode.EXACT,
        fc_turn: int = 0,
        estimated_total: int = 0,
        intermediate_result: object = None,
    ) -> ProgressReport:
        if mode == ProgressMode.EXACT:
            percent = ProgressCalculator.exact(completed_steps, total_steps)
        else:
            percent = ProgressCalculator.estimated(fc_turn, estimated_total)

        report = ProgressReport(
            execution_id=execution_id,
            progress_percent=round(percent, 2),
            mode=mode,
            stage_description=stage_description,
            completed_steps=completed_steps,
            total_steps=total_steps,
            timestamp=datetime.now().isoformat(),
            intermediate_result=intermediate_result,
        )
        return report

    async def emit_progress(self, report: ProgressReport) -> None:
        await self._bridge.emit(TestEvent.PROGRESS_UPDATE, report.to_dict())

    async def check_stagnation(self, completed_steps: int, execution_id: str) -> bool:
        is_stagnant = self._stagnation_detector.check(completed_steps)
        if is_stagnant:
            await self._bridge.emit(TestEvent.STAGNATION_ALERT, {
                "execution_id": execution_id,
                "completed_steps": completed_steps,
                "message": "进度停滞：连续多个报告无新步骤完成",
                "timestamp": datetime.now().isoformat(),
            })
        return is_stagnant

    def cache_report(self, report: ProgressReport) -> None:
        self._pending_cache.append(report)

    async def flush_cache(self) -> None:
        for report in self._pending_cache:
            await self.emit_progress(report)
        self._pending_cache.clear()
