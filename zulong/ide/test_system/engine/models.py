from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from test_system.definition.enums import (
    AssertionType,
    AttentionRestoreStatus,
    ProgressMode,
    StepStatus,
    TestStatus,
    TestCaseType,
)


@dataclass
class AssertionResult:
    assertion_type: AssertionType
    field_path: str
    expected: Any
    actual: Any
    passed: bool
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "assertion_type": self.assertion_type.value,
            "field_path": self.field_path,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AssertionResult:
        return cls(
            assertion_type=AssertionType(data["assertion_type"]),
            field_path=data["field_path"],
            expected=data["expected"],
            actual=data["actual"],
            passed=data["passed"],
            message=data.get("message", ""),
        )


@dataclass
class StepResult:
    step_id: str
    name: str
    status: StepStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    output: Any = None
    assertion_results: list[AssertionResult] = field(default_factory=list)
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "output": self.output,
            "assertion_results": [a.to_dict() for a in self.assertion_results],
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StepResult:
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            status=StepStatus(data["status"]),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            output=data.get("output"),
            assertion_results=[
                AssertionResult.from_dict(a) for a in data.get("assertion_results", [])
            ],
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
        )


@dataclass
class InterruptPoint:
    step_id: str
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    snapshot_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "snapshot_id": self.snapshot_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InterruptPoint:
        return cls(
            step_id=data["step_id"],
            reason=data["reason"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            snapshot_id=data.get("snapshot_id"),
        )


@dataclass
class StateSnapshot:
    snapshot_id: str
    execution_id: str
    step_id: str
    state_data: dict
    integrity_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def compute_hash(self) -> str:
        import hashlib
        import json

        canonical = json.dumps(self.state_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify_integrity(self) -> bool:
        return self.compute_hash() == self.integrity_hash

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "execution_id": self.execution_id,
            "step_id": self.step_id,
            "state_data": self.state_data,
            "integrity_hash": self.integrity_hash,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StateSnapshot:
        return cls(
            snapshot_id=data["snapshot_id"],
            execution_id=data["execution_id"],
            step_id=data["step_id"],
            state_data=data["state_data"],
            integrity_hash=data["integrity_hash"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class ProgressReport:
    execution_id: str
    progress_percent: float
    mode: ProgressMode
    stage_description: str
    completed_steps: int
    total_steps: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    intermediate_result: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "progress_percent": self.progress_percent,
            "mode": self.mode.value,
            "stage_description": self.stage_description,
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp,
            "intermediate_result": self.intermediate_result,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProgressReport:
        return cls(
            execution_id=data["execution_id"],
            progress_percent=data["progress_percent"],
            mode=ProgressMode(data["mode"]),
            stage_description=data["stage_description"],
            completed_steps=data["completed_steps"],
            total_steps=data["total_steps"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            intermediate_result=data.get("intermediate_result"),
        )


@dataclass
class AttentionContext:
    session_id: str
    attention_window_data: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "attention_window_data": self.attention_window_data,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttentionContext:
        return cls(
            session_id=data["session_id"],
            attention_window_data=data["attention_window_data"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class AttentionSwitchEvent:
    from_session_id: str
    to_session_id: str
    restore_status: AttentionRestoreStatus
    elapsed_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "from_session_id": self.from_session_id,
            "to_session_id": self.to_session_id,
            "restore_status": self.restore_status.value,
            "elapsed_ms": self.elapsed_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AttentionSwitchEvent:
        return cls(
            from_session_id=data["from_session_id"],
            to_session_id=data["to_session_id"],
            restore_status=AttentionRestoreStatus(data["restore_status"]),
            elapsed_ms=data["elapsed_ms"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class TestRun:
    execution_id: str
    test_case_id: str
    name: str
    type: TestCaseType
    status: TestStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    steps: list[StepResult] = field(default_factory=list)
    interrupt_point: Optional[InterruptPoint] = None
    progress_reports: list[ProgressReport] = field(default_factory=list)
    attention_switches: list[AttentionSwitchEvent] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "test_case_id": self.test_case_id,
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps": [s.to_dict() for s in self.steps],
            "interrupt_point": self.interrupt_point.to_dict() if self.interrupt_point else None,
            "progress_reports": [p.to_dict() for p in self.progress_reports],
            "attention_switches": [a.to_dict() for a in self.attention_switches],
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TestRun:
        return cls(
            execution_id=data["execution_id"],
            test_case_id=data["test_case_id"],
            name=data["name"],
            type=TestCaseType(data["type"]),
            status=TestStatus(data["status"]),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            steps=[StepResult.from_dict(s) for s in data.get("steps", [])],
            interrupt_point=InterruptPoint.from_dict(data["interrupt_point"]) if data.get("interrupt_point") else None,
            progress_reports=[ProgressReport.from_dict(p) for p in data.get("progress_reports", [])],
            attention_switches=[AttentionSwitchEvent.from_dict(a) for a in data.get("attention_switches", [])],
            error=data.get("error"),
        )
