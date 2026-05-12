from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .enums import AssertionType, TestCaseType


@dataclass
class AssertionDef:
    assertion_type: AssertionType
    field_path: str
    expected: Any
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "assertion_type": self.assertion_type.value,
            "field_path": self.field_path,
            "expected": self.expected,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AssertionDef:
        return cls(
            assertion_type=AssertionType(data["assertion_type"]),
            field_path=data["field_path"],
            expected=data["expected"],
            message=data.get("message", ""),
        )


@dataclass
class StepDefinition:
    step_id: str
    name: str
    tool: str
    tool_input: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    condition: Optional[str] = None
    timeout_seconds: int = 300
    assertions: list[AssertionDef] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "tool": self.tool,
            "tool_input": self.tool_input,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "timeout_seconds": self.timeout_seconds,
            "assertions": [a.to_dict() for a in self.assertions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> StepDefinition:
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            tool=data["tool"],
            tool_input=data.get("tool_input", {}),
            depends_on=data.get("depends_on", []),
            condition=data.get("condition"),
            timeout_seconds=data.get("timeout_seconds", 300),
            assertions=[AssertionDef.from_dict(a) for a in data.get("assertions", [])],
        )


@dataclass
class TestConfig:
    timeout_seconds: int = 3600
    on_step_fail: str = "abort"
    progress_interval_turns: int = 10
    max_renewals: int = 5
    snapshot_on_interrupt: bool = True

    def to_dict(self) -> dict:
        return {
            "timeout_seconds": self.timeout_seconds,
            "on_step_fail": self.on_step_fail,
            "progress_interval_turns": self.progress_interval_turns,
            "max_renewals": self.max_renewals,
            "snapshot_on_interrupt": self.snapshot_on_interrupt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TestConfig:
        return cls(
            timeout_seconds=data.get("timeout_seconds", 3600),
            on_step_fail=data.get("on_step_fail", "abort"),
            progress_interval_turns=data.get("progress_interval_turns", 10),
            max_renewals=data.get("max_renewals", 5),
            snapshot_on_interrupt=data.get("snapshot_on_interrupt", True),
        )


@dataclass
class TestCaseDefinition:
    test_case_id: str
    name: str
    type: TestCaseType
    description: str = ""
    steps: list[StepDefinition] = field(default_factory=list)
    config: TestConfig = field(default_factory=TestConfig)

    def to_dict(self) -> dict:
        return {
            "test_case_id": self.test_case_id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TestCaseDefinition:
        return cls(
            test_case_id=data["test_case_id"],
            name=data["name"],
            type=TestCaseType(data["type"]),
            description=data.get("description", ""),
            steps=[StepDefinition.from_dict(s) for s in data.get("steps", [])],
            config=TestConfig.from_dict(data.get("config", {})),
        )
