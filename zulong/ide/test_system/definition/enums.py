from enum import Enum


class TestCaseType(str, Enum):
    COMPLEX_TASK = "complex_task"
    INTERRUPT_RESUME = "interrupt_resume"
    LONG_TASK_REPORT = "long_task_report"
    ATTENTION_SWITCH = "attention_switch"


class TestStatus(str, Enum):
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INTERRUPTED = "interrupted"


class AssertionType(str, Enum):
    VALUE_MATCH = "value_match"
    STATUS_MATCH = "status_match"
    PATTERN_MATCH = "pattern_match"


class ProgressMode(str, Enum):
    EXACT = "exact"
    ESTIMATED = "estimated"


class AttentionRestoreStatus(str, Enum):
    RESTORED = "restored"
    COLD_START = "cold_start"
    FAILED = "failed"
