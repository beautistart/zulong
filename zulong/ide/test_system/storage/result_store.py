from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import aiosqlite

from test_system.definition.enums import TestCaseType, TestStatus
from test_system.definition.models import TestCaseDefinition
from test_system.engine.models import (
    AttentionSwitchEvent,
    InterruptPoint,
    ProgressReport,
    StepResult,
    TestRun,
)

logger = logging.getLogger(__name__)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS test_cases (
    test_case_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    description TEXT DEFAULT '',
    config_json TEXT DEFAULT '{}',
    steps_json TEXT DEFAULT '[]',
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS test_executions (
    execution_id TEXT PRIMARY KEY,
    test_case_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    interrupt_point_json TEXT,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS step_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    output_json TEXT,
    assertion_results_json TEXT DEFAULT '[]',
    error TEXT,
    duration_ms INTEGER,
    FOREIGN KEY (execution_id) REFERENCES test_executions(execution_id)
);

CREATE TABLE IF NOT EXISTS state_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    execution_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    state_data_json TEXT NOT NULL,
    integrity_hash TEXT NOT NULL,
    timestamp TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (execution_id) REFERENCES test_executions(execution_id)
);

CREATE TABLE IF NOT EXISTS progress_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    progress_percent REAL NOT NULL,
    mode TEXT NOT NULL,
    stage_description TEXT,
    completed_steps INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    timestamp TEXT DEFAULT (datetime('now')),
    intermediate_result_json TEXT,
    FOREIGN KEY (execution_id) REFERENCES test_executions(execution_id)
);

CREATE TABLE IF NOT EXISTS attention_switches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    from_session_id TEXT NOT NULL,
    to_session_id TEXT NOT NULL,
    restore_status TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    timestamp TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (execution_id) REFERENCES test_executions(execution_id)
);

CREATE INDEX IF NOT EXISTS idx_executions_case ON test_executions(test_case_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON test_executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_time ON test_executions(started_at);
CREATE INDEX IF NOT EXISTS idx_steps_execution ON step_results(execution_id);
CREATE INDEX IF NOT EXISTS idx_progress_execution ON progress_reports(execution_id);
CREATE INDEX IF NOT EXISTS idx_attention_execution ON attention_switches(execution_id);
"""


class TestResultStore:
    def __init__(self, db_path: str = "test_system.db"):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_CREATE_TABLES_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save_test_case(self, tc: TestCaseDefinition) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO test_cases (test_case_id, name, type, description, config_json, steps_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (tc.test_case_id, tc.name, tc.type.value, tc.description, json.dumps(tc.config.to_dict(), ensure_ascii=False), json.dumps([s.to_dict() for s in tc.steps], ensure_ascii=False)),
        )
        await self._db.commit()

    async def get_test_case(self, test_case_id: str) -> Optional[TestCaseDefinition]:
        cursor = await self._db.execute("SELECT * FROM test_cases WHERE test_case_id = ?", (test_case_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        return TestCaseDefinition.from_dict({
            "test_case_id": row["test_case_id"],
            "name": row["name"],
            "type": row["type"],
            "description": row["description"],
            "config": json.loads(row["config_json"]),
            "steps": json.loads(row["steps_json"]),
        })

    async def list_test_cases(self, type_filter: Optional[str] = None, page: int = 1, size: int = 20) -> list[dict]:
        sql = "SELECT test_case_id, name, type, description, updated_at FROM test_cases"
        params = []
        if type_filter:
            sql += " WHERE type = ?"
            params.append(type_filter)
        sql += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([size, (page - 1) * size])
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def save_execution(self, run: TestRun) -> None:
        interrupt_json = json.dumps(run.interrupt_point.to_dict(), ensure_ascii=False) if run.interrupt_point else None
        await self._db.execute(
            "INSERT OR REPLACE INTO test_executions (execution_id, test_case_id, name, type, status, started_at, completed_at, interrupt_point_json, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run.execution_id, run.test_case_id, run.name, run.type.value, run.status.value, run.started_at, run.completed_at, interrupt_json, run.error),
        )
        await self._db.commit()

    async def update_execution_status(self, execution_id: str, status: str, completed_at: Optional[str] = None, error: Optional[str] = None) -> None:
        await self._db.execute(
            "UPDATE test_executions SET status = ?, completed_at = ?, error = ? WHERE execution_id = ?",
            (status, completed_at, error, execution_id),
        )
        await self._db.commit()

    async def get_execution(self, execution_id: str) -> Optional[dict]:
        cursor = await self._db.execute("SELECT * FROM test_executions WHERE execution_id = ?", (execution_id,))
        return dict(await cursor.fetchone()) if await cursor.fetchone() else None

    async def list_executions(self, test_case_id: Optional[str] = None, status: Optional[str] = None, page: int = 1, size: int = 20) -> list[dict]:
        sql = "SELECT * FROM test_executions"
        conditions, params = [], []
        if test_case_id:
            conditions.append("test_case_id = ?")
            params.append(test_case_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([size, (page - 1) * size])
        cursor = await self._db.execute(sql, params)
        return [dict(r) for r in await cursor.fetchall()]

    async def save_step_result(self, execution_id: str, step: StepResult) -> None:
        await self._db.execute(
            "INSERT INTO step_results (execution_id, step_id, name, status, started_at, completed_at, output_json, assertion_results_json, error, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (execution_id, step.step_id, step.name, step.status.value, step.started_at, step.completed_at, json.dumps(step.output, ensure_ascii=False, default=str) if step.output else None, json.dumps([a.to_dict() for a in step.assertion_results], ensure_ascii=False), step.error, step.duration_ms),
        )
        await self._db.commit()

    async def get_step_results(self, execution_id: str) -> list[dict]:
        cursor = await self._db.execute("SELECT * FROM step_results WHERE execution_id = ? ORDER BY id", (execution_id,))
        return [dict(r) for r in await cursor.fetchall()]

    async def save_snapshot(self, snapshot_id: str, execution_id: str, step_id: str, state_data: dict, integrity_hash: str) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO state_snapshots (snapshot_id, execution_id, step_id, state_data_json, integrity_hash) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, execution_id, step_id, json.dumps(state_data, ensure_ascii=False), integrity_hash),
        )
        await self._db.commit()

    async def get_snapshot(self, snapshot_id: str) -> Optional[dict]:
        cursor = await self._db.execute("SELECT * FROM state_snapshots WHERE snapshot_id = ?", (snapshot_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def save_progress_report(self, report: ProgressReport) -> None:
        await self._db.execute(
            "INSERT INTO progress_reports (execution_id, progress_percent, mode, stage_description, completed_steps, total_steps, timestamp, intermediate_result_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (report.execution_id, report.progress_percent, report.mode.value, report.stage_description, report.completed_steps, report.total_steps, report.timestamp, json.dumps(report.intermediate_result, ensure_ascii=False, default=str) if report.intermediate_result else None),
        )
        await self._db.commit()

    async def save_attention_switch(self, execution_id: str, event: AttentionSwitchEvent) -> None:
        await self._db.execute(
            "INSERT INTO attention_switches (execution_id, from_session_id, to_session_id, restore_status, elapsed_ms, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (execution_id, event.from_session_id, event.to_session_id, event.restore_status.value, event.elapsed_ms, event.timestamp),
        )
        await self._db.commit()

    async def get_history(self, days: int = 30, test_case_id: Optional[str] = None) -> list[dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        sql = "SELECT * FROM test_executions WHERE started_at >= ?"
        params: list = [cutoff]
        if test_case_id:
            sql += " AND test_case_id = ?"
            params.append(test_case_id)
        sql += " ORDER BY started_at DESC"
        cursor = await self._db.execute(sql, params)
        return [dict(r) for r in await cursor.fetchall()]
