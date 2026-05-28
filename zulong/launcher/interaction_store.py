"""Lightweight persistent store for Web-first interactions.

The launcher owns user-facing conversation state.  MemoryGraph can still keep
long-term semantic memory, but Web sessions need a small event ledger that is
fast, durable, and independent from whichever reasoning path handles a turn.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_db_path() -> Path:
    env_path = os.environ.get("ZULONG_INTERACTION_DB")
    if env_path:
        return Path(env_path)
    return _project_root() / "data" / "interaction" / "interaction_store.sqlite3"


def _json_dumps(value: Optional[Dict[str, Any]]) -> str:
    return json.dumps(value or {}, ensure_ascii=False, default=str)


class InteractionStore:
    """SQLite-backed ledger for conversations, voice records, and links."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT,
                    source TEXT,
                    workspace_path TEXT,
                    project_id TEXT,
                    task_graph_id TEXT,
                    session_node_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_active_at REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS interaction_event (
                    event_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    turn_id TEXT,
                    event_type TEXT NOT NULL,
                    role TEXT,
                    source TEXT,
                    text TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    workspace_path TEXT,
                    project_id TEXT,
                    task_graph_id TEXT,
                    mirrored_from_voice_event_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_interaction_event_conversation
                    ON interaction_event(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_interaction_event_turn
                    ON interaction_event(turn_id);

                CREATE TABLE IF NOT EXISTS voice_record (
                    voice_event_id TEXT PRIMARY KEY,
                    text TEXT,
                    source TEXT,
                    confidence REAL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    linked_conversation_id TEXT
                );

                CREATE TABLE IF NOT EXISTS conversation_link (
                    link_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    link_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversation_link_lookup
                    ON conversation_link(link_type, target_id);
                """
            )
            # TSD 23.11.6: Migration — 为已有 conversation 表添加 session_node_id 列
            try:
                conn.execute("SELECT session_node_id FROM conversation LIMIT 0")
            except Exception:
                conn.execute("ALTER TABLE conversation ADD COLUMN session_node_id TEXT")

    def upsert_conversation(
        self,
        conversation_id: str,
        *,
        title: str = "",
        source: str = "web_chat",
        workspace_path: Optional[str] = None,
        project_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
        session_node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        active: bool = True,
    ) -> Dict[str, Any]:
        now = time.time()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM conversation WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if active:
                conn.execute("UPDATE conversation SET active = 0 WHERE active = 1")
            if existing:
                old_meta = {}
                try:
                    old_meta = json.loads(existing["metadata_json"] or "{}")
                except Exception:
                    old_meta = {}
                old_meta.update(metadata or {})
                conn.execute(
                    """
                    UPDATE conversation
                    SET title = COALESCE(NULLIF(?, ''), title),
                        source = COALESCE(NULLIF(?, ''), source),
                        workspace_path = COALESCE(?, workspace_path),
                        project_id = COALESCE(?, project_id),
                        task_graph_id = COALESCE(?, task_graph_id),
                        session_node_id = COALESCE(?, session_node_id),
                        updated_at = ?,
                        last_active_at = ?,
                        active = CASE WHEN ? THEN 1 ELSE active END,
                        metadata_json = ?
                    WHERE conversation_id = ?
                    """,
                    (
                        title,
                        source,
                        workspace_path,
                        project_id,
                        task_graph_id,
                        session_node_id,
                        now,
                        now,
                        1 if active else 0,
                        _json_dumps(old_meta),
                        conversation_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO conversation (
                        conversation_id, title, source, workspace_path, project_id,
                        task_graph_id, session_node_id, created_at, updated_at, last_active_at,
                        active, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        title or "新会话",
                        source,
                        workspace_path,
                        project_id,
                        task_graph_id,
                        session_node_id,
                        now,
                        now,
                        now,
                        1 if active else 0,
                        _json_dumps(metadata),
                    ),
                )
        return self.get_conversation(conversation_id) or {"conversation_id": conversation_id}

    def set_active_conversation(self, conversation_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE conversation SET active = 0 WHERE active = 1")
            conn.execute(
                """
                UPDATE conversation
                SET active = 1, last_active_at = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (time.time(), time.time(), conversation_id),
            )

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversation WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def find_active_conversation(self, max_age_seconds: float = 1800.0) -> Optional[Dict[str, Any]]:
        cutoff = time.time() - max_age_seconds
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversation
                WHERE active = 1 AND last_active_at >= ?
                ORDER BY last_active_at DESC
                LIMIT 1
                """,
                (cutoff,),
            ).fetchone()
        return dict(row) if row else None

    def find_conversation_by_hint(self, hint: str) -> Optional[Dict[str, Any]]:
        hint = (hint or "").strip()
        if not hint:
            return None
        pattern = f"%{hint}%"
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM conversation
                WHERE conversation_id = ?
                   OR title LIKE ?
                   OR project_id LIKE ?
                   OR task_graph_id LIKE ?
                ORDER BY last_active_at DESC
                LIMIT 1
                """,
                (hint, pattern, pattern, pattern),
            ).fetchone()
        return dict(row) if row else None

    def find_recent_coding_conversation(
        self,
        *,
        workspace_path: Optional[str] = None,
        project_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []
        if task_graph_id:
            clauses.append("task_graph_id = ?")
            params.append(task_graph_id)
        if project_id:
            clauses.append("project_id = ?")
            params.append(project_id)
        if workspace_path:
            clauses.append("workspace_path = ?")
            params.append(workspace_path)
        if not clauses:
            clauses.append("(workspace_path IS NOT NULL OR task_graph_id IS NOT NULL)")
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM conversation
                WHERE {' OR '.join(clauses)}
                ORDER BY last_active_at DESC
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
        return dict(row) if row else None

    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM conversation
                ORDER BY last_active_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_conversation_for_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        turn_id = (turn_id or "").strip()
        if not turn_id:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT conversation_id, turn_id, workspace_path, project_id,
                       task_graph_id, source, created_at
                FROM interaction_event
                WHERE turn_id = ?
                  AND conversation_id IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (turn_id,),
            ).fetchone()
        return dict(row) if row else None

    def append_event(
        self,
        *,
        conversation_id: Optional[str],
        turn_id: Optional[str],
        event_type: str,
        role: Optional[str] = None,
        source: str = "system",
        text: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        workspace_path: Optional[str] = None,
        project_id: Optional[str] = None,
        task_graph_id: Optional[str] = None,
        mirrored_from_voice_event_id: Optional[str] = None,
    ) -> str:
        event_id = f"evt_{uuid.uuid4().hex}"
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interaction_event (
                    event_id, conversation_id, turn_id, event_type, role, source,
                    text, payload_json, workspace_path, project_id, task_graph_id,
                    mirrored_from_voice_event_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    conversation_id,
                    turn_id,
                    event_type,
                    role,
                    source,
                    text,
                    _json_dumps(payload),
                    workspace_path,
                    project_id,
                    task_graph_id,
                    mirrored_from_voice_event_id,
                    now,
                ),
            )
            if conversation_id:
                conn.execute(
                    """
                    UPDATE conversation
                    SET last_active_at = ?, updated_at = ?,
                        workspace_path = COALESCE(?, workspace_path),
                        project_id = COALESCE(?, project_id),
                        task_graph_id = COALESCE(?, task_graph_id)
                    WHERE conversation_id = ?
                    """,
                    (now, now, workspace_path, project_id, task_graph_id, conversation_id),
                )
        return event_id

    def get_messages(self, conversation_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM interaction_event
                WHERE conversation_id = ?
                  AND role IN ('user', 'assistant', 'tool')
                  AND COALESCE(text, '') != ''
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            payload = {}
            try:
                payload = json.loads(item.get("payload_json") or "{}")
            except Exception:
                payload = {}
            content = item.get("text") or ""
            if isinstance(payload, dict):
                if item.get("event_type", "").startswith("pipeline.") and payload.get("interaction"):
                    pass
                else:
                    for key in ("raw_markdown", "display_text", "text", "message"):
                        value = payload.get(key)
                        if value:
                            content = str(value)
                            break
                if isinstance(payload.get("interaction"), dict) and (
                    not content or item.get("event_type", "").startswith("pipeline.")
                ):
                    interaction = payload["interaction"]
                    content = str(interaction.get("detail") or interaction.get("title") or "")
                if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("interaction"), dict) and (
                    not content or item.get("event_type", "").startswith("pipeline.")
                ):
                    interaction = payload["data"]["interaction"]
                    content = str(interaction.get("detail") or interaction.get("title") or "")
            item["payload"] = payload
            item["content"] = content
            item["node_id"] = item.get("event_id")
            result.append(item)
        return result

    def get_events(
        self,
        conversation_id: str,
        *,
        turn_id: Optional[str] = None,
        limit: int = 1000,
        include_system: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return raw interaction events for trace extraction and audit.

        Unlike get_messages(), this keeps tool/system events and preserves the
        original event_type so TaskExecutionTrace can reconstruct tool chains,
        approvals, terminal status, and source event ids.
        """
        clauses = ["conversation_id = ?"]
        params: List[Any] = [conversation_id]
        if turn_id:
            clauses.append("turn_id = ?")
            params.append(turn_id)
        if not include_system:
            clauses.append("role IN ('user', 'assistant', 'tool')")
        params.append(limit)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM interaction_event
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at ASC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except Exception:
                item["payload"] = {}
            item["content"] = item.get("text") or ""
            item["node_id"] = item.get("event_id")
            result.append(item)
        return result

    def save_voice_record(
        self,
        *,
        text: str,
        source: str = "voice_page",
        confidence: Optional[float] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        voice_event_id = f"voice_{uuid.uuid4().hex}"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO voice_record (
                    voice_event_id, text, source, confidence, payload_json,
                    created_at, linked_conversation_id
                ) VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    voice_event_id,
                    text,
                    source,
                    confidence,
                    _json_dumps(payload),
                    time.time(),
                ),
            )
        return voice_event_id

    def link_voice_to_conversation(
        self,
        *,
        voice_event_id: str,
        conversation_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        link_id = f"link_{uuid.uuid4().hex}"
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE voice_record SET linked_conversation_id = ? WHERE voice_event_id = ?",
                (conversation_id, voice_event_id),
            )
            conn.execute(
                """
                INSERT INTO conversation_link (
                    link_id, conversation_id, link_type, target_id,
                    payload_json, created_at
                ) VALUES (?, ?, 'voice_record', ?, ?, ?)
                """,
                (link_id, conversation_id, voice_event_id, _json_dumps(payload), time.time()),
            )
        return link_id

    def list_voice_records(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        linked_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """列出语音记录，支持按关联状态过滤。"""
        with self._lock, self._connect() as conn:
            conn.row_factory = sqlite3.Row
            sql = """
                SELECT voice_event_id, text, source, confidence,
                       linked_conversation_id, payload_json, created_at
                FROM voice_record
            """
            conditions = []
            params: list = []
            if linked_only is True:
                conditions.append("linked_conversation_id IS NOT NULL")
            elif linked_only is False:
                conditions.append("linked_conversation_id IS NULL")
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def delete_voice_record(self, voice_event_id: str) -> bool:
        """删除单条语音记录，成功返回 True。"""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM voice_record WHERE voice_event_id = ?",
                (voice_event_id,),
            )
            return cursor.rowcount > 0


_store: Optional[InteractionStore] = None


def get_interaction_store() -> InteractionStore:
    global _store
    if _store is None:
        _store = InteractionStore()
    return _store
