"""
IDE Session 管理

跨 HTTP 请求维护 FC 循环状态。
每个 IDE 对话（通过 session fingerprint 识别）对应一个 AgentSession，
其中包含 IDEFCState（可序列化的 FC 循环状态）。
"""

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class IDEFCState:
    """FC 循环可序列化状态

    IDEFCRunner 在暂停时将所有状态保存到此结构，
    恢复时从此结构重建循环上下文。
    对应 fc_graph.py 的 FCLoopState 的所有 16 个字段 + 5 个 IDE 扩展字段。
    """
    messages: List[Dict] = field(default_factory=list)
    fc_turn: int = 0
    tool_results_buffer: List[Dict] = field(default_factory=list)
    cb_force_no_tools: bool = False
    gap_continue_count: int = 0
    tool_definitions: List[Dict] = field(default_factory=list)
    is_resume: bool = False
    resume_automark_count: int = 0
    null_response_count: int = 0
    api_timeout_count: int = 0
    intent_max_tokens: int = 1024
    user_input_text: str = ""

    # IDE 扩展字段
    pending_remote_calls: List[Dict] = field(default_factory=list)
    pending_call_ids: List[str] = field(default_factory=list)
    phase: str = "idle"  # "idle" | "running" | "waiting_remote" | "done"
    last_response_content: Optional[str] = None
    vllm_model_id: str = ""

    # 意图感知扩展（融合原生 3 层编排）
    ide_intent: str = "complex"  # "complex" | "resume"（IDE 模式无 CHAT）
    force_first_tool: bool = False  # RESUME 首轮强制 task_view_overview
    force_graph_id: str = ""  # 确定性恢复: 前端传入的 graph_id，非空时跳过启发式

    # 错误恢复计数器
    loop_error_count: int = 0  # 连续循环体异常次数，>=3 时终止

    # CB 模式工具调用连续计数（防止 CB 模式下模型持续调用保留工具导致死循环）
    cb_tool_streak: int = 0

    # 压力 RED: 约束工具列表为仅注意力工具
    pressure_force_attention: bool = False

    # P2: 进度报告 + 自动继续
    progress_reports: List[Dict] = field(default_factory=list)
    last_report_turn: int = 0
    auto_continue_count: int = 0  # 已自动续期次数

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IDEFCState":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class AgentSession:
    """单个 IDE 对话的会话"""
    session_id: str
    fc_state: Optional[IDEFCState] = None
    attention_window_data: Optional[Dict[str, Any]] = None  # 序列化的注意力窗口
    rule_guardian_data: Optional[Dict[str, Any]] = None      # 序列化的 RuleGuardian 状态
    circuit_breaker_data: Optional[Dict[str, Any]] = None    # 序列化的 CircuitBreaker 状态
    active_task_graph_id: Optional[str] = None
    dialogue_round_id: Optional[str] = None       # 当前对话轮次节点 ID
    dialogue_session_id: Optional[str] = None      # 当前对话会话节点 ID
    created_at: float = 0.0
    last_accessed: float = 0.0
    request_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "session_id": self.session_id,
            "fc_state": self.fc_state.to_dict() if self.fc_state else None,
            "attention_window_data": self.attention_window_data,
            "rule_guardian_data": self.rule_guardian_data,
            "circuit_breaker_data": self.circuit_breaker_data,
            "active_task_graph_id": self.active_task_graph_id,
            "dialogue_round_id": self.dialogue_round_id,
            "dialogue_session_id": self.dialogue_session_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "request_count": self.request_count,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        fc_data = data.get("fc_state")
        fc_state = IDEFCState.from_dict(fc_data) if fc_data else None
        return cls(
            session_id=data["session_id"],
            fc_state=fc_state,
            attention_window_data=data.get("attention_window_data"),
            rule_guardian_data=data.get("rule_guardian_data"),
            circuit_breaker_data=data.get("circuit_breaker_data"),
            active_task_graph_id=data.get("active_task_graph_id"),
            dialogue_round_id=data.get("dialogue_round_id"),
            dialogue_session_id=data.get("dialogue_session_id"),
            created_at=data.get("created_at", 0.0),
            last_accessed=data.get("last_accessed", 0.0),
            request_count=data.get("request_count", 0),
        )


class AgentSessionStore:
    """内存 + 磁盘 Session 存储

    线程安全的 Session 管理器，支持：
    - fingerprint 识别（无需自定义 HTTP header）
    - TTL 过期清理
    - 最大 Session 数限制
    - 磁盘持久化（进程重启后可恢复）
    """

    def __init__(self, ttl_seconds: int = 7200, max_sessions: int = 50,
                 persist_path: str = "./data/sessions"):
        self._sessions: Dict[str, AgentSession] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._last_cleanup_time: float = 0.0
        self._cleanup_interval: float = 300.0
        self._persist_path = persist_path
        self._dirty = False
        os.makedirs(persist_path, exist_ok=True)
        self._load_from_disk()
        logger.info(
            f"[AgentSessionStore] 初始化: ttl={ttl_seconds}s, "
            f"max_sessions={max_sessions}, persisted={len(self._sessions)}"
        )

    def get_or_create(
        self,
        system_prompt: str,
        first_user_msg: str,
    ) -> AgentSession:
        """获取或创建 Session

        Args:
            system_prompt: IDE 的 system prompt
            first_user_msg: 第一条 user 消息

        Returns:
            AgentSession 实例
        """
        fp = self.compute_fingerprint(system_prompt, first_user_msg)
        now = time.time()

        with self._lock:
            # 定期清理：超过上限或距离上次清理超过 5 分钟
            if (len(self._sessions) > self._max_sessions
                    or now - self._last_cleanup_time > self._cleanup_interval):
                self._cleanup_expired_locked(now)
                self._last_cleanup_time = now

            session = self._sessions.get(fp)
            if session is not None:
                # 检查是否过期
                if now - session.last_accessed > self._ttl:
                    logger.info(
                        f"[AgentSessionStore] Session {fp[:12]}... 已过期，重建"
                    )
                    session = self._create_session(fp, now)
                    self._sessions[fp] = session
                else:
                    session.last_accessed = now
                    session.request_count += 1
                    logger.info(
                        f"[AgentSessionStore] 复用 Session {fp[:12]}..., "
                        f"请求 #{session.request_count}, "
                        f"phase={session.fc_state.phase if session.fc_state else 'none'}"
                    )
            else:
                session = self._create_session(fp, now)
                self._sessions[fp] = session
                logger.info(
                    f"[AgentSessionStore] 新建 Session {fp[:12]}..., "
                    f"总数={len(self._sessions)}"
                )

            return session

    @staticmethod
    def compute_fingerprint(system_prompt: str, first_user_msg: str) -> str:
        """计算会话指纹

        使用完整 system_prompt 和 first_user_msg 的 hash（而非截断），
        避免不同对话因前缀相同而发生指纹碰撞。
        IDE 同一对话的 system_prompt 是固定的，first_user_msg 标识具体任务。
        """
        key = (system_prompt or "") + "|" + (first_user_msg or "")
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """通过 session_id 直接获取 Session"""
        with self._lock:
            return self._sessions.get(session_id)

    def cleanup_expired(self) -> int:
        """清理过期 Session，返回清理数量"""
        now = time.time()
        with self._lock:
            return self._cleanup_expired_locked(now)

    def _cleanup_expired_locked(self, now: float) -> int:
        """内部清理（需持有锁）"""
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_accessed > self._ttl
        ]
        for sid in expired:
            logger.info(f"[AgentSessionStore] 清理过期 Session {sid[:12]}...")
            del self._sessions[sid]

        # 如果仍超过上限，移除最旧的
        if len(self._sessions) > self._max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].last_accessed,
            )
            excess = len(self._sessions) - self._max_sessions
            for sid, _ in sorted_sessions[:excess]:
                logger.info(f"[AgentSessionStore] 淘汰 LRU Session {sid[:12]}...")
                del self._sessions[sid]
            expired.extend([sid for sid, _ in sorted_sessions[:excess]])

        return len(expired)

    def _create_session(self, session_id: str, now: float) -> AgentSession:
        """创建新 Session"""
        return AgentSession(
            session_id=session_id,
            fc_state=None,
            attention_window_data=None,
            active_task_graph_id=None,
            created_at=now,
            last_accessed=now,
            request_count=1,
        )

    @property
    def active_count(self) -> int:
        """当前活跃 Session 数"""
        with self._lock:
            return len(self._sessions)

    def delete(self, session_id: str) -> bool:
        """删除指定 session（内存+磁盘）"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._dirty = True
                try:
                    path = os.path.join(self._persist_path, f"{session_id}.json")
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                logger.info(f"[AgentSessionStore] 已删除 Session {session_id[:12]}")
                return True
        return False

    def save_session(self, session: AgentSession) -> None:
        """保存单个session到磁盘（异步友好，在FC循环暂停/完成时调用）"""
        try:
            path = os.path.join(self._persist_path, f"{session.session_id}.json")
            data = session.to_dict()
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"[AgentSessionStore] Session持久化失败: {e}")

    def _load_from_disk(self) -> None:
        """启动时从磁盘恢复所有session"""
        try:
            if not os.path.isdir(self._persist_path):
                return
            for fname in os.listdir(self._persist_path):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(self._persist_path, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    session = AgentSession.from_dict(data)
                    sid = session.session_id
                    now = time.time()
                    if now - session.last_accessed > self._ttl:
                        os.remove(path)
                        continue
                    self._sessions[sid] = session
                except Exception as e:
                    logger.warning(f"[AgentSessionStore] 加载session文件失败 {fname}: {e}")
        except Exception as e:
            logger.warning(f"[AgentSessionStore] 磁盘恢复失败: {e}")

    def _mark_dirty_and_save(self, session: AgentSession) -> None:
        """标记脏并异步保存（防抖）"""
        self.save_session(session)
