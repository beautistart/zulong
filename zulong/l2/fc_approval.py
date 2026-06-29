"""Web 端 FC 审批模块
从 IDE 审批机制提取共享部分，供 Web FC runner/fc_nodes 使用。

审批流程：
 1. 工具执行前调用 get_tool_risk_level() 获取风险等级。
 2. 非 LOW 风险工具调用 request_tool_approval_sync() 阻塞等待。
 3. 若 full_auto 或白名单命中，立即通过。
 4. 否则广播 APPROVAL_REQUIRED 到前端并阻塞等待，超时(60s)则拒绝。
 5. 前端通过 /ws 的 approval_result 消息通知审批结果。
"""

import json as _json
import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 工具风险分级（与 IDE 对齐，补充 Web 端工具名映射）──

# Web 端工具名 → IDE 端工具名，用于复用风险列表
_WEB_TOOL_RISK_ALIAS: Dict[str, str] = {
    "exec_write_file": "write_to_file",
    "exec_run_command": "execute_command",
    "exec_read_file": "read_file",
    "ide_open_workspace": "write_to_file",  # 切换工作区视为高风险
}

_HIGH_RISK_TOOLS = frozenset({
    "write_to_file", "replace_in_file", "execute_command",
    "delete_files", "create_rule", "delete_files_by_pattern",
    "preview_url", "use_skill", "web_fetch", "web_search",
    "ask_followup_question", "attempt_completion",
    "delete_memory_node", "task_create_plan",  # 记忆/TaskGraph 高风险操作
})

_CRITICAL_RISK_TOOLS = frozenset({
    "execute_command",   # 命令执行可被 CRITICAL
    "delete_files",      # 不可逆删除
    "delete_memory_node",  # 不可逆记忆删除
})

_APPROVAL_TIMEOUT = 60.0  # 审批等待超时秒数


def _resolve_tool_name(tool_name: str) -> str:
    """解析 Web 工具名为 IDE 等效名，用于风险判断。"""
    return _WEB_TOOL_RISK_ALIAS.get(tool_name, tool_name)


def get_tool_risk_level(tool_name: str, tool_args: str = "{}") -> str:
    """返回工具的风险等级: LOW / HIGH / CRITICAL
    Args:
        tool_name: 工具名称（Web 端或 IDE 端均可）
        tool_args: 工具参数的 JSON 字符串
    """
    resolved = _resolve_tool_name(tool_name)
    if resolved in _CRITICAL_RISK_TOOLS:
        if resolved == "execute_command":
            try:
                args = _json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                command = args.get("command", "").lower()
                dangerous = ("rm -rf", "sudo ", "chmod 777", "mkfs.", "dd if=",
                            ":(){ :|:& };:", "> /dev/sda", "format ")
                if any(d in command for d in dangerous):
                    return "CRITICAL"
            except Exception:
                pass
        return "HIGH"
    if resolved in _HIGH_RISK_TOOLS:
        return "HIGH"
    return "LOW"


# ── 同步审批等待池 ───────────────────────────────────────

class _ApprovalRequest:
    """一次审批请求"""
    __slots__ = ("approval_id", "tool_name", "risk_level", "call_id",
                 "conversation_id", "session_id", "request_id",
                 "workspace_path", "event", "result")

    def __init__(self, **kwargs):
        self.approval_id: str = kwargs.get("approval_id", "")
        self.tool_name: str = kwargs.get("tool_name", "")
        self.risk_level: str = kwargs.get("risk_level", "LOW")
        self.call_id: str = kwargs.get("call_id", "")
        self.conversation_id: str = kwargs.get("conversation_id", "")
        self.session_id: str = kwargs.get("session_id", "")
        self.request_id: str = kwargs.get("request_id", "")
        self.workspace_path: str = kwargs.get("workspace_path", "")
        self.event: threading.Event = threading.Event()
        self.result: Dict[str, Any] = {"approved": False, "reason": ""}


class _ApprovalContext:
    """线程安全的审批等待池"""
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Dict[str, _ApprovalRequest] = {}

    def submit(self, req: _ApprovalRequest) -> None:
        with self._lock:
            self._pending[req.approval_id] = req

    def resolve(self, approval_id: str, approved: bool, reason: str = "") -> bool:
        with self._lock:
            req = self._pending.pop(approval_id, None)
            if req is None:
                return False
            req.result = {"approved": approved, "reason": reason}
            req.event.set()
            return True

    def remove_stale(self, max_age_seconds: float = 300.0) -> None:
        with self._lock:
            now = time.time()
            stale = []
            for aid, req in list(self._pending.items()):
                if req.event.is_set():
                    stale.append(aid)
            for aid in stale:
                self._pending.pop(aid, None)

    def clear_session(self, conversation_id: str) -> None:
        with self._lock:
            stale = []
            for aid, req in list(self._pending.items()):
                if req.conversation_id == conversation_id:
                    req.event.set()  # 唤醒等待线程
                    stale.append(aid)
            for aid in stale:
                self._pending.pop(aid, None)

    def reset(self) -> None:
        with self._lock:
            for req in self._pending.values():
                req.event.set()
            self._pending.clear()


# 全局单例
_approval_ctx = _ApprovalContext()


def reset_approval_context() -> None:
    """重置所有待审批项（在新任务开始时调用）。"""
    _approval_ctx.reset()


def clear_session_approvals(conversation_id: str) -> None:
    """清除指定会话的待审批项。"""
    _approval_ctx.clear_session(conversation_id)


def resolve_approval(approval_id: str, approved: bool, reason: str = "") -> bool:
    """外部（WebChatRouter）调用此函数来唤醒等待中的审批请求。"""
    return _approval_ctx.resolve(approval_id, approved, reason)


def _publish_approval_required(req: _ApprovalRequest) -> None:
    """发布 APPROVAL_REQUIRED 事件到前端。
    通过 WebChatRouter._schedule_broadcast 直接广播到 /ws 客户端。
    """
    try:
        from zulong.launcher.web_chat_router import _schedule_broadcast

        approval_mode = "manual"
        risk_text = f"{req.risk_level} 风险" if req.risk_level else "风险"
        try:
            from zulong.config.approval_config import get_runtime_approval_mode
            approval_mode = str(get_runtime_approval_mode() or "manual").strip().lower()
        except Exception:
            pass
        if req.risk_level == "CRITICAL":
            approval_mode = "popup"

        friendly_name = {
            "exec_write_file": "写入文件",
            "exec_run_command": "执行命令",
            "delete_memory_node": "删除记忆",
            "task_create_plan": "创建任务计划",
            "web_search": "网络搜索",
            "web_fetch": "获取网页内容",
        }.get(req.tool_name, req.tool_name)

        message = {
            "type": "APPROVAL_REQUIRED",
            "approval_id": req.approval_id,
            "call_id": req.call_id,
            "tool_name": req.tool_name,
            "friendly_name": friendly_name,
            "approval_mode": approval_mode,
            "risk_level": req.risk_level,
            "reason": f"{risk_text}: {friendly_name}",
            "action_summary": f"{friendly_name}",
            "conversation_id": req.conversation_id,
            "session_id": req.session_id,
            "request_id": req.request_id,
            "workspace_path": req.workspace_path,
            "interaction": {
                "approval_id": req.approval_id,
                "pair_id": req.call_id,
                "kind": "approval",
                "status": "awaiting_approval",
                "title": "需要确认后继续",
                "detail": f"{friendly_name} 需要你确认后才会继续执行。",
                "tool_name": req.tool_name,
                "risk_level": req.risk_level,
                "risk_reason": f"{risk_text}: {friendly_name}",
                "approval_mode": approval_mode,
                "confirmation_state": "pending",
            },
            "timestamp": time.time(),
        }

        _schedule_broadcast(message)
        logger.info(
            "[FCApproval] 已发布审批请求: approval_id=%s tool=%s risk=%s mode=%s conv=%s",
            req.approval_id, req.tool_name, req.risk_level, approval_mode, req.conversation_id,
        )
    except Exception as exc:
        logger.warning("[FCApproval] 发布审批请求失败: %s", exc)


def request_tool_approval_sync(
    tool_name: str,
    tool_args: str = "{}",
    *,
    conversation_id: str = "",
    session_id: str = "",
    request_id: str = "",
    workspace_path: str = "",
    call_id: str = "",
) -> Tuple[bool, Dict[str, Any]]:
    """同步阻塞等待前端审批。
    Returns:
        (approved, info_dict) — approved 为 True 表示前端点✓允许；False 表示拒绝或超时。
    """
    risk_level = get_tool_risk_level(tool_name, tool_args)
    if risk_level == "LOW":
        return True, {"approved": True, "reason": "low_risk"}

    # 检查 full_auto / 白名单
    try:
        from zulong.config.approval_config import (
            get_runtime_approval_mode,
            should_runtime_auto_approve,
        )
        runtime_mode = get_runtime_approval_mode()
        if should_runtime_auto_approve(tool_name, _json.loads(tool_args), risk_level=risk_level):
            logger.info(
                "[FCApproval] %s 自动通过: tool=%s mode=%s risk=%s",
                "full_auto" if runtime_mode == "full_auto" else "whitelist",
                tool_name, runtime_mode, risk_level,
            )
            return True, {"approved": True, "reason": f"auto_approved_{runtime_mode}"}
    except Exception as exc:
        logger.debug("[FCApproval] 自动审批检查跳过: %s", exc)

    approval_id = f"approval:{uuid.uuid4().hex[:12]}"
    if not call_id:
        call_id = f"call:{uuid.uuid4().hex[:8]}"

    req = _ApprovalRequest(
        approval_id=approval_id,
        tool_name=tool_name,
        risk_level=risk_level,
        call_id=call_id,
        conversation_id=conversation_id,
        session_id=session_id,
        request_id=request_id,
        workspace_path=workspace_path,
    )
    _approval_ctx.submit(req)
    _publish_approval_required(req)

    # 阻塞等待审批结果
    if not req.event.wait(timeout=_APPROVAL_TIMEOUT):
        logger.warning("[FCApproval] 审批超时: approval_id=%s tool=%s", approval_id, tool_name)
        _approval_ctx.resolve(approval_id, False, f"审批等待超时 ({_APPROVAL_TIMEOUT}s)")
        return False, {"approved": False, "reason": f"审批等待超时 ({_APPROVAL_TIMEOUT}s)"}

    return req.result.get("approved", False), dict(req.result)
