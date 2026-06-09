"""
TSD v1.7 §23.3.3: InteractionPayload Python dataclass
对齐 zulong-ide/src/shared/ExtensionMessage.ts 中的 TypeScript 定义
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Literal


KindType = Literal["plan", "action", "observation", "progress", "approval", "summary", "user_interject"]
StatusType = Literal[
    "pending", "running", "awaiting_approval",
    "approved", "rejected", "succeeded",
    "failed", "blocked", "cancelled",
]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ApprovalMode = Literal["full_auto", "whitelist", "manual", "popup"]


@dataclass
class MemoryChanges:
    """本次任务产生的记忆变化"""
    created: int = 0
    strengthened: int = 0
    pruned: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {"created": self.created, "strengthened": self.strengthened, "pruned": self.pruned}


@dataclass
class InteractionPayload:
    """交互事件数据模型 — 对齐 TSD §23.3.3"""
    interaction_id: str
    pair_id: str
    kind: KindType
    status: StatusType = "pending"
    protocol_version: str = "2.0"

    # 5W 内容
    title: str = ""
    detail: Optional[str] = None
    thought: Optional[str] = None  # L2 思考: "为什么选择这个工具"
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None

    # 启动说明 (kind=plan)
    plan_steps: Optional[List[str]] = None

    # 审批
    approval_id: Optional[str] = None
    action_summary: Optional[str] = None
    risk_level: Optional[RiskLevel] = None
    risk_reason: Optional[str] = None
    confirmation_state: Optional[str] = None
    approval_mode: Optional[ApprovalMode] = None

    # 进度
    progress: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    progress_items: Optional[List[Dict[str, Any]]] = None

    # 总结 (kind=summary)
    completed_items: Optional[List[str]] = None
    verified_items: Optional[List[str]] = None
    pending_items: Optional[List[str]] = None
    risks_summary: Optional[str] = None
    next_step: Optional[str] = None
    memory_changes: Optional[MemoryChanges] = None

    # 元数据
    timestamp: Optional[float] = None
    turn: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """序列化为前端兼容的 dict"""
        result: Dict[str, Any] = {
            "interaction_id": self.interaction_id,
            "pair_id": self.pair_id,
            "kind": self.kind,
            "status": self.status,
            "protocol_version": self.protocol_version,
            "title": self.title,
        }
        if self.detail is not None:
            result["detail"] = self.detail
        if self.thought is not None:
            result["thought"] = self.thought
        if self.tool_name is not None:
            result["tool_name"] = self.tool_name
        if self.tool_args is not None:
            result["tool_args"] = self.tool_args
        if self.plan_steps is not None:
            result["plan_steps"] = self.plan_steps
        if self.approval_id is not None:
            result["approval_id"] = self.approval_id
        if self.action_summary is not None:
            result["action_summary"] = self.action_summary
        if self.risk_level is not None:
            result["risk_level"] = self.risk_level
        if self.risk_reason is not None:
            result["risk_reason"] = self.risk_reason
        if self.confirmation_state is not None:
            result["confirmation_state"] = self.confirmation_state
        if self.approval_mode is not None:
            result["approval_mode"] = self.approval_mode
        if self.progress is not None:
            result["progress"] = self.progress
        if self.current_step is not None:
            result["current_step"] = self.current_step
        if self.total_steps is not None:
            result["total_steps"] = self.total_steps
        if self.progress_items is not None:
            result["progress_items"] = self.progress_items
        if self.completed_items is not None:
            result["completed_items"] = self.completed_items
        if self.verified_items is not None:
            result["verified_items"] = self.verified_items
        if self.pending_items is not None:
            result["pending_items"] = self.pending_items
        if self.risks_summary is not None:
            result["risks_summary"] = self.risks_summary
        if self.next_step is not None:
            result["next_step"] = self.next_step
        if self.memory_changes is not None:
            result["memory_changes"] = self.memory_changes.to_dict()
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp
        if self.turn is not None:
            result["turn"] = self.turn
        return result
