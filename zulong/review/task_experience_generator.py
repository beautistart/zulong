"""Task execution experience generation.

P2 of the task-memory-experience loop: turn a compact TaskExecutionTrace into
high-confidence reusable experiences. This module deliberately avoids storing
raw tool logs as experiences; raw events stay in InteractionStore/MemoryGraph.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _compact_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value or ""))


def _short(value: Any, limit: int = 180) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    # Avoid writing full local commands/absolute paths into long-term experience text.
    text = re.sub(r"[A-Za-z]:\\[^\s，,。;；]+", "<path>", text)
    text = re.sub(r"(?<!\w)/(?:[^/\s]+/){2,}[^\s，,。;；]+", "<path>", text)
    return text[:limit]


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        item = str(value or "").strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


@dataclass
class TaskExperienceCandidate:
    """Reusable experience candidate derived from one execution trace."""

    candidate_key: str
    experience_type: str
    content: str
    success: bool
    confidence: float
    importance_score: float
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    if_condition: str = ""
    then_action: str = ""
    avoid_action: str = ""
    evidence_summary: str = ""
    source_graph_nodes: List[str] = field(default_factory=list)
    source_task_nodes: List[str] = field(default_factory=list)


class TaskExperienceGenerator:
    """Generate and persist reusable experiences from TaskExecutionTrace."""

    min_confidence: float = 0.58

    def generate_candidates(self, trace: Dict[str, Any]) -> List[TaskExperienceCandidate]:
        """Build high-confidence candidates from a trace."""
        if not self._has_signal(trace):
            return []

        candidates: List[TaskExperienceCandidate] = []
        if trace.get("success"):
            procedure = self._procedure_candidate(trace)
            if procedure:
                candidates.append(procedure)
        else:
            workspace_bootstrap = self._workspace_bootstrap_failure_candidate(trace)
            if workspace_bootstrap:
                candidates.append(workspace_bootstrap)
            else:
                failure = self._failure_candidate(trace)
                if failure:
                    candidates.append(failure)

        # 用户偏好/审批偏好属于图记忆，不进入 ExperienceStore。
        # 这里仅创建/更新 MemoryGraph 记忆节点，避免污染通用经验库。
        self._project_preference_memories(trace)
        return [c for c in candidates if c.confidence >= self.min_confidence]

    def generate_and_save(
        self,
        trace: Dict[str, Any],
        *,
        experience_store: Any = None,
        memory_graph: Any = None,
    ) -> List[Dict[str, Any]]:
        """Generate candidates, save them to ExperienceStore, and project nodes."""
        candidates = self.generate_candidates(trace)
        if not candidates:
            return []

        if experience_store is None:
            from zulong.memory.enhanced_experience_store import get_enhanced_experience_store

            experience_store = get_enhanced_experience_store()
        if memory_graph is None:
            from zulong.memory.memory_graph import get_memory_graph

            memory_graph = get_memory_graph()

        saved: List[Dict[str, Any]] = []
        for candidate in candidates:
            try:
                existing_id = self._find_existing_experience_id(experience_store, candidate)
                created = False
                if existing_id:
                    exp_id = existing_id
                    candidate.metadata["candidate_key"] = candidate.candidate_key
                    self._refresh_existing_experience(
                        experience_store,
                        exp_id,
                        candidate,
                        trace,
                    )
                else:
                    candidate.metadata["candidate_key"] = candidate.candidate_key
                    exp_id = experience_store.add_experience(
                        content=self._experience_store_summary(candidate),
                        experience_type=candidate.experience_type,
                        task_id=trace.get("task_graph_id") or trace.get("trace_id"),
                        success=candidate.success,
                        metadata=candidate.metadata,
                        tags=candidate.tags,
                        importance_score=candidate.importance_score,
                    )
                    created = True

                node_id = self._project_experience_node(
                    memory_graph=memory_graph,
                    trace=trace,
                    candidate=candidate,
                    experience_id=exp_id,
                    created=created,
                )
                saved.append({
                    "experience_id": exp_id,
                    "experience_node_id": node_id,
                    "candidate_key": candidate.candidate_key,
                    "experience_type": candidate.experience_type,
                    "created": created,
                })
            except Exception as exc:
                logger.debug("[TaskExperienceGenerator] 经验候选保存跳过: %s", exc)

        if saved:
            logger.info(
                "[TaskExperienceGenerator] 已生成任务经验: trace=%s count=%s",
                trace.get("trace_id", ""),
                len(saved),
            )
        return saved

    def _has_signal(self, trace: Dict[str, Any]) -> bool:
        if not trace or not trace.get("trace_id"):
            return False
        tool_chain = trace.get("tool_chain") or []
        approvals = trace.get("approval_trace") or []
        if tool_chain or approvals:
            return True
        if not trace.get("success") and trace.get("failure_reason"):
            return True
        return False

    @staticmethod
    def _is_fully_completed(trace: Dict[str, Any]) -> bool:
        completion = trace.get("task_completion") or {}
        total = int(completion.get("total") or 0)
        completed = int(completion.get("completed") or 0)
        pending = int(completion.get("pending") or 0)
        blocked = int(completion.get("blocked") or 0)
        if total:
            return completed >= total and pending == 0 and blocked == 0
        return bool(trace.get("success"))

    def _procedure_candidate(self, trace: Dict[str, Any]) -> Optional[TaskExperienceCandidate]:
        if not self._is_fully_completed(trace):
            return None
        tool_chain = trace.get("tool_chain") or []
        if not tool_chain:
            return None
        tool_names = self._tool_names(tool_chain)
        if not tool_names:
            return None

        goal = _short(trace.get("goal"), 160) or "类似任务"
        result = _short(trace.get("result"), 180)
        files = trace.get("files") or []
        verification = trace.get("verification") or []
        tool_part = " -> ".join(tool_names[:8])
        file_part = f"；关键产物/文件 {len(files)} 个" if files else ""
        verify_part = "；已包含验证步骤" if verification else ""
        result_part = f"；结果摘要：{result}" if result else ""
        content = (
            f"流程经验：处理「{goal}」这类任务时，可优先采用工具链 "
            f"{tool_part}{file_part}{verify_part}{result_part}。"
            "下次类似任务先复用该执行顺序，并在写入、执行或打开外部环境前保留审批说明。"
        )
        confidence = 0.68 + min(0.18, 0.03 * len(tool_names))
        if files:
            confidence += 0.05
        if verification:
            confidence += 0.05
        return TaskExperienceCandidate(
            candidate_key=f"procedure:{trace.get('trace_id', '')}",
            experience_type="procedure",
            content=content,
            success=True,
            confidence=min(confidence, 0.95),
            importance_score=1.15 if verification else 1.05,
            tags=_dedupe(["task_execution", "procedure", "success", *tool_names[:5]]),
            metadata=self._metadata(trace, "procedure", confidence),
            if_condition=f"遇到类似「{goal}」的任务",
            then_action=f"优先采用工具链 {tool_part}",
            avoid_action="不要跳过必要的任务记录、审批说明和验证步骤",
            evidence_summary=result or tool_part,
            source_graph_nodes=self._source_graph_nodes(trace),
            source_task_nodes=self._source_task_nodes(trace, completed=True),
        )

    def _failure_candidate(self, trace: Dict[str, Any]) -> Optional[TaskExperienceCandidate]:
        failure_reason = _short(trace.get("failure_reason"), 220)
        failed_tools = [
            str(item.get("tool_name") or item.get("pair_id") or "")
            for item in (trace.get("tool_chain") or [])
            if item.get("success") is False or str(item.get("status") or "").lower() == "failed"
        ]
        if not failure_reason and not failed_tools:
            return None

        goal = _short(trace.get("goal"), 160) or "类似任务"
        tool_names = self._tool_names(trace.get("tool_chain") or [])
        failed_part = f"失败工具：{', '.join(_dedupe(failed_tools)[:5])}；" if failed_tools else ""
        chain_part = f"原工具链：{' -> '.join(tool_names[:8])}；" if tool_names else ""
        content = (
            f"失败经验：处理「{goal}」时任务未完成。"
            f"{failed_part}{chain_part}原因：{failure_reason or '未知'}。"
            "下次类似任务先检查前置路径、权限、事件循环和审批状态，避免重复调用同一失败工具。"
        )
        confidence = 0.72 if failure_reason else 0.64
        if failed_tools:
            confidence += 0.08
        return TaskExperienceCandidate(
            candidate_key=f"failure:{trace.get('trace_id', '')}",
            experience_type="failure",
            content=content,
            success=False,
            confidence=min(confidence, 0.92),
            importance_score=1.2,
            tags=_dedupe(["task_execution", "failure", *tool_names[:5], *_dedupe(failed_tools)[:3]]),
            metadata=self._metadata(trace, "failure", confidence),
            if_condition=f"处理类似「{goal}」的任务且出现相同失败信号",
            then_action="先检查前置路径、权限、事件循环和审批状态，再换用替代工具链",
            avoid_action=f"避免重复调用失败工具：{', '.join(_dedupe(failed_tools)[:5])}" if failed_tools else "避免重复相同失败路径",
            evidence_summary=failure_reason or "工具链失败",
            source_graph_nodes=self._source_graph_nodes(trace),
            source_task_nodes=self._source_task_nodes(trace, completed=False),
        )

    def _workspace_bootstrap_failure_candidate(self, trace: Dict[str, Any]) -> Optional[TaskExperienceCandidate]:
        """Specialize repeated new-project workspace bootstrapping failures."""
        tool_chain = trace.get("tool_chain") or []
        tool_names = self._tool_names(tool_chain)
        haystack = " ".join([
            str(trace.get("goal") or ""),
            str(trace.get("result") or ""),
            str(trace.get("failure_reason") or ""),
            " ".join(tool_names),
            " ".join(str(item.get("result_preview") or item.get("action_summary") or "") for item in tool_chain),
        ]).lower()

        has_ide_dir_tool = any(name in {"ide_write_file", "create_directory"} for name in tool_names) or (
            "ide_write_file" in haystack or "create_directory" in haystack
        )
        missing_task_plan = "task_create_plan" not in tool_names
        complex_project = any(k in haystack for k in (
            "项目", "游戏", "web", "代码", "编写", "开发", "复杂", "多文件",
            "project", "game", "workspace", "vscode",
        ))
        workspace_problem = any(k in haystack for k in (
            "workspace_required", "workspace_not_found", "工作目录不存在",
            "任务图未完成", "父目录", "尚不存在", "新项目工作区", "create_directory",
        ))
        if not (has_ide_dir_tool and missing_task_plan and complex_project and workspace_problem):
            return None

        goal = _short(trace.get("goal"), 160) or "新建项目并写代码"
        content = (
            f"失败经验：处理「{goal}」这类新建项目目录并开发代码/网页/小游戏的复杂任务时，"
            "不能用 ide_write_file(create_directory=true) 代替 task_create_plan。"
            "正确顺序是先检索历史经验，再调用 task_create_plan 创建任务图并绑定 workspace_dir，"
            "由任务创建流程负责创建目标目录和打开 VS Code；之后再用 IDE/文件工具写入具体文件。"
        )
        confidence = 0.86
        return TaskExperienceCandidate(
            candidate_key="failure:workspace_bootstrap:ide_write_file_create_directory_without_task_graph",
            experience_type="failure",
            content=content,
            success=False,
            confidence=confidence,
            importance_score=1.35,
            tags=_dedupe([
                "task_execution",
                "failure",
                "workspace_bootstrap",
                "task_create_plan",
                "ide_write_file",
                "create_directory",
                "vscode_workspace",
            ]),
            metadata=self._metadata(trace, "workspace_bootstrap_failure", confidence),
            if_condition="新建项目根目录并开发代码、网页或小游戏",
            then_action="先调用 task_create_plan 创建任务图并绑定 workspace_dir，再用 IDE/文件工具写入具体文件",
            avoid_action="不要用 ide_write_file(create_directory=true) 代替项目工作区 bootstrap",
            evidence_summary="新项目目录创建与 VS Code 工作区绑定失败",
            source_graph_nodes=self._source_graph_nodes(trace),
            source_task_nodes=self._source_task_nodes(trace, completed=False),
        )

    @staticmethod
    def _tool_names(tool_chain: List[Dict[str, Any]]) -> List[str]:
        return _dedupe(str(item.get("tool_name") or item.get("pair_id") or "") for item in tool_chain)

    @staticmethod
    def _source_graph_nodes(trace: Dict[str, Any]) -> List[str]:
        values = []
        for event_id in trace.get("source_event_ids") or []:
            if event_id:
                values.append(str(event_id))
        return _dedupe(values)[:20]

    @staticmethod
    def _source_task_nodes(trace: Dict[str, Any], *, completed: bool) -> List[str]:
        completion = trace.get("task_completion") or {}
        key = "completed_nodes" if completed else "pending_nodes"
        nodes = completion.get(key) or []
        if not nodes and not completed:
            nodes = completion.get("blocked_nodes") or []
        return _dedupe(str(item.get("id") or "") for item in nodes if isinstance(item, dict))[:20]

    @staticmethod
    def _experience_store_summary(candidate: TaskExperienceCandidate) -> str:
        """Keep ExperienceStore as a compact index; full data lives on MemoryGraph nodes."""
        parts = [
            f"经验摘要：{_short(candidate.content, 260)}",
            f"IF: {_short(candidate.if_condition, 120)}" if candidate.if_condition else "",
            f"THEN: {_short(candidate.then_action, 160)}" if candidate.then_action else "",
            f"AVOID: {_short(candidate.avoid_action, 120)}" if candidate.avoid_action else "",
        ]
        return "；".join(part for part in parts if part)

    def _metadata(
        self,
        trace: Dict[str, Any],
        candidate_kind: str,
        confidence: float,
        *,
        approval: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        compact_tools = []
        for item in trace.get("tool_chain") or []:
            compact_tools.append({
                "pair_id": item.get("pair_id", ""),
                "tool_name": item.get("tool_name", ""),
                "status": item.get("status", ""),
                "success": item.get("success"),
                "action_event_id": item.get("action_event_id", ""),
                "result_event_ids": item.get("result_event_ids", []),
            })
        metadata = {
            "created_from": "task_experience_generator",
            "candidate_kind": candidate_kind,
            "candidate_key": "",
            "confidence": confidence,
            "task_execution_trace_id": trace.get("trace_id", ""),
            "trace_id": trace.get("trace_id", ""),
            "conversation_id": trace.get("conversation_id", ""),
            "turn_id": trace.get("turn_id", ""),
            "task_graph_id": trace.get("task_graph_id", ""),
            "source_event_ids": trace.get("source_event_ids", []),
            "tool_chain": compact_tools,
            "approval_trace": trace.get("approval_trace", []),
            "files": trace.get("files", []),
            "success": bool(trace.get("success")),
            "failure_reason": trace.get("failure_reason", ""),
            "verification": trace.get("verification", []),
            "generated_at": time.time(),
            "llm_candidate_status": "rule_generated_pending_llm_review",
            "llm_review_status": "pending",
        }
        if approval:
            metadata["approval_evidence"] = approval
        return metadata

    @staticmethod
    def _find_existing_experience_id(experience_store: Any, candidate: TaskExperienceCandidate) -> str:
        experiences = getattr(experience_store, "_experiences", {}) or {}
        for exp_id, exp in experiences.items():
            meta = getattr(exp, "metadata", {}) or {}
            if meta.get("candidate_key") == candidate.candidate_key:
                return str(exp_id)
        return ""

    @staticmethod
    def _refresh_existing_experience(
        experience_store: Any,
        exp_id: str,
        candidate: TaskExperienceCandidate,
        trace: Dict[str, Any],
    ) -> None:
        """Refresh metadata for a deduped experience without creating a duplicate."""
        experiences = getattr(experience_store, "_experiences", {}) or {}
        exp = experiences.get(exp_id)
        if not exp:
            return
        try:
            exp.metadata = candidate.metadata
            exp.tags = candidate.tags
            exp.success = candidate.success
            exp.task_id = trace.get("task_graph_id") or trace.get("trace_id")
            exp.importance_score = candidate.importance_score
            if hasattr(experience_store, "_save_to_disk"):
                experience_store._save_to_disk()
        except Exception:
            return

    def _project_experience_node(
        self,
        *,
        memory_graph: Any,
        trace: Dict[str, Any],
        candidate: TaskExperienceCandidate,
        experience_id: str,
        created: bool,
    ) -> str:
        from zulong.memory.memory_graph import EdgeType, GraphNode, NodeType

        node_id = self._experience_node_id(candidate)
        metadata = {
            "sub_type": "task_execution_experience",
            "content": candidate.content,
            "summary": candidate.content[:300],
            "if": candidate.if_condition,
            "then": candidate.then_action,
            "avoid": candidate.avoid_action,
            "evidence_summary": candidate.evidence_summary,
            "experience_id": experience_id,
            "experience_type": candidate.experience_type,
            "candidate_key": candidate.candidate_key,
            "task_execution_trace_id": trace.get("trace_id", ""),
            "trace_id": trace.get("trace_id", ""),
            "conversation_id": trace.get("conversation_id", ""),
            "request_id": trace.get("turn_id", ""),
            "task_graph_id": trace.get("task_graph_id", ""),
            "source_event_ids": trace.get("source_event_ids", []),
            "tags": candidate.tags,
            "confidence": candidate.confidence,
            "success": candidate.success,
            "source_graph_nodes": candidate.source_graph_nodes,
            "source_task_nodes": candidate.source_task_nodes,
            "created_from": "task_experience_generator",
            "experience_created": created,
        }
        memory_graph.add_node(GraphNode(
            node_id=node_id,
            node_type=NodeType.EXPERIENCE,
            label=candidate.content[:60],
            backend_ref=f"enhanced_experience:{experience_id}",
            metadata=metadata,
        ))

        summary_id = self._find_trace_summary_node_id(memory_graph, trace)
        if summary_id and not self._edge_exists(memory_graph, summary_id, node_id):
            memory_graph.add_edge(
                summary_id,
                node_id,
                EdgeType.REFERENCE,
                weight=0.95,
                protected=True,
                metadata={
                    "link_type": "trace_experience",
                    "trace_id": trace.get("trace_id", ""),
                    "candidate_key": candidate.candidate_key,
                },
            )
            self._attach_experience_to_summary(memory_graph, summary_id, experience_id, node_id)
        self._attach_experience_edges(memory_graph, trace, candidate, node_id, summary_id)
        return node_id

    def _attach_experience_edges(
        self,
        memory_graph: Any,
        trace: Dict[str, Any],
        candidate: TaskExperienceCandidate,
        experience_node_id: str,
        summary_id: str = "",
    ) -> None:
        """Create graph edges that make an experience usable as part of a chain."""
        try:
            from zulong.memory.memory_graph import EdgeType
        except Exception:
            return

        task_graph_id = str(trace.get("task_graph_id") or "")
        task_node_id = self._find_task_node_id(memory_graph, task_graph_id)
        if summary_id:
            self._safe_add_edge(
                memory_graph,
                experience_node_id,
                summary_id,
                EdgeType.DERIVED_FROM,
                {"link_type": "experience_derived_from_trace", "trace_id": trace.get("trace_id", "")},
                weight=0.95,
            )
        if task_node_id:
            self._safe_add_edge(
                memory_graph,
                experience_node_id,
                task_node_id,
                EdgeType.APPLIES_TO,
                {"link_type": "experience_applies_to_task", "task_graph_id": task_graph_id},
                weight=0.8,
            )

        for event_node_id in self._find_tool_event_nodes(memory_graph, trace):
            self._safe_add_edge(
                memory_graph,
                experience_node_id,
                event_node_id,
                EdgeType.DERIVED_FROM,
                {"link_type": "experience_derived_from_tool_event"},
                weight=0.75,
            )

        failed_nodes = self._resolve_task_detail_nodes(memory_graph, trace, completed=False)
        success_nodes = self._resolve_task_detail_nodes(memory_graph, trace, completed=True)
        for failed_id in failed_nodes:
            self._safe_add_edge(
                memory_graph,
                experience_node_id,
                failed_id,
                EdgeType.CORRECTS,
                {"link_type": "experience_corrects_failed_task_node"},
                weight=0.85,
            )
            for success_id in success_nodes:
                self._safe_add_edge(
                    memory_graph,
                    failed_id,
                    success_id,
                    EdgeType.FAILED_THEN_SUCCEEDED,
                    {
                        "link_type": "failure_to_success_after_correction",
                        "experience_node_id": experience_node_id,
                    },
                    weight=0.9,
                )
                self._safe_add_edge(
                    memory_graph,
                    experience_node_id,
                    success_id,
                    EdgeType.APPLIES_TO,
                    {"link_type": "experience_success_target"},
                    weight=0.75,
                )

    @staticmethod
    def _experience_node_id(candidate: TaskExperienceCandidate) -> str:
        digest = hashlib.blake2b(candidate.content.encode("utf-8"), digest_size=6).hexdigest()
        return f"experience:task_exec:{_compact_id(candidate.candidate_key)}:{digest}"

    def _find_trace_summary_node_id(self, memory_graph: Any, trace: Dict[str, Any]) -> str:
        conversation_id = str(trace.get("conversation_id") or "")
        turn_id = str(trace.get("turn_id") or "")
        if conversation_id and turn_id:
            session_id = f"dialogue:session_{_compact_id(conversation_id)}"
            round_id = f"{session_id}/round_{_compact_id(turn_id)}"
            summary_id = f"{round_id}/episode_task_execution_trace"
            try:
                if memory_graph.has_node(summary_id):
                    return summary_id
            except Exception:
                pass

        try:
            from zulong.memory.memory_graph import NodeType

            for node in memory_graph.get_nodes_by_type(NodeType.EPISODE):
                meta = getattr(node, "metadata", {}) or {}
                if meta.get("trace_id") == trace.get("trace_id"):
                    return getattr(node, "node_id", "")
        except Exception:
            return ""
        return ""

    @staticmethod
    def _attach_experience_to_summary(memory_graph: Any, summary_id: str, experience_id: str, node_id: str) -> None:
        try:
            node = memory_graph.get_node(summary_id)
            if not node:
                return
            meta = getattr(node, "metadata", {}) or {}
            exp_ids = list(meta.get("experience_ids") or [])
            exp_nodes = list(meta.get("experience_node_ids") or [])
            if experience_id not in exp_ids:
                exp_ids.append(experience_id)
            if node_id not in exp_nodes:
                exp_nodes.append(node_id)
            meta["experience_ids"] = exp_ids
            meta["experience_node_ids"] = exp_nodes
            trace = meta.get("trace")
            if isinstance(trace, dict):
                trace["experience_ids"] = exp_ids
                trace["experience_node_ids"] = exp_nodes
                meta["trace"] = trace
            meta["task_execution_experience_updated_at"] = time.time()
            node.metadata = meta
            if hasattr(memory_graph, "update_node"):
                memory_graph.update_node(node)
        except Exception:
            return

    @staticmethod
    def _edge_exists(graph: Any, source: str, target: str) -> bool:
        if hasattr(graph, "has_edge"):
            try:
                return bool(graph.has_edge(source, target))
            except Exception:
                pass
        try:
            return bool(getattr(graph, "_graph").has_edge(source, target))
        except Exception:
            return False


_task_experience_generator: Optional[TaskExperienceGenerator] = None


def get_task_experience_generator() -> TaskExperienceGenerator:
    global _task_experience_generator
    if _task_experience_generator is None:
        _task_experience_generator = TaskExperienceGenerator()
    return _task_experience_generator


def maybe_generate_task_experiences(trace: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Best-effort terminal hook used after TaskExecutionTrace is persisted."""
    if not trace:
        return []
    try:
        return get_task_experience_generator().generate_and_save(trace)
    except Exception as exc:
        logger.debug("[TaskExperienceGenerator] generation skipped: %s", exc)
        return []
