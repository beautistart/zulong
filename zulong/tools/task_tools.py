# File: zulong/tools/task_tools.py
# 任务管理 FC 工具集 — 让模型通过 Function Calling 自主管理任务图
#
# 6 个工具:
# - task_create_plan: 创建新的任务图
# - task_add_node: 向任务图添加节点
# - task_mark_status: 更新节点状态
# - task_view_overview: 查看任务图概览
# - task_suspend: 挂起当前任务（持久化到磁盘）
# - task_list_suspended: 列出所有挂起的任务

import logging
import time
import os
import json
import asyncio
import threading
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)

# 按 workspace 隔离的任务图字典（修复跨窗口污染问题）
_workspace_task_graphs: Dict[str, Any] = {}
_workspace_graph_ids: Dict[str, str] = {}

# 向后兼容的全局单例（内部调用无 workspace 上下文时使用）
_active_task_graph = None
_active_graph_id = None
_active_workspace_dir = None  # 当前活跃任务的工作目录绝对路径
_active_graph_lock = threading.RLock()

# 任务图磁盘备份目录
_GRAPH_BACKUP_DIR = os.path.join(".", "data", "graph_backups")


def normalize_task_graph_id(value: Any) -> str:
    """Normalize TaskGraph identifiers and addresses to the runtime graph id.

    Accepted examples:
    - tg_1780901327
    - tg:tg_1780901327
    - tg:1780901327
    - tg:tg_1780901327/task:o1
    """
    raw = str(value or "").strip().strip("`'\" ")
    if not raw:
        return ""
    raw = raw.replace("\\", "/")
    match = re.search(r"\btg_\d+\b", raw)
    if match:
        return match.group(0)
    head = raw.split("/", 1)[0].strip()
    if head.startswith("tg:"):
        head = head[3:].strip()
    elif head.startswith("task:"):
        head = head[5:].strip()
    head = head.strip()
    if head.startswith("tg_"):
        return head
    if re.fullmatch(r"\d{6,}", head):
        return f"tg_{head}"
    return head


def normalize_task_graph_address(value: Any, default_node_id: str = "req") -> str:
    """Return a canonical MemoryGraph task address for a TaskGraph reference."""
    raw = str(value or "").strip().strip("`'\" ").replace("\\", "/")
    graph_id = normalize_task_graph_id(raw)
    if not graph_id:
        return ""
    node_id = str(default_node_id or "req").strip() or "req"
    if "/" in raw:
        tail = raw.split("/", 1)[1].strip()
        if tail:
            first = tail.split("/", 1)[0].strip()
            node_id = first[5:] if first.startswith("task:") else first
    return f"tg:{graph_id}/task:{node_id}"


def _compact_dialogue_id(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value or ""))


def infer_task_graph_owner_session_node_id(
    conversation_id: Any = "",
    session_node_id: Any = "",
) -> str:
    """Infer the dialogue session node address that owns a task graph."""
    node_id = str(session_node_id or "").strip()
    if node_id:
        return node_id
    conv_id = str(conversation_id or "").strip()
    if not conv_id:
        return ""
    if conv_id.startswith("dialogue:session_") and "/" not in conv_id:
        return conv_id
    return f"dialogue:session_{_compact_dialogue_id(conv_id)}"


def get_task_graph_owner(tg) -> Dict[str, str]:
    """Return immutable owner metadata for a TaskGraph."""
    meta = getattr(tg, "metadata", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    conversation_id = str(
        meta.get("owner_conversation_id")
        or meta.get("conversation_id")
        or ""
    ).strip()
    session_node_id = str(
        meta.get("owner_session_node_id")
        or meta.get("session_node_id")
        or meta.get("dialogue_session_id")
        or ""
    ).strip()
    if not session_node_id and conversation_id:
        session_node_id = infer_task_graph_owner_session_node_id(conversation_id)
    return {
        "owner_conversation_id": conversation_id,
        "owner_session_node_id": session_node_id,
    }


def bind_task_graph_owner(
    tg,
    conversation_id: Any = "",
    session_node_id: Any = "",
    *,
    claim_unowned: bool = False,
) -> bool:
    """Validate or claim the one-time owner binding of a TaskGraph.

    A task graph inherits the dialogue session node address at first binding.
    Once owner metadata exists, later calls may only activate the graph for the
    same owner; they must not rebind it to another conversation.
    """
    if tg is None:
        return False
    expected_conversation_id = str(conversation_id or "").strip()
    expected_session_node_id = infer_task_graph_owner_session_node_id(
        expected_conversation_id,
        session_node_id,
    )
    if not expected_conversation_id and not expected_session_node_id:
        return True

    owner = get_task_graph_owner(tg)
    owner_conversation_id = owner.get("owner_conversation_id", "")
    owner_session_node_id = owner.get("owner_session_node_id", "")

    if owner_conversation_id or owner_session_node_id:
        node_matches = bool(
            expected_session_node_id
            and owner_session_node_id
            and expected_session_node_id == owner_session_node_id
        )
        conversation_matches = bool(
            expected_conversation_id
            and owner_conversation_id
            and expected_conversation_id == owner_conversation_id
        )
        if not (node_matches or conversation_matches):
            logger.warning(
                "[TaskGraphOwner] 拒绝跨会话改绑: graph=%s owner=(%s,%s) expected=(%s,%s)",
                getattr(tg, "id", "") or getattr(tg, "graph_id", ""),
                owner_conversation_id or "-",
                owner_session_node_id or "-",
                expected_conversation_id or "-",
                expected_session_node_id or "-",
            )
            return False
        meta = getattr(tg, "metadata", None)
        if isinstance(meta, dict):
            if owner_conversation_id:
                meta.setdefault("owner_conversation_id", owner_conversation_id)
            elif expected_conversation_id:
                meta["owner_conversation_id"] = expected_conversation_id
            if owner_session_node_id:
                meta.setdefault("owner_session_node_id", owner_session_node_id)
            elif expected_session_node_id:
                meta["owner_session_node_id"] = expected_session_node_id
        return True

    if not claim_unowned:
        logger.warning(
            "[TaskGraphOwner] 图谱尚未绑定 owner，拒绝被动激活: graph=%s expected=(%s,%s)",
            getattr(tg, "id", "") or getattr(tg, "graph_id", ""),
            expected_conversation_id or "-",
            expected_session_node_id or "-",
        )
        return False

    meta = getattr(tg, "metadata", None)
    if not isinstance(meta, dict):
        tg.metadata = {}
        meta = tg.metadata
    if expected_conversation_id:
        meta["owner_conversation_id"] = expected_conversation_id
        meta.setdefault("conversation_id", expected_conversation_id)
    if expected_session_node_id:
        meta["owner_session_node_id"] = expected_session_node_id
        meta.setdefault("session_node_id", expected_session_node_id)
        meta.setdefault("dialogue_session_id", expected_session_node_id)
    logger.info(
        "[TaskGraphOwner] 首次绑定任务图 owner: graph=%s owner=(%s,%s)",
        getattr(tg, "id", "") or getattr(tg, "graph_id", ""),
        expected_conversation_id or "-",
        expected_session_node_id or "-",
    )
    return True


def _interaction_store_claims_graph(conversation_id: Any, graph_id: Any) -> bool:
    conversation_id = str(conversation_id or "").strip()
    graph_id = normalize_task_graph_id(graph_id)
    if not conversation_id or not graph_id:
        return False
    try:
        from zulong.launcher.interaction_store import get_interaction_store

        conv = get_interaction_store().get_conversation(conversation_id)
        return normalize_task_graph_id((conv or {}).get("task_graph_id")) == graph_id
    except Exception:
        return False


def _interaction_store_bound_graph(conversation_id: Any) -> str:
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return ""
    try:
        from zulong.launcher.interaction_store import get_interaction_store

        conv = get_interaction_store().get_conversation(conversation_id)
        return normalize_task_graph_id((conv or {}).get("task_graph_id"))
    except Exception:
        return ""


def _request_owner_context(request_or_params: Any) -> Tuple[str, str]:
    """Extract the dialogue session owner carried by FC tool parameters."""
    params: Dict[str, Any] = {}
    if isinstance(request_or_params, ToolRequest):
        params = request_or_params.parameters or {}
    elif isinstance(request_or_params, dict):
        params = request_or_params
    conversation_id = str(
        params.get("conversation_id")
        or params.get("session_id")
        or ""
    ).strip()
    session_node_id = infer_task_graph_owner_session_node_id(
        conversation_id,
        params.get("session_node_id") or params.get("dialogue_session_id") or "",
    )
    return conversation_id, session_node_id


def ensure_task_graph_owner_for_request(
    tg,
    request_or_params: Any,
    *,
    operation: str = "task_tool",
    allow_store_claim: bool = True,
) -> Tuple[bool, str]:
    """Guard active TaskGraph access for the current dialogue session.

    TaskGraph owner binding is immutable. Passive FC tools may only use an
    unowned graph when InteractionStore already says this conversation owns it.
    """
    if tg is None:
        return False, "当前没有活跃的任务图"
    conversation_id, session_node_id = _request_owner_context(request_or_params)
    if not conversation_id and not session_node_id:
        return True, ""
    graph_id = normalize_task_graph_id(
        getattr(tg, "id", "") or getattr(tg, "graph_id", "") or _active_graph_id
    )
    claim_unowned = bool(
        allow_store_claim and _interaction_store_claims_graph(conversation_id, graph_id)
    )
    if bind_task_graph_owner(
        tg,
        conversation_id=conversation_id,
        session_node_id=session_node_id,
        claim_unowned=claim_unowned,
    ):
        return True, ""
    return (
        False,
        (
            f"{operation} 被拒绝：当前任务图 {graph_id or '-'} 不属于当前会话 "
            f"{conversation_id or session_node_id or '-'}，已阻止跨会话改绑。"
        ),
    )


def _bind_conversation_task_graph_once(
    *,
    conversation_id: Any,
    graph_id: Any,
    title: str = "",
    workspace_dir: str = "",
    session_node_id: Any = "",
    metadata: Optional[Dict[str, Any]] = None,
    source: str = "task_tools",
) -> bool:
    """Persist the first task graph binding for a dialogue conversation."""
    conversation_id = str(conversation_id or "").strip()
    graph_id = normalize_task_graph_id(graph_id)
    if not conversation_id or not graph_id:
        return False
    owner_session_node_id = infer_task_graph_owner_session_node_id(
        conversation_id,
        session_node_id,
    )
    try:
        from zulong.launcher.interaction_store import get_interaction_store

        store = get_interaction_store()
        existing = store.get_conversation(conversation_id)
        existing_graph_id = normalize_task_graph_id((existing or {}).get("task_graph_id"))
        if existing_graph_id and existing_graph_id != graph_id:
            logger.warning(
                "[TaskGraphOwner] 会话已有图谱绑定，拒绝改绑: conversation=%s existing=%s incoming=%s",
                conversation_id,
                existing_graph_id,
                graph_id,
            )
            return False
        merged_meta = dict(metadata or {})
        merged_meta.update({
            "owner_conversation_id": conversation_id,
            "owner_session_node_id": owner_session_node_id,
        })
        store.upsert_conversation(
            conversation_id,
            title=title or "",
            source=source,
            workspace_path=workspace_dir or None,
            task_graph_id=graph_id,
            session_node_id=owner_session_node_id or None,
            metadata=merged_meta,
            active=True,
        )
        return True
    except Exception as exc:
        logger.debug("[TaskGraphOwner] 会话图谱绑定写入跳过: %s", exc)
        return False

# ─── 守卫常量 ─────────────────────────────────────────────
DUPLICATE_LABEL_THRESHOLD = 0.65   # bigram Jaccard 阈值，>=此值视为重复
SEMANTIC_DEDUP_THRESHOLD = 0.85    # SequenceMatcher(label+desc) 阈值，>=此值视为语义重复
MAX_LEAF_NODES = 0                 # 叶子节点数量上限（0=不限制，超大项目可达数千节点）
LEAF_SOFT_WARNING_THRESHOLD = 100  # 软警告阈值，超过此值提示LLM优先执行而非继续拆解
FUZZY_AUTO_CORRECT_THRESHOLD = 0.7 # 模糊匹配自动纠正阈值

# 中文序号映射
_CN_ORDINAL = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


def _normalize_label(label: str) -> str:
    """标签预处理：去除前缀序号标点、统一小写，但保留序号数字以区分不同项。
    
    例如: "1. 第一天健身计划" → "1 健身计划"
          "Day 1 - Fitness Plan" → "day 1 fitness plan"
          "Day 2 - Fitness Plan" → "day 2 fitness plan"
    
    注意：不能完全去除序号，否则 "Day 1 xxx" 和 "Day 2 xxx" 会被视为相同标签。
    """
    if not label:
        return ""
    s = label.strip().lower()
    # 去除前缀分隔符但保留序号数字: "1. " → "1 ", "1、" → "1 ", "1)" → "1 "
    s = re.sub(r"^([\d]+)[.、)：:\-]+\s*", r"\1 ", s)
    s = re.sub(r"^\(([\d]+)\)\s*", r"\1 ", s)
    # 中文序号：保留数字部分 "第一天" → "1天", "第3步" → "3步"
    def _cn_ordinal_replace(m):
        cn = m.group(1)
        num = _CN_ORDINAL.get(cn, cn)
        suffix = m.group(2) if m.group(2) else ""
        return f"{num}{suffix} "
    s = re.sub(r"^第([一二三四五六七八九十\d]+)([天步个项条])?\s*", _cn_ordinal_replace, s)
    # "day 1 - xxx" → "day 1 xxx"（保留 day N，只去分隔符）
    s = re.sub(r"^(day\s*\d+)[\s:=\-]+", r"\1 ", s, flags=re.IGNORECASE)
    # 去除标点
    s = re.sub(r"[，。！？、；：\u201c\u201d\u2018\u2019（）【】—_.,!?;:(){}\[\]-]", "", s)
    return s.strip()


def _label_similarity(a: str, b: str) -> float:
    """计算两个标签的相似度（归一化后的 bigram Jaccard）。
    
    Returns: 0.0 ~ 1.0
    """
    na = _normalize_label(a)
    nb = _normalize_label(b)
    if not na or not nb:
        return 0.0
    # 快捷路径：精确匹配或子串包含
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 1.0
    # 序号差异保护：如果两个标签含有不同序号，视为不同节点
    # 避免 "Day 1 xxx" 和 "Day 2 xxx" 因文本相似而被误判为重复
    ordinal_a = _extract_ordinal(a)
    ordinal_b = _extract_ordinal(b)
    if ordinal_a is not None and ordinal_b is not None and ordinal_a != ordinal_b:
        return 0.0
    # bigram Jaccard（复用 task_suspension._bigram_overlap 算法）
    if len(na) < 2 or len(nb) < 2:
        return 0.0
    bigrams_a = {na[i:i+2] for i in range(len(na) - 1)}
    bigrams_b = {nb[i:i+2] for i in range(len(nb) - 1)}
    intersection = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    return intersection / union if union > 0 else 0.0


_WINDOWS_ABS_PATH_RE = re.compile(r"(?P<path>[A-Za-z]:[\\/][A-Za-z0-9_.$~ (){}\[\]\-\\/]+)")
_POSIX_ABS_PATH_RE = re.compile(r"(?P<path>/(?:[^，。；;,\n\r\"'`]+))")
_EXPLICIT_WORKSPACE_PATH_RE = re.compile(
    r"(?:工作区|工作目录|项目目录|项目文件夹|workspace|project\s+dir|project\s+directory)"
    r"\s*(?:为|是|:|：)?\s*"
    r"(?P<path>[A-Za-z]:[\\/][A-Za-z0-9_.$~ (){}\[\]\-\\/]+|/(?:[^，。；;,\n\r\"'`]+))",
    re.IGNORECASE,
)
_PROJECT_NAME_PATTERNS = (
    re.compile(r"(?:项目文件夹|项目目录|文件夹|目录)\s*[“\"'](?P<name>[^”\"']{1,80})[”\"']"),
    re.compile(r"以\s*[“\"'](?P<name>[^”\"']{1,80})[”\"']\s*为项目"),
    re.compile(r"创建(?:一个)?(?:项目文件夹|项目目录|文件夹|目录)\s*(?P<name>[\w\-\u4e00-\u9fff]{1,80})"),
    re.compile(r"(?:项目文件夹|项目目录|文件夹|目录)\s*(?P<name>[A-Za-z][A-Za-z0-9_-]{0,79})"),
)
_PATH_TRAILING_ACTION_RE = re.compile(
    r"(?i)(?:\.\s+|\s+)(?=(?:write|create|make|implement|add|run|final|then|please|package|pkg|module|file|files|test|tests)\b)"
)


def _trim_inferred_path_candidate(candidate: str) -> str:
    """Trim natural-language action text accidentally captured after a path."""
    cleaned = str(candidate or "").strip().strip("“”\"'` ")
    cleaned = re.split(_PATH_TRAILING_ACTION_RE, cleaned, maxsplit=1)[0].strip()
    return cleaned.rstrip("\\/")


def _normalize_path_for_match(path_value: str) -> str:
    try:
        return str(Path(str(path_value or "")).resolve()).lower().replace("/", "\\")
    except Exception:
        return str(path_value or "").lower().replace("/", "\\")


def _task_graph_leaf_counts(tg) -> Tuple[int, int]:
    try:
        leaves = [
            n for n in tg.get_leaf_nodes()
            if getattr(n, "id", "") != "req"
            and not getattr(n, "id", "").startswith("crg_")
        ]
        unfinished = [
            n for n in leaves
            if getattr(n, "status", "") not in ("completed", "skipped")
        ]
        return len(leaves), len(unfinished)
    except Exception:
        return 0, 0


def _score_suspend_candidate(state, query: str, workspace_hint: str = "") -> float:
    text = f"{getattr(state, 'description', '')} {getattr(getattr(state, 'task_graph', None), 'title', '')}".lower()
    query_clean = str(query or "").lower().strip()
    score = float(getattr(state, "suspended_at", 0) or 0) / 1_000_000_000.0

    if query_clean:
        if query_clean in text:
            score += 10.0
        elif any(token and token in text for token in re.split(r"\s+", query_clean)):
            score += 4.0
        else:
            score += SequenceMatcher(None, query_clean, text).ratio() * 3.0

    tg = getattr(state, "task_graph", None)
    if tg:
        leaves, unfinished = _task_graph_leaf_counts(tg)
        if unfinished > 0:
            score += 5.0
        if leaves > 0:
            score += min(3.0, max(0.0, (leaves - unfinished) / leaves * 3.0))
        workspace = getattr(tg, "metadata", {}).get("workspace_dir", "")
        if workspace_hint and workspace:
            if _normalize_path_for_match(workspace) == _normalize_path_for_match(workspace_hint):
                score += 8.0

    return score


def _clean_project_folder_name(name: str) -> str:
    cleaned = str(name or "").strip().strip("“”\"'` ")
    cleaned = re.split(r"[，。；;,\s]", cleaned)[0].strip()
    return cleaned.rstrip(".。")


def infer_project_workspace_hint(text: str) -> Tuple[str, str]:
    """Infer explicit parent path and project folder name from a user request.

    The parser only preserves concrete placement constraints such as
    "在 D:/AI/project 下创建项目文件夹 mao"; L2 still owns planning.
    """
    raw = str(text or "")
    target_path = ""
    project_name = ""

    workspace_match = _EXPLICIT_WORKSPACE_PATH_RE.search(raw)
    if workspace_match:
        candidate = _trim_inferred_path_candidate(workspace_match.group("path"))
        candidate = re.sub(r"(?:下|中|里|作为|创建).*$", "", candidate).strip()
        final_path = Path(candidate.rstrip("\\/"))
        if final_path.name:
            return str(final_path.parent), _clean_project_folder_name(final_path.name)

    path_match = _WINDOWS_ABS_PATH_RE.search(raw) or _POSIX_ABS_PATH_RE.search(raw)
    if path_match:
        candidate = _trim_inferred_path_candidate(path_match.group("path"))
        candidate = re.sub(r"(?:文件夹|目录|下|中|里|作为|为|创建).*$", "", candidate).strip()
        target_path = candidate.rstrip("\\/")

    for pattern in _PROJECT_NAME_PATTERNS:
        match = pattern.search(raw)
        if match:
            project_name = _clean_project_folder_name(match.group("name"))
            if project_name:
                break

    if target_path and not project_name:
        try:
            final_path = Path(target_path)
            if final_path.name:
                return str(final_path.parent), _clean_project_folder_name(final_path.name)
        except Exception:
            pass

    if target_path and project_name:
        try:
            final_path = Path(target_path) / project_name
            if final_path.exists():
                target_path = str(final_path.parent)
        except Exception:
            pass

    return target_path, project_name


def _explicit_recreate_requested(text: str) -> bool:
    """Whether the user explicitly chose a fresh task over old task recovery."""
    raw = str(text or "").lower()
    if not raw:
        return False
    recreate_cues = (
        "全新任务",
        "新任务",
        "重新创建",
        "重新新建",
        "删除旧任务",
        "丢弃旧任务",
        "不要恢复",
        "不恢复",
        "不是恢复",
        "重新开始",
        "已明确选择删除旧任务",
        "delete old task",
        "discard old task",
        "do not resume",
        "don't resume",
        "fresh task",
        "new task",
        "recreate",
    )
    return any(cue in raw for cue in recreate_cues)


def _extract_title_core(title: str) -> str:
    """从任务标题中提取核心主题词。"""
    s = (title or "").strip()
    s = re.sub(r"^(帮我|请帮我|请|麻烦|把|将|给我|帮忙)\s*", "", s)
    s = re.sub(
        r"(写|做|设计|开发|创建|搭建|实现|生成|构建|完成|编写)(一个|一下)?\s*",
        "",
        s,
    )
    s = re.sub(r"(出来|一下|吧|呢|了)$", "", s)
    s = re.sub(r"(简单的|简单|基本的|基本|完整的|完整)", "", s)
    return s.strip()


def _strip_title_stopwords(core: str) -> str:
    """移除标题中的通用技术停用词，避免无关任务误匹配。"""
    stopwords = [
        "数据库表", "数据库", "数据结构",
        "管理系统", "应用程序", "管理平台",
        "系统的", "系统", "应用的", "应用",
        "程序的", "程序", "平台的", "平台",
        "功能的", "功能", "模块的", "模块",
        "页面的", "页面", "界面的", "界面",
        "服务的", "服务", "接口的", "接口",
        "的", "和", "与", "及",
    ]
    s = core
    for word in stopwords:
        s = s.replace(word, "")
    return s.strip()


def _titles_related(old_title: str, new_title: str) -> bool:
    """判断两个任务标题是否属于同一项目/领域的关联任务。"""
    core_a = _extract_title_core(old_title)
    core_b = _extract_title_core(new_title)
    if not core_a or not core_b:
        return False
    if core_a in core_b or core_b in core_a:
        return True

    clean_a = _strip_title_stopwords(core_a)
    clean_b = _strip_title_stopwords(core_b)
    if clean_a and clean_b and (clean_a in clean_b or clean_b in clean_a):
        return True

    a = clean_a if len(clean_a) >= 2 else core_a
    b = clean_b if len(clean_b) >= 2 else core_b
    if len(a) < 2 or len(b) < 2:
        return False
    bigrams_a = {a[i:i + 2] for i in range(len(a) - 1)}
    bigrams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    union = len(bigrams_a | bigrams_b)
    if union == 0:
        return False
    return (len(bigrams_a & bigrams_b) / union) >= 0.3


def _graph_workspace_health(tg, workspace_dir: Optional[str]) -> Dict[str, Any]:
    """检查活跃任务图绑定的工作目录是否仍可用于继续任务。"""
    health: Dict[str, Any] = {
        "ok": True,
        "workspace_dir": workspace_dir or "",
        "missing": [],
        "reason": "",
    }
    if not workspace_dir:
        return health

    workspace = Path(workspace_dir)
    if not workspace.exists() or not workspace.is_dir():
        health.update({
            "ok": False,
            "reason": f"任务工作目录不存在: {workspace_dir}",
            "missing": [workspace_dir],
        })
        return health

    try:
        completed_nodes = [
            node for node in getattr(tg, "_nodes", {}).values()
            if getattr(node, "status", "") == "completed"
        ]
        user_files = [
            p for p in workspace.rglob("*")
            if p.is_file() and ".zulong" not in p.parts and ".zlong" not in p.parts
        ]
        if completed_nodes and not user_files:
            health.update({
                "ok": False,
                "reason": "任务图已有完成节点，但工作目录里没有可见项目文件",
                "missing": ["project_files"],
            })
            return health

        mentioned_files = set()
        file_re = re.compile(r"(?<![\w./\\-])([\w.-]+\.(?:html|css|js|ts|tsx|jsx|json|md|py|txt|png|jpg|jpeg|webp|svg|yml|yaml))(?![\w./\\-])", re.IGNORECASE)
        for node in completed_nodes:
            text = "\n".join(
                str(value or "")
                for value in (
                    getattr(node, "label", ""),
                    getattr(node, "desc", ""),
                    getattr(node, "result", ""),
                )
            )
            for match in file_re.finditer(text):
                mentioned_files.add(match.group(1))

        missing_files = []
        existing_names = {p.name.lower() for p in user_files}
        for filename in sorted(mentioned_files):
            if filename.lower() not in existing_names:
                missing_files.append(filename)
        if missing_files:
            health.update({
                "ok": False,
                "reason": "任务图记录中已完成的文件在工作目录中缺失",
                "missing": missing_files[:20],
            })
    except Exception as exc:
        logger.debug(f"[task_create_plan] 工作目录健康检查跳过: {exc}")

    return health


def _extract_ordinal(text: str) -> Optional[int]:
    """从任意字符串中提取序号。
    
    "day2" → 2, "第三天" → 3, "node_5" → 5, "o7" → 7
    优先匹配语义更明确的模式（day > step > task > node > 兜底）
    """
    if not text:
        return None
    # 中文序号: "第三天" "第3天"
    m = re.search(r"第([一二三四五六七八九十])[\u4e00-\u9fff]?", text)
    if m:
        return _CN_ORDINAL.get(m.group(1))
    m = re.search(r"第(\d+)", text)
    if m:
        return int(m.group(1))
    # 高优先级英文模式: "day2" "day 3" "step_1"
    m = re.search(r"(?:day|step|item)\s*[_\-]?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 节点 ID 模式: "o7" "task_5"（短 ID）
    m = re.search(r"(?:^|[^a-zA-Z])o(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:task|node)\s*[_\-]?\s*(\d{1,3})(?:\D|$)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # 兜底：提取最后一个短数字（<=3位，排除长数字串如时间戳）
    nums = re.findall(r"\b(\d{1,3})\b", text)
    if nums:
        return int(nums[-1])
    return None


def _fuzzy_resolve_node_id(tg, raw_id: str) -> Tuple[Optional[str], float, str]:
    """三级模糊匹配：当 node_id 不存在时尝试纠正。
    
    Returns: (resolved_id 或 None, 置信度 0.0~1.0, 匹配方法描述)
    """
    if not raw_id or not tg:
        return (None, 0.0, "empty")
    
    all_ids = [nid for nid in tg._nodes.keys() if nid != "req"]
    if not all_ids:
        return (None, 0.0, "no_nodes")
    
    raw_lower = raw_id.lower().strip()
    
    # ── 第一级：前缀匹配（置信度 0.9）──
    prefix_matches = []
    for nid in all_ids:
        nid_lower = nid.lower()
        if nid_lower.startswith(raw_lower) or raw_lower.startswith(nid_lower):
            prefix_matches.append(nid)
    if len(prefix_matches) == 1:
        return (prefix_matches[0], 0.9, "prefix")
    
    # ── 第二级：序号匹配（置信度 0.8）──
    ordinal = _extract_ordinal(raw_id)
    if ordinal is not None:
        # 尝试直接映射到 o{N}
        candidate = f"o{ordinal}"
        if candidate in tg._nodes:
            return (candidate, 0.8, "ordinal")
        # 尝试在叶子节点中按序号位置匹配
        leaves = tg.get_leaf_nodes()
        if 1 <= ordinal <= len(leaves):
            return (leaves[ordinal - 1].id, 0.75, "ordinal_position")
    
    # ── 第三级：标签 bigram 相似度（置信度 0.5-0.7）──
    # 当 raw_id 看起来像标签文本（含中文或长度>5）时启用
    if re.search(r"[\u4e00-\u9fff]", raw_id) or len(raw_id) > 5:
        best_id = None
        best_score = 0.0
        for nid in all_ids:
            node = tg.get_node(nid)
            if node:
                score = _label_similarity(raw_id, node.label)
                if score > best_score:
                    best_score = score
                    best_id = nid
        if best_id and best_score >= 0.4:
            conf = 0.5 + (best_score * 0.2)  # 映射到 0.5-0.7
            return (best_id, min(conf, 0.7), "label_bigram")
    
    return (None, 0.0, "no_match")


def _is_user_task_node(node) -> bool:
    """任务执行节点过滤：CRG/code graph 节点不参与完成度和归档判断。"""
    node_id = str(getattr(node, "id", "") or "")
    return bool(node_id) and node_id != "req" and not node_id.startswith("crg_")


def _user_leaf_nodes(tg) -> List[Any]:
    try:
        return [n for n in tg.get_leaf_nodes() if _is_user_task_node(n)]
    except Exception:
        return []


def _compact_semantic_summary(text: str, limit: int = 500) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip()


def _find_or_create_summary_node(tg):
    """找到或创建任务总结节点，承载最终答案的结构化投射。"""
    try:
        for node in getattr(tg, "_nodes", {}).values():
            metadata = getattr(node, "metadata", {}) or {}
            if metadata.get("role") == "final_summary":
                return node

        summary_id = "summary"
        if tg.get_node(summary_id):
            idx = 2
            while tg.get_node(f"summary_{idx}"):
                idx += 1
            summary_id = f"summary_{idx}"

        node = tg.add_node(
            id=summary_id,
            label="任务总结",
            type="summary",
            status="completed",
            desc="本轮任务最终总结",
            result="",
        )
        node.metadata["role"] = "final_summary"
        node.metadata["source"] = "submit_final_answer"
        tg.add_h_edge("req", summary_id)
        return node
    except Exception as e:
        logger.warning("[TaskGraph] 创建总结节点失败（非致命）: %s", e)
        return None


def _sync_task_graph_to_memory(tg) -> None:
    """将完整 TaskGraph 投射到 MemoryGraph，确保新增总结节点也能进入图记忆。"""
    try:
        from zulong.memory.memory_graph import get_memory_graph
        from zulong.memory.graph_adapters import TaskGraphAdapter
        mg = get_memory_graph()
        if mg is not None:
            TaskGraphAdapter().sync(mg, tg)
    except Exception as e:
        logger.debug("[TaskGraph] MemoryGraph 全量同步跳过: %s", e)


def _persist_task_graph_backup(tg) -> None:
    try:
        graph_id = normalize_task_graph_id(getattr(tg, "id", "")) or _active_graph_id
        if graph_id:
            _backup_graph_to_disk(tg, graph_id)
    except Exception as e:
        logger.debug("[TaskGraph] 备份跳过: %s", e)


def _write_final_answer_to_task_graph(tg, answer: str, source: str = "submit_final_answer") -> bool:
    """把最终答案同时写入根节点和总结节点，并同步到 MemoryGraph。"""
    answer = str(answer or "").strip()
    if tg is None or not answer:
        return False

    summary = _compact_semantic_summary(answer)
    updated = False

    try:
        root = tg.get_node("req")
        if root is not None:
            root.status = "completed"
            root.result = answer
            root.semantic_summary = summary
            root.analysis_content = answer
            root.metadata["role"] = "task_root"
            root.metadata["final_answer_length"] = len(answer)
            root.metadata["final_answer_updated_at"] = time.time()
            root.metadata["final_answer_source"] = source
            updated = True

        summary_node = _find_or_create_summary_node(tg)
        if summary_node is not None:
            summary_node.status = "completed"
            summary_node.result = answer
            summary_node.semantic_summary = summary
            summary_node.analysis_content = answer
            summary_node.desc = summary_node.desc or "本轮任务最终总结"
            summary_node.metadata["role"] = "final_summary"
            summary_node.metadata["source"] = source
            summary_node.metadata["final_answer_length"] = len(answer)
            summary_node.metadata["final_answer_updated_at"] = time.time()
            updated = True

        if updated:
            try:
                tg._mark_dirty()
            except Exception:
                pass
            _sync_task_graph_to_memory(tg)
            _persist_task_graph_backup(tg)
            if getattr(tg, "on_change_callback", None):
                tg.on_change_callback("final_answer_update", {
                    "node_id": "req",
                    "summary_node_id": getattr(summary_node, "id", ""),
                    "answer_length": len(answer),
                    "source": source,
                })
            logger.info(
                "[TaskGraph] 最终答案已写入根节点和总结节点: graph=%s len=%s",
                getattr(tg, "id", ""),
                len(answer),
            )
        return updated
    except Exception as e:
        logger.warning("[TaskGraph] 写入最终答案失败（非致命）: %s", e)
        return False


def _auto_archive_completed(tg):
    """将已完成的任务图归档到 completed_tasks（幂等，重复调用安全）"""
    try:
        from zulong.l2.task_archive import CompletedTaskArchiveManager, CompletedTaskArchive
        mgr = CompletedTaskArchiveManager()

        root = tg.get_node("req")
        description = root.label if root else getattr(tg, 'title', '未命名任务')
        graph_id = getattr(tg, 'id', '') or _active_graph_id or f"tg_{int(time.time())}"
        root_result = str(getattr(root, "result", "") or "").strip() if root else ""
        if not root or getattr(root, "status", "") != "completed" or not root_result:
            logger.info(
                "[TaskArchive] 跳过自动归档: %s 根节点尚无最终总结",
                graph_id,
            )
            return

        leaves = _user_leaf_nodes(tg)
        unfinished = [
            n for n in leaves
            if getattr(n, "status", "") not in ("completed", "skipped")
        ]
        if leaves and unfinished:
            logger.warning(
                "[TaskArchive] 跳过自动归档: %s 仍有 %s/%s 个叶子节点未完成: %s",
                graph_id,
                len(unfinished),
                len(leaves),
                ", ".join(f"{n.id}({n.label})" for n in unfinished[:5]),
            )
            return

        archive = CompletedTaskArchive(
            task_id=graph_id,
            description=description,
            final_answer=root_result,
            duration=(time.time() - getattr(tg, "created_at", time.time())) if hasattr(tg, "created_at") else 0,
            total_turns=tg.metadata.get("total_turns", 0) if hasattr(tg, "metadata") else 0,
            completion_status="completed",
            task_graph_snapshot=tg.serialize(),
            workspace_dir=_active_workspace_dir or "",
            metadata={"graph_id": graph_id},
        )

        _run_async(mgr.archive_task(archive))
        logger.info(f"[TaskArchive] 任务已自动归档: {graph_id} ({description})")
    except Exception as e:
        logger.warning(f"[TaskArchive] 自动归档失败（非致命）: {e}")


def get_active_task_graph(workspace_dir=None):
    """获取当前活跃的 TaskGraph
    
    Args:
        workspace_dir: 可选，传入时返回该 workspace 专属图谱，避免跨窗口污染
    """
    with _active_graph_lock:
        if workspace_dir:
            key = os.path.abspath(workspace_dir)
            return _workspace_task_graphs.get(key)
        return _active_task_graph


def get_active_workspace_dir():
    """获取当前活跃任务的工作目录路径，无活跃任务时返回 None"""
    with _active_graph_lock:
        return _active_workspace_dir


def _normalize_workspace_dir(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return os.path.abspath(os.path.normpath(os.path.expanduser(os.path.expandvars(str(value)))))
    except Exception:
        return str(value or "").strip()


def _infer_workspace_from_file_refs(tg) -> str:
    """Infer a task workspace from absolute file attachments."""
    candidates: List[str] = []
    try:
        for node in getattr(tg, "_nodes", {}).values():
            for ref in getattr(node, "files", []) or []:
                ref_path = str(getattr(ref, "path", "") or "").strip()
                if not ref_path or not os.path.isabs(ref_path):
                    continue
                target = Path(_normalize_workspace_dir(ref_path))
                if target.exists() and target.is_dir():
                    candidates.append(str(target))
                else:
                    candidates.append(str(target.parent))
    except Exception:
        return ""

    existing_dirs = []
    for candidate in candidates:
        try:
            path = Path(candidate)
            if path.exists() and path.is_dir():
                existing_dirs.append(str(path.resolve()))
        except Exception:
            continue
    if not existing_dirs:
        return ""

    try:
        common = os.path.commonpath(existing_dirs)
        if common and os.path.isdir(common):
            return _normalize_workspace_dir(common)
    except Exception:
        pass
    return _normalize_workspace_dir(existing_dirs[0])


def _infer_workspace_from_project_registry(graph_id: str) -> str:
    graph_id = normalize_task_graph_id(graph_id)
    if not graph_id:
        return ""
    try:
        from zulong.workspace.project_registry import get_project_registry
        project = get_project_registry().get_project_by_graph_id(graph_id)
        if project and project.path:
            workspace = _normalize_workspace_dir(project.path)
            if os.path.isdir(workspace):
                return workspace
    except Exception:
        pass
    return ""


def _resolve_task_workspace_dir(tg, graph_id: str = "", workspace_dir: Optional[str] = None) -> str:
    """Resolve and persist the workspace bound to a TaskGraph."""
    graph_id = normalize_task_graph_id(graph_id or getattr(tg, "id", "") or getattr(tg, "graph_id", ""))
    candidates = [
        workspace_dir,
        getattr(tg, "metadata", {}).get("workspace_dir", "") if tg is not None else "",
        _infer_workspace_from_project_registry(graph_id),
        _infer_workspace_from_file_refs(tg) if tg is not None else "",
    ]

    # Preserve the current binding only for the same graph.
    try:
        if graph_id and graph_id == _active_graph_id:
            candidates.append(_active_workspace_dir)
    except Exception:
        pass

    for candidate in candidates:
        workspace = _normalize_workspace_dir(candidate)
        if not workspace:
            continue
        if os.path.isdir(workspace):
            if tg is not None:
                try:
                    tg.metadata["workspace_dir"] = workspace
                except Exception:
                    pass
            return workspace
    return ""


_OUTPUT_FILE_RE = re.compile(
    r"(?<![\w.-])([A-Za-z0-9][A-Za-z0-9_\-]*(?:\.[A-Za-z0-9][A-Za-z0-9_\-]{0,10})+)(?![\w.-])"
)
_OUTPUT_FILE_EXTENSIONS = {
    ".html", ".css", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".xml",
    ".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".csv", ".sql",
    ".sh", ".ps1", ".bat", ".cmd", ".env", ".ini", ".scss", ".less",
    ".vue", ".svelte", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".cs",
    ".php", ".rb",
}


def _expected_output_files_for_node(node) -> List[str]:
    """Extract explicit file outputs promised by a task node label/desc."""
    text = f"{getattr(node, 'label', '')}\n{getattr(node, 'desc', '')}"
    seen: set[str] = set()
    files: List[str] = []
    for match in _OUTPUT_FILE_RE.finditer(text):
        name = match.group(1).strip()
        suffix = Path(name).suffix.lower()
        if suffix not in _OUTPUT_FILE_EXTENSIONS:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        files.append(name)
    return files


def _file_ref_matches_expected(ref, expected_name: str, workspace: Optional[Path]) -> bool:
    ref_path = str(getattr(ref, "path", "") or "").strip()
    ref_name = str(getattr(ref, "name", "") or os.path.basename(ref_path)).strip()
    if not ref_path:
        return False

    expected_norm = expected_name.replace("\\", "/").lower()
    ref_name_norm = ref_name.replace("\\", "/").lower()
    ref_path_norm = ref_path.replace("\\", "/").lower()
    if (
        expected_norm != ref_name_norm
        and not ref_path_norm.endswith("/" + expected_norm)
        and os.path.basename(expected_norm) != os.path.basename(ref_name_norm)
    ):
        return False

    target = Path(ref_path)
    if not target.is_absolute():
        if workspace is None:
            return False
        target = workspace / ref_path

    try:
        target = target.resolve()
        if workspace is not None:
            workspace_resolved = workspace.resolve()
            if os.path.commonpath([str(workspace_resolved), str(target)]) != str(workspace_resolved):
                return False
        return target.is_file()
    except Exception:
        return False


def _missing_expected_output_files(tg, node) -> Tuple[List[str], str]:
    """Return missing explicit output files for completed file-production nodes."""
    expected = _expected_output_files_for_node(node)
    if not expected:
        return [], ""

    workspace_dir = _resolve_task_workspace_dir(
        tg,
        getattr(tg, "id", "") or getattr(tg, "graph_id", ""),
        getattr(tg, "metadata", {}).get("workspace_dir") or get_active_workspace_dir() or "",
    )
    workspace: Optional[Path] = None
    if workspace_dir:
        workspace = Path(workspace_dir).resolve()
        if not workspace.exists():
            return expected, str(workspace)

    missing: List[str] = []
    for file_name in expected:
        attached_files = getattr(node, "files", []) or []
        if any(_file_ref_matches_expected(ref, file_name, workspace) for ref in attached_files):
            continue

        if workspace is None:
            missing.append(file_name)
            continue

        candidate = Path(file_name)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (workspace / file_name).resolve()
        try:
            if os.path.commonpath([str(workspace), str(target)]) != str(workspace):
                missing.append(file_name)
                continue
        except ValueError:
            missing.append(file_name)
            continue
        if not target.is_file():
            missing.append(file_name)
    return missing, str(workspace)


def _is_user_seeded_empty_task_graph(tg) -> bool:
    """Return True for a user-created placeholder graph with no real plan yet."""
    if tg is None:
        return False
    try:
        if not getattr(tg, "metadata", {}).get("user_seeded_empty_graph"):
            return False
        nodes = getattr(tg, "_nodes", {}) or {}
        non_root_nodes = [nid for nid in nodes if nid != "req"]
        if non_root_nodes:
            return False
        root = tg.get_node("req") if hasattr(tg, "get_node") else nodes.get("req")
        if root and getattr(root, "status", "") not in ("pending", "in_progress"):
            return False
        return True
    except Exception:
        return False


def set_active_task_graph(
    tg,
    graph_id,
    workspace_dir=None,
    conversation_id: Any = "",
    session_node_id: Any = "",
    claim_unowned: bool = False,
):
    """设置当前活跃的 TaskGraph，并自动备份到磁盘"""
    global _active_task_graph, _active_graph_id, _active_workspace_dir
    graph_id = normalize_task_graph_id(graph_id)
    if tg is not None and graph_id:
        try:
            tg.id = graph_id
            tg.graph_id = graph_id
        except Exception:
            pass
        if not bind_task_graph_owner(
            tg,
            conversation_id=conversation_id,
            session_node_id=session_node_id,
            claim_unowned=claim_unowned,
        ):
            return False
    resolved_workspace = _resolve_task_workspace_dir(tg, graph_id, workspace_dir)
    with _active_graph_lock:
        if tg is None:
            clear_workspace = _normalize_workspace_dir(workspace_dir) or _active_workspace_dir
            if clear_workspace:
                key = os.path.abspath(clear_workspace)
                _workspace_task_graphs.pop(key, None)
                _workspace_graph_ids.pop(key, None)
            if graph_id:
                for key, gid in list(_workspace_graph_ids.items()):
                    if normalize_task_graph_id(gid) == graph_id:
                        _workspace_task_graphs.pop(key, None)
                        _workspace_graph_ids.pop(key, None)
            _active_task_graph = None
            _active_graph_id = None
            _active_workspace_dir = None
            try:
                from zulong.l2.task_state_manager import task_state_manager
                task_state_manager.clear_active_task(graph_id or None, clear_stack=True)
            except Exception as e:
                logger.debug(f"[TaskTools] TaskStateManager 清理跳过: {e}")
            return True

        _active_task_graph = tg
        _active_graph_id = graph_id
        _active_workspace_dir = resolved_workspace or None

        # 按 workspace 隔离存储（修复跨窗口污染）
        if resolved_workspace:
            key = os.path.abspath(resolved_workspace)
            _workspace_task_graphs[key] = tg
            _workspace_graph_ids[key] = graph_id

        # 磁盘备份：每次设置活跃图时保存一份，防止数据丢失
        if tg is not None and graph_id:
            _backup_graph_to_disk(tg, graph_id)
        # 自动注入 Web 监控回调（如果 IDE Server 已启动）
        if tg is not None and not tg.on_change_callback:
            try:
                from zulong.ide.ide_server import _task_graph_change_callback
                tg.on_change_callback = _task_graph_change_callback
            except Exception:
                pass
        # 同步到 TaskStateManager，保持两套状态一致
        try:
            from zulong.l2.task_state_manager import task_state_manager
            current_tsm_task = task_state_manager.get_active_task()
            if tg is not None and graph_id and current_tsm_task != graph_id:
                task_state_manager.create_task(graph_id, [], freeze_existing=False)
                logger.debug(
                    f"[TaskTools] 已同步 TaskGraph {graph_id} 到 TaskStateManager"
                )
        except Exception as e:
            logger.debug(f"[TaskTools] TaskStateManager 同步跳过: {e}")

    # 同步到 MemoryGraph（保持双重持久化一致）
    if tg is not None:
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.memory.graph_adapters import TaskGraphAdapter
            mg = get_memory_graph()
            if mg is not None:
                adapter = TaskGraphAdapter()
                adapter.sync(mg, tg)
                logger.debug(f"[TaskTools] TaskGraph {graph_id} 已同步到 MemoryGraph")
        except Exception as e:
            logger.debug(f"[TaskTools] MemoryGraph 同步跳过: {e}")
    return True


def _create_task_workspace(
    graph_id: str,
    project_mode: bool = False,
    project_name: str = "",
    project_desc: str = "",
    target_path: str = "",
) -> str:
    """为任务创建独立工作目录，返回绝对路径

    Args:
        graph_id: 任务图 ID
        project_mode: 是否使用项目模式（创建到统一工作空间）
        project_name: 项目名称（project_mode=True 时使用）
        project_desc: 项目描述/任务全文（project_mode=True 时使用）
        target_path: 可选：用户指定的目标父目录。若提供，项目将创建在该目录下
                     而非 workspace_root。

    Returns:
        工作目录绝对路径

    目录结构:
        project_mode=False: ./agent_workspace/{YYYYMMDD}_{HHMMSS}_{graph_id}/
        project_mode=True (无 target_path): <workspace_root>/<project_name>/
        project_mode=True (有 target_path): <target_path>/<project_name>/
    """
    from pathlib import Path

    if project_mode and project_name:
        try:
            from zulong.workspace.project_registry import get_project_registry

            registry = get_project_registry()
            info = registry.create_project(
                name=project_name,
                description=project_desc,
                task_graph_id=graph_id,
                source="web",
                target_path=target_path,
                reuse_existing=bool(target_path),
            )

            # 广播 PROJECT_CREATED 事件到 Web 前端
            try:
                from zulong.core.event_bus import get_event_bus
                from zulong.core.types import EventType, ZulongEvent
                bus = get_event_bus()
                if bus:
                    bus.publish(ZulongEvent(
                        type=EventType.PROJECT_CREATED,
                        source="workspace",
                        payload={
                            "project_id": info.project_id,
                            "name": info.name,
                            "path": info.path,
                            "task_graph_id": graph_id,
                            "status": info.status,
                        },
                    ))
            except Exception as e:
                logger.debug(f"[Workspace] PROJECT_CREATED 事件发送跳过: {e}")

            logger.info(f"[Workspace] 项目模式工作目录: {info.path}")
            return info.path
        except Exception as e:
            logger.error(f"[Workspace] 项目模式创建失败，回退到默认模式: {e}")
            # 回退到默认行为

    # 默认行为：agent_workspace 下的时间戳目录
    root = "./agent_workspace"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{graph_id}"
    full_path = os.path.join(root, folder_name)
    os.makedirs(full_path, exist_ok=True)
    abs_path = str(Path(full_path).resolve())
    logger.info(f"[Workspace] 创建任务工作目录: {abs_path}")
    return abs_path


def _backup_graph_to_disk(tg, graph_id: str):
    """将任务图序列化备份到磁盘（原子写入：先写临时文件再替换）"""
    try:
        os.makedirs(_GRAPH_BACKUP_DIR, exist_ok=True)
        backup_path = os.path.join(_GRAPH_BACKUP_DIR, f"{graph_id}.json")
        tmp_path = backup_path + ".tmp"
        data = tg.serialize()
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, backup_path)
        logger.debug(f"[GraphBackup] 已备份任务图到 {backup_path}")

        custom_path = ""
        try:
            custom_path = str(getattr(tg, "metadata", {}).get("task_graph_file_path") or "")
        except Exception:
            custom_path = ""
        if custom_path:
            custom_abs = os.path.abspath(os.path.normpath(os.path.expanduser(os.path.expandvars(custom_path))))
            if os.path.abspath(backup_path) != custom_abs:
                os.makedirs(os.path.dirname(custom_abs), exist_ok=True)
                custom_tmp = custom_abs + ".tmp"
                with open(custom_tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(custom_tmp, custom_abs)
                logger.debug(f"[GraphBackup] 已同步任务图到用户文件 {custom_abs}")

        # 增量更新语义索引
        try:
            title = getattr(tg, 'title', '') or data.get('title', '')
            if title and len(title) >= 3:
                from zulong.memory.task_search_index import (
                    get_task_search_index, TaskIndexEntry,
                )
                idx = get_task_search_index()
                idx.add_entry(TaskIndexEntry(
                    entry_id=graph_id,
                    title=title,
                    source="backup",
                    file_path=backup_path,
                ))
                # 不立即 save（backup 频繁），由 dirty 计数器控制
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[GraphBackup] 备份失败（非致命）: {e}")
        # 清理可能残留的临时文件
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def load_graph_from_backup(graph_id: str):
    """从磁盘备份恢复任务图（当挂起任务中找不到时的降级方案）

    Returns:
        TaskGraph 实例，或 None
    """
    graph_id = normalize_task_graph_id(graph_id)
    try:
        backup_path = os.path.join(_GRAPH_BACKUP_DIR, f"{graph_id}.json")
        if not os.path.exists(backup_path):
            return None
        with open(backup_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        from zulong.l2.task_graph import TaskGraph
        tg = TaskGraph.deserialize(data)
        logger.info(f"[GraphBackup] 从备份恢复任务图: {graph_id}")
        return tg
    except Exception as e:
        logger.warning(f"[GraphBackup] 从备份恢复失败: {e}")
        return None


def load_task_graph_deterministic(
    graph_id: str,
    workspace_dir: Optional[str] = None,
    conversation_id: Any = "",
    session_node_id: Any = "",
    claim_unowned: bool = False,
) -> bool:
    """Load a TaskGraph by exact id and make it active.

    Order: current memory binding -> disk backup -> MemoryGraph rebuild.
    This is used when the Web/IDE payload explicitly carries task_graph_id;
    heuristic suspended-task recovery must not override it.
    """
    graph_id = normalize_task_graph_id(graph_id)
    if not graph_id:
        return False

    current = get_active_task_graph(workspace_dir=workspace_dir) or get_active_task_graph()
    if current and normalize_task_graph_id(getattr(current, "id", "")) == graph_id:
        if not set_active_task_graph(
            current,
            graph_id,
            workspace_dir=workspace_dir,
            conversation_id=conversation_id,
            session_node_id=session_node_id,
            claim_unowned=claim_unowned,
        ):
            return False
        logger.info("[TaskTools] 确定性恢复 Level 1 (内存): %s", graph_id)
        return True

    tg = load_graph_from_backup(graph_id)
    if tg:
        if not set_active_task_graph(
            tg,
            graph_id,
            workspace_dir=workspace_dir,
            conversation_id=conversation_id,
            session_node_id=session_node_id,
            claim_unowned=claim_unowned,
        ):
            return False
        logger.info("[TaskTools] 确定性恢复 Level 2 (磁盘): %s", graph_id)
        return True

    try:
        from zulong.memory.memory_graph import get_memory_graph
        from zulong.memory.graph_adapters import rebuild_task_graph_from_memory
        mg = get_memory_graph()
        if mg:
            tg = rebuild_task_graph_from_memory(mg, graph_id)
            if tg:
                if not set_active_task_graph(
                    tg,
                    graph_id,
                    workspace_dir=workspace_dir,
                    conversation_id=conversation_id,
                    session_node_id=session_node_id,
                    claim_unowned=claim_unowned,
                ):
                    return False
                logger.info("[TaskTools] 确定性恢复 Level 3 (MemoryGraph): %s", graph_id)
                return True
    except Exception as exc:
        logger.debug("[TaskTools] 确定性恢复 MemoryGraph 跳过: %s", exc)

    logger.warning("[TaskTools] 确定性恢复失败: %s", graph_id)
    return False


def load_latest_uncompleted_backup():
    """加载最近修改的未全部完成的备份图谱（用于 resume 时无活跃 TG 的场景）

    扫描 graph_backups/ 目录，按文件修改时间倒序，返回第一个含有未完成节点的图。

    Returns:
        (TaskGraph, graph_id) 或 (None, None)
    """
    if not os.path.exists(_GRAPH_BACKUP_DIR):
        return None, None

    from zulong.l2.task_graph import TaskGraph

    # 按修改时间倒序排列
    files = []
    for fname in os.listdir(_GRAPH_BACKUP_DIR):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(_GRAPH_BACKUP_DIR, fname)
        try:
            mtime = os.path.getmtime(fpath)
            files.append((mtime, fpath, fname))
        except Exception:
            continue

    files.sort(key=lambda x: x[0], reverse=True)

    for mtime, fpath, fname in files[:5]:  # 最多检查最近 5 个
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            tg = TaskGraph.deserialize(data)
            graph_id = data.get("id") or fname.replace(".json", "")

            # 检查是否有未完成节点
            leaves = tg.get_leaf_nodes()
            uncompleted = [n for n in leaves
                           if n.status not in ("completed", "skipped")]
            if uncompleted:
                logger.info(
                    f"[GraphBackup] 找到最近未完成备份: {graph_id}, "
                    f"未完成节点={len(uncompleted)}")
                return tg, graph_id
        except Exception as e:
            logger.debug(f"[GraphBackup] 跳过备份 {fname}: {e}")
            continue

    return None, None


def load_latest_backup():
    """加载最近修改的备份图谱（不限状态，用于 session_resume 恢复）

    扫描 graph_backups/ 目录，按文件修改时间倒序，返回最近的有效图谱。
    跳过只有 req 单节点的骨架图（那是之前错误创建的空壳）。

    Returns:
        (TaskGraph, graph_id) 或 (None, None)
    """
    if not os.path.exists(_GRAPH_BACKUP_DIR):
        return None, None

    from zulong.l2.task_graph import TaskGraph

    files = []
    for fname in os.listdir(_GRAPH_BACKUP_DIR):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        fpath = os.path.join(_GRAPH_BACKUP_DIR, fname)
        try:
            mtime = os.path.getmtime(fpath)
            files.append((mtime, fpath, fname))
        except Exception:
            continue

    files.sort(key=lambda x: x[0], reverse=True)

    for mtime, fpath, fname in files[:5]:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            tg = TaskGraph.deserialize(data)
            graph_id = data.get("id") or fname.replace(".json", "")

            # 跳过只有 req 单节点的骨架图
            if len(tg._nodes) <= 1:
                continue

            leaves = [n for n in tg.get_leaf_nodes() if getattr(n, "id", "") != "req"]
            unfinished = [
                n for n in leaves
                if getattr(n, "status", "") not in ("completed", "skipped")
            ]
            if leaves and not unfinished:
                logger.info(
                    f"[GraphBackup] 跳过已完成备份: {graph_id}, "
                    f"nodes={len(tg._nodes)}"
                )
                continue

            logger.info(
                f"[GraphBackup] 加载最近备份: {graph_id}, "
                f"nodes={len(tg._nodes)}")
            return tg, graph_id
        except Exception as e:
            logger.debug(f"[GraphBackup] 跳过备份 {fname}: {e}")
            continue

    return None, None


def _save_active_backup():
    """将当前活跃任务图备份到磁盘（供工具修改后调用）"""
    with _active_graph_lock:
        if _active_task_graph is not None and _active_graph_id:
            _backup_graph_to_disk(_active_task_graph, _active_graph_id)


# ── 异常退出恢复 ─────────────────────────────────────────────

# 错误结果检测模式（与 Rule B 保持一致）
_ERROR_RESULT_PATTERNS = [
    "抱歉", "响应较慢", "请稍后再试", "请稍后",
    "rate limit", "timeout", "timed out",
    "too many requests", "server error",
    "internal error", "服务繁忙", "请求过于频繁",
]


def _is_error_result(result_text: str) -> bool:
    """判断 result 是否为错误/限流/超时响应"""
    if not result_text:
        return False
    lower = result_text.lower()
    return any(p.lower() in lower for p in _ERROR_RESULT_PATTERNS)


def repair_corrupted_task_graphs() -> dict:
    """扫描 graph_backups/ 中的任务图，修复被异常退出损坏的节点状态。

    修复规则：
    - status=completed 但 result 匹配错误模式 → 重置为 pending + 清空 result
    - status=completed 但 result 为空/过短（<20字）→ 重置为 pending

    Returns:
        {"scanned": int, "repaired": int, "details": [...]}
    """
    stats = {"scanned": 0, "repaired": 0, "details": []}

    if not os.path.exists(_GRAPH_BACKUP_DIR):
        return stats

    for fname in os.listdir(_GRAPH_BACKUP_DIR):
        if not fname.endswith(".json") or fname.endswith(".tmp"):
            continue
        stats["scanned"] += 1
        fpath = os.path.join(_GRAPH_BACKUP_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        nodes = data.get("nodes", {})
        graph_id = data.get("id", fname.replace(".json", ""))
        modified = False

        for nid, ndata in nodes.items():
            if nid == "req":
                continue  # req 节点的状态由子节点级联决定
            node_status = ndata.get("status", "")
            node_result = (ndata.get("result") or "").strip()

            if node_status != "completed":
                continue

            needs_repair = False
            reason = ""

            if _is_error_result(node_result):
                needs_repair = True
                reason = f"error_result: '{node_result[:60]}'"
            elif not node_result:
                needs_repair = True
                reason = "empty_result"

            if needs_repair:
                ndata["status"] = "pending"
                ndata["result"] = ""
                modified = True
                stats["details"].append(
                    f"{graph_id}/{nid}: {reason} → reset to pending"
                )
                logger.info(
                    f"[TaskRecovery] {graph_id}/{nid}: "
                    f"completed → pending ({reason})"
                )

        # 如果有子节点被重置，req 节点也需要重置
        if modified:
            req = nodes.get("req", {})
            if req.get("status") == "completed":
                req["status"] = "in_progress"
                stats["details"].append(
                    f"{graph_id}/req: cascade reset to in_progress"
                )

            # 原子写回
            try:
                tmp = fpath + ".repair.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, fpath)
                stats["repaired"] += 1
                logger.info(
                    f"[TaskRecovery] 已修复 {graph_id} "
                    f"({len([d for d in stats['details'] if graph_id in d])} 个节点)"
                )
            except Exception as e:
                logger.warning(f"[TaskRecovery] 写回失败 {graph_id}: {e}")

    if stats["repaired"]:
        logger.info(
            f"[TaskRecovery] 扫描 {stats['scanned']} 个图谱，"
            f"修复 {stats['repaired']} 个"
        )
    return stats


class TaskCreatePlanTool(BaseTool):
    """task_create_plan — 创建新任务规划图"""

    def __init__(self):
        super().__init__(name="task_create_plan", category=ToolCategory.CUSTOM)
        self.description = (
            "创建一个新的任务规划图。仅当用户明确要求'规划'、'设计'、'制定计划'或"
            "要求完成需要分解为多个步骤的复杂项目时调用。"
            "关键词：规划、设计、制定、分解、步骤、项目、开发、实施。"
            "注意：如果用户只是要求'记住'信息，请使用save_memory_note工具。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        title = request.parameters.get("title", "未命名任务")
        user_requirement = request.parameters.get("user_requirement", "")
        conversation_id = (
            request.parameters.get("conversation_id")
            or request.parameters.get("session_id")
            or ""
        )
        session_node_id = infer_task_graph_owner_session_node_id(
            conversation_id,
            request.parameters.get("session_node_id")
            or request.parameters.get("dialogue_session_id")
            or "",
        )
        inferred_target_path, inferred_project_name = infer_project_workspace_hint(
            f"{user_requirement}\n{title}"
        )
        requested_target_path = request.parameters.get("target_path", "")
        requested_project_name = request.parameters.get("project_name", "")
        requested_target_norm = _normalize_path_for_match(requested_target_path)
        inferred_final_norm = ""
        if inferred_target_path and inferred_project_name:
            try:
                inferred_final_norm = _normalize_path_for_match(
                    str(Path(inferred_target_path) / inferred_project_name)
                )
            except Exception:
                inferred_final_norm = ""
        if (
            user_requirement
            and inferred_target_path
            and inferred_project_name
            and requested_target_norm
            and requested_target_norm == inferred_final_norm
            and requested_project_name
            and requested_project_name != inferred_project_name
        ):
            target_path = inferred_target_path
            project_name = inferred_project_name
        elif requested_target_path or requested_project_name:
            target_path = requested_target_path or inferred_target_path
            project_name = requested_project_name or inferred_project_name or title
        else:
            target_path = inferred_target_path
            project_name = inferred_project_name or title
        existing_task_policy = str(
            request.parameters.get("existing_task_policy")
            or request.parameters.get("existingTaskPolicy")
            or "ask"
        ).strip().lower()
        if existing_task_policy not in {"ask", "reuse", "recreate"}:
            existing_task_policy = "ask"
        if existing_task_policy == "ask" and _explicit_recreate_requested(
            f"{user_requirement}\n{title}"
        ):
            existing_task_policy = "recreate"

        try:
            # 🔥 拦截：如果当前有活跃任务图且仍有未完成节点，不创建新图谱
            # 直接返回现有图谱概览，引导模型继续使用 task_view_overview
            old_tg = get_active_task_graph()
            if old_tg is not None and not bind_task_graph_owner(
                old_tg,
                conversation_id=conversation_id,
                session_node_id=session_node_id,
                claim_unowned=_interaction_store_claims_graph(
                    conversation_id,
                    getattr(old_tg, "id", "") or _active_graph_id,
                ),
            ):
                logger.info(
                    "[task_create_plan] 当前 active graph 不属于本会话，已清空后创建新图: active=%s conversation=%s",
                    getattr(old_tg, "id", "") or _active_graph_id,
                    conversation_id or "-",
                )
                set_active_task_graph(None, None)
                old_tg = None
            if old_tg is not None and _is_user_seeded_empty_task_graph(old_tg):
                logger.info(
                    "[task_create_plan] 当前为空会话预建图谱，转为真实任务: %s -> %s",
                    getattr(old_tg, "title", ""),
                    title,
                )
                old_root = old_tg.get_node("req") if hasattr(old_tg, "get_node") else None
                if old_root is not None:
                    old_root.label = title
                    old_root.desc = user_requirement or title
                    old_root.status = "in_progress"
                old_tg.title = title
                old_tg.metadata["user_requirement"] = user_requirement or title
                old_tg.metadata["user_seeded_empty_graph"] = False
                if project_name:
                    old_tg.metadata["project_name"] = project_name
                if target_path:
                    old_tg.metadata["target_path"] = target_path
                set_active_task_graph(
                    old_tg,
                    getattr(old_tg, "id", "") or _active_graph_id,
                    workspace_dir=(
                        getattr(old_tg, "metadata", {}).get("workspace_dir")
                        or get_active_workspace_dir()
                        or target_path
                    ),
                    conversation_id=conversation_id,
                    session_node_id=session_node_id,
                    claim_unowned=_interaction_store_claims_graph(
                        conversation_id,
                        getattr(old_tg, "id", "") or _active_graph_id,
                    ),
                )
                _bind_conversation_task_graph_once(
                    conversation_id=conversation_id,
                    graph_id=getattr(old_tg, "id", "") or _active_graph_id,
                    title=title,
                    workspace_dir=getattr(old_tg, "metadata", {}).get("workspace_dir") or "",
                    session_node_id=session_node_id,
                    metadata={
                        "user_requirement": user_requirement or title,
                        "task_graph_file_path": getattr(old_tg, "metadata", {}).get("task_graph_file_path") or "",
                    },
                )
                return self._create_result(
                    success=True,
                    data={
                        "graph_id": getattr(old_tg, "id", "") or _active_graph_id,
                        "root_node_id": "req",
                        "title": title,
                        "user_requirement": user_requirement or title,
                        "workspace_dir": getattr(old_tg, "metadata", {}).get("workspace_dir") or "",
                        "task_graph_file_path": getattr(old_tg, "metadata", {}).get("task_graph_file_path") or "",
                        "message": "已使用空会话预建图谱承接当前任务。请用 task_add_node 添加子任务。",
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            if old_tg is not None:
                old_root = old_tg.get_node("req")
                old_title = old_root.label if old_root else old_tg.title
                old_workspace_dir = (
                    getattr(old_tg, "metadata", {}).get("workspace_dir")
                    or get_active_workspace_dir()
                    or ""
                )
                old_workspace_health = _graph_workspace_health(old_tg, old_workspace_dir)
                if (
                    _titles_related(old_title, title)
                    and not old_workspace_health.get("ok", True)
                    and existing_task_policy == "ask"
                ):
                    decision_message = (
                        f"检测到当前任务图标题仍是「{old_title}」，但绑定目录/文件状态异常："
                        f"{old_workspace_health.get('reason') or '工作目录不完整'}。"
                        "请用户明确选择：回复「恢复上个任务」继续旧图谱，或回复「删除旧任务并重新创建」后创建全新任务。"
                    )
                    try:
                        from zulong.launcher.web_chat_router import update_task_execution_status
                        update_task_execution_status(
                            state="blocked",
                            phase="workspace_missing_needs_decision",
                            message=decision_message,
                            workspace_path=old_workspace_dir,
                            task_graph_id=_active_graph_id,
                            progress_items=[{
                                "label": "等待用户选择旧任务处理方式",
                                "status": "blocked",
                                "source": "task_graph",
                                "detail": decision_message,
                                "timestamp": time.time(),
                            }],
                        )
                    except Exception:
                        pass
                    return self._create_result(
                        success=True,
                        data={
                            "graph_id": _active_graph_id,
                            "title": old_title,
                            "already_exists": True,
                            "needs_user_decision": True,
                            "workspace_dir": old_workspace_dir,
                            "workspace_health": old_workspace_health,
                            "message": decision_message,
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                if existing_task_policy == "recreate":
                    if conversation_id and _interaction_store_claims_graph(
                        conversation_id,
                        getattr(old_tg, "id", "") or _active_graph_id,
                    ):
                        return self._create_result(
                            success=False,
                            error=(
                                "当前会话的任务图谱已一次性绑定，不能在同一会话中重建为另一张图。"
                                "请新建会话，或在当前图谱中追加/调整节点。"
                            ),
                            execution_time=time.time() - start_time,
                            request_id=request.request_id,
                        )
                    logger.info(
                        "[task_create_plan] 用户/LLM 选择重建任务，清除旧活跃图谱: %s -> %s",
                        old_title,
                        title,
                    )
                    set_active_task_graph(None, None)
                    old_tg = None

            if old_tg is not None:
                old_root = old_tg.get_node("req")
                old_title = old_root.label if old_root else old_tg.title
                # 统计现有图谱进度
                total = 0
                completed = 0
                next_pending_id = None
                next_pending_label = None
                for nid, node in old_tg._nodes.items():
                    total += 1
                    if node.status == "completed":
                        completed += 1
                    elif node.status in ("pending", "not_started") and next_pending_id is None and nid != "req":
                        next_pending_id = nid
                        next_pending_label = node.label

                # 🔥 [Fix-7] + [Fix-7C] 如果旧图所有叶子节点已完成，
                # 先检查新旧任务关联性再决定清除或复用
                _old_leaves = old_tg.get_leaf_nodes()
                _old_uncompleted = [
                    n for n in _old_leaves
                    if n.status not in ("completed", "skipped")
                ]
                if not _old_uncompleted and _old_leaves:
                    # 旧图已完成：检查新任务是否与旧任务关联
                    if _titles_related(old_title, title):
                        # 关联任务 → 复用旧图，不创建新图
                        logger.info(
                            f"[task_create_plan] 旧图谱 '{old_title}' 已完成 "
                            f"({completed}/{total})，新任务 '{title}' 与之关联，复用旧图"
                        )
                        return self._create_result(
                            success=True,
                            data={
                                "graph_id": _active_graph_id,
                                "title": old_title,
                                "already_exists": True,
                                "progress": (
                                    f"任务图「{old_title}」已完成 ({completed}/{total})。"
                                    f"新需求「{title}」与之相关，请用 task_add_node 添加新节点。"
                                ),
                                "message": (
                                    f"任务图「{old_title}」已完成，新需求与之相关。"
                                    f"请调用 task_add_node 在现有图谱上追加新功能节点。"
                                ),
                            },
                            execution_time=time.time() - start_time,
                            request_id=request.request_id,
                        )
                    else:
                        # 无关任务 → 清除旧图，创建新图
                        if conversation_id and _interaction_store_claims_graph(
                            conversation_id,
                            getattr(old_tg, "id", "") or _active_graph_id,
                        ):
                            return self._create_result(
                                success=False,
                                error=(
                                    "当前会话已绑定完成任务图谱，不能改绑到新图谱。"
                                    "请新建会话承接无关新任务。"
                                ),
                                execution_time=time.time() - start_time,
                                request_id=request.request_id,
                            )
                        logger.info(
                            f"[task_create_plan] 旧图谱 '{old_title}' 已全部完成 "
                            f"({completed}/{total})，新任务 '{title}' 无关，清除后创建新图谱"
                        )
                        set_active_task_graph(None, None)
                        # 继续往下走，创建新图谱
                else:
                    logger.info(f"[task_create_plan] 拦截重复创建：已有活跃图谱 '{old_title}' ({completed}/{total})")
                    if existing_task_policy == "ask":
                        decision_message = (
                            f"当前已有未完成任务图「{old_title}」({completed}/{total} 已完成)。"
                            "请先由 LLM 结合用户语义决定：复用旧任务、删除旧任务并重建，或向用户澄清。"
                        )
                        try:
                            from zulong.launcher.web_chat_router import update_task_execution_status
                            update_task_execution_status(
                                state="blocked",
                                phase="existing_task_needs_policy",
                                message=decision_message,
                                workspace_path=(
                                    getattr(old_tg, "metadata", {}).get("workspace_dir")
                                    or get_active_workspace_dir()
                                    or ""
                                ),
                                task_graph_id=_active_graph_id,
                                progress_items=[{
                                    "label": "等待 LLM/用户确认任务归属",
                                    "status": "blocked",
                                    "source": "task_graph",
                                    "detail": decision_message,
                                    "timestamp": time.time(),
                                }],
                            )
                        except Exception:
                            pass
                        return self._create_result(
                            success=True,
                            data={
                                "graph_id": _active_graph_id,
                                "title": old_title,
                                "already_exists": True,
                                "needs_policy_decision": True,
                                "existing_task_policy": existing_task_policy,
                                "progress": f"已有活跃任务图「{old_title}」({completed}/{total} 已完成)。",
                                "next_pending_node_id": next_pending_id,
                                "message": decision_message,
                            },
                            execution_time=time.time() - start_time,
                            request_id=request.request_id,
                        )
                    return self._create_result(
                        success=True,
                        data={
                            "graph_id": _active_graph_id,
                            "title": old_title,
                            "already_exists": True,
                            "progress": f"已有活跃任务图「{old_title}」({completed}/{total} 已完成)。",
                            "next_pending_node_id": next_pending_id,
                            "message": (
                                f"当前已有活跃任务图「{old_title}」，无需重新创建。"
                                f"请调用 task_view_overview 查看完整进度，"
                                f"然后用 task_mark_status 继续执行未完成的节点。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            bound_graph_id = _interaction_store_bound_graph(conversation_id)
            if bound_graph_id:
                if load_task_graph_deterministic(
                    bound_graph_id,
                    conversation_id=conversation_id,
                    session_node_id=session_node_id,
                    claim_unowned=True,
                ):
                    return self._create_result(
                        success=True,
                        data={
                            "graph_id": bound_graph_id,
                            "already_exists": True,
                            "message": (
                                f"当前会话已绑定任务图谱 {bound_graph_id}，不能再创建第二张图谱。"
                                "请在该图谱上追加或更新节点。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                return self._create_result(
                    success=False,
                    error=(
                        f"当前会话已绑定任务图谱 {bound_graph_id}，但暂时无法恢复。"
                        "为避免改绑到新图谱，请先打开原会话图谱或新建会话。"
                    ),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            from zulong.l2.task_graph import TaskGraph
            graph_id = f"tg_{int(time.time())}"
            tg = TaskGraph(title=title, graph_id=graph_id)

            # 创建根节点
            root_desc = user_requirement or title
            root = tg.add_node(
                id="req",
                label=title,
                type="requirement",
                status="in_progress",
                desc=root_desc,
            )

            # 创建独立工作目录（使用项目模式）
            workspace_dir = _create_task_workspace(
                graph_id,
                project_mode=True,
                project_name=project_name,
                project_desc=root_desc,
                target_path=target_path,
            )
            tg.metadata["workspace_dir"] = workspace_dir
            tg.metadata["project_name"] = project_name
            tg.metadata["target_path"] = target_path
            bind_task_graph_owner(
                tg,
                conversation_id=conversation_id,
                session_node_id=session_node_id,
                claim_unowned=True,
            )

            set_active_task_graph(
                tg,
                graph_id,
                workspace_dir=workspace_dir,
                conversation_id=conversation_id,
                session_node_id=session_node_id,
                claim_unowned=True,
            )
            bound_ok = _bind_conversation_task_graph_once(
                conversation_id=conversation_id,
                graph_id=graph_id,
                title=title,
                workspace_dir=workspace_dir,
                session_node_id=session_node_id,
                metadata={
                    "user_requirement": root_desc,
                    "project_name": project_name,
                    "target_path": target_path,
                },
            )
            if conversation_id and not bound_ok:
                set_active_task_graph(None, None)
                return self._create_result(
                    success=False,
                    error=(
                        "当前会话已有任务图谱绑定或绑定写入失败，已拒绝创建新图谱。"
                        "请打开原会话图谱继续，或新建会话承接新任务。"
                    ),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            vscode_open_result = {
                "ok": False,
                "status": "not_attempted",
                "reason": (
                    "task_create_plan 只创建并绑定任务工作区；"
                    "前台 VS Code 窗口必须由 ide_open_workspace 或 Web 端按钮显式触发。"
                ),
            }

            # 同步到 MemoryGraph
            try:
                from zulong.memory.memory_graph import get_memory_graph, GraphNode, NodeType, EdgeType
                mg = get_memory_graph()
                if mg:
                    task_node = GraphNode(
                        node_id=f"task:{graph_id}",
                        node_type=NodeType.TASK,
                        label=title,
                        activation=1.0,
                        created_at=time.time(),
                        last_accessed=time.time(),
                        access_count=1,
                        metadata={"graph_id": graph_id, "status": "active"},
                    )
                    mg.add_node(task_node)
                    mg.update_focus_to_node(task_node.node_id)
            except Exception as e:
                logger.debug(f"[task_create_plan] MemoryGraph 同步跳过: {e}")

            logger.info(f"[task_create_plan] 创建任务图 {graph_id}: {title}")

            return self._create_result(
                success=True,
                data={
                    "graph_id": graph_id,
                    "root_node_id": "req",
                    "title": title,
                    "user_requirement": root_desc,
                    "project_name": project_name,
                    "target_path": target_path,
                    "workspace_dir": workspace_dir,
                    "vscode_opened": bool(vscode_open_result.get("ok")),
                    "vscode_open_result": vscode_open_result,
                    "message": (
                        f"任务规划图已创建，工作目录已创建为 {workspace_dir}。"
                        "如需查看代码，可通过打开 VS Code 动作进入该目录。"
                        "请用 task_add_node 添加子任务。"
                    ),
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_create_plan] 创建失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"任务图创建失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "任务标题，描述整体目标",
                },
                "target_path": {
                    "type": "string",
                    "description": "可选：用户指定的项目父目录绝对路径。例如 Windows 的 D:/project/，或 Linux/macOS 的 /home/user/project/。留空则使用默认工作区。",
                },
                "project_name": {
                    "type": "string",
                    "description": "可选：用户明确指定的项目文件夹名，例如 tank。若用户只描述任务标题，可留空由系统从需求文本推断。",
                },
                "existing_task_policy": {
                    "type": "string",
                    "enum": ["ask", "reuse", "recreate"],
                    "description": (
                        "当内存里已有同名或相关任务图时的处理策略。"
                        "该字段必须由 LLM 根据用户语义决定：ask=需要向用户澄清；"
                        "reuse=复用当前旧任务图；recreate=删除/丢弃当前旧任务图并创建新任务。"
                        "不要由代码关键词推断。"
                    ),
                    "default": "ask",
                },
            },
            "required": ["title"],
        }


class TaskAddNodeTool(BaseTool):
    """task_add_node — 添加任务节点"""

    def __init__(self):
        super().__init__(name="task_add_node", category=ToolCategory.CUSTOM)
        self.description = (
            "向当前任务图添加一个子节点。"
            "通过 parent_id 指定父节点：顶层模块挂到 'req'，具体步骤挂到模块节点的 ID 下。"
            "系统自动根据深度确定节点类型（分析/大纲/任务/子任务/微任务）。"
            "建议先创建阶段节点再创建子步骤，保持 3-5 层深度以支持复杂任务分解。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        parent_id = request.parameters.get("parent_id", "req")
        label = request.parameters.get("label", "") or request.parameters.get("name", "")
        desc = request.parameters.get("desc", "") or request.parameters.get("description", "")

        # 4B 模型常在参数值中多加引号
        if isinstance(parent_id, str):
            parent_id = parent_id.strip().strip('"').strip("'")
        if isinstance(label, str):
            label = label.strip().strip('"').strip("'")
        if isinstance(desc, str):
            desc = desc.strip().strip('"').strip("'")

        if not label:
            return self._create_result(
                success=False,
                error="label 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            # 🔥 [Fix] 自动创建任务图：当没有活跃任务图时，自动创建一个
            logger.warning("[task_add_node] 无活跃任务图，自动创建默认任务图")
            auto_title = f"自动任务规划 - {label[:30]}" if len(label) > 30 else f"自动任务规划 - {label}"
            auto_request = ToolRequest(
                tool_name="task_create_plan",
                action="create",
                parameters={
                    "title": auto_title,
                    "conversation_id": request.parameters.get("conversation_id") or request.parameters.get("session_id") or "",
                    "session_id": request.parameters.get("session_id") or request.parameters.get("conversation_id") or "",
                    "session_node_id": request.parameters.get("session_node_id") or request.parameters.get("dialogue_session_id") or "",
                    "dialogue_session_id": request.parameters.get("dialogue_session_id") or request.parameters.get("session_node_id") or "",
                },
                request_id=f"auto_{request.request_id}",
            )
            create_tool = TaskCreatePlanTool()
            create_result = create_tool.execute(auto_request)
            
            if not create_result.success:
                return self._create_result(
                    success=False,
                    error=f"自动创建任务图失败: {create_result.error}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            
            # 重新获取任务图
            tg = get_active_task_graph()
            if tg is None:
                return self._create_result(
                    success=False,
                    error="自动创建任务图后仍无法获取活跃任务图",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            
            logger.info(f"[task_add_node] 已自动创建任务图: {tg.title} (graph_id={tg.graph_id})")

        try:
            # ── 守卫 A: 重复标签检查 ──
            best_match_id = None
            best_match_label = None
            best_match_score = 0.0
            for nid, node in tg._nodes.items():
                if nid == "req":
                    continue
                score = _label_similarity(label, node.label)
                if score > best_match_score:
                    best_match_score = score
                    best_match_id = nid
                    best_match_label = node.label
            if best_match_score >= DUPLICATE_LABEL_THRESHOLD and best_match_id:
                logger.info(
                    f"[task_add_node] 拦截重复标签: '{label}' ≈ '{best_match_label}' "
                    f"(id={best_match_id}, score={best_match_score:.2f})"
                )
                return self._create_result(
                    success=True,
                    data={
                        "duplicate": True,
                        "existing_node_id": best_match_id,
                        "existing_label": best_match_label,
                        "similarity": round(best_match_score, 2),
                        "message": (
                            f"已存在相似节点 {best_match_id}（{best_match_label}）。"
                            f"请直接调用 task_mark_status(node_id='{best_match_id}', "
                            f"status='in_progress') 操作该节点，不要添加新节点。"
                        ),
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # ── 守卫 A2: 语义指纹去重 (label + desc 组合) ──
            if desc:
                new_fingerprint = f"{label} {desc}".strip().lower()
                sem_best_id = None
                sem_best_label = None
                sem_best_score = 0.0
                for nid, node in tg._nodes.items():
                    if nid == "req":
                        continue
                    existing_fp = f"{node.label} {node.desc or ''}".strip().lower()
                    ratio = SequenceMatcher(None, new_fingerprint, existing_fp).ratio()
                    if ratio > sem_best_score:
                        sem_best_score = ratio
                        sem_best_id = nid
                        sem_best_label = node.label
                if sem_best_score >= SEMANTIC_DEDUP_THRESHOLD and sem_best_id:
                    logger.info(
                        f"[task_add_node] 语义指纹去重拦截: '{label}' ≈ '{sem_best_label}' "
                        f"(id={sem_best_id}, SequenceMatcher={sem_best_score:.2f})"
                    )
                    return self._create_result(
                        success=True,
                        data={
                            "duplicate": True,
                            "existing_node_id": sem_best_id,
                            "existing_label": sem_best_label,
                            "similarity": round(sem_best_score, 2),
                            "message": (
                                f"语义重复: 已存在相似节点 {sem_best_id}（{sem_best_label}），"
                                f"相似度 {sem_best_score:.0%}。"
                                f"请直接操作该节点，不要添加新节点。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # ── 守卫 B: 节点数量软警告 ──
            leaf_nodes = tg.get_leaf_nodes()
            leaf_count = len(leaf_nodes)
            if MAX_LEAF_NODES > 0 and leaf_count >= MAX_LEAF_NODES:
                uncompleted = [n for n in leaf_nodes if n.status != "completed"][:5]
                hint_nodes = ", ".join(f"{n.id}({n.label})" for n in uncompleted)
                logger.info(
                    f"[task_add_node] 节点数量达上限: {leaf_count}/{MAX_LEAF_NODES}"
                )
                return self._create_result(
                    success=True,
                    data={
                        "cap_reached": True,
                        "leaf_count": leaf_count,
                        "max_allowed": MAX_LEAF_NODES,
                        "message": (
                            f"任务图已有 {leaf_count} 个工作项，达到上限。"
                            f"请调用 task_view_overview 查看现有节点，"
                            f"然后用 task_mark_status 逐个执行。"
                            f"待执行: {hint_nodes}"
                        ),
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            elif leaf_count >= LEAF_SOFT_WARNING_THRESHOLD:
                logger.info(
                    f"[task_add_node] 节点数量软警告: {leaf_count}/{LEAF_SOFT_WARNING_THRESHOLD}"
                )

            # ── 守卫 C: parent_id 有效性检查 ──
            # 4B 模型可能传入不存在的 parent_id（如 '??'），
            # 导致创建孤儿节点、前端图谱渲染崩溃
            if parent_id != "req" and tg.get_node(parent_id) is None:
                logger.warning(
                    f"[task_add_node] 无效 parent_id '{parent_id}'，"
                    f"自动降级为 'req'"
                )
                parent_id = "req"

            # 生成节点 ID（基于已有最大后缀 +1，避免删除后碰撞）
            children = tg.get_children(parent_id)
            if parent_id == "req":
                max_idx = 0
                for c in children:
                    if c.id.startswith("o") and c.id[1:].isdigit():
                        max_idx = max(max_idx, int(c.id[1:]))
                node_id = f"o{max_idx + 1}"
            else:
                prefix = f"{parent_id}_"
                max_idx = 0
                for c in children:
                    if c.id.startswith(prefix) and c.id[len(prefix):].isdigit():
                        max_idx = max(max_idx, int(c.id[len(prefix):]))
                node_id = f"{parent_id}_{max_idx + 1}"

            # 根据深度确定类型
            depth = tg.get_node_depth(parent_id) + 1
            node_type = tg.depth_to_type(depth)

            node = tg.add_node(
                id=node_id,
                label=label,
                type=node_type,
                status="pending",
                desc=desc or label,
            )

            tg.add_h_edge(parent_id, node_id)

            # 拆解只改变结构，不代表父节点执行完成。
            # 父节点完成状态必须来自子节点完成级联或最终答案写入。
            parent = tg.get_node(parent_id)
            if parent and parent.status == "in_progress":
                parent.metadata["decomposed_child_count"] = len(tg.get_children(parent_id))
                parent.metadata["decomposed_at"] = time.time()
                logger.info(
                    f"[task_add_node] 父节点 {parent_id} 已拆解，保持 {parent.status} 状态"
                    f"（{len(tg.get_children(parent_id))}个子任务）"
                )

            logger.info(f"[task_add_node] 添加节点 {node_id} ({node_type}): {label}")

            _save_active_backup()  # 磁盘备份

            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "type": node_type,
                    "label": label,
                    "parent_id": parent_id,
                    "depth": depth,
                    "hint": f"可用 parent_id='{node_id}' 为此节点添加子步骤" if depth < 5 else "",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_add_node] 添加失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"节点添加失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": "父节点 ID。顶层模块/阶段用 'req'（根节点），子步骤用所属模块节点的 ID",
                },
                "label": {
                    "type": "string",
                    "description": "节点名称",
                },
                "desc": {
                    "type": "string",
                    "description": "节点详细描述（可选）",
                },
            },
            "required": ["parent_id", "label"],
        }


class TaskMarkStatusTool(BaseTool):
    """task_mark_status — 更新任务节点状态"""

    def __init__(self):
        super().__init__(name="task_mark_status", category=ToolCategory.CUSTOM)
        self.description = (
            "更新任务节点的执行状态。"
            "当开始执行、完成或遇到阻塞时调用。"
            "完成时请提供 result 说明执行结果。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "")
        status = request.parameters.get("status", "")
        result = request.parameters.get("result", "")

        # 4B 模型常在参数值中多加引号，如 '"o3"' 或 '"in_progress"'
        if isinstance(node_id, str):
            node_id = node_id.strip().strip('"').strip("'")
        if isinstance(status, str):
            status = status.strip().strip('"').strip("'")
        if isinstance(result, str):
            result = result.strip().strip('"').strip("'")

        valid_statuses = {"pending", "in_progress", "completed", "blocked", "skipped"}

        if not node_id or not status:
            return self._create_result(
                success=False,
                error="node_id 和 status 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if status not in valid_statuses:
            return self._create_result(
                success=False,
                error=f"无效状态 '{status}'，有效值: {valid_statuses}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            node = tg.get_node(node_id)
            if node is None:
                # ── 模糊匹配守卫：尝试纠正错误的 node_id ──
                resolved_id, confidence, method = _fuzzy_resolve_node_id(tg, node_id)

                if resolved_id and confidence >= FUZZY_AUTO_CORRECT_THRESHOLD:
                    # 高置信度：自动纠正
                    logger.info(
                        f"[task_mark_status][FuzzyResolve] '{node_id}' → '{resolved_id}' "
                        f"(conf={confidence:.2f}, method={method})"
                    )
                    node = tg.get_node(resolved_id)
                    node_id = resolved_id  # 后续逻辑使用纠正后的 ID

                elif resolved_id and confidence >= 0.5:
                    # 中等置信度：返回候选建议，不自动纠正
                    resolved_node = tg.get_node(resolved_id)
                    resolved_label = resolved_node.label if resolved_node else ""
                    return self._create_result(
                        success=False,
                        error=(
                            f"节点 '{node_id}' 不存在。"
                            f"你是否想操作: {resolved_id}（{resolved_label}）？"
                            f"请用 task_mark_status(node_id='{resolved_id}', "
                            f"status='{status}') 重新调用。"
                        ),
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

                else:
                    # 无匹配或低置信度：返回可用节点列表
                    leaves = tg.get_leaf_nodes()
                    node_list = ", ".join(
                        f"{n.id}({n.label})" for n in leaves[:10]
                    )
                    return self._create_result(
                        success=False,
                        error=(
                            f"节点 '{node_id}' 不存在。"
                            f"可用节点: {node_list}。"
                            f"请使用正确的 node_id。"
                        ),
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # Rule A: 门卫检查（在更新状态之前）
            # 如果标记为 completed 且该节点有子节点，检查子任务完成情况
            if status == "completed":
                children = tg.get_children(node_id)
                if children:
                    uncompleted_children = [c for c in children if c.status != "completed"]
                    if uncompleted_children:
                        _reject_msg = (
                            f"操作被拒绝：节点 {node_id} 有 {len(uncompleted_children)} "
                            f"个子任务未完成："
                        )
                        for _uc in uncompleted_children:
                            _reject_msg += f"\n  - {_uc.id}: {_uc.label} ({_uc.status})"
                        _reject_msg += "\n请先完成这些子任务，再标记父节点为 completed。"
                        logger.info(f"[task_mark_status] Rule A 拒绝: {node_id} 有未完成子任务")
                        return self._create_result(
                            success=False,
                            error=_reject_msg,
                            execution_time=time.time() - start_time,
                            request_id=request.request_id,
                        )

            # Rule B: result 质量门卫（标记 completed 时验证 result 内容）
            # 防止 LLM 将错误/限流响应当作有效结果标记完成
            _result_text = ""
            _rule_b_reject = None
            if status == "completed" and node_id != "req":
                _result_text = (result or "").strip()

                if not _result_text:
                    _rule_b_reject = (
                        f"操作被拒绝：标记 {node_id} 为 completed 时必须提供 result 参数，"
                        "至少 50 字说明关键产出。"
                    )
                elif len(_result_text) < 20:
                    _rule_b_reject = (
                        f"操作被拒绝：result 过短（{len(_result_text)} 字），"
                        "请至少提供 50 字说明关键产出、核心结论或文件路径。"
                    )
                else:
                    # 检测常见错误/限流/超时响应模式
                    _error_patterns = [
                        "抱歉", "响应较慢", "请稍后再试", "请稍后",
                        "rate limit", "timeout", "timed out",
                        "too many requests", "server error",
                        "internal error", "服务繁忙", "请求过于频繁",
                        "任务执行中断", "强制收敛", "未产出",
                        "进行中但未产出", "尚未完成", "还未完成",
                        "未完成节点", "未完成", "待完成", "待开始",
                        "需要继续执行", "是否继续", "请继续执行",
                        "未生成", "未创建", "没有产出",
                        "无法完成", "不能完成", "触发循环保护",
                        "系统当前出问题", "stalled", "blocked", "interrupted",
                    ]
                    _lower = _result_text.lower()
                    for _pat in _error_patterns:
                        if _pat.lower() in _lower:
                            _rule_b_reject = (
                                f"操作被拒绝：result 内容疑似错误响应"
                                f"（匹配 '{_pat}'）。"
                                f"当前 result: '{_result_text[:80]}...'\n"
                                "请实际完成该任务后再标记为 completed，"
                                "或将状态改为 blocked。"
                            )
                            break

            if _rule_b_reject:
                logger.info(
                    f"[task_mark_status] Rule B 拒绝: {node_id}, "
                    f"result='{(_result_text or '')[:60]}'"
                )
                return self._create_result(
                    success=False,
                    error=_rule_b_reject,
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # Rule C: 文件产出真实性门卫。
            # 如果节点标题/描述明确包含 game.js、README.md 等文件名，
            # 标记完成前必须能在当前任务工作区看到真实文件。
            if status == "completed" and node_id != "req":
                missing_files, workspace_for_check = _missing_expected_output_files(tg, node)
                if missing_files:
                    workspace_hint = (
                        f" 当前任务工作区: {workspace_for_check}。"
                        if workspace_for_check else " 当前任务缺少绑定工作区。"
                    )
                    reject_msg = (
                        f"操作被拒绝：节点 {node_id}（{node.label}）声明产出文件，"
                        f"但未检测到真实文件: {', '.join(missing_files)}。"
                        f"{workspace_hint}"
                        "请先调用 ide_write_file 等真实写入工具完成落盘，"
                        "再标记该节点为 completed。"
                    )
                    logger.info(
                        f"[task_mark_status] Rule C 拒绝: {node_id}, missing={missing_files}"
                    )
                    return self._create_result(
                        success=False,
                        error=reject_msg,
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            tg.update_node_status(node_id, status, result=result or None)

            logger.info(f"[task_mark_status] {node_id} → {status}")

            _save_active_backup()  # 磁盘备份

            # 🔥 [P0] 检测任务整体完成 → 自动归档
            if status == "completed":
                req_node = tg.get_node("req")
                if req_node and req_node.status == "completed":
                    _auto_archive_completed(tg)

            # Rule E: 大纲阶段软警告（仅信息性提示，不阻止操作）
            _outline_hint = ""
            if status == "in_progress":
                parent_id = tg.get_parent(node_id)
                if parent_id and parent_id == "req":
                    siblings = tg.get_children("req")
                    if len(siblings) <= 2:
                        _outline_hint = (
                            f"\n提示：当前任务图只有 {len(siblings)} 个子任务节点。"
                            "建议先用 task_add_node 搭建完整大纲，再开始执行。"
                            "不过如果你确认大纲已经完整，可以继续。"
                        )

            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "status": status,
                    "label": node.label,
                    "message": f"节点 {node_id} ({node.label}) 状态已更新为 {status}" + _outline_hint,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_mark_status] 更新失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"状态更新失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要更新的节点 ID",
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "blocked", "skipped"],
                    "description": "新状态",
                },
                "result": {
                    "type": "string",
                    "description": "完成时必填，至少50字。写明关键产出、核心结论或文件路径。示例：'已生成 api_server.py（120行），实现了 REST API 服务，支持用户注册和登录接口'",
                },
            },
            "required": ["node_id", "status"],
        }


class TaskRollbackNodeTool(BaseTool):
    """task_rollback_node — 回滚任务节点到pending状态（失败后重试）"""

    def __init__(self):
        super().__init__(name="task_rollback_node", category=ToolCategory.CUSTOM)

    @property
    def description(self) -> str:
        return "回滚任务节点及其子节点到pending状态，清除执行结果，允许重新执行"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "要回滚的节点ID"},
                "include_children": {
                    "type": "boolean",
                    "description": "是否同时回滚所有子节点（默认true）",
                    "default": True,
                },
            },
            "required": ["node_id"],
        }

    def execute(self, params: Dict[str, Any]) -> Any:
        node_id = params.get("node_id", "")
        include_children = params.get("include_children", True)
        tg = get_active_task_graph()
        if not tg:
            return {"error": "无活跃任务图"}
        node = tg.get_node(node_id)
        if not node:
            return {"error": f"节点 {node_id} 不存在"}
        if include_children:
            count = tg.rollback_subtree(node_id)
            return {"success": True, "rolled_back": count, "message": f"已回滚 {count} 个节点"}
        else:
            ok = tg.rollback_node(node_id)
            return {"success": ok, "message": "节点已回滚" if ok else "节点不可回滚"}


class TaskViewOverviewTool(BaseTool):
    """task_view_overview — 查看任务图全局概览"""

    def __init__(self):
        super().__init__(name="task_view_overview", category=ToolCategory.CUSTOM)
        self.description = (
            "查看当前任务图的全局概览，包括所有节点的层次结构、"
            "状态和进度。帮助你了解整体任务进展。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=True,
                data={"message": "当前没有活跃的任务图", "overview": ""},
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            overview = tg.to_planning_table()

            # 统计节点状态，明确标注下一步应执行哪个节点
            total = 0
            completed = 0
            next_pending_id = None
            next_pending_label = None
            for nid, node in tg._nodes.items():
                total += 1
                if node.status == "completed":
                    completed += 1
                elif node.status in ("pending", "not_started") and next_pending_id is None and nid != "req":
                    next_pending_id = nid
                    next_pending_label = node.label

            # 获取所有未完成的叶子节点（实际工作项）
            leaf_nodes = tg.get_leaf_nodes()
            uncompleted_leaves = [n for n in leaf_nodes if n.status != "completed"]

            progress_hint = f"进度: {completed}/{total} 已完成。"
            if uncompleted_leaves:
                progress_hint += f"\n⚠️ 还有 {len(uncompleted_leaves)} 个工作项未完成："
                for _ul in uncompleted_leaves:
                    _st = {"pending": "待开始", "not_started": "待开始",
                           "in_progress": "进行中", "blocked": "阻塞"}.get(_ul.status, _ul.status)
                    progress_hint += f"\n  - {_ul.id}: {_ul.label} ({_st})"
                progress_hint += f"\n请从 {uncompleted_leaves[0].id}（{uncompleted_leaves[0].label}）开始执行。"
                progress_hint += "\n注意：不要用 task_add_node 添加新节点，现有节点已经完整。"
            elif completed == total:
                progress_hint += " 所有节点已完成。"

            return self._create_result(
                success=True,
                data={
                    "graph_id": _active_graph_id,
                    "overview": overview,
                    "progress": progress_hint,
                    "total_nodes": total,
                    "completed_nodes": completed,
                    "next_pending_node_id": next_pending_id,
                    "uncompleted_count": len(uncompleted_leaves),
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_view_overview] 查看失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"概览获取失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }


def _run_async(coro):
    """在同步上下文中运行异步协程"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result(timeout=10)
    else:
        return asyncio.run(coro)


class TaskSuspendTool(BaseTool):
    """task_suspend — 挂起当前任务到磁盘"""

    def __init__(self):
        super().__init__(name="task_suspend", category=ToolCategory.CUSTOM)
        self.description = (
            "挂起当前正在执行的任务，将完整状态持久化到磁盘。"
            "适用于任务过于复杂需要分阶段完成、用户要求暂停、"
            "或需要切换到其他任务的情况。挂起的任务后续可以恢复。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        reason = request.parameters.get("reason", "user_requested")
        description = request.parameters.get("description", "")

        try:
            from zulong.l2.task_suspension import TaskSuspensionManager, SuspendableTaskState

            tg = get_active_task_graph()
            if not description and tg:
                root = tg.get_node("req")
                description = root.label if root else "未命名任务"

            if not description:
                description = "未命名任务"

            # 从 ToolEngine 上下文获取当前对话信息
            messages = []
            try:
                from zulong.tools.tool_engine import ToolEngine
                te = ToolEngine()
                ctx = te.get_context()
                if isinstance(ctx, dict):
                    user_input = ctx.get("user_input", "")
                    if user_input:
                        messages.append({"role": "user", "content": user_input})
            except Exception:
                pass

            state = SuspendableTaskState(
                task_id=TaskSuspensionManager.generate_task_id(),
                description=description,
                messages=messages,
                accumulated_links="",
                circuit_breaker_state={},
                iteration_count=0,
                task_graph=tg,
                suspended_reason=reason,
                metadata={"graph_id": _active_graph_id or ""},
            )

            mgr = TaskSuspensionManager()
            task_id = _run_async(mgr.suspend_task(state))

            if task_id:
                set_active_task_graph(None, None)
                logger.info(f"[task_suspend] 任务已挂起: {task_id}")
                return self._create_result(
                    success=True,
                    data={
                        "task_id": task_id,
                        "description": description,
                        "reason": reason,
                        "message": (
                            f"任务 '{description}' 已挂起 (ID: {task_id})。"
                            "后续由 LLM 根据用户语义判断是否恢复该任务。"
                        ),
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            else:
                return self._create_result(
                    success=False,
                    error="任务挂起失败",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

        except Exception as e:
            logger.error(f"[task_suspend] 挂起失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"任务挂起失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": ["user_requested", "complexity", "time_limit"],
                    "description": "挂起原因",
                },
                "description": {
                    "type": "string",
                    "description": "任务描述（用于后续恢复时匹配，不填则从任务图获取）",
                },
            },
            "required": [],
        }


class TaskListSuspendedTool(BaseTool):
    """task_list_suspended — 列出所有挂起的任务"""

    def __init__(self):
        super().__init__(name="task_list_suspended", category=ToolCategory.CUSTOM)
        self.description = (
            "列出所有已挂起的任务。当 LLM 判断用户语义是在恢复旧任务时，"
            "先调用此工具查看有哪些挂起的任务，然后决定恢复哪个。"
            "也可以传入 query 参数来按描述模糊匹配。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        query = request.parameters.get("query", "")
        workspace_hint = (
            request.parameters.get("workspace_path")
            or request.parameters.get("workspace_dir")
            or get_active_workspace_dir()
            or ""
        )

        try:
            active_tg = get_active_task_graph()
            if query and active_tg:
                active_graph_id = normalize_task_graph_id(getattr(active_tg, "id", ""))
                active_title = getattr(active_tg, "title", "") or ""
                try:
                    root = active_tg.get_node("req")
                    if root and getattr(root, "label", ""):
                        active_title = root.label
                except Exception:
                    pass
                active_workspace = (
                    getattr(active_tg, "metadata", {}).get("workspace_dir")
                    or get_active_workspace_dir()
                    or ""
                )
                same_workspace = bool(
                    workspace_hint
                    and active_workspace
                    and _normalize_path_for_match(workspace_hint) == _normalize_path_for_match(active_workspace)
                )
                query_related = (
                    (active_graph_id and active_graph_id in str(query))
                    or _titles_related(active_title, str(query))
                    or str(query).lower() in active_title.lower()
                )
                if same_workspace and query_related:
                    logger.info(
                        "[task_list_suspended] 当前活跃图已匹配查询，跳过挂起任务恢复: graph=%s query=%s",
                        active_graph_id,
                        query,
                    )
                    return self._create_result(
                        success=True,
                        data={
                            "resumed": False,
                            "already_active": True,
                            "graph_id": active_graph_id,
                            "title": active_title,
                            "workspace_dir": active_workspace,
                            "message": (
                                f"当前任务图「{active_title}」已绑定到该工作区，"
                                "无需从挂起任务恢复；请继续查看或扩展当前任务图。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            from zulong.l2.task_suspension import TaskSuspensionManager
            mgr = TaskSuspensionManager()

            if query:
                candidates = []
                for summary in _run_async(mgr.list_suspended_tasks()):
                    task_id = summary.get("task_id")
                    if not task_id:
                        continue
                    state = _run_async(mgr.resume_task(task_id, consume=False))
                    if not state:
                        continue
                    score = _score_suspend_candidate(state, query, workspace_hint)
                    if score <= 0:
                        continue
                    candidates.append((score, state))

                if not candidates:
                    return self._create_result(
                        success=True,
                        data={"tasks": [], "message": f"没有找到匹配 '{query}' 的挂起任务"},
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

                candidates.sort(
                    key=lambda item: (
                        item[0],
                        getattr(item[1], "suspended_at", 0) or 0,
                    ),
                    reverse=True,
                )
                match = candidates[0][1]

                if hasattr(match, 'task_id'):
                    if match.task_graph:
                        _ws = match.task_graph.metadata.get("workspace_dir", "") if hasattr(match.task_graph, 'metadata') else ""
                        matched_graph_id = normalize_task_graph_id(match.metadata.get("graph_id", ""))
                        conversation_id, session_node_id = _request_owner_context(request)
                        restored = set_active_task_graph(
                            match.task_graph,
                            matched_graph_id,
                            workspace_dir=_ws,
                            conversation_id=conversation_id,
                            session_node_id=session_node_id,
                            claim_unowned=_interaction_store_claims_graph(conversation_id, matched_graph_id),
                        )
                        if not restored:
                            return self._create_result(
                                success=False,
                                error=(
                                    f"挂起任务图 {matched_graph_id or '-'} 不属于当前会话，"
                                    "已拒绝恢复以避免跨会话改绑。"
                                ),
                                execution_time=time.time() - start_time,
                                request_id=request.request_id,
                            )
                        logger.info(f"[task_list_suspended] 已恢复任务图: {match.task_id}")

                    # 确认恢复成功，显式消费（删除）磁盘文件
                    try:
                        _run_async(mgr.cancel_task(match.task_id))
                    except Exception:
                        pass

                    return self._create_result(
                        success=True,
                        data={
                            "resumed": True,
                            "task_id": match.task_id,
                            "description": match.description,
                            "iteration_count": match.iteration_count,
                            "messages_count": len(match.messages),
                            "has_task_graph": match.task_graph is not None,
                            "workspace_dir": _ws if match.task_graph else "",
                            "candidate_count": len(candidates),
                            "message": (
                                f"已恢复任务 '{match.description}' (ID: {match.task_id})。"
                                + (f" TaskGraph 已加载。" if match.task_graph else "")
                                + " 请继续执行。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                else:
                    return self._create_result(
                        success=True,
                        data={"tasks": [match], "message": "找到匹配的挂起任务"},
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
            else:
                tasks = _run_async(mgr.list_suspended_tasks())
                return self._create_result(
                    success=True,
                    data={
                        "tasks": tasks,
                        "count": len(tasks),
                        "message": f"共 {len(tasks)} 个挂起的任务" if tasks else "没有挂起的任务",
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

        except Exception as e:
            logger.error(f"[task_list_suspended] 查询失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"查询挂起任务失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于模糊匹配任务描述的关键词（可选，不填则列出全部）",
                },
                "workspace_path": {
                    "type": "string",
                    "description": "可选：用于恢复候选打分的目标工作区绝对路径，例如 D:/AI/project/tank。",
                },
            },
            "required": [],
        }


# ═══════════════════════════════════════════════════════════════
# CRUD 补全工具集（阶段 1）
# task_add_dependency / task_get_detail / task_update_node / task_remove_node
# ═══════════════════════════════════════════════════════════════


class TaskAddDependencyTool(BaseTool):
    """task_add_dependency — 添加任务节点间的依赖关系"""

    def __init__(self):
        super().__init__(name="task_add_dependency", category=ToolCategory.CUSTOM)
        self.description = (
            "在两个任务节点之间添加依赖关系。"
            "source_id 是前置任务（先完成的），target_id 是后续任务（依赖前者的）。"
            "系统会自动检测循环依赖并拒绝。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        source_id = request.parameters.get("source_id", "").strip().strip('"').strip("'")
        target_id = request.parameters.get("target_id", "").strip().strip('"').strip("'")
        via = request.parameters.get("via", "").strip()

        if not source_id or not target_id:
            return self._create_result(
                success=False,
                error="source_id 和 target_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            # 校验节点存在
            if tg.get_node(source_id) is None:
                return self._create_result(
                    success=False,
                    error=f"源节点 '{source_id}' 不存在",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            if tg.get_node(target_id) is None:
                return self._create_result(
                    success=False,
                    error=f"目标节点 '{target_id}' 不存在",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 检查是否已存在相同依赖边
            for edge in tg._d_edges:
                if edge.s == source_id and edge.t == target_id:
                    return self._create_result(
                        success=True,
                        data={
                            "already_exists": True,
                            "message": f"依赖关系 {source_id} → {target_id} 已存在",
                        },
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # 先临时添加边，再做环检测
            from zulong.l2.task_graph import DependencyEdge, TaskScheduler
            tg._d_edges.append(DependencyEdge(s=source_id, t=target_id, via=via))

            scheduler = TaskScheduler(tg)
            is_valid, msg = scheduler.validate_dependencies()
            if not is_valid:
                # 有环，回滚
                tg._d_edges.pop()
                return self._create_result(
                    success=False,
                    error=f"添加依赖会造成循环依赖: {msg}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 边已添加，标记 dirty
            tg._mark_dirty()

            # 同步到 MemoryGraph
            try:
                from zulong.memory.memory_graph import get_memory_graph, EdgeType
                mg = get_memory_graph()
                if mg:
                    src_mem_id = f"task:{tg.id}/{source_id}"
                    tgt_mem_id = f"task:{tg.id}/{target_id}"
                    mg.add_edge(src_mem_id, tgt_mem_id, EdgeType.DEPENDENCY)
            except Exception as e:
                logger.debug(f"[task_add_dependency] MemoryGraph 同步跳过: {e}")

            logger.info(f"[task_add_dependency] 添加依赖: {source_id} → {target_id}")
            _save_active_backup()

            return self._create_result(
                success=True,
                data={
                    "source_id": source_id,
                    "target_id": target_id,
                    "via": via,
                    "message": f"已添加依赖: {source_id} → {target_id}" + (f" ({via})" if via else ""),
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_add_dependency] 失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"添加依赖失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "前置任务节点 ID（先完成的）",
                },
                "target_id": {
                    "type": "string",
                    "description": "后续任务节点 ID（依赖前者的）",
                },
                "via": {
                    "type": "string",
                    "description": "依赖传递的数据描述（可选）",
                },
            },
            "required": ["source_id", "target_id"],
        }


class TaskGetDetailTool(BaseTool):
    """task_get_detail — 读取指定节点的完整详情"""

    def __init__(self):
        super().__init__(name="task_get_detail", category=ToolCategory.CUSTOM)
        self.description = (
            "读取任务图中指定节点的完整详情，包括标签、描述、状态、"
            "产出结果（result）、父节点、子节点、依赖关系等。"
            "用于在执行阶段查看前置任务的产出或检查节点信息。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "").strip().strip('"').strip("'")

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            node = tg.get_node(node_id)
            if node is None:
                # 模糊匹配
                resolved_id, confidence, method = _fuzzy_resolve_node_id(tg, node_id)
                if resolved_id and confidence >= FUZZY_AUTO_CORRECT_THRESHOLD:
                    node = tg.get_node(resolved_id)
                    node_id = resolved_id
                else:
                    leaves = tg.get_leaf_nodes()
                    node_list = ", ".join(f"{n.id}({n.label})" for n in leaves[:10])
                    return self._create_result(
                        success=False,
                        error=f"节点 '{node_id}' 不存在。可用节点: {node_list}",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # 获取关系信息
            parent_id = tg.get_parent(node_id)
            children = tg.get_children(node_id)
            dep_ids = tg.get_dependencies(node_id)
            dependent_ids = tg.get_dependents(node_id)

            detail = {
                "node_id": node_id,
                "label": node.label,
                "type": node.type,
                "status": node.status,
                "desc": node.desc,
                "result": node.result,
                "files": [f.to_dict() for f in node.files] if node.files else [],
                "parent_id": parent_id,
                "children": [{"id": c.id, "label": c.label, "status": c.status} for c in children],
                "dependencies": dep_ids,
                "dependents": dependent_ids,
                "depth": tg.get_node_depth(node_id),
            }

            logger.info(f"[task_get_detail] 查询节点: {node_id}")
            return self._create_result(
                success=True,
                data=detail,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_get_detail] 查询失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"节点详情查询失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要查询的节点 ID",
                },
            },
            "required": ["node_id"],
        }


class TaskUpdateNodeTool(BaseTool):
    """task_update_node — 修改已有节点的标签、描述或产出内容"""

    def __init__(self):
        super().__init__(name="task_update_node", category=ToolCategory.CUSTOM)
        self.description = (
            "修改任务图中已有节点的标签（label）、描述（desc）或产出内容（result）。"
            "只修改传入的字段，未传入的字段保持不变。"
            "用于在规划阶段调整任务定义或在执行阶段更新产出。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "").strip().strip('"').strip("'")
        new_label = request.parameters.get("label")
        new_desc = request.parameters.get("desc")
        new_result = request.parameters.get("result")

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if new_label is None and new_desc is None and new_result is None:
            return self._create_result(
                success=False,
                error="至少需要提供 label、desc 或 result 中的一个字段",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            node = tg.get_node(node_id)
            if node is None:
                # 模糊匹配
                resolved_id, confidence, method = _fuzzy_resolve_node_id(tg, node_id)
                if resolved_id and confidence >= FUZZY_AUTO_CORRECT_THRESHOLD:
                    node_id = resolved_id
                else:
                    return self._create_result(
                        success=False,
                        error=f"节点 '{node_id}' 不存在",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # 清理参数
            if isinstance(new_label, str):
                new_label = new_label.strip().strip('"').strip("'") or None
            if isinstance(new_desc, str):
                new_desc = new_desc.strip().strip('"').strip("'") or None
            if isinstance(new_result, str):
                new_result = new_result.strip().strip('"').strip("'") or None

            ok = tg.update_node_content(
                node_id,
                label=new_label,
                desc=new_desc,
                result=new_result,
            )
            if not ok:
                return self._create_result(
                    success=False,
                    error=f"节点 '{node_id}' 更新失败",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            logger.info(f"[task_update_node] 更新节点: {node_id}")
            _save_active_backup()

            updated_node = tg.get_node(node_id)
            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "label": updated_node.label,
                    "desc": updated_node.desc[:200] if updated_node.desc else "",
                    "result_updated": new_result is not None,
                    "message": f"节点 {node_id} 内容已更新",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_update_node] 更新失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"节点更新失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要修改的节点 ID",
                },
                "label": {
                    "type": "string",
                    "description": "新的节点标签（可选，不传则不修改）",
                },
                "desc": {
                    "type": "string",
                    "description": "新的节点描述（可选，不传则不修改）",
                },
                "result": {
                    "type": "string",
                    "description": "新的产出内容（可选，不传则不修改）",
                },
            },
            "required": ["node_id"],
        }


class TaskRemoveNodeTool(BaseTool):
    """task_remove_node — 删除任务节点及其后代"""

    def __init__(self):
        super().__init__(name="task_remove_node", category=ToolCategory.CUSTOM)
        self.description = (
            "从任务图中删除指定节点及其所有后代节点。"
            "不能删除根节点（req）和需求分析节点（analysis）。"
            "删除后相关的层级边和依赖边会自动清理。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "").strip().strip('"').strip("'")

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            # 检查是否尝试删除受保护节点
            if node_id in ("req", "analysis"):
                return self._create_result(
                    success=False,
                    error=f"不能删除受保护的节点: {node_id}",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            node = tg.get_node(node_id)
            if node is None:
                return self._create_result(
                    success=False,
                    error=f"节点 '{node_id}' 不存在",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 检查是否有其他已完成节点依赖此节点
            dependent_ids = tg.get_dependents(node_id)
            completed_dependents = [
                d for d in dependent_ids
                if tg.get_node(d) and tg.get_node(d).status == "completed"
            ]
            warning = ""
            if completed_dependents:
                warning = (
                    f"注意：有 {len(completed_dependents)} 个已完成节点依赖此节点"
                    f"（{', '.join(completed_dependents)}），删除后这些依赖关系将被清除。"
                )

            label = node.label
            removed_ids = tg.remove_node(node_id)

            if not removed_ids:
                return self._create_result(
                    success=False,
                    error=f"删除节点 '{node_id}' 失败",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 同步到 MemoryGraph：标记为 pruned
            try:
                from zulong.memory.memory_graph import get_memory_graph
                mg = get_memory_graph()
                if mg:
                    for rid in removed_ids:
                        mem_node_id = f"task:{tg.id}/{rid}"
                        mem_node = mg.get_node(mem_node_id)
                        if mem_node:
                            mem_node.metadata["task_status"] = "pruned"
            except Exception as e:
                logger.debug(f"[task_remove_node] MemoryGraph 同步跳过: {e}")

            logger.info(f"[task_remove_node] 删除节点: {node_id} ({label}), 共移除 {len(removed_ids)} 个节点")
            _save_active_backup()

            data = {
                "node_id": node_id,
                "label": label,
                "removed_count": len(removed_ids),
                "removed_ids": removed_ids,
                "message": f"已删除节点 {node_id}（{label}）及 {len(removed_ids) - 1} 个后代",
            }
            if warning:
                data["warning"] = warning

            return self._create_result(
                success=True,
                data=data,
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_remove_node] 删除失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"节点删除失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要删除的节点 ID（不能删除 req 和 analysis）",
                },
            },
            "required": ["node_id"],
        }


class TaskUpdateContentTool(BaseTool):
    """task_update_content — 向节点写入分析内容和语义摘要

    专为项目级分析设计：将分析正文和语义摘要保存到 TaskNode，
    使节点成为结构化知识容器。支持内容修订（自动递增版本号）。
    """

    def __init__(self):
        super().__init__(name="task_update_content", category=ToolCategory.CUSTOM)
        self.description = (
            "向任务节点写入分析内容（analysis_content）和语义摘要（semantic_summary）。"
            "分析内容无长度限制，用于保存详细分析；语义摘要≤500字，用于检索和上下文恢复。"
            "每次更新自动递增 content_version，支持追踪修订历史。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "").strip().strip('"').strip("'")
        content = request.parameters.get("content")
        summary = request.parameters.get("summary")
        append = request.parameters.get("append", False)

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if content is None and summary is None:
            return self._create_result(
                success=False,
                error="至少需要提供 content 或 summary 中的一个",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            node = tg.get_node(node_id)
            if node is None:
                resolved_id, confidence, _ = _fuzzy_resolve_node_id(tg, node_id)
                if resolved_id and confidence >= FUZZY_AUTO_CORRECT_THRESHOLD:
                    node_id = resolved_id
                    node = tg.get_node(node_id)
                else:
                    return self._create_result(
                        success=False,
                        error=f"节点 '{node_id}' 不存在",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            # 处理分析内容
            analysis = content
            if isinstance(analysis, str):
                analysis = analysis.strip() or None
            if analysis is not None and append and node.analysis_content:
                analysis = node.analysis_content + "\n\n" + analysis

            # 处理语义摘要
            sem_summary = summary
            if isinstance(sem_summary, str):
                sem_summary = sem_summary.strip()[:500] or None

            ok = tg.update_node_content(
                node_id,
                analysis_content=analysis,
                semantic_summary=sem_summary,
            )
            if not ok:
                return self._create_result(
                    success=False,
                    error=f"节点 '{node_id}' 内容更新失败",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            logger.info(
                f"[task_update_content] 更新节点内容: {node_id} "
                f"(version={node.content_version})")
            _save_active_backup()

            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "content_version": node.content_version,
                    "content_length": len(node.analysis_content or ""),
                    "summary_length": len(node.semantic_summary or ""),
                    "message": f"节点 {node_id} 分析内容已更新 (v{node.content_version})",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_update_content] 更新失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"内容更新失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "目标节点 ID",
                },
                "content": {
                    "type": "string",
                    "description": "分析正文内容（无长度限制）",
                },
                "summary": {
                    "type": "string",
                    "description": "语义摘要（≤500字，用于检索和上下文恢复）",
                },
                "append": {
                    "type": "boolean",
                    "description": "是否追加到已有内容（默认 false 覆盖）",
                },
            },
            "required": ["node_id"],
        }


class TaskAttachFileTool(BaseTool):
    """task_attach_file — 将文件关联到任务节点

    在项目级分析中，将被分析的源文件关联到对应的 TaskNode，
    建立文件与知识节点的双向映射。
    """

    def __init__(self):
        super().__init__(name="task_attach_file", category=ToolCategory.CUSTOM)
        self.description = (
            "将一个文件关联到任务节点。关联后该文件会出现在节点的文件列表中，"
            "同时会在 MemoryGraph 中创建 FILE 节点和 REFERENCE 边。"
            "用于建立分析内容与源代码文件的映射关系。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "").strip().strip('"').strip("'")
        file_path = request.parameters.get("file_path", "").strip().strip('"').strip("'")
        file_name = request.parameters.get("file_name", "").strip().strip('"').strip("'")

        if not node_id:
            return self._create_result(
                success=False,
                error="node_id 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        if not file_path:
            return self._create_result(
                success=False,
                error="file_path 参数不能为空",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 自动从路径提取文件名
        if not file_name:
            file_name = os.path.basename(file_path)

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        try:
            node = tg.get_node(node_id)
            if node is None:
                resolved_id, confidence, _ = _fuzzy_resolve_node_id(tg, node_id)
                if resolved_id and confidence >= FUZZY_AUTO_CORRECT_THRESHOLD:
                    node_id = resolved_id
                else:
                    return self._create_result(
                        success=False,
                        error=f"节点 '{node_id}' 不存在",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

            ok = tg.add_file_to_node(node_id, file_name, file_path)
            if not ok:
                return self._create_result(
                    success=False,
                    error=f"文件关联失败",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            logger.info(f"[task_attach_file] 关联文件: {node_id} <- {file_path}")
            _save_active_backup()

            # 同步到 MemoryGraph（创建 FILE 节点 + REFERENCE 边）
            try:
                from zulong.memory.memory_graph import get_memory_graph, GraphNode, NodeType, EdgeType
                mg = get_memory_graph()
                if mg:
                    file_mg_id = f"file:{file_path}"
                    if not mg.has_node(file_mg_id):
                        fnode = GraphNode(
                            node_id=file_mg_id,
                            node_type=NodeType.FILE,
                            label=file_name,
                            backend_ref=f"file:{file_path}",
                            metadata={"path": file_path},
                        )
                        mg.add_node(fnode, touch=False)
                    # 查找 TaskNode 对应的 MemoryGraph 节点
                    task_mg_id = None
                    candidate_task_id = f"task:{tg.id}/{node_id}"
                    if mg.has_node(candidate_task_id):
                        task_mg_id = candidate_task_id
                    else:
                        for n in mg.get_nodes_by_type(NodeType.TASK):
                            nid = getattr(n, "node_id", "")
                            meta = getattr(n, "metadata", {}) or {}
                            if nid.endswith(f"/{node_id}") and meta.get("graph_id"):
                                task_mg_id = nid
                                break
                    if task_mg_id:
                        mg.add_edge(task_mg_id, file_mg_id, EdgeType.REFERENCE, weight=0.8)
            except Exception as e:
                logger.debug(f"[task_attach_file] MemoryGraph 同步跳过: {e}")

            updated_node = tg.get_node(node_id)
            return self._create_result(
                success=True,
                data={
                    "node_id": node_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "total_files": len(updated_node.files) if updated_node else 0,
                    "message": f"文件 {file_name} 已关联到节点 {node_id}",
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[task_attach_file] 关联失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"文件关联失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "目标节点 ID",
                },
                "file_path": {
                    "type": "string",
                    "description": "文件路径（相对或绝对路径）",
                },
                "file_name": {
                    "type": "string",
                    "description": "文件显示名称（可选，默认从路径提取）",
                },
            },
            "required": ["node_id", "file_path"],
        }


class SubmitFinalAnswerTool(BaseTool):
    """submit_final_answer — LLM 主动调用以结束当前任务并提交最终答案

    语义：该工具是 FC 循环的终止信号。调用后：
    1. 将 answer 写入活跃 TaskGraph 根节点的 output
    2. 将根节点标记为 completed
    3. 返回 success，FC 循环检测到该工具后应终止迭代
    """

    def __init__(self):
        super().__init__(name="submit_final_answer", category=ToolCategory.CUSTOM)
        self.description = (
            "提交最终回答并结束当前任务。当你认为任务已完成、"
            "可以向用户交付最终结果时调用此工具。参数：answer（最终回答文本）"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        answer = request.parameters.get("answer", "")
        if not answer:
            return self._create_result(
                success=False,
                error="缺少 answer 参数",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is not None:
            ok, guard_error = ensure_task_graph_owner_for_request(
                tg,
                request,
                operation="submit_final_answer",
            )
            if not ok:
                return self._create_result(
                    success=False,
                    error=guard_error,
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            try:
                _write_final_answer_to_task_graph(
                    tg,
                    answer,
                    source="submit_final_answer",
                )
                _auto_archive_completed(tg)
            except Exception as e:
                logger.warning("[SubmitFinalAnswer] 更新 TaskGraph 失败: %s", e)

        return self._create_result(
            success=True,
            data={
                "message": "最终答案已提交，任务完成",
                "answer_length": len(answer),
                "has_task_graph": tg is not None,
            },
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "最终回答内容（Markdown 格式）",
                },
            },
            "required": ["answer"],
        }


# ============================================================
# task_resume_by_address — 通过 MemoryGraph 地址恢复历史任务
# ============================================================

class TaskResumeByAddressTool(BaseTool):
    """task_resume_by_address — 通过 MemoryGraph 节点地址恢复历史 TaskGraph

    当 LLM 持有历史任务节点的地址（格式: tg:{graph_id}/task:{node_id}）时，
    可以通过此工具从 MemoryGraph 轻量级重建 TaskGraph 并将其激活为当前活跃图。
    无需磁盘 I/O，直接利用 MemoryGraph 中已投射的节点和边进行恢复。
    """

    def __init__(self):
        super().__init__(name="task_resume_by_address", category=ToolCategory.CUSTOM)
        self.description = (
            "通过 MemoryGraph 节点地址恢复历史任务图谱。"
            "当你知道一个历史任务节点的地址时调用此工具。"
            "地址格式: tg:{graph_id}/task:{node_id}。"
            "该工具会从记忆图谱中重建整个任务图并将其激活为当前任务。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        address = request.parameters.get("address", "").strip()
        reason = request.parameters.get("reason", "")

        if not address:
            return self._create_result(
                success=False,
                error="缺少 address 参数。格式: tg:{graph_id}/task:{node_id}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        address = normalize_task_graph_address(address)
        graph_id = normalize_task_graph_id(address)
        conversation_id, session_node_id = _request_owner_context(request)
        claim_unowned = _interaction_store_claims_graph(conversation_id, graph_id)

        if not graph_id:
            return self._create_result(
                success=False,
                error=f"无法从地址 '{address}' 中解析 graph_id。"
                      f"期望格式: tg:{{graph_id}}/task:{{node_id}}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 检查是否已经是活跃图
        current_tg = get_active_task_graph()
        if current_tg and getattr(current_tg, 'id', '') == graph_id:
            ok, guard_error = ensure_task_graph_owner_for_request(
                current_tg,
                request,
                operation="task_resume_by_address",
            )
            if not ok:
                return self._create_result(
                    success=False,
                    error=guard_error,
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )
            try:
                overview = current_tg.to_planning_table()
            except Exception:
                overview = f"活跃图: {graph_id}"
            return self._create_result(
                success=True,
                data={
                    "message": f"该任务图谱已是活跃状态: {graph_id}",
                    "graph_id": graph_id,
                    "overview": overview,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 通过 MemoryGraph 重建
        try:
            from zulong.memory.memory_graph import get_memory_graph
            from zulong.memory.graph_adapters import rebuild_task_graph_from_memory

            mg = get_memory_graph()

            # 可选: 先验证地址是否可解析
            node = mg.resolve_address(address)
            if node is None:
                logger.warning(
                    f"[TaskResumeByAddress] 地址无法解析: {address}")
                # 不阻断——仍尝试重建（graph_id 可能有对应的其他节点）

            if load_task_graph_deterministic(
                graph_id,
                conversation_id=conversation_id,
                session_node_id=session_node_id,
                claim_unowned=claim_unowned,
            ):
                rebuilt_tg = get_active_task_graph()
            else:
                rebuilt_tg = rebuild_task_graph_from_memory(mg, graph_id)
            if rebuilt_tg is None:
                return self._create_result(
                    success=False,
                    error=f"无法从 MemoryGraph 重建任务图 {graph_id}（节点数据不足或不存在）",
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 激活
            if not set_active_task_graph(
                rebuilt_tg,
                graph_id,
                conversation_id=conversation_id,
                session_node_id=session_node_id,
                claim_unowned=claim_unowned,
            ):
                return self._create_result(
                    success=False,
                    error=(
                        f"任务图 {graph_id} 不属于当前会话，已拒绝恢复，"
                        "避免跨会话改绑。"
                    ),
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
                )

            # 生成概览
            try:
                overview = rebuilt_tg.to_planning_table()
            except Exception:
                overview = f"已恢复: {graph_id}, 共 {len(rebuilt_tg._nodes)} 个节点"

            # 统计进度
            total = len(rebuilt_tg._nodes)
            completed = sum(
                1 for n in rebuilt_tg._nodes.values()
                if n.status in ("completed", "skipped")
            )
            pending = [
                n for n in rebuilt_tg._nodes.values()
                if n.status in ("pending", "not_started")
            ]

            logger.info(
                f"[TaskResumeByAddress] 成功恢复 graph={graph_id}, "
                f"nodes={total}, completed={completed}, reason={reason}")

            return self._create_result(
                success=True,
                data={
                    "message": f"已从 MemoryGraph 恢复任务图谱: {graph_id}",
                    "graph_id": graph_id,
                    "total_nodes": total,
                    "completed_nodes": completed,
                    "next_pending": pending[0].label if pending else None,
                    "overview": overview,
                },
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        except Exception as e:
            logger.error(f"[TaskResumeByAddress] 恢复失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"恢复任务图谱失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": (
                        "MemoryGraph 节点地址。"
                        "格式: tg:{graph_id}/task:{node_id}"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "恢复原因说明（可选）",
                },
            },
            "required": ["address"],
        }


# ============================================================
# task_revise_node — 修订已完成的任务节点（重新打开）
# ============================================================

class TaskReviseNodeTool(BaseTool):
    """task_revise_node — 修订已完成任务节点的内容

    允许将已完成（completed）的节点重新设置为 in_progress，以便进行修改。
    这是轻量级任务恢复的关键组件——通过地址引用恢复图谱后，
    可以对已完成节点进行修订。
    """

    def __init__(self):
        super().__init__(name="task_revise_node", category=ToolCategory.CUSTOM)
        self.description = (
            "修订已完成的任务节点。将节点状态从 completed 重置为 in_progress，"
            "并可选择性地更新任务描述。当需要修改或补充已完成任务的结果时调用。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        node_id = request.parameters.get("node_id", "").strip().strip('"').strip("'")
        reason = request.parameters.get("reason", "").strip()
        new_desc = request.parameters.get("new_desc", "").strip()

        if not node_id:
            return self._create_result(
                success=False,
                error="缺少 node_id 参数",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        tg = get_active_task_graph()
        if tg is None:
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        node = tg.get_node(node_id)
        if node is None:
            return self._create_result(
                success=False,
                error=f"节点 '{node_id}' 不存在于当前任务图中",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        old_status = node.status
        old_result = node.result

        # 只有 completed 或 skipped 的节点可以被修订
        if old_status not in ("completed", "skipped"):
            return self._create_result(
                success=False,
                error=f"节点 '{node_id}' 当前状态为 '{old_status}'，"
                      f"只有 completed/skipped 状态的节点可以修订",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 重置状态
        node.status = "in_progress"
        if new_desc:
            node.desc = new_desc

        # 保留旧结果到 metadata 供参考
        if old_result:
            if not hasattr(node, 'metadata'):
                node.metadata = {}
            if isinstance(getattr(node, 'metadata', None), dict):
                node.metadata["previous_result"] = old_result
                node.metadata["revise_reason"] = reason

        # 触发变更回调
        if hasattr(tg, 'on_change_callback') and tg.on_change_callback:
            try:
                tg.on_change_callback(tg)
            except Exception:
                pass

        logger.info(
            f"[TaskReviseNode] 节点 {node_id} 已从 {old_status} → in_progress, "
            f"reason={reason}")

        return self._create_result(
            success=True,
            data={
                "message": f"节点 '{node.label}' 已重新打开进行修订",
                "node_id": node_id,
                "old_status": old_status,
                "new_status": "in_progress",
                "reason": reason,
                "previous_result_preserved": bool(old_result),
            },
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "要修订的节点 ID",
                },
                "reason": {
                    "type": "string",
                    "description": "修订原因说明",
                },
                "new_desc": {
                    "type": "string",
                    "description": "新的任务描述（可选，留空保留原描述）",
                },
            },
            "required": ["node_id", "reason"],
        }
