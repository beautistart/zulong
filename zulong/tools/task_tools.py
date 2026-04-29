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
from typing import Dict, Any, Optional, List, Tuple

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)

# 当前活跃的 TaskGraph（模块级单例，模型按需创建）
_active_task_graph = None
_active_graph_id = None
_active_workspace_dir = None  # 当前活跃任务的工作目录绝对路径
_active_graph_lock = threading.RLock()

# 任务图磁盘备份目录
_GRAPH_BACKUP_DIR = os.path.join(".", "data", "graph_backups")

# ─── 守卫常量 ─────────────────────────────────────────────
DUPLICATE_LABEL_THRESHOLD = 0.65   # bigram Jaccard 阈值，>=此值视为重复
MAX_LEAF_NODES = 30                # 叶子节点数量上限
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


def _auto_archive_completed(tg):
    """将已完成的任务图归档到 completed_tasks（幂等，重复调用安全）"""
    try:
        from zulong.l2.task_archive import CompletedTaskArchiveManager, CompletedTaskArchive
        mgr = CompletedTaskArchiveManager()

        root = tg.get_node("req")
        description = root.label if root else getattr(tg, 'title', '未命名任务')
        graph_id = getattr(tg, 'id', '') or _active_graph_id or f"tg_{int(time.time())}"

        archive = CompletedTaskArchive(
            task_id=graph_id,
            description=description,
            final_answer="",
            duration=0,
            total_turns=0,
            completion_status="completed",
            task_graph_snapshot=tg.serialize(),
            workspace_dir=_active_workspace_dir or "",
            metadata={"graph_id": graph_id},
        )

        _run_async(mgr.archive_task(archive))
        logger.info(f"[TaskArchive] 任务已自动归档: {graph_id} ({description})")
    except Exception as e:
        logger.warning(f"[TaskArchive] 自动归档失败（非致命）: {e}")


def get_active_task_graph():
    """获取当前活跃的 TaskGraph"""
    with _active_graph_lock:
        return _active_task_graph


def get_active_workspace_dir():
    """获取当前活跃任务的工作目录路径，无活跃任务时返回 None"""
    with _active_graph_lock:
        return _active_workspace_dir


def set_active_task_graph(tg, graph_id, workspace_dir=None):
    """设置当前活跃的 TaskGraph，并自动备份到磁盘"""
    global _active_task_graph, _active_graph_id, _active_workspace_dir
    with _active_graph_lock:
        _active_task_graph = tg
        _active_graph_id = graph_id
        _active_workspace_dir = workspace_dir
        # 磁盘备份：每次设置活跃图时保存一份，防止数据丢失
        if tg is not None and graph_id:
            _backup_graph_to_disk(tg, graph_id)
        # 同步到 TaskStateManager，保持两套状态一致
        try:
            from zulong.l2.task_state_manager import task_state_manager
            current_tsm_task = task_state_manager.get_active_task()
            if tg is not None and graph_id and current_tsm_task != graph_id:
                task_state_manager.create_task(graph_id, [])
                logger.debug(
                    f"[TaskTools] 已同步 TaskGraph {graph_id} 到 TaskStateManager"
                )
        except Exception as e:
            logger.debug(f"[TaskTools] TaskStateManager 同步跳过: {e}")


def _create_task_workspace(graph_id: str) -> str:
    """为任务创建独立工作目录，返回绝对路径

    目录结构: ./agent_workspace/{YYYYMMDD}_{HHMMSS}_{graph_id}/
    """
    from pathlib import Path
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


def _save_active_backup():
    """将当前活跃任务图备份到磁盘（供工具修改后调用）"""
    with _active_graph_lock:
        if _active_task_graph is not None and _active_graph_id:
            _backup_graph_to_disk(_active_task_graph, _active_graph_id)


class TaskCreatePlanTool(BaseTool):
    """task_create_plan — 创建新任务规划图"""

    def __init__(self):
        super().__init__(name="task_create_plan", category=ToolCategory.CUSTOM)
        self.description = (
            "创建一个新的任务规划图。当用户要求完成复杂的多步骤任务时调用，"
            "如开发项目、设计系统、写报告等。"
            "创建后可用 task_add_node 添加子任务节点。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        title = request.parameters.get("title", "未命名任务")

        try:
            # 🔥 拦截：如果当前有活跃任务图且仍有未完成节点，不创建新图谱
            # 直接返回现有图谱概览，引导模型继续使用 task_view_overview
            old_tg = get_active_task_graph()
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
                    from zulong.tools.session_tool import _titles_related
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
                        logger.info(
                            f"[task_create_plan] 旧图谱 '{old_title}' 已全部完成 "
                            f"({completed}/{total})，新任务 '{title}' 无关，清除后创建新图谱"
                        )
                        set_active_task_graph(None, None)
                        # 继续往下走，创建新图谱
                else:
                    logger.info(f"[task_create_plan] 拦截重复创建：已有活跃图谱 '{old_title}' ({completed}/{total})")
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

            from zulong.l2.task_graph import TaskGraph
            graph_id = f"tg_{int(time.time())}"
            tg = TaskGraph(title=title, graph_id=graph_id)

            # 创建根节点
            root = tg.add_node(
                id="req",
                label=title,
                type="requirement",
                status="in_progress",
                desc=title,
            )

            # 创建独立工作目录
            workspace_dir = _create_task_workspace(graph_id)
            tg.metadata["workspace_dir"] = workspace_dir

            set_active_task_graph(tg, graph_id, workspace_dir=workspace_dir)

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
                    "workspace_dir": workspace_dir,
                    "message": f"任务规划图已创建。根节点: req ({title})。请用 task_add_node 添加子任务。",
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
            },
            "required": ["title"],
        }


class TaskAddNodeTool(BaseTool):
    """task_add_node — 添加任务节点"""

    def __init__(self):
        super().__init__(name="task_add_node", category=ToolCategory.CUSTOM)
        self.description = (
            "向当前任务图添加一个子节点。"
            "通过 parent_id 指定父节点（根节点为 'req'），"
            "系统自动根据深度确定节点类型（分析/大纲/任务/子任务）。"
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
            return self._create_result(
                success=False,
                error="当前没有活跃的任务图，请先调用 task_create_plan",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

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

            # ── 守卫 B: 节点数量上限 ──
            leaf_nodes = tg.get_leaf_nodes()
            if len(leaf_nodes) >= MAX_LEAF_NODES:
                uncompleted = [n for n in leaf_nodes if n.status != "completed"][:5]
                hint_nodes = ", ".join(f"{n.id}({n.label})" for n in uncompleted)
                logger.info(
                    f"[task_add_node] 节点数量达上限: {len(leaf_nodes)}/{MAX_LEAF_NODES}"
                )
                return self._create_result(
                    success=True,
                    data={
                        "cap_reached": True,
                        "leaf_count": len(leaf_nodes),
                        "max_allowed": MAX_LEAF_NODES,
                        "message": (
                            f"任务图已有 {len(leaf_nodes)} 个工作项，达到上限。"
                            f"请调用 task_view_overview 查看现有节点，"
                            f"然后用 task_mark_status 逐个执行。"
                            f"待执行: {hint_nodes}"
                        ),
                    },
                    execution_time=time.time() - start_time,
                    request_id=request.request_id,
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
                    "description": "父节点 ID（根节点为 'req'）",
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
                    "description": "执行结果说明（完成时建议填写）",
                },
            },
            "required": ["node_id", "status"],
        }


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
                        "message": f"任务 '{description}' 已挂起 (ID: {task_id})。用户可以说'继续'来恢复。",
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
            "列出所有已挂起的任务。当用户说'继续'、'接着做'、'上次那个任务'等，"
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

        try:
            from zulong.l2.task_suspension import TaskSuspensionManager
            mgr = TaskSuspensionManager()

            if query:
                match = _run_async(mgr.find_by_description(query, return_full_state=True))
                if match is None:
                    return self._create_result(
                        success=True,
                        data={"tasks": [], "message": f"没有找到匹配 '{query}' 的挂起任务"},
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )

                if hasattr(match, 'task_id'):
                    if match.task_graph:
                        _ws = match.task_graph.metadata.get("workspace_dir", "") if hasattr(match.task_graph, 'metadata') else ""
                        set_active_task_graph(match.task_graph, match.metadata.get("graph_id", ""), workspace_dir=_ws)
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
            },
            "required": [],
        }
