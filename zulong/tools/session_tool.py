# File: zulong/tools/session_tool.py
# StartSessionTool — Round 1 专用工具
#
# 此工具仅在两阶段 FC 意图分类的 Round 1 中使用，不注册到 ToolRegistry。
# 根据模型输出的意图分类，执行确定性的骨架操作：
# - CHAT:    无操作，直接返回
# - COMPLEX: 创建 TaskGraph 骨架 + 同步 MemoryGraph
# - RESUME:  查找并恢复挂起的任务图
#
# 设计原则：固化骨架 + 自由决策
# 此工具负责"固化骨架"部分，模型在 Round 2 中负责"自由决策"。

import logging
import time
import asyncio
import re
from typing import Dict, Any

from zulong.tools.base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


def _run_async(coro):
    """在同步上下文中运行异步协程（复用 task_tools.py 的模式）"""
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


def _extract_title_core(title: str) -> str:
    """从任务标题中提取核心主题词（去除动词前缀、虚词、修饰词）。

    例如:
      "帮我设计一个博客系统的数据库" → "博客系统的数据库"
      "把博客系统的前端写出来"       → "博客系统的前端"
      "帮我写一个猜数字游戏"         → "猜数字游戏"
    """
    s = title.strip()
    # 去除常见动词前缀
    s = re.sub(r'^(帮我|请帮我|请|麻烦|把|将|给我|帮忙)\s*', '', s)
    s = re.sub(
        r'(写|做|设计|开发|创建|搭建|实现|生成|构建|完成|编写)(一个|一下)?\s*',
        '', s,
    )
    # 去除常见后缀
    s = re.sub(r'(出来|一下|吧|呢|了)$', '', s)
    # 去除修饰词
    s = re.sub(r'(简单的|简单|基本的|基本|完整的|完整)', '', s)
    return s.strip()


def _strip_stopwords(core: str) -> str:
    """从核心主题词中移除通用技术停用词，避免无关任务因共享通用词而误匹配。

    例如:
      "学生成绩管理系统的数据库表结构" → "学生成绩管理表结构"
      "博客系统的数据库"               → "博客"
    """
    # 越长的先替换，防止 "数据库" 只移除 "数据" 留下 "库"
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
    for w in stopwords:
        s = s.replace(w, "")
    return s.strip()


def _titles_related(old_title: str, new_title: str) -> bool:
    """判断两个任务标题是否属于同一项目/领域的关联任务。

    策略：提取核心主题词 → 移除通用技术停用词 → bigram Jaccard ≥ 0.3。
    停用词过滤可避免 "学生成绩管理系统的数据库" 和 "博客系统的数据库"
    因共享 "系统"/"数据库" 而误判为相关。

    示例:
      ("博客系统的数据库", "博客系统的前端") → True  (核心词 "博客" 匹配)
      ("学生成绩管理系统", "博客系统的数据库") → False (停用词去除后无交集)
      ("猜数字游戏",       "博客系统")       → False
    """
    core_a = _extract_title_core(old_title or "")
    core_b = _extract_title_core(new_title or "")
    if not core_a or not core_b:
        return False
    # 快捷路径：核心互为子串（去停用词前检查）
    if core_a in core_b or core_b in core_a:
        return True
    # 去除通用技术停用词后再比较
    clean_a = _strip_stopwords(core_a)
    clean_b = _strip_stopwords(core_b)
    # 去停用词后也检查子串
    if clean_a and clean_b and (clean_a in clean_b or clean_b in clean_a):
        return True
    # bigram Jaccard（使用去停用词后的版本）
    a = clean_a if len(clean_a) >= 2 else core_a
    b = clean_b if len(clean_b) >= 2 else core_b
    if len(a) < 2 or len(b) < 2:
        return False
    bigrams_a = {a[i:i + 2] for i in range(len(a) - 1)}
    bigrams_b = {b[i:i + 2] for i in range(len(b) - 1)}
    intersection = len(bigrams_a & bigrams_b)
    union = len(bigrams_a | bigrams_b)
    if union == 0:
        return False
    return (intersection / union) >= 0.3


def _load_task_graph_from_entry(entry):
    """根据 TaskIndexEntry 从磁盘加载 TaskGraph。

    Returns:
        (TaskGraph, graph_id) 或 None
    """
    import os
    import json
    from zulong.l2.task_graph import TaskGraph

    try:
        fpath = entry.file_path
        if not os.path.exists(fpath):
            return None

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if entry.source == "completed":
            snapshot = data.get("task_graph_snapshot", {})
            if not snapshot:
                return None
            tg = TaskGraph.deserialize(snapshot)
            graph_id = data.get("metadata", {}).get("graph_id", entry.entry_id)
        else:  # backup
            tg = TaskGraph.deserialize(data)
            graph_id = data.get("id", entry.entry_id)

        return (tg, graph_id)
    except Exception as e:
        logger.debug(f"[HistSearch] 加载任务图失败 ({entry.entry_id}): {e}")
        return None


def _search_historical_task(new_title: str):
    """在历史归档和磁盘备份中搜索与新标题相关的任务图。

    搜索策略（两阶段）：
      阶段 1：语义检索（embedding + FAISS，优先）
      阶段 2：文本匹配（bigram Jaccard，降级安全网）

    Returns:
        (task_graph, graph_id, old_title, source) 或 None
    """
    import os
    import json

    if not new_title:
        return None

    logger.info(f"[HistSearch] 开始搜索历史任务: '{new_title[:60]}'")

    # ===== 阶段 1：语义检索（优先） =====
    try:
        from zulong.memory.task_search_index import get_task_search_index
        index = get_task_search_index()
        if index.is_available():
            results = index.search(new_title, top_k=3, similarity_threshold=0.55)
            logger.info(
                f"[HistSearch] 语义检索返回 {len(results)} 条结果"
                + (f" (top: sim={results[0][1]:.3f}, '{results[0][0].title[:40]}')"
                   if results else "")
            )
            for entry, sim in results:
                loaded = _load_task_graph_from_entry(entry)
                if loaded:
                    tg, graph_id = loaded
                    logger.info(
                        f"[HistSearch] 语义检索命中: '{entry.title}' "
                        f"(sim={sim:.3f}, source={entry.source}, "
                        f"graph_id={graph_id})"
                    )
                    return (tg, graph_id, entry.title, entry.source)
                else:
                    # 文件不存在，从索引中移除
                    index.remove_entry(entry.entry_id)
        else:
            logger.info("[HistSearch] 语义检索不可用（embedding 模型未就绪），跳到文本匹配")
    except Exception as e:
        logger.debug(f"[HistSearch] 语义检索失败，降级到文本匹配: {e}")

    # ===== 阶段 2：文本匹配（降级安全网） =====

    # 2a. 搜索已完成归档
    try:
        from zulong.l2.task_archive import CompletedTaskArchiveManager
        archive_mgr = CompletedTaskArchiveManager()
        results = _run_async(archive_mgr.search_tasks(new_title))
        for r in results:
            old_title = r.get("description", "")
            if _titles_related(old_title, new_title):
                task_id = r["task_id"]
                full = _run_async(archive_mgr.get_task(task_id))
                if full and full.task_graph_snapshot:
                    from zulong.l2.task_graph import TaskGraph
                    tg = TaskGraph.deserialize(full.task_graph_snapshot)
                    graph_id = full.metadata.get("graph_id", task_id)
                    logger.info(
                        f"[HistSearch] 文本匹配(归档): "
                        f"'{old_title}' (graph_id={graph_id})"
                    )
                    return (tg, graph_id, old_title, "completed")
    except Exception as e:
        logger.debug(f"[HistSearch] 已完成归档搜索失败: {e}")

    # 2b. 搜索磁盘备份
    try:
        backup_dir = os.path.join(".", "data", "graph_backups")
        if os.path.isdir(backup_dir):
            for fname in os.listdir(backup_dir):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(backup_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    old_title = data.get("title", "")
                    if old_title and _titles_related(old_title, new_title):
                        from zulong.l2.task_graph import TaskGraph
                        tg = TaskGraph.deserialize(data)
                        graph_id = data.get("id", fname.replace(".json", ""))
                        logger.info(
                            f"[HistSearch] 文本匹配(备份): "
                            f"'{old_title}' (graph_id={graph_id})"
                        )
                        return (tg, graph_id, old_title, "backup")
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"[HistSearch] 磁盘备份搜索失败: {e}")

    return None


class StartSessionTool(BaseTool):
    """start_session — Round 1 意图分类执行工具

    此工具不注册到 ToolRegistry，由 InferenceEngine._classify_intent() 直接实例化调用。
    模型通过 tool_choice=required 被强制调用此工具，输出意图分类。
    工具的 execute() 方法执行对应场景的确定性骨架操作。
    """

    def __init__(self):
        super().__init__(name="start_session", category=ToolCategory.SYSTEM)
        self.description = "对用户输入进行意图分类并执行对应的初始化操作。"

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        """执行意图分类对应的骨架操作

        Args:
            request: 包含 intent, reason, task_description 参数的请求

        Returns:
            ToolResult: 包含骨架操作结果的响应
        """
        start_time = time.time()
        intent = request.parameters.get("intent", "chat")
        reason = request.parameters.get("reason", "")
        task_description = request.parameters.get("task_description", "")
        user_input = request.parameters.get("user_input", "")

        logger.info(f"[StartSession] intent={intent}, reason={reason}, task_desc={task_description}")

        if intent == "complex":
            return self._handle_complex(task_description, reason, start_time, request.request_id, user_input)
        elif intent == "resume":
            return self._handle_resume(task_description, reason, start_time, request.request_id)
        else:
            return self._handle_chat(reason, start_time, request.request_id)

    def _handle_chat(self, reason: str, start_time: float, request_id: str) -> ToolResult:
        """CHAT 分支：无骨架操作"""
        logger.info(f"[StartSession] CHAT 模式: {reason}")
        return self._create_result(
            success=True,
            data={
                "intent": "chat",
                "message": "ready for conversation",
                "reason": reason,
            },
            execution_time=time.time() - start_time,
            request_id=request_id,
        )

    def _handle_complex(self, task_description: str, reason: str,
                        start_time: float, request_id: str, user_input: str = "") -> ToolResult:
        """COMPLEX 分支：创建 TaskGraph 骨架

        复用 TaskCreatePlanTool (task_tools.py) 的创建逻辑：
        1. 检查是否已有活跃任务图（避免重复创建）
        2. 创建 TaskGraph + 根节点
        3. 同步到 MemoryGraph
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph

            # 检查是否已有活跃任务图
            old_tg = get_active_task_graph()
            if old_tg is not None:
                old_root = old_tg.get_node("req")
                old_title = old_root.label if old_root else old_tg.title
                new_title = task_description or ""

                # ---------- 判断新旧任务是否相关 ----------
                # 🔥 [Fix-7A] 已完成图谱不应阻止新任务创建
                # 先检查旧图是否已全部完成
                _old_leaves = old_tg.get_leaf_nodes()
                _old_all_done = (
                    _old_leaves and
                    all(n.status in ("completed", "skipped") for n in _old_leaves)
                )

                if _old_all_done:
                    # 🔥 [Fix-7C] 旧图已完成，但需先检查新任务是否是旧任务的延续
                    if _titles_related(old_title, new_title):
                        # 关联任务 → 复用已完成图谱，让模型在其上追加新节点
                        logger.info(
                            f"[StartSession] 旧图谱 '{old_title}' 已完成，"
                            f"新任务 '{new_title}' 与之关联，将在旧图谱上追加"
                        )
                        return self._create_result(
                            success=True,
                            data={
                                "intent": "complex",
                                "already_exists": True,
                                "graph_id": old_tg.id,
                                "title": old_title,
                                "message": (
                                    f"任务图「{old_title}」的所有节点已完成。"
                                    f"新需求「{new_title}」与之相关，将作为追加功能添加到现有图谱。"
                                ),
                            },
                            execution_time=time.time() - start_time,
                            request_id=request_id,
                        )
                    else:
                        # 无关任务 → 归档旧图（P3 安全网），清除后创建新图
                        logger.info(
                            f"[StartSession] 旧图谱 '{old_title}' 已全部完成，"
                            f"新任务 '{new_title}' 与之无关，归档后创建新图谱"
                        )
                        from zulong.tools.task_tools import _auto_archive_completed
                        _auto_archive_completed(old_tg)
                        set_active_task_graph(None, "")
                        # 继续往下走，创建新图谱
                else:
                    # 旧图仍有未完成节点 → 检查新旧任务是否相关
                    # 方法1：图谱刚创建不久（10分钟内）且有未完成节点
                    _is_recent = False
                    try:
                        _created_ts = int(old_tg.id.split("_", 1)[1])
                        _age_seconds = time.time() - _created_ts
                        _is_recent = (0 < _age_seconds < 600)  # 10 分钟
                    except (ValueError, IndexError):
                        pass

                    # 方法2：标题互为子串
                    _is_substr = (
                        new_title and old_title and (
                            new_title in old_title or old_title in new_title
                        )
                    )

                    _is_related = _is_recent or _is_substr

                    if _is_related:
                        # 同一任务的追加请求 → 复用现有图谱
                        logger.info(f"[StartSession] COMPLEX 追加：'{new_title}' 属于现有图谱 '{old_title}'")
                        return self._create_result(
                            success=True,
                            data={
                                "intent": "complex",
                                "already_exists": True,
                                "graph_id": old_tg.id,
                                "title": old_title,
                                "message": f"已有活跃任务图「{old_title}」，将在此基础上添加新要求。",
                            },
                            execution_time=time.time() - start_time,
                            request_id=request_id,
                        )
                    else:
                        # 全新的不同任务 → 挂起旧任务，创建新图谱
                        logger.info(f"[StartSession] COMPLEX 切换：挂起旧图谱 '{old_title}'，为新任务 '{new_title}' 创建新图谱")
                    try:
                        from zulong.l2.task_state_manager import task_state_manager
                        task_state_manager.freeze_current()
                        # 清除活跃图谱引用，让后续代码创建新的
                        set_active_task_graph(None, "")
                    except Exception as _e:
                        logger.warning(f"[StartSession] 挂起旧图谱失败: {_e}")
                        # freeze 失败时，紧急将旧图谱备份到磁盘，防止数据丢失
                        try:
                            from zulong.tools.task_tools import _backup_graph_to_disk
                            _old_gid = getattr(old_tg, 'id', '') or f"tg_orphan_{int(time.time())}"
                            _backup_graph_to_disk(old_tg, _old_gid)
                            logger.info(f"[StartSession] 已将旧图谱紧急备份到磁盘: {_old_gid}")
                        except Exception as _backup_err:
                            logger.error(f"[StartSession] 旧图谱紧急备份也失败: {_backup_err}")
                        set_active_task_graph(None, "")

            # 🔥 [P1] 创建新图之前，先搜索历史任务（已完成归档 + 磁盘备份）
            _hist_title = task_description or ""
            _hist_match = _search_historical_task(_hist_title)
            if _hist_match:
                _hist_tg, _hist_gid, _hist_old_title, _hist_source = _hist_match
                _hist_ws = _hist_tg.metadata.get("workspace_dir", "") if hasattr(_hist_tg, 'metadata') else ""
                set_active_task_graph(_hist_tg, _hist_gid, workspace_dir=_hist_ws)
                logger.info(
                    f"[StartSession] 从历史任务恢复: '{_hist_old_title}' "
                    f"(source={_hist_source}, graph_id={_hist_gid})"
                )
                return self._create_result(
                    success=True,
                    data={
                        "intent": "complex",
                        "already_exists": True,
                        "graph_id": _hist_gid,
                        "title": _hist_old_title,
                        "restored_from": _hist_source,
                        "message": (
                            f"找到历史任务「{_hist_old_title}」（来源: {_hist_source}），"
                            f"已恢复为活跃图谱。新需求「{_hist_title}」将在此基础上追加。"
                        ),
                    },
                    execution_time=time.time() - start_time,
                    request_id=request_id,
                )

            # 创建新的 TaskGraph
            title = task_description or "未命名任务"
            from zulong.l2.task_graph import TaskGraph
            graph_id = f"tg_{int(time.time())}"
            tg = TaskGraph(title=title, graph_id=graph_id)

            # 创建根节点
            tg.add_node(
                id="req",
                label=title,
                type="requirement",
                status="in_progress",
                desc=title,
            )

            # Rule B: 自动存储用户原始需求（不依赖模型传参）
            if user_input:
                tg.metadata["user_requirement"] = user_input
                logger.info(f"[StartSession] Rule B: 已存储用户原始需求 ({len(user_input)} 字符)")

            # 创建独立工作目录
            from zulong.tools.task_tools import _create_task_workspace
            workspace_dir = _create_task_workspace(graph_id)
            tg.metadata["workspace_dir"] = workspace_dir

            set_active_task_graph(tg, graph_id, workspace_dir=workspace_dir)

            # 同步到 MemoryGraph
            try:
                from zulong.memory.memory_graph import get_memory_graph, GraphNode, NodeType
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
                logger.debug(f"[StartSession] MemoryGraph 同步跳过: {e}")

            logger.info(f"[StartSession] COMPLEX: 创建任务图 {graph_id}: {title}")
            return self._create_result(
                success=True,
                data={
                    "intent": "complex",
                    "graph_id": graph_id,
                    "root_node_id": "req",
                    "title": title,
                    "message": f"任务图已创建 ({graph_id})，根节点: req ({title})。",
                },
                execution_time=time.time() - start_time,
                request_id=request_id,
            )

        except Exception as e:
            logger.error(f"[StartSession] COMPLEX 创建失败: {e}", exc_info=True)
            return self._create_result(
                success=False,
                error=f"任务图创建失败: {e}",
                execution_time=time.time() - start_time,
                request_id=request_id,
            )

    def _handle_resume(self, task_description: str, reason: str,
                       start_time: float, request_id: str) -> ToolResult:
        """RESUME 分支：查找并恢复挂起的任务

        复用 TaskListSuspendedTool (task_tools.py) 的恢复逻辑：
        1. 调用 TaskSuspensionManager.find_by_description() 模糊匹配
        2. 如果找到且有 TaskGraph，恢复到活跃状态
        3. 如果未找到，降级返回 fallback=True
        """
        try:
            from zulong.tools.task_tools import get_active_task_graph, set_active_task_graph

            # 如果已有活跃任务图，直接返回（graph_id 记录到响应中供调试）
            old_tg = get_active_task_graph()
            if old_tg is not None:
                old_root = old_tg.get_node("req")
                old_title = old_root.label if old_root else old_tg.title
                old_graph_id = getattr(old_tg, 'id', '')
                logger.info(f"[StartSession] RESUME：已有活跃图谱 '{old_title}' (graph_id={old_graph_id})，直接使用")
                return self._create_result(
                    success=True,
                    data={
                        "intent": "resume",
                        "already_active": True,
                        "title": old_title,
                        "graph_id": old_graph_id,
                        "has_task_graph": True,
                        "message": f"当前已有活跃任务「{old_title}」，将继续执行。",
                    },
                    execution_time=time.time() - start_time,
                    request_id=request_id,
                )

            from zulong.l2.task_suspension import TaskSuspensionManager
            mgr = TaskSuspensionManager()

            # 模糊匹配挂起任务
            query = task_description or ""
            match = _run_async(mgr.find_by_description(query, return_full_state=True))

            if match is None:
                # 🔥 [P2] 挂起任务未找到 → 搜索已完成归档和磁盘备份
                _hist = _search_historical_task(query)
                if _hist:
                    _hist_tg, _hist_gid, _hist_title, _hist_source = _hist
                    _hist_ws = _hist_tg.metadata.get("workspace_dir", "") if hasattr(_hist_tg, 'metadata') else ""
                    set_active_task_graph(_hist_tg, _hist_gid, workspace_dir=_hist_ws)
                    logger.info(
                        f"[StartSession] RESUME: 从历史任务恢复: "
                        f"'{_hist_title}' (source={_hist_source})"
                    )
                    return self._create_result(
                        success=True,
                        data={
                            "intent": "resume",
                            "description": _hist_title,
                            "graph_id": _hist_gid,
                            "has_task_graph": True,
                            "restored_from": _hist_source,
                            "message": (
                                f"已从历史归档恢复任务「{_hist_title}」"
                                f"（来源: {_hist_source}）。任务图已加载。"
                            ),
                        },
                        execution_time=time.time() - start_time,
                        request_id=request_id,
                    )

                logger.info("[StartSession] RESUME：未找到挂起任务，降级为 CHAT")
                # 列出所有可用挂起任务，帮助用户精准选择
                _available_tasks = []
                _available_hint = ""
                try:
                    _all_tasks = _run_async(mgr.list_suspended_tasks())
                    if _all_tasks:
                        for _t in _all_tasks[:5]:
                            _desc = _t.get("description", "") if isinstance(_t, dict) else getattr(_t, "description", "")
                            _tid = _t.get("task_id", "") if isinstance(_t, dict) else getattr(_t, "task_id", "")
                            if _desc:
                                _available_tasks.append({"task_id": _tid, "description": _desc})
                        _task_names = "、".join(t["description"] for t in _available_tasks)
                        _available_hint = f" 当前有 {len(_all_tasks)} 个挂起任务：{_task_names}。你可以说出具体任务名称来恢复。"
                except Exception:
                    pass
                return self._create_result(
                    success=True,
                    data={
                        "intent": "chat",
                        "fallback": True,
                        "original_intent": "resume",
                        "available_tasks": _available_tasks,
                        "message": f"没有找到匹配的挂起任务。{_available_hint}" if _available_hint
                                   else "没有找到之前挂起的任务，将以对话模式继续。",
                    },
                    execution_time=time.time() - start_time,
                    request_id=request_id,
                )

            # 有 task_id 属性 → 完整状态对象（SuspendableTaskState）
            if hasattr(match, 'task_id'):
                task_id = match.task_id
                description = match.description
                has_graph = match.task_graph is not None

                if has_graph:
                    graph_id = match.metadata.get("graph_id", "") if hasattr(match, 'metadata') else ""
                    _resume_ws = match.task_graph.metadata.get("workspace_dir", "") if hasattr(match.task_graph, 'metadata') else ""
                    set_active_task_graph(match.task_graph, graph_id, workspace_dir=_resume_ws)
                    logger.info(f"[StartSession] RESUME: 已恢复任务图 '{description}' (task_id={task_id})")

                    # 同步到 TaskStateManager（内存状态 + MemoryGraph）
                    try:
                        from zulong.l2.task_state_manager import task_state_manager
                        task_state_manager.create_task(task_id, match.messages or [])
                        task_state_manager.resume_task(task_id, task_graph=match.task_graph)
                        logger.info(f"[StartSession] RESUME: 已同步到 TaskStateManager")
                    except Exception as _tsm_err:
                        logger.warning(f"[StartSession] RESUME: TaskStateManager 同步失败: {_tsm_err}")

                # 确认恢复成功，显式消费（删除）磁盘文件
                try:
                    _run_async(mgr.cancel_task(task_id))
                    logger.info(f"[StartSession] RESUME: 已消费挂起文件 (task_id={task_id})")
                except Exception as _ce:
                    logger.warning(f"[StartSession] RESUME: 消费挂起文件失败: {_ce}")

                return self._create_result(
                    success=True,
                    data={
                        "intent": "resume",
                        "task_id": task_id,
                        "description": description,
                        "has_task_graph": has_graph,
                        "message": (
                            f"已恢复任务「{description}」(ID: {task_id})。"
                            + (" 任务图已加载。" if has_graph else " 无关联任务图。")
                        ),
                    },
                    execution_time=time.time() - start_time,
                    request_id=request_id,
                )

            # dict 形式 → 摘要信息（无完整状态）
            if isinstance(match, dict) and match.get('task_id'):
                logger.info(f"[StartSession] RESUME：找到任务摘要 '{match.get('description', '')}'，但无完整状态")
                return self._create_result(
                    success=True,
                    data={
                        "intent": "resume",
                        "task_id": match.get('task_id', ''),
                        "description": match.get('description', ''),
                        "has_task_graph": False,
                        "message": f"找到挂起任务「{match.get('description', '')}」，但无关联任务图。",
                    },
                    execution_time=time.time() - start_time,
                    request_id=request_id,
                )

            # 兜底
            logger.info("[StartSession] RESUME：匹配结果无法识别，降级为 CHAT")
            return self._create_result(
                success=True,
                data={
                    "intent": "chat",
                    "fallback": True,
                    "original_intent": "resume",
                    "message": "未能成功恢复任务，将以对话模式继续。",
                },
                execution_time=time.time() - start_time,
                request_id=request_id,
            )

        except Exception as e:
            logger.error(f"[StartSession] RESUME 恢复失败: {e}", exc_info=True)
            # 异常时降级为 CHAT
            return self._create_result(
                success=True,
                data={
                    "intent": "chat",
                    "fallback": True,
                    "original_intent": "resume",
                    "error": str(e),
                    "message": f"任务恢复失败 ({e})，将以对话模式继续。",
                },
                execution_time=time.time() - start_time,
                request_id=request_id,
            )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["chat", "complex", "resume"],
                    "description": "用户意图分类",
                },
                "reason": {
                    "type": "string",
                    "description": "分类理由的简短说明",
                },
                "task_description": {
                    "type": "string",
                    "description": "任务描述（用于 complex 创建任务图或 resume 匹配挂起任务）。必须使用与用户输入相同的语言。",
                },
            },
            "required": ["intent", "reason"],
        }
