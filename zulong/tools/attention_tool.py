# File: zulong/tools/attention_tool.py

# navigate_attention FC 工具 — L2 模型主动控制注意力焦点
# adjust_attention_mode FC 工具 — L2 模型直接切换注意力模式（P2-1）

#

# 提供三种导航方式: deeper（深入子节点）、broader（上浮父节点）、jump（跳转指定节点）

# 模型通过此工具主动调整自己在图记忆空间中的注意力焦点，

# 实现"思维深度索引"中的动态注意力控制。



import logging

import time

from typing import Dict, Any, Optional



from .base import BaseTool, ToolCategory, ToolRequest, ToolResult



logger = logging.getLogger(__name__)


def _node_type_value(node: Any) -> str:
    node_type = getattr(node, "node_type", "")
    return str(getattr(node_type, "value", node_type) or "")


def _focus_path_summary(memory_graph: Any) -> str:
    """Return a compact focus path summary for MemoryGraph and ShardedMemoryGraph."""
    if hasattr(memory_graph, "get_focus_path_summary"):
        try:
            return memory_graph.get_focus_path_summary() or ""
        except Exception:
            pass

    ctx = {}
    try:
        ctx = memory_graph.get_last_focus_context() or {}
    except Exception:
        ctx = {}
    focus_path = list(ctx.get("focus_path") or [])
    if not focus_path:
        return ""

    lines = ["【思维导航】"]
    for idx, node_id in enumerate(focus_path[-8:]):
        node = None
        try:
            node = memory_graph.get_node(node_id)
        except Exception:
            node = None
        label = (getattr(node, "label", "") if node else "") or node_id
        type_label = _node_type_value(node) if node else "node"
        cursor = " ← 当前焦点" if idx == min(len(focus_path), 8) - 1 else ""
        indent = "  " * idx
        prefix = "└─ " if idx else ""
        short_id = node_id if len(node_id) <= 56 else node_id[:26] + "..." + node_id[-24:]
        lines.append(
            f"{indent}{prefix}L{idx + 1} [{type_label}] {str(label)[:40]} @{short_id}{cursor}"
        )
    lines.append("提示: navigate_attention deeper/broader/jump 调整注意力焦点")
    return "\n".join(lines)[:500]


def _sync_task_graph_to_memory(memory_graph: Any, task_graph: Any) -> None:
    if not task_graph:
        return
    try:
        from zulong.memory.graph_adapters import TaskGraphAdapter
        TaskGraphAdapter().sync(memory_graph, task_graph)
    except Exception as exc:
        logger.debug(f"[NavigateAttention] TaskGraph 同步到 MemoryGraph 跳过: {exc}")


def _resolve_focus_node_id(memory_graph: Any, target_node_id: str) -> Optional[str]:
    """Resolve raw MemoryGraph id or TaskGraph node id to a graph-memory node id."""
    if not target_node_id:
        return None

    try:
        if memory_graph.has_node(target_node_id):
            return target_node_id
    except Exception:
        pass

    task_graph = None
    try:
        from zulong.tools.task_tools import get_active_task_graph
        task_graph = get_active_task_graph()
    except Exception:
        task_graph = None

    graph_id = getattr(task_graph, "id", "") if task_graph else ""
    if task_graph:
        try:
            if not task_graph.get_node(target_node_id):
                task_graph = None
        except Exception:
            task_graph = None

    if task_graph:
        _sync_task_graph_to_memory(memory_graph, task_graph)
        candidates = []
        if graph_id:
            candidates.extend([
                f"task:{graph_id}/{target_node_id}",
                f"task:{graph_id}" if target_node_id in {"req", graph_id} else "",
            ])
        candidates.append(f"task:{target_node_id}")
        for candidate in [c for c in candidates if c]:
            try:
                if memory_graph.has_node(candidate):
                    return candidate
            except Exception:
                pass

    try:
        nodes = memory_graph.get_nodes_by_type("task")
    except Exception:
        nodes = []
    for node in nodes or []:
        node_id = getattr(node, "node_id", "")
        meta = getattr(node, "metadata", {}) or {}
        if graph_id and meta.get("graph_id") and meta.get("graph_id") != graph_id:
            continue
        if node_id == target_node_id or node_id.endswith(f"/{target_node_id}"):
            return node_id
        if meta.get("backend_ref", "").endswith(f"/{target_node_id}"):
            return node_id
        if meta.get("graph_address", "").endswith(f"task:{target_node_id}"):
            return node_id

    return None





class NavigateAttentionTool(BaseTool):

    """navigate_attention 工具



    让 L2 模型主动控制思维注意力焦点在图记忆空间中的位置。

    - deeper: 深入当前焦点的子节点，获取更细粒度的上下文

    - broader: 返回父节点，获取更宏观的视角

    - jump: 跳转到指定节点，切换注意力到任意图位置

    """



    def __init__(self):

        super().__init__(name="navigate_attention", category=ToolCategory.CUSTOM)

        self.description = (

            "导航思维注意力焦点。当你需要深入某个子任务的细节时使用 deeper，"

            "需要返回上层获取全局视角时使用 broader，"

            "需要切换到特定任务或对话节点时使用 jump。"

            "调用后系统会更新你的思维导航上下文。"

        )



    def initialize(self) -> bool:

        return True



    def cleanup(self) -> None:

        pass



    def execute(self, request: ToolRequest) -> ToolResult:

        """执行注意力导航



        Args:

            request.parameters:

                - direction: "deeper" | "broader" | "jump" (必填)

                - target_node_id: 跳转目标节点 ID (jump 时必填)

        """

        start_time = time.time()



        direction = request.parameters.get("direction", "")

        target_node_id = request.parameters.get("target_node_id", "")



        if direction not in ("deeper", "broader", "jump"):

            return self._create_result(

                success=False,

                error="direction 必须是 'deeper'、'broader' 或 'jump'",

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



        if direction == "jump" and not target_node_id:

            return self._create_result(

                success=False,

                error="jump 方向需要提供 target_node_id",

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



        try:

            from zulong.memory.memory_graph import get_memory_graph

            mg = get_memory_graph()

            if mg is None:

                return self._create_result(

                    success=False,

                    error="MemoryGraph 未初始化",

                    execution_time=time.time() - start_time,

                    request_id=request.request_id,

                )



            # 获取当前焦点

            ctx = mg.get_last_focus_context()

            focus_path = (ctx or {}).get("focus_path") or []

            current_focus_id = focus_path[-1] if focus_path else None



            if direction == "jump":

                # 跳转到指定节点

                resolved_target_id = _resolve_focus_node_id(mg, target_node_id)

                if not resolved_target_id:

                    return self._create_result(

                        success=False,

                        error=f"节点 '{target_node_id}' 不存在",

                        execution_time=time.time() - start_time,

                        request_id=request.request_id,

                    )

                success = mg.update_focus_to_node(resolved_target_id)



            elif direction == "deeper":

                if not current_focus_id:

                    return self._create_result(

                        success=False,

                        error="当前无焦点，请先使用 jump 指定一个节点",

                        execution_time=time.time() - start_time,

                        request_id=request.request_id,

                    )

                children = mg.get_children(current_focus_id)

                if not children:

                    return self._create_result(

                        success=True,

                        data={

                            "message": "当前焦点无子节点，已处于最深层",

                            "focus_path_summary": _focus_path_summary(mg),

                        },

                        execution_time=time.time() - start_time,

                        request_id=request.request_id,

                    )

                # 选择最近访问的子节点

                target = max(children, key=lambda n: n.last_accessed)

                success = mg.update_focus_to_node(target.node_id)



            else:  # broader

                if not current_focus_id:

                    return self._create_result(

                        success=False,

                        error="当前无焦点，请先使用 jump 指定一个节点",

                        execution_time=time.time() - start_time,

                        request_id=request.request_id,

                    )

                parent = mg.get_parent(current_focus_id)

                if not parent:

                    return self._create_result(

                        success=True,

                        data={

                            "message": "当前焦点已在最顶层，无法再上浮",

                            "focus_path_summary": _focus_path_summary(mg),

                        },

                        execution_time=time.time() - start_time,

                        request_id=request.request_id,

                    )

                success = mg.update_focus_to_node(parent.node_id)



            if not success:

                return self._create_result(

                    success=False,

                    error="焦点更新失败",

                    execution_time=time.time() - start_time,

                    request_id=request.request_id,

                )



            # 返回更新后的焦点路径

            new_summary = _focus_path_summary(mg)

            new_ctx = mg.get_last_focus_context() or {}

            new_depth = new_ctx.get("focus_depth", 0)



            logger.info(

                f"[NavigateAttention] {direction} → depth={new_depth}"

            )



            return self._create_result(

                success=True,

                data={

                    "message": f"注意力焦点已{ {'deeper': '深入', 'broader': '上浮', 'jump': '跳转'}[direction] }",

                    "direction": direction,

                    "focus_depth": new_depth,

                    "focus_path_summary": new_summary,

                },

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



        except Exception as e:

            logger.error(f"[NavigateAttention] 导航失败: {e}", exc_info=True)

            return self._create_result(

                success=False,

                error=f"注意力导航失败: {e}",

                execution_time=time.time() - start_time,

                request_id=request.request_id,

            )



    def _get_parameters_schema(self) -> Dict[str, Any]:

        return {

            "type": "object",

            "properties": {

                "direction": {

                    "type": "string",

                    "enum": ["deeper", "broader", "jump"],

                    "description": (

                        "导航方向: deeper=深入子节点细节, "

                        "broader=返回上层全局视角, "

                        "jump=跳转到指定节点"

                    ),

                },

                "target_node_id": {

                    "type": "string",

                    "description": (

                        "跳转目标节点 ID (仅 jump 方向时必填)。"

                        "可从思维导航中获取节点 ID"

                    ),

                },

            },

            "required": ["direction"],

        }


class AdjustAttentionModeTool(BaseTool):
    """adjust_attention_mode — 直接切换注意力模式

    让 L2 模型直接设置注意力窗口的运行模式：
    - global: 全局视角，关注大纲和整体结构
    - focus: 聚焦模式，关注某节点的细节和关联
    - single_chain: 单链推理，只保留当前执行链路

    此工具是 LLM 显式注意力控制能力。工具本身返回确认信息，
    调用结果由 AttentionWindowManager 作为显式注意力选择应用；
    普通读写/命令/检索工具不得借此路径自动切换注意力。
    """

    def __init__(self):
        super().__init__(name="adjust_attention_mode", category=ToolCategory.CUSTOM)
        self.description = (
            "直接切换注意力模式。"
            "当你需要切换到全局视角（global）查看整体结构、"
            "聚焦模式（focus）深入某个节点细节、"
            "或单链模式（single_chain）进行深度推理时调用。"
            "与 navigate_attention 的区别：此工具直接指定目标模式，"
            "而 navigate_attention 通过 deeper/broader 相对切换。"
        )

    def initialize(self) -> bool:
        return True

    def cleanup(self) -> None:
        pass

    def execute(self, request: ToolRequest) -> ToolResult:
        start_time = time.time()
        mode = request.parameters.get("mode", "")

        valid_modes = {"global", "focus", "single_chain"}
        if mode not in valid_modes:
            return self._create_result(
                success=False,
                error=f"mode 必须是 {valid_modes} 之一，收到: '{mode}'",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )

        # 工具本身只做验证和返回确认。
        # 模式切换只允许通过显式注意力控制能力进入 AttentionWindowManager；
        # 普通工具调用不能触发 GLOBAL/FOCUS/SINGLE_CHAIN 切换。
        mode_names = {
            "global": "全局视角",
            "focus": "聚焦模式",
            "single_chain": "单链推理",
        }

        logger.info(f"[AdjustAttentionMode] 请求切换到: {mode}")

        return self._create_result(
            success=True,
            data={
                "mode": mode,
                "mode_name": mode_names[mode],
                "message": f"注意力模式已请求切换到: {mode_names[mode]}",
            },
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["global", "focus", "single_chain"],
                    "description": (
                        "目标注意力模式: "
                        "global=全局视角（关注大纲和整体进度）, "
                        "focus=聚焦模式（关注某节点的细节和关联）, "
                        "single_chain=单链推理（深度推理，暂排当前阶段无关上下文）"
                    ),
                },
            },
            "required": ["mode"],
        }
