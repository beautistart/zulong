# File: zulong/tools/attention_tool.py
# navigate_attention FC 工具 — L2 模型主动控制注意力焦点
#
# 提供三种导航方式: deeper（深入子节点）、broader（上浮父节点）、jump（跳转指定节点）
# 模型通过此工具主动调整自己在图记忆空间中的注意力焦点，
# 实现"思维深度索引"中的动态注意力控制。

import logging
import time
from typing import Dict, Any

from .base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


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
                if not mg.has_node(target_node_id):
                    return self._create_result(
                        success=False,
                        error=f"节点 '{target_node_id}' 不存在",
                        execution_time=time.time() - start_time,
                        request_id=request.request_id,
                    )
                success = mg.update_focus_to_node(target_node_id)

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
                            "focus_path_summary": mg.get_focus_path_summary(),
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
                            "focus_path_summary": mg.get_focus_path_summary(),
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
            new_summary = mg.get_focus_path_summary()
            new_ctx = mg.get_last_focus_context() or {}
            new_depth = new_ctx.get("focus_depth", 0)

            logger.info(
                f"[NavigateAttention] {direction} → depth={new_depth}"
            )

            return self._create_result(
                success=True,
                data={
                    "message": f"注意力焦点已{{'deeper': '深入', 'broader': '上浮', 'jump': '跳转'}}[direction]",
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
