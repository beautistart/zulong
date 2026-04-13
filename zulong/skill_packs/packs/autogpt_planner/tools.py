# File: zulong/skill_packs/packs/autogpt_planner/tools.py
"""
AutoGPT任务拆解技能包 - 注册的工具类
"""

import logging
from typing import Dict, Any

from zulong.tools.base import BaseTool, ToolCategory, ToolRequest, ToolResult
from zulong.skill_packs.packs.autogpt_planner.planner import TaskDecomposeAlgorithm

logger = logging.getLogger(__name__)


class TaskDecomposeTool(BaseTool):
    """任务拆解工具
    
    将复杂用户请求拆解为可执行的子任务列表。
    """
    
    def __init__(self, planner: TaskDecomposeAlgorithm):
        super().__init__(name="task_decompose", category=ToolCategory.CUSTOM)
        self.description = (
            "任务拆解工具。将复杂用户请求拆解为可执行的子任务列表。"
            "适用于包含多个步骤的复合任务，如'搜索新闻并分析趋势并写报告'。"
            "参数：goal(任务目标), context(可选上下文)"
        )
        self.planner = planner
    
    def initialize(self) -> bool:
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        import time
        start_time = time.time()
        
        goal = request.parameters.get("goal", "")
        context = request.parameters.get("context", "")
        
        if not goal:
            return self._create_result(
                success=False,
                error="缺少任务目标(goal参数)",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        
        result = self.planner.decompose(goal, context if context else None)
        
        return self._create_result(
            success=result.get("success", False),
            data=result,
            error=result.get("error"),
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    
    def cleanup(self) -> None:
        pass
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "需要拆解的复杂任务目标"
                },
                "context": {
                    "type": "string",
                    "description": "当前上下文信息（可选）"
                }
            },
            "required": ["goal"]
        }


class PriorityRankTool(BaseTool):
    """优先级排序工具
    
    对子任务列表进行优先级排序。
    """
    
    def __init__(self, planner: TaskDecomposeAlgorithm):
        super().__init__(name="priority_rank", category=ToolCategory.CUSTOM)
        self.description = (
            "优先级排序工具。对子任务列表进行优先级排序。"
            "参数：subtasks(子任务列表)"
        )
        self.planner = planner
    
    def initialize(self) -> bool:
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        import time
        start_time = time.time()
        
        subtasks = request.parameters.get("subtasks", [])
        
        if not subtasks:
            return self._create_result(
                success=False,
                error="缺少子任务列表(subtasks参数)",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        
        ranked = self.planner.rank_priorities(subtasks)
        
        return self._create_result(
            success=True,
            data={"ranked_subtasks": ranked},
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    
    def cleanup(self) -> None:
        pass
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "description": "需要排序的子任务列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "tool_hint": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["subtasks"]
        }


class DependencyAnalyzeTool(BaseTool):
    """依赖分析工具
    
    分析子任务间的依赖关系。
    """
    
    def __init__(self, planner: TaskDecomposeAlgorithm):
        super().__init__(name="dependency_analyze", category=ToolCategory.CUSTOM)
        self.description = (
            "依赖分析工具。分析子任务间的依赖关系和并行可能性。"
            "参数：subtasks(子任务列表), goal(原始目标)"
        )
        self.planner = planner
    
    def initialize(self) -> bool:
        return True
    
    def execute(self, request: ToolRequest) -> ToolResult:
        import time
        start_time = time.time()
        
        subtasks = request.parameters.get("subtasks", [])
        goal = request.parameters.get("goal", "")
        
        if not subtasks:
            return self._create_result(
                success=False,
                error="缺少子任务列表(subtasks参数)",
                execution_time=time.time() - start_time,
                request_id=request.request_id,
            )
        
        dependencies = self.planner.analyze_dependencies(subtasks, goal)
        parallel_groups = self.planner._compute_parallel_groups(subtasks, dependencies)
        
        return self._create_result(
            success=True,
            data={
                "dependencies": dependencies,
                "parallel_groups": parallel_groups,
            },
            execution_time=time.time() - start_time,
            request_id=request.request_id,
        )
    
    def cleanup(self) -> None:
        pass
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "description": "子任务列表"
                },
                "goal": {
                    "type": "string",
                    "description": "原始任务目标"
                }
            },
            "required": ["subtasks"]
        }
