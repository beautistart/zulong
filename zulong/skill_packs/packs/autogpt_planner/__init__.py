# File: zulong/skill_packs/packs/autogpt_planner/__init__.py
"""
AutoGPT任务拆解技能包

从AutoGPT的任务规划逻辑中提取核心算法，包装为ISkillPack接口。
提供以下能力：
- task_decompose: 将复杂请求拆解为子任务列表
- priority_rank: 对子任务排序
- dependency_analyze: 分析子任务依赖关系
"""

from zulong.skill_packs.packs.autogpt_planner.planner import TaskDecomposeAlgorithm
from zulong.skill_packs.packs.autogpt_planner.tools import (
    TaskDecomposeTool,
    PriorityRankTool,
    DependencyAnalyzeTool,
)
from zulong.skill_packs.interface import ISkillPack, SkillPackManifest

__all__ = [
    "AutoGPTPlannerPack",
    "TaskDecomposeAlgorithm",
    "TaskDecomposeTool",
    "PriorityRankTool",
    "DependencyAnalyzeTool",
]


class AutoGPTPlannerPack(ISkillPack):
    """AutoGPT任务拆解技能包"""
    
    def __init__(self):
        self._planner = None
        self._tools = []
    
    def get_manifest(self) -> SkillPackManifest:
        return SkillPackManifest(
            pack_id="autogpt_planner",
            name="AutoGPT任务拆解",
            version="1.0.0",
            description="从AutoGPT提取的任务拆解技能，擅长将复杂请求拆解为可执行子任务",
            capabilities=["task_decompose", "priority_rank", "dependency_analyze"],
            learning_objectives=["将复杂请求拆解为子任务", "判断子任务优先级和依赖关系"],
            source="autogpt"
        )
    
    def install(self, tool_registry, config=None):
        config = config or {}
        max_subtasks = config.get("max_subtasks", 10)
        model_id = config.get("planning_model", "default")
        
        self._planner = TaskDecomposeAlgorithm(
            max_subtasks=max_subtasks,
            model_id=model_id,
        )
        
        self._tools = [
            TaskDecomposeTool(self._planner),
            PriorityRankTool(self._planner),
            DependencyAnalyzeTool(self._planner),
        ]
        
        for tool in self._tools:
            try:
                tool_registry.register(tool)
            except Exception:
                pass
        
        return True
    
    def execute(self, capability, params):
        if not self._planner:
            return {"success": False, "error": "技能包未安装"}
        
        if capability == "task_decompose":
            goal = params.get("goal", params.get("user_request", ""))
            context = params.get("context", "")
            return self._planner.decompose(goal, context if context else None)
        
        elif capability == "priority_rank":
            subtasks = params.get("subtasks", [])
            ranked = self._planner.rank_priorities(subtasks)
            return {"success": True, "ranked_subtasks": ranked}
        
        elif capability == "dependency_analyze":
            subtasks = params.get("subtasks", [])
            goal = params.get("goal", "")
            deps = self._planner.analyze_dependencies(subtasks, goal)
            groups = self._planner._compute_parallel_groups(subtasks, deps)
            return {"success": True, "dependencies": deps, "parallel_groups": groups}
        
        return {"success": False, "error": "未知能力: %s" % capability}
    
    def get_tools(self):
        return self._tools
    
    def uninstall(self):
        self._planner = None
        self._tools = []
        return True
