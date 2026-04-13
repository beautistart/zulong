# File: zulong/skill_packs/packs/autogpt_planner/planner.py
"""
AutoGPT任务拆解技能包 - 核心拆解算法

从AutoGPT的任务规划逻辑中提取核心算法：
1. 将复杂目标拆解为可执行的子任务列表
2. 分析子任务之间的依赖关系
3. 标记可并行执行的任务组
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TaskDecomposeAlgorithm:
    """任务拆解算法（从AutoGPT提取的核心逻辑）
    
    设计思路：
    1. 使用LLM将复杂目标拆解为结构化子任务
    2. 分析子任务间的依赖关系
    3. 识别可并行的任务组
    """
    
    def __init__(self, max_subtasks: int = 10, llm_client=None, model_id: str = "default"):
        """初始化
        
        Args:
            max_subtasks: 最大子任务数量
            llm_client: LLM客户端（可选，不传则使用默认prompt模板）
            model_id: 使用的模型ID
        """
        self.max_subtasks = max_subtasks
        self.llm_client = llm_client
        self.model_id = model_id
    
    def decompose(self, goal: str, context: Optional[str] = None) -> Dict[str, Any]:
        """将复杂目标拆解为子任务
        
        Args:
            goal: 复杂目标描述
            context: 当前上下文信息（可选）
        
        Returns:
            {
                "success": bool,
                "subtasks": [{"step": int, "task": str, "tool_hint": str}],
                "dependencies": {"task_name": ["dependency1", ...]},
                "parallel_groups": [["task1", "task2"], ["task3"]],
                "subtask_count": int,
            }
        """
        start_time = time.time()
        
        try:
            # 1. 生成任务拆解
            subtasks = self._generate_subtasks(goal, context)
            
            if not subtasks:
                return {
                    "success": False,
                    "error": "未能生成子任务拆解",
                    "subtasks": [],
                    "dependencies": {},
                    "parallel_groups": [],
                    "subtask_count": 0,
                }
            
            # 2. 分析依赖关系
            dependencies = self._analyze_dependencies(subtasks, goal)
            
            # 3. 计算并行组
            parallel_groups = self._compute_parallel_groups(subtasks, dependencies)
            
            result = {
                "success": True,
                "subtasks": subtasks,
                "dependencies": dependencies,
                "parallel_groups": parallel_groups,
                "subtask_count": len(subtasks),
                "execution_time": time.time() - start_time,
            }
            
            logger.info(f"[AutoGPTPlanner] 任务拆解完成: {len(subtasks)} 个子任务, "
                       f"{len(parallel_groups)} 个并行组, 耗时 {time.time() - start_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"[AutoGPTPlanner] 任务拆解失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "subtasks": [],
                "dependencies": {},
                "parallel_groups": [],
                "subtask_count": 0,
            }
    
    def rank_priorities(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对子任务进行优先级排序
        
        Args:
            subtasks: 子任务列表
        
        Returns:
            排序后的子任务列表（按优先级从高到低）
        """
        # 简单规则排序：
        # 1. 信息收集类任务优先级最高
        # 2. 分析计算类其次
        # 3. 输出生成类最后
        
        priority_order = {
            "search": 1, "gather": 1, "collect": 1, "fetch": 1,
            "analyze": 2, "calculate": 2, "compute": 2, "process": 2,
            "generate": 3, "write": 3, "create": 3, "output": 3,
        }
        
        def get_priority(task: Dict) -> int:
            task_desc = task.get("task", "").lower()
            tool_hint = task.get("tool_hint", "").lower()
            combined = task_desc + " " + tool_hint
            
            for keyword, priority in priority_order.items():
                if keyword in combined:
                    return priority
            return 2  # 默认中等优先级
        
        return sorted(subtasks, key=get_priority)
    
    def analyze_dependencies(self, subtasks: List[Dict[str, Any]], goal: str) -> Dict[str, List[str]]:
        """分析子任务间的依赖关系
        
        Args:
            subtasks: 子任务列表
            goal: 原始目标
        
        Returns:
            依赖关系字典 {"task_name": ["dependency1", "dependency2"]}
        """
        dependencies = {}
        
        for i, task in enumerate(subtasks):
            task_name = task.get("task", f"task_{i}")
            deps = []
            
            # 简单规则：信息收集任务通常不依赖其他任务
            # 分析和输出任务通常依赖前面的收集任务
            tool_hint = task.get("tool_hint", "").lower()
            
            if tool_hint in ("search", "gather", "collect", "fetch"):
                # 收集类任务通常没有前置依赖
                deps = []
            elif tool_hint in ("analyze", "calculate", "process"):
                # 分析类任务依赖前面的收集任务
                for j in range(i):
                    prev_tool = subtasks[j].get("tool_hint", "").lower()
                    if prev_tool in ("search", "gather", "collect"):
                        deps.append(subtasks[j].get("task", f"task_{j}"))
            else:
                # 输出类任务依赖前面的分析和收集任务
                for j in range(i):
                    deps.append(subtasks[j].get("task", f"task_{j}"))
            
            dependencies[task_name] = deps
        
        return dependencies
    
    # ========== 内部方法 ==========
    
    def _generate_subtasks(self, goal: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """生成子任务列表"""
        
        if self.llm_client and self.model_id:
            return self._generate_subtasks_with_llm(goal, context)
        else:
            return self._generate_subtasks_with_prompt(goal, context)
    
    def _generate_subtasks_with_llm(self, goal: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """使用LLM生成子任务"""
        try:
            system_prompt = """你是一个任务拆解专家。你的职责是将复杂目标拆解为可执行的子任务。

请严格按照以下JSON格式返回：
{
  "subtasks": [
    {"step": 1, "task": "子任务描述", "tool_hint": "search|analyze|write|execute"}
  ]
}

规则：
1. 子任务数量不超过10个
2. 每个子任务应该是独立可执行的
3. tool_hint 提示可能需要使用的工具类型
4. 按执行顺序排列"""
            
            user_prompt = f"目标: {goal}"
            if context:
                user_prompt += f"\n当前上下文: {context}"
            
            response = self.llm_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("subtasks", [])
            
        except Exception as e:
            logger.error(f"[AutoGPTPlanner] LLM子任务生成失败: {e}")
            return self._generate_subtasks_with_prompt(goal, context)
    
    def _generate_subtasks_with_prompt(self, goal: str, context: Optional[str] = None) -> List[Dict[str, Any]]:
        """使用预定义规则生成子任务（LLM不可用时的降级方案）"""
        # 基于关键词的简单拆解规则
        subtasks = []
        goal_lower = goal.lower()
        
        # 检测任务类型并生成对应子任务
        if any(kw in goal_lower for kw in ["搜索", "查找", "search", "查询"]):
            subtasks.append({"step": 1, "task": "搜索相关信息", "tool_hint": "search"})
        
        if any(kw in goal_lower for kw in ["分析", "analyze", "评估", "评价"]):
            if not subtasks:
                subtasks.append({"step": 1, "task": "收集原始数据", "tool_hint": "search"})
            subtasks.append({"step": len(subtasks) + 1, "task": "分析数据和趋势", "tool_hint": "analyze"})
        
        if any(kw in goal_lower for kw in ["报告", "report", "总结", "summary", "写"]):
            subtasks.append({"step": len(subtasks) + 1, "task": "编写报告/总结", "tool_hint": "write"})
        
        if any(kw in goal_lower for kw in ["发送", "邮件", "email", "send"]):
            subtasks.append({"step": len(subtasks) + 1, "task": "发送结果", "tool_hint": "execute"})
        
        # 如果没有匹配任何规则，生成默认拆解
        if not subtasks:
            subtasks = [
                {"step": 1, "task": "理解任务需求", "tool_hint": "analyze"},
                {"step": 2, "task": "收集必要信息", "tool_hint": "search"},
                {"step": 3, "task": "生成结果", "tool_hint": "write"},
            ]
        
        return subtasks[:self.max_subtasks]
    
    def _analyze_dependencies(self, subtasks: List[Dict], goal: str) -> Dict[str, List[str]]:
        """分析依赖"""
        return self.analyze_dependencies(subtasks, goal)
    
    def _compute_parallel_groups(self, subtasks: List[Dict], dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """计算可并行执行的任务组
        
        基于依赖关系，将不互相依赖的任务分在同一组。
        """
        if not subtasks:
            return []
        
        # 构建任务到步骤的映射
        task_to_step = {t.get("task", f"task_{i}"): t.get("step", i + 1) for i, t in enumerate(subtasks)}
        
        # 拓扑排序分组
        groups = []
        completed = set()
        remaining = set(task_to_step.keys())
        
        while remaining:
            # 找出所有依赖已满足的任务
            ready = set()
            for task_name in remaining:
                deps = set(dependencies.get(task_name, []))
                if deps.issubset(completed):
                    ready.add(task_name)
            
            if not ready:
                # 循环依赖：将剩余任务全部放入一组
                groups.append(list(remaining))
                break
            
            groups.append(list(ready))
            completed.update(ready)
            remaining -= ready
        
        return groups
