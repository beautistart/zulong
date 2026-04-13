# File: zulong/skill_packs/packs/openmanus_reasoner/__init__.py
"""
OpenManus 深度推理技能包

从 OpenManus 中提取的推理链（Reasoning Chain）逻辑。
擅长处理复杂逻辑推理、科研计算、数学证明等需要深度思考的任务。

与 AutoGPT 的分工：
- AutoGPT: 流程化任务拆解（搜索+分析+总结）
- OpenManus: 深度逻辑推理（设计算法、数学证明、架构设计）
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import logging
import time

from zulong.skill_packs.interface import ISkillPack, SkillPackManifest
from zulong.tools.base import BaseTool, ToolCategory, ToolRequest, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    step_type: str  # "analyze", "hypothesize", "verify", "conclude"
    description: str
    content: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningChain:
    """推理链"""
    problem: str
    context: str
    steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class DeepReasoningTool(BaseTool):
    """深度推理工具"""
    
    def __init__(self, reasoner_ref=None):
        super().__init__(name="deep_reasoning", category=ToolCategory.CUSTOM)
        self.description = "深度推理工具。适用于算法设计、数学证明、复杂系统架构等高难度任务。参数：problem(问题描述), context(可选上下文)"
        self.reasoner_ref = reasoner_ref
    
    def initialize(self):
        return True
    
    def cleanup(self):
        pass
    
    def execute(self, request: ToolRequest) -> ToolResult:
        problem = request.parameters.get("problem", "")
        if not problem:
            return self._create_result(success=False, error="缺少问题描述", request_id=request.request_id)
        
        context = request.parameters.get("context", "")
        reasoning_depth = request.parameters.get("reasoning_depth", 3)
        max_hypotheses = request.parameters.get("max_hypotheses", 5)
        
        if self.reasoner_ref:
            result = self.reasoner_ref._deep_reason(
                problem=problem,
                context=context,
                reasoning_depth=reasoning_depth,
                max_hypotheses=max_hypotheses
            )
            return self._create_result(success=True, data=result, request_id=request.request_id)
        
        return self._create_result(
            success=False, 
            error="推理器未初始化", 
            request_id=request.request_id
        )
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "需要深度推理的问题"},
                "context": {"type": "string", "description": "当前上下文（可选）"},
                "reasoning_depth": {"type": "integer", "description": "推理深度（1-5），默认 3"},
                "max_hypotheses": {"type": "integer", "description": "最大假设数量，默认 5"}
            },
            "required": ["problem"]
        }


class OpenManusReasonerPack(ISkillPack):
    """OpenManus 深度推理技能包
    
    核心算法从 OpenManus 的 Reasoning Chain 中提取：
    1. 问题分析：理解问题本质，识别关键约束
    2. 假设生成：提出多个可能的解决路径
    3. 假设验证：逐一检验每条路径的可行性
    4. 方案选择：选出最优路径并输出详细步骤
    """
    
    def __init__(self):
        self._manifest = None
        self._tools = [DeepReasoningTool(self)]
        self._reasoning_history: List[ReasoningChain] = []
        logger.info("[OpenManusReasonerPack] Initialized")
    
    def get_manifest(self) -> SkillPackManifest:
        """返回技能包清单"""
        if self._manifest is None:
            self._manifest = SkillPackManifest(
                pack_id="openmanus_reasoner",
                name="OpenManus深度推理",
                version="1.0.0",
                description="基于 OpenManus 推理链的深度逻辑推理能力",
                capabilities=["deep_reasoning", "logic_chain", "problem_decompose"],
                dependencies=[],
                resource_requirements={"cpu_mb": 256, "gpu_mb": 0},
                learning_objectives=["复杂逻辑推理模式", "多步推理链构建", "假设验证策略"],
                source="openmanus"
            )
        return self._manifest
    
    def install(self, tool_registry, config: Dict[str, Any] = None) -> bool:
        """安装技能包，注册工具"""
        try:
            for tool in self._tools:
                try:
                    tool_registry.register(tool)
                except Exception:
                    pass
            logger.info("[OpenManusReasonerPack] DeepReasoningTool registered")
            return True
        except Exception as e:
            logger.error(f"[OpenManusReasonerPack] Install failed: {e}")
            return False
    
    def execute(self, capability: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行技能包能力"""
        if capability == "deep_reasoning":
            return self._deep_reason(
                problem=params.get("problem", ""),
                context=params.get("context", ""),
                reasoning_depth=params.get("reasoning_depth", 3),
                max_hypotheses=params.get("max_hypotheses", 5)
            )
        else:
            return {
                "success": False,
                "error": f"Unknown capability: {capability}"
            }
    
    def get_tools(self):
        """返回提供的工具列表"""
        return self._tools
    
    def uninstall(self) -> bool:
        """卸载技能包"""
        self._tools = []
        self._reasoning_history.clear()
        logger.info("[OpenManusReasonerPack] Uninstalled")
        return True
    
    def _deep_reason(self, problem: str, context: str, 
                     reasoning_depth: int = 3, max_hypotheses: int = 5) -> Dict[str, Any]:
        """核心深度推理算法
        
        Args:
            problem: 问题描述
            context: 上下文信息
            reasoning_depth: 推理深度（1-5）
            max_hypotheses: 最大假设数量
            
        Returns:
            推理结果
        """
        logger.info(f"[OpenManusReasonerPack] Starting deep reasoning: {problem[:50]}...")
        
        reasoning_chain = ReasoningChain(
            problem=problem,
            context=context
        )
        
        try:
            # 步骤 1: 问题分析
            logger.info("[OpenManusReasonerPack] Step 1: Analyzing problem...")
            analysis_step = self._analyze_problem(problem, context)
            reasoning_chain.steps.append(analysis_step)
            
            # 步骤 2: 假设生成
            logger.info(f"[OpenManusReasonerPack] Step 2: Generating hypotheses (max={max_hypotheses})...")
            hypotheses = self._generate_hypotheses(problem, context, max_hypotheses)
            for hyp in hypotheses:
                reasoning_chain.steps.append(hyp)
            
            # 步骤 3: 假设验证
            logger.info(f"[OpenManusReasonerPack] Step 3: Verifying hypotheses (depth={reasoning_depth})...")
            verified_hypotheses = self._verify_hypotheses(hypotheses, problem, context, reasoning_depth)
            
            # 步骤 4: 方案选择
            logger.info("[OpenManusReasonerPack] Step 4: Selecting optimal solution...")
            conclusion_step = self._select_solution(verified_hypotheses, problem)
            reasoning_chain.steps.append(conclusion_step)
            
            # 完成推理链
            reasoning_chain.conclusion = conclusion_step.content
            reasoning_chain.confidence = conclusion_step.confidence
            reasoning_chain.completed_at = time.time()
            
            # 保存历史
            self._reasoning_history.append(reasoning_chain)
            
            # 返回结果
            return {
                "success": True,
                "reasoning_chain": [
                    {
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                        "description": step.description,
                        "content": step.content,
                        "confidence": step.confidence
                    }
                    for step in reasoning_chain.steps
                ],
                "conclusion": reasoning_chain.conclusion,
                "confidence": reasoning_chain.confidence,
                "elapsed_time": reasoning_chain.completed_at - reasoning_chain.started_at
            }
            
        except Exception as e:
            logger.error(f"[OpenManusReasonerPack] Deep reasoning failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "problem": problem
            }
    
    def _analyze_problem(self, problem: str, context: str) -> ReasoningStep:
        """步骤 1: 问题分析
        
        识别问题的关键要素、约束条件和目标。
        """
        # 提取关键要素
        key_elements = self._extract_key_elements(problem)
        
        # 识别约束条件
        constraints = self._identify_constraints(problem, context)
        
        # 构建分析结果
        analysis_content = (
            f"【问题分析】\n"
            f"关键要素: {', '.join(key_elements)}\n"
            f"约束条件: {', '.join(constraints)}\n"
            f"目标: 为问题 '{problem[:100]}' 寻找最优解决方案"
        )
        
        return ReasoningStep(
            step_id="analysis_1",
            step_type="analyze",
            description="问题分析",
            content=analysis_content,
            confidence=0.9
        )
    
    def _generate_hypotheses(self, problem: str, context: str, 
                            max_hypotheses: int) -> List[ReasoningStep]:
        """步骤 2: 假设生成
        
        基于问题分析和上下文，生成多个可能的解决路径。
        """
        hypotheses = []
        
        # 策略 1: 分解法 - 将问题拆分为子问题
        hypotheses.append(ReasoningStep(
            step_id="hypothesis_decompose",
            step_type="hypothesize",
            description="分解法：将问题拆解为子问题",
            content="采用分治策略，将复杂问题分解为可独立解决的子问题，然后组合解决方案。",
            confidence=0.8
        ))
        
        # 策略 2: 类比法 - 寻找相似问题的解决方案
        hypotheses.append(ReasoningStep(
            step_id="hypothesis_analogy",
            step_type="hypothesize",
            description="类比法：借鉴相似问题的解决方案",
            content="寻找与当前问题结构相似的已知问题，借鉴其解决思路和方法。",
            confidence=0.7
        ))
        
        # 策略 3: 逆向法 - 从目标反推
        hypotheses.append(ReasoningStep(
            step_id="hypothesis_reverse",
            step_type="hypothesize",
            description="逆向法：从目标状态反推",
            content="从期望的最终状态出发，逆向推导达到该状态需要的条件和步骤。",
            confidence=0.75
        ))
        
        # 策略 4: 归纳法 - 从特例到一般
        if len(hypotheses) < max_hypotheses:
            hypotheses.append(ReasoningStep(
                step_id="hypothesis_induction",
                step_type="hypothesize",
                description="归纳法：从特例推导一般规律",
                content="先解决简化版本的特例，从中发现一般规律，再应用到原问题。",
                confidence=0.65
            ))
        
        # 策略 5: 转化法 - 转化为已知问题
        if len(hypotheses) < max_hypotheses:
            hypotheses.append(ReasoningStep(
                step_id="hypothesis_transform",
                step_type="hypothesize",
                description="转化法：转化为已知的可解问题",
                content="将原问题通过等价变换转化为已有解决方案的问题类型。",
                confidence=0.7
            ))
        
        return hypotheses[:max_hypotheses]
    
    def _verify_hypotheses(self, hypotheses: List[ReasoningStep], 
                          problem: str, context: str,
                          depth: int) -> List[ReasoningStep]:
        """步骤 3: 假设验证
        
        对每个假设进行深度验证，评估可行性。
        """
        verified = []
        
        for hyp in hypotheses:
            # 模拟验证过程
            feasibility = self._assess_feasibility(hyp, problem, context, depth)
            
            # 更新置信度
            hyp.confidence = hyp.confidence * feasibility
            hyp.content += f"\n\n【验证结果】\n可行性评分: {feasibility:.2f}\n"
            
            if feasibility > 0.5:  # 只保留可行性 > 50% 的假设
                verified.append(hyp)
        
        # 按置信度排序
        verified.sort(key=lambda x: x.confidence, reverse=True)
        
        return verified
    
    def _select_solution(self, verified_hypotheses: List[ReasoningStep], 
                        problem: str) -> ReasoningStep:
        """步骤 4: 方案选择
        
        从验证通过的假设中选择最优方案。
        """
        if not verified_hypotheses:
            return ReasoningStep(
                step_id="conclusion",
                step_type="conclude",
                description="无可行方案",
                content="经过分析，未能找到可行的解决方案。建议重新审视问题或增加约束条件。",
                confidence=0.0
            )
        
        # 选择置信度最高的方案
        best_hypothesis = verified_hypotheses[0]
        
        conclusion_content = (
            f"【推理结论】\n"
            f"最优方案: {best_hypothesis.description}\n"
            f"方案详情: {best_hypothesis.content}\n"
            f"置信度: {best_hypothesis.confidence:.2f}\n"
            f"备选方案数量: {len(verified_hypotheses) - 1}"
        )
        
        return ReasoningStep(
            step_id="conclusion",
            step_type="conclude",
            description="推理结论",
            content=conclusion_content,
            confidence=best_hypothesis.confidence
        )
    
    def _extract_key_elements(self, problem: str) -> List[str]:
        """提取问题的关键要素"""
        elements = []
        
        # 简单的关键词提取（实际应使用 NLP）
        keywords = ["设计", "算法", "系统", "优化", "分析", "实现", "构建", "评估"]
        for kw in keywords:
            if kw in problem:
                elements.append(kw)
        
        if not elements:
            elements.append("通用问题")
        
        return elements
    
    def _identify_constraints(self, problem: str, context: str) -> List[str]:
        """识别约束条件"""
        constraints = []
        
        constraint_keywords = ["必须", "限制", "约束", "不超过", "至少", "最大", "最小"]
        for kw in constraint_keywords:
            if kw in problem:
                constraints.append(kw)
        
        if not constraints:
            constraints.append("无明显约束")
        
        return constraints
    
    def _assess_feasibility(self, hypothesis: ReasoningStep, 
                           problem: str, context: str, 
                           depth: int) -> float:
        """评估假设的可行性
        
        基于以下因素：
        - 与问题的匹配度
        - 资源的可用性
        - 历史的成功率
        - 推理深度
        """
        # 基础可行性
        base_feasibility = 0.7
        
        # 深度惩罚：深度越大，可行性越低
        depth_penalty = max(0.5, 1.0 - (depth - 1) * 0.1)
        
        # 历史成功率（如果有历史数据）
        history_bonus = 0.1 if len(self._reasoning_history) > 0 else 0.0
        
        feasibility = base_feasibility * depth_penalty + history_bonus
        
        return min(1.0, feasibility)
