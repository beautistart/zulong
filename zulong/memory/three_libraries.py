# File: zulong/memory/three_libraries.py
# 三库分立架构实现 (TSD v2.2)
# 技能库 + 经验库 + 知识库

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# 导入增强版经验库
from .enhanced_experience_store import (
    EnhancedExperienceStore,
    get_enhanced_experience_store,
    Experience as EnhancedExperience
)

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """技能数据结构"""
    name: str
    description: str
    instructions: List[str] = field(default_factory=list)
    safety_rules: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


class SkillStore:
    """技能库：内存常驻，极速访问
    
    特点：
    - 内存直读，0ms 延迟
    - 开发者预设 (Hard-coded)
    - 存储通用能力、系统指令、安全红线
    
    对应 TSD v2.2 第 9.5.1 节
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化技能库"""
        if not hasattr(self, '_initialized'):
            self._skills: Dict[str, Skill] = {}
            self._load_default_skills()
            self._initialized = True
            logger.info(f"[SkillStore] 初始化完成，已加载 {len(self._skills)} 个技能")
    
    def _load_default_skills(self):
        """加载默认技能集"""
        default_skills = {
            "coding": Skill(
                name="coding",
                description="代码编写能力",
                instructions=[
                    "编写清晰、可维护的代码",
                    "添加适当的注释和文档",
                    "遵循项目的代码规范",
                    "处理边界情况和异常"
                ],
                safety_rules=[
                    "不执行危险的系统命令",
                    "不访问敏感文件路径",
                    "不暴露敏感信息"
                ]
            ),
            "math": Skill(
                name="math",
                description="数学计算能力",
                instructions=[
                    "进行精确的数学计算",
                    "验证计算结果的合理性",
                    "解释计算步骤"
                ],
                safety_rules=[]
            ),
            "navigation": Skill(
                name="navigation",
                description="导航与路径规划能力",
                instructions=[
                    "规划安全、高效的路径",
                    "实时避障",
                    "动态调整路线"
                ],
                safety_rules=[
                    "禁止进入禁区",
                    "保持安全距离",
                    "紧急情况立即停止"
                ]
            ),
            "manipulation": Skill(
                name="manipulation",
                description="物体操作能力",
                instructions=[
                    "识别物体位置和姿态",
                    "规划抓取轨迹",
                    "轻柔操作避免损坏"
                ],
                safety_rules=[
                    "不操作危险物品",
                    "避免碰撞人体",
                    "操作失败时安全回退"
                ]
            ),
            "conversation": Skill(
                name="conversation",
                description="对话交互能力",
                instructions=[
                    "理解用户意图",
                    "提供有用的回答",
                    "保持礼貌和专业"
                ],
                safety_rules=[
                    "不泄露隐私信息",
                    "不执行有害指令",
                    "遇到不确定问题时诚实告知"
                ]
            ),
            "vision": Skill(
                name="vision",
                description="视觉感知能力",
                instructions=[
                    "识别物体和场景",
                    "检测运动和变化",
                    "理解空间关系"
                ],
                safety_rules=[
                    "不存储敏感图像",
                    "尊重隐私区域"
                ]
            ),
            "audio": Skill(
                name="audio",
                description="音频处理能力",
                instructions=[
                    "语音识别和理解",
                    "声音定位",
                    "音乐和声音分析"
                ],
                safety_rules=[
                    "不录制敏感对话",
                    "音量控制在安全范围"
                ]
            ),
            "task_planning": Skill(
                name="task_planning",
                description="任务规划能力",
                instructions=[
                    "分解复杂任务",
                    "确定执行顺序",
                    "预估时间和资源",
                    "处理任务依赖关系"
                ],
                safety_rules=[
                    "优先处理安全相关任务",
                    "任务失败时安全回退"
                ]
            )
        }
        
        for name, skill in default_skills.items():
            self._skills[name] = skill
    
    def get_all(self) -> Dict[str, Skill]:
        """获取所有技能，0ms 延迟
        
        Returns:
            Dict[str, Skill]: 技能字典
        """
        return self._skills.copy()
    
    def get(self, skill_name: str) -> Optional[Skill]:
        """获取单个技能
        
        Args:
            skill_name: 技能名称
            
        Returns:
            Optional[Skill]: 技能对象，不存在返回 None
        """
        return self._skills.get(skill_name)
    
    def get_instructions(self, skill_names: List[str]) -> List[str]:
        """获取多个技能的指令列表
        
        Args:
            skill_names: 技能名称列表
            
        Returns:
            List[str]: 合并后的指令列表
        """
        instructions = []
        for name in skill_names:
            skill = self._skills.get(name)
            if skill:
                instructions.extend(skill.instructions)
        return instructions
    
    def get_safety_rules(self, skill_names: List[str]) -> List[str]:
        """获取多个技能的安全规则
        
        Args:
            skill_names: 技能名称列表
            
        Returns:
            List[str]: 合并后的安全规则列表
        """
        rules = []
        for name in skill_names:
            skill = self._skills.get(name)
            if skill:
                rules.extend(skill.safety_rules)
        return rules
    
    def add_skill(self, skill: Skill):
        """添加新技能（运行时动态添加）
        
        Args:
            skill: 技能对象
        """
        self._skills[skill.name] = skill
        logger.info(f"[SkillStore] 添加技能: {skill.name}")
    
    def to_prompt_context(self, skill_names: Optional[List[str]] = None) -> str:
        """将技能转换为 Prompt 上下文
        
        Args:
            skill_names: 指定的技能名称列表，None 表示全部
            
        Returns:
            str: 格式化的上下文字符串
        """
        if skill_names is None:
            skills = list(self._skills.values())
        else:
            skills = [self._skills.get(name) for name in skill_names if name in self._skills]
        
        context_parts = ["## 系统能力"]
        for skill in skills:
            context_parts.append(f"\n### {skill.name} ({skill.description})")
            if skill.instructions:
                context_parts.append("指令:")
                for inst in skill.instructions:
                    context_parts.append(f"- {inst}")
            if skill.safety_rules:
                context_parts.append("安全规则:")
                for rule in skill.safety_rules:
                    context_parts.append(f"- {rule}")
        
        return "\n".join(context_parts)


@dataclass
class Experience:
    """经验数据结构"""
    id: str
    content: str
    experience_type: str  # "logic", "failure", "success", "preference"
    task_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class ExperienceStore:
    """经验库：向量检索 + 元数据过滤
    
    特点：
    - 语义检索 + 强过滤
    - 存储历史任务轨迹、避坑指南、用户偏好
    - 来自 L2 历史运行产生 (Auto-log)
    
    对应 TSD v2.2 第 9.5.2 节
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """初始化经验库
        
        Args:
            db_path: 向量数据库路径
        """
        if not hasattr(self, '_initialized'):
            self.db_path = db_path or "data/experience_db"
            self._experiences: Dict[str, Experience] = {}
            self._embedding_model = None
            self._initialized = True
            logger.info(f"[ExperienceStore] 初始化完成，路径: {self.db_path}")
    
    def set_embedding_model(self, model):
        """设置 Embedding 模型
        
        Args:
            model: Embedding 模型实例
        """
        self._embedding_model = model
        logger.info("[ExperienceStore] Embedding 模型已设置")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示
        
        Args:
            text: 输入文本
            
        Returns:
            np.ndarray: 向量表示
        """
        if self._embedding_model is None:
            return np.random.rand(768).astype(np.float32)
        
        try:
            if hasattr(self._embedding_model, 'encode'):
                return self._embedding_model.encode([text])[0]
            else:
                return np.random.rand(768).astype(np.float32)
        except Exception as e:
            logger.error(f"[ExperienceStore] Embedding 失败: {e}")
            return np.random.rand(768).astype(np.float32)
    
    def search(self, query_vector: np.ndarray,
               filter_type: Optional[str] = "logic",
               limit: int = 5) -> List[Experience]:
        """语义检索 + 强过滤
        
        Args:
            query_vector: 查询向量
            filter_type: 过滤类型 ("logic", "failure", "success", "preference")
            limit: 返回数量限制
            
        Returns:
            List[Experience]: 匹配的经验列表
        """
        results = []
        
        for exp in self._experiences.values():
            if filter_type and exp.experience_type != filter_type:
                continue
            
            if exp.embedding is not None:
                similarity = np.dot(query_vector, exp.embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(exp.embedding) + 1e-8
                )
                results.append((similarity, exp))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in results[:limit]]
    
    def search_by_text(self, query: str,
                       filter_type: Optional[str] = "logic",
                       limit: int = 5) -> List[Experience]:
        """通过文本查询
        
        Args:
            query: 查询文本
            filter_type: 过滤类型
            limit: 返回数量限制
            
        Returns:
            List[Experience]: 匹配的经验列表
        """
        query_vector = self._get_embedding(query)
        return self.search(query_vector, filter_type, limit)
    
    def add_experience(self, content: str,
                       experience_type: str = "logic",
                       task_id: Optional[str] = None,
                       success: bool = True,
                       metadata: Optional[Dict] = None) -> str:
        """添加新经验
        
        Args:
            content: 经验内容
            experience_type: 经验类型
            task_id: 关联的任务 ID
            success: 是否成功
            metadata: 元数据
            
        Returns:
            str: 经验 ID
        """
        exp_id = str(uuid.uuid4())
        embedding = self._get_embedding(content)
        
        experience = Experience(
            id=exp_id,
            content=content,
            experience_type=experience_type,
            task_id=task_id,
            success=success,
            metadata=metadata or {},
            embedding=embedding
        )
        
        self._experiences[exp_id] = experience
        logger.info(f"[ExperienceStore] 添加经验: {exp_id[:8]}... (类型: {experience_type})")
        
        return exp_id
    
    def get(self, exp_id: str) -> Optional[Experience]:
        """获取单个经验
        
        Args:
            exp_id: 经验 ID
            
        Returns:
            Optional[Experience]: 经验对象
        """
        return self._experiences.get(exp_id)
    
    def delete(self, exp_id: str) -> bool:
        """删除单个经验
        
        Args:
            exp_id: 经验 ID
            
        Returns:
            bool: 是否删除成功
        """
        if exp_id in self._experiences:
            del self._experiences[exp_id]
            logger.info(f"[ExperienceStore] 删除经验: {exp_id[:8]}...")
            return True
        return False
    
    def delete_by_task_id(self, task_id: str) -> int:
        """删除指定任务 ID 的所有经验
        
        Args:
            task_id: 任务 ID
            
        Returns:
            int: 删除的数量
        """
        to_delete = [
            eid for eid, exp in self._experiences.items()
            if exp.task_id == task_id
        ]
        for eid in to_delete:
            del self._experiences[eid]
        
        if to_delete:
            logger.info(f"[ExperienceStore] 删除 {len(to_delete)} 条经验 (task_id: {task_id})")
        return len(to_delete)
    
    def clear_all(self) -> int:
        """清空所有经验
        
        Returns:
            int: 删除的数量
        """
        count = len(self._experiences)
        self._experiences.clear()
        logger.info(f"[ExperienceStore] 清空所有经验: {count} 条")
        return count
    
    def get_recent(self, limit: int = 10) -> List[Experience]:
        """获取最近的经验
        
        Args:
            limit: 返回数量限制
            
        Returns:
            List[Experience]: 最近的经验列表
        """
        sorted_exps = sorted(
            self._experiences.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )
        return sorted_exps[:limit]
    
    def to_prompt_context(self, experiences: List[Experience]) -> str:
        """将经验转换为 Prompt 上下文
        
        Args:
            experiences: 经验列表
            
        Returns:
            str: 格式化的上下文字符串
        """
        if not experiences:
            return ""
        
        context_parts = ["## 相关经验"]
        for exp in experiences:
            status = "✅" if exp.success else "❌"
            context_parts.append(f"\n{status} [{exp.experience_type}] {exp.content}")
        
        return "\n".join(context_parts)


@dataclass
class Knowledge:
    """知识数据结构"""
    id: str
    content: str
    source: str  # 来源：文档名、URL 等
    domain: str  # 领域：产品手册、法规、百科等
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class KnowledgeStore:
    """知识库：按需检索，异步 I/O
    
    特点：
    - 异步检索，不阻塞主线程
    - 存储外部事实、产品手册、文档、行业法规
    - 来自用户上传/外部导入 (RAG)
    
    对应 TSD v2.2 第 9.5.3 节
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """初始化知识库
        
        Args:
            db_path: 向量数据库路径
        """
        if not hasattr(self, '_initialized'):
            self.db_path = db_path or "data/knowledge_db"
            self._knowledge: Dict[str, Knowledge] = {}
            self._embedding_model = None
            self._domain_index: Dict[str, List[str]] = {}
            self._initialized = True
            logger.info(f"[KnowledgeStore] 初始化完成，路径: {self.db_path}")
    
    def set_embedding_model(self, model):
        """设置 Embedding 模型
        
        Args:
            model: Embedding 模型实例
        """
        self._embedding_model = model
        logger.info("[KnowledgeStore] Embedding 模型已设置")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        if self._embedding_model is None:
            return np.random.rand(768).astype(np.float32)
        
        try:
            if hasattr(self._embedding_model, 'encode'):
                return self._embedding_model.encode([text])[0]
            else:
                return np.random.rand(768).astype(np.float32)
        except Exception as e:
            logger.error(f"[KnowledgeStore] Embedding 失败: {e}")
            return np.random.rand(768).astype(np.float32)
    
    async def search_async(self, query_vector: np.ndarray,
                           domain: Optional[str] = None,
                           limit: int = 10) -> List[Knowledge]:
        """异步检索，不阻塞主线程
        
        Args:
            query_vector: 查询向量
            domain: 领域过滤
            limit: 返回数量限制
            
        Returns:
            List[Knowledge]: 匹配的知识列表
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._sync_search(query_vector, domain, limit)
        )
        return results
    
    def _sync_search(self, query_vector: np.ndarray,
                     domain: Optional[str] = None,
                     limit: int = 10) -> List[Knowledge]:
        """同步检索实现"""
        results = []
        
        candidates = self._knowledge.values()
        if domain and domain in self._domain_index:
            candidates = [self._knowledge[kid] for kid in self._domain_index[domain]]
        
        for knowledge in candidates:
            if knowledge.embedding is not None:
                similarity = np.dot(query_vector, knowledge.embedding) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(knowledge.embedding) + 1e-8
                )
                results.append((similarity, knowledge))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [k for _, k in results[:limit]]
    
    async def search_by_text_async(self, query: str,
                                    domain: Optional[str] = None,
                                    limit: int = 10) -> List[Knowledge]:
        """通过文本异步查询
        
        Args:
            query: 查询文本
            domain: 领域过滤
            limit: 返回数量限制
            
        Returns:
            List[Knowledge]: 匹配的知识列表
        """
        query_vector = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._get_embedding(query)
        )
        return await self.search_async(query_vector, domain, limit)
    
    def add_knowledge(self, content: str,
                      source: str,
                      domain: str,
                      metadata: Optional[Dict] = None) -> str:
        """添加新知识
        
        Args:
            content: 知识内容
            source: 来源
            domain: 领域
            metadata: 元数据
            
        Returns:
            str: 知识 ID
        """
        knowledge_id = str(uuid.uuid4())
        embedding = self._get_embedding(content)
        
        knowledge = Knowledge(
            id=knowledge_id,
            content=content,
            source=source,
            domain=domain,
            metadata=metadata or {},
            embedding=embedding
        )
        
        self._knowledge[knowledge_id] = knowledge
        
        if domain not in self._domain_index:
            self._domain_index[domain] = []
        self._domain_index[domain].append(knowledge_id)
        
        logger.info(f"[KnowledgeStore] 添加知识: {knowledge_id[:8]}... (领域: {domain})")
        
        return knowledge_id
    
    def get(self, knowledge_id: str) -> Optional[Knowledge]:
        """获取单个知识"""
        return self._knowledge.get(knowledge_id)
    
    def delete(self, knowledge_id: str) -> bool:
        """删除单个知识
        
        Args:
            knowledge_id: 知识 ID
            
        Returns:
            bool: 是否删除成功
        """
        if knowledge_id in self._knowledge:
            knowledge = self._knowledge[knowledge_id]
            domain = knowledge.domain
            if domain in self._domain_index:
                if knowledge_id in self._domain_index[domain]:
                    self._domain_index[domain].remove(knowledge_id)
            del self._knowledge[knowledge_id]
            logger.info(f"[KnowledgeStore] 删除知识: {knowledge_id[:8]}...")
            return True
        return False
    
    def delete_by_domain(self, domain: str) -> int:
        """删除指定领域的所有知识
        
        Args:
            domain: 领域名称
            
        Returns:
            int: 删除的数量
        """
        if domain not in self._domain_index:
            return 0
        
        to_delete = self._domain_index[domain]
        for kid in to_delete:
            if kid in self._knowledge:
                del self._knowledge[kid]
        
        del self._domain_index[domain]
        logger.info(f"[KnowledgeStore] 删除 {len(to_delete)} 条知识 (领域: {domain})")
        return len(to_delete)
    
    def clear_all(self) -> int:
        """清空所有知识
        
        Returns:
            int: 删除的数量
        """
        count = len(self._knowledge)
        self._knowledge.clear()
        self._domain_index.clear()
        logger.info(f"[KnowledgeStore] 清空所有知识: {count} 条")
        return count
    
    def get_by_domain(self, domain: str) -> List[Knowledge]:
        """获取指定领域的所有知识
        
        Args:
            domain: 领域名称
            
        Returns:
            List[Knowledge]: 知识列表
        """
        if domain not in self._domain_index:
            return []
        return [self._knowledge[kid] for kid in self._domain_index[domain]]
    
    def to_prompt_context(self, knowledge_list: List[Knowledge]) -> str:
        """将知识转换为 Prompt 上下文
        
        Args:
            knowledge_list: 知识列表
            
        Returns:
            str: 格式化的上下文字符串
        """
        if not knowledge_list:
            return ""
        
        context_parts = ["## 相关知识"]
        for k in knowledge_list:
            context_parts.append(f"\n[{k.domain}] {k.content}")
            context_parts.append(f"  来源: {k.source}")
        
        return "\n".join(context_parts)


class ThreeLibraryManager:
    """三库管理器：统一管理技能库、经验库、知识库
    
    对应 TSD v2.2 第 9.3 节
    """
    
    _instance = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化三库管理器"""
        if not hasattr(self, '_initialized'):
            self.skill_store = SkillStore()
            self.experience_store = ExperienceStore()
            self.knowledge_store = KnowledgeStore()
            self._initialized = True
            logger.info("[ThreeLibraryManager] 三库管理器初始化完成")
    
    def set_embedding_model(self, model):
        """设置 Embedding 模型（用于经验库和知识库）"""
        self.experience_store.set_embedding_model(model)
        self.knowledge_store.set_embedding_model(model)
    
    async def retrieve_all(self, query: str,
                           skill_names: Optional[List[str]] = None,
                           experience_type: Optional[str] = "logic",
                           knowledge_domain: Optional[str] = None,
                           experience_limit: int = 5,
                           knowledge_limit: int = 10) -> Dict[str, Any]:
        """并行检索三个库
        
        Args:
            query: 查询文本
            skill_names: 指定的技能名称
            experience_type: 经验类型过滤
            knowledge_domain: 知识领域过滤
            experience_limit: 经验返回数量
            knowledge_limit: 知识返回数量
            
        Returns:
            Dict[str, Any]: 检索结果
        """
        skills = self.skill_store.get_all()
        if skill_names:
            skills = {k: v for k, v in skills.items() if k in skill_names}
        
        experiences = self.experience_store.search_by_text(
            query, experience_type, experience_limit
        )
        
        knowledge = await self.knowledge_store.search_by_text_async(
            query, knowledge_domain, knowledge_limit
        )
        
        return {
            "skills": skills,
            "experiences": experiences,
            "knowledge": knowledge
        }
    
    def build_super_prompt(self, query: str,
                           skills: Optional[Dict] = None,
                           experiences: Optional[List] = None,
                           knowledge: Optional[List] = None) -> str:
        """构建超级 Prompt
        
        Args:
            query: 用户查询
            skills: 技能字典
            experiences: 经验列表
            knowledge: 知识列表
            
        Returns:
            str: 完整的 Prompt
        """
        prompt_parts = []
        
        if skills:
            prompt_parts.append(self.skill_store.to_prompt_context(list(skills.keys())))
        
        if experiences:
            prompt_parts.append(self.experience_store.to_prompt_context(experiences))
        
        if knowledge:
            prompt_parts.append(self.knowledge_store.to_prompt_context(knowledge))
        
        prompt_parts.append(f"\n## 用户请求\n{query}")
        
        return "\n\n".join(prompt_parts)


def get_skill_store() -> SkillStore:
    """获取技能库单例"""
    return SkillStore()


def get_experience_store() -> ExperienceStore:
    """获取经验库单例"""
    return ExperienceStore()


def get_knowledge_store() -> KnowledgeStore:
    """获取知识库单例"""
    return KnowledgeStore()


def get_three_library_manager() -> ThreeLibraryManager:
    """获取三库管理器单例"""
    return ThreeLibraryManager()
