# File: zulong/memory/tagging_engine.py
# 信息分类与打标引擎 - RAG 系统的核心智能

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import re

logger = logging.getLogger(__name__)


class ImportanceLevel(Enum):
    """重要性等级"""
    MUST_LEARN = "must_learn"      # 必须学习（关键信息）
    SHOULD_LEARN = "should_learn"  # 应该学习（有用信息）
    PENDING = "pending"            # 待定（需要进一步判断）
    NOT_NEEDED = "not_needed"      # 不需要（垃圾信息）


class MemorabilityLevel(Enum):
    """记忆性等级"""
    MUST_REMEMBER = "must_remember"  # 必须记住（长期记忆）
    SHOULD_REMEMBER = "should_remember"  # 应该记住（中期记忆）
    PENDING = "pending"              # 待定
    FORGET = "forget"                # 不用记住（临时信息）


class DomainType(Enum):
    """领域类型"""
    NAVIGATION = "navigation"    # 导航
    MANIPULATION = "manipulation"  # 操作
    VISION = "vision"            # 视觉
    DIALOG = "dialog"            # 对话
    GENERAL = "general"          # 通用
    EMERGENCY = "emergency"      # 紧急情况


class MemoryType(Enum):
    """记忆类型"""
    CONTEXT = "context"          # 上下文
    EVENT = "event"              # 事件
    CONVERSATION = "conversation"  # 对话
    INSTRUCTION = "instruction"    # 指令
    EXPERIENCE = "experience"      # 经验


class ExperienceCategory(Enum):
    """经验类别"""
    TASK_SUCCESS = "task_success"      # 任务成功
    TASK_FAILURE = "task_failure"      # 任务失败
    SKILL_USAGE = "skill_usage"        # 技能使用
    SYSTEM_PROMPT = "system_prompt"    # 系统提示词


@dataclass
class TaggingResult:
    """打标结果"""
    importance: ImportanceLevel = ImportanceLevel.PENDING
    memorability: MemorabilityLevel = MemorabilityLevel.PENDING
    domain: DomainType = DomainType.GENERAL
    memory_type: MemoryType = MemoryType.CONTEXT
    experience_category: Optional[ExperienceCategory] = None
    
    # 附加标签
    sentiment: str = "neutral"  # 情感：positive/negative/neutral
    urgency: str = "normal"     # 紧急度：high/normal/low
    confidence: float = 0.0     # 置信度（0-1）
    
    # 目标 RAG 库
    target_rag: str = "memory"  # memory/experience/knowledge
    
    # 元数据
    tags: List[str] = field(default_factory=list)  # 自定义标签列表
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "importance": self.importance.value,
            "memorability": self.memorability.value,
            "domain": self.domain.value,
            "memory_type": self.memory_type.value,
            "experience_category": self.experience_category.value if self.experience_category else None,
            "sentiment": self.sentiment,
            "urgency": self.urgency,
            "confidence": self.confidence,
            "target_rag": self.target_rag,
            "tags": self.tags,
            "metadata": self.metadata
        }


class RuleBasedTagger:
    """基于规则的打标器
    
    TSD v1.7 对应规则:
    - 信息分类打标规则
    - 关键词匹配
    - 模式识别
    """
    
    def __init__(self):
        """初始化规则打标器"""
        
        # 领域关键词
        self.domain_keywords = {
            DomainType.NAVIGATION: [
                "导航", "路径", "位置", "地图", "坐标", "方向", "距离",
                "前往", "到达", "路线", " waypoint", "gps"
            ],
            DomainType.MANIPULATION: [
                "抓取", "操作", "夹持", "物体", "机械臂", "末端",
                "pick", "place", "grasp", "manipulate"
            ],
            DomainType.VISION: [
                "视觉", "图像", "识别", "检测", "摄像头", "rgb",
                "depth", "segmentation", "detection", "recognition"
            ],
            DomainType.DIALOG: [
                "对话", "聊天", "说话", "回答", "问题", "指令",
                "你好", "谢谢", "再见", "请问", "帮我"
            ],
            DomainType.EMERGENCY: [
                "紧急", "危险", "警告", "错误", "故障", "异常",
                "救命", "急", "stop", "emergency"  # 移除"帮助"，避免误判
            ]
        }
        
        # 重要性关键词
        self.importance_keywords = {
            ImportanceLevel.MUST_LEARN: [
                "必须", "一定", "绝对", "关键", "重要", "记住",
                "永远", "always", "must", "critical"
            ],
            ImportanceLevel.SHOULD_LEARN: [
                "应该", "最好", "建议", "有用", "useful", "should"
            ],
            ImportanceLevel.NOT_NEEDED: [
                "不用", "不必", "无所谓", "不重要", "随便",
                "forget", "ignore", "doesn't matter"
            ]
        }
        
        # 记忆性关键词
        self.memorability_keywords = {
            MemorabilityLevel.MUST_REMEMBER: [
                "记住", "牢记", "永远记住", "别忘了", "重要",
                "remember", "never forget", "important"
            ],
            MemorabilityLevel.FORGET: [
                "忘了", "不用记", "临时", "暂时", "一会儿",
                "forget", "temporary", "just now"
            ]
        }
        
        # 经验类别模式
        self.experience_patterns = {
            ExperienceCategory.TASK_SUCCESS: [
                r"成功.*任务", r"完成.*了", r"做到了", r"success",
                r"accomplished", r"completed"
            ],
            ExperienceCategory.TASK_FAILURE: [
                r"失败", r"没成功", r"错误", r"有问题",
                r"fail", r"error", r"wrong"
            ],
            ExperienceCategory.SKILL_USAGE: [
                r"使用.*工具", r"用.*方法", r"通过.*技能",
                r"use.*tool", r"skill"
            ]
        }
        
        logger.info("[RuleBasedTagger] Initialized with rules")
    
    def tag(self, content: str, context: Optional[Dict] = None) -> TaggingResult:
        """对内容进行打标"""
        result = TaggingResult()
        
        # 1. 领域识别
        result.domain = self._identifyDomain(content)
        
        # 2. 重要性评估
        result.importance = self._assessImportance(content)
        
        # 3. 记忆性评估
        result.memorability = self._assessMemorability(content)
        
        # 4. 记忆类型识别
        result.memory_type = self._identifyMemoryType(content)
        
        # 5. 经验类别识别
        result.experience_category = self._identifyExperienceCategory(content)
        
        # 6. 情感分析（简化版）
        result.sentiment = self._analyzeSentiment(content)
        
        # 7. 紧急度评估
        result.urgency = self._assessUrgency(content)
        
        # 8. 确定目标 RAG 库
        result.target_rag = self._determineTargetRAG(result)
        
        # 9. 计算置信度
        result.confidence = self._calculateConfidence(content, result)
        
        # 10. 提取自定义标签
        result.tags = self._extractTags(content, result)
        
        logger.debug(f"[RuleBasedTagger] Tagged: domain={result.domain.value}, "
                    f"importance={result.importance.value}, target={result.target_rag}")
        
        return result
    
    def _identifyDomain(self, content: str) -> DomainType:
        """领域识别"""
        content_lower = content.lower()
        
        # 检查紧急领域（优先级最高）
        for keyword in self.domain_keywords[DomainType.EMERGENCY]:
            if keyword.lower() in content_lower:
                return DomainType.EMERGENCY
        
        # 检查其他领域
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            if domain == DomainType.EMERGENCY:
                continue
            score = sum(1 for kw in keywords if kw.lower() in content_lower)
            domain_scores[domain] = score
        
        # 返回得分最高的领域
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            if best_domain[1] > 0:
                return best_domain[0]
        
        return DomainType.GENERAL
    
    def _assessImportance(self, content: str) -> ImportanceLevel:
        """重要性评估"""
        content_lower = content.lower()
        
        for level, keywords in self.importance_keywords.items():
            for kw in keywords:
                if kw.lower() in content_lower:
                    return level
        
        return ImportanceLevel.PENDING
    
    def _assessMemorability(self, content: str) -> MemorabilityLevel:
        """记忆性评估"""
        content_lower = content.lower()
        
        for level, keywords in self.memorability_keywords.items():
            for kw in keywords:
                if kw.lower() in content_lower:
                    return level
        
        return MemorabilityLevel.PENDING
    
    def _identifyMemoryType(self, content: str) -> MemoryType:
        """记忆类型识别"""
        content_lower = content.lower()
        
        # 🔥 关键修复 1: 优先检测用户个人信息（职业、姓名、爱好等）
        # 检测姓名介绍
        if any(kw in content_lower for kw in ["我叫", "我是", "名字", "姓名"]):
            # 排除疑问句
            if "?" not in content and "？" not in content:
                return MemoryType.CONVERSATION  # 自我介绍归类为对话，但会存储到 memory RAG
        
        # 检测职业信息
        if any(kw in content_lower for kw in ["是...设计师", "是...老师", "是...医生", "是...工程师", 
                                               "职业", "工作", "从事"]):
            return MemoryType.CONVERSATION  # 职业信息也归类为对话
        
        # 检测爱好/喜好
        if any(kw in content_lower for kw in ["喜欢", "爱", "爱好", "最爱", "经常"]):
            return MemoryType.CONVERSATION  # 爱好信息归类为对话
        
        # 指令识别（增强版：要求完整指令模式）
        if any(kw in content_lower for kw in ["请帮我", "请去", "请执行", "请你", "帮我做", "去做", "instruction"]):
            return MemoryType.INSTRUCTION
        
        # 对话识别
        if any(kw in content_lower for kw in ["你好", "谢谢", "再见", "hello", "thanks"]):
            return MemoryType.CONVERSATION
        
        # 事件识别
        if any(kw in content_lower for kw in ["发生", "出现", "看到", "遇到", "happened"]):
            return MemoryType.EVENT
        
        # 经验识别
        if any(kw in content_lower for kw in ["上次", "之前", "曾经", "经验", "before", "experience"]):
            return MemoryType.EXPERIENCE
        
        return MemoryType.CONTEXT
    
    def _identifyExperienceCategory(self, content: str) -> Optional[ExperienceCategory]:
        """经验类别识别"""
        content_lower = content.lower()
        
        for category, patterns in self.experience_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return category
        
        return None
    
    def _analyzeSentiment(self, content: str) -> str:
        """情感分析（简化版）"""
        content_lower = content.lower()
        
        positive_words = ["好", "棒", "成功", "开心", "满意", "good", "great", "happy"]
        negative_words = ["坏", "差", "失败", "生气", "失望", "bad", "wrong", "angry"]
        
        pos_score = sum(1 for w in positive_words if w in content_lower)
        neg_score = sum(1 for w in negative_words if w in content_lower)
        
        if pos_score > neg_score:
            return "positive"
        elif neg_score > pos_score:
            return "negative"
        else:
            return "neutral"
    
    def _assessUrgency(self, content: str) -> str:
        """紧急度评估"""
        content_lower = content.lower()
        
        urgent_words = ["紧急", "急", "快", "马上", "立刻", "urgent", "hurry", "now"]
        
        if any(w in content_lower for w in urgent_words):
            return "high"
        
        # 检查是否有感叹号（表示紧急）
        if content.count("!") > 1 or content.count("！") > 1:
            return "high"
        
        return "normal"
    
    def _determineTargetRAG(self, result: TaggingResult) -> str:
        """确定目标 RAG 库"""
        # 经验 -> 经验 RAG
        if result.experience_category is not None:
            return "experience"
        
        # 紧急领域 -> 知识 RAG（高优先级）
        if result.domain == DomainType.EMERGENCY:
            return "knowledge"
        
        # 紧急/重要信息 -> 知识 RAG
        if result.importance == ImportanceLevel.MUST_LEARN:
            return "knowledge"
        
        # 领域知识 -> 知识 RAG
        if result.domain in [DomainType.NAVIGATION, DomainType.MANIPULATION, DomainType.VISION]:
            return "knowledge"
        
        # 对话/上下文 -> 记忆 RAG
        if result.memory_type in [MemoryType.CONVERSATION, MemoryType.CONTEXT]:
            return "memory"
        
        # 事件 -> 记忆 RAG
        if result.memory_type == MemoryType.EVENT:
            return "memory"
        
        # 默认 -> 记忆 RAG
        return "memory"
    
    def _calculateConfidence(self, content: str, result: TaggingResult) -> float:
        """计算置信度"""
        # 基于关键词匹配数量计算置信度
        content_lower = content.lower()
        match_count = 0
        total_keywords = 0
        
        # 领域关键词
        for keywords in self.domain_keywords.values():
            total_keywords += len(keywords)
            match_count += sum(1 for kw in keywords if kw.lower() in content_lower)
        
        # 重要性关键词
        for keywords in self.importance_keywords.values():
            total_keywords += len(keywords)
            match_count += sum(1 for kw in keywords if kw.lower() in content_lower)
        
        # 简单置信度计算
        if total_keywords > 0:
            confidence = min(1.0, match_count / total_keywords * 10)
        else:
            confidence = 0.5
        
        return round(confidence, 2)
    
    def _extractTags(self, content: str, result: TaggingResult) -> List[str]:
        """提取自定义标签"""
        tags = []
        
        # 提取领域标签
        tags.append(f"domain:{result.domain.value}")
        
        # 提取重要性标签
        tags.append(f"importance:{result.importance.value}")
        
        # 提取时间标签（如果有时问词）
        time_words = ["今天", "明天", "昨天", "现在", "刚才", "today", "tomorrow", "now"]
        if any(w in content.lower() for w in time_words):
            tags.append("time_sensitive")
        
        # 提取实体标签（简化版：提取数字）
        if re.search(r"\d+", content):
            tags.append("has_numbers")
        
        return tags


class TaggingEngine:
    """信息分类与打标引擎
    
    TSD v1.7 对应规则:
    - 2.2.5 基础设施层：信息分类打标
    - 自动分类 + 人工规则
    - 支持多标签
    
    功能:
    - 内容分析
    - 智能打标
    - 分类决策
    - 置信度评估
    """
    
    def __init__(self, use_rule_based: bool = True):
        """初始化打标引擎
        
        Args:
            use_rule_based: 是否使用规则打标器
        """
        self.use_rule_based = use_rule_based
        
        if use_rule_based:
            self.rule_tagger = RuleBasedTagger()
            logger.info("[TaggingEngine] Initialized with rule-based tagger")
        else:
            self.rule_tagger = None
            logger.info("[TaggingEngine] Initialized (no tagger)")
        
        # 统计信息
        self.total_tags = 0
        self.tag_history: List[Dict] = []
    
    def tag_content(self, content: str, context: Optional[Dict] = None) -> TaggingResult:
        """对内容进行打标
        
        Args:
            content: 文本内容
            context: 上下文信息（可选）
            
        Returns:
            TaggingResult: 打标结果
        """
        if self.rule_tagger:
            result = self.rule_tagger.tag(content, context)
        else:
            result = TaggingResult()
        
        self.total_tags += 1
        
        # 记录历史
        self.tag_history.append({
            "content": content[:100],  # 只记录前 100 字符
            "result": result.to_dict(),
            "timestamp": time.time()
        })
        
        # 限制历史记录大小
        if len(self.tag_history) > 1000:
            self.tag_history = self.tag_history[-1000:]
        
        logger.debug(f"[TaggingEngine] Tagged content: {result.target_rag}")
        
        return result
    
    def tag_and_store(self, content: str, rag_manager) -> str:
        """打标并存储到 RAG 库
        
        Args:
            content: 文本内容
            rag_manager: RAG 管理器实例
            
        Returns:
            str: 文档 ID
        """
        # 打标
        result = self.tag_content(content)
        
        # 根据目标 RAG 库存储
        if result.target_rag == "experience":
            doc_id = rag_manager.add_experience(
                content=content,
                category=result.experience_category.value if result.experience_category else "general",
                importance=result.importance.value,
                domain=result.domain.value
            )
        elif result.target_rag == "knowledge":
            doc_id = rag_manager.add_knowledge(
                content=content,
                domain=result.domain.value,
                certainty="confirmed" if result.confidence > 0.7 else "uncertain"
            )
        else:  # memory
            doc_id = rag_manager.add_memory(
                content=content,
                memory_type=result.memory_type.value,
                time_span="long_term" if result.memorability == MemorabilityLevel.MUST_REMEMBER else "short_term",
                memorability=result.memorability.value
            )
        
        logger.info(f"[TaggingEngine] Tagged and stored: {doc_id} -> {result.target_rag}")
        
        return doc_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tags": self.total_tags,
            "history_size": len(self.tag_history),
            "tagger_type": "rule_based" if self.rule_tagger else "none"
        }
    
    def print_status(self):
        """打印状态信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("信息分类与打标引擎状态")
        print("=" * 60)
        print(f"总打标数：{stats['total_tags']}")
        print(f"历史记录：{stats['history_size']} 条")
        print(f"打标器类型：{stats['tagger_type']}")
        print("=" * 60 + "\n")
