# -*- coding: utf-8 -*-
# File: zulong/memory/experience_generator.py
# 经验自动生成器 - 从对话历史中提取经验并保存到经验库

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class ExperienceCandidate:
    """经验候选"""
    content: str  # 经验内容
    category: str  # 分类（成功/失败/偏好/知识）
    confidence: float  # 置信度（0-1）
    source: str  # 来源（用户反馈/对话模式/错误日志）
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ExperienceGenerator:
    """经验自动生成器
    
    功能:
    1. 从对话历史中提取成功模式
    2. 从错误日志中提取失败教训
    3. 从用户反馈中提取偏好
    4. 自动分类并添加到经验库
    """
    
    def __init__(self, rag_manager=None):
        """初始化经验生成器
        
        Args:
            rag_manager: RAG 管理器（用于添加到经验库）
        """
        self.rag_manager = rag_manager
        
        # 配置
        self.min_confidence = 0.6  # 最小置信度（低于此值不生成经验）
        self.max_experience_length = 500  # 经验最大长度
        
        # 统计信息
        self.total_extracted = 0
        self.total_added = 0
        self.total_skipped = 0
        
        # 模式库
        self.success_patterns = [
            r"成功.*",
            r"完成.*",
            r"解决了.*",
            r"正确.*",
            r"太好了.*",
            r"谢谢.*",
            r"非常好.*",
            r"完美.*",
        ]
        
        self.failure_patterns = [
            r"错误.*",
            r"失败.*",
            r"不正确.*",
            r"有问题.*",
            r"不行.*",
            r"错误：.*",
            r"Exception.*",
            r"Failed.*",
        ]
        
        self.preference_patterns = [
            r"我喜欢.*",
            r"偏好.*",
            r"希望.*",
            r"想要.*",
            r"最好.*",
            r"不喜欢.*",
            r"避免.*",
        ]
        
        logger.info("[ExperienceGenerator] Initialized")
    
    def set_rag_manager(self, rag_manager):
        """设置 RAG 管理器
        
        Args:
            rag_manager: RAG 管理器
        """
        self.rag_manager = rag_manager
        logger.info("[ExperienceGenerator] RAG manager set")
    
    def extract_from_dialogue(self, dialogue_history: List[Dict[str, str]]) -> List[ExperienceCandidate]:
        """从对话历史中提取经验
        
        Args:
            dialogue_history: 对话历史列表
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            List[ExperienceCandidate]: 经验候选列表
        """
        logger.info(f"[ExperienceGenerator] Extracting from {len(dialogue_history)} dialogue turns")
        
        candidates = []
        
        # 分析对话模式
        for i, turn in enumerate(dialogue_history):
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if not content:
                continue
            
            # 检查用户反馈
            if role == "user":
                candidate = self._analyze_user_feedback(content, dialogue_history, i)
                if candidate:
                    candidates.append(candidate)
            
            # 检查 AI 回复模式
            elif role == "assistant":
                candidate = self._analyze_ai_response(content, dialogue_history, i)
                if candidate:
                    candidates.append(candidate)
        
        self.total_extracted += len(candidates)
        logger.info(f"[ExperienceGenerator] Extracted {len(candidates)} candidates")
        
        return candidates
    
    def _analyze_user_feedback(self, user_content: str, dialogue_history: List[Dict], turn_index: int) -> Optional[ExperienceCandidate]:
        """分析用户反馈
        
        Args:
            user_content: 用户输入内容
            dialogue_history: 对话历史
            turn_index: 当前轮次索引
            
        Returns:
            Optional[ExperienceCandidate]: 经验候选
        """
        # 检查是否是反馈（通常在 AI 回复之后）
        if turn_index == 0:
            return None
        
        prev_turn = dialogue_history[turn_index - 1]
        if prev_turn.get("role") != "assistant":
            return None
        
        ai_response = prev_turn.get("content", "")
        
        # 1. 检查正面反馈
        for pattern in self.success_patterns:
            if re.search(pattern, user_content):
                # 提取成功经验
                experience = self._extract_success_experience(user_content, ai_response)
                if experience:
                    return experience
        
        # 2. 检查负面反馈
        for pattern in self.failure_patterns:
            if re.search(pattern, user_content):
                # 提取失败教训
                experience = self._extract_failure_experience(user_content, ai_response)
                if experience:
                    return experience
        
        # 3. 检查偏好
        for pattern in self.preference_patterns:
            if re.search(pattern, user_content):
                # 提取偏好
                experience = self._extract_preference(user_content)
                if experience:
                    return experience
        
        return None
    
    def _analyze_ai_response(self, ai_content: str, dialogue_history: List[Dict], turn_index: int) -> Optional[ExperienceCandidate]:
        """分析 AI 回复模式
        
        检测 AI 回复中的成功模式（如解决问题、提供有用信息）
        
        Args:
            ai_content: AI 回复内容
            dialogue_history: 对话历史
            turn_index: 当前轮次索引
            
        Returns:
            Optional[ExperienceCandidate]: 经验候选
        """
        # 检查是否包含解决方案
        solution_indicators = [
            "解决方案是",
            "方法是",
            "步骤如下",
            "代码如下",
            "答案是",
        ]
        
        for indicator in solution_indicators:
            if indicator in ai_content:
                # 可能是一个成功的解决方案
                # 检查用户下一轮是否满意
                if turn_index < len(dialogue_history) - 1:
                    next_turn = dialogue_history[turn_index + 1]
                    if next_turn.get("role") == "user":
                        next_content = next_turn.get("content", "")
                        
                        # 检查用户是否满意
                        for pattern in self.success_patterns:
                            if re.search(pattern, next_content):
                                # 用户满意，提取经验
                                return ExperienceCandidate(
                                    content=f"当用户询问问题时，提供详细的解决方案：{ai_content[:200]}",
                                    category="成功模式",
                                    confidence=0.7,
                                    source="对话模式",
                                    metadata={
                                        "ai_response_length": len(ai_content),
                                        "indicator": indicator
                                    }
                                )
        
        return None
    
    def _extract_success_experience(self, user_feedback: str, ai_response: str) -> Optional[ExperienceCandidate]:
        """提取成功经验
        
        Args:
            user_feedback: 用户正面反馈
            ai_response: AI 之前的回复
            
        Returns:
            Optional[ExperienceCandidate]: 成功经验
        """
        # 简化：直接使用 AI 回复作为经验内容
        content = f"有效的回复模式：{ai_response[:self.max_experience_length]}"
        
        # 计算置信度
        confidence = 0.8  # 基础置信度
        
        # 如果反馈包含强烈正面词汇，提高置信度
        strong_positive = ["太好了", "完美", "非常好", "特别感谢"]
        for word in strong_positive:
            if word in user_feedback:
                confidence += 0.1
                break
        
        confidence = min(confidence, 1.0)
        
        if confidence < self.min_confidence:
            return None
        
        return ExperienceCandidate(
            content=content,
            category="成功模式",
            confidence=confidence,
            source="用户反馈",
            metadata={
                "user_feedback": user_feedback,
                "feedback_sentiment": "positive"
            }
        )
    
    def _extract_failure_experience(self, user_feedback: str, ai_response: str) -> Optional[ExperienceCandidate]:
        """提取失败教训
        
        Args:
            user_feedback: 用户负面反馈
            ai_response: AI 之前的回复
            
        Returns:
            Optional[ExperienceCandidate]: 失败教训
        """
        content = f"需要避免的回复模式：{ai_response[:self.max_experience_length]}"
        
        confidence = 0.75
        
        return ExperienceCandidate(
            content=content,
            category="失败教训",
            confidence=confidence,
            source="用户反馈",
            metadata={
                "user_feedback": user_feedback,
                "feedback_sentiment": "negative"
            }
        )
    
    def _extract_preference(self, user_content: str) -> Optional[ExperienceCandidate]:
        """提取用户偏好
        
        Args:
            user_content: 用户输入内容
            
        Returns:
            Optional[ExperienceCandidate]: 偏好经验
        """
        content = f"用户偏好：{user_content[:self.max_experience_length]}"
        
        return ExperienceCandidate(
            content=content,
            category="用户偏好",
            confidence=0.85,  # 偏好通常很明确
            source="用户直接表达",
            metadata={
                "preference_type": "explicit"
            }
        )
    
    def add_experience_to_rag(self, candidate: ExperienceCandidate) -> Optional[str]:
        """将经验添加到 RAG 经验库
        
        Args:
            candidate: 经验候选
            
        Returns:
            Optional[str]: 文档 ID，如果添加失败返回 None
        """
        if not self.rag_manager:
            logger.warning("[ExperienceGenerator] ⚠️ RAG manager not set - 经验无法保存（但不影响系统运行）")
            self.total_skipped += 1
            return None
        
        # 检查置信度
        if candidate.confidence < self.min_confidence:
            logger.debug(f"[ExperienceGenerator] Skipped (low confidence={candidate.confidence})")
            self.total_skipped += 1
            return None
        
        try:
            # 确定重要性
            importance = "useful"
            if candidate.category == "失败教训":
                importance = "critical"  # 失败教训最重要
            elif candidate.confidence > 0.9:
                importance = "important"
            
            # 添加到经验库（RAGManager.add_experience 不支持 metadata）
            doc_id = self.rag_manager.add_experience(
                content=candidate.content,
                category=candidate.category,
                importance=importance,
                domain="general"  # 默认领域
            )
            
            self.total_added += 1
            logger.info(f"[ExperienceGenerator] Added experience: {doc_id[:8]}... (category={candidate.category}, confidence={candidate.confidence:.2f})")
            
            return doc_id
        
        except Exception as e:
            logger.error(f"[ExperienceGenerator] Failed to add experience: {e}", exc_info=True)
            return None
    
    def process_dialogue_batch(self, dialogue_history: List[Dict[str, str]]) -> Dict[str, int]:
        """批量处理对话历史
        
        Args:
            dialogue_history: 对话历史列表
            
        Returns:
            Dict[str, int]: 处理统计
        """
        logger.info(f"[ExperienceGenerator] Processing batch of {len(dialogue_history)} turns")
        
        # 提取经验候选
        candidates = self.extract_from_dialogue(dialogue_history)
        
        # 添加到 RAG
        added_count = 0
        skipped_count = 0
        
        for candidate in candidates:
            doc_id = self.add_experience_to_rag(candidate)
            if doc_id:
                added_count += 1
            else:
                skipped_count += 1
        
        stats = {
            "extracted": len(candidates),
            "added": added_count,
            "skipped": skipped_count,
            "total_extracted_all_time": self.total_extracted,
            "total_added_all_time": self.total_added,
        }
        
        logger.info(f"[ExperienceGenerator] Batch complete: {stats}")
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            "total_extracted": self.total_extracted,
            "total_added": self.total_added,
            "total_skipped": self.total_skipped,
            "extraction_rate": self.total_added / max(self.total_extracted, 1),
        }
