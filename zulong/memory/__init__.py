# -*- coding: utf-8 -*-
# File: zulong/memory/__init__.py
# 记忆系统模块初始化

"""
ZULONG 记忆系统模块

包含:
- 短期记忆 (ShortTermMemory)
- 长期记忆 (LongTermMemory via RAG)
- 经验生成器 (ExperienceGenerator)
- 记忆进化机制 (MemoryEvolution)
- RAG 管理系统 (RAGManager)
- 打标引擎 (TaggingEngine)
- 知识图谱 (KnowledgeGraph) ⭐ NEW v2.5
- 人物画像 (PersonProfileManager) ⭐ NEW v2.5
- LLM 记忆审查 (LLMMemoryReviewer) ⭐ NEW v2.5
- 双索引摘要库 (DualIndexSummaryStore) ⭐ NEW v2.5
- 记忆图谱 (MemoryGraph) ⭐ NEW v3.0
"""

from .short_term_memory import ShortTermMemory
from .memory_evolution import MemoryConsolidator, MemoryStrength, MemoryEvolutionEngine
from .experience_generator import ExperienceGenerator, ExperienceCandidate
from .rag_manager import RAGManager, RAGConfig
from .tagging_engine import TaggingEngine
from .knowledge_graph import KnowledgeGraph, get_knowledge_graph
from .person_profile import PersonProfileManager, get_person_profile_manager
from .llm_memory_reviewer import LLMMemoryReviewer, get_llm_memory_reviewer
from .summary_store import DualIndexSummaryStore, get_dual_index_summary_store
from .memory_graph import MemoryGraph, get_memory_graph
from .task_search_index import HistoricalTaskIndex, get_task_search_index

__all__ = [
    "ShortTermMemory",
    "MemoryConsolidator",
    "MemoryStrength",
    "MemoryEvolutionEngine",
    "ExperienceGenerator",
    "ExperienceCandidate",
    "RAGManager",
    "RAGConfig",
    "TaggingEngine",
    "KnowledgeGraph",
    "get_knowledge_graph",
    "PersonProfileManager",
    "get_person_profile_manager",
    "LLMMemoryReviewer",
    "get_llm_memory_reviewer",
    "DualIndexSummaryStore",
    "get_dual_index_summary_store",
    "MemoryGraph",
    "get_memory_graph",
    "HistoricalTaskIndex",
    "get_task_search_index",
]
