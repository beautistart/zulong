# -*- coding: utf-8 -*-
# File: zulong/memory/__init__.py
# ZULONG Memory System - Community Edition

"""
ZULONG Memory System

Community Edition includes:
- RAG Manager (RAGManager)
- Tagging Engine (TaggingEngine)
- Experience Generator (ExperienceGenerator)

Enterprise Edition adds:
- Short-term Memory (ShortTermMemory)
- Memory Evolution (MemoryConsolidator)
- Knowledge Graph (KnowledgeGraph)
- Person Profile (PersonProfileManager)
- LLM Memory Reviewer (LLMMemoryReviewer)
- Dual-Index Summary Store (DualIndexSummaryStore)

For enterprise licensing, see COMMERCIAL_LICENSE.md
"""

# --- Community Edition (always available) ---
from .rag_manager import RAGManager, RAGConfig
from .tagging_engine import TaggingEngine
from .experience_generator import ExperienceGenerator, ExperienceCandidate

__all__ = [
    "RAGManager",
    "RAGConfig",
    "TaggingEngine",
    "ExperienceGenerator",
    "ExperienceCandidate",
]

# --- Enterprise Edition (optional) ---
try:
    from .short_term_memory import ShortTermMemory
    from .memory_evolution import MemoryConsolidator, MemoryStrength
    __all__.extend(["ShortTermMemory", "MemoryConsolidator", "MemoryStrength"])
except ImportError:
    pass

try:
    from .knowledge_graph import KnowledgeGraph, get_knowledge_graph
    __all__.extend(["KnowledgeGraph", "get_knowledge_graph"])
except ImportError:
    pass

try:
    from .person_profile import PersonProfileManager, get_person_profile_manager
    __all__.extend(["PersonProfileManager", "get_person_profile_manager"])
except ImportError:
    pass

try:
    from .llm_memory_reviewer import LLMMemoryReviewer, get_llm_memory_reviewer
    __all__.extend(["LLMMemoryReviewer", "get_llm_memory_reviewer"])
except ImportError:
    pass

try:
    from .summary_store import DualIndexSummaryStore, get_dual_index_summary_store
    __all__.extend(["DualIndexSummaryStore", "get_dual_index_summary_store"])
except ImportError:
    pass
