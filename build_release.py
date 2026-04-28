#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZULONG GitHub 发布构建脚本

功能:
1. 复制公开文件到 release/ 目录 (排除闭源模块/敏感数据)
2. 替换硬编码绝对路径为相对路径
3. 替换硬编码密码为占位符
4. 为闭源模块添加 try/except 导入保护
5. 生成示例配置文件
6. 最终验证: 确认无泄漏

用法:
    python build_release.py
    python build_release.py --output ../zulong_release
    python build_release.py --dry-run   # 仅检查，不复制
"""

import os
import re
import shutil
import argparse
from pathlib import Path


# ============================================================================
# 配置区
# ============================================================================

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 默认输出目录 (项目同级)
DEFAULT_OUTPUT = PROJECT_ROOT.parent / "zulong_release"

# --- 闭源模块列表 (企业版，完全不公开) ---
CLOSED_SOURCE_FILES = [
    # 记忆系统核心
    "zulong/memory/knowledge_graph.py",
    "zulong/memory/person_profile.py",
    "zulong/memory/llm_memory_reviewer.py",
    "zulong/memory/summary_store.py",
    "zulong/memory/memory_evolution.py",
    # short_term_memory.py -> 已开放给社区版
    "zulong/memory/hot_update_engine.py",
    "zulong/memory/three_libraries.py",
    "zulong/memory/enhanced_experience_store.py",
    "zulong/memory/episodic_memory.py",
    "zulong/memory/semantic_drift_detector.py",
    # L1-B 核心调度
    "zulong/l1b/hotswap_scheduler.py",
    "zulong/l1b/scheduler_gatekeeper.py",
    "zulong/l1b/dynamic_threshold_manager.py",
    "zulong/l1b/memory_config_initializer.py",
    # L3 双脑
    "zulong/l3/dual_brain_container.py",
    # L2 推理引擎
    "zulong/l2/inference_engine.py",
    "zulong/l2/inference_engine.py.backup",
    "zulong/l2/inference_engine_fixed.py",
    # L1-C 优化视觉
    "zulong/l1c/optimized_vision_processor.py",
]

# --- 闭源目录 (整个目录不公开) ---
CLOSED_SOURCE_DIRS = [
    "zulong/replay",
    "zulong/review",
]

# --- 完全排除的目录/文件 (不进入 release) ---
# 仅按目录名排除 (任何深度)
EXCLUDE_DIRS_BY_NAME = [
    "__pycache__", "node_modules", ".vscode", ".idea", ".trae", ".qoder", ".git",
]
# 按相对路径排除 (仅根目录级别)
EXCLUDE_DIRS_BY_PATH = [
    # 运行时数据
    "data", "dossiers", "experience_store", "checkpoints",
    "debug_data", "diagnostics", "stability_test_data",
    "camera_test_data", "gesture_test_data", "test_reports",
    # 备份
    "safetensorsvers", "backup_old_vision_arch", "backups",
    # 内部文档 (含核心架构细节)
    "资料",
    # docs 目录 (含大量内部实现细节，将由脚本生成精简版)
    "docs",
    # 独立子项目
    "openclaw",
    # 虚拟环境
    "zulong_env", "venv", ".venv",
    # 模型权重 (根目录级别，非 zulong/models/)
    "models",
    # config 目录 (本地配置)
    "config",
    # 测试目录 (引用闭源模块和本地硬件环境)
    "tests",
    # 截图 (调试截图)
    "screenshots",
    # 脚本目录 (含内部诊断工具)
    "scripts",
]

EXCLUDE_FILES_PATTERNS = [
    # 开发过程文档 (含实现细节)
    r"^ARCHITECTURE_.*\.md$",
    r"^ASYNC_.*\.md$",
    r"^CLEANUP_.*\.md$",
    r"^CROSS_SESSION_.*\.md$",
    r"^DEBUG_.*\.md$",
    r"^DOWNGRADE_.*\.md$",
    r"^FIX_.*\.md$",
    r"^GATEKEEPER_.*\.md$",
    r"^L2_.*\.md$",
    r"^MEMORY_.*\.md$",
    r"^PHASE_.*\.md$",
    r"^QUICK_REVIEW_.*\.md$",
    r"^RAG_.*\.md$",
    r"^REVIEW_.*\.md$",
    r"^RUNTIME_.*\.md$",
    r"^SHARED_POOL_.*\.md$",
    r"^SOFTWARE_.*\.md$",
    r"^WEB_RESPONSE_.*\.md$",
    r"^TODO_ENHANCEMENTS\.md$",
    r"^TESTING_GUIDE\.md$",
    r"^TEST_.*\.md$",
    r"^code_review_files\.txt$",
    r"^启动说明\.md$",
    # 根目录测试文件
    r"^test_.*\.(py|txt|json|html)$",
    r"^test2_output\.txt$",
    # 启动脚本 (含本地路径)
    r"^start_.*\.bat$",
    # 模型权重
    r".*\.safetensors$",
    r".*\.gguf$",
    r".*\.pt$",
    r".*\.pth$",
    r"^gesture_recognizer\.task$",
    # Python 缓存
    r".*\.pyc$",
    r".*\.pyo$",
    # 系统
    r".*\.swp$",
    r".*\.log$",
    # 本地配置
    r"^config\.yaml$",
    r"^docker-compose\.yml$",
    r"^\.env$",
    r"^\.env\..*$",
    r"^\.dockerignore$",
    # 临时文件
    r"^nul$",
    r"^LICENSE_AGPL_RAW\.txt$",
    r"^LICENSE_HEADER\.txt$",
    # 构建脚本本身
    r"^build_release\.py$",
    # 源 README (release 使用独立版本)
    r"^README\.md$",
]

# --- 硬编码路径替换规则 ---
# 格式: (文件相对路径, 原字符串, 替换字符串)
PATH_REPLACEMENTS = [
    # zulong/models/model_configs.py
    (
        "zulong/models/model_configs.py",
        'MODEL_BASE_DIR = Path(r"d:\\AI\\project\\zulong_beta4\\models")',
        'PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent\nMODEL_BASE_DIR = PROJECT_ROOT / "models"',
    ),
    # zulong/l2/l2_config.py
    (
        "zulong/l2/l2_config.py",
        'L2_CORE_MODEL_PATH = Path(r"d:\\AI\\project\\zulong_beta4\\models\\Qwen\\Qwen3___5-2B")',
        'PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent\nL2_CORE_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen" / "Qwen3___5-2B"',
    ),
    # zulong/l1b/l1b_config.py
    (
        "zulong/l1b/l1b_config.py",
        'L1B_AUDIO_MODEL_PATH = Path(r"d:\\AI\\project\\zulong_beta4\\zulong\\models\\Qwen3.5-0.8B-int4-L1B")',
        'PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent\nL1B_AUDIO_MODEL_PATH = PROJECT_ROOT / "models" / "Qwen3.5-0.8B-int4-L1B"',
    ),
    # zulong/tts/cosyvoice_config.py
    (
        "zulong/tts/cosyvoice_config.py",
        'MODEL_BASE_DIR = Path(r"d:\\AI\\project\\zulong_beta4\\models")',
        'PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent\nMODEL_BASE_DIR = PROJECT_ROOT / "models"',
    ),
    # zulong/l3/expert_config.py
    (
        "zulong/l3/expert_config.py",
        'MODEL_BASE_DIR = Path(r"d:\\AI\\project\\zulong_beta4\\models")',
        'MODEL_BASE_DIR = Path(__file__).resolve().parent.parent.parent / "models"',
    ),
    # zulong/l3/tts_expert_node.py - model_path
    (
        "zulong/l3/tts_expert_node.py",
        'self.model_path = Path(r"d:\\AI\\project\\zulong_beta4\\models\\CosyVoice3-0.5B\\FunAudioLLM\\Fun-CosyVoice3-0___5B-2512")',
        '_project_root = Path(__file__).resolve().parent.parent.parent\n        self.model_path = _project_root / "models" / "CosyVoice3-0.5B" / "FunAudioLLM" / "Fun-CosyVoice3-0___5B-2512"',
    ),
    # zulong/l3/tts_expert_node.py - ttsfrd_path
    (
        "zulong/l3/tts_expert_node.py",
        'self.ttsfrd_path = Path(r"d:\\AI\\project\\zulong_beta4\\models\\iic\\CosyVoice-ttsfrd")',
        'self.ttsfrd_path = _project_root / "models" / "iic" / "CosyVoice-ttsfrd"',
    ),
    # --- short_term_memory.py: 闭源导入降级 (社区版开放) ---
    # 1. 顶层 MemoryConsolidator 导入
    (
        "zulong/memory/short_term_memory.py",
        'from zulong.memory.memory_evolution import MemoryConsolidator',
        'try:\n    from zulong.memory.memory_evolution import MemoryConsolidator\nexcept ImportError:\n    MemoryConsolidator = None',
    ),
    # 2. MemoryConsolidator 初始化降级
    (
        "zulong/memory/short_term_memory.py",
        'self.consolidator = MemoryConsolidator(RAGManager())',
        'self.consolidator = MemoryConsolidator(RAGManager()) if MemoryConsolidator else None',
    ),
    # 3. get_stats 动态阈值降级
    (
        "zulong/memory/short_term_memory.py",
        'thresholds = self.threshold_manager.get_thresholds()',
        'thresholds = self.threshold_manager.get_thresholds() if self.threshold_manager else None',
    ),
    # 4. get_stats 阈值字段降级
    (
        "zulong/memory/short_term_memory.py",
        '            "hard_token_limit": thresholds.hard_token_limit,\n            "soft_turn_limit": thresholds.soft_turn_limit,\n            "is_emergency_mode": thresholds.is_emergency_mode,\n            "vram_usage": thresholds.vram_usage,',
        '            "hard_token_limit": thresholds.hard_token_limit if thresholds else 0,\n            "soft_turn_limit": thresholds.soft_turn_limit if thresholds else 0,\n            "is_emergency_mode": thresholds.is_emergency_mode if thresholds else False,\n            "vram_usage": thresholds.vram_usage if thresholds else 0,',
    ),
]

# --- BaiduNetdisk 路径替换 (使用正则) ---
BAIDU_PATH_REPLACEMENTS = [
    # tts_cosyvoice_server.py
    (
        "zulong/l3/tts_cosyvoice_server.py",
        [
            (r'model_dir: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'model_dir: str = ""  # TODO: Set your CosyVoice model directory'),
            (r'code_path: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'code_path: str = ""  # TODO: Set your CosyVoice code path'),
            (r'default_prompt_audio: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'default_prompt_audio: str = ""  # TODO: Set your prompt audio path'),
            (r'integrated_python = r"D:\\BaiduNetdiskDownload\\.*?"',
             'integrated_python = ""  # TODO: Set your Python executable path'),
        ],
    ),
    # tts_cosyvoice_direct.py
    (
        "zulong/l3/tts_cosyvoice_direct.py",
        [
            (r'integrated_python_path: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'integrated_python_path: str = ""  # TODO: Set your Python executable path'),
            (r'model_dir: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'model_dir: str = ""  # TODO: Set your CosyVoice model directory'),
            (r'code_path: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'code_path: str = ""  # TODO: Set your CosyVoice code path'),
            (r'default_prompt_audio: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'default_prompt_audio: str = ""  # TODO: Set your prompt audio path'),
        ],
    ),
    # tts_cosyvoice_wrapper.py
    (
        "zulong/l3/tts_cosyvoice_wrapper.py",
        [
            (r'integrated_python_path: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'integrated_python_path: str = ""  # TODO: Set your Python executable path'),
            (r'cosyvoice_code_path: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'cosyvoice_code_path: str = ""  # TODO: Set your CosyVoice code path'),
            (r'model_dir: str = r"D:\\BaiduNetdiskDownload\\.*?"',
             'model_dir: str = ""  # TODO: Set your CosyVoice model directory'),
        ],
    ),
    # tts_cosyvoice_gradio.py
    (
        "zulong/l3/tts_cosyvoice_gradio.py",
        [
            (r'prompt_audio = r"D:\\BaiduNetdiskDownload\\.*?"',
             'prompt_audio = ""  # TODO: Set your prompt audio path'),
        ],
    ),
]

# --- __init__.py 导入保护替换 ---
# 格式: (文件路径, 原导入行, 模块名, 导入的符号列表)
INIT_IMPORT_GUARDS = {
    "zulong/memory/__init__.py": {
        "replace_full": True,
        "content": '''# -*- coding: utf-8 -*-
# File: zulong/memory/__init__.py
# ZULONG Memory System - Community Edition

"""
ZULONG Memory System

Community Edition includes:
- RAG Manager (RAGManager)
- Tagging Engine (TaggingEngine)
- Experience Generator (ExperienceGenerator)
- Short-term Memory (ShortTermMemory)

Enterprise Edition adds:
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
from .short_term_memory import ShortTermMemory

__all__ = [
    "RAGManager",
    "RAGConfig",
    "TaggingEngine",
    "ExperienceGenerator",
    "ExperienceCandidate",
    "ShortTermMemory",
]

# --- Enterprise Edition (optional) ---
try:
    from .memory_evolution import MemoryConsolidator, MemoryStrength
    __all__.extend(["MemoryConsolidator", "MemoryStrength"])
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
''',
    },
    "zulong/l1b/__init__.py": {
        "replace_full": True,
        "content": '''# File: zulong/l1b/__init__.py
# L1-B Scheduler & Gatekeeper Layer - Community Edition

"""
ZULONG L1-B Layer

Community Edition includes:
- Audio Understanding Node
- Async Scheduler

Enterprise Edition adds:
- Scheduler Gatekeeper (hotswap, dynamic thresholds)
"""

# --- Community Edition ---
from .audio_understanding_node import l1b_audio_understanding, L1BAudioUnderstandingNode
from .async_scheduler import AsyncL1BScheduler, async_scheduler, get_async_scheduler

__all__ = [
    'l1b_audio_understanding',
    'L1BAudioUnderstandingNode',
    'AsyncL1BScheduler',
    'async_scheduler',
    'get_async_scheduler',
]

# --- Enterprise Edition (optional) ---
try:
    from .scheduler_gatekeeper import gatekeeper
    __all__.append('gatekeeper')
except ImportError:
    pass
''',
    },
    "zulong/l3/__init__.py": {
        "replace_full": True,
        "content": '''# File: zulong/l3/__init__.py
# L3 Expert Skill Layer - Community Edition

"""
ZULONG L3 Layer - Expert Skills

Community Edition includes:
- Base Expert Node framework
- Navigation, Manipulation, Vision expert nodes
- Model Switcher
- Expert Config, Loader, Container

Enterprise Edition adds:
- Dual Brain Container (KV Cache hot-swap)
"""

# --- Community Edition ---
from .base_expert_node import BaseExpertNode, ExpertExecutionError
from .nav_expert_node import NavExpertNode
from .manipulation_expert_node import ManipulationExpertNode
from .vision_expert_node import VisionExpertNode
from .model_switcher import ModelSwitcher
from .expert_config import (
    ExpertConfig,
    ExpertModelType,
    ExpertRole,
    ExpertQuantizationConfig,
    QuantizationPreset,
    ModelPathRegistry,
    ExpertContainerConfig,
)
from .expert_loader import (
    ExpertLoader,
    ExpertContainer,
    get_expert_container,
)
from .expert_container import (
    ExpertPoolContainer,
    ExpertInstance,
    ExpertContext,
    get_expert_pool,
)

__all__ = [
    "BaseExpertNode",
    "ExpertExecutionError",
    "NavExpertNode",
    "ManipulationExpertNode",
    "VisionExpertNode",
    "ModelSwitcher",
    "ExpertConfig",
    "ExpertModelType",
    "ExpertRole",
    "ExpertQuantizationConfig",
    "QuantizationPreset",
    "ModelPathRegistry",
    "ExpertContainerConfig",
    "ExpertLoader",
    "ExpertContainer",
    "get_expert_container",
    "ExpertPoolContainer",
    "ExpertInstance",
    "ExpertContext",
    "get_expert_pool",
]

# --- Enterprise Edition (optional) ---
try:
    from .dual_brain_container import DualBrainContainer
    __all__.append("DualBrainContainer")
except ImportError:
    pass
''',
    },
    "zulong/l1c/__init__.py": {
        "replace_full": True,
        "content": '''# File: zulong/l1c/__init__.py
# L1-C Silent Visual Attention Layer - Community Edition

"""
ZULONG L1-C Layer - Silent Visual Attention

Community Edition includes:
- Action Classifier (MobileNetV4-TSM)
- MediaPipe Gesture Recognizer

Enterprise Edition adds:
- Optimized Vision Processor (four-layer architecture)
"""

# --- Community Edition ---
from zulong.l1c.action_classifier import MobileNetV4_TSM
from zulong.l1c.mediapipe_gesture_recognizer import MediaPipeGestureRecognizer

__all__ = [
    'MobileNetV4_TSM',
    'MediaPipeGestureRecognizer',
]

# --- Enterprise Edition (optional) ---
try:
    from zulong.l1c.optimized_vision_processor import OptimizedVisionProcessor, init_vision_processor, get_vision_processor
    __all__.extend(['OptimizedVisionProcessor', 'init_vision_processor', 'get_vision_processor'])
except ImportError:
    pass
''',
    },
}


def apply_stm_community_patches(file_path: Path, rel_path: str):
    """对社区版 short_term_memory.py 应用闭源导入降级（多行替换）"""
    rel_forward = rel_path.replace("\\", "/")
    if rel_forward != "zulong/memory/short_term_memory.py":
        return

    content = file_path.read_text(encoding="utf-8")
    original = content

    # 1. __init__ 中的 dynamic_threshold_manager 导入 + 初始化
    content = re.sub(
        r'(        )from zulong\.l1b\.dynamic_threshold_manager import get_dynamic_threshold_manager\n'
        r'        self\.threshold_manager = get_dynamic_threshold_manager\(\)',
        r'\g<1>try:\n'
        r'\g<1>    from zulong.l1b.dynamic_threshold_manager import get_dynamic_threshold_manager\n'
        r'\g<1>    self.threshold_manager = get_dynamic_threshold_manager()\n'
        r'\g<1>except ImportError:\n'
        r'\g<1>    self.threshold_manager = None',
        content
    )

    # 2. threshold_manager 回调注册保护
    content = re.sub(
        r'(        )self\.threshold_manager\.register_emergency_trigger_callback\(self\._on_emergency_trigger\)',
        r'\g<1>if self.threshold_manager:\n\g<1>    self.threshold_manager.register_emergency_trigger_callback(self._on_emergency_trigger)',
        content
    )

    # 3. semantic_drift_detector 导入 + 初始化
    content = re.sub(
        r'(        )from zulong\.memory\.semantic_drift_detector import get_semantic_drift_detector\n'
        r'        self\.drift_detector = get_semantic_drift_detector\(\)',
        r'\g<1>try:\n'
        r'\g<1>    from zulong.memory.semantic_drift_detector import get_semantic_drift_detector\n'
        r'\g<1>    self.drift_detector = get_semantic_drift_detector()\n'
        r'\g<1>except ImportError:\n'
        r'\g<1>    self.drift_detector = None',
        content
    )

    # 4. backup_scheduler 导入 + 初始化
    content = re.sub(
        r'(        )from zulong\.l2\.backup_scheduler import get_l2_backup_scheduler\n'
        r'        self\.backup_scheduler = get_l2_backup_scheduler\(\)',
        r'\g<1>try:\n'
        r'\g<1>    from zulong.l2.backup_scheduler import get_l2_backup_scheduler\n'
        r'\g<1>    self.backup_scheduler = get_l2_backup_scheduler()\n'
        r'\g<1>except ImportError:\n'
        r'\g<1>    self.backup_scheduler = None',
        content
    )

    # 5. backup_scheduler 回调注册保护
    content = re.sub(
        r'(        )self\.backup_scheduler\.register_completion_callback\(self\._on_summarization_complete\)',
        r'\g<1>if self.backup_scheduler:\n\g<1>    self.backup_scheduler.register_completion_callback(self._on_summarization_complete)',
        content
    )

    # 6. _check_dynamic_thresholds 方法顶部：企业版模块不可用时跳过
    content = re.sub(
        r'(    async def _check_dynamic_thresholds\(self[^)]*\)[^:]*:.*?""")\n(        # 1\.)',
        r'\1\n        if not self.threshold_manager or not self.drift_detector or not self.backup_scheduler:\n            return False\n\n\2',
        content,
        flags=re.DOTALL
    )

    if content != original:
        file_path.write_text(content, encoding="utf-8")
        print(f"  [STM] {rel_forward}: applied community edition import guards")


# ============================================================================
# 核心逻辑
# ============================================================================

def should_exclude_dir(dir_name: str, rel_path: str) -> bool:
    """检查目录是否应被排除"""
    # 按名称排除 (任何深度)
    if dir_name in EXCLUDE_DIRS_BY_NAME:
        return True
    # 按路径排除 (仅匹配根目录级别的目录)
    rel_forward = rel_path.replace("\\", "/")
    if rel_forward in EXCLUDE_DIRS_BY_PATH:
        return True
    # 闭源目录
    for closed_dir in CLOSED_SOURCE_DIRS:
        if rel_forward == closed_dir or rel_forward.startswith(closed_dir + "/"):
            return True
    return False


def should_exclude_file(file_name: str, rel_path: str) -> bool:
    """检查文件是否应被排除"""
    # 检查闭源文件列表
    rel_forward = rel_path.replace("\\", "/")
    if rel_forward in CLOSED_SOURCE_FILES:
        return True

    # 检查文件名模式 (仅根目录文件)
    parts = rel_forward.split("/")
    if len(parts) == 1:  # 根目录文件
        for pattern in EXCLUDE_FILES_PATTERNS:
            if re.match(pattern, file_name):
                return True
    else:
        # 非根目录: 检查通用排除模式
        for pattern in [r".*\.pyc$", r".*\.pyo$", r".*\.swp$", r".*\.log$",
                        r".*\.safetensors$", r".*\.gguf$", r".*\.pt$", r".*\.pth$",
                        r".*\.backup$"]:
            if re.match(pattern, file_name):
                return True

    return False


def apply_path_replacements(file_path: Path, rel_path: str):
    """对复制后的文件应用路径替换"""
    rel_forward = rel_path.replace("\\", "/")

    # 检查精确替换
    for target_file, old_str, new_str in PATH_REPLACEMENTS:
        if rel_forward == target_file:
            content = file_path.read_text(encoding="utf-8")
            if old_str in content:
                content = content.replace(old_str, new_str)
                file_path.write_text(content, encoding="utf-8")
                print(f"  [PATH] {rel_forward}: replaced hardcoded path")

    # 检查 BaiduNetdisk 正则替换
    for target_file, replacements in BAIDU_PATH_REPLACEMENTS:
        if rel_forward == target_file:
            content = file_path.read_text(encoding="utf-8")
            changed = False
            for pattern, replacement in replacements:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    changed = True
            if changed:
                file_path.write_text(content, encoding="utf-8")
                print(f"  [BAIDU] {rel_forward}: replaced BaiduNetdisk paths")


def apply_init_guards(file_path: Path, rel_path: str):
    """替换 __init__.py 为带导入保护的版本"""
    rel_forward = rel_path.replace("\\", "/")
    if rel_forward in INIT_IMPORT_GUARDS:
        guard_info = INIT_IMPORT_GUARDS[rel_forward]
        if guard_info.get("replace_full"):
            file_path.write_text(guard_info["content"], encoding="utf-8")
            print(f"  [GUARD] {rel_forward}: replaced with import-guarded version")


def generate_example_configs(output_dir: Path):
    """生成示例配置文件"""

    # config.yaml.example
    config_example = output_dir / "config.yaml.example"
    config_example.write_text("""# ZULONG Configuration Example
# Copy this file to config.yaml and modify as needed

system:
  name: "ZULONG"
  version: "beta4"
  log_level: "INFO"

# Model paths (relative to project root)
models:
  base_dir: "./models"
  l1a_model: "Qwen3.5-0.8B-int4-L1A"
  l1b_model: "Qwen3.5-0.8B-int4-L1B"
  l2_model: "Qwen/Qwen3___5-2B"
  tts_model: "CosyVoice3-0.5B/FunAudioLLM/Fun-CosyVoice3-0___5B-2512"

# Device configuration
device:
  gpu_memory_limit: "6GB"  # RTX 3060 6GB
  tts_device: "cpu"        # TTS runs on CPU

# Memory system
memory:
  stm_capacity: 50
  ltm_backend: "chromadb"
  embedding_model: "BAAI/bge-small-zh-v1.5"

# TTS configuration
tts:
  sample_rate: 22050
  speaker: "default"

# Server configuration
server:
  host: "0.0.0.0"
  port: 8000
  websocket_port: 8001
""", encoding="utf-8")
    print(f"  [CONFIG] Created config.yaml.example")

    # docker-compose.yml.example
    docker_example = output_dir / "docker-compose.yml.example"
    docker_example.write_text("""# ZULONG Docker Compose Example
# Copy this file to docker-compose.yml and modify as needed

version: '3.8'

services:
  zulong:
    build: .
    container_name: zulong-main
    ports:
      - "8000:8000"
      - "8001:8001"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # Optional: Monitoring with Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: zulong-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-changeme}  # Set via .env file
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana_data:
""", encoding="utf-8")
    print(f"  [CONFIG] Created docker-compose.yml.example")

    # .env.example
    env_example = output_dir / ".env.example"
    env_example.write_text("""# ZULONG Environment Variables
# Copy this file to .env and set your values

# Grafana monitoring password
GRAFANA_PASSWORD=changeme

# Model directory (absolute path, if not using default ./models)
# MODEL_BASE_DIR=/path/to/your/models

# GPU device index
# CUDA_VISIBLE_DEVICES=0

# Log level (DEBUG, INFO, WARNING, ERROR)
# LOG_LEVEL=INFO
""", encoding="utf-8")
    print(f"  [CONFIG] Created .env.example")


def verify_release(output_dir: Path) -> list:
    """验证发布目录中没有泄漏"""
    issues = []

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            file_path = Path(root) / f
            rel = file_path.relative_to(output_dir)
            rel_forward = str(rel).replace("\\", "/")

            # 检查是否有闭源文件意外包含
            if rel_forward in CLOSED_SOURCE_FILES:
                issues.append(f"LEAK: Closed-source file found: {rel_forward}")

            # 检查文件内容
            if f.endswith((".py", ".yml", ".yaml", ".txt", ".md", ".bat")):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                # 检查硬编码绝对路径
                if r"d:\AI\project\zulong_beta4" in content.lower() or \
                   r"d:/ai/project/zulong_beta4" in content.lower():
                    issues.append(f"LEAK: Hardcoded project path in {rel_forward}")

                # 检查 BaiduNetdisk 路径
                if "BaiduNetdisk" in content:
                    issues.append(f"LEAK: BaiduNetdisk path in {rel_forward}")

                # 检查硬编码密码
                if "zulong123" in content:
                    issues.append(f"LEAK: Hardcoded password in {rel_forward}")

    # 检查闭源目录
    for closed_dir in CLOSED_SOURCE_DIRS:
        full_path = output_dir / closed_dir
        if full_path.exists():
            issues.append(f"LEAK: Closed-source directory exists: {closed_dir}")

    return issues


def build_release(output_dir: Path, dry_run: bool = False):
    """主构建流程"""
    print("=" * 60)
    print("ZULONG GitHub Release Builder")
    print("=" * 60)
    print(f"Source: {PROJECT_ROOT}")
    print(f"Output: {output_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'BUILD'}")
    print()

    if not dry_run:
        # 清理旧的输出目录 (保留 .git 目录)
        if output_dir.exists():
            print(f"Removing old release directory (preserving .git)...")
            for item in output_dir.iterdir():
                if item.name == ".git":
                    continue
                if item.is_dir():
                    shutil.rmtree(item, onerror=lambda f, p, e: (os.chmod(p, 0o777), f(p)))
                else:
                    item.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    excluded_count = 0
    sanitized_count = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        rel_root = root_path.relative_to(PROJECT_ROOT)
        rel_root_str = str(rel_root).replace("\\", "/")

        # 过滤目录 (就地修改 dirs 以阻止 os.walk 递归)
        dirs[:] = [
            d for d in dirs
            if not should_exclude_dir(d, f"{rel_root_str}/{d}".lstrip("./"))
        ]

        for f in files:
            src_file = root_path / f
            rel_file = src_file.relative_to(PROJECT_ROOT)
            rel_file_str = str(rel_file).replace("\\", "/")

            if should_exclude_file(f, rel_file_str):
                excluded_count += 1
                if dry_run:
                    print(f"  [EXCLUDE] {rel_file_str}")
                continue

            if dry_run:
                print(f"  [INCLUDE] {rel_file_str}")
                copied_count += 1
                continue

            # 复制文件
            dst_file = output_dir / rel_file
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            copied_count += 1

            # 应用替换
            apply_path_replacements(dst_file, rel_file_str)
            apply_init_guards(dst_file, rel_file_str)
            apply_stm_community_patches(dst_file, rel_file_str)

    if not dry_run:
        # 生成示例配置文件
        print()
        print("Generating example config files...")
        generate_example_configs(output_dir)

    # 验证
    print()
    print("=" * 60)
    print("Verification")
    print("=" * 60)

    if not dry_run:
        issues = verify_release(output_dir)
        if issues:
            print(f"FOUND {len(issues)} ISSUES:")
            for issue in issues:
                print(f"  !! {issue}")
        else:
            print("  All checks passed! No leaks detected.")
    else:
        print("  (skipped in dry-run mode)")

    # 统计
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Files copied:   {copied_count}")
    print(f"  Files excluded: {excluded_count}")
    if not dry_run:
        print(f"  Output: {output_dir}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ZULONG release for GitHub")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory (default: ../zulong_release)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show what would be done, don't copy files")
    args = parser.parse_args()

    output_path = Path(args.output).resolve()
    build_release(output_path, dry_run=args.dry_run)
