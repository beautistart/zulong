# File: zulong/l3/__init__.py
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
