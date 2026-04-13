# File: zulong/l1b/__init__.py
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
