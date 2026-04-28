# File: zulong/skill_packs/__init__.py
"""
技能包模块

提供"借用→学习→内化→丢弃"的技能包生命周期管理。

使用方式：
    from zulong.skill_packs import SkillPackRuntime, ISkillPack, SkillPackManifest

    runtime = SkillPackRuntime(tool_engine, experience_store, hot_update_engine)
    runtime.install_pack(MySkillPack())
    runtime.execute_capability("my_pack", "my_capability", {"param": "value"})
    runtime.check_internalization("my_pack")
    runtime.uninstall_pack("my_pack")
"""

from zulong.skill_packs.interface import ISkillPack, SkillPackManifest, SkillPackStatus
from zulong.skill_packs.runtime import SkillPackRuntime
from zulong.skill_packs.loader import SkillPackLoader
from zulong.skill_packs.module_router import quick_class, classify_with_timing

__all__ = [
    "ISkillPack",
    "SkillPackManifest",
    "SkillPackStatus",
    "SkillPackRuntime",
    "SkillPackLoader",
    "quick_class",
    "classify_with_timing",
]
