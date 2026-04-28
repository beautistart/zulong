# File: zulong/skill_packs/packs/__init__.py
"""
具体技能包实现目录

每个技能包是一个子目录，包含：
- __init__.py
- planner.py / reasoner.py / coder.py (核心算法)
- tools.py (注册的工具类)
- manifest.yaml (可选，替代代码中的 manifest)

已实现的技能包：
- cline_coder: Cline编程

已卸载的技能包：
- complex_task: [已移除] 旧版复杂任务处理（融合 OpenClaw + AutoGPT 逻辑）
"""
