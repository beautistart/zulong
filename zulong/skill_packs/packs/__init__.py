# File: zulong/skill_packs/packs/__init__.py
"""
具体技能包实现目录

每个技能包是一个子目录，包含：
- __init__.py
- planner.py / reasoner.py / coder.py (核心算法)
- tools.py (注册的工具类)
- manifest.yaml (可选，替代代码中的 manifest)

已规划的技能包：
- autogpt_planner: AutoGPT任务拆解
- openmanus_reasoner: OpenManus深度推理
- cline_coder: Cline编程
"""
