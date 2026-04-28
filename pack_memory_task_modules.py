# -*- coding: utf-8 -*-
"""
祖龙系统 - 记忆与任务执行模块打包脚本
将图记忆、动态注意力、任务编排、任务恢复、长短期记忆索引、标签、思维深度导航等模块
及其相关中间件代码打包成单一文件，用于代码审查。
"""

import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(r"d:\AI\project\zulong_beta4")

# ====== 按模块分类的文件列表 ======
MODULES = {
    "1. 图记忆板块 (Graph Memory)": [
        "zulong/memory/memory_graph.py",
        "zulong/memory/graph_adapters.py",
        "zulong/memory/knowledge_graph.py",
        "zulong/tools/memory_graph_tools.py",
    ],
    "2. 动态注意力 / 上下文 (Dynamic Attention / Context)": [
        "zulong/l2/attention_window.py",
        "zulong/core/attention_atoms.py",
        "zulong/l1b/attention_controller.py",
        "zulong/l1a/context_tracker.py",
        "zulong/tools/attention_tool.py",
    ],
    "3. 任务编排 (Task Orchestration)": [
        "zulong/l2/fc_graph.py",
        "zulong/l2/task_graph.py",
        "zulong/l2/task_state_manager.py",
        "zulong/l2/task_archive.py",
        "zulong/tools/task_tools.py",
    ],
    "4. 任务恢复 (Task Recovery)": [
        "zulong/l2/task_snapshot.py",
        "zulong/l2/interrupt_handler.py",
        "zulong/l2/interrupt_controller.py",
        "zulong/l2/snapshot_manager.py",
        "zulong/l2/environment_snapshot.py",
        "zulong/l2/recovery_notifier.py",
        "zulong/l2/re_eval_node.py",
        "zulong/l2/task_suspension.py",
        "zulong/l2/l2_snapshot_interface.py",
    ],
    "5. 长短期记忆索引 (Long/Short-term Memory Index)": [
        "zulong/memory/short_term_memory.py",
        "zulong/memory/three_libraries.py",
        "zulong/memory/rag_libraries.py",
        "zulong/memory/rag_manager.py",
        "zulong/memory/base_rag_library.py",
        "zulong/memory/enhanced_experience_store.py",
        "zulong/memory/episodic_memory.py",
        "zulong/memory/summary_store.py",
        "zulong/memory/vector_cache.py",
        "zulong/memory/embedding_manager.py",
        "zulong/memory/memory_evolution.py",
        "zulong/memory/hybrid_search_config.py",
        "zulong/memory/experience_generator.py",
    ],
    "6. 标签系统 (Tagging System)": [
        "zulong/memory/smart_tagging.py",
        "zulong/memory/tagging_engine.py",
        "zulong/memory/time_tags.py",
    ],
    "7. 中间件与支撑模块 (Middleware & Support)": [
        "zulong/memory/__init__.py",
        "zulong/l2/__init__.py",
        "zulong/memory/integration.py",
        "zulong/memory/hot_update_engine.py",
        "zulong/memory/rollback.py",
        "zulong/memory/patch_applier.py",
        "zulong/memory/semantic_drift_detector.py",
        "zulong/memory/llm_memory_reviewer.py",
        "zulong/memory/person_profile.py",
        "zulong/memory/tool_rag.py",
        "zulong/l2/circuit_breaker.py",
        "zulong/l2/rule_guardian.py",
        "zulong/l2/types.py",
        "zulong/l2/info_gap_detector.py",
        "zulong/adapters/memory_backend.py",
        "zulong/infrastructure/shared_memory_pool.py",
        "zulong/utils/memory_manager.py",
        "zulong/l1b/memory_config_initializer.py",
        "zulong/tools/experience_tool.py",
        "zulong/replay/experience_store.py",
        "zulong/replay/context_snapshot.py",
    ],
}

SEPARATOR = "=" * 80


def count_lines(content: str) -> int:
    return content.count('\n') + (1 if content and not content.endswith('\n') else 0)


def pack_modules(output_path: Path):
    """将所有模块打包到单一文件"""
    total_files = 0
    total_lines = 0
    missing_files = []
    file_stats = []

    lines = []
    lines.append(SEPARATOR)
    lines.append("  祖龙系统 - 记忆与任务执行相关模块 源代码合集")
    lines.append(f"  打包时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  项目路径: {BASE_DIR}")
    lines.append(SEPARATOR)
    lines.append("")

    for section_name, file_list in MODULES.items():
        lines.append("")
        lines.append(SEPARATOR)
        lines.append(f"  [{section_name}]")
        lines.append(SEPARATOR)
        lines.append("")

        for rel_path in file_list:
            full_path = BASE_DIR / rel_path
            lines.append("")
            lines.append("#" + "-" * 79)
            lines.append(f"# FILE: {rel_path}")

            if full_path.exists():
                try:
                    content = full_path.read_text(encoding="utf-8")
                    line_count = count_lines(content)
                    file_size = full_path.stat().st_size
                    lines.append(f"# LINES: {line_count}  |  SIZE: {file_size:,} bytes")
                    lines.append("#" + "-" * 79)
                    lines.append("")
                    lines.append(content)
                    if not content.endswith('\n'):
                        lines.append("")
                    total_files += 1
                    total_lines += line_count
                    file_stats.append((rel_path, line_count, file_size))
                except Exception as e:
                    lines.append(f"# ERROR: 读取失败 - {e}")
                    lines.append("#" + "-" * 79)
                    missing_files.append((rel_path, str(e)))
            else:
                lines.append(f"# STATUS: 文件不存在")
                lines.append("#" + "-" * 79)
                missing_files.append((rel_path, "文件不存在"))

    # 尾部统计
    lines.append("")
    lines.append("")
    lines.append(SEPARATOR)
    lines.append("  打包统计")
    lines.append(SEPARATOR)
    lines.append(f"  成功打包文件数: {total_files}")
    lines.append(f"  总代码行数:     {total_lines:,}")
    lines.append(f"  缺失文件数:     {len(missing_files)}")
    if missing_files:
        lines.append("")
        lines.append("  缺失文件列表:")
        for fp, reason in missing_files:
            lines.append(f"    - {fp}: {reason}")
    lines.append(SEPARATOR)

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return total_files, total_lines, missing_files, file_stats


def generate_directory(output_path: Path, file_stats, total_files, total_lines, missing_files):
    """生成目录索引文件"""
    lines = []
    lines.append("# 祖龙系统 - 记忆与任务执行模块 代码目录")
    lines.append("")
    lines.append(f"打包时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"合集文件: zulong_memory_task_bundle.py")
    lines.append(f"总文件数: {total_files}")
    lines.append(f"总代码行: {total_lines:,}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 按模块输出目录
    file_index = 0
    global_index = 1
    for section_name, file_list in MODULES.items():
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| # | 文件路径 | 行数 | 大小 |")
        lines.append("|---|---------|------|------|")
        for rel_path in file_list:
            # 查找对应统计
            found = False
            for stat_path, stat_lines, stat_size in file_stats:
                if stat_path == rel_path:
                    lines.append(f"| {global_index} | `{rel_path}` | {stat_lines} | {stat_size:,} B |")
                    found = True
                    break
            if not found:
                lines.append(f"| {global_index} | `{rel_path}` | - | 缺失 |")
            global_index += 1
        lines.append("")

    # 模块说明
    lines.append("---")
    lines.append("")
    lines.append("## 模块功能说明")
    lines.append("")
    lines.append("### 1. 图记忆板块 (Graph Memory)")
    lines.append("基于 NetworkX 的异构图结构，支持 9 种节点类型和 7 种边类型。实现 BFS 扩散激活、赫布学习和艾宾浩斯衰减机制。")
    lines.append("")
    lines.append("### 2. 动态注意力 / 上下文 (Dynamic Attention / Context)")
    lines.append("三模式注意力窗口（GLOBAL/FOCUS/SINGLE_CHAIN），Token 预算管理，思维深度导航（deeper/broader/jump 三方向）。")
    lines.append("")
    lines.append("### 3. 任务编排 (Task Orchestration)")
    lines.append("基于 LangGraph 的 4 节点有向图任务编排系统，具备 5 层防护链，支持复杂任务的分解与调度。")
    lines.append("")
    lines.append("### 4. 任务恢复 (Task Recovery)")
    lines.append("完整的任务快照和环境重评估机制，支持中断恢复、任务挂起与恢复通知。")
    lines.append("")
    lines.append("### 5. 长短期记忆索引 (Long/Short-term Memory Index)")
    lines.append("三库架构：SkillStore（内存常驻）、ExperienceStore（向量检索+时间衰减）、KnowledgeStore（异步 RAG 检索）。")
    lines.append("")
    lines.append("### 6. 标签系统 (Tagging System)")
    lines.append("智能打标引擎，支持语义标签、时间标签和自动标签生成。")
    lines.append("")
    lines.append("### 7. 中间件与支撑模块 (Middleware & Support)")
    lines.append("包括记忆集成、热更新引擎、回滚机制、熔断器、规则守卫、共享内存池等基础设施。")
    lines.append("")

    if missing_files:
        lines.append("---")
        lines.append("")
        lines.append("## 缺失文件")
        lines.append("")
        for fp, reason in missing_files:
            lines.append(f"- `{fp}`: {reason}")
        lines.append("")

    output_path.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == "__main__":
    bundle_path = BASE_DIR / "zulong_memory_task_bundle.py"
    directory_path = BASE_DIR / "zulong_memory_task_directory.md"

    print("正在打包祖龙系统记忆与任务执行模块...")
    total_files, total_lines, missing_files, file_stats = pack_modules(bundle_path)
    print(f"  合集文件: {bundle_path}")
    print(f"  文件数: {total_files}, 总行数: {total_lines:,}")

    print("正在生成目录文件...")
    generate_directory(directory_path, file_stats, total_files, total_lines, missing_files)
    print(f"  目录文件: {directory_path}")

    if missing_files:
        print(f"\n警告: {len(missing_files)} 个文件缺失:")
        for fp, reason in missing_files:
            print(f"  - {fp}: {reason}")

    print("\n完成!")
