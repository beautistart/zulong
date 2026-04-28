# 祖龙系统 - 记忆与任务执行模块 代码目录

打包时间: 2026-04-27 18:37:17
合集文件: zulong_memory_task_bundle.py
总文件数: 60
总代码行: 30,961

---

## 1. 图记忆板块 (Graph Memory)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 1 | `zulong/memory/memory_graph.py` | 2843 | 115,992 B |
| 2 | `zulong/memory/graph_adapters.py` | 1319 | 51,685 B |
| 3 | `zulong/memory/knowledge_graph.py` | 715 | 27,458 B |
| 4 | `zulong/tools/memory_graph_tools.py` | 540 | 19,665 B |

## 2. 动态注意力 / 上下文 (Dynamic Attention / Context)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 5 | `zulong/l2/attention_window.py` | 582 | 20,538 B |
| 6 | `zulong/core/attention_atoms.py` | 199 | 6,733 B |
| 7 | `zulong/l1b/attention_controller.py` | 425 | 15,928 B |
| 8 | `zulong/l1a/context_tracker.py` | 106 | 3,740 B |
| 9 | `zulong/tools/attention_tool.py` | 201 | 8,356 B |

## 3. 任务编排 (Task Orchestration)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 10 | `zulong/l2/fc_graph.py` | 1234 | 56,905 B |
| 11 | `zulong/l2/task_graph.py` | 774 | 30,294 B |
| 12 | `zulong/l2/task_state_manager.py` | 235 | 9,290 B |
| 13 | `zulong/l2/task_archive.py` | 242 | 9,567 B |
| 14 | `zulong/tools/task_tools.py` | 1104 | 44,582 B |

## 4. 任务恢复 (Task Recovery)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 15 | `zulong/l2/task_snapshot.py` | 260 | 10,587 B |
| 16 | `zulong/l2/interrupt_handler.py` | 422 | 15,288 B |
| 17 | `zulong/l2/interrupt_controller.py` | 212 | 8,158 B |
| 18 | `zulong/l2/snapshot_manager.py` | 470 | 18,350 B |
| 19 | `zulong/l2/environment_snapshot.py` | 349 | 11,785 B |
| 20 | `zulong/l2/recovery_notifier.py` | 176 | 6,788 B |
| 21 | `zulong/l2/re_eval_node.py` | 316 | 12,286 B |
| 22 | `zulong/l2/task_suspension.py` | 356 | 15,279 B |
| 23 | `zulong/l2/l2_snapshot_interface.py` | 308 | 11,621 B |

## 5. 长短期记忆索引 (Long/Short-term Memory Index)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 24 | `zulong/memory/short_term_memory.py` | 1290 | 53,926 B |
| 25 | `zulong/memory/three_libraries.py` | 878 | 28,115 B |
| 26 | `zulong/memory/rag_libraries.py` | 759 | 28,311 B |
| 27 | `zulong/memory/rag_manager.py` | 333 | 11,974 B |
| 28 | `zulong/memory/base_rag_library.py` | 622 | 21,672 B |
| 29 | `zulong/memory/enhanced_experience_store.py` | 883 | 33,476 B |
| 30 | `zulong/memory/episodic_memory.py` | 725 | 29,920 B |
| 31 | `zulong/memory/summary_store.py` | 719 | 27,616 B |
| 32 | `zulong/memory/vector_cache.py` | 553 | 20,089 B |
| 33 | `zulong/memory/embedding_manager.py` | 339 | 11,733 B |
| 34 | `zulong/memory/memory_evolution.py` | 588 | 20,747 B |
| 35 | `zulong/memory/hybrid_search_config.py` | 357 | 11,350 B |
| 36 | `zulong/memory/experience_generator.py` | 418 | 14,391 B |

## 6. 标签系统 (Tagging System)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 37 | `zulong/memory/smart_tagging.py` | 452 | 16,920 B |
| 38 | `zulong/memory/tagging_engine.py` | 542 | 19,568 B |
| 39 | `zulong/memory/time_tags.py` | 461 | 14,686 B |

## 7. 中间件与支撑模块 (Middleware & Support)

| # | 文件路径 | 行数 | 大小 |
|---|---------|------|------|
| 40 | `zulong/memory/__init__.py` | 52 | 1,696 B |
| 41 | `zulong/l2/__init__.py` | 16 | 434 B |
| 42 | `zulong/memory/integration.py` | 510 | 17,106 B |
| 43 | `zulong/memory/hot_update_engine.py` | 441 | 14,933 B |
| 44 | `zulong/memory/rollback.py` | 505 | 16,095 B |
| 45 | `zulong/memory/patch_applier.py` | 420 | 13,191 B |
| 46 | `zulong/memory/semantic_drift_detector.py` | 244 | 8,743 B |
| 47 | `zulong/memory/llm_memory_reviewer.py` | 509 | 19,677 B |
| 48 | `zulong/memory/person_profile.py` | 649 | 25,052 B |
| 49 | `zulong/memory/tool_rag.py` | 233 | 8,910 B |
| 50 | `zulong/l2/circuit_breaker.py` | 469 | 20,743 B |
| 51 | `zulong/l2/rule_guardian.py` | 216 | 8,663 B |
| 52 | `zulong/l2/types.py` | 79 | 2,806 B |
| 53 | `zulong/l2/info_gap_detector.py` | 230 | 10,089 B |
| 54 | `zulong/adapters/memory_backend.py` | 347 | 11,437 B |
| 55 | `zulong/infrastructure/shared_memory_pool.py` | 1173 | 48,057 B |
| 56 | `zulong/utils/memory_manager.py` | 476 | 14,654 B |
| 57 | `zulong/l1b/memory_config_initializer.py` | 212 | 7,203 B |
| 58 | `zulong/tools/experience_tool.py` | 135 | 4,806 B |
| 59 | `zulong/replay/experience_store.py` | 375 | 12,401 B |
| 60 | `zulong/replay/context_snapshot.py` | 363 | 12,561 B |

---

## 模块功能说明

### 1. 图记忆板块 (Graph Memory)
基于 NetworkX 的异构图结构，支持 9 种节点类型和 7 种边类型。实现 BFS 扩散激活、赫布学习和艾宾浩斯衰减机制。

### 2. 动态注意力 / 上下文 (Dynamic Attention / Context)
三模式注意力窗口（GLOBAL/FOCUS/SINGLE_CHAIN），Token 预算管理，思维深度导航（deeper/broader/jump 三方向）。

### 3. 任务编排 (Task Orchestration)
基于 LangGraph 的 4 节点有向图任务编排系统，具备 5 层防护链，支持复杂任务的分解与调度。

### 4. 任务恢复 (Task Recovery)
完整的任务快照和环境重评估机制，支持中断恢复、任务挂起与恢复通知。

### 5. 长短期记忆索引 (Long/Short-term Memory Index)
三库架构：SkillStore（内存常驻）、ExperienceStore（向量检索+时间衰减）、KnowledgeStore（异步 RAG 检索）。

### 6. 标签系统 (Tagging System)
智能打标引擎，支持语义标签、时间标签和自动标签生成。

### 7. 中间件与支撑模块 (Middleware & Support)
包括记忆集成、热更新引擎、回滚机制、熔断器、规则守卫、共享内存池等基础设施。
