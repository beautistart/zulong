# 祖龙存储层深度重构与 OpenHands 融合分析

> 生成日期: 2026-05-20
> 来源: Qoder 深度分析对话

---

## 一、模块清单 — 现状全貌

经过彻底梳理，祖龙与存储/记忆/持久化相关的代码共 **~60 个文件**，分属 5 个目录。

### 1.1 废弃文件清单（可直接删除，~4800行）

| 文件 | 行数 | 原因 |
|------|------|------|
| `zulong/core/graph.py` | 89 | 引用不存在的模块 |
| `zulong/core/state.py` | 66 | 与 state_manager.py 完全重叠 |
| `zulong/memory/rollback.py` | 505 | 无外部引用 |
| `zulong/memory/patch_applier.py` | 420 | 无外部引用 |
| `zulong/memory/time_tags.py` | 461 | 无外部引用 |
| `zulong/memory/hybrid_search_config.py` | 357 | 无外部引用 |
| `zulong/memory/integration.py` | 510 | 无外部引用 |
| `zulong/memory/smart_tagging.py` | 452 | 与 tagging_engine.py 功能重复 |
| `zulong/storage/cold_storage.py` | 434 | 无外部引用 |
| `zulong/storage/logger.py` | 320 | 无外部引用 |
| `zulong/storage/migration.py` | 445 | 无外部引用 |
| `zulong/l2/types.py` | ~80 | 与其他 types 重复 |
| `zulong/l2/intent_schema.py` | ~24 | 无外部引用 |

### 1.2 职责重叠文件（需要合并，~2000行可节省）

| 重叠区 | 文件组 |
|--------|--------|
| 经验存储 | `three_libraries.py` vs `enhanced_experience_store.py` vs `rag_libraries.py` |
| 打标 | `tagging_engine.py` vs `smart_tagging.py` |
| 状态管理 | `core/state.py` vs `core/state_manager.py` |
| FC循环 | `fc_graph.py` vs `unified_fc_runner.py` vs `ide_fc_runner.py` |
| 类型定义 | `core/types.py` vs `l2/types.py` vs `l2/task_snapshot.py` |

### 1.3 分片存储问题

| 问题 | 严重度 | 位置 |
|------|--------|------|
| 无分片大小上限 | 高 | `sharded_memory_graph.py` |
| `compact()` 空实现 | 高 | `property_store.py:392` |
| 跨分片 BFS 无法发现未加载分片节点 | 高 | `sharded_memory_graph.py:317` |
| 分片未启用 (`use_sharding: false`) | 中 | `config/zulong_config.yaml` |

---

## 二、分片大小控制方案

### 目标
- 冷加载延迟 < 200ms / 每分片
- 每分片 ≤ 150,000 节点
- 按时间切分(月) → 超限自动分裂为子分片

### 分片索引格式
```json
{
  "shards": {
    "2026_05": {
      "parts": ["2026_05_part_0", "2026_05_part_1"],
      "total_nodes": 280000
    }
  },
  "max_nodes_per_shard": 150000,
  "max_cold_load_ms": 200
}
```

### 自动分裂机制
- 预警比例: 95% → 日志警告
- 强制分裂: 110% → 异步分裂为两个等大小子分片
- 冷加载超预算(>200ms) → 后台触发分裂

---

## 三、OpenHands 融合分析

### OpenHands 核心架构
- 三层分离: Frontend(React) → App Server(FastAPI) → Agent Server(Sandbox内)
- 一套代码多平台，通过 Docker 统一
- 前端自主开发(React+Vite)，非VS Code扩展
- 通过 iframe 嵌入 code-server 作为工具

### 吸收清单

| 吸收点 | 优先级 | 复杂度 |
|--------|--------|--------|
| Event 持久化存储 | P0 | 低 |
| Skills 模板系统 | P0 | 低 |
| AgentState 状态机 | P1 | 中 |
| Thin Proxy 模式 | P1 | 中 |
| Pause/Resume 机制 | P1 | 中 |
| Sandbox 隔离执行 | P2 | 高 |

### 交互体验吸收
- **ActionEvent 模型**: thought + action + tool_name + security_risk + summary
- **AgentState 状态机**: running/awaiting_user_input/awaiting_user_confirmation/paused/stopped/error
- **Confirmation 机制**: 安全风险等级 + 确认/拒绝按钮 + 键盘快捷键
- **Observation 替换 Action**: UI 事件流中 observation 到达时替换对应 action

---

## 四、详细开发计划 (18步, 5阶段)

### 阶段0: 准备
- 0.1: 创建分支 + 全量回归测试基线

### 阶段1: 删除废弃代码 (~4800行)
- 1.1: 删除13个零引用文件
- 1.2: 删除废弃修复文件
- 1.3: 提交

### 阶段2: 合并去重 (~2000行)
- 2.1: vector_cache → short_term_memory
- 2.2: fc_graph 节点工厂 → fc_nodes.py
- 2.3: three_libraries → enhanced_experience_store
- 2.4: hot_update_engine 清理

### 阶段3: 启用并完善分片
- 3.1: 配置启用分片
- 3.2: 实现分片大小控制
- 3.3: 实现 LMDB compact
- 3.4: 修复跨分片 BFS
- 3.5: 数据迁移

### 阶段4: Event 持久化
- 4.1: 创建 EventStore 模块
- 4.2: 集成到 EventBus
- 4.3: 事件归档定时任务

### 阶段5: 收尾
- 5.1: 全量回归测试 + 性能基准
- 5.2: 代码审查 + 提交

---

## 五、清理后的最终模块清单

```
zulong/memory/ (17个核心文件):
├── memory_graph.py              # 统一异构图
├── short_term_memory.py         # 短期记忆(合并vector_cache)
├── episodic_memory.py           # 临时记忆
├── knowledge_graph.py           # 知识图谱
├── person_profile.py            # 人物画像
├── llm_memory_reviewer.py       # 记忆审查
├── summary_store.py             # 摘要库
├── enhanced_experience_store.py # 经验库(合并three_libraries)
├── rag_manager.py               # RAG管理器
├── rag_libraries.py             # RAG库
├── base_rag_library.py          # RAG基类
├── tool_rag.py                  # 工具RAG
├── memory_evolution.py          # 记忆进化
├── experience_generator.py      # 经验生成
├── tagging_engine.py            # 打标引擎
├── graph_adapters.py            # 适配器层
├── task_search_index.py         # 任务搜索
├── code_anchor.py               # 代码锚点
├── embedding_manager.py         # Embedding管理
├── semantic_drift_detector.py   # 语义漂移
├── memory_graph_factory.py      # 工厂+分片配置
│
└── storage_hybrid/ (4个文件):
    ├── sharded_memory_graph.py  # 分片图+大小控制
    ├── memory_graph_hybrid.py   # 混合图
    ├── property_store.py        # LMDB属性(含compact)
    └── topology_index.py        # igraph拓扑

zulong/storage/ (1个文件):
└── hot_storage.py               # MongoDB热存储

zulong/events/ (新增):
└── event_store.py               # Event持久化

总结: 60文件→42文件, 35000行→26000行 (-26%)
```
