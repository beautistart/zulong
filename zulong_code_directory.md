# 祖龙(ZULONG) 图记忆系统及协调板块 - 代码目录索引

> 生成时间: 2026-04-23
> 合并源码文件: `zulong_graph_memory_full_source.py` (1204 KB, 30926 行)
> 文件总数: 49 个

---

## 一、板块总览

| 板块 | 文件数 | 总大小 | 核心职责 |
|------|--------|--------|----------|
| 核心层 (Core) | 4 | 27 KB | 事件类型/状态管理/事件总线/注意力原子类 |
| 基础设施 (Infra) | 1 | 46 KB | 共享记忆池(跨模块数据交换中心) |
| 记忆系统-核心 | 7 | 239 KB | MemoryGraph统一记忆中枢+适配器+向量+剪枝 |
| 记忆系统-标签 | 3 | 50 KB | 智能标签(重要度/温度/时间段) |
| 记忆系统-RAG | 6 | 101 KB | 向量+BM25混合检索/4个RAG库 |
| 记忆系统-旧组件 | 3 | 109 KB | 已废弃(短期/临时/三库),被MG替代 |
| 记忆系统-知识 | 2 | 51 KB | 知识图谱/人物档案 |
| 记忆系统-经验 | 2 | 47 KB | 经验存储/自动生成 |
| 记忆系统-维护 | 5 | 70 KB | 集成/热更新/补丁/回滚/漂移检测 |
| L1-B层 | 2 | 130 KB | 调度守门员/注意力控制器 |
| L2-任务系统 | 6 | 89 KB | TaskGraph/挂起恢复/快照/状态管理 |
| L2-中断系统 | 2 | 23 KB | 生成中断/冻结/恢复 |
| L2-恢复系统 | 2 | 19 KB | 环境变化检测/启动恢复通知 |
| L2-推理核心 | 1 | 126 KB | FC循环/记忆检索/prompt构建 |
| FC工具 | 3 | 52 KB | LLM通过FC操作任务/注意力/记忆 |

**总计: 49 个文件, 1176 KB, 30615 行**

---

## 二、系统架构层级图

```
+---------------------------------------------------------------+
|                    FC工具层 (LLM自主调用)                       |
|  task_tools / attention_tool / memory_graph_tools              |
+---------------------------------------------------------------+
|                    L2 推理核心                                  |
|  inference_engine (FC循环/prompt构建/记忆检索)                  |
|  +-- task_graph (任务图谱)                                     |
|  +-- task_suspension / task_state_manager (挂起/恢复)          |
|  +-- interrupt_handler / interrupt_controller (中断)           |
|  +-- snapshot_manager / environment_snapshot (快照)            |
+---------------------------------------------------------------+
|              L1-B 调度门控层                                    |
|  scheduler_gatekeeper (事件过滤/打包)                          |
|  attention_controller (优先级路由/中断管理)                     |
+---------------------------------------------------------------+
|              记忆系统 (MemoryGraph 统一中枢)                    |
|  memory_graph.py (异构图/BFS/赫布/衰减)                        |
|  +-- graph_adapters (6适配器:对话/任务/知识/情景/人物/经验)     |
|  +-- embedding_manager + summary_store (向量/FAISS)            |
|  +-- smart_tagging + memory_evolution (标签/巩固/遗忘)         |
|  +-- llm_memory_reviewer (LLM剪枝守卫)                        |
|  +-- rag_manager (4个RAG库)                                   |
|  +-- knowledge_graph + person_profile (知识/人物)              |
+---------------------------------------------------------------+
|              基础设施层                                         |
|  event_bus / shared_memory_pool / state_manager               |
+---------------------------------------------------------------+
```

---

## 三、详细文件列表

> 带 ★ 的是各板块核心文件，审查时优先关注

### 3.1 核心层 (Core)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `types.py` | 6 KB | 159 | 事件类型(EventType)/优先级(EventPriority)/ZulongEvent/L2Status等枚举 |
| `state_manager.py` | 5 KB | 169 | 全局状态管理器(L2状态/中断标志/上下文读写,带锁保护) |
| `event_bus.py` | 9 KB | 220 | 事件总线(发布/订阅/路由,FIFO队列) |
| `attention_atoms.py` | 7 KB | 199 | AttentionEvent/ContextSnapshot/MacroCommand/SensorFusionData |

### 3.2 基础设施 (Infrastructure)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `shared_memory_pool.py` | 46 KB | 1159 | 共享记忆池(4分区/DataEnvelope/gzip持久化/任务队列) |

### 3.3 记忆系统-核心 (Memory Core)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `__init__.py` | 2 KB | 52 | 记忆模块初始化和导出 |
| ★`memory_graph.py` | 110 KB | 2784 | **MemoryGraph核心**: 异构图(NetworkX DiGraph)/BFS扩散激活/赫布学习/艾宾浩斯衰减剪枝/双路径检索(热遍历+冷FAISS)/SummarySidecarIndex/地址解析 |
| ★`graph_adapters.py` | 50 KB | 1298 | **6个图适配器**: DialogueAdapter(对话生命周期/话题检测/session分配/地址继承)/TaskGraphAdapter(任务投射/增量同步)/KnowledgeGraphAdapter/EpisodeAdapter/PersonProfileAdapter/ExperienceAdapter |
| `embedding_manager.py` | 11 KB | 339 | 向量嵌入管理器(BAAI/bge-small-zh-v1.5, 512维) |
| `summary_store.py` | 27 KB | 719 | 双索引摘要存储(FAISS向量索引+node_id指针映射) |
| `memory_evolution.py` | 20 KB | 588 | 记忆巩固/遗忘引擎(MemoryStrength/Consolidator/Forgetter/EvolutionEngine) |
| `llm_memory_reviewer.py` | 19 KB | 509 | LLM剪枝守卫(PRE_STORE/PRE_EVICT/PERIODIC_REVIEW三种审查模式) |

### 3.4 记忆系统-标签 (Memory Tags)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `smart_tagging.py` | 17 KB | 452 | 智能标签引擎(重要度/温度自动标注规则) |
| `tagging_engine.py` | 19 KB | 542 | 标签引擎(规则匹配+LLM辅助标注) |
| `time_tags.py` | 14 KB | 461 | 时间标签(HOT/WARM/COLD基于last_accessed动态计算) |

### 3.5 记忆系统-RAG (Memory RAG & Search)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `base_rag_library.py` | 21 KB | 622 | RAG基类(向量+BM25混合检索框架) |
| `rag_libraries.py` | 28 KB | 759 | 3个RAG库(ExperienceRAG/MemoryRAG/KnowledgeRAG) |
| `rag_manager.py` | 12 KB | 333 | RAG统一管理器(4库注册/路由/按需调用) |
| `hybrid_search_config.py` | 11 KB | 357 | 混合检索配置(向量权重/BM25权重/TopK参数) |
| `vector_cache.py` | 20 KB | 553 | 向量缓存(减少重复embedding计算开销) |
| `tool_rag.py` | 9 KB | 233 | 工具RAG(FC工具能力索引,独立于记忆系统) |

### 3.6 记忆系统-旧组件 (Memory Legacy)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `short_term_memory.py` | 53 KB | 1290 | 短期记忆(已废弃,对话缓存/向量检索被图节点替代) |
| `episodic_memory.py` | 29 KB | 725 | 临时记忆(已废弃,摘要逻辑迁入EpisodeAdapter) |
| `three_libraries.py` | 27 KB | 878 | 三库分离架构(第一代,已被MemoryGraph统一替代) |

### 3.7 记忆系统-知识 (Memory Knowledge)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `knowledge_graph.py` | 27 KB | 715 | 知识图谱(实体/关系/三元组管理) |
| `person_profile.py` | 24 KB | 649 | 人物档案(身份/偏好/关系持久化) |

### 3.8 记忆系统-经验 (Memory Experience)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `enhanced_experience_store.py` | 33 KB | 883 | 增强经验存储(复盘提取/置信度评估/去重) |
| `experience_generator.py` | 14 KB | 418 | 经验自动生成(模式匹配提取经验) |

### 3.9 记忆系统-维护 (Memory Maintenance)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `integration.py` | 17 KB | 510 | 集成层(各子系统协调初始化) |
| `hot_update_engine.py` | 15 KB | 441 | 热更新引擎(运行时替换组件) |
| `patch_applier.py` | 13 KB | 420 | 补丁应用器(增量更新图结构) |
| `rollback.py` | 16 KB | 505 | 回滚机制(操作撤销/状态恢复) |
| `semantic_drift_detector.py` | 9 KB | 244 | 语义漂移检测(概念偏移监控) |

### 3.10 L1-B 调度门控层

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `attention_controller.py` | 15 KB | 413 | 注意力控制器(IDLE/BUSY/SUSPENDED状态机/事件优先级路由/中断冻结恢复) |
| ★`scheduler_gatekeeper.py` | 115 KB | 2411 | **调度守门员**: 用户输入入口/复盘触发/语音检测/上下文打包/事件路由到L2 |

### 3.11 L2 任务系统

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| ★`task_graph.py` | 28 KB | 734 | **TaskGraph**: 任务图谱(节点/层级边hEdges/依赖边dEdges/地址get_node_address/序列化) |
| `task_suspension.py` | 13 KB | 315 | 任务挂起/恢复(SuspendableTaskState/JSON持久化/模糊匹配find_by_description) |
| `task_state_manager.py` | 9 KB | 226 | 任务状态管理器(活跃任务/冻结字典/任务栈LIFO) |
| `task_snapshot.py` | 10 KB | 260 | 任务快照(对话历史/执行进度/工作记忆序列化) |
| `snapshot_manager.py` | 18 KB | 470 | 快照管理器(冻结freeze/解冻thaw/磁盘持久化) |
| `l2_snapshot_interface.py` | 11 KB | 308 | L2快照接口(InferenceEngine对接冻结/恢复) |

### 3.12 L2 中断系统

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `interrupt_handler.py` | 15 KB | 422 | 中断处理器(检测中断/冻结快照/GenerationLoop包装器) |
| `interrupt_controller.py` | 8 KB | 212 | 中断控制器(任务栈管理/自动恢复最近冻结) |

### 3.13 L2 恢复系统

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| `environment_snapshot.py` | 12 KB | 349 | 环境快照(ObjectState/UserState/TaskCondition/变化检测compare_snapshots) |
| `recovery_notifier.py` | 7 KB | 176 | 恢复通知器(启动时扫描checkpoints和suspended_tasks,通知用户) |

### 3.14 L2 推理核心

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| ★`inference_engine.py` | 126 KB | 2709 | **推理引擎**: _on_l2_command入口/retrieve_context记忆检索/_build_messages_with_history构建prompt(含任务规则+层级感知+记忆注入+注意力状态)/FC自主循环(100步限制+中断检查+超时降级)/信息缺口检测/_update_memory异步记忆写回 |

### 3.15 FC工具 (Function Calling)

| 文件 | 大小 | 行数 | 说明 |
|------|------|------|------|
| ★`task_tools.py` | 25 KB | 684 | 任务FC工具: task_create_plan(含重复拦截)/task_add_node/task_add_dependency/task_mark_status/task_view_overview/task_suspend/task_list_suspended(含自动恢复) |
| `attention_tool.py` | 8 KB | 201 | 注意力导航: navigate_attention(deeper/broader/jump) |
| `memory_graph_tools.py` | 19 KB | 540 | 记忆图谱: recall_memory(关键词搜索)/recall_node_context(子图摘要)/discover_related(语义近邻) |

---

## 四、关键数据流

### 4.1 用户输入 -> 记忆检索 -> LLM推理 -> 记忆写入
```
scheduler_gatekeeper._handle_normal_command()
  -> DialogueAdapter.add_round()           // 创建对话节点
  -> event_bus.publish(SYSTEM_L2_COMMAND)   // 发送到L2
  -> inference_engine._on_l2_command()
    -> _process_with_memory()
      -> MemoryGraph.retrieve_context()     // 双路径检索(热遍历+冷FAISS)
      -> _build_messages_with_history()     // 构建prompt(身份+规则+层级+记忆)
      -> FC循环 (模型自主调用工具, 最多100步)
      -> _update_memory()                   // 写回记忆
        -> DialogueAdapter.finalize_round() // FAISS摘要索引
        -> assign_session_by_similarity()   // embedding话题匹配分配session
        -> _propagate_address_to_tasks()    // 地址继承传播
```

### 4.2 任务创建 -> 图谱同步
```
LLM调用 task_create_plan (FC工具)
  -> TaskGraph 创建根节点 "req"
  -> set_active_task_graph() 设为活跃
  -> TaskGraphAdapter.incremental_sync()
    -> MemoryGraph.add_node(TASK类型)
    -> 建立 round <-> task REFERENCE边
    -> 地址继承: dialogue:session_xxx/task:tg_xxx/req
```

### 4.3 注意力中断处理
```
紧急事件(priority>=8)到达 AttentionController
  -> Freeze: _create_l2_snapshot() 冻结当前任务
  -> Recompose: 组合紧急上下文+旧任务摘要
  -> Inject: _force_l2_respond() 强制L2响应
  -> on_l2_idle(): 恢复冻结的任务继续执行
```

---

## 五、审查重点建议

1. **memory_graph.py** (2784行) - 最核心文件,重点审查 retrieve_context() 检索逻辑和 compute_activations() BFS扩散
2. **graph_adapters.py** (1298行) - 重点审查 DialogueAdapter 的 session 分配和地址继承传播
3. **inference_engine.py** (2709行) - 重点审查 FC循环、system prompt 构建、记忆检索注入
4. **task_tools.py** (684行) - 重点审查任务创建拦截、挂起恢复逻辑
5. **scheduler_gatekeeper.py** (2411行) - 重点审查事件打包和路由逻辑
