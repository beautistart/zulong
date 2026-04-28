# 图记忆机制深度审查与 FC/非FC 工具全景分析

> 审查日期：2026-04-21
> 状态：架构讨论文档，待决策
> 更新：新增第九节「MG 能力二分法分类」

---

## 九、MG 能力二分法：系统层自动运行 vs 模型按需调用

### 判定标准

- **系统层**：代码自动执行，不需要模型决策，执行结果直接作用于模型的上下文/行为
- **工具层**：需要模型根据当前情况判断"是否调用、参数是什么"

### A. 系统层自动运行（不需要 FC，直接对模型产生作用）

#### A1. 图维护循环（后台定时器，30 分钟一轮）

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `decay_and_prune()` | 后台异步循环 | 艾宾浩斯衰减 + 边修剪 | 模型下次检索时，弱关联已被自动清理 |
| `run_importance_review()` | 后台异步循环 | access_count≥3 → 自动提升 IMPORTANT | 模型自动获得更重要的上下文权重 |
| `cleanup_orphan_nodes()` | 每 3 轮衰减周期 | 清除语义孤立节点 | 减少噪声节点干扰模型 |
| `save()` | 脏标记防抖 + 周期保存 | 持久化图到磁盘 | 模型重启后状态不丢失 |
| `update_temperature()` | 衰减周期内 | 更新节点温度缓存(HOT/WARM/COLD) | 影响检索时的冷热路径选择 |

#### A2. 适配器同步（代码驱动，每次对话自动触发）

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `TaskGraphAdapter.sync()` | Orchestrator 初始化 | 投影 TaskGraph → MG 节点 | 模型通过 recall 能看到任务结构 |
| `TaskGraphAdapter.incremental_sync()` | 每次图变更回调 | 实时同步任务状态变更 | 模型实时感知任务进度 |
| `DialogueAdapter.ensure_session()` | 用户消息到达 | 创建/复用会话节点 | 模型获得会话上下文 |
| `DialogueAdapter.add_round()` | 用户消息到达 | 创建对话轮次节点 | 模型获得对话历史链 |
| `DialogueAdapter.add_sub_dialogue()` | 重要节点变更 | 创建 Agent turn 节点 | 模型行为被记录为图节点 |
| `DialogueAdapter.finalize_round()` | 对话轮完成 | 归档轮次 + FAISS 索引 | 后续可通过向量搜索找到 |
| `DialogueAdapter._detect_importance()` | 节点创建时 | 模式匹配自动标注重要性 | "我叫X"→IDENTITY，"帮我记住"→MUST_REMEMBER |

#### A3. 焦点追踪（代码自动，每轮/检查点触发）

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `update_focus_to_node()` | 对话创建 + 检查点 | 构建思维深度路径 | `get_focus_path_summary()` 注入 system prompt |
| `set_last_focus_context()` | 每 20 轮 + 关闭时 | 保存焦点上下文 | 重启后恢复模型注意力位置 |
| `set_active_nodes()` | 每轮 Agent 循环 | 标记活跃节点 | 前端高亮 + 检索权重加成 |

#### A4. 激活计算（代码逻辑触发）

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `compute_activations()` | 焦点恢复 + 注意力导航 | BFS 扩散激活 | 激活值影响检索排序 + context boosting |
| `update_node_activation()` | compute_activations 内部 | 更新节点激活值 + 访问计数 | access_count≥3 触发自动重要性提升 |

#### A5. 上下文检索（代码在构建 prompt 时自动调用）

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `retrieve_context()` | AttentionWindow 构建 prompt | 双路径热/冷混合检索 | 模型的 system prompt 中自动注入相关记忆 |
| `_retrieve_hot()` | retrieve_context 内部 | 近期节点关键词+BFS 扩展 | 热记忆自动出现在上下文中 |
| `_retrieve_cold()` | retrieve_context 内部 | FAISS 向量搜索冷数据 | 冷记忆按相关性自动注入 |
| `get_focus_path_summary()` | prompt 构建时 | 渲染思维导航路径 | "【思维导航】L1...L2...←当前焦点" 直接注入 prompt |

#### A6. 节点创建时的自动行为

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `_auto_embed_node()` | add_node() 内部 | 自动生成 embedding | 后续语义搜索可用 |
| `discover_semantic_neighbors()` | _auto_embed_node 内部 | 余弦相似度发现近邻 | 自动创建 SEMANTIC 边 |
| `_update_coactivation_counter()` | compute_activations 内部 | 追踪共激活对 | 达阈值自动创建 ASSOCIATION 边 |

#### A7. 后端溯源（内部自动解析）

| 方法 | 触发方式 | 作用 | 对模型的影响 |
|---|---|---|---|
| `resolve_backend_ref()` | _node_to_result 内部 | 查找 KG/RAG 原始内容 | 模型看到的不只是摘要，还有原文 |
| `set_rag_manager()` | Orchestrator 初始化 | 注册 RAG 后端 | 使 resolve_backend_ref 能工作 |

**系统层小结：共 25+ 个方法/流程，完全自动运行，模型无感知但直接受益**

---

### B. 工具层（需要模型按需调用，应封装为 FC 工具）

#### B1. 记忆搜索与发现（模型决定搜什么、怎么搜）

| MG 方法 | 当前 FC 封装 | 缺失的能力 | 建议新 FC 工具 |
|---|---|---|---|
| `search_nodes()` | recall_memory | 仅关键词子串匹配，max_results=8 硬编码 | **recall_memory** (已有，需增强: 放开 max_results，加 mode 参数) |
| `search_summaries()` | 无 | FAISS 向量搜索完全不可用 | **recall_memory** mode="vector" |
| `retrieve_context()` | 无 | 双路径混合检索不可用 | **recall_context_hybrid**(query, top_k, hot_window_min) |
| `discover_semantic_neighbors()` | 部分(在 recall_node_context 中) | 不能主动发现任意节点的语义近邻 | **discover_neighbors**(node_id, top_k, threshold) |

#### B2. 图导航与遍历（模型决定去哪里、看什么）

| MG 方法 | 当前 FC 封装 | 缺失的能力 | 建议新 FC 工具 |
|---|---|---|---|
| `resolve_address()` | 无 | 无法通过地址字符串定位节点 | **navigate_to**(address) → 返回节点 + 焦点路径 |
| `get_ancestors()` | 无 | 无法上行查看思维深度 | **navigate_to** 包含 |
| `get_children()` | 无 | 无法下行展开子任务 | **navigate_to** 包含 |
| `get_parent()` | 无 | 无法回溯到上层 | **navigate_to** 包含 |
| `get_neighbors()` | 无 | 无法发现任意深度的邻居 | **explore_neighbors**(node_id, edge_types, depth) |
| `get_subgraph_summary()` | 部分(在 recall_node_context 中) | 可独立使用 | 已在 recall_node_context 中覆盖 |
| `get_focus_path_summary()` | 无(仅前端展示) | 模型看不到当前思维深度 | **view_focus_path**() |

#### B3. 图写入操作（模型决定创建/修改什么）

| MG 方法 | 当前 FC 封装 | 缺失的能力 | 建议新 FC 工具 |
|---|---|---|---|
| `add_node()` | 部分(通过 plan_add_node 操作 TaskGraph) | 不能直接创建 KNOWLEDGE/EPISODE 等非任务节点 | **create_memory_node**(type, label, content) |
| `add_edge()` | 部分(通过 plan_add_dependency) | 不能创建语义关联/引用等非结构边 | **link_nodes**(source_id, target_id, edge_type) |
| `set_importance()` | 无 | 模型无法手动标注重要性 | **set_memory_importance**(node_id, level) |
| `promote_importance()` | 无 | 模型无法提升节点重要性 | **set_memory_importance** 包含 |

#### B4. 激活与学习（模型决定何时强化/探索）

| MG 方法 | 当前 FC 封装 | 缺失的能力 | 建议新 FC 工具 |
|---|---|---|---|
| `compute_activations()` | 无 | 模型不能主动触发 BFS 扩散 | **activate_and_retrieve**(seed_ids, depth, min_score) |
| `hebbian_strengthen()` | 无 | **从未被调用！** 共激活边永远不强化 | **strengthen_associations**() |

#### B5. 跨任务与历史（模型决定访问哪个历史任务）

| MG 方法 | 当前 FC 封装 | 缺失的能力 | 建议新 FC 工具 |
|---|---|---|---|
| `get_node()` + 历史节点 | recall_memory 可搜索到 | 但无法直接读取节点完整内容 | **recall_node_detail**(node_id) |
| `resolve_backend_ref()` | 无 | 无法溯源到 KG/RAG 原始数据 | **resolve_source**(node_id) |

**工具层小结：需要新增 8-10 个 FC 工具覆盖上述缺失能力**

---

### C. 纯基础设施（既不是系统自动，也不是模型需要，是内部实现）

| 方法 | 性质 | 说明 |
|---|---|---|
| `has_node()` | 内部查询 | 其他方法的前置检查 |
| `get_edge()` | 内部查询 | 边属性查询 |
| `has_edge()` | 内部查询 | 边存在性检查 |
| `remove_node()` | 内部维护 | 由 decay_and_prune 自动调用 |
| `remove_edge()` | 内部维护 | 由 decay_and_prune 自动调用 |
| `get_node()` | 内部查询 | 其他方法的基础，不直接暴露 |
| `get_nodes_by_type()` | 内部查询 | 适配器用，模型用 search_nodes |
| `get_temperature()` | 内部计算 | 动态计算，不需要模型知道 |
| `get_importance()` | 内部查询 | 被 decay_and_prune/review 使用 |
| `update_temperature()` | 内部维护 | 衰减周期调用 |
| `is_recent()` | 内部判断 | 检索路径内使用 |
| `get_embedding()` / `set_embedding()` | 内部存储 | embedding 管理 |
| `index_summary()` | 内部索引 | 适配器调用 |
| `summary_index_count` | 内部状态 | 统计用 |
| `register_adapter()` | 启动配置 | 一次性设置 |
| `sync_all()` | 启动配置 | 一次性全量同步 |
| `start_prune_loop()` / `stop_prune_loop()` | 生命周期 | 启动/关闭 |
| `submit_prune_review()` / `process_review_result()` | LLM 审查回调 | 衰减审查机制 |
| `get_active_node_ids()` | 前端查询 | UI 用 |
| `to_frontend_dict()` / `flush_changes()` | 前端 API | 序列化输出 |
| `stats` | 监控 | 运维用 |
| `_load()` / `_mark_dirty()` / `_do_auto_save()` | 持久化内部 | 自动 |
| `_serialize_embeddings()` / `_deserialize_embeddings()` | 序列化内部 | 自动 |

**纯基础设施小结：24+ 个方法，不需要暴露给模型，也不需要自动运行逻辑**

---

### D. 关键发现

1. **hebbian_strengthen() 从未被调用** — compute_activations 计算了 `_last_activated_edges`，但没有代码调用 `hebbian_strengthen()` 来强化这些边。赫布学习机制是**死代码**。

2. **系统层自动运行 ≈ 25 个方法/流程** — 这些已经在对模型产生作用，但模型"不知道它们存在"

3. **工具层需要 ≈ 8-10 个新 FC 工具** — 覆盖搜索发现、图导航、写入操作、激活学习、跨任务历史

4. **纯基础设施 ≈ 24 个方法** — 不需要暴露，是图的内部实现

5. **当前 FC 暴露 2/65+ = 3.1%** — 扩展后预计 12/65+ ≈ 18.5%，加上系统层 25 个自动运行，模型可感知 MG 约 37/65+ ≈ 57% 的能力

---

## 一、Orchestrator 生命周期与"必须重建"问题

### 1.1 当前生命周期

```
创建 Orchestrator → run() 执行 FC 循环 → finally: _active_orchestrator = None → GC 回收
                                                                   ↑
                                                             这里一切归零
```

Orchestrator 是**一次性容器**（disposable instance），设计上等同于无状态 HTTP 请求处理器。

### 1.2 恢复任务时丢失/保留的状态

| 保留的（序列化到磁盘） | 丢失的（实例内存） |
|---|---|
| TaskGraph 节点/边/状态 | ScratchPad 临时笔记 |
| 对话历史 messages | CircuitBreaker 状态（RED/YELLOW/GREEN） |
| session_id / request_id | ToolRegistry 调用记录 |
| workspace_dir | AttentionWindow 上下文模式 |
| | GuardrailEngine 计数器 |

### 1.3 为什么不能复用旧 Orchestrator？

1. **线程屏障**：旧 Orchestrator 运行在守护线程中，线程退出后 thread-local 状态无效
2. **组件初始化依赖**：scratch_pad、tool_registry、guardrail_engine 都在 `run()` 中创建，与单次执行绑定
3. **上下文语义**：AttentionWindow 维护轮次级历史、模式状态，是瞬态的

**核心洞察**：丢失的都是 FC 循环的运行时状态。如果不依赖 FC 循环，这些"丢失"本身就不存在。

---

## 二、FC 工具全景

### 2.1 当前 25 个 FC 工具

| 类别 | 工具名 | 是否访问 MemoryGraph |
|---|---|---|
| **规划** | plan_add_node, plan_update_node, plan_remove_node, plan_add_dependency, plan_mark_status, plan_add_file | 间接（操作 TaskGraph，通过 Adapter 同步到 MG） |
| **查看** | view_graph_overview, view_node_detail, view_focused_context | 间接 |
| **执行** | exec_write_file, exec_run_command | 否 |
| **交互** | ask_user | 否 |
| **控制** | submit_final_answer | 否 |
| **记忆** | recall_memory, recall_node_context | **直接访问 MemoryGraph** |
| **跨任务** | list_task_workspaces | 否 |
| **桥接系统** | BackTraceTool 系列 (5), ToolEngine 桥接 (9+) | 否 |

### 2.2 MemoryGraph 能力暴露率

```
MemoryGraph 公开方法:  53 个
通过 FC 暴露给模型:    2 个 (recall_memory + recall_node_context)
覆盖率:               3.8%
```

### 2.3 recall_memory 的实际能力

- **搜索方式**：纯关键词子串匹配
- **评分**：node_id 精确匹配(1.0) > label 包含(0.8) > metadata 包含(0.5) + activation*0.2
- **结果上限**：硬编码 8 条
- **无**：向量搜索、扩散激活、层级导航、语义近邻

---

## 三、MemoryGraph 内部机制详解

### 3.1 9 种节点类型 & 7 种边类型

**节点**：TASK, DIALOGUE, KNOWLEDGE, EXPERIENCE, EPISODE, FILE, CONCEPT, PERSON, DOCUMENT

**边**：
- **结构边**（protected=True，永不修剪）：HIERARCHY, DEPENDENCY, TEMPORAL
- **语义边**：SEMANTIC（余弦相似度 > 0.7）, CAUSAL
- **学习边**：ASSOCIATION（赫布学习产生）, REFERENCE

### 3.2 BFS 扩散激活

```python
def compute_activations(self, seed_node_ids, max_depth=3, decay=0.5, min_activation=0.01):
    """加权 BFS 扩散"""
    # 初始化: 种子节点 activation = 1.0
    # 传播公式: activation[neighbor] = activation[current] × edge_weight × decay
    # 示例 (decay=0.5):
    #   Seed A: 1.0
    #   1-hop B (weight=1.0): 1.0 × 1.0 × 0.5 = 0.5
    #   2-hop C (weight=0.8): 0.5 × 0.8 × 0.5 = 0.2
    #   3-hop D (weight=0.6): 0.2 × 0.6 × 0.5 = 0.06
```

### 3.3 赫布学习

```python
def hebbian_strengthen(self):
    """共激活边强化"""
    # 公式: new_weight = old_weight + eta * (1 - old_weight)
    # eta = 0.1（固定学习率）
    # 渐近趋近 1.0，永不超出
    # protected 边跳过（保持结构完整性）
```

### 3.4 突触修剪（艾宾浩斯衰减）

```python
def decay_and_prune(self):
    """衰减公式: decayed = weight × exp(-elapsed_hours × ln(2) / half_life)"""
    # 半衰期按重要性分级:
    #   TRIVIAL: 6h | NORMAL: 24h | IMPORTANT: 7天
    #   FACT: 15天 | IDENTITY: 30天 | MUST_REMEMBER: ∞
    # 衰减后:
    #   < 0.05 → 删除边
    #   0.05~0.15 → 标记濒危（可送 LLM 审查）
    #   ≥ 0.15 → 更新权重
```

### 3.5 动态注意力（焦点机制）

```python
def update_focus_to_node(self, node_id):
    """构建焦点路径（思维深度导航）"""
    # 1. get_ancestors() → 从当前节点上行到根
    # 2. 构建路径: [root, ..., grandparent, parent, node_id]
    # 3. focus_depth = len(path) - 1
    # 4. 保存到 _last_focus_context

def get_focus_path_summary(self):
    """渲染思维深度路径"""
    # 输出示例:
    # 【思维导航】
    # L1 [任务] 项目开发
    #  └─ L2 [任务] 实现API
    #   └─ L3 [对话] 帮我调试异常 ← 当前焦点
```

### 3.6 地址解析

```python
def resolve_address(self, address: str) -> Optional[GraphNode]:
    """解析图地址字符串"""
    # 支持格式:
    #   "tg:{graph_id}/task:{node_id}"  → TaskGraph 地址
    #   "task:o1_1"                      → 直接 node_id
    #   "dialogue:round_42"              → 对话节点
    #   metadata 中的 graph_address       → 后备匹配
```

### 3.7 层级导航

```python
def get_ancestors(node_id, edge_type=HIERARCHY) → List[GraphNode]  # 上行到根
def get_parent(node_id, edge_type=HIERARCHY) → Optional[GraphNode]  # 直接父节点
def get_children(node_id, edge_type=HIERARCHY) → List[GraphNode]    # 子节点列表
```

### 3.8 记忆搜索完整链路

```
LLM 调用 recall_memory(query="部署配置", node_type="task")
  → tools.py: _handle_recall_memory()
    → mg = get_memory_graph()  # 单例
    → results = mg.search_nodes(query, node_types=[NodeType.TASK], max_results=8)
      → 遍历所有节点
      → 评分: node_id精确(1.0) > label包含(0.8) > metadata包含(0.5)
      → 加成: + activation * 0.2
      → 排序返回 top 8
    → 返回 JSON {"found": N, "results": [...]}
```

---

## 四、非 FC 工具（直接代码调用）

### 4.1 图适配器（直接操作 MemoryGraph）

| 适配器 | 调用的 MG 方法 | 触发时机 |
|---|---|---|
| TaskGraphAdapter | add_node, add_edge, get_node | 每轮 FC 循环结束后 sync |
| DialogueAdapter | add_node, add_edge, get_nodes_by_type, set_importance | 每次对话创建/恢复 |
| KnowledgeGraphAdapter | add_node, add_edge | 知识注入时 |
| EpisodicMemoryAdapter | add_node, add_edge | 经验归档时 |

### 4.2 系统级直接调用

| 调用者 | 目标 | 说明 |
|---|---|---|
| Gatekeeper 闭包 | DialogueAdapter, MemoryGraph | 创建对话节点、焦点更新 |
| _handle_resume_task | TaskSuspensionManager | 磁盘读写 |
| RecoveryNotifier | 检查点提升 | 崩溃恢复 |
| 后台定时器 | decay_and_prune() | 突触修剪 |
| InferenceEngine | mg.set_rag_manager() | RAG 后端注册 |

---

## 五、模型能用但 FC 未暴露的 MemoryGraph 能力

| MemoryGraph 方法 | 功能 | 为何未暴露 |
|---|---|---|
| `compute_activations()` | BFS 扩散激活 | 无对应 FC 工具 |
| `retrieve_context()` | 热/冷混合检索 + boosting | 未集成到 FC 调度 |
| `search_summaries()` | FAISS 向量搜索 | embedding 基础设施未暴露 |
| `get_ancestors/get_children/get_parent` | 层级导航 | Agent 看不到 MG 层级 |
| `discover_semantic_neighbors()` | 余弦相似度搜索 | embedding 管理未暴露 |
| `promote_importance()` | 重要性提升 | 仅有自动检测 |
| `get_subgraph_summary()` | 节点 + 邻居摘要 | 未加入 FC 工具 |
| `resolve_address()` | 地址解析 | 未加入 FC 工具 |
| `resolve_backend_ref()` | 溯源（KG/RAG） | 未加入 FC 工具 |
| `get_focus_path_summary()` | 思维深度路径 | 仅前端展示 |
| `hebbian_strengthen()` | 赫布学习强化 | 自动触发未接入 |
| `update_focus_to_node()` | 焦点切换 | 未暴露为主动操作 |

---

## 六、两条路径分析

### 路径 1：纯系统工具调用，完全舍弃 FC

```
用户输入 → LLM 推理 → 自然语言输出 → 意图解析器（代码/规则） → 直接调用 MemoryGraph API
```

**优势**：零 FC 开销、确定性操作不等待模型决策
**致命问题**：意图解析器 = 另一种形式的硬编码，是刚拆掉的东西的翻版
**结论：倒退，不可取**

### 路径 2：FC 与图空间完全融合

```
用户输入 → Gatekeeper（仅 L1-A 反射 + 唤醒） → 模型直接在图上操作
├─ navigate_graph(address="tg:g1/task:o1_1")     ← 思维深度导航
├─ activate_and_retrieve(seeds=["task:o1"], depth=3) ← BFS 扩散
├─ recall_context_hybrid(query="部署配置", mode="hybrid") ← 混合检索
├─ set_memory_importance(node_id="kg:安全协议", level="MUST_REMEMBER")
├─ discover_neighbors(node_id="task:o1", top_k=5) ← 语义近邻
└─ view_focus_path()                               ← 当前思维深度
```

**优势**：
- 模型获得图的完整操作能力
- 不再需要"实例化 Orchestrator"来访问任务 — 任务就在图上
- FC 参数有 schema 约束，比自然语言解析可靠
- 模型自主决定搜索策略

**需要解决的问题**：
- FC 工具定义占 context window → 按需加载工具集
- 模型可能误操作图 → 写操作加确认/回滚
- 每次新 Orchestrator 状态丢失 → **根本不需要 Orchestrator 了，状态就在图上**

### 分层实施建议

**第一层（立即可做）**：扩展 FC 工具，暴露 MemoryGraph 核心能力

| 新 FC 工具 | 包装的 MG 方法 | 用途 |
|---|---|---|
| `navigate_graph` | resolve_address + get_ancestors/get_children | 地址解析 + 层级导航 |
| `activate_and_retrieve` | compute_activations + 结果聚合 | BFS 扩散 + 返回激活节点 |
| `recall_context_hybrid` | retrieve_context + search_summaries | 混合检索（关键词+向量+激活度） |
| `set_memory_importance` | promote_importance | 重要性标注 |
| `discover_semantic_neighbors` | discover_semantic_neighbors | 语义近邻发现 |
| `view_focus_path` | get_focus_path_summary | 当前思维深度 |

**第二层（架构重构）**：解耦 Orchestrator 与任务状态
- 任务状态只存在 MemoryGraph 节点上
- Orchestrator 退化为"推理会话"概念 — 不持有状态
- 恢复任务 = 导航到图上的任务节点，不需要重建 Orchestrator

**第三层（愿景）**：模型即图的操作者
- 每轮对话 = 图上的一个 DIALOGUE 节点
- 模型每次操作 = 在图上创建/修改/连接节点
- 不需要"开始任务"或"结束任务"—— 模型始终在图上

---

## 七、核心问题：FC 与 MG 内部能力的本质关系

### 为什么模型能调 FC 工具却不能调 MG 内部能力？

**答案：没有技术原因。FC 工具就是 MG 方法的薄封装。**

```
LLM 输出: {"name": "recall_memory", "arguments": {"query": "部署"}}
    ↓
FC 分发器: _handlers["recall_memory"] → _handle_recall_memory()
    ↓
实际调用: mg.search_nodes(query="部署", max_results=8)
    ↓
返回 JSON → LLM 下一轮输入
```

**recall_memory 就是 `mg.search_nodes()` 的 FC 封装**。同理：
- `navigate_graph` = `mg.resolve_address()` + `mg.get_ancestors()` 的 FC 封装
- `activate_and_retrieve` = `mg.compute_activations()` 的 FC 封装
- `set_memory_importance` = `mg.promote_importance()` 的 FC 封装

**模型不能调用 MG 内部能力，不是因为 LLM 有什么限制，而是因为我们只写了 2 个封装函数。** FC 是 LLM 与 Python 代码交互的唯一通道，我们选择只暴露了 3.8% 的通道。

### FC 不是"框"—— 是模型的双手

FC（Function Calling）不是限制模型的笼子，而是模型操作外部世界的**双手**。

当前的问题不是"模型被困在 FC 里"，而是"模型的双手只接了 2 根手指"。

把 MG 的 53 个方法都封装为 FC 工具，模型就能"看到"并"操作"整个图空间。

---

## 八、待决策

1. 是否采纳路径 2（FC 与图空间完全融合）？
2. 第一层 6 个新 FC 工具的优先级排序？
3. 是否需要重构 Orchestrator 使其不再持有任务状态？
4. 工具定义膨胀（25→30+）对 context window 的影响如何管理？
