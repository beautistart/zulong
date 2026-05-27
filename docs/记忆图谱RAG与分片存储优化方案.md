# 记忆图谱 RAG 与分片存储优化方案

> 状态：讨论稿，供后续增量删改  
> 对齐依据：`TSD/祖龙 (ZULONG) 机器人系统技术规格说明书 (TSD)1.7.txt`  
> 范围：MemoryGraph、DualIndexSummaryStore、RAGManager、InteractionStore、TaskGraph、Web 端记忆图谱展示

## 1. 背景与问题

当前运行态已采用分片 Hybrid 记忆图谱：

- 图谱权威属性：`data/memory_graph_hybrid/*/properties`，LMDB 存储节点与边属性。
- 拓扑缓存：`topology.graphml`，用于快速 BFS、邻居发现和 Web 展开。
- 分片索引：`shard_index.json`，记录分片统计。
- 会话事件账本：`data/interaction/interaction_store.sqlite3`。
- 任务图谱备份：`data/graph_backups/*.json`、`data/completed_tasks/*.json`。

已观察到的问题：

- LMDB 中真实节点数量远多于 `topology.graphml`，说明拓扑缓存可能滞后。
- Web 首屏若只读根节点，会被误解为“只有根节点，没有子节点”。
- Web 会话与 `task_graph_id` 绑定不足时，任务图谱虽已落盘，但刷新后无法稳定恢复。
- 后期记忆规模极大、分片极多时，Web 端若全量重建 MemoryGraph，会带来明显卡顿。

## 2. 目标

1. 保持 MemoryGraph 的图结构与 RAG 摘要索引协同工作。
2. 利用现有 RAG 库能力，对图记忆生成“向量摘要 + 图记忆地址”。
3. 检索时采用冷热并行：
   - 热数据：遍历活跃/近期图节点，做语义 + 关键词检索。
   - 冷数据：走摘要向量、详情向量、FAISS/RAG 检索。
4. Web 端只做索引级首屏与按需展开，禁止全量扫描全部分片。
5. 任务图谱恢复优先走结构化索引，不依赖全文日志或正则。

## 3. 总体原则

- `LMDB properties` 是记忆图谱真实数据权威。
- `topology.graphml` 是拓扑缓存，不是唯一事实来源。
- `SummaryStore/RAG` 是冷数据导航层，不替代 MemoryGraph。
- `InteractionStore` 是 Web 会话恢复与审计账本，不承载完整长期语义记忆。
- Web 展示走分页、按需展开、懒加载，不返回全量图。
- 后台维护可以修复和重建索引，前台请求不得同步等待重建。

## 4. 存储分层设计

### 4.1 MemoryGraph Hybrid

职责：

- 存储会话、轮次、任务、工具、审批、代码锚点等图节点。
- 存储 HIERARCHY、TEMPORAL、REFERENCE、SEMANTIC、ASSOCIATION 等边。
- 支持 BFS 扩散、赫布增强、注意力路径、按需展开。

数据形态：

- 拓扑层：节点 ID、节点类型、边、边权重。
- 属性层：label、content、metadata、importance、temperature、activation、backend_ref。

### 4.2 RAG / 双索引摘要库

职责：

- 对 MemoryGraph 中冷却后的会话、任务、经验生成摘要。
- 保存摘要向量、详情向量和关键词索引。
- 每条摘要必须回写图记忆地址，例如：
  - `node_id`
  - `session_node_id`
  - `round_node_id`
  - `task_graph_id`
  - `shard_id`
  - `full_path`

推荐摘要对象：

- 会话摘要：一个 session 下多个 round 的主题摘要。
- 任务摘要：TaskGraph 根节点及关键子任务摘要。
- 工具执行摘要：工具调用、审批、结果、失败原因。
- 经验摘要：可复用的错误修复、项目规则、用户偏好。

### 4.3 InteractionStore

职责：

- 保存 Web 会话列表、消息账本、`conversation_id`、`session_node_id`、`task_graph_id`。
- 支撑 Web 刷新后的快速恢复。
- 作为 MemoryGraph 写入失败时的短期回填来源。

InteractionStore 不应成为长期语义检索主库。

## 5. 冷热并行检索方案

### 5.1 热数据检索

热数据范围：

- 当前 session 根节点及 3 跳 BFS 邻域。
- 最近活跃 session / round。
- 当前活跃 TaskGraph。
- `temperature=hot` 或最近访问节点。
- InteractionStore 最近 N 条事件。

检索方式：

- 图 BFS：从 `session_node_id`、当前 round、active task 开始扩散。
- 关键词检索：label、content、metadata.goal、metadata.user_text、metadata.bot_text。
- 轻量语义检索：对热节点摘要或 content 做向量相似度。
- 会话优先加权：`session_id` 匹配应高于全局相似节点。

热路径目标：

- 不跨全部分片扫描。
- 响应时间保持在几十毫秒级。
- 召回当前上下文和近期任务状态。

### 5.2 冷数据检索

冷数据范围：

- Warm/Cold 记忆节点。
- 历史会话摘要。
- 历史 TaskGraph 摘要。
- 经验库与 RAG 库。

检索方式：

- `DualIndexSummaryStore`：SQLite 条件过滤 + FAISS 摘要向量。
- 详情向量：摘要命中后再加载对应详情。
- 经验库混合检索：向量 0.7 + BM25 0.3。
- 图地址回跳：根据摘要命中的 `node_id/shard_id/full_path` 回到 MemoryGraph，加载邻域。

冷路径目标：

- 不加载完整历史图。
- 先找摘要，再按图地址取局部“肉”。
- 只把高置信结果注入 L1-B 上下文包。

### 5.2.1 冷记忆摘要导航与增量回忆

用户输入后，系统先对输入内容进行记忆检索。冷记忆流程建议采用“摘要先行、图地址回跳、工具式增量回忆”：

```text
用户输入
→ L1-B / Memory Retriever 对输入做检索
→ 记忆 RAG 命中摘要
→ 将摘要文本 + 图记忆地址拼接进 LLM 上下文
→ LLM 判断是否需要了解详情
→ LLM 调用记忆工具，以摘要对应的图记忆 ID 作为 BFS 种子
→ MemoryGraph 做一次增量关联扩散
→ 返回局部详细记忆内容
→ LLM 基于增量回忆继续推理
```

首次注入 LLM 上下文的冷记忆不应是完整历史正文，而应是轻量导航信息：

```text
[冷记忆摘要]
- summary_id: summary_20260527_xxx
- graph_memory_id: dialogue:session_xxx/round_xxx
- shard_id: 2026_05
- full_path: dialogue:session_xxx/round_xxx
- summary: 曾讨论过 Web 端任务图谱刷新后无法恢复，原因是 conversation 未绑定 task_graph_id。
- relevance: 0.86
- recall_hint: 如需详情，可调用 recall_memory，以 graph_memory_id 为 seed 做 BFS 扩散。
```

LLM 若需要细节，不直接要求系统全量加载历史，而是调用记忆工具：

```text
recall_memory({
  "seed_node_id": "dialogue:session_xxx/round_xxx",
  "max_depth": 2,
  "limit": 20,
  "include_content": true
})
```

工具返回的内容应是局部增量，而不是整库回放：

- 种子节点正文或摘要。
- 直接父子节点。
- 时间相邻 round。
- 关联 TaskGraph 节点。
- 关联工具/审批/结果节点。
- 高权重 SEMANTIC / ASSOCIATION 邻居。

这样冷记忆检索分为两步：

1. **摘要召回**：低成本把可能相关的历史放入 LLM 注意力范围。
2. **增量回忆**：只有当 LLM 明确需要细节时，才以图记忆 ID 为种子做 BFS 局部扩散。

### 5.3 融合排序

建议统一融合分：

```text
score =
  0.35 * semantic_score
+ 0.20 * keyword_score
+ 0.20 * graph_proximity_score
+ 0.15 * recency_score
+ 0.10 * importance_score
+ session/task bonus
```

热路径优先保证上下文连续性，冷路径优先保证历史相关性。若热路径已有高置信结果，冷路径可以异步补充，不阻塞首字输出。

## 6. 分片索引优化

当前 `shard_index.json` 只记录粗统计，后期应扩展或拆出独立索引库。

最低索引字段：

- `session_id -> shard_id`
- `conversation_id -> session_node_id`
- `task_graph_id -> shard_id`
- `task_graph_id -> backup_path`
- `node_id -> shard_id`
- `root_nodes`
- `recent_sessions`
- `active_nodes`
- `hot_nodes`
- `updated_at`

索引更新时机：

- 新增节点时更新 `node_id -> shard_id`。
- 新建 Web 会话时更新 `conversation_id -> session_node_id`。
- 创建或复用 TaskGraph 时更新 `task_graph_id` 相关索引。
- 会话活跃时更新 `recent_sessions`。
- 节点温度变化时更新 `hot_nodes` 或摘要状态。

## 7. Web 端加载策略

### 7.1 首屏

只返回：

- 最近会话列表。
- 根级 session/task 节点。
- 活跃节点 ID。
- 当前任务图谱摘要。
- 统计信息。

禁止首屏返回所有节点和所有边。

### 7.2 展开

用户展开节点时：

1. `node_id` 通过索引定位 `shard_id`。
2. 从拓扑缓存读取一层 children。
3. 从 LMDB 批量加载 children 属性。
4. 返回分页结果。

默认限制：

- 单次最多 50-100 个节点。
- 默认深度 1。
- 更深层由用户继续展开。

### 7.3 搜索

搜索流程：

1. 热路径：当前 session / active task / hot nodes。
2. 冷路径：摘要向量 + 详情向量 + 经验库。
3. 命中摘要后，先把摘要 + 图记忆地址注入上下文。
4. 如 LLM 需要详情，再调用记忆工具，以图记忆 ID 为种子做 BFS 增量回忆。

Web 搜索不应直接遍历全部分片。

## 8. 拓扑一致性维护

### 8.1 启动校验

启动时比较：

- `topology.graphml` 节点/边数量。
- LMDB properties 节点/边数量。
- `shard_index.json` 统计。

若差异超过阈值，标记该分片为 `topology_stale=true`。

### 8.2 后台修复

后台异步执行：

```text
LMDB properties
→ rebuild_topology_from_properties()
→ save topology.graphml
→ 更新 shard_index
→ 广播索引修复完成事件
```

前台请求处理策略：

- 若拓扑可用，继续用拓扑缓存。
- 若拓扑陈旧但 LMDB 可用，对指定节点可走 LMDB 索引兜底。
- 禁止 Web 请求同步重建完整拓扑。

## 9. TaskGraph 持久化与恢复

TaskGraph 恢复优先级：

1. 当前内存活跃 TaskGraph。
2. `InteractionStore.conversation.task_graph_id`。
3. `task_graph_id -> backup_path` 索引。
4. MemoryGraph 中 task 节点地址。
5. BFS 兜底恢复。

必须保证：

- 创建 TaskGraph 后写入 InteractionStore。
- 复用 TaskGraph 后刷新 conversation 的 `task_graph_id`。
- TaskGraph 根节点写入 MemoryGraph，并与 session/round 建立 REFERENCE 或 HIERARCHY 边。
- 任务完成后生成摘要，写入 SummaryStore/RAG，并保留图地址。

## 10. 写入闭环

一次 Web 用户消息建议写入顺序：

```text
Web user message
→ InteractionStore.append_event
→ MemoryGraph: session/round/message nodes
→ 热路径索引更新
→ L1-B 上下文检索
→ L2/FC 执行
→ TaskGraph 更新与备份
→ assistant/tool/status events 写入 InteractionStore
→ MemoryGraph 补充 tool/approval/task summary 节点
→ SummaryStore/RAG 异步摘要和向量化
→ ExperienceStore 可复用经验提炼
```

## 11. 后台维护任务

建议后台任务：

- 拓扑一致性校验与修复。
- Hot/Warm/Cold 状态迁移。
- 冷数据摘要生成。
- 摘要向量索引保存。
- 低价值节点审查与遗忘。
- 分片索引压缩与校验。
- InteractionStore 到 MemoryGraph 的回填。

这些任务不得阻塞 Web 聊天主链。

## 12. 性能边界

目标：

- Web 首屏：只读索引，避免全图扫描。
- 单节点展开：个位数到几十毫秒，取决于 children 数量。
- 热路径检索：几十毫秒级。
- 冷路径检索：百毫秒级以内，允许异步补充。
- 拓扑修复：后台执行，可耗时，不阻塞用户操作。

风险：

- 如果 `node_id -> shard_id` 缺失，将退化为跨分片扫描。
- 如果 GraphML 缓存长期滞后，BFS 与 Web 展开会漏节点。
- 如果 `task_graph_id` 未绑定 conversation，任务图谱刷新恢复会失败。
- 如果摘要缺少图地址，RAG 命中后无法回跳 MemoryGraph 局部上下文。

## 13. 分阶段落地

### P0：修复当前可见问题

- 启动时检测 GraphML 与 LMDB 数量差异。
- 差异明显时后台重建拓扑。
- 修复 Web 会话与 `task_graph_id` 绑定。
- Web 首屏明确返回根节点 + `children_count`，避免误判。

### P1：索引增强

- 建立 `node_id -> shard_id`。
- 建立 `conversation_id -> session_node_id`。
- 建立 `task_graph_id -> shard_id/backup_path`。
- Web 展开接口改为索引定位分片。

### P2：RAG 深度融合

- 对 session、round、task、experience 生成摘要。
- 摘要写入 `DualIndexSummaryStore`。
- 摘要 metadata 保存图记忆地址。
- 冷路径命中摘要后回跳 MemoryGraph 邻域。

### P3：长期规模优化

- 摘要分片化。
- 冷分片按需 mmap。
- 跨分片语义边延迟加载。
- Web 图谱搜索分页与流式返回。

## 14. 设计决策与推荐

### 14.1 摘要粒度推荐：混合粒度

推荐采用混合摘要，而不是只按 session、round 或任务阶段单一切分。

分层粒度：

- **Round 摘要**：单轮用户输入、助手回复、关键工具调用。用于精确回忆和冷记忆图地址回跳。
- **Session 摘要**：一个会话窗口的主题、长期上下文、用户偏好和未完成事项。用于 Web 会话恢复和跨轮上下文召回。
- **Task Stage 摘要**：TaskGraph 的阶段性目标、已完成节点、失败原因、验证结果。用于复杂任务恢复。
- **Experience 摘要**：从多次任务中提炼出的可复用经验。用于跨项目/跨会话迁移。

推荐写入关系：

```text
session_summary
  → round_summary[]
  → task_stage_summary[]
  → experience_summary[]
```

LLM 上下文默认注入 Session/Task Stage 级摘要；当 LLM 需要细节时，再通过摘要中的 `graph_memory_id` 回跳到 Round 或 Task 节点做 BFS 增量回忆。

理由：

- 只按 round：检索精准，但上下文碎片化严重。
- 只按 session：上下文稳定，但细节损失明显。
- 只按任务阶段：适合任务恢复，但无法覆盖普通聊天和用户偏好。
- 混合粒度最符合 TSD 的 MemoryGraph + SummaryStore + ExperienceStore 三层闭环。

### 14.2 热数据划分：按时间为主

热数据按时间划分，遵循 TSD 与当前代码：

- 当前 `ShardedMemoryGraph.retrieve_context()` 默认 `hot_window_minutes=30`。
- TSD 中 MemoryGraph 节点温度阈值为 `hot_max=3600`、`warm_max=86400`。
- 当前 Hybrid 节点温度更新逻辑为：1 小时内 hot，24 小时内 warm，超过后 cold。

推荐策略：

- 检索热窗口默认保持 30 分钟。
- 节点温度继续保持 1 小时 hot、24 小时 warm、之后 cold。
- 访问次数、重要度、当前 session/task 匹配只作为加权，不改变时间分层本身。

### 14.3 Web 默认展示

Web 默认展示：

```text
根节点 + 最近活跃路径
```

首屏内容：

- 根级 session/task。
- 最近活跃 session。
- 当前 focus path / active node path。
- 每个根节点的 `children_count`。

这样既避免全量加载，又能让用户看到“当前记忆正在活跃哪里”。

### 14.4 RAG 回跳 BFS 深度

RAG 命中后回跳 MemoryGraph 的 BFS 深度默认保持现有设置：

- `compute_activations()` 默认 `max_depth=3`。
- `TopologyIndex` 加权 BFS 默认 `max_depth=3`。
- 当前跨分片冷路径发现局部实现中有 `max_depth=2` 的性能保护，可继续作为跨分片检索保护。

推荐：

- 单分片或已定位精确 `shard_id`：默认深度 3。
- 跨分片兜底发现：默认深度 2。
- 工具参数允许 LLM 显式请求更小深度或更小 `limit`，但不默认扩大。

### 14.5 Dialogue 与 TaskGraph 的主连接边

推荐使用 `REFERENCE` 作为 Dialogue 与 TaskGraph 的主连接边，不使用 `HIERARCHY`。

依据：

- TSD 中明确：`HIERARCHY` 用于父子关系，如 `session→round`、TaskGraph 内部层级；`REFERENCE` 用于跨类型引用，如 `dialogue↔tasks`。
- 当前 `DialogueAdapter.add_round()` 中，当存在 `task_graph_id` 时，创建 `round → task` 的 `REFERENCE` 边。
- 当前 `DialogueAdapter.bind_session_to_task()` 中，会话与任务绑定也使用 `REFERENCE` 边。
- 当前 `memory_mirror._attach_task_if_present()` 也使用 `REFERENCE` 边连接 `round → task_node`。

推荐结构：

```text
Dialogue Session
  └─ HIERARCHY → Dialogue Round
       └─ REFERENCE → TaskGraph Root
TaskGraph Root
  └─ HIERARCHY / DEPENDENCY → Task Nodes
```

原因：

- Dialogue 和 TaskGraph 是不同类型的语义对象，不是父子包含关系。
- 使用 `REFERENCE` 可以保留跨类型引用语义，避免把 TaskGraph 错误挂成会话子树。
- BFS 仍可通过 `REFERENCE` 跨到任务图谱，不影响上下文召回。

### 14.6 分片索引存储推荐：LMDB 独立索引库 + JSON 元信息

推荐采用：

```text
LMDB 独立索引库为主，JSON 仅保存低频元信息和人工可读 manifest。
```

建议结构：

```text
data/memory_graph_hybrid/
  shard_index.json              # 人工可读分片 manifest：分片列表、统计、版本
  global_index/
    index.lmdb                  # mmap / B+树 / 低延时查询
  shard_2026_05/
    topology.graphml
    properties/data.mdb
```

LMDB 独立索引库保存：

- `node_id -> shard_id`
- `session_id -> shard_id`
- `conversation_id -> session_node_id`
- `task_graph_id -> shard_id`
- `task_graph_id -> backup_path`
- `root_node_id -> shard_id`
- `active_node_id -> shard_id`
- `hot_node_id -> shard_id`

JSON 保存：

- 分片列表。
- 分片创建时间。
- 粗略节点/边数量。
- 索引版本。
- 最近一次一致性校验时间。

不推荐只用 JSON：

- 分片和节点数量极大后，JSON 需要整体读写。
- 并发写入和崩溃恢复弱。
- 查询 `node_id -> shard_id` 会退化成内存加载大字典。

不推荐 SQLite 作为主索引：

- SQLite 适合 InteractionStore 这种关系账本和审计查询。
- 对高频点查可以胜任，但与当前 Hybrid 的 LMDB mmap 设计不完全一致。
- 在“磁盘虚拟内存 + 最低加载延时”目标下，LMDB 更贴合 TSD 的骨肉分离方案。

推荐 LMDB 的原因：

- TSD 年级记忆方案明确偏向 mmap/LMDB。
- LMDB B+ 树点查适合 `node_id -> shard_id`。
- mmap 按需加载，冷读也能保持低延时。
- 读取无锁，适合 Web 展开和检索并发。
- 崩溃安全和事务语义强于 JSON。
