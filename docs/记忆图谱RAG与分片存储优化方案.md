# 记忆图谱 RAG 与分片存储优化方案

> 状态：讨论稿，供后续增量删改  
> 对齐依据：`TSD/祖龙 (ZULONG) 机器人系统技术规格说明书 (TSD)1.7.txt`  
> 范围：MemoryGraph、DualIndexSummaryStore、RAGManager、InteractionStore、TaskGraph、Web 端记忆图谱展示

## 1. 背景与问题

当前运行态已采用分片 Hybrid 记忆图谱：

- 图谱权威属性：`data/memory_graph_hybrid/*/properties`，LMDB 存储节点与边属性。
- 拓扑缓存：当前实现存在 `topology.graphml`，用于快速 BFS、邻居发现和 Web 展开。
- 分片索引：`shard_index.json`，记录分片统计。
- 会话事件账本：`data/interaction/interaction_store.sqlite3`。
- 任务图谱备份：`data/graph_backups/*.json`、`data/completed_tasks/*.json`。

已观察到的问题：

- LMDB 中真实节点数量远多于 `topology.graphml`，说明拓扑缓存可能滞后。
- Web 首屏若只读根节点，会被误解为“只有根节点，没有子节点”。
- Web 会话与 `task_graph_id` 绑定不足时，任务图谱虽已落盘，但刷新后无法稳定恢复。
- 后期记忆规模极大、分片极多时，Web 端若全量重建 MemoryGraph，会带来明显卡顿。

根据更新后的 TSD 第 21 章，长期大规模运行态应升级为“骨肉分离 + LMDB/mmap + 自定义骨架”的年级记忆方案：

- 骨架：常驻或 mmap 的紧凑二进制拓扑，用于 BFS、邻居展开、注意力路径。
- 肉：LMDB payload/properties，按需 mmap 冷加载节点/边详情。
- 向量：摘要向量/详情向量使用 mmap/FAISS 侧车。
- 分片：按时间或容量分片，冷分片按需打开。
- GraphML/JSON：仅用于调试、导出、人类可读 manifest，不作为大规模运行态主索引。

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
- 大规模运行态以二进制/内存拓扑骨架作为 BFS 主路径；`topology.graphml` 仅作为调试/导出快照，不是运行态主索引。
- `SummaryStore/RAG` 是冷数据导航层，不替代 MemoryGraph。
- `InteractionStore` 是 Web 会话恢复与审计账本，不承载完整长期语义记忆。
- Web 展示走分页、按需展开、懒加载，不返回全量图。
- 后台维护可以修复和重建索引，前台请求不得同步等待重建。
- 存储层升级不得改变任务编排、工具调用、前端交互、图谱可视化的外部协议和数据契约。

## 4. 存储分层设计

### 4.1 MemoryGraph 大规模运行态

职责：

- 存储会话、轮次、任务、工具、审批、代码锚点等图节点。
- 存储 HIERARCHY、TEMPORAL、REFERENCE、SEMANTIC、ASSOCIATION 等边。
- 支持 BFS 扩散、赫布增强、注意力路径、按需展开。

数据形态：

- 骨架层：节点 ID、节点类型、边、边权重，紧凑二进制或内存结构，负责 BFS。
- 属性层：label、content、metadata、importance、temperature、activation、backend_ref，LMDB/mmap 按需读取。
- 全局索引层：`node_id/session_id/task_graph_id -> shard_id/address`，LMDB 独立索引库。
- 导出层：GraphML/JSON 仅作调试、备份、人工检查，不参与高频读写路径。

推荐目录：

```text
data/memory_graph_hybrid/
  manifest.json                 # 人工可读：分片列表、版本、统计、校验时间
  global_index/
    index.lmdb                  # node_id/session_id/task_graph_id 等全局定位索引
  active_topology/
    skeleton.bin                # 热/活跃骨架，启动加载或 mmap
  shards/
    shard_2026_05/
      topology.bin              # 本分片紧凑拓扑骨架
      properties.lmdb           # 节点/边属性，mmap 按需冷加载
      local_index.lmdb          # 本分片局部索引
      topology.graphml          # 可选，仅调试/导出
```

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

当前 `shard_index.json` 只记录粗统计。大规模运行态采用 LMDB 独立全局索引库，JSON 退化为人工可读 manifest。

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

高频路径必须只读 LMDB 全局索引，不扫描所有分片：

- Web 展开节点。
- RAG 摘要命中后回跳图记忆地址。
- TaskGraph 刷新恢复。
- `read_memory_node` / `discover_related` 等工具调用。
- MemoryGraph 快照生成中的 active path / execution view 定位。

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
2. 从 `topology.bin` / active skeleton 读取一层 children。
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

- `topology.bin` / active skeleton 节点/边数量。
- LMDB properties 节点/边数量。
- LMDB global_index 统计。
- `manifest.json` 统计。

若差异超过阈值，标记该分片为 `topology_stale=true`。

### 8.2 后台修复

后台异步执行：

```text
LMDB properties
→ rebuild_topology_from_properties()
→ save topology.bin / active skeleton
→ 可选导出 topology.graphml
→ 更新 global_index.lmdb
→ 更新 manifest.json
→ 广播索引修复完成事件
```

前台请求处理策略：

- 若拓扑可用，继续用二进制/内存骨架。
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
- TaskGraph 根节点写入 MemoryGraph，并与 session/round 建立 `REFERENCE` 边。
- 任务完成后生成摘要，写入 SummaryStore/RAG，并保留图地址。

## 10. 跨板块兼容边界

存储升级只允许替换 MemoryGraph 的内部存储实现，不允许破坏任务编排、工具调用、前端交互和图谱可视化的外部契约。

### 10.1 对任务编排的兼容

必须保持：

- TaskGraph 仍由 `zulong/l2/task_graph.py` 负责运行态编排和 `to_frontend_dict()` 序列化。
- TaskGraph 备份仍写入 `data/graph_backups/*.json`，完成归档仍写入 `data/completed_tasks/*.json`。
- `TaskGraphAdapter` 继续把 TaskGraph 投射到 MemoryGraph。
- `task_graph_id` 仍通过 InteractionStore 和 MemoryGraph metadata 绑定 Web conversation。
- Dialogue 与 TaskGraph 的主边继续使用 `REFERENCE`。

禁止：

- 因存储升级改变 TaskGraph 节点 ID、状态值、`h_edges/d_edges` 语义。
- 因 MemoryGraph 分片改造阻塞 FC 循环或 TaskGraph 自动保存。
- 把 TaskGraph 恢复改成必须全图扫描。

### 10.2 对工具调用链路的兼容

必须保持 TSD v2.7 工具袋与工具预判链路：

```text
用户输入
→ L1-B 工具预判 + 上下文/记忆打包
→ L2 单次决策
→ 只有真实 tool_call 出现才进入工具循环
```

记忆工具接口必须稳定：

- `recall_memory`
- `read_memory_node`
- `save_memory_note`
- `discover_related`

存储升级后，这些工具只改变内部查询路径：

```text
global_index.lmdb 定位分片
→ topology.bin / skeleton 做 BFS
→ properties.lmdb 加载详情
→ 返回原有工具结果结构
```

禁止：

- 改变工具名、参数名、风险等级语义。
- 让 LLM 看到底层分片实现细节。
- 因索引修复或拓扑重建阻塞工具调用。

### 10.3 对前端交互协议的兼容

必须保持现有 WebSocket / unified protocol 消息类型：

- `tool:prediction`
- `task:plan`
- `task:summary`
- `approval:required`
- `attention:update`
- `graph:memory:diff`
- `MEMORY_GRAPH_UPDATE`
- `TASK_GRAPH_UPDATE`
- `INTERACTION_EVENT`

必须保持 `InteractionPayload` 的 7 类 kind：

- `plan`
- `action`
- `observation`
- `approval`
- `progress`
- `summary`
- `user_interject`

存储升级只影响 `MEMORY_GRAPH_UPDATE` 的数据来源，不影响消息名、字段语义和前端卡片渲染逻辑。

### 10.4 对图谱可视化的兼容

必须保持 MemoryGraph 前端接口：

- `to_frontend_dict(depth=...)`
- `get_node_children_for_frontend(node_id)`
- `get_active_node_ids()`
- `compute_activations(seed_node_ids, max_depth=3, decay=0.5)`

`/api/memory-graph/snapshot` 与 `MEMORY_GRAPH_UPDATE` 必须继续返回：

- `nodes`
- `edges`
- `stats`
- `active_node_ids`
- `thought_view`

根据更新后的 TSD §23.8.5，快照还必须显式携带 `execution_view`，用于展示工具调用、工具结果和审批事件：

```json
{
  "execution_view": {
    "nodes": [],
    "edges": [],
    "execution_node_ids": [],
    "stats": {
      "total_execution_nodes": 0,
      "tool_call": 0,
      "tool_result": 0,
      "approval": 0
    }
  }
}
```

首屏可以继续是根节点折叠视图，但不得隐藏执行事件节点的验收入口。大规模图谱必须采用渐进式加载、分页和渲染降级。

### 10.5 对任务执行记忆化的兼容

更新后的 TSD 要求任务执行事件采用独立 NodeType：

- `TOOL_CALL`
- `TOOL_RESULT`
- `APPROVAL`

存储升级必须支持这些节点类型，且同时兼容枚举模式和字符串模式查询。

图边规则必须保持：

| source | target | edge | 含义 |
|---|---|---|---|
| session | round | `HIERARCHY` | 会话包含轮次 |
| round | `TOOL_CALL` / `TOOL_RESULT` / `APPROVAL` | `HIERARCHY` | 轮次包含工具/审批事件 |
| round | task_graph/task_node | `REFERENCE` | 对话轮引用任务图 |
| task_node | `TOOL_CALL` | `DEPENDENCY` 或 `REFERENCE` | 任务节点触发工具 |
| `TOOL_CALL` | `TOOL_RESULT` | `CAUSAL` | 工具调用与结果配对 |
| `APPROVAL(request)` | `APPROVAL(decision)` | `CAUSAL` | 审批请求与决策配对 |
| `TOOL_RESULT` | file/code_symbol | `REFERENCE` | 工具结果影响文件/代码 |
| `TOOL_CALL` | next `TOOL_CALL` | `TEMPORAL` | 同一任务内的工具链顺序 |

三层闭环必须保持：

```text
InteractionStore 原始事件
→ MemoryGraph 结构化语义节点和边
→ ExperienceStore 可复用经验
```

## 11. 写入闭环

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

## 12. 后台维护任务

建议后台任务：

- 拓扑一致性校验与修复。
- Hot/Warm/Cold 状态迁移。
- 冷数据摘要生成。
- 摘要向量索引保存。
- 低价值节点审查与遗忘。
- 分片索引压缩与校验，包括 `global_index.lmdb` 与分片 properties/topology 的一致性校验。
- InteractionStore 到 MemoryGraph 的回填。

这些任务不得阻塞 Web 聊天主链。

## 13. 大规模实现细节

### 13.1 `global_index.lmdb` 写入策略

`global_index.lmdb` 是高频点查路径，但新增节点、更新会话、绑定 TaskGraph 时也会产生索引写入。LMDB 同一环境同一时刻只能有一个写事务，因此不能让业务热路径对每个节点都同步开启独立写事务。

推荐采用写缓冲 + 批量提交：

```text
业务写入 MemoryGraph
→ IndexUpdateQueue 追加索引变更
→ 后台 writer 单线程批量提交 LMDB transaction
→ 提交成功后更新内存 hot index
```

写入策略：

- 单 writer 协程/线程串行写 `global_index.lmdb`。
- 按数量或时间批量提交，例如 `100-500` 条或 `50-200ms` 一批。
- 对强一致路径提供 `flush_index_updates()`，仅在任务完成、会话切换、关闭前调用。
- Web 展开和工具读取优先查内存 hot index，再查 LMDB。
- 若 LMDB 索引尚未提交，允许通过当前进程的 pending index buffer 命中最新节点。

一致性要求：

- MemoryGraph 节点写入成功后，索引更新至少进入队列。
- 队列持久化可采用轻量 WAL，防止进程崩溃后索引丢失。
- 启动时若发现 WAL 未清空，先 replay，再开放 Web 请求。
- 索引写失败不得阻断 L2 回复，但必须标记 `index_degraded=true` 并触发后台修复。

### 13.2 `topology.bin` 格式建议

`topology.bin` 需要服务 BFS、邻居展开和注意力路径，目标是 mmap 友好、可增量更新、读路径尽量零拷贝。

推荐第一阶段采用自定义数组结构，而不是直接引入 Cap'n Proto / FlatBuffers 作为主拓扑：

- BFS 需要连续数组和邻接表，CSR/CSC 结构比通用对象序列化更合适。
- Cap'n Proto / FlatBuffers 适合稳定对象读取，但动态图增量更新仍需要额外索引和重写策略。
- 自定义结构更贴合 TSD 第 21 章的“自定义骨架 + LMDB/mmap”方向。

建议格式：

```text
Header
  magic/version/schema_version
  node_count/edge_count
  node_table_offset
  edge_table_offset
  out_index_offset
  in_index_offset
  string_table_offset

NodeTable[]
  node_int_id
  node_type
  shard_local_id
  flags
  label_ref/hash

EdgeTable[]
  src_int_id
  dst_int_id
  edge_type
  weight
  flags

OutIndex[] / InIndex[]
  node_int_id -> edge range

StringTable / IdMap
  compact int id <-> node_id
```

增量更新策略：

- 热写入先追加到 delta log：`topology_delta.log`。
- 读路径合并 `base topology.bin + delta overlay`。
- 后台 compactor 定期把 delta 合并回新的 `topology.bin.tmp`，校验后原子替换。
- 单次业务写入不得全量重写大分片 topology。

当 delta 过大时触发合并：

- delta 边数超过 base 的 5%-10%。
- delta 文件超过阈值，例如 32-128MB。
- 分片从 hot 迁移到 warm/cold 前。
- 系统空闲窗口。

### 13.3 跨分片边存储与发现

跨分片边不能只散落在各分片本地拓扑中，否则跨冷分片 BFS 会退化为多分片扫描。

推荐新增跨分片边索引：

```text
global_index/
  cross_edges.lmdb
```

索引内容：

- `src_node_id -> [(dst_node_id, dst_shard_id, edge_type, weight)]`
- `dst_node_id -> [(src_node_id, src_shard_id, edge_type, weight)]`
- `edge_type -> edge ids` 可选，用于 SEMANTIC/REFERENCE/CAUSAL 过滤。

存储规则：

- 分片内边写入本地 `topology.bin`。
- 跨分片边写入 `cross_edges.lmdb`，并在源/目标分片 local index 中保留轻量引用。
- BFS 默认先走本分片拓扑；到边界节点时查询 `cross_edges.lmdb`。
- 跨分片 BFS 必须有 `max_cross_shards` 和 `max_nodes` 限制。

建议限制：

- 普通 RAG 回跳：最多跨 2 个分片。
- 当前 session/task 恢复：最多跨 3 个分片。
- 全局探索工具：需要显式工具调用并分页返回。

### 13.4 冷分片按需打开策略

冷分片按需 mmap 打开不能粗暴地直接打开完整 `properties.lmdb`。`lmdb.open()` 和大 map size 会有管理开销，分片多时需要分级唤醒。

推荐分级打开：

```text
Level 0: manifest.json / global_index.lmdb 常驻
Level 1: local_index.lmdb 小索引按需打开
Level 2: topology.bin mmap 打开，做局部 BFS
Level 3: properties.lmdb 打开，加载最终 TopK 节点详情
```

冷分片唤醒流程：

1. `global_index.lmdb` 定位候选 shard。
2. 打开或复用 `local_index.lmdb`，确认节点/摘要存在。
3. 需要扩散时 mmap `topology.bin`。
4. BFS 得到 TopK 节点后，才打开 `properties.lmdb` 加载详情。

缓存策略：

- LRU 管理已打开分片句柄。
- 热分片常驻。
- Warm 分片保留 local index 和 topology，properties 延迟打开。
- Cold 分片默认只保留 manifest/global index 命中能力。
- 后台预热当前 session/task 相关分片。

### 13.5 摘要向量分片化

当摘要量达到百万级，单个 FAISS 内存索引会成为瓶颈。摘要向量需要与 MemoryGraph 分片策略协同。

推荐：

- 热摘要：小型内存 FAISS Flat/HNSW。
- Warm 摘要：按月/主题分段 FAISS。
- Cold 摘要：IVF/PQ 或磁盘索引，按 shard/topical cluster 加载。

检索流程：

```text
query
→ 热摘要索引 topK
→ global summary routing index 粗筛候选分片/主题
→ 并行查询 2-5 个候选分段 FAISS
→ 融合排序
→ 命中摘要携带 graph_memory_id/shard_id
→ 回跳 MemoryGraph
```

摘要 metadata 必须包含：

- `summary_id`
- `graph_memory_id`
- `shard_id`
- `full_path`
- `time_range`
- `topic_cluster`
- `status` = hot/warm/cold
- `source_node_ids`

### 13.6 Active Skeleton 生命周期

`active_topology/skeleton.bin` 是热/活跃骨架，不应无限膨胀成全量拓扑，也不能过小导致频繁冷分片 fallback。

进入 active skeleton 的条件：

- 当前 session 根节点及 3 跳邻域。
- 当前 active TaskGraph。
- 最近 30 分钟热路径命中节点。
- 最近被 RAG 摘要回跳命中的节点及 1-2 跳邻域。
- 高重要度节点：`IMPORTANT`、`FACT`、`IDENTITY`、`MUST_REMEMBER`。

剔除条件：

- 超过热窗口且访问次数低。
- 不在当前 session/task 相关邻域。
- 非高重要度节点。
- active skeleton 超过内存预算。

维护策略：

- 使用 LRU + importance 加权淘汰。
- 每次检索只更新激活计数，不同步重写 skeleton。
- 后台定期生成新的 skeleton 快照并原子替换。
- 保留 delta overlay，避免频繁全量重建。

建议预算：

- 桌面端初始预算：50k-200k 节点骨架。
- 嵌入式/低内存端：10k-50k 节点骨架。
- 超出预算时保留当前 session/task、执行事件节点、重要记忆节点。

### 13.7 分片大小阈值与当前落地策略

分片阈值不能只看节点数。TSD 第 21 章给出的长期量级是 `skeleton.bin ~15MB`、`payloads.lmdb ~2GB@10年`，并可按年拆成约 `200MB/片`。结合当前代码的月度分片和 Hybrid LMDB 属性库，运行态采用“节点数 + 物理大小”双阈值。

当前兼容期默认配置：

```yaml
memory:
  hybrid_storage:
    global_index_map_size_mb: 256
    local_index_map_size_mb: 64
    max_nodes_per_shard: 150000
    max_shard_property_mb_warning: 150
    max_shard_property_mb_split: 200
    max_shard_topology_mb_warning: 64
    max_shard_topology_delta_mb_compact: 32
    max_active_skeleton_nodes: 50000
    shard_size_check_interval_nodes: 100
```

触发规则：

- 节点数达到 `max_nodes_per_shard * 95%`：预警。
- 节点数达到 `max_nodes_per_shard * 110%`：触发自动分裂。
- `properties` LMDB 实际使用量达到 `150MB`：预警。
- `properties` LMDB 实际使用量达到 `200MB`：触发自动分裂。
- `topology.graphml` 达到 `64MB`：预警，提示进入 `topology.bin` 迁移窗口。

当前代码仍保留 `topology.graphml`，所以 `max_shard_topology_mb_warning` 只做风险提示，不把 GraphML 大小作为自动分裂主条件。进入 P3 后，阈值应改为 `topology_delta.log` 和 `topology.bin` 的大小。

自动分裂兼容策略：

- 父分片继续保留历史节点。
- 新写入路由到最新 `parent_part_N` 子分片。
- 同一分片内的边继续按源/目标节点共同所在分片写入。
- 跨分片边写入 `global_index/index.lmdb` 中的 `cross_edges_by_src` / `cross_edges_by_dst`，保证 `HIERARCHY` / `REFERENCE` 在 Web 展开、BFS 和前端快照中不断链。
- 跨分片边数量单独计入 `cross_edge_count`，不重复计入源/目标分片的本地 `edge_count`。
- 旧版 `shard_index.json.cross_edges` 仅作为迁移兜底，启动时自动写入 LMDB 跨边索引；迁移成功后清空 JSON 中的跨边列表，仅保留 `legacy_cross_edge_count` 和迁移时间戳。

这样可以先防止单分片无限膨胀，又不改变 MemoryGraph 外部 API、Web 快照结构、工具 schema 和 TaskGraph 编排逻辑。

## 14. 性能边界

目标：

- Web 首屏：只读索引，避免全图扫描。
- 单节点展开：个位数到几十毫秒，取决于 children 数量。
- 热路径检索：几十毫秒级。
- 冷路径检索：百毫秒级以内，允许异步补充。
- 拓扑修复：后台执行，可耗时，不阻塞用户操作。
- 点查定位：通过 `global_index.lmdb` 达到毫秒级或亚毫秒级。
- 冷节点加载：通过 LMDB/mmap 按需读取，目标个位数毫秒内。

风险：

- 如果 `node_id -> shard_id` 缺失，将退化为跨分片扫描。
- 如果二进制骨架或 GraphML 导出长期滞后，BFS 与 Web 展开会漏节点。
- 如果 `task_graph_id` 未绑定 conversation，任务图谱刷新恢复会失败。
- 如果摘要缺少图地址，RAG 命中后无法回跳 MemoryGraph 局部上下文。
- 如果存储升级改变 `to_frontend_dict()` / `get_node_children_for_frontend()` 输出结构，前端图谱可视化会受损。
- 如果新增节点类型没有兼容枚举和字符串查询，工具执行记忆化会漏掉 `TOOL_CALL` / `TOOL_RESULT` / `APPROVAL`。

## 15. 分阶段落地

### P0：修复当前可见问题

- 兼容期启动时检测当前 GraphML 与 LMDB 数量差异。
- 差异明显时后台重建拓扑。
- 修复 Web 会话与 `task_graph_id` 绑定。
- Web 首屏明确返回根节点 + `children_count`，避免误判。
- 分片阈值配置化，新增节点数与 LMDB 物理大小双阈值保护。
- 修正自动分裂后的写入路由，避免继续写入已超限父分片。
- 保持 MemoryGraph 外部 API、工具 schema、WebSocket 消息类型不变。

当前已落地 P0 兼容子集：

- 分片加载时会比较 `topology.graphml` 拓扑节点/边数量与 LMDB properties 节点/边数量；GraphML 为空或明显少于 LMDB 时，以 LMDB 为权威自动重建当前分片拓扑缓存。
- 拓扑健康信息写入 `shard_index.json.shards[*].topology_health`，并统计 `topology_rebuild_count`。
- 已加入节点数 + LMDB properties 物理大小双阈值，超限触发预警或自动分裂。
- 自动分裂后，新写入路由到最新 `parent_part_N`，避免继续写爆父分片。
- Web 首屏节点序列化携带 `children_count`，展开接口继续按需返回子节点。
- Web/IDE/MemoryMirror 已补充 `task_graph_id` 绑定与恢复入口，刷新后优先通过索引定位图记忆地址。

兼容期说明：当前重建仍是分片加载时同步修复，适合 P0 避免旧 GraphML 缓存遮住真实 LMDB 数据。P3 后应迁移为后台 compactor 维护 `topology.bin` / active skeleton。

### P1：索引增强

- 建立 `global_index/index.lmdb`。
- 建立 `node_id -> shard_id`，并接入 `get_node()`、`has_node()`、`get_children()`、`add_edge()` 等高频路径。
- 基于节点 metadata 建立 `conversation_id -> session_node_id`、`session_id -> node_id`、`task_graph_id -> task_node_id` 的轻量二级索引。
- 保留全分片扫描兜底，并在兜底命中后回填 `global_index.lmdb`。
- Web 展开接口改为索引定位分片。

当前已落地 P1 兼容子集：

- `global_index/index.lmdb` 已作为 P1 点查层引入。
- 写入新节点时同步维护 `node_id -> shard_id`。
- 读取节点时优先查 LMDB 全局索引，索引缺失时扫描分片并回填。
- `get_total_stats()` 暴露 `global_index` 统计，用于观察索引健康度。
- 跨分片边已迁入 `global_index/index.lmdb` 的 `cross_edges_by_src` / `cross_edges_by_dst`，JSON `cross_edges` 仅保留旧数据迁移兜底。
- MemoryGraph 已提供显式恢复接口：`get_session_node_id_for_conversation()`、`get_task_node_id_for_graph()`、`get_context_seed_for_conversation()`。
- Web 已提供 `/api/memory-graph/context-seed` 和 `REQUEST_MEMORY_CONTEXT_SEED` / `GET_MEMORY_CONTEXT_SEED`，用于按 conversation/task graph 直接定位图记忆地址。
- IDE BFS 自恢复已优先使用全局索引定位 session seed，MemoryMirror 任务绑定已优先使用全局索引定位 Task 节点。

### P2：RAG 深度融合

- 对 session、round、task、experience 生成摘要。
- 摘要写入 `DualIndexSummaryStore`。
- 摘要 metadata 保存图记忆地址。
- 冷路径命中摘要后回跳 MemoryGraph 邻域。

当前已落地 P2 兼容子集：

- `recall_memory` 返回结果追加 `graph_memory_id`、`shard_id`、`full_path`、`memory_address`，不改变原有 `node_id/type/label/content/score` 字段。
- `read_memory_node`、`discover_related`、`activate_memory_network`、`list_memory` 均返回图记忆地址，并兼容 Hybrid/Sharded 后端的字符串节点类型。
- `recall_memory` 新增可选 `seed_node_id` / `graph_memory_id` 与 `expand_depth`，支持 LLM 以摘要命中的图记忆 ID 作为种子做一次增量回忆。
- `DualIndexSummaryStore` 的摘要条目新增 `graph_memory_id`、`shard_id`、`full_path`、`source_node_ids`，旧 SQLite 库通过 `ALTER TABLE` 自动补列。
- `MemoryGraph.index_summary()` / `ShardedMemoryGraph.index_summary()` 已同步写入 `DualIndexSummaryStore` 的 SQLite L1 摘要导航层，摘要 ID 采用 `graph_summary:{graph_memory_id}`，不会在对话热链路强制触发 embedding。
- `DualIndexSummaryStore.search_graph_summaries()` 已提供低延迟摘要导航检索入口，命中结果直接携带 `graph_memory_id/shard_id/full_path`，可回跳 MemoryGraph 做 BFS 增量回忆。
- `MemoryGraph.retrieve_context()` 与 `ShardedMemoryGraph.retrieve_context()` 已接入 SQLite 摘要导航检索；即使热路径没有命中，也可以返回 `source=summary_navigation` 的冷摘要结果，并携带 `graph_memory_id/shard_id/full_path/memory_address`。
- L2/Web/IDE 的记忆上下文注入会显示图记忆 ID，方便 LLM 后续调用 `read_memory_node` 或 `discover_related` 以该摘要地址为种子做 BFS 增量回忆。
- 该阶段只追加字段和可选参数，不改变工具名、必填 schema、WebSocket 消息类型、TaskGraph 编排协议和前端图谱 payload 契约。

### P3：长期规模优化

- 摘要分片化。
- 冷分片按需 mmap。
- 跨分片语义边延迟加载。
- Web 图谱搜索分页与流式返回。
- GraphML 从运行态主拓扑降级为调试/导出。
- 分片拓扑升级为 `topology.bin` / active skeleton。

当前已落地 P3 兼容子集：

- `TopologyIndex` 已支持 `topology.bin` 二进制侧车保存/加载。
- `MemoryGraphHybrid.save()` 会同时写入 `topology.bin` 与 `topology.graphml`；`load()` 默认优先读取 `topology.bin`，失败再回退 GraphML。
- `MemoryGraphHybrid` 已追加 `topology_delta.log` 增量日志；节点/边新增删除先写 delta，加载时回放 `topology.bin + delta`，保存快照时压实并清空 delta。
- 分片 usage 统计已增加 `topology_bin_bytes/topology_bin_mb` 与 `topology_delta_bytes/topology_delta_mb`，可观察二进制拓扑和增量日志体积。
- 新增 `max_shard_topology_delta_mb_compact` 配置；当 delta 超过阈值时，`ShardedMemoryGraph` 会启动后台 compactor，把 delta 合并进 `topology.bin` / GraphML 并清空日志。
- topology compactor 已改为去重队列 + 单后台 worker，并增加短静默窗口调度；失败会记录到 shard index，并最多重试 3 次。
- 新增 `active_topology/skeleton.bin` 兼容骨架；检索命中、焦点更新、BFS 激活会刷新 active 节点与一跳邻接，`to_frontend_dict()` 和 stats 会返回 `active_skeleton`。
- 新增 `max_active_skeleton_nodes` 配置，默认 `50000`；刷新 active skeleton 时按“当前中心/焦点路径优先、激活值、重要度、访问次数、最近访问、任务/会话元数据”打分，超预算时只淘汰非关键候选节点，避免 active skeleton 膨胀成全量拓扑。
- active skeleton 已增加后台防抖刷新队列；检索、焦点更新和 BFS 激活会排队刷新，减少热链路频繁同步写快照。
- 新增分片级 `local_index.lmdb` 侧车索引；新写节点会同步记录轻量节点头，旧分片可通过 `rebuild_local_index()` 重建。完整 local index 可在 global index 指向错误或节点缺失时给出确定性未命中，避免无效打开完整冷分片。
- 新增 `topology.csr` CSR/CSC mmap 读侧车；稳定分片在 delta 压实后可优先用 CSR reader 做邻居查询、边查询和加权 BFS，缺失或陈旧时自动回退 igraph。
- 摘要导航已增加 `shard_id/topic/status` 路由索引与 `route_graph_summaries()` 入口，为后续分段 FAISS/IVF/PQ 选择候选向量段。
- 当前 `topology.bin` 仍作为 igraph 拓扑缓存的二进制侧车；`topology.csr` 已承担兼容读路径，但写入和 GraphML 导出仍保留，避免破坏外部契约。
- 当前 active skeleton 仍是 Web/调试/后续迁移用的热骨架快照，还不是 BFS 主路径。

仍未完成：

- 生产级分段 FAISS/IVF/PQ 向量文件仍可在摘要路由索引之上继续增强。
- `topology.csr` 当前是读侧车，写入主路径仍由 igraph + delta 承担；完全去 igraph 化可作为后续性能专项。

## 16. 设计决策与推荐

### 16.1 摘要粒度推荐：混合粒度

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

### 16.2 热数据划分：按时间为主

热数据按时间划分，遵循 TSD 与当前代码：

- 当前 `ShardedMemoryGraph.retrieve_context()` 默认 `hot_window_minutes=30`。
- TSD 中 MemoryGraph 节点温度阈值为 `hot_max=3600`、`warm_max=86400`。
- 当前 Hybrid 节点温度更新逻辑为：1 小时内 hot，24 小时内 warm，超过后 cold。

推荐策略：

- 检索热窗口默认保持 30 分钟。
- 节点温度继续保持 1 小时 hot、24 小时 warm、之后 cold。
- 访问次数、重要度、当前 session/task 匹配只作为加权，不改变时间分层本身。

### 16.3 Web 默认展示

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

### 16.4 RAG 回跳 BFS 深度

RAG 命中后回跳 MemoryGraph 的 BFS 深度默认保持现有设置：

- `compute_activations()` 默认 `max_depth=3`。
- `TopologyIndex` 加权 BFS 默认 `max_depth=3`。
- 当前跨分片冷路径发现局部实现中有 `max_depth=2` 的性能保护，可继续作为跨分片检索保护。

推荐：

- 单分片或已定位精确 `shard_id`：默认深度 3。
- 跨分片兜底发现：默认深度 2。
- 工具参数允许 LLM 显式请求更小深度或更小 `limit`，但不默认扩大。

### 16.5 Dialogue 与 TaskGraph 的主连接边

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

### 16.6 分片索引与运行态存储推荐

推荐采用：

```text
LMDB 全局索引库 + LMDB 分片属性库 + topology.bin/active skeleton。
JSON 仅保存低频元信息和人工可读 manifest。
GraphML 仅作为可选调试/导出格式。
```

建议结构：

```text
data/memory_graph_hybrid/
  manifest.json                 # 人工可读分片 manifest：分片列表、统计、版本、校验时间
  global_index/
    index.lmdb                  # mmap / B+树 / 低延时查询
  active_topology/
    skeleton.bin                # 热/活跃骨架，BFS 主路径
  shards/
    shard_2026_05/
      topology.bin              # 本分片紧凑拓扑
      properties.lmdb           # 节点/边属性，mmap 按需读取
      local_index.lmdb          # 本分片局部索引
      topology.graphml          # 可选，仅调试/导出
```

LMDB 全局索引库保存：

- `node_id -> shard_id`
- `session_id -> shard_id`
- `conversation_id -> session_node_id`
- `task_graph_id -> shard_id`
- `task_graph_id -> backup_path`
- `root_node_id -> shard_id`
- `active_node_id -> shard_id`
- `hot_node_id -> shard_id`

JSON manifest 保存：

- 分片列表。
- 分片创建时间。
- 粗略节点/边数量。
- 索引版本。
- 最近一次一致性校验时间。

`topology.bin` / active skeleton 保存：

- 节点 ID 紧凑映射。
- 节点类型轻量标签。
- 出入边邻接表。
- 边类型和边权。
- BFS 所需的最小字段。

`properties.lmdb` 保存：

- label/content/summary。
- metadata。
- importance/temperature/activation。
- backend_ref/full_path。
- 工具调用、审批、任务执行节点详情。

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

GraphML 的定位：

- 保留为导出、诊断、可视化调试格式。
- 不作为超大规模运行态主拓扑。
- 不参与高频 BFS、节点展开、RAG 回跳和 TaskGraph 恢复。
