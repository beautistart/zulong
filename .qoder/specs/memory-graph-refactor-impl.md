# MemoryGraph 记忆架构改造 - 实现规划

## Context

祖龙系统的 MemoryGraph 当前处于"只写不读"状态——LLM 上下文注入完全依赖旧系统（conversation_history / STM / EpisodicMemory / RAG），MemoryGraph 仅在推理结束后写入，从未被查询。同时所有节点统一 24h 半衰期，无长期记忆能力，重要冷数据面临被误剪风险。

本改造将 MemoryGraph 升级为完整记忆生命周期管理系统：通过多维标签（温度/重要度）实现灵活的记忆分类，基于标签互斥过滤的并行检索（热数据遍历 + 冷数据 FAISS）实现高效查询，最终切换读写主通道到 MemoryGraph。

**已完成**: `Importance`/`Temperature` 枚举和 `_IMPORTANCE_ORDER`/`_IMPORTANCE_HALF_LIFE`/`_TEMPERATURE_THRESHOLDS` 常量已添加到 memory_graph.py 第 76-118 行。

**任务文档**: `docs/记忆架构改造任务文档.md`（12 个任务，5 个阶段）

---

## 执行顺序

```
阶段 1: 任务1 → 任务2 → 任务3  (基础设施，严格顺序)
阶段 2: 任务5 → 任务6           (检索能力，严格顺序)
阶段 3: 任务4 | 任务7 | 任务12  (安全守卫，可并行)
阶段 4: 任务8 | 任务9           (外部集成，可并行)
阶段 5: 任务10 → 任务11         (切换主通道，严格顺序)
```

---

## 阶段 1 - 基础设施

### 任务 1: GraphNode 多维标签体系

**文件**: `zulong/memory/memory_graph.py`

**1a. 修改 `add_node()` (行 264-295)**
- 新节点分支（行 287）: `graph.add_node()` 前为 metadata 设默认值:
  ```python
  node.metadata.setdefault("temperature", Temperature.HOT.value)
  node.metadata.setdefault("importance", Importance.NORMAL.value)
  ```
- 更新分支（行 272）: 不覆盖已有标签

**1b. 新增 5 个辅助方法（在节点 CRUD 区块后，约行 328 后）**:

| 方法 | 签名 | 逻辑 |
|------|------|------|
| `get_temperature` | `(node_id) -> Optional[Temperature]` | 动态计算: `now - last_accessed` 与 `_TEMPERATURE_THRESHOLDS` 比较。不读 metadata，实时算 |
| `get_importance` | `(node_id) -> Optional[Importance]` | 读 `metadata.get("importance", "normal")`，返回 `Importance(值)`，try/except 容错 |
| `set_importance` | `(node_id, importance: Importance) -> bool` | 写 `metadata["importance"]`，调 `_mark_dirty()` |
| `update_temperature` | `(node_id) -> Optional[Temperature]` | 同 `get_temperature()` 但同步写入 `metadata["temperature"]`（前端展示用） |
| `is_recent` | `(node_id, window_seconds=1800) -> bool` | `now - last_accessed < window_seconds` |

**关键决策**: 温度标签由 `last_accessed` 实时计算（符合"时间段标签查询时动态计算"约束），metadata 中的 temperature 仅为缓存/前端展示。重要度存 metadata，持久化自动兼容。

---

### 任务 2: 差异化衰减 decay_and_prune() 改造

**文件**: `zulong/memory/memory_graph.py`

**2a. 修改边衰减逻辑（行 675-693 的循环体）**:
```
原: decayed = weight * exp(-elapsed / 24)
改: 
  1. imp_src, imp_tgt = get_importance(src), get_importance(tgt)
  2. higher = max(imp_src, imp_tgt, key=_IMPORTANCE_ORDER.get)
  3. half_life = _IMPORTANCE_HALF_LIFE[higher]
  4. 若 half_life == inf: 跳过衰减 + 设 protected=True
  5. decayed = weight * exp(-elapsed * ln(2) / half_life)
```

**2b. 修改孤立节点清理（行 696-713）**:
```
原: 统一 86400s (24h) 容忍
改:
  - importance >= IDENTITY → 永不因孤立删除
  - TRIVIAL → 6h (21600s)
  - NORMAL → 24h (86400s)
  - IMPORTANT/FACT → 7天 (604800s)
  - TASK 类型永不删除（保留原逻辑）
```

**2c. 在衰减循环中顺便更新温度缓存**:
- 遍历所有节点时，调 `update_temperature(node_id)` 更新 metadata

---

### 任务 3: 重要度动态提升

**文件**: `zulong/memory/memory_graph.py`

**3a. 新增 `promote_importance(node_id, target: Importance) -> bool`**:
- 只升不降: `_IMPORTANCE_ORDER[target] > _IMPORTANCE_ORDER[current]` 才执行
- 提升为 MUST_REMEMBER 时自动设所有关联边 `protected=True`
- 记录 `metadata["importance_history"]` 列表

**3b. 新增 `run_importance_review() -> Dict[str, int]`**:
- `access_count >= 3 且 NORMAL` → 自动提升 IMPORTANT
- `access_count >= 5 且 IMPORTANT` → 标记为 LLM 审查候选（不直接调 LLM）
- 返回 `{"auto_promoted": N, "llm_candidates": M}`

**3c. 修改 `start_prune_loop()` (行 720)**:
- 每轮 `decay_and_prune()` 后调 `run_importance_review()`

---

## 阶段 2 - 检索能力

### 任务 5: FAISS 摘要侧车索引

**文件**: `zulong/memory/memory_graph.py`

**5a. 新增 `SummarySidecarIndex` 类（MemoryGraph 类定义前，约行 185）**:

```python
class SummarySidecarIndex:
    def __init__(self, dimension=512, persist_path="")
    def add_summary(self, node_id: str, summary_text: str) -> bool
    def search(self, query_text: str, top_k=5, exclude_ids=None) -> List[Tuple[str, float]]
    def remove(self, node_id: str) -> bool
    def save(self, path: str) -> bool
    def load(self, path: str) -> bool
```

- 内部复用 `FAISSVectorStore(dim=512, index_type="Flat")` (来自 `base_rag_library.py`)
- 使用 `EmbeddingModelManager().encode_document()` 生成 512 维向量（来自 `embedding_manager.py`）
- 只索引 Session/EPISODE 摘要向量 + node_id 指针，不向量化节点实际内容
- 维护 `node_id_to_faiss_id` 和 `faiss_id_to_node_id` 双向映射

**5b. MemoryGraph.__init__() 中初始化**:
- `self._summary_index = SummarySidecarIndex(persist_path=persist_path)`

**5c. 索引维护触发点**:
- `remove_node()` 中同步删除 FAISS 条目
- `save()` / `_load()` 中保存/加载 FAISS 文件（独立于 JSON）

**5d. 持久化文件**:
- `{persist_path}/summary_sidecar.index` — FAISS 索引
- `{persist_path}/summary_sidecar.maps.json` — ID 映射

---

### 任务 6: 并行检索策略

**文件**: `zulong/memory/memory_graph.py`

**6a. 新增核心方法**:

```python
async def retrieve_context(self, query_text: str, top_k=10, hot_window_minutes=30) -> List[Dict]
async def _retrieve_hot(self, query_text: str, window_minutes: int) -> List[Dict]
async def _retrieve_cold(self, query_text: str, top_k: int) -> List[Dict]
```

**6b. 路径 A (热数据遍历)**:
- 筛选 `is_recent(node_id, window_seconds=hot_window_minutes*60) == True` 的节点
- 关键词匹配 + BFS 扩散（`compute_activations`），过滤非热节点
- NetworkX 操作是同步的，用 `run_in_executor()` 包装为 async

**6c. 路径 B (冷数据 FAISS)**:
- `_summary_index.search(query_text, top_k)`
- 过滤掉热数据（`is_recent() == True` 的结果丢弃）
- BFS 下钻获取子节点详情

**6d. 合并**:
- `asyncio.gather(path_a, path_b)` 并行执行
- 两路结果合并，按 activation 降序排序，截取 top_k
- 返回格式: `[{"node_id", "node_type", "label", "content", "score", "source"}]`

---

## 阶段 3 - 安全守卫

### 任务 4: LLM 剪枝守卫

**文件**: `zulong/memory/memory_graph.py`, `zulong/memory/llm_memory_reviewer.py`

- 修改 `decay_and_prune()`: 边权 0.05-0.15 且 importance != TRIVIAL → 加入 `pending_review` 列表暂不删除
- 新增 `_submit_prune_review()` 异步提交 `LLMMemoryReviewer.review_before_evict()`
- 新增 `process_review_callback()` 处理结果: KEEP→reinforce+promote, DISCARD→删除, COMPRESS→摘要
- 适配 `_build_evict_prompt_template()` 为图节点格式

### 任务 7: 重要信息自动检测

**文件**: `zulong/memory/graph_adapters.py`

- 在 `DialogueAdapter` 中新增 `_IMPORTANCE_RULES` 正则规则表和 `_detect_importance(text) -> Tuple[Importance, List[str]]`
- 修改 `add_round()` (行 495-549): 创建节点后调 `_detect_importance(goal)` 设初始 importance
- 修改 `add_sub_dialogue()` (行 551-604): 对 content 调 `_detect_importance()` 设 importance
- 规则: "我叫"→IDENTITY, "帮我记住"→MUST_REMEMBER, "我的电话"→FACT, "嗯/好/哦"→TRIVIAL 等

### 任务 12: 空节点清理

**文件**: `zulong/memory/memory_graph.py`

- 新增 `cleanup_orphan_nodes() -> Dict[str, int]`
- 语义孤立检测: 仅剩 TEMPORAL/HIERARCHY 边的节点
- 分级: trivial/normal+cold→标记可丢弃, importance>=identity→保留为沉睡记忆
- 在 `start_prune_loop` 中每 3 轮修剪后调一次

---

## 阶段 4 - 外部集成

### 任务 8: ExperienceRAG 被动化

**文件**: `zulong/l2/inference_engine.py`, 新建 `zulong/tools/experience_rag_tool.py`, `zulong/tools/tool_engine.py`

- 新建 `ExperienceRAGTool(BaseTool)`: name="search_experience", 参数 query/top_k/domain
- 在 `tool_engine.py` 的 `_register_builtin_tools()` 中注册
- 修改 `inference_engine.py` 的 `_retrieve_from_rag()` (行 2044): 直接 `return None`（短路，取消自动注入）

### 任务 9: KnowledgeRAG 打通

**文件**: `zulong/memory/memory_graph.py`, `zulong/memory/graph_adapters.py`

- 新增 `resolve_backend_ref(node_id) -> Optional[Dict]`: 按 backend_ref 前缀路由到 KnowledgeRAG/ExperienceRAG
- 在 `retrieve_context()` BFS 下钻到 KNOWLEDGE 节点时调 `resolve_backend_ref()` 获取完整内容
- 确保 `KnowledgeGraphAdapter.sync()` 正确设置 backend_ref 格式

---

## 阶段 5 - 切换主通道

### 任务 10: 记忆注入通道切换

**文件**: `zulong/l2/inference_engine.py`

- 替换 `_build_messages_with_history_async()` 中行 2500-2664 的旧 3 层记忆检索
- 改为: `memory_results = await mg.retrieve_context(user_input, top_k=10)`
- 按 node_type 格式化: DIALOGUE→【历史对话】, TASK→【相关任务】, KNOWLEDGE→【知识参考】, EPISODE→【历史摘要】
- 保留视觉/工具/搜索上下文注入逻辑不变
- 加 feature flag `USE_MEMORY_GRAPH` 控制灰度切换
- MemoryGraph 为空时 fallback 到 `conversation_history[-4:]`

### 任务 11: 记忆写入通道统一

**文件**: `zulong/l2/inference_engine.py`

- 替换 `_update_memory_async()` 中行 2712-2744 的 STM/EpisodicMemory 写入
- 改为: `DialogueAdapter.ensure_session()` + `add_round()` 写入 MemoryGraph
- AI 回复写入 round 节点 `metadata["ai_response"]` 和 `metadata["content"]`
- 保留 `conversation_history`（供经验生成器和 fallback 使用）
- 新增 `_last_round_id`/`_current_session_id` 实例变量跟踪对话状态

---

## 关键文件清单

| 文件 | 涉及任务 | 变更类型 | 预估新增行数 |
|------|---------|---------|------------|
| `zulong/memory/memory_graph.py` | 1,2,3,4,5,6,9,12 | 重度修改 | ~500 行 |
| `zulong/memory/graph_adapters.py` | 7,9 | 中度修改 | ~80 行 |
| `zulong/l2/inference_engine.py` | 8,10,11 | 中度修改 | ~100 行 (替换~300行) |
| `zulong/tools/experience_rag_tool.py` | 8 | 新建 | ~80 行 |
| `zulong/tools/tool_engine.py` | 8 | 轻微修改 | ~5 行 |
| `zulong/memory/llm_memory_reviewer.py` | 4 | 轻微修改 | ~20 行 |

## 向后兼容保障

- 所有新标签存入 `GraphNode.metadata` dict，`to_dict()`/`from_dict()` 自动序列化，无需改 JSON schema
- 旧 JSON 数据无 importance 字段时，`get_importance()` 返回 NORMAL（24h 半衰期），行为不变
- 温度标签由 `last_accessed` 实时计算，不依赖持久化值
- FAISS 索引文件独立于 `memory_graph.json`，加载失败不影响图谱功能
- 任务 10 用 feature flag 控制，可灰度切换

## 验证方案

1. **任务 1-3 验证**: 创建测试节点，验证默认标签、差异化衰减速率、自动提升逻辑
2. **任务 5 验证**: 添加摘要到 FAISS 索引，验证搜索返回正确 node_id
3. **任务 6 验证**: 构造热+冷节点场景，验证并行检索的正确性和互斥过滤
4. **任务 7 验证**: 输入"我叫小明"，验证节点自动标记为 identity
5. **任务 10-11 验证**: 端到端对话测试，验证 MemoryGraph 读写通道工作正常
6. **兼容性验证**: 加载旧 JSON 数据，确认不报错且旧节点默认 NORMAL/24h
