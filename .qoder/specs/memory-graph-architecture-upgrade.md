# 记忆图谱 (Memory Graph) 架构升级规划

## Context

### 问题

祖龙系统当前的记忆架构存在根本性的 **结构性割裂** 问题：

1. **孤岛式记忆**: ShortTermMemory、EpisodicMemory、KnowledgeGraph、RAG Libraries、PersonProfile 各自独立运行，彼此无连接。一个对话记忆无法自动关联到相关的知识实体或历史任务。

2. **一维注意力**: AttentionWindowManager 在消息序列上做 `weight = base x time_decay x mode_multiplier` 的线性评分，只能按"时间远近 + 工具关联"来筛选消息。无法发现跨类型的语义关联（如：当前任务依赖的知识点出现在30轮前的对话中）。

3. **任务图谱隔离**: TaskGraph 仅存在于 pipeline/ 内的 Agent 循环中，与记忆系统完全无交集。任务完成后的经验无法自动回流到记忆网络。

### 目标

构建 **记忆图谱 (MemoryGraph)** -- 一个统一的异构类型图，将所有记忆子系统的数据投射为图节点和图边，实现：

- 注意力从"在消息列表上滑动"升级为"在图上做加权 BFS 扩散"
- 任务图谱成为记忆图谱的一个子图层级
- 边权通过赫布学习自动增强（共激活=共强化），通过艾宾浩斯衰减自动修剪
- 从一维线性记忆跃升为三维立体结构化记忆

### 核心范式转换：上下文问题 -> 检索问题

记忆图谱的本质是将 **上下文窗口限制** 转化为 **图检索问题**：

- **传统做法**: 模型必须"同时看到"所有相关信息 -> 受限于 context window 大小
- **图谱做法**: 模型只需看到"当前最相关的子图" -> 图负责发现关联，模型负责推理

这意味着：
1. **有效无限上下文**: 图可存储任意规模的结构化知识，100万token的信息只需在当前窗口内精确投射 3-4K 关键token
2. **任意节点 -> 全局注意**: 从任何一个节点出发，沿着依赖/引用/语义边做 BFS 扩散，即可追溯到全图中所有相关节点
3. **对模型大小不敏感**: 图检索的质量由图结构和边权决定，不依赖模型参数量。模型越大推理质量越高，但即使小模型也能通过图获得正确的上下文输入
4. **对数据量不敏感**: 图的 BFS 扩散有 max_depth + 阈值剪枝，检索复杂度与总数据量解耦

类比人脑：人的工作记忆容量只有 7+-2 个 chunk，但通过联想网络（海马体）可以从任意线索追溯到整个长期记忆网络。记忆图谱就是祖龙的"海马体"。

### 设计原则

- **适配器模式，不替换**: MemoryGraph 是现有模块之上的集成层，各后端模块保持不变
- **最终一致性**: 图是后端数据的投影，后端仍为数据源头，图可从后端重建
- **渐进式启用**: 每个阶段独立可用，图注意力可通过开关回退到1D模式
- **异步优先**: 所有操作 async/await，匹配现有架构风格
- **模型无关**: 架构不针对特定模型大小优化，对 0.8B 到 100B+ 均适用
- **可扩展节点/边类型**: NodeType/EdgeType 设计为可扩展枚举，未来新增文档/知识等类型无需修改核心逻辑

---

## 架构总览

```
+-----------------------------------------------------+
|                   消费层 (Consumers)                  |
|  AttentionWindowManager | AgentOrchestrator           |
|  L1-B Gatekeeper        | Review/Replay System        |
+-------------+-------------------------------------------+
              | query / traverse / activate
+-------------v-------------------------------------------+
|              MemoryGraph (新增集成层)                    |
|  +------------------------------------------------+    |
|  |  NetworkX DiGraph (异构类型图)                    |    |
|  |  - GraphNode (类型: task/dialogue/knowledge/...) |    |
|  |  - GraphEdge (类型: hierarchy/semantic/causal/.)|    |
|  |  - 边权: 赫布学习 + 艾宾浩斯衰减                  |    |
|  +------------------------------------------------+    |
|  +----------------+ +-----------+ +-----------------+   |
|  | FAISS 语义索引  | | 异步修剪器 | | JSON 持久化      |   |
|  | (512-dim)       | | (30min)   | |                 |   |
|  +----------------+ +-----------+ +-----------------+   |
+-------------+-------------------------------------------+
              | adapters (只读适配器)
+-------------v-------------------------------------------+
|           现有后端 (保持不变)                              |
|  ShortTermMemory | EpisodicMemory  | KnowledgeGraph      |
|  RAG Libraries   | SummaryStore    | PersonProfile        |
|  VectorCache     | MemoryEvolution | TaskGraph (pipeline) |
+-----------------------------------------------------+
```

---

## 数据模型

### GraphNode

```python
@dataclass
class GraphNode:
    node_id: str           # 全局唯一, 带类型前缀: "task:o1_1", "dialogue:42", "kg:person_mom"
    node_type: NodeType    # TASK | DIALOGUE | KNOWLEDGE | EXPERIENCE | EPISODE | FILE | CONCEPT | PERSON
    label: str             # 人类可读标签
    embedding: Optional[np.ndarray]  # 512-dim (BAAI/bge-small-zh-v1.5)
    activation: float      # 当前激活水平 (0.0-1.0), 随时间衰减
    created_at: float
    last_accessed: float
    access_count: int
    backend_ref: str       # 后端来源指针, 如 "stm:turn_42", "kg:entity_person_mom"
    metadata: Dict[str, Any]
```

### GraphEdge (存储为 NetworkX 边属性)

```python
# edge_data on nx.DiGraph edges
{
    "edge_type": EdgeType,  # HIERARCHY | DEPENDENCY | REFERENCE | TEMPORAL | SEMANTIC | CAUSAL | ASSOCIATION
    "weight": float,        # 0.0-1.0+, 赫布增强 + 衰减
    "created_at": float,
    "last_activated": float,
    "activation_count": int,
    "protected": bool,      # True = 永不修剪 (结构性边)
    "metadata": Dict
}
```

### 类型枚举

```python
class NodeType(Enum):
    TASK = "task"           # 来自 TaskGraph.TaskNode
    DIALOGUE = "dialogue"   # 来自 ShortTermMemory 对话轮次
    KNOWLEDGE = "knowledge" # 来自 KnowledgeGraph.Entity
    EXPERIENCE = "experience" # 来自 ExperienceRAG 文档
    EPISODE = "episode"     # 来自 EpisodicMemory 摘要
    FILE = "file"           # 来自 TaskNode.files
    CONCEPT = "concept"     # 来自 KG 中 entity_type=CONCEPT
    PERSON = "person"       # 来自 PersonProfile / KG中PERSON实体
    DOCUMENT = "document"   # 预留: 未来文档/知识切片摄入

class EdgeType(Enum):
    HIERARCHY = "hierarchy"     # 父子关系 (task h_edges)
    DEPENDENCY = "dependency"   # 数据依赖 (task d_edges)
    REFERENCE = "reference"     # 跨类型引用 (task->file, dialogue->knowledge)
    TEMPORAL = "temporal"       # 时间序列 (dialogue->dialogue)
    SEMANTIC = "semantic"       # 语义相似 (embedding cosine > 0.7)
    CAUSAL = "causal"          # 因果关系 (KG中 CAUSED 关系)
    ASSOCIATION = "association" # 赫布学习产生的关联
```

---

## 分阶段实施计划

### 阶段 1: 基础层 -- 核心图数据结构

**目标**: 创建 MemoryGraph 类，实现基础 CRUD、BFS 遍历、持久化

**新建文件**:
- `zulong/memory/memory_graph.py` (~400行)

**修改文件**:
- `zulong/memory/__init__.py` -- 添加导出

**实现内容**:

```python
class MemoryGraph:  # 单例, __new__ + _initialized 模式 (参考 knowledge_graph.py)
    _graph: nx.DiGraph
    _persist_path: str  # "./data/memory_graph/graph.json"

    # --- CRUD ---
    add_node(node: GraphNode) -> str
    get_node(node_id: str) -> Optional[GraphNode]
    remove_node(node_id: str) -> bool
    update_node_activation(node_id: str, activation: float)
    
    add_edge(source_id: str, target_id: str, edge_type: EdgeType, weight: float, protected: bool = False) -> bool
    get_edge(source_id: str, target_id: str) -> Optional[Dict]
    remove_edge(source_id: str, target_id: str) -> bool
    
    # --- 查询 ---
    get_neighbors(node_id: str, edge_types: Optional[Set[EdgeType]] = None, max_depth: int = 1) -> List[GraphNode]
    get_nodes_by_type(node_type: NodeType) -> List[GraphNode]
    
    # --- BFS 扩散激活 (核心算法) ---
    compute_activations(seed_node_ids: List[str], max_depth: int = 3, decay: float = 0.5) -> Dict[str, float]
    
    # --- 持久化 ---
    save() -> bool       # JSON (embeddings base64编码)
    _load() -> bool
    
    # --- 统计 ---
    stats -> Dict[str, Any]
```

**BFS 扩散激活算法**:
```
输入: seed_nodes (种子节点列表), max_depth=3, decay=0.5
输出: Dict[node_id -> activation_score]

1. 初始化: 每个种子节点 activation = 1.0, 加入队列
2. BFS 循环:
   - 取出 (node, depth, act)
   - 如果 depth >= max_depth: 跳过
   - 遍历所有邻居 (出边 + 入边, 视为无向传播):
     - propagated = act x edge_weight x decay
     - 如果 propagated < 0.01: 剪枝
     - 如果 neighbor 未访问 或 新值更大: 更新并入队
3. 返回所有节点的激活值
```

预期性能: 1000节点图 <5ms, 因为 max_depth=3 + 0.01阈值剪枝

**持久化格式**:
```json
{
  "version": "1.0",
  "nodes": { "task:o1_1": { "node_type": "task", "label": "...", ... } },
  "edges": [ { "source": "...", "target": "...", "edge_type": "...", "weight": 0.8, ... } ],
  "meta": { "saved_at": 1234567890, "node_count": 42, "edge_count": 78 }
}
```

**风险**: 低。纯新增代码，不触及现有模块。

---

### 阶段 2: 后端适配器 -- 数据投射入图

**目标**: 为每个现有记忆模块创建适配器，将数据投射为 GraphNode + GraphEdge

**新建文件**:
- `zulong/memory/graph_adapters.py` (~350行)

**修改文件**:
- `zulong/memory/memory_graph.py` -- 添加 `register_adapter()`, `sync_all()`, `sync_adapter(name)`

**6个适配器**:

| 适配器 | 数据源 | 节点类型 | 边类型 |
|--------|--------|---------|--------|
| TaskGraphAdapter | TaskGraph._nodes, _h_edges, _d_edges | TASK, FILE | HIERARCHY(protected), DEPENDENCY(protected), REFERENCE |
| KnowledgeGraphAdapter | KnowledgeGraph.entities, .graph.edges() | KNOWLEDGE, PERSON, CONCEPT | REFERENCE, CAUSAL, ASSOCIATION |
| DialogueAdapter | ShortTermMemory._turn_index | DIALOGUE | TEMPORAL(protected) |
| EpisodeAdapter | EpisodicMemory._episode_index | EPISODE | TEMPORAL |
| PersonProfileAdapter | PersonProfileManager.profiles | PERSON | REFERENCE |
| ExperienceAdapter | ExperienceRAG.documents | EXPERIENCE | (无自动边, 靠语义层补充) |

**适配器接口**:
```python
class BaseGraphAdapter:
    def sync(self, memory_graph: MemoryGraph, source: Any) -> int:
        """将后端数据投射到图中, 返回新增/更新节点数"""
        ...
```

**调用策略**:
- `sync_all()` -- 启动时全量同步一次
- 增量同步 -- 通过 EventBus 事件触发 (阶段 6 实现)
- TaskGraphAdapter 特殊处理: TaskGraph 是每次请求临时创建的，需在 orchestrator 中显式传入

**关键设计**: 适配器是 **只读** 的 -- 只从后端读取数据投射到图中，不修改后端。MemoryGraph 是后端的投影，可随时重建。

**风险**: 中等。需处理后端未初始化的情况（检查 `_initialized` 标记，静默跳过）。

---

### 阶段 3: 图注意力 -- 从1D到图BFS

**目标**: 在 AttentionWindowManager 中集成图激活加权，实现跨类型上下文发现

**修改文件**:
- `pipeline/attention_window.py` -- 核心修改

**具体改动**:

1. **构造函数新增参数**:
```python
def __init__(self, context_window_size, task_graph=None, 
             memory_graph=None,              # 新增
             use_graph_attention: bool = False  # 新增: 开关, 默认关闭
             reserved_tokens=7096):
    self._memory_graph = memory_graph
    self._use_graph_attention = use_graph_attention
```

2. **新增方法 `_compute_graph_activations()`**:
```python
def _compute_graph_activations(self) -> Dict[str, float]:
    """计算图激活值, 每次 apply_window() 调用一次"""
    if not self._memory_graph or not self._use_graph_attention:
        return {}
    
    # 种子节点: 当前聚焦的任务节点 + 最近3轮对话节点
    seeds = []
    if self._current_node_id:
        seeds.append(f"task:{self._current_node_id}")
    # 加入最近对话节点
    recent_dialogue_ids = [f"dialogue:{env.turn}" for env in self.envelopes[-6:] if env.msg.get("role") == "user"]
    seeds.extend(recent_dialogue_ids[-3:])
    
    return self._memory_graph.compute_activations(seeds, max_depth=3, decay=0.5)
```

3. **修改 `_score_message()` 评分公式**:
```python
def _score_message(self, env: MessageEnvelope) -> float:
    base = 1.0
    age = max(0, self._current_turn - env.turn)
    time_decay = 0.95 ** age
    mode_mult = self._mode_multiplier(env)
    
    # 新增: 图激活加成
    graph_boost = 1.0
    if self._graph_activations and env.node_id:
        graph_node_id = f"task:{env.node_id}"  # 映射到图节点ID
        activation = self._graph_activations.get(graph_node_id, 0.0)
        graph_boost = 1.0 + activation  # 最大2.0x加成
    
    return base * time_decay * mode_mult * graph_boost
```

4. **在 `apply_window()` 开头调用**:
```python
def apply_window(self) -> List[Dict]:
    # 新增: 计算图激活 (每次LLM调用前执行一次)
    self._graph_activations = self._compute_graph_activations()
    
    # ... 原有逻辑不变 ...
```

**安全机制**:
- `use_graph_attention` 默认 False，行为与现有完全一致
- 即使开启，图激活只是一个 **额外乘数** (1.0-2.0)，不改变原有的 time_decay 和 mode_multiplier 逻辑
- 如果 MemoryGraph 未初始化或出错，`_compute_graph_activations()` 返回空 dict，graph_boost = 1.0

**修改文件**:
- `pipeline/orchestrator.py` -- 在 AttentionWindowManager 初始化时传入 memory_graph 引用

```python
# 约第219行, 现有:
attn_window = AttentionWindowManager(
    context_window_size=self.context_window_size,
    task_graph=self.task_graph,
)
# 改为:
attn_window = AttentionWindowManager(
    context_window_size=self.context_window_size,
    task_graph=self.task_graph,
    memory_graph=self._memory_graph,        # 新增
    use_graph_attention=True,                # 新增
)
```

**风险**: 高（行为变更）。缓解措施：默认关闭开关；A/B对比测试；回退到1D模式。

---

### 阶段 4: 赫布学习 + 突触修剪

**目标**: 实现图的自组织能力 -- 共激活边增强，弱连接衰减修剪

**修改文件**:
- `zulong/memory/memory_graph.py` -- 新增方法

**4.1 赫布增强**:

在 `compute_activations()` 执行期间，记录共激活的节点对。BFS 结束后：

```python
def _hebbian_strengthen(self, activated_pairs: List[Tuple[str, str]]):
    """赫布增强: 共激活的节点对, 边权增加"""
    eta = 0.1  # 学习率
    for src, tgt in activated_pairs:
        edge = self.get_edge(src, tgt)
        if edge and not edge.get("protected"):
            old_w = edge["weight"]
            new_w = old_w + eta * (1 - old_w)  # 渐近趋向 1.0
            edge["weight"] = new_w
            edge["last_activated"] = time.time()
            edge["activation_count"] += 1
```

**4.2 关联边自动创建**:

维护一个 `_coactivation_counter: Dict[Tuple[str,str], int]`。当两个无直接边的节点共激活次数 >= 3 时，自动创建 ASSOCIATION 边 (初始权重 0.3)。

**4.3 突触修剪 (异步定时任务)**:

```python
async def _prune_loop(self):
    """每30分钟执行一次修剪"""
    while self._running:
        await asyncio.sleep(1800)  # 30分钟
        self._decay_and_prune()

def _decay_and_prune(self):
    """衰减非结构性边权, 移除弱连接"""
    now = time.time()
    edges_to_remove = []
    
    for src, tgt, data in self._graph.edges(data=True):
        if data.get("protected"):
            continue
        
        elapsed_hours = (now - data["last_activated"]) / 3600
        # 艾宾浩斯衰减, 24小时半衰期 (匹配 memory_evolution.py 的 MemoryStrength.decay())
        decayed = data["weight"] * math.exp(-elapsed_hours / 24)
        
        if decayed < 0.05:
            edges_to_remove.append((src, tgt))
        else:
            data["weight"] = decayed
    
    for src, tgt in edges_to_remove:
        self._graph.remove_edge(src, tgt)
    
    # 移除孤立的非结构性节点 (度为0 且 >24小时未访问)
    nodes_to_remove = []
    for node_id in list(self._graph.nodes):
        node = self.get_node(node_id)
        if node and node.node_type not in (NodeType.TASK,) and self._graph.degree(node_id) == 0:
            if (now - node.last_accessed) > 86400:
                nodes_to_remove.append(node_id)
    
    for nid in nodes_to_remove:
        self._graph.remove_node(nid)
```

**保护机制**:
- `protected=True` 的边永不修剪 (HIERARCHY, DEPENDENCY, TEMPORAL)
- TASK 类型节点永不因孤立而删除
- 所有修剪操作记录 INFO 日志

**风险**: 低。自包含在 MemoryGraph 内部。

---

### 阶段 5: 语义边层 -- FAISS 自动发现

**目标**: 基于 embedding 相似度自动创建/维护 SEMANTIC 边

**修改文件**:
- `zulong/memory/memory_graph.py` -- 新增 FAISS 侧车索引

**实现**:

```python
# MemoryGraph 内部
_faiss_index: faiss.IndexFlatIP  # 内积 (embeddings已归一化)
_faiss_id_map: Dict[str, int]    # node_id -> faiss_idx
_faiss_reverse_map: Dict[int, str]

def _ensure_embedding(self, node: GraphNode):
    """确保节点有 embedding, 没有则计算"""
    if node.embedding is not None:
        return
    text = self._get_node_text(node)  # 根据类型拼接文本
    node.embedding = embedding_manager.encode([text])[0]

def discover_semantic_neighbors(self, node_id: str, top_k: int = 5, threshold: float = 0.7):
    """按需发现语义邻居, 创建 SEMANTIC 边"""
    node = self.get_node(node_id)
    self._ensure_embedding(node)
    
    # FAISS 搜索 top_k
    distances, indices = self._faiss_index.search(node.embedding.reshape(1, -1), top_k + 1)
    
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        neighbor_id = self._faiss_reverse_map.get(idx)
        if neighbor_id == node_id or neighbor_id is None:
            continue
        if dist >= threshold:  # 内积 >= 0.7 (已归一化 = cosine相似度)
            if not self.get_edge(node_id, neighbor_id):
                self.add_edge(node_id, neighbor_id, EdgeType.SEMANTIC, weight=float(dist))
```

**调用时机**: 懒惰式 -- 只在节点被激活/访问时才计算语义邻居，不批量计算

**风险**: 中等。FAISS 索引需与图状态同步。缓解：启动时重建索引。

---

### 阶段 6: 任务图谱集成 + 启动序列

**目标**: 将 TaskGraph 双向连接到 MemoryGraph；将 MemoryGraph 接入系统启动流程

**修改文件**:

1. `pipeline/orchestrator.py` -- 在 run() 中注册 TaskGraph 变更回调
```python
# TaskGraph 创建后 (~line 204):
if self._memory_graph:
    task_adapter = TaskGraphAdapter()
    task_adapter.sync(self._memory_graph, self.task_graph)
    
    # 注册双向回调: TaskGraph 变更时增量同步到 MemoryGraph
    original_callback = self.task_graph.on_change_callback
    def _combined_callback(event_type, data):
        if original_callback:
            original_callback(event_type, data)
        task_adapter.incremental_sync(self._memory_graph, event_type, data)
    self.task_graph.on_change_callback = _combined_callback
```

2. `zulong/bootstrap.py` -- 在 SharedMemoryPool 之后初始化 MemoryGraph
```python
# 在 SharedMemoryPool.get_instance() 之后:
try:
    from zulong.memory.memory_graph import MemoryGraph
    memory_graph = MemoryGraph(persist_path="./data/memory_graph")
    memory_graph._load()
    memory_graph.sync_all()  # 全量同步所有后端
    logger.info("MemoryGraph 初始化完成")
except Exception as e:
    logger.error(f"MemoryGraph 初始化失败 (降级运行): {e}")
    memory_graph = None  # 降级: 不影响其他模块
```

3. `zulong/core/types.py` -- 新增事件类型
```python
# 新增:
MEMORY_GRAPH_UPDATED = "memory_graph_updated"
MEMORY_GRAPH_PRUNED = "memory_graph_pruned"
```

4. `zulong/memory/memory_graph.py` -- EventBus 订阅
```python
def _setup_event_subscriptions(self):
    """订阅后端事件, 触发增量同步"""
    event_bus = EventBus.get_instance()
    event_bus.subscribe(EventType.TASK_COMPLETED, self._on_task_event, "MemoryGraph")
    event_bus.subscribe(EventType.EXPERIENCE_STORED, self._on_experience_event, "MemoryGraph")
```

**风险**: 中等。启动顺序依赖。缓解：try/except 包裹，MemoryGraph 初始化失败不阻塞系统启动。

---

## 阶段间依赖关系

```
阶段1 (基础层)
    |
    v
阶段2 (适配器)        阶段5 (语义边) [可并行]
    |                      |
    v                      v
阶段3 (图注意力) <--------+
    |
    v
阶段4 (赫布学习 + 修剪)
    |
    v
阶段6 (集成 + 启动序列)
```

关键路径: 1 -> 2 -> 3 -> 4 -> 6
并行路径: 5 可与 3 同时进行

---

## 关键文件清单

| 操作 | 文件路径 | 变更说明 |
|------|---------|---------|
| 新建 | `zulong/memory/memory_graph.py` | 核心 MemoryGraph 类 (~600行) |
| 新建 | `zulong/memory/graph_adapters.py` | 6个后端适配器 (~350行) |
| 修改 | `zulong/memory/__init__.py` | 添加 MemoryGraph 导出 |
| 修改 | `pipeline/attention_window.py` | 集成图激活加成到评分公式 |
| 修改 | `pipeline/orchestrator.py` | 传入 MemoryGraph, 注册 TaskGraph 回调 |
| 修改 | `zulong/bootstrap.py` | 启动序列加入 MemoryGraph 初始化 |
| 修改 | `zulong/core/types.py` | 新增 2 个事件类型 |

---

## 验证计划

### 单元测试

1. **MemoryGraph 基础测试** (`tests/test_memory_graph.py`):
   - 节点 CRUD (add/get/remove)
   - 边 CRUD (add/get/remove)
   - BFS 扩散激活: 验证种子节点激活=1.0, 一跳邻居=0.5, 两跳=0.25
   - 赫布增强: 共激活后边权增加
   - 修剪: 弱边被移除, protected 边不被移除
   - 持久化: save -> load -> 验证图结构一致

2. **适配器测试** (`tests/test_graph_adapters.py`):
   - TaskGraphAdapter: 创建 TaskGraph + 节点/边, sync 后验证图中存在对应 GraphNode/GraphEdge
   - KnowledgeGraphAdapter: 创建 KG Entity/Relation, sync 后验证映射正确
   - DialogueAdapter: 模拟对话轮次, sync 后验证 TEMPORAL 边连接

3. **注意力窗口测试** (`tests/test_attention_window.py` 扩展):
   - use_graph_attention=False: 行为与现有完全一致 (回归测试)
   - use_graph_attention=True: 验证图关联节点获得更高评分
   - MemoryGraph=None 时的降级行为

### 集成测试

4. **端到端流程测试**:
   - 启动系统 -> 验证 MemoryGraph 初始化成功
   - 发送用户消息 -> 验证对话节点被投射到图中
   - 创建任务 -> 验证任务节点投射到图中
   - 多轮对话后 -> 验证赫布学习产生了 ASSOCIATION 边
   - 等待修剪周期 -> 验证弱边被清理

5. **降级测试**:
   - 删除 memory_graph 数据文件 -> 系统正常启动
   - 破坏 memory_graph JSON -> 系统降级运行
   - 禁用 use_graph_attention -> 注意力窗口行为不变

### 手动验证

6. **前端观察**:
   - 通过日志观察注意力模式切换时的 graph_boost 值
   - 对比开启/关闭图注意力时的消息保留差异
   - 检查 `./data/memory_graph/graph.json` 的内容是否合理

---

## 风险矩阵

| 阶段 | 风险等级 | 主要风险 | 缓解措施 |
|------|---------|---------|---------|
| 1 基础层 | 低 | 大图内存占用 | 阶段4修剪; 节点上限保护 |
| 2 适配器 | 中 | 后端未初始化时适配器失败 | 检查 _initialized; 静默跳过 |
| 3 图注意力 | **高** | 注意力行为回归 | 默认关闭开关; 对比测试; 1D回退 |
| 4 赫布/修剪 | 低 | 误删重要边 | protected标记; 结构性边不修剪 |
| 5 语义边 | 中 | FAISS索引与图状态不同步 | 启动时重建; 一致性校验 |
| 6 集成 | 中 | 启动顺序依赖导致系统不可用 | try/except降级; 日志告警 |

---

## 未来扩展方向 (本次不实施)

以下扩展能力建立在记忆图谱基础架构之上，待核心6阶段完成后按需启用：

### 扩展 A: 文档/知识摄入 (DocumentAdapter)

将长文本（小说、论文、技术文档、教程等）结构化切片并内化到图谱中：

```
长文本 -> 按章节/段落切片 -> DOCUMENT 类型节点
       -> NER 抽取人物/地点/概念 -> KNOWLEDGE/PERSON/CONCEPT 节点
       -> 相邻切片建 TEMPORAL 边 (顺序)
       -> 实体与出现段落建 REFERENCE 边
       -> embedding 自动发现 SEMANTIC 边
       -> 因果/情节链建 CAUSAL 边
```

实现方式: 在 `graph_adapters.py` 中新增 `DocumentAdapter`，NodeType.DOCUMENT 已预留。
核心价值: 将"阅读"转化为"内化" -- 不只是存储文本，而是将知识编织进图的关联网络中。

### 扩展 B: 跨会话知识积累

利用图的持久化特性，实现跨会话的知识积累与成长：
- 会话结束时自动将高激活节点和边固化
- 新会话启动时加载持久化图，自动获得历史上下文
- 赫布学习使常用知识路径越来越强，形成"专业技能"

### 扩展 C: 多智能体共享图谱

多个 Agent 实例共享同一个 MemoryGraph：
- Agent A 的任务经验自动对 Agent B 可见
- 通过图分区实现访问控制
- 协作任务中的依赖关系在图中自然表达

### 扩展 D: 图谱可视化

将 MemoryGraph 的实时状态推送到前端：
- 在现有任务图谱可视化的基础上，扩展显示跨类型节点和边
- 节点激活度用颜色深浅表达
- 边权用线条粗细表达
- 用户可点击任意节点查看关联上下文
