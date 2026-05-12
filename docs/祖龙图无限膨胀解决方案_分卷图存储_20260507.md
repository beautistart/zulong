# 祖龙图本位架构：无限膨胀问题解决方案

> 生成时间：2026-05-07
> 核心思想：借鉴分卷压缩（RAR/7z分卷）理念——逻辑上仍是一张图，物理上分成多个分片存储，任意分片可独立加载，无延迟调用

---

## 一、问题本质

祖龙的设计理念是**图本位**：记忆、注意力、任务编排、知识、代码锚点全部在一张大图上。
理论最优：所有数据在一张图上 → 全局推理、跨域关联、赫布学习。

现实瓶颈：
| 瓶颈 | 数据 | 影响 |
|------|------|------|
| 单文件膨胀 | memory_graph.json 已达15MB，embeddings占60.3% | 全量读写慢，启动加载阻塞 |
| 内存全量驻留 | 3286节点双写(_nodes+_graph)≈2MB冗余 | COLD节点(91.7%)白占内存 |
| 全遍历扫描 | _retrieve_hot 遍历全部3286节点 | 热路径O(N)复杂度 |
| 修剪未生效 | start_prune_loop()从未调用 | 图只增不减 |

---

## 二、核心设计：分卷图存储（Sharded Graph Storage）

### 2.1 类比分卷压缩

```
分卷压缩：
  大文件 → 分成 part1.rar, part2.rar, part3.rar
  点击任意一个 → 解压得到完整文件
  特点：逻辑上是一个文件，物理上多个分片

分卷图存储（借鉴）：
  大图 → 分成 shard_0.mg, shard_1.mg, shard_2.mg, ... + catalog.mg
  访问任意节点 → 自动定位所在分片，瞬间加载
  特点：逻辑上是一张图，物理上多个分片
```

**关键区别**：分卷压缩的任意分片都能解压出完整文件，是因为冗余存储。
分卷图存储不需要每个分片包含完整图，而是通过**目录(Catalog)**实现"任意入口、瞬间定位"。

### 2.2 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                 MemoryGraph (逻辑层)                      │
│  对外API不变：add_node / get_node / retrieve_context     │
│  内部感知不到分片，仍然是"一张图"                            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              ShardManager (分片管理层)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Catalog     │  │  LRU Cache   │  │  Prefetcher   │  │
│  │  (常驻内存)   │  │  (热分片缓存) │  │  (预测预加载)  │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              FileSystem (物理存储层)                       │
│  catalog.mg          ← 目录索引（常驻，<100KB）           │
│  shard_hot.mg        ← 热分片（常驻内存）                 │
│  shard_warm_0.mg     ← 温分片（LRU缓存，按需加载）        │
│  shard_warm_1.mg     ← 温分片                            │
│  shard_cold_0.mg     ← 冷分片（仅磁盘，按需加载）         │
│  shard_cold_1.mg     ← 冷分片                            │
│  shard_cold_2.mg     ← 冷分片                            │
│  embeddings/         ← 向量独立存储                       │
│    emb_shard_0.bin   ← FAISS二进制分片                   │
│    emb_shard_1.bin   ← FAISS二进制分片                   │
│  faiss_summary/      ← FAISS摘要侧车（已有，保持不变）    │
└─────────────────────────────────────────────────────────┘
```

---

## 三、核心组件设计

### 3.1 Catalog（目录索引）— 常驻内存，<100KB

Catalog是整个方案的"任意入口"关键。类似分卷压缩的恢复记录(recovery record)。

```python
@dataclass
class CatalogEntry:
    node_id: str
    shard_id: str          # "hot" / "warm_0" / "cold_3" 等
    node_type: NodeType     # 枚举，1 byte
    temperature: str        # "hot"/"warm"/"cold"，1 byte
    importance: str         # "must_remember"/"important"/"normal"/"trivial"
    label_hash: int         # label的hash，用于快速关键词过滤
    has_embedding: bool     # 是否有对应embedding

class Catalog:
    _entries: Dict[str, CatalogEntry]    # node_id → entry，常驻
    _type_index: Dict[NodeType, Set[str]]  # 按类型索引
    _shard_index: Dict[str, Set[str]]     # shard_id → node_ids
    _label_trie: TrieTree                 # label前缀树，用于快速搜索

    def locate(self, node_id: str) -> str:
        """O(1) 定位节点所在分片"""
        return self._entries[node_id].shard_id

    def search_by_label(self, prefix: str) -> List[Tuple[str, str]]:
        """前缀搜索，返回 [(node_id, shard_id), ...]"""
        return self._label_trie.search(prefix)
```

**为什么Catalog能做到<100KB**：
- 每个CatalogEntry约 50 bytes（node_id~40 + shard_id~10 + 枚举~5 + hash~8 + bool~1）
- 10000节点 × 50 bytes = 500KB，但实际可用更紧凑的编码（如node_id用数字映射）
- 使用label_hash替代完整label，TrieTree仅存高频访问的label前缀
- 即使10万节点，Catalog也可控制在1-2MB以内，远小于全量加载

### 3.2 ShardManager（分片管理器）

```python
class ShardManager:
    _catalog: Catalog                    # 常驻
    _hot_shard: GraphShard               # 常驻内存（HOT节点）
    _warm_cache: LRUCache[str, GraphShard]  # LRU缓存，容量=3个分片
    _cold_loader: ColdShardLoader        # 按需加载，mmap
    _emb_store: ShardedEmbeddingStore    # 向量独立存储
    _prefetcher: Prefetcher              # 预测预加载

    def get_node(self, node_id: str) -> GraphNode:
        """
        获取节点——对调用者透明，无感知分片
        """
        entry = self._catalog.locate(node_id)
        if entry.shard_id == "hot":
            return self._hot_shard.get(node_id)
        elif entry.shard_id.startswith("warm"):
            shard = self._warm_cache.get(entry.shard_id)
            if shard is None:
                shard = self._load_shard(entry.shard_id)
                self._warm_cache.put(entry.shard_id, shard)
            return shard.get(node_id)
        else:  # cold
            return self._cold_loader.load_node(entry.shard_id, node_id)

    def add_node(self, node: GraphNode) -> None:
        """
        添加节点——自动路由到热分片
        """
        self._hot_shard.add(node)
        self._catalog.add(node.id, CatalogEntry(
            node_id=node.id, shard_id="hot",
            node_type=node.node_type, temperature="hot",
            ...
        ))
        self._mark_dirty("hot")

    def promote(self, node_id: str) -> None:
        """
        提升：冷→温→热（访问时自动触发）
        """
        entry = self._catalog.locate(node_id)
        if entry.temperature == "cold":
            node = self._cold_loader.load_node(entry.shard_id, node_id)
            self._migrate_to_warm(node)
        elif entry.temperature == "warm":
            node = self._warm_cache.get_node(entry.shard_id, node_id)
            self._migrate_to_hot(node)
```

### 3.3 GraphShard（图分片）— 物理存储单元

```python
class GraphShard:
    """
    单个分片，包含一部分节点和这些节点之间的内部边
    跨分片边由Catalog的cross_edge_index管理
    """
    _nodes: Dict[str, GraphNode]
    _internal_edges: List[Edge]          # 两端都在本分片内的边
    _cross_edge_refs: List[CrossEdgeRef] # 指向其他分片的边（仅存node_id引用）
    _modified: bool

    def save(self, path: str) -> None:
        """原子写入单个分片"""
        data = {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "internal_edges": [...],
            "cross_edge_refs": [...],
        }
        atomic_write_json(path, data)

    @classmethod
    def load(cls, path: str) -> 'GraphShard':
        """加载单个分片"""
        ...
```

### 3.4 分片策略—温度驱动

**核心原则**：温度决定驻留级别，访问触发晋升，衰减触发降级。

```
┌──────────┐  访问   ┌──────────┐  访问   ┌──────────┐
│   COLD   │ ──────→ │   WARM   │ ──────→ │    HOT   │
│ 磁盘+mmap│ ←────── │ LRU缓存  │ ←────── │ 常驻内存  │
│          │  衰减   │          │  衰减   │          │
└──────────┘         └──────────┘         └──────────┘
  冷分片:              温分片:              热分片:
  - 超过24h未访问      - 1h-24h前访问      - 最近1h内访问
  - 完整写入磁盘       - 磁盘+LRU缓存      - 内存常驻
  - mmap零拷贝加载     - 容量3个分片       - 单个分片
  - 不可修改(只读)     - 可修改           - 可修改
```

**分片分配规则**：

| 温度 | 分片命名 | 内存驻留 | 写入时机 | 大小限制 |
|------|---------|---------|---------|---------|
| HOT | `shard_hot.mg` | 常驻 | 实时追加 | 节点数≤500 |
| WARM | `shard_warm_{i}.mg` | LRU缓存3个 | 防抖2秒 | 每分片≤1000节点 |
| COLD | `shard_cold_{i}.mg` | 无(mmap) | 降级时写入 | 每分片≤2000节点 |

**自动再平衡**：
- 热分片超过500节点 → 最冷的热节点降级到温分片
- 温分片超过1000节点 → 按last_accessed排序，最冷的降级到冷分片
- 冷分片超过2000节点 → 拆分为两个冷分片

### 3.5 向量独立分片（ShardedEmbeddingStore）

**问题**：embeddings占60.3%，是膨胀主因。

**方案**：完全独立于节点分片，按FAISS自然分片。

```python
class ShardedEmbeddingStore:
    """
    向量分片存储，与节点分片解耦
    每个分片是一个FAISS IndexFlatIP索引
    """
    _catalog_emb: Dict[str, int]       # node_id → shard_index
    _shards: List[faiss.IndexFlatIP]   # 按需加载的FAISS索引
    _hot_index: faiss.IndexFlatIP      # 热向量常驻内存
    _shard_size: int = 1000            # 每分片1000个向量

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """
        向量检索：先查热索引，不足再按需加载冷分片
        """
        # 1. 热索引检索
        hot_scores, hot_ids = self._hot_index.search(query_vec, top_k)
        # 2. 如果热结果不足，加载最可能命中的冷分片
        if len(hot_scores) < top_k:
            cold_shards = self._predict_relevant_shards(query_vec)
            for shard_id in cold_shards:
                shard = self._load_emb_shard(shard_id)
                scores, ids = shard.search(query_vec, top_k)
                hot_scores = merge_top_k(hot_scores, scores, top_k)
        return hot_scores

    def add(self, node_id: str, vec: np.ndarray) -> None:
        """添加向量到热索引，满时分片"""
        self._hot_index.add(vec)
        self._catalog_emb[node_id] = 0  # shard 0 = hot
        if self._hot_index.ntotal > self._shard_size:
            self._split_hot_shard()
```

**关键优化**：
- 向量分片使用FAISS原生二进制格式（.index文件），比base64 JSON快10x加载
- 冷向量分片使用mmap模式（`faiss.read_index(path, faiss.IO_FLAGS_MMAP)`），零拷贝
- 向量检索不再需要全量加载，按预测加载相关分片

### 3.6 跨分片边管理

**核心挑战**：节点A在热分片，节点B在冷分片，边(A,B)如何管理？

```python
@dataclass
class CrossEdgeRef:
    """跨分片边引用"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    target_shard: str    # 目标节点所在分片（从Catalog查）

class CrossEdgeIndex:
    """
    跨分片边索引，常驻内存
    空间优化：仅存 node_id + shard_id + edge_type + weight
    每条边约 60 bytes，10000条跨分片边 ≈ 600KB
    """
    _outgoing: Dict[str, List[CrossEdgeRef]]   # source_id → 跨分片出边
    _incoming: Dict[str, List[CrossEdgeRef]]   # target_id → 跨分片入边

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[Tuple[str, EdgeType, float]]:
        """获取跨分片邻居——无需加载目标分片，仅返回引用"""
        if direction in ("out", "both"):
            results = [(r.target_id, r.edge_type, r.weight) for r in self._outgoing.get(node_id, [])]
        if direction in ("in", "both"):
            results += [(r.source_id, r.edge_type, r.weight) for r in self._incoming.get(node_id, [])]
        return results
```

**关键**：`get_neighbors()` 返回的是(node_id, edge_type, weight)引用，**不需要加载目标分片**。
只有当调用方需要目标节点的完整数据时，才通过ShardManager.get_node()按需加载。

### 3.7 Prefetcher（预测预加载）— 实现零延迟

**目标**：当热路径BFS扩散可能访问温/冷分片时，提前异步加载，消除IO等待。

```python
class Prefetcher:
    """
    基于BFS扩散预测的异步预加载器
    """
    _shard_mgr: ShardManager
    _prefetch_queue: asyncio.Queue

    async def on_bfs_seed(self, seed_ids: List[str]) -> None:
        """
        BFS种子确定后，预测可能访问的分片并预加载
        """
        for seed_id in seed_ids:
            # 获取跨分片邻居
            cross_neighbors = self._cross_edge_index.get_neighbors(seed_id)
            for neighbor_id, _, _ in cross_neighbors:
                entry = self._catalog.locate(neighbor_id)
                if entry.temperature in ("warm", "cold"):
                    # 异步预加载，不阻塞当前操作
                    await self._prefetch_queue.put(entry.shard_id)

    async def _prefetch_worker(self) -> None:
        """后台预加载worker"""
        while True:
            shard_id = await self._prefetch_queue.get()
            if shard_id not in self._warm_cache:
                self._shard_mgr._load_shard(shard_id)  # 异步加载到LRU缓存
```

**预加载时机**：
1. `retrieve_context()` 调用前：根据query_text的FAISS检索结果，预加载命中节点所在分片
2. BFS扩散前：根据种子节点的跨分片边，预加载目标分片
3. `navigate_attention("deeper")` 时：预加载子节点分片

**延迟分析**：
- 热分片：0ms（常驻内存）
- 温分片（LRU命中）：0ms
- 温分片（LRU未命中+预加载）：~5ms（磁盘读取单分片<1MB）
- 冷分片（mmap预加载）：~2ms（mmap零拷贝）
- 冷分片（未预加载）：~10-50ms（首次mmap + 页面加载）

---

## 四、关键操作在分片架构下的实现

### 4.1 retrieve_context() — 热路径

```python
async def retrieve_context(self, query_text: str, top_k: int = 10) -> List[NodeScore]:
    # 1. Catalog前缀过滤（O(1)，无需加载任何分片）
    candidate_types = {NodeType.DIALOGUE, NodeType.TASK, NodeType.KNOWLEDGE}

    # 2. 热路径：仅访问热分片（常驻内存）
    hot_results = self._retrieve_hot_from_shard(
        query_text, self._shard_mgr._hot_shard, hot_window_minutes=30
    )

    # 3. 冷路径：FAISS侧车检索（已独立，无需改动）
    cold_results = await self._retrieve_cold(query_text, top_k)

    # 4. 冷路径命中的节点，按需从分片加载（预加载已提前触发）
    for node_id, score in cold_results:
        node = self._shard_mgr.get_node(node_id)  # 可能触发温/冷分片加载
        ...

    # 5. BFS扩散：跨分片边由CrossEdgeIndex提供（常驻内存）
    for seed_id in hot_seed_ids:
        # 内分片邻居：从当前分片获取
        internal_neighbors = current_shard.get_neighbors(seed_id)
        # 跨分片邻居：从CrossEdgeIndex获取
        cross_neighbors = self._cross_edge_index.get_neighbors(seed_id)
        # 跨分片邻居的完整数据按需加载
        for neighbor_id, _, _ in cross_neighbors:
            neighbor = self._shard_mgr.get_node(neighbor_id)
            ...
```

### 4.2 add_node() — 追加写入

```python
def add_node(self, node: GraphNode) -> None:
    # 新节点总是进入热分片
    self._shard_mgr._hot_shard.add(node)
    self._catalog.add(node.id, entry)
    self._mark_dirty("hot")

    # 热分片超限 → 再平衡
    if len(self._shard_mgr._hot_shard) > 500:
        self._rebalance()
```

### 4.3 save() — 分片独立保存

```python
def save(self) -> None:
    # Catalog先保存（小文件，快速）
    self._catalog.save(self._persist_path + "/catalog.mg")

    # 各分片独立保存（仅保存脏分片）
    if self._hot_shard._modified:
        self._hot_shard.save(self._persist_path + "/shard_hot.mg")
    for shard_id, shard in self._warm_cache.items():
        if shard._modified:
            shard.save(self._persist_path + f"/{shard_id}.mg")
    # 冷分片不可修改，无需保存

    # 向量分片独立保存
    self._emb_store.save()

    # 跨分片边索引保存
    self._cross_edge_index.save(self._persist_path + "/cross_edges.mg")
```

**对比当前**：当前save()全量写入15MB单文件；分片后仅保存脏分片，典型场景<1MB。

### 4.4 _load() — 按需加载启动

```python
def _load(self) -> None:
    # 仅加载Catalog（<100KB，瞬间完成）
    self._catalog = Catalog.load(self._persist_path + "/catalog.mg")

    # 加载热分片（通常<1MB，快速）
    self._hot_shard = GraphShard.load(self._persist_path + "/shard_hot.mg")

    # 加载跨分片边索引（通常<1MB）
    self._cross_edge_index = CrossEdgeIndex.load(self._persist_path + "/cross_edges.mg")

    # 温/冷分片不加载！按需加载
    # 向量FAISS侧车不加载！检索时按需mmap
```

**对比当前**：当前_load()全量加载15MB；分片后仅加载Catalog+热分片+跨边索引，<3MB，启动时间从秒级降到毫秒级。

---

## 五、温度升降级机制

### 5.1 升级（访问触发）

```python
def get_node(self, node_id: str) -> GraphNode:
    entry = self._catalog[node_id]
    now = time.time()

    # 更新访问时间
    entry.last_accessed = now

    # 升级判断
    if entry.temperature == "cold" and now - entry.last_accessed < 3600:
        # 冷→温（最近1h内第二次访问）
        node = self._cold_loader.load_node(entry.shard_id, node_id)
        self._migrate_to_warm(node)
        entry.temperature = "warm"
    elif entry.temperature == "warm" and now - entry.last_accessed < 600:
        # 温→热（最近10min内访问）
        node = self._warm_cache.get_node(entry.shard_id, node_id)
        self._migrate_to_hot(node)
        entry.temperature = "hot"

    return self._get_from_current_shard(node_id)
```

### 5.2 降级（衰减循环触发，修复P1#5）

```python
async def _decay_and_rebalance_loop(self) -> None:
    """
    每30分钟执行一次的衰减+再平衡循环
    替代原来未启动的start_prune_loop()
    """
    while True:
        await asyncio.sleep(1800)  # 30分钟

        # 1. 边权衰减（艾宾浩斯）
        self._decay_edges()

        # 2. 弱边修剪
        self._prune_weak_edges()

        # 3. 孤立节点清理
        self._prune_orphan_nodes()

        # 4. 温度降级检查
        now = time.time()
        for entry in self._catalog.all_entries():
            age = now - entry.last_accessed
            if entry.temperature == "hot" and age > 3600:
                # 热→温：超过1h未访问
                self._demote_hot_to_warm(entry)
            elif entry.temperature == "warm" and age > 86400:
                # 温→冷：超过24h未访问
                self._demote_warm_to_cold(entry)

        # 5. 热分片再平衡
        if len(self._hot_shard) > 500:
            self._rebalance_hot()

        # 6. 保存脏分片
        self.save()
```

### 5.3 迁移实现

```python
def _demote_hot_to_warm(self, entry: CatalogEntry) -> None:
    """热→温降级"""
    node = self._hot_shard.remove(entry.node_id)
    # 同时迁移该节点的内边到跨分片边索引
    internal_edges = self._hot_shard.pop_edges(entry.node_id)
    for edge in internal_edges:
        self._cross_edge_index.add(edge)

    # 选择目标温分片（轮询或最小分片）
    target_shard = self._select_warm_shard()
    target_shard.add(node)
    # 将跨分片边中目标为本节点的边转为内边
    ...

    entry.shard_id = target_shard.shard_id
    entry.temperature = "warm"
    self._mark_dirty(target_shard.shard_id)

def _demote_warm_to_cold(self, entry: CatalogEntry) -> None:
    """温→冷降级"""
    source_shard = self._warm_cache.get(entry.shard_id)
    node = source_shard.remove(entry.node_id)
    # 迁移边逻辑同上

    target_shard = self._select_or_create_cold_shard()
    target_shard.save_node(node)  # 直接写入磁盘，不从LRU缓存

    # 从LRU缓存移除源分片（如果分片已空）
    if len(source_shard) == 0:
        self._warm_cache.evict(entry.shard_id)
        os.remove(f"{self._persist_path}/{entry.shard_id}.mg")

    entry.shard_id = target_shard.shard_id
    entry.temperature = "cold"
```

---

## 六、与现有MemoryGraph API的兼容

**核心原则**：分片存储是内部实现，MemoryGraph的公开API不变。

```python
class MemoryGraph:
    """
    公开API完全不变，内部委托给ShardManager
    """
    def __init__(self, persist_path: str):
        self._shard_mgr = ShardManager(persist_path)
        # 兼容：_nodes属性改为属性代理
        # 不再维护独立的 _nodes dict

    @property
    def _nodes(self) -> Dict[str, GraphNode]:
        """兼容旧代码：返回热分片+LRU缓存中所有节点的视图"""
        return self._shard_mgr.all_loaded_nodes()

    def add_node(self, node: GraphNode) -> None:
        self._shard_mgr.add_node(node)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._shard_mgr.get_node(node_id)

    def get_neighbors(self, node_id: str, max_depth: int = 1) -> List:
        return self._shard_mgr.get_neighbors(node_id, max_depth)

    async def retrieve_context(self, query: str, top_k: int = 10) -> List:
        # 内部走分片路径，外部不变
        ...
```

**兼容性策略**：
- `_nodes` 属性通过 property 代理，返回当前已加载节点的视图（热+LRU温）
- 遍历 `_nodes` 的旧代码（如_retrieve_hot全遍历）改为通过Catalog过滤+按需加载
- `_graph` (NetworkX) 仅在热分片+LRU温分片中维护，不再包含冷节点
- 边遍历通过CrossEdgeIndex补充跨分片边

---

## 七、TaskGraph同步改造

TaskGraph同样需要分片，但策略更简单（TaskGraph远小于MemoryGraph）：

```python
class TaskGraphShardManager:
    """
    TaskGraph分片策略：按任务完成状态分片
    - active.mg: 进行中的任务图（常驻内存）
    - completed_{date}.mg: 已完成的任务图（按日期归档，冷存储）
    """
    _active_shard: TaskGraphShard     # 常驻
    _archive_index: Dict[str, str]    # graph_id → archive_file

    def get_graph(self, graph_id: str) -> TaskGraph:
        if graph_id in self._active_shard:
            return self._active_shard.get(graph_id)
        archive = self._archive_index.get(graph_id)
        if archive:
            return self._load_archive(archive)  # 按需加载
```

**TaskGraph自动归档**：
- 任务完成后，TaskGraph从active移到completed归档
- 归档文件按日期命名：`completed_20260507.mg`
- 超过30天的归档文件自动压缩（gzip）
- Level 2恢复时从归档按需加载

---

## 八、性能对比

| 指标 | 当前（单文件） | 分片后 | 提升 |
|------|--------------|--------|------|
| 启动加载时间 | ~2s (15MB全量) | ~50ms (Catalog+热分片) | **40x** |
| 节点查找 | O(1) Dict | O(1) Catalog定位+分片内O(1) | 持平 |
| 热路径检索 | O(N) 全遍历3286节点 | O(H) 仅遍历热节点≤500 | **6.5x** |
| 冷路径检索 | FAISS已独立 | 不变 | 持平 |
| save() | 15MB全量写 | 仅脏分片<1MB | **15x** |
| 内存占用 | ~30MB (全量双写) | ~5MB (热分片+LRU+Catalog) | **6x** |
| BFS 1跳扩散 | O(1) NetworkX | O(1) 内边+O(1) CrossEdgeIndex | 持平 |
| 跨分片边访问 | 不适用 | 预加载~2ms，未预加载~50ms | 可接受 |
| 节点迁移延迟 | 不适用 | 热→温<1ms，温→冷~5ms | 可忽略 |

---

## 九、实施路线

### Phase 1：基础设施（1-2天）
1. 实现 Catalog 类（目录索引）
2. 实现 GraphShard 类（分片读写）
3. 实现 CrossEdgeIndex 类（跨分片边）

### Phase 2：分片管理器（2-3天）
4. 实现 ShardManager（分片管理+LRU缓存）
5. 实现 ShardedEmbeddingStore（向量独立分片）
6. 实现温度升降级逻辑
7. 实现 Prefetcher（预测预加载）

### Phase 3：MemoryGraph改造（2-3天）
8. MemoryGraph内部委托ShardManager
9. _load()改为按需加载
10. save()改为分片独立保存
11. retrieve_context()适配分片
12. 兼容性：_nodes属性代理

### Phase 4：启动修剪循环（1天）
13. 在__init__中启动_decay_and_rebalance_loop()
14. 修复P1#5：修剪循环从未启动

### Phase 5：TaskGraph改造（1-2天）
15. TaskGraph按完成状态分片
16. 自动归档机制
17. Level 2恢复适配

### Phase 6：验证与压测（1-2天）
18. 10万节点压测
19. 并发读写压测
20. 启动/恢复时间测试
21. 端到端功能测试
