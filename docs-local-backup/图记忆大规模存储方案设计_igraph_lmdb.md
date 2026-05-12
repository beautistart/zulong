# 图记忆大规模存储方案设计 (igraph + LMDB)

> 面向年级别巨量记忆的存储架构设计，满足：超低延时（个位数毫秒）、离散文件存储、未加载即可发现关联

---

## 一、现状问题分析

### 1.1 当前方案瓶颈（NetworkX + JSON）

| 问题 | 影响场景 | 严重程度 |
|-----|---------|---------|
| 全量JSON加载 | 百万节点图启动耗时>30秒 | 致命 |
| 内存常驻 | 1百万节点≈2-5GB内存 | 严重 |
| 无索引支持 | BFS需遍历全图 | 严重 |
| 单文件存储 | 无法并行读写 | 中等 |

**结论**: NetworkX适合<10万节点场景，年级别记忆（预估千万级节点）需要全新架构。

---

## 二、igraph + LMDB 组合方案

### 2.1 技术选型对比

| 技术 | 优势 | 劣势 | 适用场景 |
|-----|-----|-----|---------|
| **igraph** | C后端、内存效率高10-100倍、BFS/邻接查询微秒级 | Python绑定略受限 | 图拓扑结构存储 |
| **LMDB** | B+树、mmap零拷贝、读延时<1ms、支持事务 | 写并发受限 | 节点/边属性KV存储 |
| **FAISS** | 向量检索高效、支持索引持久化 | 仅支持向量 | 语义相似度检索 |

**核心思想**: 
- **图拓扑分离**: 仅在内存中保留igraph的邻接结构（节点ID + 边列表）
- **属性按需加载**: LMDB存储完整节点/边属性，通过mmap零拷贝读取
- **向量侧车索引**: FAISS独立索引，支持语义检索

---

## 三、存储架构设计

### 3.1 三层分离存储

```
┌─────────────────────────────────────────────────────────────────┐
│                        Memory Graph API                          │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Layer 1       │  │   Layer 2       │  │   Layer 3       │
│   igraph        │  │   LMDB          │  │   FAISS         │
│   图拓扑索引     │  │   属性存储       │  │   向量索引       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
│                    │                    │
│ - 节点ID映射       │ - 节点完整属性     │ - embedding向量   │
│ - 邻接列表         │ - 边完整属性       │ - node_id映射     │
│ - 边类型标签       │ - 元数据           │ - 相似度索引      │
│                    │                    │
│ 内存常驻           │ mmap按需加载       │ 内存常驻          │
└────────────────────┴────────────────────┴──────────────────┘
```

### 3.2 离散文件存储设计

```
data/memory_graph/
├── topology/                      # Layer 1: igraph图拓扑
│   ├── graph_2024_01.graphml      # 按月分片（igraph GraphML格式）
│   ├── graph_2024_02.graphml
│   └── index.json                 # 时间分片索引
│
├── nodes/                         # Layer 2: LMDB节点属性
│   ├── lmdb_2024_01/              # 按月分片LMDB
│   │   ├── data.mdb               # 数据文件（mmap）
│   │   └── lock.mdb               # 锁文件
│   ├── lmdb_2024_02/
│   └── index.json                 # 分片索引
│
├── edges/                         # Layer 2: LMDB边属性
│   ├── lmdb_2024_01/
│   ├── lmdb_2024_02/
│   └── index.json
│
└── vectors/                       # Layer 3: FAISS向量索引
    ├── faiss_2024_01.index
    ├── faiss_2024_02.index
    ├── node_map_2024_01.json      # 向量→节点ID映射
    └── index.json
```

**分片策略**:
- **时间分片**: 按月切分，每个分片独立管理
- **大小控制**: 单分片目标<50万节点，文件大小约50-200MB
- **按需加载**: 仅加载当前活跃时间窗口的分片（近3个月）

---

## 四、核心数据结构设计

### 4.1 igraph层：图拓扑索引

```python
import igraph as ig

class TopologyIndex:
    """igraph图拓扑索引 - 仅存储节点ID和邻接关系"""
    
    def __init__(self):
        # igraph图对象（内存常驻）
        self.graph = ig.Graph(directed=True)
        
        # 节点ID ↔ igraph内部索引双向映射
        self.node_id_to_idx: Dict[str, int] = {}  # "task:o1_1" → 0
        self.idx_to_node_id: Dict[int, str] = {}  # 0 → "task:o1_1"
        
    def add_node(self, node_id: str, node_type: str):
        """添加节点 - 仅存储ID和类型"""
        idx = self.graph.add_vertex(name=node_id)
        self.node_id_to_idx[node_id] = idx.index
        self.idx_to_node_id[idx.index] = node_id
        # 节点类型作为属性（极小开销）
        self.graph.vs[idx.index]["type"] = node_type
        
    def add_edge(self, src_id: str, dst_id: str, edge_type: str, weight: float = 1.0):
        """添加边 - 存储类型和权重"""
        src_idx = self.node_id_to_idx[src_id]
        dst_idx = self.node_id_to_idx[dst_id]
        eid = self.graph.add_edge(src_idx, dst_idx)
        self.graph.es[eid.index]["type"] = edge_type
        self.graph.es[eid.index]["weight"] = weight
        
    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[str]:
        """获取邻居 - igraph微秒级查询"""
        idx = self.node_id_to_idx[node_id]
        neighbors = self.graph.neighborhood(idx, order=1, mode="out")
        if edge_type:
            # 过滤边类型
            neighbors = [
                n for n in neighbors 
                if self.graph.es.select(_source=idx, _target=n)["type"] == edge_type
            ]
        return [self.idx_to_node_id[n] for n in neighbors]
        
    def bfs_spread(self, seed_ids: List[str], max_depth: int = 3) -> List[str]:
        """BFS扩散 - igraph原生高效实现"""
        seed_indices = [self.node_id_to_idx[sid] for sid in seed_ids]
        
        # igraph BFS返回（节点索引列表, 距离列表）
        visited_indices, distances = self.graph.bfs(
            seed_indices, 
            mode="out",
            distance_limit=max_depth
        )
        
        # 按距离加权排序（近者优先）
        weighted = sorted(
            zip(visited_indices, distances),
            key=lambda x: 1.0 / (x[1] + 1)
        )
        return [self.idx_to_node_id[idx] for idx, _ in weighted]
```

**性能估算**:
- 内存开销: 每节点约50字节（ID + 类型 + 邻接指针）
- 百万节点: ~50MB内存（vs NetworkX的2-5GB）
- BFS查询: <1ms（igraph C后端）

### 4.2 LMDB层：节点属性存储

```python
import lmdb
import pickle
import msgspec  # 高性能序列化

class NodePropertyStore:
    """LMDB节点属性存储 - mmap零拷贝读取"""
    
    def __init__(self, db_path: str, map_size: int = 10 * 1024**3):  # 10GB虚拟空间
        self.env = lmdb.open(
            db_path,
            map_size=map_size,         # 虚拟内存映射大小
            max_dbs=3,                 # 3个子数据库
            read_only=False,
            metasync=False,            # 关闭元数据异步（性能优化）
            sync=False,                # 关闭同步（应用层控制）
        )
        
        # 子数据库
        self.node_db = self.env.open_db(b"nodes")       # 节点完整属性
        self.edge_db = self.env.open_db(b"edges")       # 边完整属性
        self.index_db = self.env.open_db(b"index")      # 辅助索引
        
        # msgspec序列化器（比pickle快5-10倍）
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder()
        
    def get_node(self, node_id: str) -> Optional[Dict]:
        """读取节点属性 - mmap零拷贝，<1ms"""
        with self.env.begin(read_only=True) as txn:
            key = node_id.encode("utf-8")
            data = txn.get(key, db=self.node_db)
            if data is None:
                return None
            # msgspec解码
            return self.decoder.decode(data)
            
    def set_node(self, node_id: str, properties: Dict):
        """写入节点属性 - 批量写入时合并事务"""
        with self.env.begin(write=True) as txn:
            key = node_id.encode("utf-8")
            data = self.encoder.encode(properties)
            txn.put(key, data, db=self.node_db)
            
    def batch_get_nodes(self, node_ids: List[str]) -> Dict[str, Dict]:
        """批量读取 - 单次事务，极低延时"""
        result = {}
        with self.env.begin(read_only=True) as txn:
            for node_id in node_ids:
                key = node_id.encode("utf-8")
                data = txn.get(key, db=self.node_db)
                if data:
                    result[node_id] = self.decoder.decode(data)
        return result
        
    def get_edge_properties(self, src_id: str, dst_id: str) -> Optional[Dict]:
        """读取边属性"""
        with self.env.begin(read_only=True) as txn:
            key = f"{src_id}→{dst_id}".encode("utf-8")
            data = txn.get(key, db=self.edge_db)
            return self.decoder.decode(data) if data else None
```

**LMDB关键参数**:
- `map_size`: 虚拟内存映射大小（建议10-50GB），实际占用取决于OS
- `mmap`: 直接映射到虚拟内存，无需显式read调用
- `read_only=True`: 读取时不加锁，极致性能

**性能估算**:
- 单点读取: 0.3-0.8ms（msgspec解码+dict构建）
- 批量读取(100节点): 2-5ms（单次事务）
- 文件大小: 每节点约500字节（含完整属性）

### 4.3 节点属性数据结构

```python
@dataclass
class NodeProperties:
    """节点完整属性 - 存储在LMDB"""
    node_id: str                      # 全局唯一ID
    node_type: str                    # NodeType枚举值
    label: str                        # 人类可读标签
    
    # 激活状态
    activation: float = 0.0           # 当前激活水平
    importance: str = "normal"        # Importance枚举值
    temperature: str = "cold"         # Temperature枚举值
    
    # 时间信息
    created_at: float = 0.0           # 创建时间戳
    last_accessed: float = 0.0        # 最后访问时间
    access_count: int = 0             # 访问次数
    
    # 内容数据（可选，按需加载）
    content: Optional[str] = None     # 节点内容（对话文本、任务描述等）
    embedding: Optional[List[float]] = None  # 嵌入向量（或存FAISS）
    
    # 后端引用
    backend_ref: str = ""             # 如 "stm:turn_42"
    storage_shard: str = ""           # 所属分片ID "2024_01"
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 五、未加载记忆发现机制

### 5.1 核心原理

**关键洞察**: igraph层仅存储节点ID和邻接关系，**无需加载节点属性即可发现关联**。

```
BFS扩散流程（未加载属性）:
1. 种子节点ID → igraph.get_neighbors() → 邻接节点ID列表
2. 邻接节点ID → igraph.get_neighbors() → 二级关联ID列表
3. 返回关联节点ID列表（无需访问LMDB）

选中加载流程（个位数毫秒）:
1. 关联节点ID列表 → LMDB.batch_get_nodes() → 节点属性字典
2. 延时 = LMDB读取(0.3-0.8ms/节点) × 批量优化
```

### 5.2 实现代码

```python
class MemoryGraphHybrid:
    """混合存储记忆图谱 - igraph拓扑 + LMDB属性"""
    
    def __init__(self, data_dir: str):
        self.topology = TopologyIndex()          # igraph层
        self.properties = NodePropertyStore(data_dir)  # LMDB层
        self.vector_index = FAISSSidecarIndex()  # FAISS层
        
    def discover_related_nodes(
        self, 
        seed_ids: List[str], 
        max_depth: int = 3,
        edge_types: Optional[List[str]] = None
    ) -> List[str]:
        """
        发现关联节点 - 无需加载属性
        
        仅操作igraph拓扑索引，返回关联节点ID列表
        延时 < 1ms
        """
        # igraph BFS扩散
        related_ids = self.topology.bfs_spread(seed_ids, max_depth)
        
        # 边类型过滤（如果指定）
        if edge_types:
            related_ids = [
                rid for rid in related_ids
                if any(
                    self.topology.get_edge_type(seed, rid) in edge_types
                    for seed in seed_ids
                )
            ]
        
        return related_ids
        
    def load_nodes_properties(self, node_ids: List[str]) -> Dict[str, NodeProperties]:
        """
        加载节点属性 - 批量读取LMDB
        
        延时 = 2-5ms（100节点）
        """
        raw_props = self.properties.batch_get_nodes(node_ids)
        return {
            nid: NodeProperties(**props) 
            for nid, props in raw_props.items()
        }
        
    def search_and_discover(
        self, 
        query: str, 
        top_k: int = 10,
        spread_depth: int = 2
    ) -> Tuple[List[str], Dict[str, NodeProperties]]:
        """
        语义搜索 + 关联发现（完整流程）
        
        1. FAISS向量搜索 → 种子节点ID
        2. igraph BFS扩散 → 关联节点ID
        3. LMDB批量加载 → 节点属性
        
        总延时: 3-10ms
        """
        # Step 1: FAISS搜索（1-2ms）
        seed_ids = self.vector_index.search(query, top_k=top_k)
        
        # Step 2: BFS扩散（<1ms）
        related_ids = self.discover_related_nodes(seed_ids, max_depth=spread_depth)
        
        # Step 3: 加载属性（2-5ms）
        properties = self.load_nodes_properties(related_ids)
        
        return related_ids, properties
```

### 5.3 延时分解

| 操作 | 延时 | 说明 |
|-----|-----|-----|
| igraph BFS扩散（100跳） | 0.5-1ms | C后端原生实现 |
| LMDB单点读取 | 0.3-0.8ms | mmap零拷贝 |
| LMDB批量读取（100节点） | 2-5ms | 单次事务 |
| msgspec反序列化 | 0.1-0.3ms | 比pickle快5-10倍 |
| **总计（发现+加载）** | **3-7ms** | 满足个位数毫秒要求 |

---

## 六、分片管理与虚拟内存优化

### 6.1 时间分片策略

```python
class ShardedMemoryGraph:
    """分片管理器 - 按时间切分存储"""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.active_shards: Dict[str, MemoryGraphHybrid] = {}  # 活跃分片缓存
        self.shard_index = self._load_shard_index()
        
    def _get_shard_id(self, timestamp: float) -> str:
        """时间戳 → 分片ID"""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return f"{dt.year}_{dt.month:02d}"
        
    def get_shard(self, shard_id: str, load_if_missing: bool = True) -> Optional[MemoryGraphHybrid]:
        """
        获取分片 - 按需加载
        
        近3个月分片常驻内存，更早分片按需加载
        """
        # 缓存命中
        if shard_id in self.active_shards:
            return self.active_shards[shard_id]
            
        # 冷分片加载
        if load_if_missing:
            shard_path = f"{self.base_dir}/{shard_id}"
            shard = MemoryGraphHybrid(shard_path)
            
            # 仅加载igraph拓扑，LMDB延迟映射
            shard.topology.load_from_graphml(f"{shard_path}/topology.graphml")
            shard.properties.env  # LMDB自动mmap
            
            # 缓存管理（LRU淘汰）
            if len(self.active_shards) > 3:  # 最多3个活跃分片
                self._evict_coldest_shard()
                
            self.active_shards[shard_id] = shard
            return shard
            
        return None
        
    def discover_across_shards(
        self, 
        seed_ids: List[str],
        max_depth: int = 3
    ) -> List[str]:
        """
        跨分片关联发现
        
        1. 从种子节点所在分片开始BFS
        2. 遇到跨分片边时，加载目标分片
        3. 合并所有关联节点ID
        """
        visited = set()
        queue = deque(seed_ids)
        
        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # 获取节点所在分片
            shard_id = self._get_node_shard(node_id)
            shard = self.get_shard(shard_id)
            
            # BFS扩展
            neighbors = shard.topology.get_neighbors(node_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    queue.append(neighbor_id)
                    
        return list(visited)
```

### 6.2 虚拟内存映射优化

**LMDB天然使用mmap**，但可进一步优化：

```python
# LMDB配置优化
env = lmdb.open(
    db_path,
    map_size=50 * 1024**3,      # 50GB虚拟空间
    writemap=True,              # 启用可写mmap（写操作也零拷贝）
    metasync=False,             # 关闭元数据异步刷新
    sync=False,                 # 应用层控制同步时机
    max_readers=128,            # 最大并发读事务
    lock=False,                 # 单进程场景可关闭锁（性能提升）
)

# 手动触发同步（应用退出时）
env.sync()
```

**OS层面优化**（Linux）:
```bash
# 预读优化
echo 128 > /sys/block/sda/queue/read_ahead_kb

# 脏页回写优化
echo 500 > /proc/sys/vm/dirty_writeback_centisecs

# 虚拟内存过度提交
echo 1 > /proc/sys/vm/overcommit_memory
```

---

## 七、性能基准估算

### 7.1 存储容量估算

| 指标 | 计算公式 | 年级别规模 |
|-----|---------|-----------|
| 节点数量 | 100万对话轮次 × 5节点/轮次 | 500万节点 |
| igraph内存 | 500万 × 50字节 | ~250MB |
| LMDB文件 | 500万 × 500字节 | ~2.5GB |
| FAISS索引 | 500万 × 512维 × 4字节 | ~10GB |
| **总计** | | **~13GB** |

**对比NetworkX方案**:
- NetworkX内存: 500万节点 × 2KB = **10GB**
- 节省内存: **40倍**

### 7.2 查询延时估算

| 操作 | 延时 | 说明 |
|-----|-----|-----|
| 单节点加载 | 0.5-1ms | LMDB读取+反序列化 |
| BFS扩散（100跳） | 1-2ms | igraph C后端 |
| 关联发现+加载（50节点） | 3-7ms | 满足个位数毫秒 |
| 跨分片发现（3分片） | 10-20ms | 含分片加载开销 |
| FAISS向量搜索 | 1-2ms | Flat索引精确搜索 |

### 7.3 并发能力

| 场景 | QPS | 说明 |
|-----|-----|-----|
| 只读查询 | 5000+ | LMDB只读事务无锁 |
| 混合读写 | 1000+ | 写事务串行化 |
| BFS扩散 | 10000+ | igraph内存操作 |

---

## 八、实现路线图

### Phase 1: 核心存储层（1周）

- [ ] `TopologyIndex` igraph拓扑索引实现
- [ ] `NodePropertyStore` LMDB属性存储实现
- [ ] 单分片CRUD单元测试

### Phase 2: 分片管理（1周）

- [ ] `ShardedMemoryGraph` 分片管理器
- [ ] 时间分片策略实现
- [ ] LRU缓存淘汰机制
- [ ] 跨分片关联发现

### Phase 3: 性能优化（1周）

- [ ] msgspec序列化集成
- [ ] LMDB mmap参数调优
- [ ] 批量读取优化
- [ ] 性能基准测试

### Phase 4: 迁移兼容（1周）

- [ ] NetworkX → igraph迁移脚本
- [ ] JSON → LMDB数据转换
- [ ] API兼容层
- [ ] 现有测试迁移验证

### Phase 5: 生产部署（1周）

- [ ] 配置参数文档
- [ ] 监控指标采集
- [ ] 故障恢复机制
- [ ] 压力测试

---

## 九、配置参数设计

```yaml
memory:
  storage_engine: igraph_lmdb  # 新增选项: networkx / igraph_lmdb
  
  igraph_lmdb:
    # 分片配置
    shard_by: month            # 分片策略: month / week / size
    max_active_shards: 3       # 最大活跃分片数
    
    # igraph配置
    topology_memory_limit_mb: 512
    
    # LMDB配置
    lmdb_map_size_gb: 50       # 虚拟内存映射大小
    lmdb_max_readers: 128      # 最大并发读事务
    lmdb_writemap: true        # 启用可写mmap
    
    # 性能参数
    batch_read_size: 100       # 批量读取节点数
    bfs_max_depth: 3           # BFS最大深度
    
    # 序列化
    serializer: msgspec        # msgspec / pickle
    
    # 文件路径
    data_dir: ./data/memory_graph
```

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|-----|-----|---------|
| igraph Python绑定兼容性 | 中等 | 封装抽象层，支持NetworkX回退 |
| LMDB写并发限制 | 低 | 写操作队列化，批量提交 |
| 跨分片查询开销 | 中等 | 分片预加载、热点缓存 |
| mmap内存压力 | 低 | map_size按需扩展、监控RSS |

---

## 十一、总结

**igraph + LMDB方案完全可行**，核心优势：

1. **内存效率提升40倍**: igraph拓扑仅250MB（vs NetworkX 10GB）
2. **查询延时个位数毫秒**: BFS扩散<1ms，属性加载2-5ms
3. **未加载即可发现**: igraph层独立支持关联发现
4. **天然mmap优化**: LMDB零拷贝读取，无需显式加载
5. **离散文件存储**: 按月分片，单分片50-200MB

**满足所有设计要求**。
