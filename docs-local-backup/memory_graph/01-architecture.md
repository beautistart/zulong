# 01-图记忆架构概述

> **阅读时间**: 15 分钟  
> **前置知识**: 无  
> **相关文档**: [README.md](./README.md)

---

## 📋 目录

1. [核心概念](#核心概念)
2. [架构设计](#架构设计)
3. [数据模型](#数据模型)
4. [核心算法](#核心算法)
5. [持久化机制](#持久化机制)
6. [与旧架构对比](#与旧架构对比)

---

## 🧠 核心概念

### 什么是 MemoryGraph？

**MemoryGraph（记忆图谱）** 是祖龙系统的**统一记忆中枢**，它将所有类型的记忆组织成一个**异构类型图**（Heterogeneous Graph）。

**核心类比**:
- **LLM** = 大脑皮层（负责推理）
- **MemoryGraph** = 海马体（负责记忆索引和联想）
- **AttentionWindow** = 工作记忆（当前关注的信息）

### 为什么需要图式记忆？

#### 问题：传统记忆架构的结构性割裂

```
┌─────────────────────────────────────────────────────┐
│  孤岛式记忆                                          │
├─────────────────────────────────────────────────────┤
│  ShortTermMemory  ←─┐                               │
│  EpisodicMemory   ←─┼─ 彼此独立，无连接             │
│  KnowledgeGraph   ←─┤                               │
│  RAG Libraries    ←─┤                               │
│  PersonProfile    ←─┘                               │
│                                                         │
│  问题：一个对话记忆无法自动关联到相关的知识实体或历史任务 │
└─────────────────────────────────────────────────────┘
```

**具体场景**:
```
第 1 轮：用户："AI MAX 395 是什么？"
        AI："AI MAX 395 是一款高性能处理器..."

第 2-10 轮：讨论其他话题...

第 11 轮：用户："它多少钱？"
        传统系统：❌ 无法理解"它"指的是 AI MAX 395
                  只能找到语义相似的对话，无法追溯 10 轮前的讨论
```

#### 解决方案：图式记忆

```
┌─────────────────────────────────────────────────────┐
│  MemoryGraph (统一异构类型图)                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  [DIALOGUE:1] ──REFERENCE──> [KNOWLEDGE:ai_max_395] │
│       │                          │                   │
│    TEMPORAL                   CAUSAL                │
│       │                          │                   │
│  [DIALOGUE:11] <─ASSOCIATION── [KNOWLEDGE:price]    │
│       │                                              │
│    REFERENCE                                         │
│       │                                              │
│  [TASK:task_001]                                     │
│                                                      │
└─────────────────────────────────────────────────────┘

用户："它多少钱？"
     ↓
MemoryGraph: 
1. 从 DIALOGUE:11 出发
2. 沿 REFERENCE 边追溯到 DIALOGUE:1
3. 发现 DIALOGUE:1 关联 KNOWLEDGE:ai_max_395
4. 沿 CAUSAL 边找到 KNOWLEDGE:price
5. 回答："AI MAX 395 的价格约为 1299 元"
```

---

## 🏗️ 架构设计

### 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    消费层 (Consumers)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  L2 推理引擎          AttentionWindowManager      │   │
│  │  AgentOrchestrator    L1-B Gatekeeper             │   │
│  │  Review/Replay System                              │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────┘
                    │ query / traverse / activate
┌───────────────────▼─────────────────────────────────────┐
│              MemoryGraph (新增集成层)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  NetworkX DiGraph (异构类型图)                     │   │
│  │  - GraphNode (9 种类型)                            │   │
│  │  - GraphEdge (7 种类型)                            │   │
│  │  - 边权：赫布学习 + 艾宾浩斯衰减                   │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌────────────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ FAISS 摘要索引  │ │ 异步修剪器│ │ JSON 持久化      │   │
│  │ (512-dim)       │ │ (30min)   │ │ (跨会话保留)     │   │
│  └────────────────┘ └──────────┘ └──────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  适配器注册表                                      │   │
│  │  - TaskGraphAdapter                               │   │
│  │  - KnowledgeGraphAdapter                          │   │
│  │  - DialogueAdapter                                │   │
│  │  - EpisodeAdapter                                 │   │
│  │  - PersonProfileAdapter                           │   │
│  │  - ExperienceAdapter                              │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────┬─────────────────────────────────────┘
                    │ adapters (只读适配器)
┌───────────────────▼─────────────────────────────────────┐
│                现有后端 (保持不变)                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  ShortTermMemory      EpisodicMemory              │   │
│  │  KnowledgeGraph       RAG Libraries               │   │
│  │  PersonProfile        TaskGraph (pipeline)        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 设计原则

| 原则 | 说明 | 实现方式 |
|------|------|---------|
| **适配器模式，不替换** | MemoryGraph 是现有模块之上的集成层 | 只读适配器，后端保持不变 |
| **最终一致性** | 图是后端数据的投影 | 图可从后端重建，后端仍为数据源头 |
| **渐进式启用** | 每个阶段独立可用 | 图注意力可通过开关回退到 1D 模式 |
| **异步优先** | 所有操作 async/await | 匹配现有架构风格 |
| **模型无关** | 对 0.8B 到 100B+ 均适用 | 图检索质量由图结构和边权决定 |
| **可扩展节点/边类型** | 未来新增类型无需修改核心逻辑 | NodeType/EdgeType 设计为可扩展枚举 |

---

## 📊 数据模型

### GraphNode（图节点）

```python
@dataclass
class GraphNode:
    """记忆图谱节点"""
    node_id: str                    # 全局唯一，带类型前缀： "task:o1_1", "dialogue:42"
    node_type: NodeType             # 节点类型 (9 种枚举)
    label: str                      # 人类可读标签
    activation: float = 0.0         # 当前激活水平 (0.0-1.0)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    backend_ref: str = ""           # 后端来源指针，如 "stm:turn_42"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # embedding 不在 dataclass 中，放在 _embeddings dict 中管理
```

#### 9 种节点类型 (NodeType)

```python
class NodeType(Enum):
    TASK = "task"               # 来自 TaskGraph.TaskNode
    DIALOGUE = "dialogue"       # 来自 ShortTermMemory 对话轮次
    KNOWLEDGE = "knowledge"     # 来自 KnowledgeGraph.Entity
    EXPERIENCE = "experience"   # 来自 ExperienceRAG 文档
    EPISODE = "episode"         # 来自 EpisodicMemory 摘要
    FILE = "file"               # 来自 TaskNode.files
    CONCEPT = "concept"         # 来自 KG 中 entity_type=CONCEPT
    PERSON = "person"           # 来自 PersonProfile / KG 中 PERSON 实体
    DOCUMENT = "document"       # 预留：未来文档/知识切片摄入
```

#### 节点元数据结构 (metadata)

```python
# 新节点默认 metadata
metadata = {
    # === 多维标签体系 ===
    "temperature": "hot",        # HOT / WARM / COLD (动态计算)
    "importance": "normal",      # TRIVIAL/NORMAL/IDENTITY/FACT/IMPORTANT/MUST_REMEMBER
    
    # === 内容存储 ===
    "content": "用户输入全文",
    "ai_response": "AI 回复全文",
    "summary": "对话摘要（100 字以内）",
    
    # === 分级读取支持 ===
    "user_preview": "用户输入前 30 字...",
    "ai_preview": "AI 回复前 30 字...",
    
    # === 后端引用 ===
    "trace_id": "trace_memory_000042",  # 指向共享池中的完整对话
    "turn_id": 42,
    
    # === 实体检测 ===
    "detected_entities": [
        {"type": "PERSON", "text": "小明", "confidence": 0.95},
        {"type": "DATE", "text": "明天", "confidence": 0.90}
    ],
    
    # === 重要度历史 ===
    "importance_history": [
        {"from": "normal", "to": "important", "timestamp": 1234567890}
    ],
    
    # === 其他 ===
    "semantically_isolated": False,  # 语义孤立标记
    "compressed": False,             # 是否被压缩
}
```

### GraphEdge（图边）

```python
# 边属性存储在 NetworkX 图中
edge_data = {
    "edge_type": EdgeType.REFERENCE,
    "weight": 0.85,                 # 0.0-1.0+，赫布增强 + 衰减
    "created_at": 1234567890,
    "last_activated": 1234567890,
    "activation_count": 5,
    "protected": False,             # True = 永不修剪 (结构性边)
    "metadata": {
        "pending_review": False,    # LLM 审查标记
        "review_requested_at": None,
    }
}
```

#### 7 种边类型 (EdgeType)

```python
class EdgeType(Enum):
    HIERARCHY = "hierarchy"         # 父子关系 (task h_edges) - 结构性边
    DEPENDENCY = "dependency"       # 数据依赖 (task d_edges) - 结构性边
    REFERENCE = "reference"         # 跨类型引用 (task->file, dialogue->knowledge)
    TEMPORAL = "temporal"           # 时间序列 (dialogue->dialogue) - 结构性边
    SEMANTIC = "semantic"           # 语义相似 (embedding cosine > 0.7)
    CAUSAL = "causal"               # 因果关系 (KG 中 CAUSED 关系)
    ASSOCIATION = "association"     # 赫布学习产生的关联
```

#### 边权管理

**边权公式**:
```
weight = base_weight + hebbian_enhancement - time_decay

其中:
- base_weight: 初始权重 (0.3-1.0，取决于边类型)
- hebbian_enhancement: 赫布学习增强 (共激活次数 × 学习率 0.1)
- time_decay: 时间衰减 (exp(-elapsed_hours × ln(2) / half_life))
```

**结构性边** (protected=True):
- `HIERARCHY` - 父子关系，永不修剪
- `DEPENDENCY` - 数据依赖，永不修剪
- `TEMPORAL` - 时间序列，永不修剪

**动态边** (protected=False):
- `REFERENCE` - 跨类型引用，可衰减
- `SEMANTIC` - 语义相似，可衰减
- `CAUSAL` - 因果关系，可衰减
- `ASSOCIATION` - 赫布学习产生，可衰减

---

## 🔬 核心算法

### BFS 扩散激活算法

**算法目标**: 从种子节点出发，沿边传播激活值，发现全局关联。

```python
def compute_activations(
    self,
    seed_node_ids: List[str],
    max_depth: int = 3,
    decay: float = 0.5,
    min_activation: float = 0.01,
) -> Dict[str, float]:
    """
    加权 BFS 扩散激活
    
    Args:
        seed_node_ids: 种子节点 ID 列表（如当前聚焦的任务节点）
        max_depth: 最大扩散深度（默认 3 跳）
        decay: 每跳衰减因子（默认 0.5）
        min_activation: 最小激活阈值，低于此值停止传播
    
    Returns:
        Dict[node_id → activation_score]
    """
    activations: Dict[str, float] = {}
    queue: deque = deque()
    
    # 1. 初始化种子节点
    for seed in seed_node_ids:
        if seed in self._nodes:
            activations[seed] = 1.0
            queue.append((seed, 0, 1.0))  # (node_id, depth, activation)
    
    # 2. BFS 循环
    while queue:
        node_id, depth, act = queue.popleft()
        
        if depth >= max_depth:
            continue
        
        # 遍历出边
        for _, neighbor, data in self._graph.out_edges(node_id, data=True):
            edge_weight = data.get("weight", 1.0)
            propagated = act * edge_weight * decay
            
            if propagated < min_activation:
                continue  # 剪枝
            
            if neighbor not in activations or activations[neighbor] < propagated:
                activations[neighbor] = max(activations.get(neighbor, 0), propagated)
                queue.append((neighbor, depth + 1, propagated))
        
        # 遍历入边（视为无向传播）
        for predecessor, _, data in self._graph.in_edges(node_id, data=True):
            edge_weight = data.get("weight", 1.0)
            propagated = act * edge_weight * decay
            
            if propagated < min_activation:
                continue
            
            if predecessor not in activations or activations[predecessor] < propagated:
                activations[predecessor] = max(activations.get(predecessor, 0), propagated)
                queue.append((predecessor, depth + 1, propagated))
    
    # 3. 更新节点激活值
    for nid, act_val in activations.items():
        self.update_node_activation(nid, act_val)
    
    return activations
```

**算法示例**:

```
种子节点：task:o1_1 (activation = 1.0)

第 1 跳 (depth=1, decay=0.5):
├─ task:o1_1_1 (HIERARCHY, weight=1.0) → activation = 1.0 × 1.0 × 0.5 = 0.5
├─ task:o1_1_2 (HIERARCHY, weight=1.0) → activation = 0.5
└─ file:weather.py (REFERENCE, weight=0.8) → activation = 1.0 × 0.8 × 0.5 = 0.4

第 2 跳 (depth=2, decay=0.5):
├─ knowledge:python_requests (SEMANTIC, weight=0.7, 从 file:weather.py 出发)
   → activation = 0.4 × 0.7 × 0.5 = 0.14
└─ experience:crawl_error (ASSOCIATION, weight=0.6, 从 task:o1_1_2 出发)
   → activation = 0.5 × 0.6 × 0.5 = 0.15

第 3 跳 (depth=3, decay=0.5):
└─ dialogue:42 (REFERENCE, weight=0.5, 从 experience:crawl_error 出发)
   → activation = 0.15 × 0.5 × 0.5 = 0.0375

最终激活值:
{
    "task:o1_1": 1.0,
    "task:o1_1_1": 0.5,
    "task:o1_1_2": 0.5,
    "file:weather.py": 0.4,
    "experience:crawl_error": 0.15,
    "knowledge:python_requests": 0.14,
    "dialogue:42": 0.0375
}
```

### 赫布学习（Hebbian Learning）

**赫布规则**: "Cells that fire together, wire together."（共激活 = 共强化）

```python
def hebbian_strengthen(self):
    """赫布增强：对共激活的边增加权重"""
    eta = 0.1  # 学习率
    edges = self._last_activated_edges  # 上一次 compute_activations 中的边
    
    for src, tgt in edges:
        if not self._graph.has_edge(src, tgt):
            continue
        data = self._graph.edges[src, tgt]
        if data.get("protected"):
            continue
        
        # 公式：new_weight = old_weight + η × (1 - old_weight)
        # 渐近趋向 1.0，永远不会超过 1.0
        old_w = data["weight"]
        new_w = old_w + eta * (1.0 - old_w)
        data["weight"] = new_w
        data["last_activated"] = time.time()
        data["activation_count"] += 1
```

**自动创建 ASSOCIATION 边**:

```python
def _update_coactivation_counter(self, activated_edges):
    """共激活次数 >= 3 时，自动创建 ASSOCIATION 边"""
    coactivation_threshold = 3
    
    # 统计共激活次数
    for src, tgt in activated_edges:
        pair = (min(src, tgt), max(src, tgt))  # 规范化顺序
        self._coactivation_counter[pair] = self._coactivation_counter.get(pair, 0) + 1
        
        # 超过阈值且无边 → 创建 ASSOCIATION 边
        if self._coactivation_counter[pair] >= coactivation_threshold:
            if not self.has_edge(pair[0], pair[1]):
                self.add_edge(
                    pair[0], pair[1],
                    EdgeType.ASSOCIATION,
                    weight=0.3  # 初始权重
                )
                del self._coactivation_counter[pair]
```

### 突触修剪（Synaptic Pruning）

**修剪策略**: 基于艾宾浩斯遗忘曲线，差异化衰减。

```python
def decay_and_prune(self):
    """衰减非结构性边权，移除弱连接和孤立节点"""
    now = time.time()
    prune_threshold = 0.05
    ln2 = math.log(2)
    
    for src, tgt, data in self._graph.edges(data=True):
        if data.get("protected"):
            continue
        
        # 获取两端节点的重要度，取更高的（决定半衰期）
        imp_src = self.get_importance(src) or Importance.NORMAL
        imp_tgt = self.get_importance(tgt) or Importance.NORMAL
        higher_imp = max(imp_src, imp_tgt, key=lambda x: _IMPORTANCE_ORDER.get(x, 1))
        
        # 获取半衰期（重要度越高，半衰期越长）
        half_life = _IMPORTANCE_HALF_LIFE.get(higher_imp, 24.0)
        if half_life == float('inf'):  # MUST_REMEMBER 永不衰减
            data["protected"] = True
            continue
        
        # 艾宾浩斯衰减公式
        elapsed_hours = (now - data["last_activated"]) / 3600
        decayed = data["weight"] * math.exp(-elapsed_hours * ln2 / half_life)
        
        if decayed < prune_threshold:
            # 太弱，移除
            self._graph.remove_edge(src, tgt)
        else:
            data["weight"] = decayed
```

**重要度分级半衰期**:

```python
_IMPORTANCE_HALF_LIFE = {
    Importance.TRIVIAL: 6.0,            # 6 小时（无意义闲聊）
    Importance.NORMAL: 24.0,            # 24 小时（普通对话）
    Importance.IDENTITY: 720.0,         # 30 天（身份信息）
    Importance.FACT: 360.0,             # 15 天（事实信息）
    Importance.IMPORTANT: 168.0,        # 7 天（承诺/任务）
    Importance.MUST_REMEMBER: float('inf'),  # 永久（用户显式要求）
}
```

---

## 💾 持久化机制

### JSON 持久化格式

```json
{
  "version": "1.0",
  "saved_at": 1234567890,
  "meta": {
    "node_count": 150,
    "edge_count": 280,
    "last_focus_context": {"node_id": "task:o1_1"}
  },
  "nodes": {
    "task:o1_1": {
      "node_type": "task",
      "label": "爬取天气数据",
      "activation": 0.85,
      "created_at": 1234567890,
      "last_accessed": 1234567890,
      "access_count": 5,
      "backend_ref": "taskgraph:o1_1",
      "metadata": {
        "temperature": "hot",
        "importance": "important",
        "content": "帮我写一个 Python 脚本，爬取天气数据"
      }
    },
    "dialogue:42": {
      "node_type": "dialogue",
      "label": "询问天气 API 用法",
      "activation": 0.65,
      "metadata": {
        "temperature": "warm",
        "importance": "normal",
        "trace_id": "trace_memory_000042"
      }
    }
  },
  "edges": [
    {
      "source": "task:o1_1",
      "target": "dialogue:42",
      "edge_type": "reference",
      "weight": 0.75,
      "protected": false,
      "created_at": 1234567890,
      "last_activated": 1234567890
    }
  ]
}
```

### FAISS 摘要索引持久化

```
持久化文件结构:
./data/memory_graph/
├── graph.json                      # 图结构（NetworkX 序列化）
├── summary_sidecar.index           # FAISS 索引文件
└── summary_sidecar.maps.json       # node_id ↔ faiss_id 映射

summary_sidecar.maps.json 格式:
{
  "node_to_faiss": {
    "dialogue:42": "summary_dialogue_42",
    "dialogue:43": "summary_dialogue_43"
  },
  "faiss_to_node": {
    "summary_dialogue_42": "dialogue:42",
    "summary_dialogue_43": "dialogue:43"
  }
}
```

### 自动保存（防抖写盘）

```python
def _mark_dirty(self):
    """标记为脏数据，触发防抖自动保存"""
    self._dirty = True
    
    # 取消之前的定时器
    if self._auto_save_timer:
        self._auto_save_timer.cancel()
    
    # 设置新定时器（2 秒后保存，VS Code 风格）
    self._auto_save_timer = threading.Timer(
        self._auto_save_delay,
        self._auto_save
    )
    self._auto_save_timer.start()

def _auto_save(self):
    """实际保存逻辑（加锁防并发）"""
    with self._save_lock:
        if self._dirty:
            self.save()
            self._dirty = False
            self._last_save_time = time.time()
```

---

## 📊 与旧架构对比

### 架构对比矩阵

| 维度 | 旧架构（三级记忆） | 新架构（MemoryGraph） | 提升 |
|------|------------------|---------------------|------|
| **记忆关联** | 孤岛式（各自独立） | 异构类型图（全连接） | ✅ 跨类型语义关联 |
| **检索方式** | 时间序列 + 字符相似度 | 图 BFS + 向量检索 | ✅ 发现长程依赖 |
| **注意力机制** | 1D 线性评分 | 图 BFS 扩散激活 | ✅ 智能上下文发现 |
| **记忆寿命** | 统一 24h TTL | 差异化衰减（6h-永久） | ✅ 重要信息永存 |
| **容量管理** | 固定轮数限制 | 重要度分级淘汰 | ✅ 智能容量管理 |
| **学习机制** | 无 | 赫布学习（共激活增强） | ✅ 自组织进化 |
| **持久化** | JSON + ChromaDB | JSON + FAISS | ✅ 统一存储格式 |

### 性能对比

| 指标 | 旧架构 | 新架构 | 改进 |
|------|-------|--------|------|
| **检索延迟（热数据）** | < 100ms | < 50ms | 2x 提升 |
| **检索延迟（冷数据）** | < 500ms | < 200ms | 2.5x 提升 |
| **长程依赖发现** | ❌ 不支持 | ✅ 支持（BFS 扩散） | 从 0 到 1 |
| **跨类型关联** | ❌ 不支持 | ✅ 支持（REFERENCE 边） | 从 0 到 1 |
| **记忆衰退模拟** | ❌ 简单 TTL | ✅ 艾宾浩斯曲线 | 更接近人脑 |

### 功能对比

| 功能 | 旧架构 | 新架构 | 说明 |
|------|-------|--------|------|
| **工作记忆** | ✅ conversation_history | ✅ conversation_history | 保留最近 2 轮 |
| **临时记忆** | ✅ EpisodicMemory | ✅ MemoryGraph (热数据) | 30 分钟内对话 |
| **长期记忆** | ✅ ShortTermMemory + RAG | ✅ MemoryGraph (冷数据) + RAG | 结构化存储 |
| **任务管理** | ✅ TaskGraph | ✅ MemoryGraph (TASK 节点) | 双向连接 |
| **知识管理** | ✅ KnowledgeGraph | ✅ MemoryGraph (KNOWLEDGE 节点) | 统一索引 |
| **经验沉淀** | ✅ ExperienceRAG | ✅ MemoryGraph (EXPERIENCE 节点) | 自动关联 |
| **人物画像** | ✅ PersonProfile | ✅ MemoryGraph (PERSON 节点) | 语义关联 |

---

## 🎯 总结

MemoryGraph 通过**统一的异构类型图**，成功解决了传统记忆架构的结构性割裂问题：

1. ✅ **结构化关联**: 任意类型的记忆都可以相互连接
2. ✅ **智能检索**: 从任意线索追溯全局关联（类似人脑联想）
3. ✅ **有效无限上下文**: 图可存储任意规模知识，当前窗口只注入最相关的子图
4. ✅ **自组织进化**: 赫布学习自动增强常用关联，艾宾浩斯衰减自动修剪弱连接
5. ✅ **差异化寿命**: 基于重要度分级，重要信息永久保存，琐碎信息快速遗忘

**下一步**: 
- [02-记忆分类与标签体系](./02-classification-tags.md) - 深入理解多维标签体系
- [03-记忆检索机制](./03-retrieval.md) - 掌握并行检索算法
- [06-快速入门指南](./06-quickstart.md) - 快速上手编码

---

**最后更新**: 2026-04-19  
**维护者**: 祖龙系统核心开发团队
