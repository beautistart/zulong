# 祖龙系统图式记忆架构完全指南

> **文档版本**: v1.0  
> **最后更新**: 2026-04-19  
> **适用系统**: 祖龙系统 (ZULONG) Beta 4  
> **目标读者**: 系统架构师、AI 工程师、开发者

---

## 📋 目录

1. [图记忆架构概述](#1-图记忆架构概述)
2. [记忆分类与标签体系](#2-记忆分类与标签体系)
3. [记忆检索机制](#3-记忆检索机制)
4. [图注意力机制](#4-图注意力机制)
5. [复杂任务编排](#5-复杂任务编排)
6. [快速入门指南](#6-快速入门指南)
7. [FAQ 与故障排查](#7-faq-与故障排查)

---

# 1. 图记忆架构概述

## 1.1 核心概念

### 什么是 MemoryGraph？

**MemoryGraph（记忆图谱）** 是祖龙系统的**统一记忆中枢**，它将所有类型的记忆（对话、任务、知识、人物、文件等）组织成一个**异构类型图**（Heterogeneous Graph）。

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

## 1.2 架构设计

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

## 1.3 数据模型

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

## 1.4 核心算法

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

# 2. 记忆分类与标签体系

## 2.1 多维标签体系概述

### 为什么需要多维标签？

**传统硬分区的问题**:
```
旧架构:
├─ L1 (工作记忆) = 最近 2 轮
├─ L2 (临时记忆) = 最近 20-50 轮
└─ L3 (长期记忆) = 向量化知识

问题：边界僵硬，无法灵活处理
- 第 2 轮的重要信息（如用户姓名）2 小时后被遗忘
- 第 50 轮的琐碎闲聊（如"嗯嗯"）仍然占用空间
```

**新架构的解决方案**:
```
MemoryGraph 使用三组正交标签实现灵活分类:

标签维度       存储方式      可选值                          用途
─────────────────────────────────────────────────────────────────
温度标签       metadata 存储  hot / warm / cold              检索路由
(temperature)  动态更新                                       (热数据直接遍历)

重要度标签     metadata 存储  trivial / normal / identity /  差异化衰减
(importance)   写入时设置    fact / important / must_remember (决定半衰期)
               可被提升                                      

时间段标签     不存储        recent / non_recent              并行检索
(time_scope)   查询时计算                                     (互斥过滤)
```

### 标签设计原则

| 原则 | 说明 | 实现方式 |
|------|------|---------|
| **正交性** | 三组标签相互独立，可自由组合 | 如：`hot + important + recent` |
| **动态性** | 温度标签实时计算，不依赖存储值 | 基于 `last_accessed` 动态算 |
| **可提升性** | 重要度只升不降，支持自动/手动提升 | `promote_importance()` 方法 |
| **查询优化** | 时间段标签查询时计算，支持互斥过滤 | `is_recent()` 方法 |

## 2.2 温度标签 (Temperature)

### 温度定义

```python
class Temperature(Enum):
    """节点温度标签"""
    HOT = "hot"    # 最近被访问/激活 (1 小时内)
    WARM = "warm"  # 中等时间未激活 (1h-24h)
    COLD = "cold"  # 长期未激活 (>24h)
```

### 动态计算逻辑

```python
def get_temperature(self, node_id: str) -> Optional[Temperature]:
    """
    动态计算节点温度（基于 last_accessed 实时计算）
    
    Returns:
        Temperature 枚举值，节点不存在返回 None
    """
    node = self._nodes.get(node_id)
    if not node:
        return None
    
    elapsed = time.time() - node.last_accessed
    
    if elapsed < _TEMPERATURE_THRESHOLDS["hot_max"]:
        return Temperature.HOT
    elif elapsed < _TEMPERATURE_THRESHOLDS["warm_max"]:
        return Temperature.WARM
    else:
        return Temperature.COLD

# 温度阈值配置（秒）
_TEMPERATURE_THRESHOLDS = {
    "hot_max": 3600,    # 1 小时内 → hot
    "warm_max": 86400,  # 1h-24h → warm
    # > 24h → cold
}
```

### 温度的实际用途

```python
# 1. 检索路由（热数据直接遍历，冷数据 FAISS 检索）
async def retrieve_context(self, query_text, top_k=10):
    # 路径 A: 热数据遍历
    hot_nodes = [n for n in self._nodes.values() 
                 if self.is_recent(n.node_id, window_seconds=1800)]
    
    # 路径 B: 冷数据 FAISS（排除热数据）
    cold_results = await self._summary_index.search(
        query_text, 
        exclude_node_ids=set(n.node_id for n in hot_nodes)
    )
    
    return merge_and_sort(hot_nodes + cold_results)[:top_k]
```

## 2.3 重要度标签 (Importance)

### 重要度分级

```python
class Importance(Enum):
    """节点重要度标签"""
    TRIVIAL = "trivial"           # 无意义闲聊（"嗯"/"好的"）
    NORMAL = "normal"             # 普通对话
    IDENTITY = "identity"         # 身份信息（姓名/年龄/称呼）
    FACT = "fact"                 # 客观事实（日期/电话/地址）
    IMPORTANT = "important"       # 承诺/任务指令/偏好
    MUST_REMEMBER = "must_remember"  # 用户显式要求记住
```

### 重要度排序

```python
_IMPORTANCE_ORDER = {
    Importance.TRIVIAL: 0,
    Importance.NORMAL: 1,
    Importance.IDENTITY: 2,
    Importance.FACT: 3,
    Importance.IMPORTANT: 4,
    Importance.MUST_REMEMBER: 5,
}
```

### 初始重要度设置（写入时检测）

```python
class DialogueAdapter:
    # 重要度检测规则表
    _IMPORTANCE_RULES = [
        (r"我叫.*", Importance.IDENTITY),              # 姓名
        (r"我今年.*岁", Importance.IDENTITY),          # 年龄
        (r"帮我记住.*", Importance.MUST_REMEMBER),     # 显式要求
        (r"我的 (电话 | 手机号 | 地址 | 生日).*", Importance.FACT),  # 事实信息
        (r"明天.*记得提醒我", Importance.IMPORTANT),   # 承诺/提醒
        (r"嗯 | 好 | 哦 | 好的 | 知道了", Importance.TRIVIAL),  # 闲聊
    ]
    
    def _detect_importance(self, text: str) -> Tuple[Importance, List[str]]:
        """根据文本内容自动检测重要度"""
        matched_reasons = []
        max_importance = Importance.NORMAL
        
        for pattern, importance in self._IMPORTANCE_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                matched_reasons.append(f"匹配规则：{pattern}")
                if _IMPORTANCE_ORDER[importance] > _IMPORTANCE_ORDER[max_importance]:
                    max_importance = importance
        
        return max_importance, matched_reasons
```

### 重要度半衰期

```python
_IMPORTANCE_HALF_LIFE = {
    Importance.TRIVIAL: 6.0,            # 6 小时（快速遗忘）
    Importance.NORMAL: 24.0,            # 24 小时（默认）
    Importance.IDENTITY: 720.0,         # 30 天（长期记忆）
    Importance.FACT: 360.0,             # 15 天（中期记忆）
    Importance.IMPORTANT: 168.0,        # 7 天（周级别）
    Importance.MUST_REMEMBER: float('inf'),  # 永不衰减
}
```

## 2.4 时间段标签 (Time Scope)

### 时间段定义

时间段标签是**查询时动态计算**的，用于并行检索的互斥过滤。

```python
class TimeScope(Enum):
    RECENT = "recent"       # 最近 T 分钟内（默认 30 分钟）
    NON_RECENT = "non_recent"  # 超过 T 分钟
```

### 查询时计算

```python
def is_recent(self, node_id: str, window_seconds: int = 1800) -> bool:
    """
    判断节点是否在热窗口内（检索路由用）
    
    Args:
        node_id: 节点 ID
        window_seconds: 热窗口秒数，默认 30 分钟
    
    Returns:
        True 表示节点在热窗口内
    """
    node = self._nodes.get(node_id)
    if not node:
        return False
    return (time.time() - node.last_accessed) < window_seconds
```

### 检索路由规则

```
用户输入
  ├── 并行路径 A: time_scope=recent 的节点
  │   └── 直接遍历 + BFS（不经过 FAISS）
  │
  └── 并行路径 B: time_scope=non_recent 的节点
      └── FAISS 摘要向量检索 → BFS 下钻
  
  合并：两路径结果按激活值排序 → Top-N 注入
```

## 2.5 标签的正交性

### 三组标签相互独立

```
温度 (Temperature)     重要度 (Importance)     时间段 (Time Scope)
─────────────────      ──────────────────      ───────────────────
HOT                    TRIVIAL                 RECENT
HOT                    NORMAL                  RECENT
HOT                    IDENTITY                RECENT
HOT                    FACT                    RECENT
HOT                    IMPORTANT               RECENT
HOT                    MUST_REMEMBER           RECENT

WARM                   TRIVIAL                 NON_RECENT (可能)
WARM                   NORMAL                  NON_RECENT (可能)
...

COLD                   TRIVIAL                 NON_RECENT
COLD                   NORMAL                  NON_RECENT
COLD                   IDENTITY                NON_RECENT (沉睡记忆)
COLD                   FACT                    NON_RECENT
COLD                   IMPORTANT               NON_RECENT
COLD                   MUST_REMEMBER           NON_RECENT (永久记忆)
```

### 典型组合示例

| 温度 | 重要度 | 时间段 | 说明 | 处理方式 |
|------|-------|--------|------|---------|
| HOT | NORMAL | RECENT | 刚发生的普通对话 | 热数据遍历，24h 后衰减 |
| HOT | IMPORTANT | RECENT | 刚发生的任务指令 | 热数据遍历，7 天衰减 |
| WARM | IDENTITY | NON_RECENT | 1 天前的身份信息 | 热/冷数据皆可，30 天衰减 |
| COLD | FACT | NON_RECENT | 2 天前的地址信息 | 冷数据 FAISS，15 天衰减 |
| COLD | MUST_REMEMBER | NON_RECENT | 用户要求永久记住 | 冷数据 FAISS，永不衰减 |
| COLD | TRIVIAL | NON_RECENT | 3 天前的闲聊 | 冷数据 FAISS，6h 后清理 |

## 2.6 重要度动态提升

### 自动提升机制

```python
def promote_importance(self, node_id: str, target: Importance) -> bool:
    """
    提升节点重要度（只升不降）
    
    Args:
        node_id: 节点 ID
        target: 目标重要度
    
    Returns:
        是否实际提升
    """
    node = self._nodes.get(node_id)
    if not node:
        return False
    
    current = self.get_importance(node_id) or Importance.NORMAL
    current_order = _IMPORTANCE_ORDER.get(current, 1)
    target_order = _IMPORTANCE_ORDER.get(target, 1)
    
    # 只允许向上提升
    if target_order <= current_order:
        return False
    
    # 执行提升
    node.metadata["importance"] = target.value
    
    # 记录提升历史
    history = node.metadata.setdefault("importance_history", [])
    history.append({
        "from": current.value,
        "to": target.value,
        "timestamp": time.time(),
    })
    
    # 提升为 MUST_REMEMBER 时，自动将所有关联边设为 protected
    if target == Importance.MUST_REMEMBER:
        for _, neighbor, data in self._graph.out_edges(node_id, data=True):
            data["protected"] = True
        for predecessor, _, data in self._graph.in_edges(node_id, data=True):
            data["protected"] = True
    
    self._mark_dirty()
    logger.info(
        f"[MemoryGraph] 节点 {node_id} 重要度提升：{current.value} → {target.value}"
    )
    return True
```

### 基于访问频率的自动提升

```python
def run_importance_review(self) -> Dict[str, Any]:
    """
    扫描所有节点，根据访问模式自动提升重要度
    
    规则:
    - access_count >= 3 且 importance == NORMAL → 自动提升为 IMPORTANT
    - access_count >= 5 且 importance == IMPORTANT → 标记为 LLM 审查候选
    
    Returns:
        {"auto_promoted": int, "llm_candidates": List[str]}
    """
    auto_promoted = 0
    llm_candidates = []
    
    for node_id, node in self._nodes.items():
        imp = self.get_importance(node_id) or Importance.NORMAL
        count = node.access_count
        
        if count >= 3 and imp == Importance.NORMAL:
            if self.promote_importance(node_id, Importance.IMPORTANT):
                auto_promoted += 1
        
        elif count >= 5 and imp == Importance.IMPORTANT:
            # 不直接调用 LLM，仅标记候选
            llm_candidates.append(node_id)
    
    if auto_promoted > 0:
        logger.info(f"[MemoryGraph] 自动提升了 {auto_promoted} 个节点的重要度")
    if llm_candidates:
        logger.info(f"[MemoryGraph] {len(llm_candidates)} 个节点待 LLM 审查确认提升")
    
    return {"auto_promoted": auto_promoted, "llm_candidates": llm_candidates}
```

---

# 3. 记忆检索机制

## 3.1 检索架构概述

### 核心挑战

传统记忆检索面临的核心挑战：

1. **效率 vs 完整性**: 
   - 快速检索可能遗漏重要信息
   - 全面检索可能延迟过高

2. **热数据 vs 冷数据**:
   - 热数据（最近对话）访问频繁，需要低延迟
   - 冷数据（历史对话）量大，需要高效索引

3. **语义 vs 时间**:
   - 语义检索能找到相关内容，但可能遗漏时间上接近的对话
   - 时间检索能保证新鲜度，但可能遗漏语义相关的内容

### MemoryGraph 的解决方案

```
并行检索架构:

用户输入："处理器价格"
         ↓
┌─────────────────────────────────────────┐
│  并行检索（asyncio.gather）              │
├─────────────────────────────────────────┤
│  路径 A: 热数据遍历                      │
│  ├─ 筛选 is_recent() == True 的节点      │
│  ├─ 关键词匹配 + BFS 扩散                │
│  └─ 返回摘要 + 索引                      │
│                                         │
│  路径 B: 冷数据 FAISS                    │
│  ├─ 向量检索摘要索引                     │
│  ├─ 过滤掉热数据（互斥）                 │
│  └─ BFS 下钻获取详情                     │
└─────────────────────────────────────────┘
         ↓
合并结果，按 activation 降序排序 → Top-10 注入上下文
```

**核心优势**:
- ✅ **低延迟**: 热数据 < 50ms，冷数据 < 200ms，平均 < 100ms
- ✅ **高召回**: 并行检索，互不干扰
- ✅ **互斥过滤**: 避免热数据被 FAISS 重复检索
- ✅ **分级读取**: 先返回摘要，按需读取详情

## 3.2 并行检索策略

### 核心方法签名

```python
class MemoryGraph:
    async def retrieve_context(
        self,
        query_text: str,
        top_k: int = 10,
        hot_window_minutes: int = 30,
    ) -> List[Dict]:
        """
        并行检索上下文
        
        Args:
            query_text: 查询文本
            top_k: 返回数量限制
            hot_window_minutes: 热窗口分钟数（默认 30 分钟）
        
        Returns:
            [
                {
                    "node_id": "dialogue:42",
                    "node_type": "dialogue",
                    "label": "询问处理器价格",
                    "summary": "询问定义：AI MAX 395 是什么 → ...",
                    "trace_id": "trace_memory_000042",
                    "score": 0.85,
                    "activation": 0.72,
                    "source": "hot"  # "hot" 或 "cold"
                },
                ...
            ]
        """
```

### 并行执行流程

```python
async def retrieve_context(self, query_text, top_k=10, hot_window_minutes=30):
    """并行检索上下文"""
    
    # 1. 计算热窗口秒数
    hot_window_seconds = hot_window_minutes * 60
    
    # 2. 并行执行两条检索路径
    hot_results, cold_results = await asyncio.gather(
        self._retrieve_hot(query_text, hot_window_seconds),
        self._retrieve_cold(query_text, top_k),
        return_exceptions=True
    )
    
    # 异常处理
    if isinstance(hot_results, Exception):
        logger.error(f"热数据检索失败：{hot_results}")
        hot_results = []
    if isinstance(cold_results, Exception):
        logger.error(f"冷数据检索失败：{cold_results}")
        cold_results = []
    
    # 3. 合并结果（去重）
    merged = self._merge_results(hot_results, cold_results)
    
    # 4. 按 activation 降序排序
    merged.sort(key=lambda x: x["activation"], reverse=True)
    
    # 5. 截取 top_k
    return merged[:top_k]
```

### 互斥过滤

```python
async def _retrieve_cold(self, query_text, top_k):
    """冷数据 FAISS 检索（排除热数据）"""
    
    # 1. 获取所有热数据节点 ID（用于互斥过滤）
    hot_node_ids = set()
    for node_id in self._nodes.keys():
        if self.is_recent(node_id, window_seconds=1800):  # 30 分钟
            hot_node_ids.add(node_id)
    
    # 2. FAISS 检索时排除热数据
    cold_results = await self._summary_index.search(
        query_text=query_text,
        top_k=top_k,
        exclude_node_ids=hot_node_ids  # ← 关键：互斥过滤
    )
    
    return cold_results
```

**关键设计**: 冷数据检索时排除热数据，避免重复。

## 3.3 热数据遍历

### 实现逻辑

```python
async def _retrieve_hot(self, query_text: str, window_seconds: int) -> List[Dict]:
    """
    热数据遍历检索
    
    策略：
    1. 筛选 is_recent() == True 的节点
    2. 关键词匹配（BM25 或简单包含）
    3. BFS 扩散激活（发现关联节点）
    4. 返回摘要 + 索引
    """
    results = []
    
    # 1. 筛选热节点
    hot_nodes = []
    for node_id, node in self._nodes.items():
        if self.is_recent(node_id, window_seconds=window_seconds):
            hot_nodes.append(node)
    
    # 2. 关键词匹配
    query_lower = query_text.lower()
    matched_nodes = []
    for node in hot_nodes:
        score = self._compute_keyword_score(node, query_lower)
        if score > 0:
            matched_nodes.append((node, score))
    
    # 3. BFS 扩散激活
    if matched_nodes:
        seed_ids = [n.node_id for n, _ in matched_nodes]
        activations = self.compute_activations(
            seed_node_ids=seed_ids,
            max_depth=2,  # 热数据只扩散 2 跳
            decay=0.5
        )
        
        # 4. 构建结果
        for node, keyword_score in matched_nodes:
            activation = activations.get(node.node_id, 0.0)
            if activation > 0.01:  # 激活阈值
                results.append({
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "label": node.label,
                    "summary": node.metadata.get("summary", ""),
                    "trace_id": node.metadata.get("trace_id"),
                    "score": keyword_score,
                    "activation": activation,
                    "source": "hot"
                })
    
    return results
```

### 关键词评分

```python
def _compute_keyword_score(self, node: GraphNode, query_lower: str) -> float:
    """
    计算关键词匹配分数
    
    策略：
    - label 精确匹配：1.0
    - label 包含：0.8
    - metadata 包含：0.5
    """
    score = 0.0
    
    # label 精确匹配
    if query_lower in node.label.lower():
        score = 0.8
        if query_lower == node.label.lower():
            score = 1.0
    
    # metadata 包含
    else:
        # 搜索 summary
        summary = node.metadata.get("summary", "")
        if summary and query_lower in summary.lower():
            score = 0.6
        
        # 搜索 content
        content = node.metadata.get("content", "")
        if content and query_lower in content.lower():
            score = max(score, 0.5)
        
        # 搜索 ai_response
        ai_response = node.metadata.get("ai_response", "")
        if ai_response and query_lower in ai_response.lower():
            score = max(score, 0.4)
    
    return score
```

## 3.4 冷数据 FAISS 检索

### FAISS 摘要侧车索引

```python
class SummarySidecarIndex:
    """
    MemoryGraph 的 FAISS 摘要侧车索引
    
    只索引 Session / EPISODE 节点的摘要向量 + node_id 指针，
    不向量化记忆节点的实际内容。FAISS 命中后通过 node_id 回到图中
    BFS 下钻获取详情。
    """
    
    def __init__(self, dimension: int = 512, persist_path: str = ""):
        self._dimension = dimension
        self._persist_path = persist_path
        self._store = None          # FAISSVectorStore，延迟初始化
        self._emb_manager = None    # EmbeddingModelManager
        self._node_to_faiss: Dict[str, str] = {}
        self._faiss_to_node: Dict[str, str] = {}
```

### 检索流程

```python
async def _retrieve_cold(self, query_text: str, top_k: int) -> List[Dict]:
    """
    冷数据 FAISS 检索
    
    策略:
    1. 使用 EmbeddingModelManager 编码查询
    2. FAISS 向量检索
    3. 过滤掉热数据（互斥）
    4. BFS 下钻获取子节点详情
    """
    results = []
    
    # 1. 获取所有热数据节点 ID（用于互斥过滤）
    hot_node_ids = set(
        node_id for node_id in self._nodes.keys()
        if self.is_recent(node_id, window_seconds=1800)
    )
    
    # 2. FAISS 检索
    faiss_results = await self._summary_index.search(
        query_text=query_text,
        top_k=top_k * 2,  # 多检索一些以弥补过滤损失
        exclude_node_ids=hot_node_ids
    )
    
    # 3. 构建结果
    for node_id, score in faiss_results:
        node = self.get_node(node_id)
        if not node:
            continue
        
        # 4. BFS 下钻获取子节点
        subgraph = self.get_subgraph_summary(node_id, max_depth=1)
        
        results.append({
            "node_id": node_id,
            "node_type": node.node_type.value,
            "label": node.label,
            "summary": node.metadata.get("summary", ""),
            "trace_id": node.metadata.get("trace_id"),
            "score": score,
            "activation": node.activation,
            "source": "cold",
            "subgraph": subgraph  # BFS 下钻结果
        })
    
    return results[:top_k]
```

## 3.5 BFS 扩散下钻

### 子图摘要提取

```python
def get_subgraph_summary(self, node_id: str, max_depth: int = 2) -> Dict[str, Any]:
    """
    提取以 node_id 为中心的子图摘要
    
    Args:
        node_id: 中心节点 ID
        max_depth: 搜索深度
    
    Returns:
        {
            "center": {...},  # 中心节点信息
            "neighbor_count": 5,
            "type_distribution": {"dialogue": 3, "knowledge": 2},
            "neighbors": [...]  # 最多返回 20 个邻居
        }
    """
    neighbors = self.get_neighbors(node_id, max_depth=max_depth)
    center = self.get_node(node_id)
    if not center:
        return {}
    
    # 统计邻居类型分布
    type_counts: Dict[str, int] = {}
    for n in neighbors:
        type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1
    
    return {
        "center": center.to_dict(),
        "neighbor_count": len(neighbors),
        "type_distribution": type_counts,
        "neighbors": [n.to_dict() for n in neighbors[:20]],  # 最多 20 个
    }
```

### 邻居获取

```python
def get_neighbors(
    self,
    node_id: str,
    edge_types: Optional[Set[EdgeType]] = None,
    max_depth: int = 1,
) -> List[GraphNode]:
    """
    获取邻居节点
    
    Args:
        node_id: 中心节点 ID
        edge_types: 限定边类型 (None = 所有)
        max_depth: 搜索深度
    
    Returns:
        邻居节点列表
    """
    if node_id not in self._nodes:
        return []
    
    visited = {node_id}
    current_layer = [node_id]
    result = []
    
    for _ in range(max_depth):
        next_layer = []
        for nid in current_layer:
            # 出边
            if self._graph.has_node(nid):
                for _, neighbor, data in self._graph.out_edges(nid, data=True):
                    if neighbor in visited:
                        continue
                    if edge_types and EdgeType(data["edge_type"]) not in edge_types:
                        continue
                    visited.add(neighbor)
                    next_layer.append(neighbor)
                    node = self._nodes.get(neighbor)
                    if node:
                        result.append(node)
            
            # 入边（视为无向查询）
            for predecessor, _, data in self._graph.in_edges(nid, data=True):
                if predecessor in visited:
                    continue
                if edge_types and EdgeType(data["edge_type"]) not in edge_types:
                    continue
                visited.add(predecessor)
                next_layer.append(predecessor)
                node = self._nodes.get(predecessor)
                if node:
                    result.append(node)
        
        current_layer = next_layer
    
    return result
```

## 3.6 结果合并与排序

### 去重合并

```python
def _merge_results(self, hot_results: List[Dict], cold_results: List[Dict]) -> List[Dict]:
    """
    合并热数据和冷数据结果（去重）
    
    策略：
    1. 使用 node_id 作为唯一标识
    2. 优先保留热数据结果
    3. 冷数据结果补充
    """
    merged = {}
    
    # 1. 先加入热数据结果
    for result in hot_results:
        node_id = result["node_id"]
        merged[node_id] = result
    
    # 2. 冷数据结果补充（跳过已存在的 node_id）
    for result in cold_results:
        node_id = result["node_id"]
        if node_id not in merged:
            merged[node_id] = result
    
    return list(merged.values())
```

---

# 4. 图注意力机制

## 4.1 注意力机制演进

### 原有架构：1D 线性注意力

**实现逻辑**:

```python
# attention_window_manager.py
class AttentionWindowManager:
    def _compute_weights(self, turn_id: int, current_turn: int) -> float:
        """
        原有架构：1D 线性注意力评分
        
        公式:
        weight = base × time_decay × mode_multiplier
        
        其中:
        - base: 基础权重 (1.0)
        - time_decay: 时间衰减因子 (0.5^(轮次差/20))
        - mode_multiplier: 模式乘数 (工具调用=1.5, 文件操作=1.3)
        """
        turn_diff = current_turn - turn_id
        time_decay = 0.5 ** (turn_diff / 20)  # 每 20 轮衰减一半
        
        mode_multiplier = 1.0
        if turn_id in tool_call_turns:
            mode_multiplier = 1.5
        elif turn_id in file_operation_turns:
            mode_multiplier = 1.3
        
        return 1.0 * time_decay * mode_multiplier
```

**局限性**:

```
问题场景:
第 1 轮：用户："AI MAX 395 是什么？"
        AI："AI MAX 395 是一款高性能处理器..."

第 2-30 轮：讨论其他话题...

第 31 轮：用户："它多少钱？"

1D 注意力评分:
- dialogue:1 的评分 = 1.0 × 0.5^(30/20) × 1.0 = 0.35
- dialogue:30 的评分 = 1.0 × 0.5^(1/20) × 1.0 = 0.97

结果：dialogue:30 评分更高，但内容与"它"无关
      dialogue:1 评分低，但包含"AI MAX 395"的定义
      → 无法正确解析"它"的指代
```

**根本缺陷**:
- ❌ 只能按"时间远近 + 工具关联"筛选
- ❌ 无法发现跨类型的语义关联
- ❌ 对长程依赖不敏感

## 4.2 从 1D 到图注意力的跃升

### 架构对比

```
┌─────────────────────────────────────────────────────────┐
│  原有架构 (1D 线性注意力)                                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [dialogue:1] -- [dialogue:2] -- ... -- [dialogue:31]    │
│       ↓                                                    │
│  评分 = base × time_decay × mode_multiplier              │
│                                                          │
│  缺陷：只能按时间远近筛选，无法发现语义关联               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  新架构 (图注意力)                                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [dialogue:31] ──REFERENCE──> [dialogue:1]               │
│                                    │                     │
│                                 CAUSAL                   │
│                                    │                     │
│                            [knowledge:ai_max_395]        │
│                                    │                     │
│                                 CAUSAL                   │
│                                    │                     │
│                              [knowledge:price]           │
│                                                          │
│  评分 = base × time_decay × graph_boost                 │
│  其中 graph_boost = 1.0 + activation (最大 2.0)          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 4.3 图注意力评分公式

### 完整公式

```python
# attention_window_manager.py
class AttentionWindowManager:
    def _compute_weights_with_graph(self, turn_id: int, current_turn: int) -> float:
        """
        新架构：图注意力评分
        
        公式:
        weight = base × time_decay × mode_multiplier × graph_boost
        
        其中:
        - base: 基础权重 (1.0)
        - time_decay: 时间衰减因子 (0.5^(轮次差/20))
        - mode_multiplier: 模式乘数 (工具调用=1.5, 文件操作=1.3)
        - graph_boost: 图注意力增强 (1.0 + activation, 最大 2.0)
        """
        turn_diff = current_turn - turn_id
        time_decay = 0.5 ** (turn_diff / 20)
        
        mode_multiplier = 1.0
        if turn_id in tool_call_turns:
            mode_multiplier = 1.5
        elif turn_id in file_operation_turns:
            mode_multiplier = 1.3
        
        # 图注意力增强
        graph_boost = 1.0
        if self.memory_graph:
            activation = self.memory_graph.get_node_activation(turn_id)
            graph_boost = 1.0 + min(activation, 1.0)  # 最大 2.0
        
        return 1.0 * time_decay * mode_multiplier * graph_boost
```

### 图注意力增强系数

```python
# graph_boost 计算逻辑
def compute_graph_boost(self, node_id: str) -> float:
    """
    计算图注意力增强系数
    
    公式:
    graph_boost = 1.0 + min(activation, 1.0)
    
    范围：[1.0, 2.0]
    
    其中 activation 来自 BFS 扩散激活算法
    """
    activation = self.get_node_activation(node_id)
    return 1.0 + min(activation, 1.0)

# 示例:
# activation = 0.0 → graph_boost = 1.0 (无增强)
# activation = 0.5 → graph_boost = 1.5 (中等增强)
# activation = 1.0 → graph_boost = 2.0 (最大增强)
```

## 4.4 BFS 扩散激活算法

### 算法详解

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

### 算法可视化

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

### 剪枝优化

```python
# 关键剪枝策略
if propagated < min_activation:
    continue  # 低于阈值，停止传播

if depth >= max_depth:
    continue  # 超过最大深度，停止传播
```

**效果**:
- `min_activation=0.01`: 剪掉弱激活分支，减少计算量
- `max_depth=3`: 限制扩散范围，避免全局遍历

## 4.5 与 AttentionWindowManager 集成

### 集成架构

```
┌─────────────────────────────────────────────────────────┐
│  AttentionWindowManager                                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 接收用户输入                                         │
│  2. 调用 MemoryGraph.retrieve_context()                  │
│  3. 获取种子节点激活值                                   │
│  4. 计算图注意力评分                                     │
│  5. 选择 Top-N 轮次注入上下文                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  MemoryGraph                                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. BFS 扩散激活                                         │
│  2. 返回 Dict[node_id → activation]                      │
│  3. 更新节点激活值                                       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 集成代码

```python
# attention_window_manager.py
class AttentionWindowManager:
    def __init__(self, memory_graph: Optional[MemoryGraph] = None):
        self.memory_graph = memory_graph
        self._activation_cache: Dict[int, float] = {}
    
    async def select_turns(
        self,
        current_turn: int,
        max_turns: int = 20,
    ) -> List[int]:
        """
        选择要注入上下文的轮次
        
        策略:
        1. 计算每轮的注意力评分
        2. 按评分降序排序
        3. 选择 Top-N
        """
        turn_scores = []
        
        for turn_id in range(max(0, current_turn - 100), current_turn):
            score = self._compute_weights_with_graph(turn_id, current_turn)
            turn_scores.append((turn_id, score))
        
        # 按评分降序排序
        turn_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择 Top-N
        selected_turns = [turn_id for turn_id, _ in turn_scores[:max_turns]]
        
        return selected_turns
```

## 4.6 注意力可视化

### 前端展示

```python
# 前端组件（伪代码）
function AttentionVisualization({ nodes, activations }) {
  return (
    <div className="attention-map">
      {nodes.map(node => (
        <NodeCard
          key={node.id}
          node={node}
          activation={activations[node.id]}
          color={getHeatColor(activations[node.id])}
        />
      ))}
    </div>
  );
}

// 热力颜色映射
function getHeatColor(activation: number): string {
  if (activation >= 0.8) return "#FF0000";  // 红色 - 高激活
  if (activation >= 0.5) return "#FFA500";  // 橙色 - 中激活
  if (activation >= 0.2) return "#FFFF00";  // 黄色 - 低激活
  return "#00FF00";  // 绿色 - 无激活
}
```

### 可视化示例

```
注意力热力图:

🔴 task:o1_1 (1.0)  ← 当前聚焦任务
│
├─ 🟠 task:o1_1_1 (0.5)  ← 子任务
├─ 🟠 task:o1_1_2 (0.5)  ← 子任务
│
└─ 🟠 file:weather.py (0.4)  ← 相关文件
    │
    └─ 🟡 knowledge:python_requests (0.14)  ← 知识文档

颜色说明:
🔴 红色：activation >= 0.8 (高激活)
🟠 橙色：0.5 <= activation < 0.8 (中激活)
🟡 黄色：0.2 <= activation < 0.5 (低激活)
🟢 绿色：activation