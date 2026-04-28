"""Replace TSD Chapter 8 with updated MemoryGraph architecture content."""
import re
import sys

NEW_CHAPTER_8 = r'''# 第 8 章：记忆系统架构

## 8.1 架构演进与当前核心

### 8.1.1 三代记忆架构演进

| 代际 | 核心组件 | 存储方式 | 状态 |
|------|---------|---------|------|
| **Gen-1** | ShortTermMemory + EpisodicMemory | 共享池分区 + JSON 索引 | 仍保留运行，提供对话缓存与情景存储 |
| **Gen-2** | 三库 RAG (ExperienceRAG / MemoryRAG / KnowledgeRAG) | FAISS 向量库 + BAAI/bge-small-zh-v1.5 | 作为冷数据检索路径运行 |
| **Gen-3 (当前)** | **MemoryGraph 统一异构图** | NetworkX DiGraph + 多维标签 + 图算法 | **主架构**，Hub 角色 |

> **关键说明**：Gen-1 的 `ShortTermMemory` 和 `EpisodicMemory` 并未废弃，仍作为独立模块运行。`ShortTermMemory` 负责最近 20 轮对话缓存（LRU + TTL 1h），`EpisodicMemory` 负责情景记忆存储与事件流管理。但所有记忆的创建、关联、检索、衰减统一通过 MemoryGraph 调度。

### 8.1.2 当前核心：MemoryGraph 统一异构图

**核心文件**: `zulong/memory/memory_graph.py` (2762 行)

MemoryGraph 是 Gen-3 记忆架构的中枢。它将所有记忆实体建模为**有向异构图** (NetworkX DiGraph)，通过 **9 种节点类型 + 7 种边类型 + 3 维标签**实现统一管理。

**架构总览**:
```
MemoryGraph (Hub)
├── 图结构层 ─── NetworkX DiGraph (节点 + 边 + 属性)
├── 标签系统 ─── Temperature / Importance / TimeScope
├── 算法引擎 ─── BFS 扩散激活 / Hebbian 学习 / 艾宾浩斯衰减
├── 适配器层 ─── 7 个领域适配器 (写入入口)
├── 检索引擎 ─── 双路径 (热数据图遍历 + 冷数据 FAISS)
└── 进化引擎 ─── MemoryStrength / Consolidator / Forgetter / LLM Reviewer
```

---

## 8.2 MemoryGraph 统一图架构

### 8.2.1 节点类型 (NodeType)

**核心文件**: `zulong/memory/memory_graph.py` — `class NodeType(str, Enum)`

| NodeType | 含义 | 典型来源 |
|----------|------|---------|
| `DIALOGUE` | 对话轮次 | DialogueAdapter |
| `TASK` | 任务节点 | TaskGraphAdapter |
| `KNOWLEDGE` | 知识片段 | KnowledgeGraphAdapter |
| `EXPERIENCE` | 经验记录 | ExperienceAdapter |
| `EPISODE` | 情景事件 | EpisodeAdapter |
| `PERSON` | 人物画像 | PersonProfileAdapter |
| `CONCEPT` | 抽象概念 | KnowledgeGraphAdapter |
| `EMOTION` | 情感状态 | EpisodeAdapter |
| `SYSTEM_EVENT` | 系统事件 | BaseGraphAdapter |

### 8.2.2 边类型 (EdgeType)

**核心文件**: `zulong/memory/memory_graph.py` — `class EdgeType(str, Enum)`

| EdgeType | 语义 | 权重范围 |
|----------|------|---------|
| `TEMPORAL` | 时间顺序关系 | 0.0–1.0 |
| `CAUSAL` | 因果关系 | 0.0–1.0 |
| `SEMANTIC` | 语义相似 | 0.0–1.0 |
| `REFERENCE` | 引用/地址继承 | 0.0–1.0 |
| `ASSOCIATION` | 联想关联 (Hebbian 自动创建) | 0.0–1.0 |
| `HIERARCHICAL` | 层级从属 | 0.0–1.0 |
| `DEPENDENCY` | 依赖关系 | 0.0–1.0 |

### 8.2.3 多维标签系统

每个节点携带 3 维标签，用于检索过滤和进化决策：

**温度标签 (Temperature)**:
| 值 | 含义 | 转换条件 |
|----|------|---------|
| `HOT` | 活跃数据，优先检索 | 最近访问 / 高频引用 |
| `WARM` | 一般数据 | 默认状态 |
| `COLD` | 低活数据，进入 RAG 检索路径 | 长时间未访问 |

**重要性标签 (Importance)** — 6 级:
| 值 | 含义 | 半衰期 |
|----|------|--------|
| `MUST_REMEMBER` | 必须记住 | 720h |
| `VERY_IMPORTANT` | 非常重要 | 360h |
| `IMPORTANT` | 重要 | 168h |
| `NORMAL` | 普通 | 72h |
| `TRIVIAL` | 琐碎 | 24h |
| `DISPOSABLE` | 可丢弃 | 6h |

**时间范围标签 (TimeScope)**:
| 值 | 含义 |
|----|------|
| `RECENT` | 最近的记忆 |
| `NON_RECENT` | 非最近的记忆 |

### 8.2.4 图算法引擎

MemoryGraph 内置 3 种图算法：

**1. BFS 扩散激活 (Spreading Activation)**

从种子节点出发，沿边权重加权扩散，收集相关节点：

```python
def spreading_activation(self, seed_ids, max_depth=3, min_activation=0.1):
    """
    BFS 扩散：activation(neighbor) = activation(current) × edge_weight × decay
    - max_depth: 最大扩散层数
    - min_activation: 最小激活阈值 (低于此值停止扩散)
    """
```

**2. Hebbian 学习 (共激活强化)**

当两个节点在同一上下文中被同时访问时，自动强化关联：

```python
def hebbian_update(self, node_a, node_b, boost=0.1):
    """
    如果 A-B 之间已有边 → weight += boost (上限 1.0)
    如果 A-B 之间无边 → 自动创建 ASSOCIATION 边 (初始 weight=boost)
    """
```

**3. 艾宾浩斯衰减 (Ebbinghaus Decay)**

```
decayed_weight = weight × exp(-elapsed_hours × ln(2) / half_life)

其中:
- weight: 当前权重
- elapsed_hours: 距离上次访问的小时数
- half_life: 由 Importance 标签决定 (见 8.2.3 表)
```

### 8.2.5 双路径上下文检索

**核心方法**: `MemoryGraph.retrieve_context(query, task_id=None)`

```
检索请求 (query + task_id)
  ↓
┌─────────────────────────────────────────────┐
│  路径 A：热数据图遍历                         │
│  1. 种子选取 → 与 query 语义最近的 HOT 节点    │
│  2. BFS 扩散激活 (max_depth=3)               │
│  3. 按 activation 得分排序                    │
├─────────────────────────────────────────────┤
│  路径 B：冷数据 FAISS 检索                    │
│  1. EmbeddingManager.encode_query(query)     │
│  2. 各 RAG 库 top_k 检索                     │
│  3. 按相似度得分排序                          │
└─────────────────────────────────────────────┘
  ↓
合并 + 去重 + 统一排序
  ↓
返回 context_items[]
```

---

## 8.3 图适配器层

**核心文件**: `zulong/memory/graph_adapters.py` (1326 行)

适配器是**所有记忆写入 MemoryGraph 的唯一入口**，负责将业务数据转换为图节点/边：

| 适配器 | 职责 | 创建的 NodeType |
|--------|------|----------------|
| `BaseGraphAdapter` | 基类，提供 add_node / add_edge 统一接口 | — |
| `TaskGraphAdapter` | 任务图节点管理 (创建/状态更新/地址继承) | TASK |
| `DialogueAdapter` | 对话轮次记录 + TEMPORAL 时间链 | DIALOGUE |
| `KnowledgeGraphAdapter` | 知识片段 + CONCEPT 概念节点 + SEMANTIC 边 | KNOWLEDGE, CONCEPT |
| `EpisodeAdapter` | 情景事件 + 情感状态关联 | EPISODE, EMOTION |
| `PersonProfileAdapter` | 用户画像构建与更新 | PERSON |
| `ExperienceAdapter` | 任务成功/失败经验记录 | EXPERIENCE |

**数据流示例** (对话写入):
```
用户输入："帮我分析这段代码"
  ↓
DialogueAdapter.record_dialogue(user_text, ai_text)
  ↓
创建 DIALOGUE 节点 (turn_id, timestamp, text, importance)
  ↓
创建 TEMPORAL 边 → 上一轮对话节点
  ↓
Hebbian 更新 → 相关 KNOWLEDGE/TASK 节点
  ↓
节点已在 MemoryGraph 中
```

---

## 8.4 RAG 向量检索系统 (4 库)

### 8.4.1 架构概览

**核心文件**: `zulong/memory/rag_manager.py` (333 行) + `zulong/memory/rag_libraries.py` (759 行)

```
RAG Manager
├── ExperienceRAG (经验库)
│   ├── 任务成功/失败案例
│   ├── 技能使用经验
│   └── 系统提示词归类
│
├── MemoryRAG (记忆库)
│   ├── 短期记忆 (< 1 小时)
│   ├── 中期记忆 (1 小时–1 天)
│   ├── 长期记忆 (> 1 天)
│   └── 记忆类型：上下文/事件/对话
│
├── KnowledgeRAG (知识库)
│   ├── 领域知识：导航/操作/视觉/对话
│   ├── 确定性分级：已确认/不确定
│   └── 知识验证机制
│
└── **ToolRAG (工具库)** ← v2.7 新增
    ├── 冷工具发现 (非 CoreToolManager 热加载工具)
    ├── 工具描述向量化索引
    └── 按 query 语义匹配推荐工具
```

> **ToolRAG** (`zulong/memory/tool_rag.py`, 233 行) 是 v2.4 新增的第 4 个 RAG 库，用于冷工具的语义发现。当 CoreToolManager 的热工具集未命中时，通过向量检索从完整工具库中推荐匹配工具。

### 8.4.2 向量化引擎

**核心文件**: `zulong/memory/embedding_manager.py` (339 行)

| 配置项 | 值 |
|--------|-----|
| 模型名称 | `BAAI/bge-small-zh-v1.5` |
| 输出维度 | 512 |
| 量化 | 4bit (节省显存) |
| 设备 | CPU 优先 (可选 GPU) |
| 向量索引 | FAISS (`IndexFlatL2` / `IndexIVFFlat`) |

**查询优化**:
```python
# 查询前缀 (提升检索效果)
query_text = f"为这个句子生成表示以用于检索：{text}"

# 文档前缀
doc_text = f"为这个句子生成表示以用于存储：{text}"
```

### 8.4.3 相似性搜索流程

```
查询："如何创建子任务"
  ↓
EmbeddingManager.encode_query()
  ↓
512 维向量 (已归一化)
  ↓
FAISS.search(query_vector, top_k=5) × 4 库
  ↓
L2 距离 → 相似度转换：similarity = 1 / (1 + distance)
  ↓
跨库合并 + 去重 + 排序
  ↓
返回 Top-K 文档
```

---

## 8.5 记忆自进化机制

### 8.5.1 记忆强度模型

**核心文件**: `zulong/memory/memory_evolution.py` (588 行)

进化系统包含 4 个核心类：

| 类 | 职责 |
|----|------|
| `MemoryStrength` | 记忆强度计算 (艾宾浩斯衰减 + 访问强化) |
| `MemoryConsolidator` | 短期→长期巩固 (重要性 > 0.7 / 访问 >= 2 / must_remember) |
| `MemoryForgetter` | 遗忘清理 (强度 < 0.1 且非 must_learn) |
| `MemoryEvolutionEngine` | 统一调度：巩固 → 衰减 → 遗忘 → 清理 |

**记忆强度公式** (艾宾浩斯遗忘曲线):
```
R = e^(-t/S)

其中:
- R: 记忆保留率
- t: 经过时间 (小时)
- S: 强度系数 (初始强度 × 10)
```

**强化机制**:
```python
def reinforce(self, boost: float = 0.2):
    """每次访问增强初始强度"""
    self.access_count += 1
    self.initial_strength *= (1 + boost)
```

**遗忘判断**:
```python
def should_forget(self, threshold: float = 0.1):
    """强度<0.1 时遗忘"""
    return self.current_strength < threshold
```

### 8.5.2 LLM 记忆审查员

**核心文件**: `zulong/memory/llm_memory_reviewer.py` (509 行)

LLM 审查员在记忆生命周期关键节点介入，通过 LLM 判断记忆的去留：

**审查时机**:
| 时机 | 触发条件 |
|------|---------|
| `PRE_STORE` | 新记忆写入前 → 判断是否值得存储 |
| `PRE_EVICT` | 遗忘前 → 确认是否真的可以丢弃 |
| `PERIODIC_REVIEW` | 定期巡检 → 批量优化存量记忆 |

**审查决策**:
| 决策 | 含义 |
|------|------|
| `KEEP` | 保留不变 |
| `DISCARD` | 丢弃 |
| `COMPRESS` | 压缩 (保留关键信息，减少冗余) |
| `PROMOTE` | 提升重要性等级 |
| `MERGE` | 与相似记忆合并 |

---

## 8.6 跨会话记忆持久化

### 8.6.1 MemoryGraph 持久化

**存储路径**: `./data/memory_graph/`

```python
# 保存
memory_graph.save(path)  # → graph.json + metadata.json

# 加载
memory_graph.load(path)  # → 恢复完整图结构 + 节点属性 + 边权重
```

### 8.6.2 共享池快照机制

**核心文件**: `zulong/infrastructure/shared_memory_pool.py`

| 配置项 | 值 |
|--------|-----|
| 快照间隔 | 30 秒 |
| 最大保留数 | 20 |
| 压缩格式 | gzip (~10:1) |
| 存储路径 | `./data/shared_memory_pool/` |

**快照文件格式**: `snapshot_20260404_185711.json.gz`

包含所有分区数据:
- Raw Zone (原始数据)
- Feature Zone (特征数据)
- Memory Zone (记忆数据)
- System Zone (系统数据)

**数据恢复流程**:
```
系统启动
  ↓
await SharedMemoryPool.get_instance()
  ↓
检查最新快照 → _load_snapshot()
  ↓
解压 + 反序列化 → 恢复所有分区数据
  ↓
启动后台任务:
  - _cleanup_loop() (自动清理)
  - _snapshot_loop() (定期保存)
  ↓
数据可用 (跨会话记忆恢复完成)
```

---

## 8.7 CircuitBreaker 上下文窗口监控

**核心文件**: `zulong/l2/circuit_breaker.py` (406 行)

CircuitBreaker 提供 5 信号智能循环检测，其中**信号 4** 专门监控上下文窗口压力：

**信号 4：上下文窗口压力**

```python
# 当前配置
_context_window_size = 65536  # 假定上下文窗口大小

# 压力计算
pressure = estimated_tokens / _context_window_size

# 阈值
if pressure > 0.85:  # 上下文占用超过 85%
    signal_4 = True   # 触发上下文压力信号
```

**5 信号综合判断**:

| 信号 | 检测内容 | 阈值 |
|------|---------|------|
| Signal 1 | 重复输出检测 | 连续 3 次相似度 > 0.9 |
| Signal 2 | 进度停滞检测 | 连续 5 轮无有效进展 |
| Signal 3 | 错误循环检测 | 相同错误出现 3 次 |
| Signal 4 | 上下文窗口压力 | token 占比 > 85% |
| Signal 5 | 时间超限检测 | Planning: 180s yellow / 300s red |

> **注意**：当前 `_context_window_size` 硬编码为 65536，但实际 Ollama 模型可能使用不同的上下文窗口大小。后续版本将通过 Ollama `/api/show` API 自动探测模型实际上下文窗口并同步此值。

---

'''

TSD_PATH = r'd:\AI\project\zulong_beta4\docs\TSD_v2.4.md'

with open(TSD_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace Chapter 8
# Pattern: from "# 第 8 章：记忆系统架构" to just before "# 第 9 章：数据存储架构"
pattern = r'# 第 8 章：记忆系统架构.*?(?=# 第 9 章：数据存储架构)'

match = re.search(pattern, content, flags=re.DOTALL)
if not match:
    print("ERROR: Could not find Chapter 8 boundaries")
    sys.exit(1)

old_text = match.group(0)
print(f"Found Chapter 8: {len(old_text)} chars, from pos {match.start()} to {match.end()}")
print(f"First 100 chars: {old_text[:100]}")
print(f"Last 100 chars: {old_text[-100:]}")

new_content = re.sub(pattern, NEW_CHAPTER_8, content, count=1, flags=re.DOTALL)

if new_content == content:
    print("ERROR: No replacement made")
    sys.exit(1)

with open(TSD_PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"\nSUCCESS: Chapter 8 replaced")
print(f"Old file size: {len(content)} chars")
print(f"New file size: {len(new_content)} chars")
print(f"Diff: {len(new_content) - len(content):+d} chars")
