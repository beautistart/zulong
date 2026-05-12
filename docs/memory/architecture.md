# 祖龙 (ZULONG) 记忆系统完整架构文档

> **文档定位**: 任何 Agent 阅读此文档后，能完整准确理解祖龙系统的记忆板块及相关协同运行机制。
> **最后更新**: 2026-04-23
> **版本**: v3.0 (含设计理念、核心模块运行机制详解、修复计划)

---

## 目录

1. [整体架构概述](#1-整体架构概述)
2. [记忆系统演进历程](#2-记忆系统演进历程)
3. [MemoryGraph 记忆图谱核心](#3-memorygraph-记忆图谱核心)
4. [图适配器与后端投射](#4-图适配器与后端投射)
5. [多维标签体系](#5-多维标签体系)
6. [地址继承系统](#6-地址继承系统)
7. [基于图记忆的自主动态注意力机制](#7-基于图记忆的自主动态注意力机制)
8. [思维深度导航](#8-思维深度导航)
9. [记忆遗忘与剪枝机制](#9-记忆遗忘与剪枝机制)
10. [记忆检索系统](#10-记忆检索系统)
11. [记忆巩固与自演进](#11-记忆巩固与自演进)
12. [经验数据与事件记忆区分](#12-经验数据与事件记忆区分)
13. [任务恢复机制](#13-任务恢复机制)
14. [MG 能力暴露: 系统层 vs 工具层](#14-mg-能力暴露-系统层-vs-工具层)
15. [各系统协同运行机制](#15-各系统协同运行机制)
16. [关键文件索引](#16-关键文件索引)
17. [关键设计决策](#17-关键设计决策)
18. [核心设计理念：3D 记忆城市](#18-核心设计理念3d-记忆城市)
19. [四大核心模块运行机制详解](#19-四大核心模块运行机制详解)
20. [当前已知问题与分阶段修复计划](#20-当前已知问题与分阶段修复计划)
21. [调试参考：五大目标效果与实现路径](#21-调试参考五大目标效果与实现路径)

---

## 1. 整体架构概述

### 1.1 祖龙系统分层架构

```
+-------------------------------------------------------------+
|                      L3 专家层                                |
|  TTS 专家 | 运动控制专家 | 其他领域专家                      |
+-------------------------------------------------------------+
|                   L2 认知层 (记忆系统核心)                     |
|  InferenceEngine | TaskGraph | MemoryGraph | RAG             |
|  +-- Function Calling 工具自主调用                            |
|  +-- MCTS 长期规划                                           |
|  +-- 记忆图谱: BFS 扩散激活 + 赫布学习 + 突触修剪            |
+-------------------------------------------------------------+
|              L1-B 调度门控层 (注意力控制)                      |
|  Gatekeeper | AttentionController                            |
|  +-- 三层注意力: L0 采集 -> L1 静默 -> L2 交互               |
|  +-- 中断处理: 冻结 -> 重组 -> 注入                          |
+-------------------------------------------------------------+
|                L1-A/C 感知层                                  |
|  ReflexController | VisionProcessor | AudioProcessor         |
+-------------------------------------------------------------+
|                    L0 传感器层                                 |
|  CameraDevice | MicrophoneDevice                             |
+-------------------------------------------------------------+
         |
         v
+-------------------------------------------------------------+
|                    基础设施层                                  |
|  EventBus | SharedMemoryPool | DataIngestion                 |
|  SkillPackRuntime | WebSocket Server                         |
+-------------------------------------------------------------+
```

### 1.2 记忆系统在架构中的定位

MemoryGraph 是 L2 认知层的**记忆中枢**，它不是一个独立模块，而是贯穿多个系统层级的**统一集成层**。所有记忆子系统（对话历史、任务状态、知识图谱、经验库、人物档案等）都通过**适配器模式**投射为图中的节点和边，共享同一个 NetworkX DiGraph 实例。

**核心类比**: LLM 是大脑皮层（负责推理），MemoryGraph 是海马体（负责记忆索引和联想）。

### 1.3 记忆系统核心子系统一览

| 子系统 | 职责 | 核心文件 |
|--------|------|----------|
| **MemoryGraph** | 统一异构图存储，BFS/FAISS 检索，衰减/修剪 | `zulong/memory/memory_graph.py` (102 KB) |
| **图适配器** (6 个) | 将各类后端数据投射为图节点 | `zulong/memory/graph_adapters.py` (49 KB) |
| **短期记忆** | 基于共享池的对话缓存，向量检索加速 | `zulong/memory/short_term_memory.py` (52 KB) |
| **临时记忆** | 对话摘要、分级读取 | `zulong/memory/episodic_memory.py` (29 KB) |
| **RAG 管理器** | 统一管理 4 个 RAG 库 (经验/记忆/知识/工具) | `zulong/memory/rag_manager.py` (11 KB) |
| **记忆演进** | 记忆巩固 + 遗忘 + 强度管理 | `zulong/memory/memory_evolution.py` (20 KB) |
| **LLM 审查器** | 剪枝前 LLM 守卫 | `zulong/memory/llm_memory_reviewer.py` (19 KB) |
| **注意力控制器** | 事件优先级、中断冻结、焦点管理 | `zulong/l1b/attention_controller.py` (14 KB) |
| **焦点导航工具** | LLM 通过 FC 自主调整注意力焦点 | `zulong/tools/attention_tool.py` (8 KB) |
| **任务挂起/恢复** | 任务断点续传、环境重评估 | `zulong/l2/task_suspension.py` (13 KB) |
| **环境快照** | 任务恢复前的环境变化检测 | `zulong/l2/environment_snapshot.py` (11 KB) |
| **恢复通知器** | 启动时扫描可恢复任务 | `zulong/l2/recovery_notifier.py` (6 KB) |
| **模型预加载器** (v2.8) | 系统启动时后台加载模型，避免冷启动延迟 | `zulong/utils/model_preloader.py` (8 KB) |

### 1.4 记忆系统主数据流

```
用户输入
   |
   v
L1-B Gatekeeper (事件过滤 + 注意力判断)
   |
   v
SYSTEM_L2_COMMAND 事件 --> EventBus
   |
   v
InferenceEngine._on_l2_command()
   |
   v
_process_with_memory()
   |--- retrieve_context() ---> MemoryGraph 双路径检索 (热遍历 + 冷 FAISS)
   |--- _retrieve_from_rag() --> RAGManager 知识/经验检索
   |--- get_focus_path_summary() --> 思维焦点注入 system prompt
   |
   v
_build_messages_with_history() 构建上下文
   |
   v
LLM 推理 (vLLM) + Function Calling 循环
   |
   v
_update_memory()
   |--- DialogueAdapter.add_round() --> 创建对话节点
   |--- DialogueAdapter._detect_importance() --> 自动标注重要度
   |--- DialogueAdapter.finalize_round() --> FAISS 摘要索引
   |--- assign_session_by_similarity() --> 话题边界检测 + session 分配
   |--- _propagate_address_to_tasks() --> 地址继承传播
   |
   v
后台定时器 (每 30 分钟):
   |--- decay_and_prune() --> 艾宾浩斯衰减 + 边修剪
   |--- run_importance_review() --> 自动重要度提升
   |--- update_temperature() --> 温度标签更新
   |--- save() --> JSON 持久化
```

---

## 2. 记忆系统演进历程

### 2.1 第一代: 三库分离架构

最初的记忆系统使用三个独立 RAG 库:

```
ExperienceRAG ── 经验库 (复盘提取的教训/最佳实践)
MemoryRAG     ── 记忆库 (对话固化的事件记忆)
KnowledgeRAG  ── 知识库 (事实性知识/概念/原理)
```

**问题**: 记忆孤岛，无法发现跨类型关联。

### 2.2 第二代: 增强型三级记忆架构

引入分层记忆检索:

```
工作记忆 (最近 2 轮完整对话) --> 时间序列直接注入
临时记忆 (EpisodicMemory 摘要索引) --> 语义检索 + 分级读取
长期记忆 (ShortTermMemory/RAG 向量) --> 向量检索
```

**改进**: 支持摘要检索和按需读取，动态容量适配 (4K/8K/128K 模型)。

**问题**: MemoryGraph 只写不读 -- 对话写入了图节点，但 LLM 上下文注入仍完全依赖旧系统。

### 2.3 第三代 (当前): MemoryGraph 统一记忆中枢

**设计目标**: MemoryGraph 成为唯一的记忆读写中枢。

**核心改造**:
1. MemoryGraph 从"只写"变为"读写统一" -- `retrieve_context()` 作为 LLM 上下文注入的主入口
2. 引入多维标签体系（温度/重要度/时间段）替代硬分区
3. 并行检索策略（热数据遍历 + 冷数据 FAISS）在同一张图上通过互斥过滤实现
4. 6 个适配器将所有后端系统投射到同一张图上

**旧组件处置**:

| 组件 | 处置 | 说明 |
|------|------|------|
| `MemoryRAG` | 废弃 | 被 DIALOGUE 节点 + FAISS 侧车索引替代 |
| `ShortTermMemory` | 废弃 | 对话存储/向量检索被图节点替代 |
| `EpisodicMemory` | 废弃 | 摘要逻辑迁入 EpisodeAdapter，分级读取由 BFS 下钻实现 |
| `conversation_history` | 废弃 | 被 DIALOGUE 节点 (Round/Sub-dialogue) 替代 |
| `ExperienceRAG` | 保留，被动化 | 取消自动注入，注册为 FC 工具按需调用 |
| `KnowledgeRAG` | 保留，打通图谱 | 通过 KNOWLEDGE 节点的 backend_ref 与图谱联动 |
| `ToolRAG` | 保留 | 与记忆系统无关，属于工具能力索引 |

---

## 3. MemoryGraph 记忆图谱核心

### 3.1 核心概念

MemoryGraph 是一张**完整的异构类型图** (NetworkX DiGraph)。所有记忆子系统统一投射为节点和边。整张图是连通的，BFS 扩散激活可以在所有节点间自由传播。

**关键特性**: 图是一个整体，不做任何物理分割。概念上的"宏空间 vs 任务空间"只是区域划分，不是物理隔离。

### 3.2 节点类型 (NodeType) -- 9 种

```python
class NodeType(Enum):
    TASK = "task"              # 来自 TaskGraph 的任务节点
    DIALOGUE = "dialogue"      # 对话轮次 (session/round/sub_dialogue)
    KNOWLEDGE = "knowledge"    # 知识图谱实体
    EXPERIENCE = "experience"  # 经验 RAG 文档
    EPISODE = "episode"        # 情景记忆摘要
    PERSON = "person"          # 人物档案
    FILE = "file"              # 文件引用
    CONCEPT = "concept"        # 概念节点
    DOCUMENT = "document"      # 文档（预留）
```

### 3.3 边类型 (EdgeType) -- 7 种

| 边类型 | 含义 | 是否受保护 | 衰减规则 | 典型场景 |
|--------|------|-----------|----------|----------|
| `HIERARCHY` | 父子层级关系 | protected | 永不修剪 | session->round, task->subtask |
| `DEPENDENCY` | 数据依赖 | protected | 永不修剪 | task1 依赖 task2 完成 |
| `TEMPORAL` | 时间序列 | protected | 永不修剪 | round1->round2 按时间串联 |
| `REFERENCE` | 跨类型引用 | 否 | 按重要度衰减 | dialogue<->task 双向关联 |
| `SEMANTIC` | 语义相似 | 否 | 按重要度衰减 | embedding 余弦相似度 > 0.7 自动创建 |
| `CAUSAL` | 因果关系 | 否 | 按重要度衰减 | 事件 A 导致事件 B |
| `ASSOCIATION` | 赫布学习产生 | 否 | 按重要度衰减 | 共激活 >= 3 次自动创建 |

### 3.4 核心数据结构

```python
@dataclass
class GraphNode:
    node_id: str               # 全局唯一标识，带类型前缀 (如 "dialogue:round_42")
    node_type: NodeType        # 节点类型
    label: str                 # 人类可读标签
    activation: float          # 当前激活水平 (0.0-1.0)
    created_at: float          # 创建时间戳
    last_accessed: float       # 最后访问时间戳
    access_count: int          # 访问次数
    backend_ref: str           # 后端来源指针 (如 "knowledge_rag:doc_123")
    metadata: Dict[str, Any]   # 扩展属性 (含多维标签)
    # metadata 标准字段:
    #   "temperature": "hot"/"warm"/"cold"
    #   "importance": "trivial"/"normal"/"identity"/"fact"/"important"/"must_remember"
    #   "content": str           用户输入全文
    #   "ai_response": str       AI 回复全文
    #   "goal": str              对话目标
    #   "detected_entities": []  自动检测到的实体列表
    #   "memory_strength": {}    衰减追踪结构
    #   "importance_history": [] 重要度变更历史

# 边属性存储在 NetworkX 图的边字典中:
edge_attrs = {
    "edge_type": EdgeType.REFERENCE,
    "weight": 0.85,            # 0.0-1.0+, 赫布增强 + 衰减影响
    "created_at": float,
    "last_activated": float,
    "protected": bool,         # True = 永不修剪 (结构边)
}
```

### 3.5 核心算法

#### 3.5.1 加权 BFS 扩散激活

```python
def compute_activations(self, seed_node_ids, max_depth=3, decay=0.5, min_activation=0.01):
    """从种子节点出发，沿边权衰减传播激活值"""
    # 初始化: 种子节点 activation = 1.0
    # 传播公式: activation[neighbor] = activation[current] x edge_weight x decay
    # 终止条件: activation < min_activation 或 depth > max_depth
    # 示例 (decay=0.5):
    #   Seed A: 1.0
    #   1-hop B (weight=1.0): 1.0 x 1.0 x 0.5 = 0.5
    #   2-hop C (weight=0.8): 0.5 x 0.8 x 0.5 = 0.2
    #   3-hop D (weight=0.6): 0.2 x 0.6 x 0.5 = 0.06
    # 返回: {node_id: activation_score}
    # 副作用: 记录被激活的边对 (_last_activated_edges) 供赫布学习使用
```

#### 3.5.2 赫布学习 (Hebbian Learning)

```python
def hebbian_strengthen(self):
    """共激活边强化 + 自动创建 ASSOCIATION 边"""
    # 强化公式: new_weight = old_weight + eta * (1 - old_weight)
    # eta = 0.1 (固定学习率)
    # 渐近趋近 1.0, 永不超出
    # protected 边跳过 (保持结构完整性)
    #
    # 自动关联:
    # 共激活计数器 _coactivation_counter[pair] += 1
    # 当 count >= hebb_threshold (默认 3) 且无直接边:
    #   自动创建 ASSOCIATION 边 (weight=0.5)
```

**重要发现**: 当前实现中 `hebbian_strengthen()` 虽然完整实现，但在 `compute_activations()` 后缺少自动调用链路。共激活计数器正确记录了数据，但边强化需要显式触发。

#### 3.5.3 突触修剪 (艾宾浩斯衰减)

```python
def decay_and_prune(self):
    """衰减公式: decayed = weight x exp(-elapsed_hours x ln(2) / half_life)"""
    # 衰减后:
    #   < 0.05 --> 删除边
    #   0.05~0.15 --> 标记濒危 (可送 LLM 审查)
    #   >= 0.15 --> 更新权重
    # protected 边永不衰减 (HIERARCHY/DEPENDENCY/TEMPORAL)
```

#### 3.5.4 语义近邻自动发现

```python
def discover_semantic_neighbors(self, node_id, top_k=5, threshold=0.7):
    """通过 embedding 余弦相似度自动发现近邻并创建 SEMANTIC 边"""
    # 1. 获取节点 embedding
    # 2. 对比所有已有 embedding (numpy 余弦相似度)
    # 3. 相似度 > 0.7 的节点对自动创建 SEMANTIC 边
```

#### 3.5.5 后端溯源

```python
def resolve_backend_ref(self, node_id) -> Optional[Dict]:
    """解析 backend_ref 指针, 回到原始后端获取完整数据"""
    # 支持前缀:
    #   "knowledge_rag:{doc_id}" --> KnowledgeRAG
    #   "experience_rag:{doc_id}" --> ExperienceRAG
    #   "task_graph:{graph_id}" --> TaskGraph
    # BFS 激活到 KNOWLEDGE 节点时, 自动通过此方法获取完整知识文档
```

---

## 4. 图适配器与后端投射

MemoryGraph 通过 6 个适配器，将各类后端系统的数据**单向投射**为图中的节点和边。适配器是只读桥接层，不修改后端数据。

### 4.1 适配器全景

| 适配器 | 数据来源 | 投射的节点类型 | 投射的边类型 |
|--------|---------|---------------|-------------|
| `TaskGraphAdapter` | TaskGraph | TASK, FILE | HIERARCHY (protected), DEPENDENCY (protected), REFERENCE |
| `DialogueAdapter` | 对话数据 | DIALOGUE (session/round/sub_dialogue) | HIERARCHY, TEMPORAL, REFERENCE |
| `KnowledgeGraphAdapter` | KnowledgeGraph | PERSON, CONCEPT, KNOWLEDGE | CAUSAL, ASSOCIATION, REFERENCE |
| `EpisodeAdapter` | EpisodicMemory | EPISODE | TEMPORAL |
| `PersonProfileAdapter` | PersonProfileManager | PERSON | REFERENCE |
| `ExperienceAdapter` | ExperienceRAG | EXPERIENCE | (无自动边) |

### 4.2 DialogueAdapter (最关键的适配器)

DialogueAdapter 负责对话数据的完整生命周期管理:

**核心方法**:

| 方法 | 触发时机 | 功能 |
|------|---------|------|
| `ensure_session()` | 用户消息到达 | 话题边界检测: bigram 重叠率 + bound_task_id 匹配, 决定创建新 session 还是复用旧 session |
| `add_round()` | 每轮对话 | 创建 round 节点 + 自动重要度检测 + 建立 round->task 的 REFERENCE 边 |
| `add_sub_dialogue()` | FC 工具调用 | 创建 agent_turn 节点 + REFERENCE 到相关 task |
| `finalize_round()` | 对话轮完成 | 归档轮次 + 将 round_summary **写入 FAISS 摘要侧车索引** |
| `assign_session_by_similarity()` | L2 推理后 | embedding 相似度匹配分配 session, 调用地址继承传播 |
| `_detect_importance()` | 节点创建时 | 正则匹配自动标注: "我叫X"->IDENTITY, "帮我记住"->MUST_REMEMBER, "嗯/好的"->TRIVIAL |
| `_link_to_knowledge()` | round 创建后 | 自动链接到 KG 中的 KNOWLEDGE/PERSON/CONCEPT 节点 |

**话题边界检测逻辑** (`_is_same_topic()`):
1. 检查 session 是否绑定了 task (bound_task_id)，如果当前有活跃任务则保持同一 session
2. 计算 bigram 重叠率 >= 20% 判定为同一话题
3. 否则创建新 session，并建立 TEMPORAL 边连接到上一个 session

### 4.3 TaskGraphAdapter

**同步模式**:
- `sync()`: 全量同步 -- 遍历 TaskGraph 所有节点/边, 投射为 TASK/FILE 节点 + HIERARCHY/DEPENDENCY 边
- `incremental_sync()`: 增量同步 -- 监听 TaskGraph 变更事件 (node_add/update, h_edge_add, d_edge_add)

**关键行为**:
- HIERARCHY 和 DEPENDENCY 边自动设为 `protected=True`
- 任务节点通过 `backend_ref` 指向原始 TaskGraph
- 节点地址继承: 子任务 full_path = 父任务 full_path + "/" + 子 ID

---

## 5. 多维标签体系

MemoryGraph 使用**三组正交标签**替代传统的硬分区 (L1/L2/L3)，实现灵活的记忆分类和检索路由:

### 5.1 标签定义

| 维度 | 存储方式 | 可选值 | 说明 |
|------|---------|-------|------|
| **温度** (temperature) | metadata 存储, 动态更新 | `HOT` / `WARM` / `COLD` | 基于 last_accessed 实时计算 |
| **重要度** (importance) | metadata 存储, 可被提升 | `TRIVIAL` / `NORMAL` / `IDENTITY` / `FACT` / `IMPORTANT` / `MUST_REMEMBER` | 写入时规则检测, LLM 审查/手动可提升 |
| **时间段** (time_scope) | 不存储, 查询时计算 | `RECENT` / `NON_RECENT` | 默认 30 分钟窗口, 可配置 |

### 5.2 温度动态更新规则

在每轮 `decay_and_prune()` 中自动计算:

| 条件 | 温度 |
|------|------|
| `last_accessed` 在最近 1 小时内 | `HOT` |
| `last_accessed` 在 1~24 小时之间 | `WARM` |
| `last_accessed` 超过 24 小时 | `COLD` |

节点被 BFS 激活或直接访问时自动设为 `HOT`。

### 5.3 重要度分级与半衰期

| 重要度 | 边权衰减半衰期 | 孤立节点容忍 | 自动识别规则 | 示例 |
|--------|---------------|-------------|-------------|------|
| `TRIVIAL` | 6 小时 | 6 小时 | 语气词/极短回复 < 5 字符 | "嗯""好的""你好" |
| `NORMAL` | 24 小时 | 24 小时 | 默认值 | 一般对话 |
| `IDENTITY` | 30 天 | 永不删除 | "我叫"/"我是"/"我姓"/"今年 XX 岁" | "我叫张三" |
| `FACT` | 15 天 | 不删除 | "我家在"/"我的电话是"/"XX 月 XX 号" | "我住在上海" |
| `IMPORTANT` | 7 天 | 24 小时 | access_count >= 3 自动提升, 或"我喜欢"/"我讨厌" | 偏好/高频访问 |
| `MUST_REMEMBER` | 永不衰减 | 永不删除 | "帮我记住"/"别忘了"/"一定要记得" | 用户显式要求 |

**重要度只升不降**: `promote_importance()` 只允许向上提升。降级需要 LLM 审查专门处理。

### 5.4 标签组合示例

一个节点可以同时是 `importance="identity"` + `temperature="hot"` (如用户刚提到的名字):
- 衰减极慢 (高重要度, 30 天半衰期)
- 同时在热检索路径中被直接遍历到

---

## 6. 地址继承系统

### 6.1 核心概念

**节点地址像文件系统路径一样，子节点地址携带完整的父级路径。** 这使得从任意子节点都能追溯完整上级路径。

### 6.2 地址格式

```
Session 根节点:    dialogue:session_abc123
Round 节点:        dialogue:session_abc123/dialogue:round_xxx
Sub-dialogue:      dialogue:session_abc123/dialogue:round_xxx/turn_agent_xxx
任务根节点:        dialogue:session_abc123/task:tg_xxx
任务子节点:        dialogue:session_abc123/task:tg_xxx/o1
任务孙节点:        dialogue:session_abc123/task:tg_xxx/o1_1
```

### 6.3 地址继承完整流程

```
1. Gatekeeper 收到用户输入
   --> DialogueAdapter.add_round(session_id=None)
   --> 创建 round 节点 (此时 session 未分配, full_path 为空)
   --> 建立 REFERENCE 边到当前活跃 task (如果存在)

2. L2 推理 --> FC 循环 --> task_create_plan (如果需要创建任务)
   --> 查找最近 round, 继承其路径创建任务根节点
   --> 建立双向 REFERENCE 边: round <-> task

3. L2 推理完成 --> _update_memory_async()
   --> assign_session_by_similarity() 话题边界检测
   --> _link_round_to_session() 分配 session
       --> 建立 HIERARCHY 边: session -> round
       --> 更新 round.full_path = "dialogue:session_xxx/dialogue:round_xxx"
   --> _propagate_address_to_tasks() 传播地址到关联任务节点
       --> 更新 task.full_path = "dialogue:session_xxx/task:tg_xxx"
   --> _propagate_address_to_task_children() 递归传播到子任务
       --> 更新 subtask.full_path = "dialogue:session_xxx/task:tg_xxx/o1"

4. 后续用户调用 task_add_node (FC 工具)
   --> 查找父节点 full_path
   --> 子节点地址 = 父路径 + "/" + 子 ID
   --> 通过 TaskGraphAdapter.incremental_sync() 同步到 MemoryGraph
```

### 6.4 地址解析

```python
def resolve_address(self, address: str) -> Optional[GraphNode]:
    """解析图地址字符串"""
    # 支持格式:
    #   "tg:{graph_id}/task:{node_id}" --> TaskGraph 地址
    #   "task:o1_1"                     --> 直接 node_id
    #   "dialogue:round_42"             --> 对话节点
    #   metadata 中的 graph_address      --> 后备匹配
```

---

## 7. 基于图记忆的自主动态注意力机制

### 7.1 核心概念

注意力系统负责**动态分配认知资源**，决定哪些信息应被优先处理，哪些应被暂时忽略。祖龙的注意力机制从传统的 1D 线性注意力（按时间远近评分）升级为**图 BFS 扩散注意力**（沿图结构发现跨类型关联）。

### 7.2 基于图记忆的动态注意力

基于图记忆机制的动态注意力分为两个板块：

#### 板块一：思维深度导航

L2 模型通过 `navigate_attention` FC 工具主动控制注意力焦点，在记忆图的树状结构中自由移动：

| 操作 | 行为 | 代码对应 |
|------|------|---------|
| `deeper` | 深入当前焦点的子节点，进入更细粒度的任务/对话细节 | `attention_tool.py` NavigateAttentionTool |
| `broader` | 回退到父节点，获取更宏观的上下文视角 | `attention_tool.py` NavigateAttentionTool |
| `jump` | 跳转到指定节点 ID，实现跨话题/跨任务的注意力转移 | `attention_tool.py` NavigateAttentionTool |

#### 板块二：基于 BFS 扩散的三类注意力

| 类型 | 机制 | 代码对应 |
|------|------|---------|
| 全局注意力 | BFS 从种子节点沿加权边扩散，每跳按 decay 衰减，覆盖整张图 | `memory_graph.py` compute_activations() |
| 单链注意力 | 沿 HIERARCHY 父子链聚焦，从当前焦点到根的纵向路径 | `attention_tool.py` deeper/broader |
| 局部注意力 | 当前焦点节点的直接邻域（1-hop neighbors） | `memory_graph_tools.py` discover_related |

> **注意：** `AttentionLayer` 枚举（L0_SENSOR ~ L3_COGNITIVE）是**事件路由层级**，用于 L1-B AttentionController 的事件分发和中断管理，不是注意力机制本身。详见 `attention_atoms.py`。

### 7.3 注意力事件 (AttentionEvent)

```python
@dataclass
class AttentionEvent:
    event_id: str
    source: str                   # 事件来源插件标识
    type: EventType               # SILENT_OBSERVATION / INTERACTION_TRIGGER / EMERGENCY_ALERT
    priority: int                 # 1-10, 10 最高
    payload: Dict[str, Any]       # 事件载荷
    timestamp: float

    def is_interrupt_level(self) -> bool:
        """优先级 >= 8 判定为中断级别"""
        return self.priority >= 8
```

### 7.4 注意力控制器 (AttentionController)

位于 L1-B 层, 是注意力系统的中枢。文件: `zulong/l1b/attention_controller.py`

**状态机**:
- `IDLE`: 空闲, 可接收新事件
- `BUSY`: L2 正在处理, 低优事件入队
- `SUSPENDED`: 当前任务被中断冻结

#### 7.4.1 事件处理策略 (tick 方法)

```python
def tick(self, events: List[AttentionEvent]):
    for evt in events:
        if evt.is_interrupt_level():           # 优先级 >= 8
            self._handle_interrupt(evt)        # 强制中断
        elif self.status == "BUSY":
            if evt.priority >= interrupt_threshold:  # >= 8
                self._handle_interrupt(evt)    # 高优打断
            else:
                self._queue_event(evt)         # 低优排队 (PriorityQueue)
        elif self.status == "IDLE":
            if evt.priority >= high_priority_threshold:  # >= 5
                self._route_to_l2_direct(evt)  # 高优直通 L2
            else:
                self._queue_event(evt)         # 批量处理

    # IDLE 状态下检查队列
    # on_l2_idle() -> 恢复挂起任务 或 处理排队事件
```

#### 7.4.2 中断处理核心流程

```
1. Freeze (冻结):
   --> _create_l2_snapshot()
   --> 创建 ContextSnapshot (KV Cache 指针 + 对话历史 + 任务摘要)
   --> status = SUSPENDED

2. Recompose (重组):
   --> _format_emergency_context(evt)
   --> 打包 "[紧急事件] {source} 检测到 {payload}"
        + "[暂停的任务] {snapshot.summary}"
        + "请给出简短的应对指令。"

3. Inject (注入):
   --> _force_l2_respond(recomposed_prompt, priority="IMMEDIATE")
   --> 强制 L2 清空当前生成流, 立即响应新 Prompt

4. 紧急事件处理完成后:
   --> 检查 active_snapshot 是否存在
   --> _load_l2_snapshot(snapshot) 恢复
   --> 继续被中断的任务
```

### 7.5 上下文快照 (ContextSnapshot)

```python
@dataclass
class ContextSnapshot:
    task_id: str                      # 任务 ID
    summary: str                      # 任务一句话摘要
    full_history: List[Dict]          # 完整对话历史 (最近 10 轮)
    kv_cache_ptr: Optional[int]       # 显存中 KV Cache 指针
    generation_state: Optional[Dict]  # 当前生成 token 进度
    pause_reason: str                 # 暂停原因
    created_at: float
```

### 7.6 其他注意力原子类

```python
@dataclass
class MacroCommand:
    """L2 -> L1-A 的高层指令"""
    cmd_id: str
    intent: str        # GRASP / NAVIGATE / FOLLOW
    targets: Dict
    constraints: Dict
    source_snapshot_id: str

@dataclass
class SensorFusionData:
    """多模态传感器融合数据"""
    vision_target_pos: Optional[List[float]]
    radar_obstacles: Optional[List[Dict]]
    audio_source_pos: Optional[List[float]]
    timestamp: float
```

---

## 8. 思维深度导航

### 8.1 核心概念

思维深度导航允许 LLM **主动调整注意力焦点**，在图记忆中沿着层级关系上下移动。这是 LLM 通过 FC 工具自主操作图空间的核心能力之一。

### 8.2 NavigateAttentionTool

文件: `zulong/tools/attention_tool.py`

| 方向 | 含义 | 实现 |
|------|------|------|
| `deeper` | 深入当前子任务 | 获取当前节点的子节点 (get_children), 选择最近访问的子节点 |
| `broader` | 返回上层 | 获取当前节点的父节点 (get_parent) |
| `jump` | 跳转到特定节点 | 直接定位到 target_node_id |

**执行后的副作用**:
1. 调用 `MemoryGraph.update_focus_to_node()` 更新焦点路径
2. 重新计算焦点路径摘要 `get_focus_path_summary()`
3. 影响下次 BFS 扩散激活的种子节点选择

### 8.3 导航层级结构

```
Session (对话根节点)
    | HIERARCHY
    v
Round (对话轮次)
    | HIERARCHY
    v
 +--------+--------+
 v                  v
Sub-dialogue    Task:tg_xxx (任务根)
                    | HIERARCHY
                    v
                Task:o1 (子任务)
                    | HIERARCHY
                    v
                Task:o1_1 (孙任务)
```

### 8.4 焦点路径注入

每次 LLM 推理时, `get_focus_path_summary()` 的输出被注入到 system prompt:

```
【思维导航】
L1 [任务] 项目开发
 +-- L2 [任务] 实现 API
  +-- L3 [对话] 帮我调试异常 <-- 当前焦点
```

### 8.5 焦点追踪的系统层自动行为

| 方法 | 触发方式 | 作用 |
|------|---------|------|
| `update_focus_to_node()` | 对话创建 + 检查点 | 构建思维深度路径 (get_ancestors) |
| `set_last_focus_context()` | 每 20 轮 + 关闭时 | 保存焦点上下文到磁盘 (重启后恢复) |
| `set_active_nodes()` | 每轮 Agent 循环 | 标记活跃节点 (前端高亮 + 检索权重加成) |

---

## 9. 记忆遗忘与剪枝机制

### 9.1 艾宾浩斯衰减公式

```
decayed_weight = weight x exp(-elapsed_hours x ln(2) / half_life)
```

其中 `half_life` 由节点 importance 决定:

| importance | half_life | 含义 |
|-----------|-----------|------|
| TRIVIAL | 6 小时 | 6 小时后权重降至 50% |
| NORMAL | 24 小时 | 1 天后权重降至 50% |
| IMPORTANT | 168 小时 (7 天) | 7 天后权重降至 50% |
| FACT | 360 小时 (15 天) | 15 天后权重降至 50% |
| IDENTITY | 720 小时 (30 天) | 30 天后权重降至 50% |
| MUST_REMEMBER | 无穷大 | 永不衰减, 边自动设为 protected |

**边的半衰期取两端节点中更高重要度的值。**

### 9.2 自动重要度检测规则

`DialogueAdapter._detect_importance()` 使用正则匹配:

```python
_IMPORTANCE_RULES = [
    # MUST_REMEMBER (用户显式要求)
    (r'(帮我记住|别忘了|一定要记得|记住了|千万别忘)', MUST_REMEMBER),
    # IDENTITY (身份信息)
    (r'(我叫|我是|我姓|我今年.*岁|我.*年生)', IDENTITY),
    # FACT (客观事实)
    (r'(我家在|我住在|我的电话|我的手机|.*月.*号|.*月.*日)', FACT),
    # IMPORTANT (偏好)
    (r'(我喜欢|我不喜欢|我爱|我讨厌|我答应|我保证)', IMPORTANT),
    # TRIVIAL (无意义)
    (r'^(嗯|好|好的|哦|ok|行|是的|对|谢谢|你好|再见)$', TRIVIAL),
]
```

### 9.3 剪枝完整流程

```
decay_and_prune() -- 后台定时器每 30 分钟触发
  |
  |-- 1. 遍历所有边, 计算衰减后权重
  |     |-- 获取两端节点的 importance (取更高者)
  |     |-- 计算 elapsed_hours = (now - last_activated) / 3600
  |     |-- 应用衰减公式
  |     |-- protected 边跳过
  |
  |-- 2. 修剪低权重边
  |     |-- weight < 0.05 --> 直接删除
  |     |-- weight 0.05~0.15 --> 标记 pending_review (送 LLM 审查)
  |     |-- weight >= 0.15 --> 更新权重
  |
  |-- 3. LLM 审查 (异步, 不阻塞)
  |     |-- submit_prune_review() 提交濒危边到 LLM
  |     |-- LLM 返回: KEEP/DISCARD/COMPRESS/PROMOTE/MERGE
  |     |-- process_review_result() 处理结果:
  |           KEEP --> 强化边权 + 提升 importance
  |           DISCARD --> 下一轮剪枝删除
  |           COMPRESS --> 压缩 metadata 为摘要, 重置边权
  |           PROMOTE --> 提升节点 importance
  |           MERGE --> 合并到关联节点
  |
  |-- 4. 同步更新温度标签
  |     |-- last_accessed < 1h --> HOT
  |     |-- last_accessed 1h~24h --> WARM
  |     |-- last_accessed > 24h --> COLD
  |
  |-- 5. 检查孤立节点
  |     |-- 无任何边连接的节点
  |     |-- importance >= IDENTITY --> 保留 ("沉睡记忆", 等待未来激活)
  |     |-- TRIVIAL 孤立 > 6h --> 删除
  |     |-- NORMAL 孤立 > 24h --> 删除
  |
  |-- 6. 语义孤立检测 (空壳清理)
  |     |-- 仅剩 TEMPORAL/HIERARCHY 边 (纯骨架)
  |     |-- importance="trivial"/"normal" + temperature="cold" --> 标记可丢弃
  |
  |-- 7. 自动重要度提升
  |     |-- run_importance_review()
  |     |-- access_count >= 3 且 importance="normal" --> 提升为 "important"
  |     |-- access_count >= 5 且 importance="important" --> 候选 LLM 审查是否提升为 "must_remember"
  |
  |-- 8. 记录统计 + 持久化
```

### 9.4 LLM 剪枝守卫

文件: `zulong/memory/llm_memory_reviewer.py`

三种审查模式:
- **PRE_STORE**: 写入前审查 (阻止低质量数据入库)
- **PRE_EVICT**: 剪枝前审查 (保护重要记忆)
- **PERIODIC_REVIEW**: 定期巡检 (批量提升/清理)

审查通过 L2-BACKUP 异步执行, 不阻塞主对话流。

---

## 10. 记忆检索系统

### 10.1 检索入口

```python
async def retrieve_context(
    self,
    query_text: str,
    top_k: int = 10,
    hot_window_minutes: int = 30,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
```

### 10.2 双路径并行检索策略

**核心原则**: 图是一个整体, 不做物理拆分。两条检索路径通过标签**互斥过滤**在同一张图上各自排除对方的数据范围, 保证无重复。

```
用户输入: "处理器价格"
         |
    asyncio.gather() 并行执行
         |
   +-----+-----+
   |             |
   v             v
路径 A:        路径 B:
热数据遍历     冷数据 FAISS
   |             |
   |  筛选 is_recent()  |  FAISS 向量搜索摘要索引
   |  == True 的节点     |  排除热数据 (互斥过滤)
   |             |
   |  关键词匹配:       |  命中 Session/EPISODE 节点
   |  - label 完全匹配  |  作为 BFS 种子
   |    --> 0.8 分       |  下钻获取详情
   |  - content 包含     |
   |    --> 0.5 分       |  resolve_backend_ref()
   |  - bigram 重叠率    |  获取 KG/RAG 原始内容
   |    >= 20%           |
   |    --> 0.3 * 率     |
   |             |
   |  BFS 扩散激活       |
   |  从匹配节点出发     |
   |             |
   |  TEMPORAL 边追溯    |
   |  最近 N 轮对话      |
   |             |
   +-----+------+
         |
         v
合并 (天然无重复, 因为互斥):
  - 热数据 boost 1.5x
  - 高 importance 额外加权
  - session_id 匹配 +0.3
  - 按 score 降序 --> Top-K
```

### 10.3 摘要侧车索引 (SummarySidecarIndex)

独立维护的 FAISS 向量索引层:

```python
class SummarySidecarIndex:
    """FAISS 索引中只存摘要向量 + node_id 指针, 不存储节点实际内容"""
    
    def index_summary(self, node_id: str, text: str) -> bool:
        """为节点摘要生成 embedding 并加入 FAISS 索引"""
        # 使用 BAAI/bge-small-zh-v1.5, 512 维
        
    def search_by_text(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """向量搜索: text -> embedding -> FAISS Top-K -> (node_id, score)"""
```

**索引维护时机**:
- `DialogueAdapter.finalize_round()` 完成时, 为 Session 节点更新摘要向量
- `EpisodeAdapter.sync()` 时, 为 EPISODE 节点生成摘要向量
- 节点删除时, 同步移除 FAISS 中的向量
- 随 MemoryGraph 一同保存/加载 (`save()`/`_load()`)

### 10.4 检索结果结构

```python
{
    "node_id": "dialogue:session_abc/dialogue:round_xxx",
    "node_type": "dialogue",
    "label": "帮我做一个 3D 赛车游戏...",
    "score": 0.85,
    "source": "keyword_match",  # keyword_match / bfs_activation / temporal_trace / faiss
    "metadata": {...},
}
```

### 10.5 上下文注入格式

检索结果按节点类型格式化后注入 LLM system prompt:

| 节点类型 | 注入格式 |
|---------|---------|
| DIALOGUE | `【历史对话】用户曾问: ... 你回答: ...` |
| TASK | `【相关任务】任务名: ... 状态: ...` |
| KNOWLEDGE | `【知识参考】...` (通过 backend_ref 获取 KG 完整内容) |
| EPISODE | `【历史摘要】...` |
| EXPERIENCE | 不自动注入 (由 FC 按需获取) |

---

## 11. 记忆巩固与自演进

### 11.1 记忆强度模型

文件: `zulong/memory/memory_evolution.py`

```python
@dataclass
class MemoryStrength:
    initial_strength: float      # 初始强度
    current_strength: float      # 当前强度
    decay_rate: float            # 衰减率 (艾宾浩斯曲线)
    last_access_time: float
    access_count: int
    emotional_weight: float      # 情感权重 (正面情感增强记忆)
    level: str                   # L1/L2/L3
    importance_level: str
    
    def decay(self):
        """应用艾宾浩斯衰减: R = e^(-t/S)"""
        
    def reinforce(self):
        """访问时强化: 重置衰减计时器, 增加 access_count"""
        
    def should_forget(self) -> bool:
        """strength < 0.1 时应遗忘"""
        
    def should_promote(self) -> bool:
        """strength > 0.7 且 access_count >= min_count 时应巩固"""
```

### 11.2 记忆巩固器

```python
class MemoryConsolidator:
    """短期记忆 -> 长期记忆转化"""
    # 巩固条件:
    #   - memorability == "must_remember"
    #   - access_count >= 2
    #   - importance == "must_learn"
    #   - emotional_weight > 1.5
    # 巩固间隔: 1 小时
```

### 11.3 记忆遗忘器

```python
class MemoryForgetter:
    """自动清理低价值记忆"""
    # 遗忘条件: strength < 0.1
    # 检查间隔: 6 小时
    # 保护: importance >= IDENTITY 永不遗忘
```

### 11.4 与 MemoryGraph 的集成

MemoryStrength 的衰减追踪结构存储在 `GraphNode.metadata["memory_strength"]` 中。`decay_and_prune()` 在衰减过程中同步更新这些值, 实现图级衰减和节点级衰减的统一。

---

## 12. 经验数据与事件记忆区分

### 12.1 五维区分机制

| 维度 | 经验数据 (ExperienceRAG) | 事件记忆 (MemoryRAG/DIALOGUE 节点) |
|------|------------------------|----------------------------------|
| **存储位置** | ExperienceRAG + EXPERIENCE 节点 | MemoryGraph DIALOGUE 节点 |
| **数据结构** | Experience (带 tags/keywords/importance_score) | GraphNode (带 importance/temperature) |
| **生成路径** | 复盘提取 / L2 自动生成 / 用户确认 | 对话自动固化 |
| **评估标准** | 置信度 >= 0.6, 可复用性, 去重 | 访问次数, 情感权重, 时间衰减 |
| **检索方式** | 混合检索 (向量 + BM25 + 标签), FC 按需调用 | 图 BFS + FAISS, 自动注入 |

### 12.2 经验数据生成流程

```
复盘路径:
用户对话 --> ReviewTempBuffer (隔离缓冲区, 不污染主记忆池)
         --> ExperienceExtractor (L2 分析提取)
         --> 用户确认
         --> ExperienceRAG (入库)

自动生成路径:
对话历史 --> ExperienceGenerator (模式匹配)
         --> ExperienceRAG (入库)
```

### 12.3 隔离设计

复盘对话本身**不存入** MemoryRAG/MemoryGraph, 仅保存在临时缓冲区。只有经过 L2 分析和用户确认的经验才存入 ExperienceRAG。缓冲区在复盘结束后自动清空。

---

## 13. 任务恢复机制

### 13.1 概述

任务恢复机制支持**任务挂起后从断点继续执行**, 包括环境变化检测、快照恢复、重评估等。涉及三个核心组件的协同。

### 13.2 TaskSuspensionManager

文件: `zulong/l2/task_suspension.py`

```python
class SuspendableTaskState:
    """可挂起的任务状态 (可序列化)"""
    task_id: str
    description: str
    messages: List[Dict]          # 对话历史
    accumulated_links: Dict       # 累积的外部链接
    circuit_breaker_state: str    # 熔断器状态
    iteration_count: int
    task_graph_serialized: str    # TaskGraph 序列化 JSON
    created_at: float
    suspended_at: float
    suspended_reason: str
    metadata: Dict

class TaskSuspensionManager:
    """任务挂起/恢复管理器 (单例)"""
    
    async def suspend_task(self, state: SuspendableTaskState) -> str:
        """挂起: 序列化 -> JSON -> 磁盘 (./data/suspended_tasks/)"""
        # 强制 max_suspended_tasks 限制, 超出时删除最旧的
        
    async def resume_task(self, task_id: str) -> Optional[SuspendableTaskState]:
        """恢复: 读取 JSON -> 反序列化 TaskGraph -> 删除文件"""
        
    async def find_by_description(self, query: str) -> Optional[Dict]:
        """模糊匹配: 字符级相似度 (对中文友好)"""
```

### 13.3 EnvironmentSnapshot

文件: `zulong/l2/environment_snapshot.py`

```python
@dataclass
class ObjectState:
    """物体状态"""
    position: List[float]
    status: str    # on_ground / held / moved / disappeared

@dataclass
class EnvironmentSnapshot:
    """环境快照"""
    objects: List[ObjectState]
    user_state: UserState
    task_conditions: List[TaskCondition]
    scene_description: str
    audio_context: str

@dataclass
class EnvironmentChange:
    """变化检测结果"""
    has_changes: bool
    changes: List[str]
    severity: str        # none / minor / major / critical
    recommendation: str  # CONTINUE / REPLAN / ABORT
```

### 13.4 RecoveryNotifier

文件: `zulong/l2/recovery_notifier.py`

系统启动时自动执行:
1. 扫描 `./data/checkpoints/` 中的崩溃检查点
2. 将检查点提升为挂起任务 (`_promote_checkpoints()`)
3. 列出所有可恢复的挂起任务
4. 通过 EventBus 发送 L2_OUTPUT 事件通知用户

### 13.5 恢复场景识别

```python
# 用户输入包含以下关键词时, 尝试匹配挂起的任务:
RESUME_KEYWORDS = ['继续', '接着', '上次', '恢复', '回到', '之前', '的', '任务', '那个']

# 匹配逻辑:
# 1. 提取用户输入中的任务描述关键词
# 2. find_by_description() 字符级模糊匹配
# 3. 找到最匹配的任务, 调用 resume_task()
# 4. 环境快照对比: compare_snapshots() --> EnvironmentChange
# 5. 根据 recommendation 决定: CONTINUE / REPLAN / ABORT
```

### 13.6 完整恢复流程（基于图记忆的 L2 自主决策）

系统层**不做恢复意图检测**，所有判断由 L2 模型通过 FC 工具自主完成：

```
用户输入（任意内容，可能涉及恢复，也可能不涉及）
  |
  v
L1-B Gatekeeper（物理层打包，不判断意图）
  |-- 语音模式检测
  |-- 上下文快照收集
  |-- 注意力层级路由
  |-- 打包发送给 L2（无 is_resume 字段）
  |
  v
L2 InferenceEngine（模型自主 FC 循环）
  |-- 模型理解用户意图后，自主决定调用哪些工具：
  |
  |-- [可选] task_list_suspended(query="...")
  |     |-- 列出/搜索挂起任务
  |
  |-- [可选] recall_memory("相关描述")
  |     |-- 从记忆图中搜索历史任务记忆
  |
  |-- [可选] discover_related(node_id="...")
  |     |-- 发现关联的任务/对话节点
  |
  |-- [可选] navigate_attention(direction="jump", target="...")
  |     |-- 跳转到旧任务的记忆节点
  |
  |-- [决策] 模型判断是否需要恢复旧任务
  |     |-- 是 → resume_task(task_id="matched_id")
  |     |       |-- 从磁盘加载 JSON + 反序列化 TaskGraph
  |     |       |-- Session 事后绑定 (bind_session_to_task)
  |     |-- 否 → 作为新任务处理
  |
  v
EnvironmentSnapshot.compare_snapshots(old, current)
  |-- 检测物体状态变化
  |-- 检测用户位置变化
  |-- 检测任务条件满足度
  |
  v
根据 recommendation:
  CONTINUE --> 直接恢复, L2 根据旧上下文继续执行
  REPLAN   --> 重新规划部分子任务
  ABORT    --> 通知用户任务条件已不满足
```

---

## 14. MG 能力暴露: 系统层 vs 工具层

### 14.1 系统层自动运行 (约 25 个方法, 模型无感知但直接受益)

#### A. 图维护循环 (后台定时器, 30 分钟一轮)

| 方法 | 作用 | 对模型的影响 |
|------|------|-------------|
| `decay_and_prune()` | 艾宾浩斯衰减 + 边修剪 | 弱关联自动清理, 下次检索更精准 |
| `run_importance_review()` | access_count >= 3 自动提升 | 高频信息自动获得更高权重 |
| `update_temperature()` | 更新节点温度 (HOT/WARM/COLD) | 影响冷热检索路径选择 |
| `save()` | 脏标记防抖持久化 | 重启后状态不丢失 |

#### B. 适配器同步 (每次对话自动触发)

| 方法 | 作用 | 对模型的影响 |
|------|------|-------------|
| `TaskGraphAdapter.sync()` | 投影 TaskGraph -> MG 节点 | 模型通过检索能看到任务结构 |
| `DialogueAdapter.add_round()` | 创建对话轮次节点 | 模型获得对话历史链 |
| `DialogueAdapter._detect_importance()` | 模式匹配标注重要性 | "我叫 X"->IDENTITY, "帮我记住"->MUST_REMEMBER |
| `DialogueAdapter.finalize_round()` | 归档 + FAISS 索引 | 冷数据可通过向量搜索找到 |

#### C. 上下文检索 (prompt 构建时自动调用)

| 方法 | 作用 | 对模型的影响 |
|------|------|-------------|
| `retrieve_context()` | 双路径热/冷混合检索 | 相关记忆自动注入 system prompt |
| `get_focus_path_summary()` | 渲染思维导航路径 | "【思维导航】L1...L2..." 直接注入 |

#### D. 节点创建时的自动行为

| 方法 | 作用 | 对模型的影响 |
|------|------|-------------|
| `_auto_embed_node()` | 自动生成 embedding | 后续语义搜索可用 |
| `discover_semantic_neighbors()` | 余弦相似度发现近邻 | 自动创建 SEMANTIC 边 |
| `_update_coactivation_counter()` | 追踪共激活对 | 达阈值自动创建 ASSOCIATION 边 |

### 14.2 工具层 (需要模型通过 FC 按需调用)

**当前已暴露的 FC 工具**:

| FC 工具 | 包装的 MG 方法 | 用途 |
|---------|---------------|------|
| `recall_memory` | `search_nodes()` | 关键词搜索记忆 (子串匹配, max=8) |
| `recall_node_context` | `get_subgraph_summary()` | 获取节点 + 邻居摘要 |
| `navigate_attention` | `update_focus_to_node()` | 思维焦点导航 (deeper/broader/jump) |
| `search_experience` | `ExperienceRAG.search()` | 经验库检索 (被动化后) |

**已识别但尚未封装的 MG 能力**:

| MG 方法 | 建议 FC 工具 | 用途 |
|---------|-------------|------|
| `retrieve_context()` | `recall_context_hybrid` | 混合检索 (关键词 + 向量 + 激活度) |
| `search_summaries()` | `recall_memory` mode="vector" | FAISS 向量搜索 |
| `resolve_address()` + 层级导航 | `navigate_graph` | 地址解析 + 层级上下移动 |
| `promote_importance()` | `set_memory_importance` | 手动标注重要性 |
| `discover_semantic_neighbors()` | `discover_neighbors` | 语义近邻发现 |
| `get_focus_path_summary()` | `view_focus_path` | 查看当前思维深度 |
| `compute_activations()` | `activate_and_retrieve` | 手动触发 BFS 扩散 |

**当前暴露率**: 约 4/53 = 7.5% (含 navigate_attention)

### 14.3 纯基础设施 (约 24 个方法, 不需要暴露)

`has_node()`, `get_edge()`, `remove_node()`, `remove_edge()`, `get_nodes_by_type()`, `get_temperature()`, `is_recent()`, `get_embedding()`, `set_embedding()`, `register_adapter()`, `sync_all()`, `start_prune_loop()`, `stop_prune_loop()`, `submit_prune_review()`, `to_frontend_dict()`, `flush_changes()`, `_load()`, `_mark_dirty()`, `_do_auto_save()`, `_serialize_embeddings()`, `_deserialize_embeddings()` 等。

---

## 15. 各系统协同运行机制

### 15.1 完整对话流程 (从用户输入到记忆写入)

```
+------------------------------------------------------------------+
| 用户输入: "帮我做一个 3D 赛车游戏"                                   |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| L1-B Gatekeeper（物理层打包，不做语义意图识别）                      |
| - 注意力层级判断: L2_INTERACTIVE                                   |
| - 语音模式检测                                                      |
| - 上下文快照收集                                                    |
| - DialogueAdapter.add_round(session_id=None) 创建 round 节点       |
|   --> _detect_importance() 标注为 NORMAL                           |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| InferenceEngine._process_with_memory()                             |
| - MemoryGraph.retrieve_context("帮我做一个3D赛车游戏")              |
|   --> 并行: 热数据遍历 + 冷数据 FAISS                              |
|   --> 合并 Top-K 相关记忆                                          |
| - get_focus_path_summary() 获取思维焦点路径                         |
| - _retrieve_from_rag() 获取 KnowledgeRAG 知识                      |
| - _build_messages_with_history() 构建 LLM 上下文:                  |
|   [system: 角色定义 + 任务规则 + 思维焦点 + 相关记忆 + 知识]        |
|   [user: "帮我做一个3D赛车游戏"]                                    |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| LLM 推理 (vLLM) + FC 循环                                         |
| - LLM 识别到复杂任务, 调用 task_create_plan                        |
|   --> 创建 TaskGraph 根节点                                        |
|   --> TaskGraphAdapter.incremental_sync() 同步到 MG                |
|   --> 建立 round <--> task 双向 REFERENCE 边                       |
| - 调用 task_add_node x N 添加子任务                                |
|   --> 子节点地址 = 父路径/子 ID                                     |
| - 调用 submit_final_answer 返回结果                                |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| _update_memory() (异步)                                            |
| - assign_session_by_similarity() embedding 话题匹配               |
|   --> 新话题: _create_session() + TEMPORAL 边                      |
|   --> 旧话题: 复用已有 session                                     |
| - _link_round_to_session()                                        |
|   --> HIERARCHY 边: session -> round                               |
|   --> 更新 round.full_path                                        |
|   --> _propagate_address_to_tasks() 传播地址到任务节点              |
|   --> _propagate_address_to_task_children() 递归传播               |
| - finalize_round()                                                |
|   --> round_summary 写入 FAISS 摘要索引                            |
|   --> _link_to_knowledge() 自动链接到 KG 节点                      |
| - _auto_embed_node() 生成 embedding                               |
| - discover_semantic_neighbors() 发现语义近邻                        |
+------------------------------------------------------------------+
```

### 15.2 注意力中断完整流程

```
+------------------------------------------------------------------+
| 紧急事件触发 (优先级 >= 8)                                          |
| 例如: 用户说 "停下" / 视觉检测到危险物体                             |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| AttentionController._handle_interrupt(evt)                         |
|                                                                    |
| 1. Freeze (冻结)                                                   |
|    - _create_l2_snapshot()                                         |
|    - 保存 ContextSnapshot:                                         |
|      task_id + summary + full_history + kv_cache_ptr               |
|    - status = SUSPENDED                                            |
|                                                                    |
| 2. Recompose (重组)                                                |
|    - "[紧急事件] vision 检测到 {payload}"                           |
|    - "[暂停的任务] {snapshot.summary}"                              |
|    - "请给出简短的应对指令。"                                       |
|                                                                    |
| 3. Inject (注入)                                                   |
|    - _force_l2_respond(prompt, priority="IMMEDIATE")               |
|    - 强制 L2 清空当前生成流, 立即响应                               |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| L2 处理紧急事件 --> 输出应对指令                                     |
+----------------------------+-------------------------------------+
                             |
                             v
+------------------------------------------------------------------+
| on_l2_idle() (紧急事件处理完毕)                                     |
| - 检查 active_snapshot 是否存在                                     |
| - _load_l2_snapshot(snapshot) 恢复                                  |
|   --> 恢复对话历史 + KV Cache                                       |
| - 继续被中断的任务                                                  |
| - status = BUSY                                                    |
+------------------------------------------------------------------+
```

### 15.3 跨系统依赖关系全景

```
MemoryGraph (核心)
    |-- 被 InferenceEngine 调用 (retrieve_context, get_focus_path_summary)
    |-- 被 AttentionController 调用 (update_focus_to_node, set_active_nodes)
    |-- 被 TaskSuspensionManager 调用 (任务节点状态查询)
    |-- 被 6 个 Graph Adapters 调用 (add_node, add_edge, 同步)
    |-- 被 NavigateAttentionTool 调用 (焦点导航)
    |-- 被 recall_memory FC 工具调用 (search_nodes)
    +-- 被 recall_node_context FC 工具调用 (get_subgraph_summary)

Address Inheritance (地址继承)
    |-- 依赖 MemoryGraph (节点/边操作)
    |-- 被 TaskCreatePlanTool 调用 (创建任务时继承 round 路径)
    |-- 被 TaskAddNodeTool 调用 (添加子任务时继承父路径)
    |-- 被 DialogueAdapter 调用 (session 分配时传播)
    +-- 被 TaskGraphAdapter 调用 (sync 时继承路径)

Attention Mechanism (注意力)
    |-- 依赖 MemoryGraph (焦点路径管理, get_ancestors/get_children)
    |-- 依赖 EventBus (事件路由)
    |-- 被 AttentionController 调用 (tick -> 事件路由/中断处理)
    |-- 被 NavigateAttentionTool 调用 (FC 焦点调整)
    +-- 被 InferenceEngine 调用 (焦点路径注入 system prompt)

Pruning Mechanism (遗忘剪枝)
    |-- 依赖 MemoryGraph (边/节点操作)
    |-- 依赖 Importance 分级 (半衰期选择)
    |-- 依赖 LLMMemoryReviewer (濒危边审查)
    +-- 定时触发 (后台 asyncio task, 每 30 分钟)

Retrieval System (检索)
    |-- 依赖 MemoryGraph (BFS 扩散激活, 关键词匹配)
    |-- 依赖 SummarySidecarIndex (FAISS 向量检索)
    |-- 依赖 EmbeddingManager (文本 -> 向量)
    +-- 被 InferenceEngine 调用 (构建 LLM 上下文)

Task Recovery (任务恢复)
    |-- 依赖 TaskSuspensionManager (序列化/反序列化)
    |-- 依赖 EnvironmentSnapshot (环境变化检测)
    |-- 依赖 RecoveryNotifier (启动时扫描)
    +-- 被 InferenceEngine 调用 (恢复关键词检测 -> find_by_description)

RAG System (知识/经验)
    |-- ExperienceRAG: 被动化, FC 按需调用 (search_experience)
    |-- KnowledgeRAG: 通过 KNOWLEDGE 节点 backend_ref 打通 MemoryGraph
    |-- ToolRAG: 独立, 不参与记忆系统
    +-- RAGManager: 统一管理 4 个 RAG 库, 注册到 MemoryGraph (set_rag_manager)
```

---

## 16. 关键文件索引

### 16.1 记忆系统核心 (zulong/memory/)

| 文件 | 大小 | 职责 | 核心类/方法 |
|------|------|------|-------------|
| `memory_graph.py` | 102 KB | 异构图核心 | `MemoryGraph`, `retrieve_context()`, `compute_activations()`, `decay_and_prune()`, `hebbian_strengthen()`, `search_nodes()`, `resolve_address()`, `resolve_backend_ref()`, `promote_importance()`, `SummarySidecarIndex` |
| `graph_adapters.py` | 49 KB | 6 个图适配器 | `DialogueAdapter`, `TaskGraphAdapter`, `KnowledgeGraphAdapter`, `EpisodeAdapter`, `PersonProfileAdapter`, `ExperienceAdapter`, `_detect_importance()`, `_propagate_address_to_tasks()` |
| `short_term_memory.py` | 52 KB | 对话缓存 | `ShortTermMemory`, `store()`, `get_recent()`, `search_similar()` |
| `episodic_memory.py` | 29 KB | 摘要/分级读取 | `EpisodicMemory`, `store_episode()`, `search_by_summary()`, `get_full_dialogue()` |
| `rag_manager.py` | 11 KB | 统一 RAG 管理 | `RAGManager`, `add_document()`, `search()` |
| `memory_evolution.py` | 20 KB | 记忆巩固/遗忘 | `MemoryStrength`, `MemoryConsolidator`, `MemoryForgetter`, `MemoryEvolutionEngine` |
| `llm_memory_reviewer.py` | 19 KB | LLM 剪枝守卫 | `LLMMemoryReviewer`, `review_before_evict()`, PRE_STORE/PRE_EVICT/PERIODIC_REVIEW |
| `embedding_manager.py` | 11 KB | 向量生成管理 | `EmbeddingManager`, BAAI/bge-small-zh-v1.5 (512 维) |
| `knowledge_graph.py` | 26 KB | 知识图谱 | 实体/关系管理 |
| `person_profile.py` | 24 KB | 人物档案 | 身份信息持久化 |
| `rag_libraries.py` | 27 KB | 3 个 RAG 库实现 | `ExperienceRAG`, `MemoryRAG`, `KnowledgeRAG` |
| `experience_generator.py` | 14 KB | 经验自动生成 | 模式匹配提取经验 |
| `summary_store.py` | 27 KB | 双索引摘要存储 | FAISS + 节点指针 |

### 16.2 注意力系统

| 文件 | 大小 | 职责 | 核心类/方法 |
|------|------|------|-------------|
| `zulong/core/attention_atoms.py` | 6 KB | 原子类定义 | `AttentionEvent`, `ContextSnapshot`, `MacroCommand`, `SensorFusionData` |
| `zulong/tools/attention_tool.py` | 8 KB | FC 焦点导航 | `NavigateAttentionTool` (deeper/broader/jump) |
| `zulong/l1b/attention_controller.py` | 14 KB | 注意力控制器 | `AttentionController`, `tick()`, `_handle_interrupt()`, `on_l2_idle()` |

### 16.3 任务系统

| 文件 | 大小 | 职责 | 核心类/方法 |
|------|------|------|-------------|
| `zulong/l2/task_graph.py` | - | 任务图谱 | `TaskGraph`, `add_node()`, `add_h_edge()` |
| `zulong/l2/task_suspension.py` | 13 KB | 挂起/恢复 | `TaskSuspensionManager`, `SuspendableTaskState` |
| `zulong/l2/environment_snapshot.py` | 11 KB | 环境快照 | `EnvironmentSnapshot`, `EnvironmentChange` |
| `zulong/l2/recovery_notifier.py` | 6 KB | 启动恢复通知 | `RecoveryNotifier`, `check_and_notify()` |
| `zulong/tools/task_tools.py` | - | 任务 FC 工具 | `TaskCreatePlanTool`, `TaskAddNodeTool` |

### 16.4 推理引擎

| 文件 | 大小 | 职责 | 核心类/方法 |
|------|------|------|-------------|
| `zulong/l2/inference_engine.py` | 99 KB | L2 推理核心 | `InferenceEngine`, `_process_with_memory()`, `_build_messages_with_history()`, `_update_memory()`, `_on_l2_command()` |

### 16.5 模型预加载器 (v2.8 新增)

| 文件 | 大小 | 职责 | 核心类/方法 |
|------|------|------|-------------|
| `zulong/utils/model_preloader.py` | 8 KB | 系统启动时后台加载模型 | `ModelPreloader`, `_detect_backend()`, `_preload_ollama()`, `_preload_openai_compatible()`, `preload_model_from_config()` |

### 16.6 相关文档 (docs/)

| 文档 | 说明 |
|------|------|
| `docs/记忆系统架构文档.md` | 本文档 |
| `docs/memory_graph/` | 图式记忆架构完全指南 (6 篇分册 + 完整版) |
| `docs/记忆架构改造任务文档.md` | MG 改造为读写统一的 12 个任务清单 |
| `docs/图记忆机制深度审查与FC融合分析.md` | MG 能力二分法 + FC 工具规划 |
| `docs/三级记忆检索架构技术实现文档.md` | 第二代增强型三级记忆架构 |
| `docs/经验数据与事件记忆区分机制.md` | 经验 vs 事件记忆的五维区分 |

---

## 17. 关键设计决策

### 17.1 为什么使用异构图而非关系型数据库?

1. **跨类型关联发现**: BFS 扩散激活可以从对话节点扩散到任务节点, 发现隐性关联
2. **灵活的边类型**: 7 种边类型支持复杂关系建模
3. **赫布学习**: 共激活节点对自动建立 ASSOCIATION 边, 模拟神经网络可塑性
4. **地址继承**: 子节点地址携带完整父级路径, 支持从任意节点追溯完整路径

### 17.2 为什么一张图 + 多维标签而非物理分区?

图是一个整体, 所有节点在同一个 NetworkX DiGraph 中。标签只是节点属性标记, 不改变图的拓扑结构, BFS 扩散可以跨任何标签传播。两条并行检索路径通过标签互斥过滤实现, 天然无重复。

### 17.3 为什么衰减速度由重要度标签决定?

不同 importance 拥有不同半衰期 (6h ~ 无穷大), 自然实现"重要信息保持更久"。无需显式的"长期记忆区"概念。`MUST_REMEMBER` 节点的关联边自动设为 protected, 永不衰减。

### 17.4 为什么 LLM 审查是异步守卫?

所有 LLM 审查通过 L2-BACKUP 异步执行。如果审查结果未返回, 濒危节点暂不剪枝 (pending_review 标记), 等审查完成后再决定。不阻塞主对话流。

### 17.5 为什么地址像文件系统路径?

1. **完整追溯**: 从任意子节点通过地址追溯完整上级路径
2. **BFS 友好**: 地址隐含层级关系, 扩散激活可沿路径传播
3. **前端可视化**: 按路径分组展示, 清晰呈现层级结构

### 17.6 为什么 ExperienceRAG 要被动化?

经验库从自动注入改为 FC 按需调用, 避免:
1. 无关经验污染 LLM 上下文
2. context window 浪费
3. 模型可以自主判断何时需要参考经验

---

**文档结束**

> 本文档基于源码 (`memory_graph.py` v2026-04-21, `graph_adapters.py` v2026-04-22, `attention_controller.py`, `task_suspension.py`, `inference_engine.py` 等) 和全部相关设计文档深度梳理而成。所有描述的机制均有对应实现代码。

---

## 18. 核心设计理念：3D 记忆城市

### 18.1 设计哲学

祖龙的记忆系统不是传统的"对话历史列表"或"向量数据库"，而是一个**三维立体的记忆城市**。记忆图谱和任务图谱共同构成了这座城市的空间结构。

**核心原则**:

1. **模型自由游走** - 模型像一个人在城市中行走，可以自由决定去哪条街、进哪栋楼、看哪个房间。不是被铁轨绑死在固定路线上。

2. **按需读取** - 模型根据当前任务需要，自己判断需要读取哪些记忆/文件，自己决定把注意力调整到什么范围。

3. **弹性注意力** - 有时候需要"全景视角"看整体，有时候需要"显微镜"聚焦到某个细节。模型自己判断什么时候需要全部注意力集中到某个局部。

4. **参考模板而非流程捆绑** - 复杂任务有参考模板（类似操作手册），但模型不是机械地按流程办事。模板是建议，不是枷锁。模型可以跳步、回退、插入新步骤。这与传统 Agent 的任务编排有本质区别——传统做法把模型栓死在任务流程上，只能机械地按流程办事。

5. **跨会话持久** - 即使系统重启、任务中断，记忆城市依然存在。模型可以随时回到之前离开的位置继续工作。

### 18.2 3D 空间比喻

```
                    ┌─────── 注意力控制层（天空视角）──────┐
                    │  全局注意力 → 单层注意力 → 焦点注意力 │
                    └──────────────┬───────────────────────┘
                                   │
          ┌──────── 宏空间（城市的街区）────────────────────┐
          │                                                │
  L1层    │  ■ 总会话节点（城市全景）                        │
          │    ├─ 会话1（街区A）                            │
  L2层    │    │   ├─ 子会话1.1（A区1号楼）                 │
  L3层    │    │   │   ├─ 对话轮次（楼层/房间）              │
  L4层    │    │   │   └─ ...                              │
          │    │   └─ 子会话1.2（A区2号楼）                 │
          │    ├─ 会话2（街区B）                            │
          │    └─ ...                                      │
          ├────────── 虚线分隔 ────────────────────────────┤
          │           任务空间（城市的工厂区）               │
  L5层    │  ■ 任务根节点（工厂大门）                        │
  L6层    │    ├─ 子任务1.1（车间1）                        │
  L7层    │    │   ├─ 子子任务1.1.1（工位1）                │
          │    │   └─ 子子任务1.1.2（工位2）                │
          │    └─ 子任务1.2（车间2）                        │
          └────────────────────────────────────────────────┘
                                   │
          ┌──────── 记忆RAG（地下档案室）──────────────────┐
          │  长/中期标签记忆索引或者摘要                     │
          │  （需要深度搜索才能找到的旧记忆）                │
          └────────────────────────────────────────────────┘
```

### 18.3 注意力切换规则（对应架构图顶部的"注意力控制"区域）

| 场景 | 注意力行为 | 比喻 |
|------|-----------|------|
| 进入一层节点 | 默认获取该层节点及全局注意力（扩展覆盖） | 站在街区入口，能看到整个街区和远处的城市天际线 |
| 情报节点以上下层概括 | 获取层节点摘要，不展开细节 | 看楼的门牌号和招牌，不进去 |
| 需要更详细联想检索 | 进入单层信息节点，下钻获取子节点内容 | 走进楼里，看每层的具体内容 |
| 进入子节点 | 全局注意力切换到节点以内，聚焦相关集合，丢弃无关长上下文 | 进入房间后，关上门，集中注意力在房间内的东西 |
| 上下文窗口达到预警 | 模型把当前交互关联的元素加标签再驱动注意力，丢弃碎片关联 | 桌子满了，把重要文件贴标签留下，不重要的先放回柜子 |
| 注入记忆后继续执行 | 把当前记忆区间内容注入，让模型接入记忆后继续原任务 | 从档案室取回需要的文件，回到工位继续工作 |

---

## 19. 四大核心模块运行机制详解

### 19.1 中断处理器 (InterruptHandler)

**文件**: `zulong/l2/interrupt_handler.py`
**全局实例**: `interrupt_handler`

**角色定位**: L2 推理层内部的"紧急逃生向导"。当收到中断信号时，负责安全地停止当前生成、保存进度、并在之后协调恢复。

#### 19.1.1 内部状态

```python
class InterruptHandler:
    _interrupt_callbacks: List[Callable]   # 中断时通知的回调列表
    _resume_callbacks: List[Callable]      # 恢复时通知的回调列表
    _is_generating: bool                   # 是否正在生成（True=生成中）
    _current_task_id: Optional[str]        # 当前正在生成的任务ID
    _lock: threading.RLock                 # 线程安全锁
```

#### 19.1.2 完整运行流程

```
[正常生成中]
  │
  │  start_generation(task_id)
  │  ├─ 检查是否已在生成（防止重复）
  │  ├─ 标记 _is_generating = True
  │  ├─ 记录 _current_task_id
  │  └─ 清除中断标志: state_manager.set_interrupt_flag(False)
  │
  │  生成循环中每隔N个token:
  │  check_interrupt()
  │  ├─ 读取 state_manager.get_interrupt_flag()
  │  └─ 返回 True/False
  │
  ▼  [收到中断信号]
  │
  │  handle_interrupt(reason)
  │  ├─ 1. state_manager.set_interrupt_flag(True)
  │  ├─ 2. 检查是否有正在生成的任务
  │  ├─ 3. _is_generating = False（停止生成）
  │  ├─ 4. snapshot_manager.freeze(task_id)（冻结快照）
  │  ├─ 5. 遍历 _interrupt_callbacks 通知所有监听者
  │  └─ 6. 返回冻结的 TaskSnapshot
  │
  ▼  [恢复生成]
  │
  │  resume_generation(task_id)
  │  ├─ 1. snapshot_manager.thaw(task_id)（解冻快照）
  │  ├─ 2. _re_evaluate_environment(task_id, snapshot)（重评估）
  │  │     ├─ 获取冻结时的环境快照
  │  │     ├─ 创建当前环境快照
  │  │     ├─ compare_snapshots(旧, 新)
  │  │     └─ 返回 EnvironmentChange:
  │  │         ├─ recommendation="CONTINUE" → 直接继续
  │  │         ├─ recommendation="REPLAN"   → 发布重规划事件
  │  │         └─ recommendation="ABORT"    → 发布任务中止事件
  │  ├─ 3. 设置 _current_task_id, _is_generating = True
  │  ├─ 4. state_manager.set_interrupt_flag(False)
  │  └─ 5. 遍历 _resume_callbacks 通知所有监听者
```

#### 19.1.3 GenerationLoop（生成循环包装器）

```python
# 使用方式:
loop = GenerationLoop(interrupt_handler)
for output in loop.generate(task_id, prompt, model):
    if output["interrupted"]:
        print("被中断了")
        break
    print(output["text"])

# 内部逻辑:
# - 每生成 update_interval 个 token 调用一次 check_interrupt()
# - 中断时 break 退出循环
# - 每个 token 都更新 snapshot_manager 中的执行进度
```

---

### 19.2 任务状态管理器 (TaskStateManager)

**文件**: `zulong/l2/task_state_manager.py`
**全局实例**: `task_state_manager`（单例）

**角色定位**: "任务调度台"。管理哪个任务正在执行、哪些任务被暂停冻结、暂停任务的恢复顺序。

#### 19.2.1 内部状态

```python
class TaskStateManager:  # 线程安全单例
    _active_task_id: Optional[str]              # 当前活跃任务ID（只能有一个）
    _frozen_tasks: Dict[str, TaskSnapshot]      # 冻结的任务字典 {task_id: 快照}
    _task_stack: List[str]                      # 任务栈（后冻先恢复）
    _lock: threading.Lock                       # 线程安全锁
```

#### 19.2.2 完整运行流程

```
[创建新任务] create_task(task_id, context)
  │
  ├─ 已有活跃任务？ → 先调用 freeze_current()
  │     ├─ 创建 TaskSnapshot（对话历史 + 执行进度 + 工作记忆）
  │     ├─ 放入 _frozen_tasks 字典
  │     ├─ 推入 _task_stack 栈
  │     └─ 清除 _active_task_id
  │
  ├─ 创建新的 TaskSnapshot
  └─ 设置 _active_task_id = task_id

[冻结当前任务] freeze_current()
  │
  ├─ 检查是否已在冻结列表（避免重复冻结）
  ├─ 创建 TaskSnapshot（execution_pointer 记录当前步骤）
  ├─ _frozen_tasks[task_id] = snapshot
  ├─ _task_stack.append(task_id)
  └─ _active_task_id = None

[恢复任务] resume_task(task_id, task_graph=None)
  │
  ├─ 从 _frozen_tasks 取出快照（pop）
  ├─ 如果提供了 task_graph:
  │     ├─ 关联 TaskGraph 到快照
  │     └─ _sync_to_memory_graph() 同步到图记忆
  │           ├─ TaskGraphAdapter.sync() 投射节点和边
  │           └─ 建立 dialogue → task 的 REFERENCE 边
  ├─ 已有活跃任务？ → 先冻结
  ├─ _active_task_id = task_id
  ├─ 从 _task_stack 移除
  └─ state_manager.set_l2_status(L2Status.BUSY, task_id)
```

#### 19.2.3 任务栈的作用

任务栈实现"后进先出"恢复策略：

```
操作序列:
1. 用户发任务A → A成为活跃任务
2. 中断，用户发任务B → A冻结入栈 [A], B成为活跃任务
3. 中断，用户发任务C → B冻结入栈 [A, B], C成为活跃任务
4. C完成 → 从栈顶弹出B恢复 [A], B成为活跃任务
5. B完成 → 从栈顶弹出A恢复 [], A成为活跃任务
```

---

### 19.3 注意力控制器 (AttentionController)

**文件**: `zulong/l1b/attention_controller.py`
**全局实例**: `get_attention_controller()` 获取单例

**角色定位**: "前台接待员 + 保安队长"。决定事件的处理策略：立刻打断、排队优先、还是稍后处理。

#### 19.3.1 内部状态

```python
class AttentionController:
    status: str                          # "IDLE" / "BUSY" / "SUSPENDED"
    active_snapshot: Optional[ContextSnapshot]  # 被冻结的任务快照
    current_task_id: Optional[str]       # 当前任务ID
    event_queue: PriorityQueue           # 优先级事件队列
    interrupt_threshold: int = 8         # 中断阈值（>=8强制中断）
    high_priority_threshold: int = 5     # 高优阈值（>=5优先处理）
    stats: Dict                          # 统计信息
```

#### 19.3.2 事件处理决策树 (tick 方法)

```
事件到达 tick(events)
  │
  ├─ 是紧急事件(EMERGENCY_ALERT)或中断级别(priority>=8)?
  │     └─ YES → _handle_interrupt(evt) [强制中断，见下方]
  │
  ├─ 当前状态是 BUSY？
  │     ├─ priority >= 8 → _handle_interrupt(evt) [高优打断]
  │     └─ priority < 8  → _queue_event(evt) [低优排队]
  │
  ├─ 当前状态是 IDLE？
  │     ├─ priority >= 5 → _route_to_l2_direct(evt) [直通L2]
  │     └─ priority < 5  → _queue_event(evt) [批量处理]
  │
  └─ 当前状态是 SUSPENDED？
        └─ 所有事件都排队
```

#### 19.3.3 中断处理核心流程 (_handle_interrupt)

```
Step 1: Freeze（冻结）
  ├─ 如果 status == "BUSY" 且有 current_task_id:
  │     ├─ 调用 _create_l2_snapshot()
  │     │     ├─ 通过 L2SnapshotInterface 冻结当前任务
  │     │     └─ 返回 ContextSnapshot（含task_id, summary, full_history, kv_cache_ptr）
  │     └─ 记录到 active_snapshot
  └─ status = "SUSPENDED"

Step 2: Recompose（重组Prompt）
  ├─ 紧急事件上下文: "⚠️ 紧急事件：{来源} 检测到 {事件}"
  ├─ 旧任务摘要: "📝 [暂停的任务]: '{snapshot.summary}'"
  └─ 组合成新的 Prompt

Step 3: Inject（注入L2）
  └─ _force_l2_respond(recomposed_prompt, priority="IMMEDIATE")
      └─ 通过 EventBus 发布 SYSTEM_INTERRUPT 事件，强制L2立即响应

[紧急事件处理完毕后]

Step 4: on_l2_idle() 被调用
  ├─ 检查 status == "SUSPENDED" 且 active_snapshot 存在
  ├─ 构造恢复 Prompt: "✅ 紧急事件已处理。现在恢复之前的任务..."
  ├─ _load_l2_snapshot(active_snapshot) 恢复快照
  ├─ _force_l2_respond(resume_prompt, priority="NORMAL")
  ├─ active_snapshot = None
  └─ status = "IDLE"
```

#### 19.3.4 排队事件处理

```
on_l2_idle() 中处理排队:
  ├─ 每次最多处理 3 个排队事件
  ├─ 从 PriorityQueue 中取出（按优先级排序）
  └─ _route_to_l2_direct() 逐个路由到 L2
```

---

### 19.4 共享记忆池 (SharedMemoryPool)

**文件**: `zulong/infrastructure/shared_memory_pool.py`
**获取方式**: `await SharedMemoryPool.get_instance()` (异步单例)

**角色定位**: 整个系统的"中央邮局"。所有模块产生的数据都先送到这里，统一登记、分类、存放。

#### 19.4.1 内部状态

```python
class SharedMemoryPool:  # 异步单例
    # 四个分区（仓库）
    _raw_zone: Dict[str, DataEnvelope]      # 原始数据（未拆的包裹）
    _feature_zone: Dict[str, DataEnvelope]  # 特征数据（分拣好的信件）
    _system_zone: Dict[str, DataEnvelope]   # 系统状态（内部通知）
    _memory_zone: Dict[str, DataEnvelope]   # 记忆数据（档案室）

    # 索引（加速查询）
    _trace_index: Dict[str, ZoneType]       # trace_id → 分区映射
    _type_index: Dict[DataType, List[str]]  # 数据类型 → trace_id列表
    _time_index: List[tuple]                # 时间索引

    # 锁（线程安全）
    _zone_locks: Dict[ZoneType, threading.Lock]  # 每个分区一把锁
    _index_lock: threading.Lock                  # 索引锁
```

#### 19.4.2 数据信封 (DataEnvelope)

所有数据进入共享池前必须包装成 DataEnvelope:

```python
@dataclass
class DataEnvelope:
    trace_id: str          # 全局唯一追踪ID
    timestamp: float       # 时间戳
    data_type: DataType    # 数据类型 (VIDEO_FRAME/AUDIO_RAW/TEXT_USER/...)
    zone: ZoneType         # 目标分区 (RAW/FEATURE/SYSTEM/MEMORY)
    payload: Any           # 实际数据
    metadata: Dict         # 附加信息
    parent_trace_id: str   # 父级追踪ID（多模态关联用）
    related_trace_ids: List[str]  # 关联追踪ID列表
    expires_at: float      # 过期时间（自动清理用）
```

#### 19.4.3 数据流向

```
[数据采集]
  │
  ▼
[封装成 DataEnvelope]
  ├─ 分配 trace_id (UUID)
  ├─ 标记 data_type 和 zone
  └─ 设置过期时间（可选）
  │
  ▼
[write() 写入对应分区]
  ├─ 获取分区锁
  ├─ 写入分区字典
  ├─ 更新三个索引 (trace/type/time)
  ├─ 释放锁
  └─ 触发异步延迟保存 (_queue_snapshot_save)
  │
  ▼
[其他模块按需读取]
  ├─ read(trace_id) — 按ID精确读取
  ├─ get_by_type(data_type) — 按类型批量读取
  ├─ get_recent(time_window) — 按时间窗口读取
  └─ build_context_pack() — 构建上下文包（多模态融合）
```

#### 19.4.4 持久化机制

```
[写入触发] write()
  └─ _queue_snapshot_save()
      ├─ 将保存请求放入队列
      └─ 如果没有正在执行的保存任务 → 创建 _process_save_queue()

[延迟合并保存] _process_save_queue()
  ├─ 等待 0.5 秒（让多个写入请求累积）
  ├─ 清空队列（合并所有请求为1次保存）
  └─ 调用 _save_snapshot()

[实际保存] _save_snapshot()
  ├─ 收集四个分区所有数据
  ├─ 序列化为 JSON
  ├─ gzip 压缩
  ├─ 先写临时文件，再原子重命名（防止写入中断）
  └─ 清理旧快照（保留最近20个）

[定时保存] _snapshot_loop() — 每 30 秒检查一次
  └─ 如果有未处理的保存请求 → 处理

[系统退出保存] _sync_shutdown() — atexit 注册
  └─ 强制保存最终快照

[启动恢复] _load_snapshot()
  ├─ 查找最新快照文件
  ├─ 解压 + 反序列化
  └─ 恢复到四个分区 + 索引
```

#### 19.4.5 任务队列功能

共享池还提供任务队列功能，用于 L2 向 L1-B 传递复杂任务子任务列表:

```python
write_task_queue(task_id, task_data)     # 写入任务
read_task_queue(task_id)                 # 读取任务
update_task_queue_status(task_id, status) # 更新状态
list_task_queue(status=None)             # 列出任务
delete_task_queue_item(task_id)          # 删除任务
```

---

## 20. 当前已知问题与分阶段修复计划

> **此章节是后续调试会话的核心参考**，避免重复诊断。
> **最后验证日期**: 2026-04-23（通过代码逐行核实）

### 20.1 问题总表

| 编号 | 模块 | 问题描述 | 严重度 | 状态 |
|------|------|---------|--------|------|
| **BUG-01** | SharedMemoryPool | `_read_from_zone()` 中 `async with self._zone_locks[zone]` 使用 `threading.Lock` 作为异步上下文管理器 | P0 | **已修复** - 现在正确使用 `with self._zone_locks[zone]:` |
| **BUG-02** | StateManager | 缺少 `set_interrupt_flag()` / `get_interrupt_flag()` 方法 | P0 | **已修复** - `core/state_manager.py` 第115-137行已实现，带锁保护 |
| **BUG-03** | TaskStateManager | `get_current_context()` 硬编码返回假数据 | P1 | **已修复** - 现在返回 `self._active_snapshot.context_history` |
| **BUG-04** | TaskStateManager | `_sync_to_memory_graph()` 调用不存在的函数 | P1 | **已修复** - 使用 `from zulong.memory.memory_graph import get_memory_graph` 正确导入 |
| **BUG-05** | AttentionController | `_load_l2_snapshot()` 访问不存在的属性 `snapshot.kv_cache.num_tokens` | P1 | **已修复** - 使用 `hasattr()` 安全检查 + 字典兼容处理 |
| **BUG-06** | InferenceEngine | FC 循环中 `_interrupt_flag` 检查无锁保护，存在竞态条件 | P1 | **已修复** - 所有5处读写均已用 `self._lock` 保护 |
| **BUG-07** | InferenceEngine | FC 循环未正常触发工具调用 | P0 | **已修复** - FC循环工具调用逻辑完整正确，Round 3测试100%通过 |
| **BUG-08** | SnapshotManager | 冻结快照仅保存在内存中，不写入磁盘 | P1 | **已修复** - freeze()时写入`data/snapshots/`，启动时自动加载，thaw()时删除 |
| **BUG-09** | InterruptController | 任务栈只读不弹出 | P2 | **低风险** - `resume_task()` 内部通过 `remove()` 补偿，功能正确但日志误导 |
| **BUG-10** | SmartTagging | 正则表达式中 `\|` 两侧有空格 | P2 | **已修复** - 所有正则表达式格式正确 |
| **BUG-11** | RecoveryNotifier | 文件操作非原子性 | P2 | **未修复** - 仍有 race condition 风险，但影响较低 |
| **BUG-12** | EventBus | 事件优先级被忽略，全部按 FIFO 处理 | P2 | **未修复** - 紧急事件无法优先处理 |
| **BUG-13** | MemoryGraph | `SummarySidecarIndex.load()` 后 reverse_id_map 可能为空导致搜索静默失败 | P2 | **已修复** - 加载后验证 reverse_id_map，node_maps.json 缺失时自动重建映射 |

### 20.2 剩余待修复问题

以下问题优先级较低，不影响核心功能，可按需修复：

| 编号 | 问题 | 建议修复方式 |
|------|------|-------------|
| BUG-09 | InterruptController 日志写"Popped"但实际用 remove | 修改日志文案为 "Resumed task" |
| BUG-11 | RecoveryNotifier 文件操作非原子 | 使用先写临时文件再原子重命名 |
| BUG-12 | EventBus 无优先级队列 | 将内部队列替换为 PriorityQueue |

### 20.3 修复依赖关系

```
第一阶段 ← 所有后续阶段的前置条件
  │
  ├─ 1.1 SharedMemoryPool锁修复 ← 2.1 FC循环依赖共享池
  ├─ 1.2 中断标志接口 ← 2.3 中断检查加锁、3.2 恢复栈操作
  ├─ 1.3 真实上下文 ← 3.5 端到端恢复
  └─ 1.4 MemoryGraph同步 ← 3.1 快照持久化、3.5 端到端恢复
      │
      ▼
第二阶段
  │
  ├─ 2.1 FC循环修复 ← 3.5 端到端恢复需要任务能执行
  └─ 2.2 快照属性修复 ← 3.5 端到端恢复需要注意力正常
      │
      ▼
第三阶段
  │
  └─ 3.5 端到端测试 = 最终验证
```

---

## 21. 调试参考：五大目标效果与实现路径

> **最后验证日期**: 2026-04-23

### 效果1: 能成功处理闲聊和复杂任务

**当前状态**: **已可用** - Round 3 测试 100% 通过
**原阻塞 BUG**: BUG-01(已修复), BUG-07(已修复)
**注意事项**: 任务发送间隔需 >= 20 秒，避免触发系统保护机制

### 效果2: 复杂任务中途退出还能找到原来的任务图谱继续（超级重点）

**当前状态**: **基本可用** - 快照持久化已实现，内存中和重启后均可恢复
**原阻塞 BUG**: BUG-02(已修复), BUG-04(已修复), BUG-05(已修复), BUG-08(已修复)
**仍需验证**: 完整端到端流程（发任务 -> 中断 -> 重启 -> 恢复）需要在实际运行环境中验证

### 效果3: 模型能实时感知自己所在的思维层级

**当前状态**: **代码完整** - `get_focus_path_summary()` 在两处 prompt 构建中注入
**原阻塞 BUG**: BUG-05(已修复)
**仍需验证**: 实际对话中检查日志确认焦点路径正常注入

### 效果4: 能感知是否需要注入记忆/丢弃部分注意力

**当前状态**: **框架可用** - 共享池读取正常，注意力控制器状态机完整
**原阻塞 BUG**: BUG-01(已修复)
**仍需完善**: 上下文窗口预警机制和动态注意力丢弃策略需要更多实际场景测试

### 效果5: 能感知是否需要更多信息并自主检索

**当前状态**: **可用** - FC 循环正常触发，recall_memory/navigate_attention 等工具已注册
**原阻塞 BUG**: BUG-07(已修复)
**仍需验证**: 实际对话中观察模型是否主动调用 recall_memory 检索历史信息

---

### 调试提示（给未来的调试会话）

1. **大部分 BUG 已修复**: 13 个 BUG 中 10 个已修复，仅 BUG-09(低风险)/BUG-11/BUG-12 待处理
2. **每修一个 bug 就验证**: 不要累积修改，逐个验证
3. **启动命令**: `python zulong/bootstrap.py`
4. **关键日志**: `zulong/l2/logs/l2_*.log`
5. **任务恢复测试流程**: 发送复杂任务 -> 等执行到一半 -> 中断 -> 发"继续之前的任务"
6. **快照文件位置**: `data/snapshots/` (冻结时写入，恢复时删除)
