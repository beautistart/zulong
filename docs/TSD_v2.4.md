# 祖龙 (ZULONG) 机器人系统技术规格说明书
## Technical Specification Document (TSD) v2.9

**版本**: v2.9  
**日期**: 2026-04-23  
**状态**: 记忆系统架构全面同步 + 四大核心模块 + 3D记忆城市设计理念

---

# 📋 修订历史

| 版本 | 日期 | 修订内容 | 作者 |
|------|------|---------|------|
| v2.2 | 2026-03-15 | 初始版本：经验库系统架构 | 架构团队 |
| v2.3 | 2026-03-29 | **架构升级增强**：<br>- 数据存储架构（热/冷分层）<br>- 复盘机制<br>- 时间标签体系<br>- 智能打标系统<br>- 系统监控 | 架构团队 |
| v2.3 Rev.2 | 2026-04-04 | **记忆系统全面增强**：<br>- 短期记忆 (基于共享池)<br>- 长期记忆 (RAG 向量库)<br>- 跨会话记忆持久化<br>- 记忆自进化机制 | 架构团队 |
| v2.4 | 2026-04-16 | **工具系统与技能包架构**：<br>- 工具系统 (ToolEngine / BaseTool / ToolRegistry)<br>- 热/冷工具管理 (CoreToolManager / ToolRAG)<br>- Function Calling 多轮推理循环<br>- 技能包系统 (ISkillPack / SkillPackRuntime)<br>- 内置技能包 (task_planner / deep_reasoner) | 架构团队 |
| v2.4.1 | 2026-04-16 | **技能包架构合并**：<br>- 合并 task_planner + deep_reasoner → ComplexTaskPack<br>- LLM 增强版 DeepReasoningEngine（复用 L2 CORE client）<br>- KV Cache Context Slot 热交换机制<br>- 新增 plan_and_reason 联合能力（拆解→推理一体化） | 架构团队 |
| v2.5 | 2026-04-17 | **五阶段流水线框架与无限深度任务图谱**：<br>- 新增 `zulong/pipeline/` 框架层目录<br>- TaskGraph 无限深度递归树架构（模板节点 + 动态生成节点）<br>- 叶子节点执行模型 + 非叶子节点状态聚合<br>- 五阶段流水线：拆解→规划→执行→审查→输出<br>- 零成本复杂度分类器（关键词路由）<br>- 两相位拆解 + 六级递归解析<br>- ReviewGate 审查纠错（最多2次重试）<br>- L2 InferenceEngine 集成<br>- 前端 pipeline 事件推送 | 架构团队 |
| v2.6 | 2026-04-22 | **地址继承系统与图谱事件推送**：<br>- 地址继承系统 (REFERENCE 边 + full_path 传播)<br>- bind_session_to_task 任务回溯机制<br>- 图谱事件推送 (L2_THINKING_STEP + FC循环三发布点)<br>- TRIVIAL 规则扩展 (闲聊问候识别)<br>- Session 分配策略 (Embedding 相似度)<br>- 突触修剪机制 (艾宾浩斯遗忘曲线)<br>- 记忆写入统一 (全部通过 MemoryGraph 适配器) | 架构团队 |
| v2.7 | 2026-04-23 | **记忆系统架构文档同步 + FC 工具补全 + 超时优化**：<br>- TSD 第 8 章重写：三库 RAG 描述 → MemoryGraph 统一异构图架构<br>- 新增 task_add_dependency FC 工具（任务依赖关系）<br>- CircuitBreaker 上下文窗口压力监控<br>- LLM 调用超时优化（CORE 120s / BACKUP 60s / FC 120s）<br>- RAG 系统从 3 库升级为 4 库（新增 ToolRAG）<br>- 工具系统注册表完善（7 个任务工具） | 架构团队 |
| v2.8 | 2026-04-23 | **模型预加载器 + FC 超时优化 + 配置变量修复**：<br>- 新增 ModelPreloader（支持 Ollama/LM Studio/vLLM 自动检测）<br>- FC 循环超时从硬编码 120s 升级为可配置（core=300s, backup=60s, fc_loop=300s）<br>- 修复配置变量跨引用问题（${llm.ollama.model_id} → 直接值）<br>- bootstrap.py 集成模型预加载（后台线程启动，不阻塞）<br>- 删除 task_planner 技能包（FC 工具提供等效功能）<br>- 系统实测验证：简单查询 6 秒响应，工具调用正常 | 架构团队 |
| v2.9 | 2026-04-23 | **记忆系统架构全面同步 v3.0 + 四大核心模块 + 3D记忆城市**：<br>- 第8章全面重写：MemoryGraph统一异构图架构（9种节点+7种边+6个适配器）<br>- 新增3D记忆城市设计理念（宏空间/任务空间/地下档案室）<br>- 新增四大核心模块详解：InterruptHandler、TaskStateManager、AttentionController、SharedMemoryPool<br>- 新增任务恢复机制完整流程（L2自主FC决策+环境快照对比）<br>- 新增记忆遗忘与剪枝完整流程（艾宾浩斯衰减+LLM审查+孤立节点清理）<br>- 新增已知问题与分阶段修复计划（13个BUG+4阶段修复）<br>- 多维标签体系完整描述（温度/重要度/时间段）<br>- 双路径并行检索策略更新（热遍历+冷FAISS互斥过滤）<br>- 地址继承系统完整流程（Session绑定→Round分配→Task传播）<br>- 文件索引更新（102KB memory_graph.py + 49KB graph_adapters.py等） | 架构团队 |


---

# 第 8 章：记忆系统架构

## 8.1 记忆系统设计理念

### 8.1.1 整体架构概述

祖龙系统采用分层架构，记忆系统位于 L2 认知层：

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

### 8.1.2 记忆系统定位

MemoryGraph 是 L2 认知层的**记忆中枢**，它不是独立模块，而是贯穿多个系统层级的**统一集成层**。所有记忆子系统（对话历史、任务状态、知识图谱、经验库、人物档案等）都通过**适配器模式**投射为图中的节点和边，共享同一个 NetworkX DiGraph 实例。

**核心类比**: LLM 是大脑皮层（负责推理），MemoryGraph 是海马体（负责记忆索引和联想）。

### 8.1.3 记忆系统核心子系统

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

### 8.1.4 记忆系统主数据流

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

## 8.2 MemoryGraph 统一异构图架构

### 8.2.1 核心概念

MemoryGraph 是一张**完整的异构类型图** (NetworkX DiGraph)。所有记忆子系统统一投射为节点和边。整张图是连通的，BFS 扩散激活可以在所有节点间自由传播。

**关键特性**: 图是一个整体，不做任何物理分割。概念上的"宏空间 vs 任务空间"只是区域划分，不是物理隔离。

### 8.2.2 节点类型 (NodeType) — 9 种

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

### 8.2.3 边类型 (EdgeType) — 7 种

| 边类型 | 含义 | 是否受保护 | 衰减规则 | 典型场景 |
|--------|------|-----------|----------|----------|
| `HIERARCHY` | 父子层级关系 | protected | 永不修剪 | session->round, task->subtask |
| `DEPENDENCY` | 数据依赖 | protected | 永不修剪 | task1 依赖 task2 完成 |
| `TEMPORAL` | 时间序列 | protected | 永不修剪 | round1->round2 按时间串联 |
| `REFERENCE` | 跨类型引用 | 否 | 按重要度衰减 | dialogue<->task 双向关联 |
| `SEMANTIC` | 语义相似 | 否 | 按重要度衰减 | embedding 余弦相似度 > 0.7 自动创建 |
| `CAUSAL` | 因果关系 | 否 | 按重要度衰减 | 事件 A 导致事件 B |
| `ASSOCIATION` | 赫布学习产生 | 否 | 按重要度衰减 | 共激活 >= 3 次自动创建 |

### 8.2.4 核心数据结构

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

---

## 8.3 多维标签体系

MemoryGraph 使用**三组正交标签**替代传统的硬分区 (L1/L2/L3)，实现灵活的记忆分类和检索路由:

### 8.3.1 标签定义

| 维度 | 存储方式 | 可选值 | 说明 |
|------|---------|-------|------|
| **温度** (temperature) | metadata 存储, 动态更新 | `HOT` / `WARM` / `COLD` | 基于 last_accessed 实时计算 |
| **重要度** (importance) | metadata 存储, 可被提升 | `TRIVIAL` / `NORMAL` / `IDENTITY` / `FACT` / `IMPORTANT` / `MUST_REMEMBER` | 写入时规则检测, LLM 审查/手动可提升 |
| **时间段** (time_scope) | 不存储, 查询时计算 | `RECENT` / `NON_RECENT` | 默认 30 分钟窗口, 可配置 |

### 8.3.2 温度动态更新规则

在每轮 `decay_and_prune()` 中自动计算:

| 条件 | 温度 |
|------|------|
| `last_accessed` 在最近 1 小时内 | `HOT` |
| `last_accessed` 在 1~24 小时之间 | `WARM` |
| `last_accessed` 超过 24 小时 | `COLD` |

节点被 BFS 激活或直接访问时自动设为 `HOT`。

### 8.3.3 重要度分级与半衰期

| 重要度 | 边权衰减半衰期 | 孤立节点容忍 | 自动识别规则 | 示例 |
|--------|---------------|-------------|-------------|------|
| `TRIVIAL` | 6 小时 | 6 小时 | 语气词/极短回复 < 5 字符 | "嗯""好的""你好" |
| `NORMAL` | 24 小时 | 24 小时 | 默认值 | 一般对话 |
| `IDENTITY` | 30 天 | 永不删除 | "我叫"/"我是"/"我姓"/"今年 XX 岁" | "我叫张三" |
| `FACT` | 15 天 | 不删除 | "我家在"/"我的电话是"/"XX 月 XX 号" | "我住在上海" |
| `IMPORTANT` | 7 天 | 24 小时 | access_count >= 3 自动提升, 或"我喜欢"/"我讨厌" | 偏好/高频访问 |
| `MUST_REMEMBER` | 永不衰减 | 永不删除 | "帮我记住"/"别忘了"/"一定要记得" | 用户显式要求 |

**重要度只升不降**: `promote_importance()` 只允许向上提升。降级需要 LLM 审查专门处理。

---

## 8.4 图适配器与后端投射

MemoryGraph 通过 6 个适配器，将各类后端系统的数据**单向投射**为图中的节点和边。适配器是只读桥接层，不修改后端数据。

### 8.4.1 适配器全景

| 适配器 | 数据来源 | 投射的节点类型 | 投射的边类型 |
|--------|---------|---------------|-------------|
| `TaskGraphAdapter` | TaskGraph | TASK, FILE | HIERARCHY (protected), DEPENDENCY (protected), REFERENCE |
| `DialogueAdapter` | 对话数据 | DIALOGUE (session/round/sub_dialogue) | HIERARCHY, TEMPORAL, REFERENCE |
| `KnowledgeGraphAdapter` | KnowledgeGraph | PERSON, CONCEPT, KNOWLEDGE | CAUSAL, ASSOCIATION, REFERENCE |
| `EpisodeAdapter` | EpisodicMemory | EPISODE | TEMPORAL |
| `PersonProfileAdapter` | PersonProfileManager | PERSON | REFERENCE |
| `ExperienceAdapter` | ExperienceRAG | EXPERIENCE | (无自动边) |

### 8.4.2 DialogueAdapter (最关键的适配器)

DialogueAdapter 负责对话数据的完整生命周期管理:

| 方法 | 触发时机 | 功能 |
|------|---------|------|
| `ensure_session()` | 用户消息到达 | 话题边界检测: bigram 重叠率 + bound_task_id 匹配 |
| `add_round()` | 每轮对话 | 创建 round 节点 + 自动重要度检测 + 建立 round->task 的 REFERENCE 边 |
| `add_sub_dialogue()` | FC 工具调用 | 创建 agent_turn 节点 + REFERENCE 到相关 task |
| `finalize_round()` | 对话轮完成 | 归档轮次 + 将 round_summary 写入 FAISS 摘要侧车索引 |
| `assign_session_by_similarity()` | L2 推理后 | embedding 相似度匹配分配 session, 调用地址继承传播 |
| `_detect_importance()` | 节点创建时 | 正则匹配自动标注: "我叫X"->IDENTITY, "帮我记住"->MUST_REMEMBER |
| `_link_to_knowledge()` | round 创建后 | 自动链接到 KG 中的 KNOWLEDGE/PERSON/CONCEPT 节点 |

### 8.4.3 TaskGraphAdapter

**同步模式**:
- `sync()`: 全量同步 -- 遍历 TaskGraph 所有节点/边
- `incremental_sync()`: 增量同步 -- 监听 TaskGraph 变更事件

**关键行为**:
- HIERARCHY 和 DEPENDENCY 边自动设为 `protected=True`
- 任务节点通过 `backend_ref` 指向原始 TaskGraph
- 节点地址继承: 子任务 full_path = 父任务 full_path + "/" + 子 ID

---

## 8.5 双路径并行检索策略

### 8.5.1 检索入口

```python
async def retrieve_context(
    self,
    query_text: str,
    top_k: int = 10,
    hot_window_minutes: int = 30,
    session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
```

### 8.5.2 并行检索流程

**核心原则**: 图是一个整体, 不做物理拆分。两条检索路径通过标签**互斥过滤**在同一张图上各自排除对方的数据范围, 保证无重复。

```
用户输入: "处理器价格"
         |
    asyncio.gather() 并行执行
         |
   +-----+-----+
   |           |
   v           v
路径 A:      路径 B:
热数据遍历    冷数据 FAISS
   |           |
   | 筛选is_recent() | FAISS 向量搜索摘要索引
   | == True的节点  | 排除热数据 (互斥过滤)
   |           |
   | 关键词匹配:    | 命中 Session/EPISODE 节点
   | - label 匹配  | 作为 BFS 种子
   | - content 包含| 下钻获取详情
   | - bigram 重叠 |
   |           |
   | BFS 扩散激活    | resolve_backend_ref()
   | 从匹配节点出发  | 获取 KG/RAG 原始内容
   |           |
   | TEMPORAL 边追溯 |
   | 最近 N 轮对话   |
   |           |
   +-----+------+
         |
         v
合并 (天然无重复, 因为互斥):
  - 热数据 boost 1.5x
  - 高 importance 额外加权
  - session_id 匹配 +0.3
  - 按 score 降序 → Top-K
```

### 8.5.3 摘要侧车索引 (SummarySidecarIndex)

独立维护的 FAISS 向量索引层:

```python
class SummarySidecarIndex:
    """FAISS 索引中只存摘要向量 + node_id 指针"""
    
    def index_summary(self, node_id: str, text: str) -> bool:
        """使用 BAAI/bge-small-zh-v1.5, 512 维"""
        
    def search_by_text(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """向量搜索: text → embedding → FAISS Top-K → (node_id, score)"""
```

**索引维护时机**:
- `DialogueAdapter.finalize_round()` 完成时为 Session 节点更新摘要向量
- `EpisodeAdapter.sync()` 时为 EPISODE 节点生成摘要向量
- 节点删除时同步移除 FAISS 中的向量
- 随 MemoryGraph 一同保存/加载

### 8.5.4 检索结果结构与上下文注入

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

| 节点类型 | 注入格式 |
|---------|---------|
| DIALOGUE | `【历史对话】用户曾问: ... 你回答: ...` |
| TASK | `【相关任务】任务名: ... 状态: ...` |
| KNOWLEDGE | `【知识参考】...` (通过 backend_ref 获取 KG 完整内容) |
| EPISODE | `【历史摘要】...` |
| EXPERIENCE | 不自动注入 (由 FC 按需获取) |

---

## 8.6 核心算法

### 8.6.1 加权 BFS 扩散激活

```python
def compute_activations(self, seed_node_ids, max_depth=3, decay=0.5, min_activation=0.01):
    """从种子节点出发，沿边权衰减传播激活值"""
    # 传播公式: activation[neighbor] = activation[current] × edge_weight × decay
    # 示例 (decay=0.5):
    #   Seed A: 1.0
    #   1-hop B (weight=1.0): 1.0 × 1.0 × 0.5 = 0.5
    #   2-hop C (weight=0.8): 0.5 × 0.8 × 0.5 = 0.2
```

### 8.6.2 赫布学习 (Hebbian Learning)

```python
def hebbian_strengthen(self):
    """共激活边强化 + 自动创建 ASSOCIATION 边"""
    # 强化公式: new_weight = old_weight + eta × (1 - old_weight), eta=0.1
    # 共激活计数器 >= 3 次 → 自动创建 ASSOCIATION 边 (weight=0.5)
```

### 8.6.3 突触修剪 (艾宾浩斯衰减)

```python
def decay_and_prune(self):
    """衰减公式: decayed = weight × exp(-elapsed_hours × ln(2) / half_life)"""
    # weight < 0.05 → 删除边
    # 0.05~0.15 → 标记 pending_review (送 LLM 审查)
    # >= 0.15 → 更新权重
    # protected 边永不衰减 (HIERARCHY/DEPENDENCY/TEMPORAL)
```

### 8.6.4 语义近邻自动发现

```python
def discover_semantic_neighbors(self, node_id, top_k=5, threshold=0.7):
    """通过 embedding 余弦相似度自动发现近邻并创建 SEMANTIC 边"""
```

---

## 8.7 地址继承系统

### 8.7.1 核心概念

**节点地址像文件系统路径一样，子节点地址携带完整的父级路径。** 这使得从任意子节点都能追溯完整上级路径。

### 8.7.2 地址格式

```
Session 根节点:    dialogue:session_abc123
Round 节点:        dialogue:session_abc123/dialogue:round_xxx
Sub-dialogue:      dialogue:session_abc123/dialogue:round_xxx/turn_agent_xxx
任务根节点:        dialogue:session_abc123/task:tg_xxx
任务子节点:        dialogue:session_abc123/task:tg_xxx/o1
任务孙节点:        dialogue:session_abc123/task:tg_xxx/o1_1
```

### 8.7.3 地址继承完整流程

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
```

---

## 8.8 记忆遗忘与剪枝机制

### 8.8.1 艾宾浩斯衰减公式

```
decayed_weight = weight × exp(-elapsed_hours × ln(2) / half_life)
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

### 8.8.2 自动重要度检测规则

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

### 8.8.3 剪枝完整流程

```
decay_and_prune() -- 后台定时器每 30 分钟触发
  |
  |-- 1. 遍历所有边, 计算衰减后权重
  |-- 2. 修剪低权重边 (weight < 0.05 删除, 0.05~0.15 标记 pending_review)
  |-- 3. LLM 审查 (异步, 不阻塞)
  |     |-- KEEP --> 强化边权 + 提升 importance
  |     |-- DISCARD --> 下一轮剪枝删除
  |     |-- COMPRESS --> 压缩 metadata 为摘要
  |     |-- PROMOTE --> 提升节点 importance
  |     |-- MERGE --> 合并到关联节点
  |-- 4. 同步更新温度标签 (HOT/WARM/COLD)
  |-- 5. 检查孤立节点 (IDENTITY 保留, TRIVIAL 孤立>6h 删除)
  |-- 6. 语义孤立检测 (空壳清理)
  |-- 7. 自动重要度提升 (access_count >= 3 提升)
  |-- 8. 记录统计 + 持久化
```

---

## 8.9 记忆巩固与自演进

### 8.9.1 记忆强度模型

**核心文件**: `zulong/memory/memory_evolution.py`

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

### 8.9.2 记忆巩固器

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

### 8.9.3 与 MemoryGraph 的集成

MemoryStrength 的衰减追踪结构存储在 `GraphNode.metadata["memory_strength"]` 中。`decay_and_prune()` 在衰减过程中同步更新这些值, 实现图级衰减和节点级衰减的统一。

---

## 8.10 经验数据与事件记忆区分

### 8.10.1 五维区分机制

| 维度 | 经验数据 (ExperienceRAG) | 事件记忆 (MemoryGraph DIALOGUE 节点) |
|------|------------------------|----------------------------------|
| **存储位置** | ExperienceRAG + EXPERIENCE 节点 | MemoryGraph DIALOGUE 节点 |
| **数据结构** | Experience (带 tags/keywords/importance_score) | GraphNode (带 importance/temperature) |
| **生成路径** | 复盘提取 / L2 自动生成 / 用户确认 | 对话自动固化 |
| **评估标准** | 置信度 >= 0.6, 可复用性, 去重 | 访问次数, 情感权重, 时间衰减 |
| **检索方式** | 混合检索 (向量 + BM25 + 标签), FC 按需调用 | 图 BFS + FAISS, 自动注入 |

### 8.10.2 隔离设计

复盘对话本身**不存入** MemoryRAG/MemoryGraph, 仅保存在临时缓冲区。只有经过 L2 分析和用户确认的经验才存入 ExperienceRAG。缓冲区在复盘结束后自动清空。

---

## 8.11 旧组件处置

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

# 第 8A 章：3D 记忆城市设计理念

## 8A.1 设计哲学

祖龙的记忆系统不是传统的"对话历史列表"或"向量数据库"，而是一个**三维立体的记忆城市**。记忆图谱和任务图谱共同构成了这座城市的空间结构。

**核心原则**:

1. **模型自由游走** - 模型像一个人在城市中行走，可以自由决定去哪条街、进哪栋楼、看哪个房间。不是被铁轨绑死在固定路线上。

2. **按需读取** - 模型根据当前任务需要，自己判断需要读取哪些记忆/文件，自己决定把注意力调整到什么范围。

3. **弹性注意力** - 有时候需要"全景视角"看整体，有时候需要"显微镜"聚焦到某个细节。模型自己判断什么时候需要全部注意力集中到某个局部。

4. **参考模板而非流程捆绑** - 复杂任务有参考模板（类似操作手册），但模型不是机械地按流程办事。模板是建议，不是枷锁。模型可以跳步、回退、插入新步骤。

5. **跨会话持久** - 即使系统重启、任务中断，记忆城市依然存在。模型可以随时回到之前离开的位置继续工作。

## 8A.2 3D 空间比喻

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

## 8A.3 注意力切换规则

| 场景 | 注意力行为 | 比喻 |
|------|-----------|------|
| 进入一层节点 | 默认获取该层节点及全局注意力（扩展覆盖） | 站在街区入口，能看到整个街区和远处的城市天际线 |
| 情报节点以上下层概括 | 获取层节点摘要，不展开细节 | 看楼的门牌号和招牌，不进去 |
| 需要更详细联想检索 | 进入单层信息节点，下钻获取子节点内容 | 走进楼里，看每层的具体内容 |
| 进入子节点 | 全局注意力切换到节点以内，聚焦相关集合，丢弃无关长上下文 | 进入房间后，关上门，集中注意力在房间内的东西 |
| 上下文窗口达到预警 | 模型把当前交互关联的元素加标签再驱动注意力，丢弃碎片关联 | 桌子满了，把重要文件贴标签留下，不重要的先放回柜子 |
| 注入记忆后继续执行 | 把当前记忆区间内容注入，让模型接入记忆后继续原任务 | 从档案室取回需要的文件，回到工位继续工作 |

---

# 第 8B 章：四大核心模块运行机制

## 8B.1 中断处理器 (InterruptHandler)

**文件**: `zulong/l2/interrupt_handler.py`
**全局实例**: `interrupt_handler`

**角色定位**: L2 推理层内部的"紧急逃生向导"。当收到中断信号时，负责安全地停止当前生成、保存进度、并在之后协调恢复。

### 8B.1.1 内部状态

```python
class InterruptHandler:
    _interrupt_callbacks: List[Callable]   # 中断时通知的回调列表
    _resume_callbacks: List[Callable]      # 恢复时通知的回调列表
    _is_generating: bool                   # 是否正在生成（True=生成中）
    _current_task_id: Optional[str]        # 当前正在生成的任务ID
    _lock: threading.RLock                 # 线程安全锁
```

### 8B.1.2 完整运行流程

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

### 8B.1.3 GenerationLoop（生成循环包装器）

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

## 8B.2 任务状态管理器 (TaskStateManager)

**文件**: `zulong/l2/task_state_manager.py`
**全局实例**: `task_state_manager`（单例）

**角色定位**: "任务调度台"。管理哪个任务正在执行、哪些任务被暂停冻结、暂停任务的恢复顺序。

### 8B.2.1 内部状态

```python
class TaskStateManager:  # 线程安全单例
    _active_task_id: Optional[str]              # 当前活跃任务ID（只能有一个）
    _frozen_tasks: Dict[str, TaskSnapshot]      # 冻结的任务字典 {task_id: 快照}
    _task_stack: List[str]                      # 任务栈（后冻先恢复）
    _lock: threading.Lock                       # 线程安全锁
```

### 8B.2.2 完整运行流程

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

### 8B.2.3 任务栈的作用

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

## 8B.3 注意力控制器 (AttentionController)

**文件**: `zulong/l1b/attention_controller.py`
**全局实例**: `get_attention_controller()` 获取单例

**角色定位**: "前台接待员 + 保安队长"。决定事件的处理策略：立刻打断、排队优先、还是稍后处理。

### 8B.3.1 内部状态

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

### 8B.3.2 事件处理决策树 (tick 方法)

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

### 8B.3.3 中断处理核心流程 (_handle_interrupt)

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

### 8B.3.4 上下文快照 (ContextSnapshot)

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

### 8B.3.5 基于图记忆的自主动态注意力机制

基于图记忆机制的动态注意力分为两个板块：

#### 板块一：思维深度导航

L2 模型通过 `navigate_attention` FC 工具主动控制注意力焦点：

| 操作 | 行为 | 代码对应 |
|------|------|---------|
| `deeper` | 深入当前焦点的子节点 | `attention_tool.py` NavigateAttentionTool |
| `broader` | 回退到父节点 | `attention_tool.py` NavigateAttentionTool |
| `jump` | 跳转到指定节点 ID | `attention_tool.py` NavigateAttentionTool |

#### 板块二：基于 BFS 扩散的三类注意力

| 类型 | 机制 | 代码对应 |
|------|------|---------|
| 全局注意力 | BFS 从种子节点沿加权边扩散 | `memory_graph.py` compute_activations() |
| 单链注意力 | 沿 HIERARCHY 父子链聚焦 | `attention_tool.py` deeper/broader |
| 局部注意力 | 当前焦点节点的直接邻域 | `memory_graph_tools.py` discover_related |

---

## 8B.4 共享记忆池 (SharedMemoryPool)

**文件**: `zulong/infrastructure/shared_memory_pool.py`
**获取方式**: `await SharedMemoryPool.get_instance()` (异步单例)

**角色定位**: 整个系统的"中央邮局"。所有模块产生的数据都先送到这里，统一登记、分类、存放。

### 8B.4.1 内部状态

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

### 8B.4.2 数据信封 (DataEnvelope)

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

### 8B.4.3 数据流向

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

### 8B.4.4 持久化机制

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

### 8B.4.5 任务队列功能

共享池还提供任务队列功能，用于 L2 向 L1-B 传递复杂任务子任务列表:

```python
write_task_queue(task_id, task_data)     # 写入任务
read_task_queue(task_id)                 # 读取任务
update_task_queue_status(task_id, status) # 更新状态
list_task_queue(status=None)             # 列出任务
delete_task_queue_item(task_id)          # 删除任务
```

---

## 8B.5 任务恢复机制

### 8B.5.1 完整恢复流程（基于图记忆的 L2 自主决策）

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

### 8B.5.2 TaskSuspensionManager

**文件**: `zulong/l2/task_suspension.py`

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
        
    async def resume_task(self, task_id: str) -> Optional[SuspendableTaskState]:
        """恢复: 读取 JSON -> 反序列化 TaskGraph -> 删除文件"""
        
    async def find_by_description(self, query: str) -> Optional[Dict]:
        """模糊匹配: 字符级相似度 (对中文友好)"""
```

### 8B.5.3 RecoveryNotifier

**文件**: `zulong/l2/recovery_notifier.py`

系统启动时自动执行:
1. 扫描 `./data/checkpoints/` 中的崩溃检查点
2. 将检查点提升为挂起任务 (`_promote_checkpoints()`)
3. 列出所有可恢复的挂起任务
4. 通过 EventBus 发送 L2_OUTPUT 事件通知用户

### 8B.5.4 恢复场景识别

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

---

# 第 9 章：数据存储架构

## 9.1 分层存储策略

### 9.1.1 设计理念

基于**数据热度**和**访问频率**，采用分层存储策略：
- **热存储**：最近 7-14 天数据，高频访问，完整日志
- **冷存储**：超过 14 天数据，低频访问，压缩归档

**目标**：
- 降低存储成本 30%+
- 保持查询性能（热数据 < 100ms）
- 满足合规要求（6 个月留存）

---

### 9.1.2 热存储（Hot Storage）

**技术选型**: MongoDB

**存储内容**:
- 完整执行日志（含传感器快照引用）
- 系统状态快照
- 用户交互记录
- 任务执行轨迹

**保留周期**: 14 天（TTL 索引自动清理）

**数据结构**:

```json
{
  "trace_id": "trace_20260329_001",
  "timestamp": "2026-03-29T10:30:00Z",
  "status": "SUCCESS",
  
  "user_input": {
    "text": "帮我拿一瓶水",
    "voice_data": "base64_encoded_audio",
    "location": "kitchen"
  },
  
  "sensor_snapshot": {
    "timestamp": "2026-03-29T10:30:00Z",
    "cameras": [
      {"id": "cam_front", "view": "front", "image_ref": "s3://img_001.jpg"}
    ],
    "lidar": {"point_cloud_ref": "s3://lidar_001.pcd"},
    "proprioception": {
      "arm_pos": [0.5, 0.3, 0.2],
      "gripper_state": "open"
    }
  },
  
  "system_state": {
    "system_prompt": "zulong_v2.3",
    "injected_context": ["exp_001", "exp_002"],
    "active_skills": ["navigation", "manipulation"]
  },
  
  "execution_steps": [
    {
      "step_id": 1,
      "agent": "L2_Planner",
      "thought": "用户需要水，先规划到厨房的路径",
      "tool_call": {"tool": "navigate", "params": {"target": "kitchen"}},
      "output": {"status": "success"},
      "result": "到达厨房"
    }
  ],
  
  "final_output": "已拿到水，放在桌子上",
  
  "cost_stats": {
    "total_time_ms": 5230,
    "llm_cost": 0.012,
    "token_usage": 1250
  }
}
```

**索引策略**:
```python
# 单字段索引
db.logs.createIndex({"timestamp": -1})
db.logs.createIndex({"trace_id": 1})
db.logs.createIndex({"status": 1})

# 复合索引
db.logs.createIndex({"timestamp": -1, "status": 1})
db.logs.createIndex({"user_input.text": 1, "timestamp": -1})

# TTL 索引（14 天自动删除）
db.logs.createIndex({"timestamp": 1}, {expireAfterSeconds: 1209600})
```

---

### 9.1.3 冷存储（Cold Storage）

**技术选型**: MinIO / S3 对象存储

**存储内容**:
- 历史日志（压缩打包）
- 已复盘案件卷宗
- 传感器原始数据（图片、点云）

**存储格式**: `.json.gz` 压缩文件

**保留周期**: 至少 6 个月

**文件命名规范**:
```
logs/YYYY/MM/DD/HH/{trace_id}.json.gz
archives/YYYY/MM/batch_{batch_id}.tar.gz
```

**目录结构**:
```
s3://zulong-cold-storage/
├── logs/
│   ├── 2026/
│   │   ├── 03/
│   │   │   ├── 29/
│   │   │   │   ├── 10/
│   │   │   │   │   ├── trace_001.json.gz
│   │   │   │   │   └── trace_002.json.gz
│   │   │   │   └── 11/
│   │   │   │       └── ...
│   │   │   └── 30/
│   │   │       └── ...
│   │   └── 04/
│   │       └── ...
│   └── archives/
│       ├── 2026-03-batch-001.tar.gz
│   └── reviews/
│       ├── case_001/
│       │   ├── log.json.gz
│       │   ├── analysis.json
│       │   └── experience.json
│       └── case_002/
│           └── ...
```

---

### 9.1.4 冷热数据迁移

**触发条件**:
- 定时任务：每日凌晨 02:00
- 手动触发：管理员命令

**迁移流程**:

```
1. 查询热存储中超过 14 天的日志
   ↓
2. 按日期分组打包（每批最多 1000 条）
   ↓
3. 压缩为 .json.gz 文件
   ↓
4. 上传到冷存储（S3/MinIO）
   ↓
5. 验证上传成功
   ↓
6. 从热存储删除（如果 enable_ttl=false）
   ↓
7. 记录迁移日志
```

---

## 9.2 日志收集器

### 9.2.1 L1-B 核心循环集成

**集成点**: `zulong/l1b/core.py`

**代码结构**:

```python
from zulong.storage.logger import get_logger

class L1BCore:
    def __init__(self):
        self.logger = get_logger()
        
    async def main_loop(self):
        """主循环"""
        while True:
            event = await self.event_bus.receive()
            
            # 开始追踪
            trace_id = self.logger.start_trace(
                event_type=event.type,
                user_input=event.payload
            )
            
            try:
                # 处理事件
                result = await self.handle_event(event)
                
                # 记录成功
                self.logger.end_trace(
                    trace_id=trace_id,
                    status="SUCCESS",
                    result=result
                )
                
            except Exception as e:
                # 记录失败
                self.logger.end_trace(
                    trace_id=trace_id,
                    status="FAILED",
                    error=str(e)
                )
                
                # 触发复盘
                if self.should_review(e):
                    await self.trigger_review(trace_id)
```

---

### 9.2.2 异步日志队列

**设计模式**: 生产者 - 消费者模式

```python
import asyncio
from typing import Dict, Any

class AsyncLogQueue:
    """异步日志队列"""
    
    def __init__(self, max_size: int = 10000):
        self.queue = asyncio.Queue(maxsize=max_size)
        self._writers = []
        
    async def put(self, log_entry: Dict[str, Any]):
        """添加日志（非阻塞）"""
        try:
            self.queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            # 队列满时丢弃最旧日志
            logger.warning("日志队列已满，丢弃旧日志")
            self.queue.get_nowait()
            self.queue.put_nowait(log_entry)
    
    async def get(self) -> Dict[str, Any]:
        """获取日志（异步阻塞）"""
        return await self.queue.get()
    
    def qsize(self) -> int:
        """队列大小"""
        return self.queue.qsize()
```

---

# 第 10 章：系统监控与运维

## 10.1 记忆系统健康度指标

**监控指标**:

| 指标 | 计算方式 | 健康阈值 | 告警阈值 |
|------|---------|---------|---------|
| 短期记忆命中率 | reads / writes | > 0.3 | < 0.1 |
| 记忆巩固率 | consolidations / writes | > 0.1 | < 0.05 |
| 快照保存间隔 | 上次保存时间 | < 60s | > 120s |
| RAG 库文档数 | len(documents) | > 10 | < 5 |
| 向量检索延迟 | search 耗时 | < 100ms | > 500ms |

**查询命令**:
```python
# 短期记忆统计
stats = short_term_memory.get_stats()
print(f"当前轮数：{stats['current_turn']}")
print(f"缓存轮数：{stats['cached_turns']}")
print(f"总写入：{stats['total_writes']}")
print(f"总读取：{stats['total_reads']}")

# RAG 库统计
stats = rag_manager.get_statistics()
print(f"经验库文档数：{stats['libraries']['experience']['total_documents']}")
print(f"记忆库文档数：{stats['libraries']['memory']['total_documents']}")
print(f"知识库文档数：{stats['libraries']['knowledge']['total_documents']}")
```

---

## 10.2 故障排查指南

### 10.2.1 跨会话记忆丢失

**症状**: 重启后历史对话丢失

**排查步骤**:
1. 检查快照文件是否存在
   ```bash
   ls -la ./data/shared_memory_pool/snapshot_*.json.gz
   ```

2. 检查最新快照时间
   ```bash
   stat ./data/shared_memory_pool/snapshot_*.json.gz | grep Modify
   ```

3. 验证快照内容
   ```python
   import gzip, json
   with gzip.open("./data/shared_memory_pool/snapshot_latest.json.gz", "rt") as f:
       data = json.load(f)
       print(f"Memory Zone: {len(data.get('memory_zone', {}))} 条")
   ```

4. 手动加载测试
   ```python
   pool = await SharedMemoryPool.get_instance()
   print(f"恢复数据：{len(pool._memory_zone)} 条")
   ```

---

### 10.2.2 RAG 检索效果差

**症状**: 检索结果不相关

**排查步骤**:
1. 检查 Embedding 模型加载
   ```python
   from zulong.memory.embedding_manager import get_embedding_manager
   emb = get_embedding_manager()
   print(f"模型已加载：{emb._model is not None}")
   print(f"向量维度：{emb.get_model_info()['dimension']}")
   ```

2. 检查向量维度匹配
   ```python
   # 应该是 512 维 (BGE-small-zh-v1.5)
   assert emb.get_model_info()['dimension'] == 512
   ```

3. 检查 FAISS 索引
   ```python
   from zulong.memory.rag_manager import RAGManager
   rag = RAGManager()
   for lib_name, lib in rag.rag_libraries.items():
       print(f"{lib_name}: {lib.vector_store.index.ntotal} 个向量")
   ```

4. 测试检索效果
   ```python
   results = rag.search("memory", query="你好", top_k=5)
   for doc in results:
       print(f"内容：{doc.content}")
       print(f"相似度：{doc.similarity}")
   ```

---

# 第 11 章：模型预加载器 (v2.8 新增)

.1 记忆系统配置.1 设计目标

#.1 记忆系统配置.1.1 问题背景

系统启动后，首次调用 LLM 需要加载模型到内存（冷启动），这个过程可能需要 30-120 秒，导致：
- 用户首次交互等待时间过长
- FC 循环中超时风险增加
- 复杂任务处理可能被中断

#.1 记忆系统配置.1.2 解决方案

**ModelPreloader** 在系统启动时后台加载常用模型，避免冷启动延迟。

**核心特性**:
- 支持多种后端自动检测：Ollama、LM Studio、vLLM
- 后台线程启动，不阻塞系统启动流程
- 灵活的配置支持（通过 `zulong_config.yaml`）

.1 记忆系统配置.2 架构设计

**核心文件**: `zulong/utils/model_preloader.py`

```python
class ModelPreloader:
    """模型预加载器，支持多种后端"""
    
    def _detect_backend(self) -> str:
        """自动检测后端类型"""
        # 1. 检测 Ollama: GET /api/ps (默认端口 11434)
        # 2. 检测 LM Studio: GET /v1/models (默认端口 1234)
        # 3. 检测 vLLM: GET /v1/models (默认端口 8000)
        # 4. 返回检测到的后端类型，默认 "ollama"
        
    def _preload_ollama(self):
        """预加载 Ollama 模型"""
        # 使用 /api/generate 接口，设置 keep_alive="60m"
        # 空请求触发加载，不等待响应
        
    def _preload_openai_compatible(self):
        """预加载 OpenAI 兼容后端模型 (LM Studio/vLLM)"""
        # 使用 OpenAI SDK 发送空请求进行预热
```

.1 记忆系统配置.3 后端检测逻辑

```
系统启动
  ↓
ModelPreloader._detect_backend()
  ↓
尝试 Ollama: http://localhost:11434/api/ps
  ├─ 成功 → 返回 "ollama"
  └─ 失败 ↓
尝试 LM Studio: http://localhost:1234/v1/models
  ├─ 成功 → 返回 "lm_studio"
  └─ 失败 ↓
尝试 vLLM: http://localhost:8000/v1/models
  ├─ 成功 → 返回 "vllm"
  └─ 失败 ↓
默认返回 "ollama"
```

.1 记忆系统配置.4 配置方式

**配置文件**: `config/zulong_config.yaml`

```yaml
llm:
  preload:
    enabled: true
    # 后端类型: auto (自动检测) / ollama / lm_studio / vllm
    backend: "auto"
    # 要预加载的模型 (支持多个)
    models:
      - "qwen3.5:4b"         # 备份模型
      - "deepseek-v3.1:671b-cloud"  # 核心模型
    # Ollama 配置
    ollama_base_url: "http://localhost:11434"
    keep_alive: "60m"        # 模型保持时间
    # OpenAI 兼容后端配置
    openai_base_url: "http://localhost:1234/v1"
    openai_api_key: "not-needed"
```

.1 记忆系统配置.5 集成方式

**集成点**: `zulong/bootstrap.py` (SharedMemoryPool 初始化后)

```python
# 模型预加载（后台进行，不阻塞启动）
logger.info("🔥 [BOOTSTRAP] 启动模型预加载（后台）...")
try:
    from zulong.utils.model_preloader import preload_model_from_config
    preload_model_from_config(config_manager)
    logger.info("✅ [BOOTSTRAP] 模型预加载已在后台启动")
except Exception as e:
    logger.warning(f"⚠️ [BOOTSTRAP] 模型预加载启动失败（非致命）: {e}")
```

.1 记忆系统配置.6 预加载流程

```
preload_model_from_config(config_manager)
  ↓
读取 zulong_config.yaml 中的 llm.preload 配置
  ↓
创建 ModelPreloader 实例
  ↓
启动后台线程: threading.Thread(target=_preload_all, daemon=True)
  ↓
后台线程执行:
  for model in models:
    if backend == "ollama":
      _preload_ollama(model)  # 调用 /api/generate
    else:
      _preload_openai_compatible(model)  # 调用 OpenAI SDK
  ↓
模型加载到内存，系统启动完成
```

.1 记忆系统配置.7 注意事项

1. **非阻塞设计**: 预加载在后台线程进行，不会阻塞系统启动
2. **失败容错**: 预加载失败不影响系统正常运行，只会记录警告日志
3. **模型保持**: Ollama 的 keep_alive="60m" 确保模型在 60 分钟内不会被卸载
4. **多模型支持**: 可以配置多个模型同时预加载

---

# 第 12A 章：已知问题与分阶段修复计划

> **此章节是后续调试会话的核心参考**，避免重复诊断。

## 12A.1 问题总表

| 编号 | 模块 | 问题描述 | 严重度 | 影响范围 |
|------|------|---------|--------|---------|
| **BUG-01** | SharedMemoryPool | `_read_from_zone()` 中 `async with self._zone_locks[zone]` 使用 `threading.Lock` 作为异步上下文管理器，运行时报 TypeError | P0 | 共享池异步读取崩溃 |
| **BUG-02** | StateManager | 缺少 `set_interrupt_flag()` / `get_interrupt_flag()` 方法，但 InterruptHandler 和 InterruptController 都在调用 | P0 | 整个中断机制失效 |
| **BUG-03** | TaskStateManager | `get_current_context()` 硬编码返回 `[{"role":"user","content":"Hello"}]`，不返回真实上下文 | P1 | 恢复任务时获取不到真实对话历史 |
| **BUG-04** | TaskStateManager | `_sync_to_memory_graph()` 调用不存在的 `get_memory_graph()` 函数 | P1 | 任务恢复时无法同步到图记忆 |
| **BUG-05** | AttentionController | `_load_l2_snapshot()` 第297行访问 `snapshot.kv_cache.num_tokens`，但 ContextSnapshot 的属性是 `kv_cache_ptr`（没有 `.num_tokens`） | P1 | 任务恢复时注意力控制器 AttributeError 崩溃 |
| **BUG-06** | InferenceEngine | FC 循环中 `_interrupt_flag` 检查无锁保护，存在竞态条件 | P1 | 中断信号可能不被及时检测 |
| **BUG-07** | InferenceEngine | FC 循环未正常触发工具调用，复杂任务全部超时降级返回固定17字符响应 | P0 | 复杂任务完全无法执行 |
| **BUG-08** | SnapshotManager | 冻结快照仅保存在内存中，不写入磁盘，系统崩溃时全部丢失 | P1 | 任务恢复不可靠 |
| **BUG-09** | InterruptController | 任务栈只读 `task_stack[-1]` 不弹出 `pop()`，每次恢复同一任务 | P2 | 恢复后死循环 |
| **BUG-10** | SmartTagging | 正则表达式中 `\|` 两侧有空格（如 `慢 \| 卡`），导致规则永远不匹配 | P2 | 记忆自动打标失效 |
| **BUG-11** | RecoveryNotifier | 文件操作非原子性（`os.remove(ckpt_path)` 可能在 `os.replace` 后崩溃导致重复处理） | P2 | 恢复通知可能重复 |
| **BUG-12** | EventBus | 事件优先级被忽略，全部按 FIFO 处理 | P2 | 紧急事件无法优先处理 |
| **BUG-13** | MemoryGraph | `SummarySidecarIndex.load()` 中访问 `self._store.reverse_id_map`，该属性可能不存在 | P2 | FAISS 索引加载可能失败 |

## 12A.2 分阶段修复计划

### 第一阶段：修通基础设施

**目标**: 系统启动后，基础模块之间的接口调用不报错

| 步骤 | 修复 | 涉及文件 | 验证方法 |
|------|------|---------|---------|
| 1.1 | **BUG-01**: `_read_from_zone()` 中 `use_lock=True` 路径把 `async with threading.Lock()` 改为 `with threading.Lock()` | `shared_memory_pool.py` | 启动系统不报 TypeError |
| 1.2 | **BUG-02**: StateManager 添加 `_interrupt_flag` 属性和 `set_interrupt_flag()` / `get_interrupt_flag()` 方法 | `state_manager.py` | InterruptHandler 能正常设置和读取中断标志 |
| 1.3 | **BUG-03**: `get_current_context()` 改为返回活跃任务的真实上下文 | `task_state_manager.py` | 返回值包含真实对话历史 |
| 1.4 | **BUG-04**: `_sync_to_memory_graph()` 改为使用 `MemoryGraph._instance` 或 `MemoryGraph()` 单例 | `task_state_manager.py` | 任务恢复时同步成功 |

### 第二阶段：修通任务系统

**目标**: 复杂任务能执行完成，工具调用能被触发

| 步骤 | 修复 | 涉及文件 | 验证方法 |
|------|------|---------|---------|
| 2.1 | **BUG-07**: 诊断并修复 FC 循环未触发工具调用的根本原因 | `inference_engine.py` | 发送复杂任务后日志出现工具调用记录 |
| 2.2 | **BUG-05**: `_load_l2_snapshot()` 中 `snapshot.kv_cache.num_tokens` 改为安全访问 `snapshot.kv_cache_ptr` | `attention_controller.py` | 恢复快照不再报 AttributeError |
| 2.3 | **BUG-06**: 推理引擎中断标志检查加锁 | `inference_engine.py` | 中断信号及时检测 |

### 第三阶段：修通任务恢复（超级重点）

**目标**: 任务中途退出后能找到原来的任务图谱，从断点继续

| 步骤 | 修复 | 涉及文件 | 验证方法 |
|------|------|---------|---------|
| 3.1 | **BUG-08**: SnapshotManager 冻结时同时写入磁盘 (`data/snapshots/`) | `snapshot_manager.py` | 冻结后磁盘有快照文件 |
| 3.2 | **BUG-09**: InterruptController 恢复时用 `pop()` 弹出栈顶 | `interrupt_controller.py` | 恢复后栈中不再有该任务 |
| 3.3 | TaskGraph 序列化/反序列化完整性验证 | `task_graph.py` | 序列化再反序列化，节点和边完全一致 |
| 3.4 | RecoveryNotifier 启动扫描可恢复任务 | `recovery_notifier.py` | 重启后提示用户可恢复的任务 |
| 3.5 | 端到端测试: 执行 → 中断 → 恢复 → 继续 | 全链路 | 从断点继续，不从头开始 |

### 第四阶段：优化和稳定

| 步骤 | 修复 | 涉及文件 |
|------|------|---------|
| 4.1 | **BUG-10**: SmartTagging 正则空格修复 | `smart_tagging.py` |
| 4.2 | **BUG-11**: RecoveryNotifier 文件操作原子性 | `recovery_notifier.py` |
| 4.3 | **BUG-12**: EventBus 实现优先级队列 | `event_bus.py` |
| 4.4 | **BUG-13**: FAISS 索引加载安全检查 | `memory_graph.py` |

### 12A.3 修复依赖关系

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

## 12A.4 调试参考：五大目标效果与实现路径

### 效果1: 能成功处理闲聊和复杂任务

**当前状态**: 闲聊可用，复杂任务超时降级
**阻塞原因**: BUG-01(共享池锁), BUG-07(FC循环未触发)
**修复路径**: 第一阶段(1.1) → 第二阶段(2.1)

### 效果2: 复杂任务中途退出还能找到原来的任务图谱继续（超级重点）

**当前状态**: 完全不工作
**阻塞原因**: BUG-02, BUG-04, BUG-05, BUG-08, BUG-09
**修复路径**: 第一阶段全部 → 第二阶段(2.2, 2.3) → 第三阶段全部

### 效果3: 模型能实时感知自己所在的思维层级

**当前状态**: BFS 和焦点路径代码已实现，但 BUG-05 导致注意力控制器崩溃
**阻塞原因**: BUG-05(快照属性不匹配)
**修复路径**: 第二阶段(2.2)，并确认 `get_focus_path_summary()` 正常注入 system prompt

### 效果4: 能感知是否需要注入记忆/丢弃部分注意力

**当前状态**: 共享池无法异步读取
**阻塞原因**: BUG-01(共享池锁)
**修复路径**: 第一阶段(1.1)，需要在推理引擎中实现上下文窗口预警和动态管理

### 效果5: 能感知是否需要更多信息并自主检索

**当前状态**: FC 循环不触发，工具无法被调用
**阻塞原因**: BUG-07(FC循环)
**修复路径**: 第二阶段(2.1)，确保 InfoGapDetector 与 FC 循环集成

### 调试提示

1. **修复顺序必须遵循依赖关系**: 第一阶段是基础，不能跳过
2. **每修一个 bug 就验证**: 不要累积修改，逐个验证
3. **启动命令**: `python zulong/bootstrap.py`
4. **关键日志**: `zulong/l2/logs/l2_*.log`
5. **任务恢复测试流程**: 发送复杂任务 → 等执行到一半 → 中断 → 发"继续之前的任务"

---

# 第 13 章：配置与部署

.1 记忆系统配置.1 记忆系统配置

**配置文件**: `config/memory_config.yaml`

```yaml
# 短期记忆配置
short_term_memory:
  max_rounds: 20          # 最大缓存轮数
  ttl_seconds: 3600       # 过期时间 (秒)
  persistence_enabled: true
  persistence_path: "./data/short_term_memory"

# 长期记忆 RAG 配置
rag:
  vector_dimension: 512   # BGE 模型输出维度
  vector_store_type: "faiss"
  base_path: "./data/rag"
  experience_rag_enabled: true
  memory_rag_enabled: true
  knowledge_rag_enabled: true

# Embedding 模型配置
embedding:
  model_name: "BAAI/bge-small-zh-v1.5"
  use_cpu: true           # CPU 优先，节省显存
  quantize: true          # 4bit 量化
  cache_dir: "./data/models"

# 记忆自进化配置
memory_evolution:
  consolidation_threshold: 0.7    # 巩固阈值
  forget_threshold: 0.1           # 遗忘阈值
  consolidation_interval_hours: 1.0
  check_interval_hours: 6.0

# 共享池快照配置
shared_pool:
  persistence_enabled: true
  snapshot_interval: 30           # 秒
  max_snapshots: 20               # 保留数量
  persistence_path: "./data/shared_memory_pool"

# 数据存储配置
storage:
  hot_storage:
    type: "mongodb"
    uri: "mongodb://localhost:27017"
    database: "zulong_logs"
    ttl_days: 14
  cold_storage:
    type: "minio"
    endpoint: "localhost:9000"
    access_key: "minioadmin"
    secret_key: "minioadmin"
    bucket: "zulong-cold-storage"
    migration_schedule: "0 2 * * *"  # 每天凌晨 2 点
```

---

.1 记忆系统配置.2 性能优化建议

#.1 记忆系统配置.2.1 批量向量化

**优化前** (单条处理):
```python
for text in texts:
    vector = embedding_model.encode_query(text)
    # 每次都要加载模型，慢
```

**优化后** (批量处理):
```python
vectors = embedding_model.encode(texts, batch_size=32)
# 一次加载，批量编码，快 3-5 倍
```

---

#.1 记忆系统配置.2.2 FAISS 索引优化

**小规模数据** (< 1000 条):
```python
index = faiss.IndexFlatL2(512)  # 精确搜索
```

**大规模数据** (> 1000 条):
```python
# IVF 索引 (近似搜索，10 倍加速)
nlist = 100  # 聚类中心数
quantizer = faiss.IndexFlatL2(512)
index = faiss.IndexIVFFlat(quantizer, 512, nlist)

# 训练索引 (需要样本)
index.train(sample_vectors)

# 搜索时指定 probe 数 (越高越精确)
index.nprobe = 10
```

---

# 第 13 章：工具系统架构

## 12.1 工具系统设计理念

### 12.1.1 设计目标

祖龙工具系统围绕以下目标设计：

1. **统一接口** - 所有工具（内置工具、技能包工具、插件工具）遵循相同的 `BaseTool` 接口
2. **动态注册** - 支持运行时注册/注销工具，技能包安装时自动注册
3. **并发调度** - 基于线程池的并发执行，支持批量并行调用
4. **防御性桥接** - 自动检测并处理 async/sync 混用（coroutine 泄漏）
5. **性能监控** - 每次调用记录耗时、成功率、调用历史

### 12.1.2 分层架构

```
┌──────────────────────────────────────────────────┐
│  InferenceEngine (Function Calling 循环)          │
│  tool_choice="auto" → 多轮迭代                   │
├──────────────────────────────────────────────────┤
│  CoreToolManager (热/冷工具管理)                  │
│  HOT: 始终在 prompt    COLD: 在 ToolRAG 中       │
├──────────────────────────────────────────────────┤
│  ToolEngine (统一调度器)                          │
│  call_tool() → validate → execute → record       │
├──────────────────────────────────────────────────┤
│  ToolRegistry (单例注册表)                        │
│  register() / unregister() / get() / list_tools()│
├──────────────────────────────────────────────────┤
│  BaseTool (抽象基类)                              │
│  initialize() / execute(ToolRequest) / cleanup() │
├──────────┬───────────┬────────────┬──────────────┤
│ OpenClaw │ OpenClaw  │ search_    │ 技能包工具   │
│ Adapter  │ Search    │ tools      │ (task_decomp │
│          │           │ (元工具)   │  deep_reason)│
└──────────┴───────────┴────────────┴──────────────┘
```

---

## 12.2 BaseTool 基类

**核心文件**: `zulong/tools/base.py`

### 12.2.1 核心数据结构

**工具请求 (ToolRequest)**:
```python
@dataclass
class ToolRequest:
    tool_name: str                    # 工具名称
    action: str                       # 动作
    parameters: Dict[str, Any]       # 参数
    timeout: float = 30.0            # 超时时间（秒）
    priority: int = 5                # 优先级（1-10，10 最高）
    request_id: str = ""             # 请求 ID（自动生成）
    callback: Optional[Callable] = None
```

**工具结果 (ToolResult)**:
```python
@dataclass
class ToolResult:
    success: bool                     # 是否成功
    data: Any = None                 # 返回数据
    error: Optional[str] = None      # 错误信息
    status_code: int = 0             # 状态码
    execution_time: float = 0.0      # 执行时间（秒）
    request_id: str = ""             # 请求 ID
```

### 12.2.2 BaseTool 接口

```python
class BaseTool(ABC):
    """所有工具必须实现的抽象基类"""
    
    def initialize(self) -> bool:       # 初始化
    def execute(self, request: ToolRequest) -> ToolResult:  # 执行
    def cleanup(self) -> None:          # 清理资源
    
    # 自动提供:
    def get_function_schema(self) -> Dict:   # OpenAI Function Calling schema
    def _get_parameters_schema(self) -> Dict: # 子类重写描述参数
    def get_statistics(self) -> Dict:        # 调用统计
```

**Function Calling Schema 自动生成**:

每个工具通过 `get_function_schema()` 方法返回标准的 OpenAI Function Calling 格式：
```json
{
    "type": "function",
    "function": {
        "name": "search_tools",
        "description": "在工具索引中搜索可用的工具...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "..."},
                "top_k": {"type": "integer", "description": "..."}
            },
            "required": ["query"]
        }
    }
}
```

### 12.2.3 工具分类

| 分类 | 枚举值 | 说明 |
|------|--------|------|
| SYSTEM | "system" | 系统工具（文件、进程等） |
| NETWORK | "network" | 网络工具（HTTP、API 等） |
| CODE | "code" | 代码工具（VSCode、执行等） |
| ROBOT | "robot" | 机器人工具（运动、感知等） |
| CUSTOM | "custom" | 自定义工具（技能包提供） |

---

## 12.3 ToolRegistry 注册表

**核心文件**: `zulong/tools/base.py`

全局单例模式，管理所有已注册工具的生命周期。

**核心功能**:

| 方法 | 说明 |
|------|------|
| `register(tool)` | 注册工具（同名拒绝重复注册） |
| `unregister(tool_name)` | 注销工具 |
| `get(tool_name)` | 按名称获取工具实例 |
| `get_by_category(category)` | 按分类查询 |
| `get_all_function_schemas()` | 聚合所有工具的 Function Calling Schema |
| `initialize_all()` | 批量初始化 |
| `cleanup_all()` | 批量清理 |

---

## 12.4 ToolEngine 调度引擎

**核心文件**: `zulong/tools/tool_engine.py`

### 12.4.1 内置工具注册

系统启动时，`ToolEngine.__init__()` 自动注册以下内置工具：

| 工具名 | 类名 | 说明 |
|--------|------|------|
| `openclaw` | `OpenClawToolAdapter` | OpenClaw 通用工具适配器 |
| `openclaw_search` | `OpenClawSearchTool` | OpenClaw 搜索工具 |
| `openclaw_plugin` | `OpenClawPluginAdapter` | OpenClaw 插件适配器 |
| `search_tools` | `SearchToolsTool` | ToolRAG 元工具（发现冷工具） |

### 12.4.2 调用流程

```
call_tool(tool_name, action, parameters)
  ↓
创建 ToolCall 记录 (call_id, start_time)
  ↓
ToolRegistry.get(tool_name) → 查找工具
  ├─ 未找到 → 返回 ToolResult(success=False)
  └─ 找到 → 检查 tool.enabled
  ↓
构建 ToolRequest(tool_name, action, parameters, timeout, priority)
  ↓
tool.execute(request) → ToolResult
  ↓
防御性检测：inspect.iscoroutine(result)?
  ├─ 是 → asyncio.run(result) 或 ThreadPoolExecutor 桥接
  └─ 否 → 直接使用
  ↓
更新统计 (total_calls, successful_calls, failed_calls)
  ↓
记录调用历史 (call_history, 最多保留 1000 条)
  ↓
返回 ToolResult
```

### 12.4.3 async/sync 桥接机制

ToolEngine 的 `call_tool()` 是同步方法，但工具可能返回 coroutine（async def 未被 await）。
系统通过以下防御性逻辑自动处理：

```python
result = tool.execute(request)

if inspect.iscoroutine(result):
    try:
        result = asyncio.run(result)        # 无事件循环时直接运行
    except RuntimeError:
        # 已有事件循环 → 新线程中运行
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(asyncio.run, result).result(timeout=30)
```

---

# 第 14 章：热/冷工具管理与 ToolRAG

## 13.1 热/冷工具设计理念

### 13.1.1 问题背景

随着技能包不断增加，工具总数可能超出 LLM 上下文窗口容量。
如果将所有工具 schema 都注入 prompt，会导致：
- 上下文窗口被工具描述占满
- LLM 选择准确率下降（工具过多时"选择困难"）
- token 成本浪费

### 13.1.2 解决方案：热/冷分层

```
┌────────────────────────────────┐
│  Prompt 上下文窗口             │
│  ┌──────────────────────────┐ │
│  │ 热工具 (HOT, ≤8 个)      │ │   ← 始终可见
│  │ openclaw_search           │ │
│  │ read_memory_detail        │ │
│  │ search_tools (元工具)     │ │   ← 发现冷工具的入口
│  └──────────────────────────┘ │
├────────────────────────────────┤
│  ToolRAG 向量索引              │
│  ┌──────────────────────────┐ │
│  │ 冷工具 (COLD, 无限)      │ │   ← 按需语义检索
│  │ task_decompose            │ │
│  │ deep_reasoning            │ │
│  │ ...更多技能包工具...      │ │
│  └──────────────────────────┘ │
└────────────────────────────────┘
```

**核心规则**:
- 热工具 ≤ 8 个，始终注入 LLM prompt
- 冷工具存入 ToolRAG（FAISS 向量索引），通过 `search_tools` 元工具按需检索
- `search_tools` 自身永远是热工具（它是发现冷工具的唯一入口）

---

## 13.2 CoreToolManager

**核心文件**: `zulong/tools/core_tool_manager.py`

### 13.2.1 默认热工具白名单

```python
DEFAULT_HOT_TOOLS: Set[str] = {
    "openclaw_search",     # 核心搜索
    "read_memory_detail",  # 记忆检索
    "search_tools",        # 元工具（发现冷工具的入口）
}

MAX_HOT_TOOLS = 8          # 热工具数量上限
```

### 13.2.2 自动分类策略

工具注册时，`CoreToolManager.register_tool()` 按以下优先级自动分类：

```
register_tool(tool_name, schema, source, force_hot)
  ↓
force_hot == True?  → HOT
  ↓
tool_name 在 DEFAULT_HOT_TOOLS 中?  → HOT
  ↓
source == "builtin" 且热工具未满 (< MAX_HOT_TOOLS)?  → HOT
  ↓
其余 → COLD (写入 ToolRAG)
```

**分类结果**:
- 内置工具（如 `openclaw_search`） → 默认热工具
- 技能包工具（如 `task_decompose`, `deep_reasoning`, `plan_and_reason`） → 默认冷工具

### 13.2.3 临时提升机制

在 InferenceEngine 的多轮推理中，如果 LLM 调用了 `search_tools` 并发现了冷工具，
这些工具会被 **临时提升** 为热工具（当前推理周期内有效）：

```
LLM 调用 search_tools(query="拆解任务")
  ↓
ToolRAG.search_tools() → 发现 task_decompose (similarity=0.92)
  ↓
CoreToolManager.promote_tools(["task_decompose"])
  ↓
_temp_promoted.add("task_decompose")
  ↓
get_active_schemas() → 返回 hot_tools ∪ temp_promoted 的 schema
  ↓
task_decompose 的 schema 被注入到下一轮迭代的 tools 列表
  ↓
推理结束 → clear_temp_promoted() → 清空临时提升
```

### 13.2.4 核心接口

| 方法 | 说明 |
|------|------|
| `register_tool(name, schema, source)` | 注册并自动分类 |
| `unregister_tool(name)` | 注销工具（同时从 ToolRAG 移除） |
| `get_active_schemas()` | 获取当前应注入 prompt 的 schema 列表 |
| `promote_tools(names)` | 临时提升冷工具（本轮推理有效） |
| `clear_temp_promoted()` | 清空临时提升（推理结束时调用） |
| `is_hot(name)` | 检查工具是否为热工具 |
| `get_stats()` | 获取热/冷工具统计 |

---

## 13.3 ToolRAG 工具索引

**核心文件**: `zulong/memory/tool_rag.py`

### 13.3.1 架构位置

ToolRAG 继承 `BaseRAGLibrary`，与 ExperienceRAG / MemoryRAG / KnowledgeRAG 并列，
由 RAGManager 统一管理。

```
RAGManager
├── ExperienceRAG (经验库)
├── MemoryRAG (记忆库)
├── KnowledgeRAG (知识库)
└── ToolRAG (工具库)
```

### 13.3.2 文档格式

每条工具摘要的向量化文本：
```
工具名: task_decompose
描述: 将复杂目标拆解为可执行的子任务列表...
参数: goal, context
```

metadata 包含完整的 Function Calling schema，供动态注入时直接使用。

### 13.3.3 检索流程

```
用户意图："我需要拆解这个复杂任务"
  ↓
search_tools 元工具收到 query
  ↓
ToolRAG.search_tools(query, top_k=3)
  ↓
EmbeddingModelManager.encode_query(query) → 512 维向量
  ↓
FAISS.search(query_vector, top_k=3)
  ↓
L2 距离 → 相似度转换: similarity = 1 / (1 + distance)
  ↓
返回结构化结果:
  [{tool_name, description, similarity, function_schema}, ...]
  ↓
_discovered_schemas 暂存到 InferenceEngine
  ↓
下一轮迭代: 动态注入到 tools 列表
```

### 13.3.4 核心接口

| 方法 | 说明 |
|------|------|
| `add_tool(name, description, params, source, schema)` | 添加工具摘要 |
| `remove_tool(name)` | 移除工具摘要 |
| `search_tools(query, top_k)` | 语义检索工具 |
| `get_tool_schema(name)` | 获取完整 Function Calling schema |
| `has_tool(name)` | 检查工具是否已索引 |
| `list_all_tools()` | 列出所有已索引工具名 |

---

## 13.4 search_tools 元工具

**核心文件**: `zulong/tools/search_tools.py`

`search_tools` 是一个 **"工具的工具"（元工具）**：

- 它本身是一个热工具（始终在 prompt 中）
- LLM 发现当前工具不够用时，调用它按语义检索更多工具
- 返回的工具 schema 被 InferenceEngine 动态注入到后续迭代

**参数**:
```json
{
    "query": "描述你需要什么类型的工具（如'拆解复杂任务'）",
    "top_k": 3
}
```

**返回**:
```json
{
    "message": "找到 2 个相关工具",
    "tools_found": 2,
    "tools": [
        {"tool_name": "task_decompose", "description": "...", "similarity": 0.92},
        {"tool_name": "deep_reasoning", "description": "...", "similarity": 0.78}
    ],
    "_discovered_schemas": [...]
}
```

其中 `_discovered_schemas` 是内部字段，被 InferenceEngine 提取后从返回结果中移除，
不会暴露给 LLM。

---

# 第 15 章：Function Calling 多轮推理

## 14.1 推理循环架构

**核心文件**: `zulong/l2/inference_engine.py`

### 14.1.1 设计目标

LLM 通过 OpenAI 兼容的 Function Calling 协议自主决定是否调用工具：
- 不需要预先判断 "是否需要搜索" — 完全由模型自主决定
- 支持多轮迭代 — 一次推理中可以连续调用多个工具
- 支持动态工具发现 — search_tools 的结果在下一轮迭代中可用

### 14.1.2 推理循环流程

```
用户输入 → 构建 messages + system_prompt
  ↓
获取 tools = CoreToolManager.get_active_schemas()  (热工具 + 临时提升)
  ↓
┌─ for iteration in range(max_tool_iterations=10): ─────────────────┐
│                                                                    │
│  检查 _pending_tool_schemas → 动态注入新发现的工具                 │
│                                                                    │
│  vllm_client.chat.completions.create(                             │
│      model=model_id,                                              │
│      messages=messages,                                           │
│      tools=tools,                                                 │
│      tool_choice="auto",    ← LLM 自主决定                       │
│      max_tokens=1024,                                             │
│      temperature=0.3,                                             │
│      top_p=0.85,                                                  │
│      timeout=300s           ← 超时保护 (v2.8 可配置)            ← 超时保护                            │
│  )                                                                │
│                                                                    │
│  response.choices[0].message                                      │
│  ├─ 有 tool_calls → 逐个执行 → 结果追加到 messages → 继续循环    │
│  └─ 无 tool_calls → 提取 content → 跳出循环                      │
│                                                                    │
│  死循环检测：记录最近 3 次调用，重复则强制跳出                     │
└───────────────────────────────────────────────────────────────────┘
  ↓
返回最终 content + 累积的 links
```

### 14.1.3 工具调用分发

InferenceEngine 在收到 LLM 的 `tool_calls` 后，按工具名分发：

| 工具名 | 处理方法 | 说明 |
|--------|---------|------|
| `search_tools` | `_handle_search_tools_meta()` | 元工具，结果包含 _discovered_schemas |
| `read_memory_detail` | `_handle_read_memory()` | 记忆检索 |
| 其他 | `ToolEngine.call_tool()` | 通用工具调用 |

### 14.1.4 动态工具注入机制

```
迭代 1: LLM 调用 search_tools(query="拆解任务")
  ↓
_handle_search_tools_meta() 执行
  ↓
ToolRAG 返回 task_decompose 的 schema
  ↓
schema 暂存到 self._pending_tool_schemas
  ↓
tool_message 返回给 LLM（告知"已发现 task_decompose 工具"）
  ↓
迭代 2 开始：
  检查 _pending_tool_schemas → 发现新 schema
  ↓
  追加到 tools 列表: tools.append(task_decompose_schema)
  ↓
  LLM 现在可以调用 task_decompose 工具
```

### 14.1.5 容错与降级

| 场景 | 处理方式 |
|------|---------|
| CORE 模型超时 (>300s，v2.8 可配置) | 自动切换到 BACKUP 模型 |
| Function Calling 不可用 | 降级到普通模式（不传 tools） |
| 死循环检测 | 最近 3 次调用相同 → 强制跳出 |
| 最大迭代限制 | 10 次迭代后强制停止 |

---

# 第 16 章：技能包系统

## 15.1 技能包设计理念

### 15.1.1 "借用→学习→内化→丢弃" 生命周期

技能包系统模拟人类学习新技能的过程：

```
AVAILABLE (可安装)
  ↓ install()
INSTALLED (已安装，工具已注册)
  ↓ execute() × N
LEARNING (学习中，积累经验)
  ↓ check_internalization() > 0.9
INTERNALIZED (已内化，经验充足)
  ↓ uninstall()
UNINSTALLED (已卸载，经验保留)
```

**核心思想**：技能包是临时的"拐杖"，安装后通过反复执行积累经验，
当经验充足（内化完成度 > 90%）时可以卸载，经验永久保留在 ExperienceStore 中。

### 15.1.2 架构总览

```
config/skill_packs.yaml
  ↓
SkillPackLoader.load_from_config()
  ↓ 动态导入 + 依赖检查
SkillPackRuntime.install_pack()
  ↓ 注册工具到 ToolEngine + CoreToolManager
ISkillPack 实例
  ├── get_manifest()    → SkillPackManifest
  ├── install()         → 注册工具
  ├── execute()         → 执行能力 + 记录经验
  ├── get_tools()       → List[BaseTool]
  └── uninstall()       → 注销工具 + 释放资源
```

---

## 15.2 ISkillPack 接口

**核心文件**: `zulong/skill_packs/interface.py`

### 15.2.1 SkillPackManifest 清单

```python
@dataclass
class SkillPackManifest:
    pack_id: str                    # 唯一标识，如 "complex_task"
    name: str                       # 显示名称，如 "任务拆解"
    version: str = "1.0.0"
    description: str = ""
    capabilities: List[str] = []    # 能力列表，如 ["task_decompose"]
    dependencies: List[str] = []    # Python 包依赖
    resource_requirements: Dict     # {"cpu_mb": 512, "gpu_mb": 0}
    learning_objectives: List[str]  # 学习目标（内化评估用）
    source: str = "custom"          # 来源: "custom" / "cline"
```

### 15.2.2 接口方法

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `get_manifest()` | `SkillPackManifest` | 返回清单（在 install 前调用） |
| `install(tool_registry, config)` | `bool` | 安装：注册工具 |
| `execute(capability, params)` | `Dict` | 执行某个能力 |
| `get_tools()` | `List[BaseTool]` | 返回提供的工具列表 |
| `uninstall()` | `bool` | 卸载：注销工具、释放资源 |

---

## 15.3 SkillPackRuntime 运行时

**核心文件**: `zulong/skill_packs/runtime.py`

### 15.3.1 初始化依赖

```python
SkillPackRuntime(
    tool_engine=ToolEngine,           # 用于工具注册/注销
    experience_store=ExperienceStore,  # 用于经验记录
    hot_update_engine=HotUpdateEngine, # 用于经验内化
    core_tool_manager=CoreToolManager, # 用于热/冷分类
)
```

### 15.3.2 安装流程

```
install_pack(pack, config)
  ↓
1. 验证状态：是否已安装？
  ↓
2. 依赖检查：manifest.dependencies 全部可 import?
  ↓
3. 注入 LLM 客户端：config['llm_client'] = vllm_client
  ↓
4. 调用 pack.install(ToolRegistry(), config)
  ↓
5. 注册工具到 ToolEngine:
   for tool in pack.get_tools():
       tool_engine.registry.register(tool)
       core_tool_manager.register_tool(tool.name, schema, source=pack_id)
  ↓
6. 更新状态: INSTALLED, 记录安装时间
```

### 15.3.3 执行与经验记录

```
execute_capability(pack_id, capability, params)
  ↓
1. 调用 pack.execute(capability, params) → result
  ↓
2. 更新状态: INSTALLED → LEARNING
  ↓
3. 记录经验到 ExperienceStore:
   content = "技能包 {pack_id} 执行 {capability}: 成功/失败"
   experience_store.add_experience(...)
  ↓
4. 通知 HotUpdateEngine（可选）
```

### 15.3.4 内化评估

**评估公式**:
```
internalization_score = quantity_score × 0.6 + quality_score × 0.4

其中:
- quantity_score = min(experience_count / 50, 1.0)
- quality_score = success_count / total_count
```

当 `internalization_score > 0.9` 时，自动更新状态为 `INTERNALIZED`。

---

## 15.4 SkillPackLoader 加载器

**核心文件**: `zulong/skill_packs/loader.py`

### 15.4.1 配置文件格式

**配置文件**: `config/skill_packs.yaml`

```yaml
skill_packs:
  - pack_id: "complex_task"
    enabled: true
    path: "zulong.skill_packs.packs.complex_task"
    config:
      max_subtasks: 10
      kv_cache_ttl: 1800
      max_slots_per_session: 8
      # llm_client 和 model_id 由系统自动注入 L2 CORE

  - pack_id: "cline_coder"
    enabled: false
    path: "zulong.skill_packs.packs.cline_coder"

internalization:
  min_experience_count: 50
  min_success_rate: 0.9
  evaluation_interval_hours: 24
```

### 15.4.2 加载流程

```
load_from_config(config_path)
  ↓
解析 YAML → 遍历 skill_packs 列表
  ↓
for each pack_config:
  ├─ enabled == false → 跳过
  ├─ 无 path → 跳过
  └─ _load_pack(pack_id, pack_path, config):
       ↓
     importlib.import_module(pack_path)      # 动态导入
       ↓
     _find_pack_class(module, pack_id)       # 查找 ISkillPack 子类
       ├─ 策略 1: 遍历模块属性，找 ISkillPack 子类
       └─ 策略 2: 按命名约定 (ComplexTask for complex_task)
       ↓
     pack_instance = pack_class()             # 实例化
       ↓
     _check_dependencies(manifest)            # 验证依赖
       ↓
     runtime.install_pack(pack_instance)      # 安装到运行时
```

---

## 15.5 内置技能包

### 15.5.1 complex_task（复杂任务处理）

> **架构变更说明**：v2.4.1 将原 `task_planner`（任务拆解）和 `deep_reasoner`（深度推理）
> 合并为统一的 `ComplexTaskPack`，共享 LLM client 和 KV Cache 策略。

**目录**: `zulong/skill_packs/packs/complex_task/`

**清单**:
```python
SkillPackManifest(
    pack_id="complex_task",
    name="复杂任务处理",
    capabilities=[
        "task_decompose",      # 任务拆解
        "priority_rank",       # 优先级排序
        "dependency_analyze",  # 依赖分析
        "deep_reasoning",      # LLM 深度推理
        "plan_and_reason",     # 联合能力：拆解→推理一体化
    ],
    source="custom"
)
```

**目录结构**:
```
complex_task/
├── __init__.py           # ComplexTaskPack 入口（ISkillPack 实现）
├── planner.py            # TaskDecomposeAlgorithm（任务拆解核心算法）
├── planning_tools.py     # TaskDecomposeTool / PriorityRankTool / DependencyAnalyzeTool
├── reasoning_engine.py   # DeepReasoningEngine（LLM 增强版四步推理链）
├── kv_cache_slot.py      # KVCacheSlotManager（Context Slot 热交换）
└── tools.py              # DeepReasoningTool / PlanAndReasonTool
```

**核心模块 1 — 任务拆解** (`TaskDecomposeAlgorithm`):

1. **子任务生成**:
   - 优先使用 LLM 生成（`_generate_subtasks_with_llm`）
   - LLM 不可用时降级为规则引擎（`_generate_subtasks_with_prompt`）

2. **依赖分析**: 基于任务类型判断依赖关系
   - 收集类 (search/gather) → 无前置依赖
   - 分析类 (analyze/calculate) → 依赖收集类任务
   - 输出类 (generate/write) → 依赖所有前置任务

3. **并行组计算**: 拓扑排序，将不互相依赖的任务分入同一并行组

**核心模块 2 — LLM 增强版深度推理** (`DeepReasoningEngine`):

```
步骤 1: 问题分析 (analyze)    ← LLM 驱动：提取关键要素 + 约束 + 复杂度评估
  ↓
步骤 2: 假设生成 (hypothesize) ← LLM 驱动：生成多方案（含方法/预期效果/风险/置信度）
  ↓
步骤 3: 假设验证 (verify)      ← LLM 驱动：逻辑完整性/约束满足/可实施性/风险评估
  ↓
步骤 4: 方案选择 (conclude)    ← LLM 驱动：选最优 + 实施步骤 + 风险 + 备选方案
```

与原 `deep_reasoner` 的关键差异：

| 维度 | 原 deep_reasoner | 新 DeepReasoningEngine |
|------|-----------------|----------------------|
| 推理驱动 | 硬编码模板（5个固定假设） | LLM 驱动（基于问题动态生成） |
| LLM 集成 | 无 | 复用 L2 CORE client |
| 降级策略 | 无（纯模板） | LLM 失败→自动降级到规则模式 |
| 可行性评估 | 公式计算（忽略问题内容） | LLM 逐项验证 |

**核心模块 3 — KV Cache Context Slot** (`KVCacheSlotManager`):

```
宏观规划 → save_slot("macro") → 保存 KV Cache
  ↓
具体分析 → save_slot("detail") → 独立 KV Cache
  ↓
恢复规划 → restore_slot("macro") → 注入分析结果，继续规划
```

- 每 session 支持多命名 slot（默认最多 8 个）
- TTL 自动过期（默认 30 分钟）
- LRU 淘汰策略（超出上限时淘汰最旧 slot）

**核心模块 4 — 联合能力** (`plan_and_reason`):

```
用户目标 → TaskDecomposeAlgorithm.decompose()
  ↓
子任务列表 → 筛选分析类子任务
  ↓
对每个分析类子任务 → DeepReasoningEngine.deep_reason()
  ↓
输出：带推理支撑的完整执行方案
```

**提供的工具**（共 5 个）:

| 工具 | 来源 | 用途 |
|------|------|------|
| `task_decompose` | TaskDecomposeAlgorithm | 将复杂目标拆解为子任务列表 |
| `priority_rank` | TaskDecomposeAlgorithm | 子任务优先级排序 |
| `dependency_analyze` | TaskDecomposeAlgorithm | 子任务依赖分析 + 并行组计算 |
| `deep_reasoning` | DeepReasoningEngine | LLM 四步深度推理 |
| `plan_and_reason` | 联合 | 先拆解再逐子任务推理的一体化流程 |

---

# 第 17 章：五阶段流水线与无限深度任务图谱

## 16.1 设计理念

### 16.1.1 三大核心原则

五阶段流水线是祖龙系统的**框架骨架**而非可选技能包，集成在 `zulong/pipeline/` 目录中。
设计遵循三大核心原则：

1. **"导演模式"** — 框架只编排流程，不约束 LLM 输出格式。任何能接受自然语言并输出自然语言的模型均可直接接入。
2. **"规划表即结构"** — LLM 生成自然语言规划表（Markdown表格），框架在多轮调用间原样传递，不做结构化改写。
3. **三层引导策略** — 通用引导模板（极简无格式要求）→ 模型适配提示（可选优化）→ 宽松解析（正则+多策略降级）。

### 16.1.2 架构定位

```
用户输入
  ↓
ComplexityClassifier (零成本关键词分类)
  ├─ is_complex=False → 标准推理 (Function Calling 循环)
  └─ is_complex=True  → PipelineOrchestrator 五阶段流水线
                           ↓
                       TaskGraph (无限深度递归树)
                           ↓
                       Stage1 → Stage2 → Stage3 → Stage4 → Stage5
                                                    ↑         |
                                                    └─ FAIL ──┘ (最多2次重试)
```

### 16.1.3 与旧 ComplexTaskPack 的关系

| 维度 | 旧 ComplexTaskPack (v2.4.1) | 新 Pipeline (v2.5) |
|------|---------------------------|-------------------|
| 架构位置 | 技能包（可装卸） | 框架骨架（`zulong/pipeline/`） |
| 任务结构 | 扁平子任务列表 | 无限深度递归树 |
| 执行模型 | 全部 type="task" 节点执行 | 仅叶子节点执行 |
| 审查机制 | 无 | ReviewGate + 重试循环 |
| 进度表达 | 线性列表 | 递归 Markdown 表格（按大纲分组） |
| 前端可视化 | 无 | TaskGraph 实时图谱推送 |

---

## 16.2 TaskGraph 数据结构

**核心文件**: `zulong/pipeline/task_graph.py`

### 16.2.1 无限深度递归树

TaskGraph 采用**模板 + 动态生成**的双层架构：

```
固定模板节点 (每次任务自动创建):
  req (原始需求, depth=0, type=requirement)
   └── analysis (需求分析, depth=1, type=analysis)

动态生成节点 (由 LLM 按任务需要生成, 无深度上限):
        ├── o1 (大纲1, depth=2, type=outline)
        │    ├── o1_1 (任务1, depth=3, type=task)
        │    │    ├── o1_1_1 (子任务1, depth=4, type=subtask)
        │    │    └── o1_1_2 (子任务2, depth=4, type=subtask)
        │    └── o1_2 (任务2, depth=3, type=task) ← 叶子
        └── o2 (大纲2, depth=2, type=outline)
             └── o2_1 (任务3, depth=3, type=task)
                  └── o2_1_1 (子任务3, depth=4, type=subtask)
                       └── o2_1_1_1 (子子任务, depth=5, type=subtask) ← 叶子
```

**ID 命名规范**: 深度2用 `o{i}`，更深层用 `{parent_id}_{i}` 递归嵌套。

### 16.2.2 深度-类型自动映射

```python
@staticmethod
def depth_to_type(depth: int) -> str:
    mapping = {0: "requirement", 1: "analysis", 2: "outline", 3: "task"}
    return mapping.get(depth, "subtask")  # depth >= 4 统一为 subtask
```

| 深度 | 类型 | 说明 | 前端颜色 |
|------|------|------|---------|
| 0 | requirement | 原始需求（模板） | #8b5cf6 紫色 |
| 1 | analysis | 需求分析（模板） | #6366f1 靛蓝 |
| 2 | outline | 大纲（LLM 生成） | #3b82f6 蓝色 |
| 3 | task | 任务（LLM 生成） | #06b6d4 青色 |
| 4+ | subtask | 子任务（LLM 生成，无限） | #10b981 绿色 |

### 16.2.3 叶子节点执行模型

**核心规则**: 只有**叶子节点**（没有子节点的非模板节点）被实际执行。

```python
def get_leaf_nodes(self) -> List[TaskNode]:
    parent_ids = {s for s, t in self._h_edges}
    template_ids = {"req", "analysis"}
    return [
        n for n in self._nodes.values()
        if n.id not in parent_ids and n.id not in template_ids
    ]
```

- 容器节点（有子节点）仅作为结构分组，不直接执行
- 模板节点（req, analysis）由框架自动管理，不参与执行
- 执行粒度由 LLM 拆解深度决定：拆得越细，叶子节点越小、执行越精确

### 16.2.4 非叶子节点状态聚合

非叶子节点的 `status` 不存储在数据中，而是在导出时从子节点递归聚合：

```
聚合规则（优先级从高到低）：
  所有子节点 completed  → "completed"
  任一子节点 in_progress 或混合 → "in_progress"
  任一子节点 blocked    → "blocked"
  任一子节点 needs_adjust → "needs_adjust"
  全部 pending          → "pending"
```

### 16.2.5 规划表输出

`to_planning_table()` 生成递归 Markdown 表格，注入到子任务 prompt 中，实现"全局注意力"：

```markdown
## 当前任务规划

### 前端开发
| 编号 | 子任务 | 状态 | 结果摘要 |
|------|--------|------|----------|
| | **组件开发** | | |
| o1_1_1 |   按钮组件 | 完成 | 按钮组件已完成... |
| o1_1_2 |   表单组件 | 进行中 | |
| o1_2 | 样式调整 | 待开始 | |

### 后端开发
| 编号 | 子任务 | 状态 | 结果摘要 |
|------|--------|------|----------|
| | **API开发** | | |
| |   **数据库设计** | | |
| o2_1_1_1 |     用户表设计 | 待开始 | |
```

非叶子节点以 **粗体分组标题** 显示，叶子节点以表格行显示。

### 16.2.6 序列化与反序列化

TaskGraph 支持完整序列化（用于任务挂起/恢复），所有节点、层级边、依赖边、并行组均可持久化为 JSON 并还原。

---

## 16.3 复杂度分类器

**核心文件**: `zulong/pipeline/complexity_classifier.py`

### 16.3.1 零成本设计

分类器基于**关键词 + 多信号加权打分**，不调用 LLM，延迟为零。

```
classify(text) → ClassifyResult(is_complex, score, signals)
```

### 16.3.2 五维信号

| 信号 | 权重 | 检测方式 |
|------|------|---------|
| 文本长度 | 0.15 | `len(text) > 200` 字符 |
| 多步骤关键词 | 0.30 | "然后"、"接着"、"第一步"等 |
| 连接词密度 | 0.15 | "并且"、"同时"、"但是"的密度 |
| 复合意图 | 0.25 | 多动词检测（"分析...并...设计"）|
| 显式标记 | 0.15 | "复杂"、"详细"、"完整方案"等 |

**阈值**: `score >= 0.4` → `is_complex = True` → 进入流水线

### 16.3.3 简单对话快速跳过

匹配 `SIMPLE_CHAT_PATTERNS`（如"你好"、"谢谢"）时直接返回 `is_complex=False`，避免不必要的信号计算。

---

## 16.4 流水线编排器

**核心文件**: `zulong/pipeline/orchestrator.py`

### 16.4.1 编排流程

```python
class PipelineOrchestrator:
    async def run(self, goal: str, request_id: str = "") -> Dict:
        # 1. 初始化 TaskGraph（自动创建 req + analysis 模板节点）
        # 2. Stage1: Decompose（两相位 + 递归解析）
        # 3. Stage2: Plan（Kahn 拓扑排序）
        # 4. Stage3: Execute（按并行组逐组执行叶子节点）
        # 5. Stage4+3 ReviewGate 循环（最多2次重试）
        # 6. Stage5: Finalize（汇总输出）
```

### 16.4.2 ReviewGate 重试机制

```
Stage3 → Stage4 → PASS? ─Yes→ Stage5
                    |
                    No (标记 needs_adjust 节点)
                    ↓
                  retry_count < MAX_REVIEW_RETRIES (2)?
                    ├── Yes → 重新执行 Stage3 → Stage4
                    └── No  → 强制进入 Stage5
```

### 16.4.3 事件回调

编排器通过 `on_event` 回调向前端推送进度事件：

| 事件类型 | 时机 | 数据 |
|---------|------|------|
| `pipeline_start` | 流水线开始 | `{goal, graph_id}` |
| `stage_start` | 阶段开始 | `{stage, stage_name}` |
| `stage_done` | 阶段完成 | `{stage, success, data}` |
| `graph_update` | 图谱变化 | `{graph: to_frontend_dict()}` |
| `node_update` | 节点状态变化 | `{node_id, status, result}` |
| `pipeline_done` | 流水线完成 | `{success, final_response}` |

---

## 16.5 五阶段详解

### 16.5.1 Stage1: 任务拆解 (Decompose)

**核心文件**: `zulong/pipeline/stages/decompose.py`

**两相位执行**:

```
Phase A: 需求分析
  ├── 调用 LLM 分析需求（核心目标/约束/产出物）
  ├── 结果填充 analysis 模板节点
  └── 失败时降级为原始需求文本

Phase B: 多级任务拆解
  ├── 调用 LLM 进行层次化拆解
  ├── 六级递归解析 LLM 输出
  ├── _write_tree_to_graph() 递归写入 TaskGraph
  └── _parse_dependencies() 解析依赖关系
```

**六级递归解析策略**（降级链）:

| 策略 | 格式 | 示例 |
|------|------|------|
| A | 嵌套 JSON | `[{"name":"...", "children":[...]}]` |
| B | Markdown 标题层级 | `## 大纲` → `### 任务` → `#### 子任务` |
| C | 多级编号列表 | `1.` → `1.1` → `1.1.1` |
| D | 段落分节 | `大纲1: ...` 段落块 |
| E | 扁平降级 | 编号/无序列表 → 包裹为单 outline |
| F | 终极兜底 | 整段文本作为单节点 |

**默认依赖策略**: 同父节点下的叶子顺序依赖（`o1_1_1→o1_1_2`），跨父节点无依赖（`o1_*` 与 `o2_*` 可并行）。

### 16.5.2 Stage2: 依赖规划 (Plan)

**核心文件**: `zulong/pipeline/stages/plan.py`

纯算法阶段，不调用 LLM：

```
提取叶子节点 → 构建依赖图 → Kahn 拓扑排序 → 计算并行组
```

输出: `parallel_groups = [["o1_1_1"], ["o1_1_2", "o2_1_1_1"], ["o1_2"]]`
每组内的任务可并行执行。含环检测（降级为全顺序执行）。

### 16.5.3 Stage3: 任务执行 (Execute)

**核心文件**: `zulong/pipeline/stages/execute.py`

按并行组逐组执行叶子节点：

```
for group in parallel_groups:
    并行执行 group 中的所有叶子
    ↓
    对每个叶子:
      1. 构建 prompt (含 planning_table 全局上下文)
      2. 调用 LLM 生成结果
      3. 更新节点状态 → completed / blocked
```

**局部 + 全局注意力**: 每个叶子的 prompt 包含：
- **局部**: 该叶子自身的 label、desc、依赖的前置结果
- **全局**: `to_planning_table()` 输出的完整规划表

### 16.5.4 Stage4: 审查纠错 (ReviewGate)

**核心文件**: `zulong/pipeline/stages/review.py`

调用 LLM 审查所有叶子节点结果的完整性、一致性、质量：

```
收集已完成叶子的结果 → 构建审查 prompt → LLM 判定
  ├── PASS → 进入 Stage5
  └── FAIL → 标记 needs_adjust 节点 → 触发重试
```

宽松解析审查结论:
- PASS 信号: "全部通过"、"没有问题"、"all pass" 等
- FAIL 信号: "需要修改"、"不一致"、"遗漏" 等
- 同时出现 PASS+FAIL: 尝试定位具体失败节点，无法定位则默认 PASS

### 16.5.5 Stage5: 汇总输出 (Finalize)

**核心文件**: `zulong/pipeline/stages/finalize.py`

汇总所有叶子节点结果，调用 LLM 生成面向用户的最终自然语言回答：

```
收集叶子结果 → 构建汇总 prompt → LLM 生成最终回答
  └── 失败时降级: 直接拼接叶子结果
```

---

## 16.6 Prompt 引导体系

**核心文件**: `zulong/pipeline/prompts.py`

### 16.6.1 三层引导策略实现

| 层级 | 实现 | 作用 |
|------|------|------|
| 通用引导模板 | `ANALYSIS_PROMPT`, `DECOMPOSE_PROMPT` 等常量 | 极简描述目标，不约束格式 |
| 模型适配提示 | `build_*_prompt(goal, model_id)` 中的 hints | 可选，按 model_id 注入偏好 |
| 宽松解析 | DecomposeStage 六级降级解析 | 正则 + 多策略兜底 |

### 16.6.2 关键 Prompt

- `ANALYSIS_PROMPT`: 需求分析（核心目标 / 约束 / 产出物）
- `DECOMPOSE_PROMPT`: 多级拆解引导（含 `{analysis}` 占位符，鼓励多层级但不强制格式）
- `build_review_prompt()`: 审查引导（检查完整性和一致性）
- `build_finalize_prompt()`: 汇总引导（整合为连贯回答）

---

## 16.7 与 InferenceEngine 的集成

**核心文件**: `zulong/l2/inference_engine.py`

### 16.7.1 复杂度路由

在 `_on_l2_command()` 中，标准推理之前先进行复杂度判断：

```python
if text and not is_wakeup_command and not emergency and not visual_attention:
    classify_result = classify_complexity(text)
    if classify_result.is_complex:
        # 启动流水线（后台线程）
        threading.Thread(target=self._run_pipeline, args=(text, request_id)).start()
        return  # 不再走标准推理
```

### 16.7.2 降级保护

流水线执行失败时，自动降级到标准推理流程。

### 16.7.3 事件复用

流水线事件通过现有的 `L2_THINKING_STEP` WebSocket 事件推送，前端通过 `event_type` 前缀 `pipeline.*` 区分。

---

# 第 17 章：地址继承系统

## 17.1 设计理念

### 17.1.1 地址继承原理

祖龙记忆图谱是异构图（Heterogeneous Graph），所有记忆子系统统一投射为节点和边。为了让任意节点都能被精确定位并追溯其上下文来源，系统引入**地址继承系统**：

```
地址格式: {parent_path}/{child_id}

示例:
  dialogue:session_a1b2c3                          ← Session（地址根）
  dialogue:session_a1b2c3/dialogue:round_req123     ← Round（挂载到 Session）
  dialogue:session_a1b2c3/task:tg_xxx               ← 任务根节点（通过 REFERENCE 边绑定）
  dialogue:session_a1b2c3/task:tg_xxx/o1            ← 大纲节点（继承 Session 路径）
  dialogue:session_a1b2c3/task:tg_xxx/o1/o1_1       ← 子任务（递归继承）
```

**核心原则**:
- Session 是地址根节点，所有子节点的 `full_path` 均以 session 路径为前缀
- 任务节点通过 **REFERENCE 边** 与对话 round 建立绑定关系
- 地址传播是递归的：父节点地址更新后，自动传播到所有 HIERARCHY 下游子节点
- 子任务节点的地址携带完整父级路径，如文件系统路径

### 17.1.2 跨空间概念

记忆图谱是**一张统一的图**，不存在物理分离。架构图中"对话空间"和"任务空间"之间的虚线仅表示**概念性分区**：

```
┌─────────────────────────────────────────────────┐
│                    MemoryGraph                    │
│                                                   │
│  ┌─ 对话空间（概念分区）────────────────┐         │
│  │  Session → Round → SubDialogue       │         │
│  │  节点类型: DIALOGUE                  │         │
│  └────│ (REFERENCE 边) ────────────────┘         │
│       │                                          │
│  ┌────│ 任务空间（概念分区）────────────────┐     │
│  │    ▼                                    │     │
│  │  TaskRoot → Outline → Task → SubTask    │     │
│  │  节点类型: TASK                         │     │
│  └─────────────────────────────────────────┘     │
│                                                   │
│  实际是一张图，所有节点和边在同一个 NetworkX DiGraph 中 │
└─────────────────────────────────────────────────┘
```

---

## 17.2 核心机制

### 17.2.1 REFERENCE 边（跨类型引用）

**核心文件**: `zulong/memory/graph_adapters.py`

REFERENCE 边是连接对话空间和任务空间的桥梁。当 Round 被创建且关联了 TaskGraph 时，自动建立双向关联：

```python
# add_round() 中建立 REFERENCE 边
if task_graph_id:
    task_node_id = f"task:{task_graph_id}"
    if graph.has_node(task_node_id):
        graph.add_edge(
            round_id, task_node_id,
            EdgeType.REFERENCE, weight=0.8,
            metadata={"link_type": "dialogue_to_task", "inherited_at": time.time()},
        )
```

**边的权重**:
- `dialogue → task` 绑定: 0.8
- `sub_dialogue → task` 关联: 0.7
- `session → task` 绑定: 0.9

### 17.2.2 地址传播流程

```
用户输入 → ensure_session() / assign_session_by_similarity()
  ↓
Round 被分配到 Session（如 dialogue:session_a1b2c3）
  ↓
_link_round_to_session():
  1. 建立 HIERARCHY 边: session → round
  2. 更新 round 的 full_path = "dialogue:session_a1b2c3/dialogue:round_req123"
  ↓
_propagate_address_to_tasks():
  1. 通过 REFERENCE 边找到所有关联的 TASK 节点
  2. 将 TASK 节点的 full_path 更新为 "dialogue:session_a1b2c3/task:tg_xxx"
  ↓
_propagate_address_to_task_children():
  1. 通过 HIERARCHY 边找到 TASK 的所有子节点
  2. 递归更新每个子节点的 full_path
  3. 格式: parent_full_path/child_id
```

### 17.2.3 TaskGraphAdapter 同步中的地址继承

**核心文件**: `zulong/memory/graph_adapters.py` — `TaskGraphAdapter.sync()`

TaskGraph 同步到 MemoryGraph 时，子节点自动继承根节点的完整路径：

```python
# 查找任务根节点在 MemoryGraph 中的完整路径
root_node = graph.get_node(f"task:{graph_id}")
parent_prefix = root_node.metadata.get("full_path", f"task:{graph_id}")

# 投射子节点时使用完整路径地址
for node_id, task_node in source._nodes.items():
    if parent_prefix:
        full_node_id = f"{parent_prefix}/{node_id}"
    else:
        full_node_id = f"task:{node_id}"
```

### 17.2.4 task_create_plan 中的地址继承

**核心文件**: `zulong/tools/task_tools.py` — `TaskCreatePlanTool.execute()`

```python
# 找到最近 round，继承其 session 路径
parent_path = ""
for nid, node in mg._nodes.items():
    if (node.node_type == NodeType.DIALOGUE
            and node.metadata.get("sub_type") == "round"
            and (now - node.created_at) < 300):
        parent_path = node.metadata.get("full_path", nid)
        break

task_node_id = f"{parent_path}/task:{graph_id}" if parent_path else f"task:{graph_id}"
```

### 17.2.5 task_add_node 中的子节点地址继承

**核心文件**: `zulong/tools/task_tools.py` — `TaskAddNodeTool.execute()`

```python
# 同步到 MemoryGraph，继承父节点完整路径
parent_node = mg.get_node(f"{parent_prefix}/{parent_id}")
if parent_node:
    parent_path = parent_node.metadata.get("full_path", parent_id)
    full_node_id = f"{parent_path}/{node_id}"
else:
    full_node_id = f"task:{node_id}"
```

---

## 17.3 bind_session_to_task 方法

**核心文件**: `zulong/memory/graph_adapters.py` — `DialogueAdapter.bind_session_to_task()`

将 session 绑定到任务 ID，用于挂起任务回溯：

```python
def bind_session_to_task(self, graph, session_id, task_id):
    node = graph.get_node(session_id)
    if node:
        node.metadata["bound_task_id"] = task_id
        # 更新任务节点地址
        task_root_id = f"task:{task_id}"
        task_node = graph.get_node(task_root_id)
        if task_node:
            new_path = f"{session_id}/{task_node.node_id}"
            task_node.metadata["full_path"] = new_path
            task_node.metadata["parent_session"] = session_id
            # 创建 REFERENCE 边
            graph.add_edge(session_id, task_node.node_id, EdgeType.REFERENCE, ...)
            # 传播到子节点
            self._propagate_address_to_task_children(graph, task_node.node_id)
```

---

## 17.4 简单任务的 Session 节点创建

### 17.4.1 设计理念

即使简单任务（如闲聊回复）也会创建 session 节点，保证所有对话都有归属，不会丢失。通过重要性分级和遗忘/剪枝机制，简单对话会被自动清理。

### 17.4.2 Importance 分级与遗忘曲线

| 重要度 | 半衰期 | 适用场景 | 示例 |
|--------|--------|----------|------|
| TRIVIAL | 6 小时 | 闲聊/问候/语气词 | "你好"、"好的"、"谢谢" |
| NORMAL | 24 小时 | 普通对话 | 一般问题与回答 |
| IMPORTANT | 7 天 | 偏好/承诺 | "我喜欢吃辣" |
| FACT | 15 天 | 客观事实 | "我家在北京" |
| IDENTITY | 30 天 | 身份信息 | "我叫张三" |
| MUST_REMEMBER | 永不衰减 | 显式要求 | "帮我记住这个" |

### 17.4.3 TRIVIAL 规则扩展

**核心文件**: `zulong/memory/graph_adapters.py` — `DialogueAdapter._IMPORTANCE_RULES`

扩展了 TRIVIAL 规则以识别闲聊问候：

```python
_IMPORTANCE_RULES = [
    # must_remember: 用户显式要求
    (r'帮我记住|记得|别忘了|一定要记住|不要忘记|你要记得', MUST_REMEMBER, "explicit_remember"),
    # identity: 身份信息
    (r'我叫|我的名字|我姓|我是.{0,4}[人]|我今年.{0,4}岁|我的名字叫', IDENTITY, "identity"),
    # fact: 客观事实
    (r'我家在|我住在|我的电话|我的手机|我的生日|我的地址|我的邮箱|号码是', FACT, "fact"),
    # important: 偏好/承诺/任务指令
    (r'我喜欢|我不喜欢|我讨厌|我爱|我习惯|我每次都|我答应|我保证|以后都', IMPORTANT, "preference"),
    # trivial: 语气词/极短回复
    (r'^(嗯|好|好的|哦|ok|OK|行|是的|对|谢谢|感谢|拜拜|再见|没了|没有了|可以|你好|嗨|哈喽|hi|hello)$', TRIVIAL, "filler"),
    # trivial: 闲聊问候/寒暄
    (r'^(你好|您好|嗨|哈喽|hi |hello |早上好|下午好|晚上好|晚安|早安|最近怎么样|你好吗|最近如何|在吗|在干嘛|忙吗|干嘛呢|在不在|有空吗|聊聊天|随便聊聊|没什么事|随便问问)$', TRIVIAL, "greeting"),
]
```

### 17.4.4 突触修剪机制

**核心文件**: `zulong/memory/memory_graph.py` — `decay_and_prune()`

基于艾宾浩斯遗忘曲线的边权衰减与弱连接移除：

```python
def decay_and_prune(self, prune_threshold: float = 0.05):
    for node_id in list(self._graph.nodes()):
        node = self.get_node(node_id)
        importance = self._importance.get(node_id, Importance.NORMAL)
        half_life = _IMPORTANCE_HALF_LIFE.get(importance, 24.0)
        
        hours_since_access = (time.time() - node.last_accessed) / 3600.0
        decay_factor = 0.5 ** (hours_since_access / half_life)
        
        for _, target, edge_data in self._graph.out_edges(node_id, data=True):
            if edge_data.get("protected"):
                continue
            edge_data["weight"] *= decay_factor
            if edge_data["weight"] < prune_threshold:
                edges_to_remove.append((node_id, target))
```

---

## 17.5 Session 分配策略（Embedding 相似度）

**核心文件**: `zulong/memory/graph_adapters.py` — `DialogueAdapter.assign_session_by_similarity()`

三级策略：

```
策略 1: 恢复场景匹配
  if is_resume and resume_task_id:
    查找 bound_task_id == resume_task_id 的 session → 直接匹配

策略 2: Embedding 余弦相似度
  对用户输入做向量编码
  与各 session 最新 round 文本比较余弦相似度
  if best_score >= 0.55 → 归入该 session

策略 3: 创建新 session
  无匹配时新建 session，建立 TEMPORAL 边连接上一个 session
```

---

# 第 18 章：图谱事件推送机制

## 18.1 设计理念

TaskGraph 存储在内存中，前端通过 WebSocket 实时查看任务图谱的变化。系统需要在 FC 循环的关键节点发布 `L2_THINKING_STEP` 事件，携带任务图谱快照到前端。

---

## 18.2 L2_THINKING_STEP 事件

**核心文件**: `zulong/core/types.py` — `EventType.L2_THINKING_STEP`

事件类型定义：

```python
class EventType(Enum):
    L2_THINKING_STEP = "l2_thinking_step"  # L2 推理过程中的思考步骤
```

**事件流转**:
```
InferenceEngine (FC 循环)
  ↓ _send_thinking_step()
EventBus.publish(ZulongEvent(type=L2_THINKING_STEP))
  ↓ WebSocket Bridge 订阅
websocket_server.py → 转发到前端
  ↓
前端 handleThinkingStep() → addTaskGraph() 渲染
```

---

## 18.3 核心方法

### 18.3.1 _publish_task_graph_event

**核心文件**: `zulong/l2/inference_engine.py`

```python
def _publish_task_graph_event(
    self, pipeline_type: str,
    fc_turn: int = 0, tool_name: str = "", tool_result: str = "",
) -> None:
    """发布任务图谱更新事件到前端"""
    from zulong.tools.task_tools import get_active_task_graph
    tg = get_active_task_graph()
    if not tg:
        return
    
    graph_data = self._task_graph_to_frontend(tg)
    event_data = {
        "graph": graph_data, "turn": fc_turn,
        "tool": tool_name, "tool_count": len(tg._nodes),
    }
    if tool_result:
        event_data["tool_result"] = tool_result[:500]
    
    step_type = f"pipeline.{pipeline_type}"
    self._send_thinking_step(step_type, event_data)
```

### 18.3.2 _task_graph_to_frontend

```python
def _task_graph_to_frontend(self, tg) -> Dict:
    """将 TaskGraph 转换为前端 addTaskGraph() 兼容格式
    
    {
        "id": "tg_xxx", "title": "...",
        "nodes": [{"id", "label", "type", "status", "desc", "result", "files", "children"}],
        "activeNodeId": "req"
    }
    """
```

### 18.3.3 _send_thinking_step

```python
def _send_thinking_step(self, step_type: str, data: Dict) -> None:
    """发送 L2_THINKING_STEP 事件到 EventBus"""
    request_id = _current_request_id_var.get() or f"req_{int(time.time() * 1000)}"
    
    payload = {
        "request_id": request_id, "step_type": step_type,
        "data": data, "timestamp": time.time(),
        "iteration": data.get("turn", 0),
    }
    
    event = ZulongEvent(
        type=EventType.L2_THINKING_STEP,
        priority=EventPriority.NORMAL,
        source="InferenceEngine",
        payload=payload,
    )
    event_bus.publish(event)
```

---

## 18.4 FC 循环发布点

InferenceEngine 的 FC 循环在 **3 个关键点** 调用 `_publish_task_graph_event()`：

```python
# 1. pipeline_start: FC 循环开始前
self._publish_task_graph_event("pipeline_start", 0, "", "")

while fc_turn < max_fc_turns:
    fc_turn += 1
    # ... LLM 调用 + 工具执行
    for tc in msg.tool_calls:
        result_text = self._execute_tool_call(tc)
        # 2. agent_tool_call: 每个工具执行后
        self._publish_task_graph_event(
            "agent_tool_call", fc_turn, tc.function.name, result_text,
        )

# 3. agent_done: FC 循环结束
self._publish_task_graph_event("agent_done", fc_turn, "", "")
```

---

## 18.5 _active_task_graph_id 跟踪

**核心文件**: `zulong/l2/inference_engine.py`

InferenceEngine 维护活跃任务图 ID 跟踪，用于地址继承：

```python
class InferenceEngine:
    def __init__(self):
        self._active_task_graph_id: Optional[str] = None
    
    def _execute_tool_call(self, tool_call) -> str:
        if func_name == "task_create_plan" and isinstance(data, dict):
            graph_id = data.get("graph_id")
            if graph_id:
                self._active_task_graph_id = graph_id
```

---

## 18.6 WebSocket 桥转发

**核心文件**: `zulong/core/websocket_server.py`

WebSocket 桥订阅 `L2_THINKING_STEP` 事件并转发到前端：

```json
{
    "event_type": "l2_thinking_step",
    "data": {
        "request_id": "req_xxx",
        "step_type": "pipeline.agent_tool_call",
        "data": {
            "graph": {"id": "tg_xxx", "title": "...", "nodes": [...], "activeNodeId": "req"},
            "turn": 1, "tool": "task_add_node", "tool_count": 5
        }
    }
}
```

---

# 附录 A：关键文件清单

## A.1 核心实现文件

```
zulong/
├── infrastructure/
│   ├── shared_memory_pool.py       # 共享池 (跨会话记忆, DataEnvelope, 持久化)
│   └── data_ingestion.py           # 数据摄入引擎
│
├── memory/
│   ├── memory_graph.py             # 记忆图谱（异构图 + BFS 扩散激活, 102 KB）
│   │   ├── MemoryGraph 核心类
│   │   ├── compute_activations()  # BFS 扩散激活
│   │   ├── hebbian_strengthen()   # 赫布学习
│   │   ├── decay_and_prune()      # 艾宾浩斯衰减 + 突触修剪
│   │   ├── retrieve_context()     # 双路径并行检索
│   │   └── SummarySidecarIndex    # FAISS 摘要侧车索引
│   │
│   ├── graph_adapters.py           # 6 个图适配器（49 KB）
│   │   ├── TaskGraphAdapter       # TaskGraph → TASK/FILE
│   │   ├── DialogueAdapter        # 对话数据 → DIALOGUE
│   │   ├── KnowledgeGraphAdapter  # 知识图谱 → KNOWLEDGE/PERSON/CONCEPT
│   │   ├── EpisodeAdapter         # EpisodicMemory → EPISODE
│   │   ├── PersonProfileAdapter   # 人物档案 → PERSON
│   │   └── ExperienceAdapter      # ExperienceRAG → EXPERIENCE
│   │
│   ├── short_term_memory.py        # 短期记忆 (52 KB)
│   ├── episodic_memory.py          # 临时记忆/摘要 (29 KB)
│   ├── rag_manager.py              # RAG 管理器 (11 KB)
│   ├── rag_libraries.py            # 3 个 RAG 库实现 (27 KB)
│   ├── memory_evolution.py         # 记忆巩固/遗忘 (20 KB)
│   ├── llm_memory_reviewer.py      # LLM 剪枝守卫 (19 KB)
│   ├── embedding_manager.py        # 向量生成管理 (11 KB)
│   ├── knowledge_graph.py          # 知识图谱 (26 KB)
│   └── person_profile.py           # 人物档案 (24 KB)
│
├── tools/
│   ├── base.py                     # BaseTool / ToolRequest / ToolResult / ToolRegistry
│   ├── tool_engine.py              # ToolEngine 调度引擎
│   ├── core_tool_manager.py        # CoreToolManager 热/冷工具管理器
│   ├── search_tools.py             # search_tools 元工具
│   ├── task_tools.py               # 任务工具（地址继承实现）
│   ├── memory_graph_tools.py       # 记忆图谱工具
│   └── attention_tool.py           # NavigateAttentionTool 思维焦点导航
│
├── pipeline/                        # 五阶段流水线框架
│   ├── __init__.py                  #   包入口
│   ├── task_graph.py                #   TaskGraph 无限深度递归树
│   ├── orchestrator.py              #   PipelineOrchestrator 流水线编排器
│   ├── complexity_classifier.py     #   零成本复杂度分类器
│   ├── prompts.py                   #   Prompt 模板与构建函数
│   └── stages/                      #   五个阶段实现
│       ├── decompose.py             #     Stage1: 两相位拆解
│       ├── plan.py                  #     Stage2: Kahn 拓扑排序
│       ├── execute.py               #     Stage3: 按并行组执行
│       ├── review.py                #     Stage4: ReviewGate 审查
│       └── finalize.py              #     Stage5: 汇总输出
│
├── skill_packs/
│   ├── interface.py                # ISkillPack / SkillPackManifest 接口定义
│   ├── runtime.py                  # SkillPackRuntime 运行时
│   ├── loader.py                   # SkillPackLoader 加载器
│   └── packs/complex_task/         # 复杂任务处理技能包
│       ├── __init__.py             #   ComplexTaskPack 入口
│       ├── planner.py              #   TaskDecomposeAlgorithm
│       ├── planning_tools.py       #   任务工具集
│       ├── reasoning_engine.py     #   DeepReasoningEngine
│       ├── kv_cache_slot.py        #   KVCacheSlotManager
│       └── tools.py                #   推理工具
│
├── l2/
│   ├── inference_engine.py         # L2 推理引擎 (99 KB)
│   ├── interrupt_handler.py        # 中断处理器
│   ├── task_state_manager.py       # 任务状态管理器
│   ├── task_suspension.py          # 任务挂起/恢复 (13 KB)
│   ├── environment_snapshot.py     # 环境快照 (11 KB)
│   ├── recovery_notifier.py        # 启动恢复通知 (6 KB)
│   └── rag_node.py                 # RAG 集成节点
│
├── l1b/
│   ├── attention_controller.py     # L1-B 注意力控制器（14 KB）
│   ├── scheduler_gatekeeper.py     # L1-B 事件路由
│   └── core.py                     # L1-B 核心循环
│
├── core/
│   ├── event_bus.py                # 事件总线
│   ├── state_manager.py            # 状态管理器
│   ├── types.py                    # 事件类型定义
│   ├── websocket_server.py         # WebSocket 桥（L2 事件转发）
│   └── attention_atoms.py          # 注意力事件和上下文快照原子类
│
├── storage/
│   ├── hot_storage.py              # 热存储 (MongoDB)
│   ├── cold_storage.py             # 冷存储 (MinIO/S3)
│   └── logger.py                   # 日志收集器
│
├── utils/
│   └── model_preloader.py          # 模型预加载器（8 KB, v2.8 新增）
│
└── models/
    ├── container.py                # 模型容器
    └── config.py                   # 模型配置
```

## A.2 配置文件

```
config/
├── skill_packs.yaml                # 技能包配置（启用/禁用、路径、参数）
├── zulong_config.yaml              # 主配置（含 enabled_packs 列表）
└── memory_config.yaml              # 记忆系统配置
```

---

## A.3 测试验证文件

```
tests/
├── test_cross_session_memory.py    # 跨会话记忆测试
├── test_memory_retrieval.py        # 记忆检索测试
├── test_rag_search.py              # RAG 搜索测试
├── test_embedding_model.py         # Embedding 模型测试
└── test_storage_migration.py       # 存储迁移测试

check_*.py:
├── check_memory.py                 # 记忆状态检查
├── check_memory_content.py         # 记忆内容检查
├── check_memory_status.py          # 记忆健康度检查
└── check_pool_memory.py            # 共享池状态检查
```

---

## A.4 文档文件

```
docs/
├── TSD_v2.4.md                     # 本文档 (v2.9)
├── 记忆系统架构文档.md              # 记忆系统完整架构文档 (v3.0)
├── memory_graph/                   # 图式记忆架构完全指南 (6 篇分册)
├── 记忆架构改造任务文档.md          # MG 改造为读写统一的 12 个任务清单
├── 图记忆机制深度审查与FC融合分析.md # MG 能力二分法 + FC 工具规划
├── 三级记忆检索架构技术实现文档.md  # 第二代增强型三级记忆架构
├── 经验数据与事件记忆区分机制.md    # 经验 vs 事件记忆的五维区分
└── 修复计划.md                     # 上下文丢失修复计划
```

---

## 附录 B：术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| STM | Short-Term Memory | 短期记忆，缓存最近 20 轮对话 |
| LTM | Long-Term Memory | 长期记忆，RAG 向量库存储 |
| RAG | Retrieval-Augmented Generation | 检索增强生成，用向量检索增强 LLM |
| FAISS | Facebook AI Similarity Search | Facebook 开源的向量检索库 |
| Embedding | Embedding | 将文本/图像映射为向量的过程 |
| LRU | Least Recently Used | 最近最少使用，缓存清理策略 |
| TTL | Time To Live | 生存时间，数据过期时间 |
| IVF | Inverted File Index | 倒排文件索引，FAISS 近似搜索算法 |
| MongoDB | MongoDB | NoSQL 文档数据库，用于热存储 |
| MinIO | MinIO | 对象存储，兼容 S3 协议，用于冷存储 |
| ToolEngine | ToolEngine | 工具调用引擎，统一调度所有工具的执行 |
| BaseTool | BaseTool | 工具基类，定义 initialize/execute/cleanup 接口 |
| ToolRAG | Tool RAG | 工具语义索引库，按意图检索冷工具 |
| CoreToolManager | CoreToolManager | 热/冷工具管理器，管理工具在 prompt 中的可见性 |
| Function Calling | Function Calling | OpenAI 工具调用协议，LLM 自主决定调用哪个工具 |
| ISkillPack | ISkillPack | 技能包统一接口，定义安装/执行/卸载生命周期 |
| SkillPackRuntime | SkillPackRuntime | 技能包运行时，管理技能包的安装、执行、内化、卸载 |
| ComplexTaskPack | ComplexTaskPack | 统一复杂任务处理技能包，融合任务拆解与 LLM 深度推理 |
| DeepReasoningEngine | DeepReasoningEngine | LLM 增强版四步推理引擎，复用 L2 CORE client |
| KVCacheSlotManager | KV Cache Slot Manager | KV Cache 多命名 Slot 管理器，支持推理上下文热交换 |
| TaskGraph | TaskGraph | 任务图谱，无限深度递归树数据结构 |
| PipelineOrchestrator | PipelineOrchestrator | 五阶段流水线编排器，协调 Stage1~5 + ReviewGate 循环 |
| ComplexityClassifier | ComplexityClassifier | 零成本复杂度分类器，基于关键词多信号加权打分路由 |
| ReviewGate | ReviewGate | Stage4 审查纠错机制，LLM 审查完整性并标记失败节点 |
| 叶子节点 | Leaf Node | 没有子节点的非模板节点，是唯一被实际执行的工作项 |
| 状态聚合 | Status Aggregation | 非叶子节点的 status 从其子节点递归聚合 |
| 规划表 | Planning Table | TaskGraph 生成的递归 Markdown 表格，注入子任务 prompt 实现全局注意力 |
| MemoryGraph | Memory Graph | 统一异构类型图集成层，所有记忆子系统投射为图节点和边 |
| GraphAdapter | Graph Adapter | 图适配器，将后端数据投射为 MemoryGraph 的节点和边 |
| BFS 扩散激活 | BFS Activation | 基于加权 BFS 的扩散激活算法，用于上下文发现 |
| 赫布学习 | Hebbian Learning | 共激活节点间的边权增强机制 |
| 突触修剪 | Synaptic Pruning | 基于艾宾浩斯遗忘曲线的边权衰减与弱连接移除 |
| 地址继承 | Address Inheritance | 节点地址携带完整父级路径的机制 |
| REFERENCE 边 | Reference Edge | 跨类型引用边，连接对话和任务节点 |
| L2_THINKING_STEP | L2 Thinking Step | L2 推理过程中的思考步骤事件，用于前端实时展示 |
| Importance | Importance | 节点重要度分级（TRIVIAL/NORMAL/IDENTITY/FACT/IMPORTANT/MUST_REMEMBER） |
| 半衰期 | Half-Life | 不同重要度节点的遗忘曲线衰减半衰期（6h/24h/15d/30d/7d/∞） |
| 3D 记忆城市 | 3D Memory City | 记忆系统的三维设计理念：宏空间/任务空间/地下档案室 |
| InterruptHandler | InterruptHandler | L2 推理层内部的中断处理器，负责安全停止和恢复生成 |
| TaskStateManager | TaskStateManager | 任务状态管理器，管理活跃/冻结任务的调度和恢复顺序 |
| AttentionController | AttentionController | 注意力控制器，决定事件处理策略（打断/排队/直通） |
| SharedMemoryPool | SharedMemoryPool | 共享记忆池，系统中央邮局，统一管理所有模块数据 |
| DataEnvelope | DataEnvelope | 数据信封，包装所有进入共享池的数据 |
| ContextSnapshot | ContextSnapshot | 上下文快照，保存被中断任务的完整状态 |
| TaskSuspensionManager | TaskSuspensionManager | 任务挂起/恢复管理器，序列化任务状态到磁盘 |
| EnvironmentSnapshot | EnvironmentSnapshot | 环境快照，记录物体状态和用户位置用于恢复对比 |
| RecoveryNotifier | RecoveryNotifier | 恢复通知器，系统启动时扫描可恢复任务 |
| SummarySidecarIndex | SummarySidecarIndex | FAISS 摘要侧车索引，只存摘要向量和节点指针 |
| 多维标签体系 | Multi-dimensional Tags | 温度/重要度/时间段三组正交标签替代硬分区 |
| 对话空间/任务空间 | Dialogue/Task Space | 概念性分区，实际在同一张图中，不做物理分离 |

---

**文档生成时间**: 2026-04-17  
**最后更新**: 2026-04-23  
**维护团队**: 祖龙架构团队  
**文档版本**: TSD v2.9  
**审查状态**: 已审查
