# 祖龙系统图式记忆架构完全指南

> **文档版本**: v1.0  
> **最后更新**: 2026-04-19  
> **适用系统**: 祖龙系统 (ZULONG) Beta 4  
> **目标读者**: 系统架构师、AI 工程师、开发者

---

## 📖 文档导航

本系列文档包含以下内容：

| 文档 | 说明 | 阅读时间 |
|------|------|---------|
| [01-图记忆架构概述](./01-architecture.md) | 核心概念、架构设计、数据模型 | 15 分钟 |
| [02-记忆分类与标签体系](./02-classification-tags.md) | 多维标签、重要度、温度管理 | 10 分钟 |
| [03-记忆检索机制](./03-retrieval.md) | 并行检索、BFS 扩散、FAISS 索引 | 12 分钟 |
| [04-图注意力机制](./04-attention.md) | 从 1D 到图 BFS、注意力评分 | 10 分钟 |
| [05-复杂任务编排](./05-task-orchestration.md) | TaskGraph 集成、Agent 协作 | 15 分钟 |
| [06-快速入门指南](./06-quickstart.md) | 代码示例、最佳实践 | 8 分钟 |
| [07-FAQ 与故障排查](./07-faq.md) | 常见问题、性能优化 | 5 分钟 |

---

## 🚀 5 分钟快速理解 MemoryGraph

### 什么是 MemoryGraph？

**MemoryGraph（记忆图谱）** 是祖龙系统的**统一记忆中枢**，它将所有类型的记忆（对话、任务、知识、人物、文件等）组织成一个**异构类型图**。

**核心类比**: 如果把 LLM 比作大脑皮层（负责推理），那么 MemoryGraph 就是**海马体**（负责记忆索引和联想）。

---

### 为什么需要图式记忆？

#### 传统记忆架构的问题

```
用户："我之前问的那个处理器多少钱？"
     ↓
传统系统：❌ 只能找到语义相似的对话
          无法理解"那个处理器"指的是 3 轮前讨论的"AI MAX 395"
```

**根本原因**:
- 记忆孤岛：对话记忆、任务记忆、知识记忆彼此独立
- 一维注意力：只能按时间远近筛选，无法发现跨类型关联

#### 图式记忆的解决方案

```
用户："我之前问的那个处理器多少钱？"
     ↓
MemoryGraph: ✅ 从当前节点出发，沿 REFERENCE 边追溯
            → 找到 3 轮前的 DIALOGUE 节点
            → 发现讨论的是"AI MAX 395"
            → 检索 KNOWLEDGE 节点中的价格信息
            → 回答："AI MAX 395 的价格约为 1299 元"
```

**核心优势**:
- ✅ **结构化关联**: 任意类型的记忆都可以相互连接
- ✅ **智能检索**: 从任意线索追溯全局关联（类似人脑联想）
- ✅ **有效无限上下文**: 图可存储任意规模知识，当前窗口只注入最相关的子图

---

### 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    消费层 (Consumers)                    │
│  L2 推理引擎 | AttentionWindowManager | AgentOrchestrator│
└───────────────────┬─────────────────────────────────────┘
                    │ query / traverse / activate
┌───────────────────▼─────────────────────────────────────┐
│              MemoryGraph (记忆图谱集成层)                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  NetworkX DiGraph (异构类型图)                     │   │
│  │  - 9 种节点类型 (TASK/DIALOGUE/KNOWLEDGE/...)      │   │
│  │  - 7 种边类型 (HIERARCHY/SEMANTIC/REFERENCE/...)   │   │
│  │  - 边权：赫布学习增强 + 艾宾浩斯衰减               │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────┐ ┌──────────┐ ┌──────────────────┐    │
│  │ FAISS 摘要索引│ │ 异步修剪器│ │ JSON 持久化       │    │
│  │ (冷数据检索)  │ │ (30min)   │ │ (跨会话保留)     │    │
│  └──────────────┘ └──────────┘ └──────────────────┘    │
└───────────────────┬─────────────────────────────────────┘
                    │ adapters (只读适配器)
┌───────────────────▼─────────────────────────────────────┐
│                现有后端 (保持不变)                        │
│  ShortTermMemory | EpisodicMemory | KnowledgeGraph      │
│  RAG Libraries   | PersonProfile    | TaskGraph          │
└─────────────────────────────────────────────────────────┘
```

---

### 核心数据模型

#### GraphNode（记忆节点）

```python
@dataclass
class GraphNode:
    node_id: str                    # 全局唯一，带类型前缀
    node_type: NodeType             # 节点类型 (9 种)
    label: str                      # 人类可读标签
    activation: float               # 当前激活水平 (0.0-1.0)
    created_at: float
    last_accessed: float
    access_count: int
    backend_ref: str                # 后端来源指针
    metadata: Dict[str, Any]        # 包含多维标签
```

**9 种节点类型**:
- `TASK` - 任务节点
- `DIALOGUE` - 对话节点
- `KNOWLEDGE` - 知识实体
- `EXPERIENCE` - 经验文档
- `EPISODE` - 情景摘要
- `FILE` - 文件引用
- `CONCEPT` - 概念
- `PERSON` - 人物信息
- `DOCUMENT` - 文档（预留）

#### GraphEdge（记忆关联）

```python
# 边属性存储在 NetworkX 图中
{
    "edge_type": EdgeType.REFERENCE,
    "weight": 0.85,                 # 0.0-1.0+，赫布增强 + 衰减
    "created_at": 1234567890,
    "last_activated": 1234567890,
    "protected": False,             # True = 永不修剪
}
```

**7 种边类型**:
- `HIERARCHY` - 父子关系（结构性边，protected）
- `DEPENDENCY` - 数据依赖（结构性边）
- `REFERENCE` - 跨类型引用
- `TEMPORAL` - 时间序列（结构性边）
- `SEMANTIC` - 语义相似（embedding cosine > 0.7）
- `CAUSAL` - 因果关系
- `ASSOCIATION` - 赫布学习产生的关联

---

### 多维标签体系

MemoryGraph 使用**三组正交标签**实现灵活的记忆分类：

| 维度 | 标签值 | 计算方式 | 用途 |
|------|-------|---------|------|
| **温度** | HOT / WARM / COLD | 基于 `last_accessed` 实时计算 | 检索路由（热数据直接遍历） |
| **重要度** | TRIVIAL / NORMAL / IDENTITY / FACT / IMPORTANT / MUST_REMEMBER | 写入时规则检测 + 动态提升 | 差异化衰减（6h-永久） |
| **时间段** | RECENT / NON_RECENT | 查询时动态计算（默认 30 分钟窗口） | 并行检索（互斥过滤） |

**示例**:
```python
# 新节点默认标签
node.metadata = {
    "temperature": "hot",        # 刚写入，是热的
    "importance": "normal",      # 普通对话
    "content": "用户输入全文",
    "ai_response": "AI 回复全文"
}

# 重要度动态提升（access_count >= 3）
# normal → important，半衰期从 24h 延长到 7 天
```

---

### 并行检索策略

```
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

**性能优势**:
- 热数据（30 分钟内）: < 50ms（直接遍历）
- 冷数据（30 分钟前）: < 200ms（FAISS 向量检索）
- 总体检索延迟：平均 < 100ms

---

### 图注意力机制

#### 从 1D 到图 BFS 的跃升

**原有架构（1D 注意力）**:
```python
# AttentionWindowManager 在消息序列上线性评分
weight = base × time_decay × mode_multiplier

缺陷：只能按"时间远近 + 工具关联"筛选
```

**新架构（图注意力）**:
```python
# 加权 BFS 扩散激活
def compute_activations(seed_nodes, max_depth=3, decay=0.5):
    """
    从种子节点出发，沿边传播激活值
    propagated = activation × edge_weight × decay
    """
    返回：Dict[node_id → activation_score]

# 注意力评分公式升级
weight = base × time_decay × mode_multiplier × graph_boost
# 其中 graph_boost = 1.0 + activation (最大 2.0x 加成)
```

**核心优势**:
- ✅ 发现跨类型的语义关联（如：当前任务依赖的知识点出现在 30 轮前的对话中）
- ✅ 对模型大小不敏感（图检索质量由图结构和边权决定）
- ✅ 对数据量不敏感（BFS 有 max_depth + 阈值剪枝）

---

### 复杂任务编排

#### TaskGraph 与 MemoryGraph 的双向连接

```
任务创建流程:
1. 用户："帮我写一个 Python 脚本，爬取天气数据"
   ↓
2. L2 推理引擎 → 创建 TaskGraph
   ├─ 根任务：o1_1 "爬取天气数据"
   ├─ 子任务：o1_1_1 "分析需求"
   ├─ 子任务：o1_1_2 "编写代码"
   └─ 子任务：o1_1_3 "测试运行"
   ↓
3. TaskGraphAdapter → 同步到 MemoryGraph
   ├─ 添加 TASK 节点 (o1_1, o1_1_1, ...)
   ├─ 添加 HIERARCHY 边 (父子关系)
   ├─ 添加 REFERENCE 边 (任务 → 相关文件)
   └─ 设置 backend_ref (指向原始 TaskGraph)
   ↓
4. 任务执行过程中
   ├─ 每个子任务完成 → 更新节点 activation
   ├─ 引用知识文档 → 添加 REFERENCE 边
   └─ 遇到问题 → 创建 EXPERIENCE 节点（经验沉淀）
   ↓
5. 任务完成后
   └─ 经验回流到 MemoryGraph（形成 ASSOCIATION 边）
```

**Agent 协作流程**:
```
L2 推理引擎 (主实例)
├─ 当前聚焦：task:o1_1
├─ 种子节点：[task:o1_1, dialogue:42, dialogue:41]
├─ BFS 扩散激活 → 发现相关节点
│  ├─ task:o1_1_1 (子任务，HIERARCHY 边)
│  ├─ file:weather_api.py (文件引用，REFERENCE 边)
│  ├─ knowledge:python_requests (知识，SEMANTIC 边)
│  └─ experience:crawl_error_handling (经验，ASSOCIATION 边)
└─ 注入上下文 → LLM 推理

L1-B 调度器
├─ 监控 L2 状态
├─ 空闲时触发后台复盘
└─ 紧急情况下切换 KV Cache
```

---

## 🎯 下一步阅读

根据您的角色和目标，推荐阅读顺序：

### 架构师/技术负责人
1. [01-图记忆架构概述](./01-architecture.md) - 理解整体设计
2. [05-复杂任务编排](./05-task-orchestration.md) - 规划系统演进
3. [03-记忆检索机制](./03-retrieval.md) - 评估性能指标

### AI 工程师/算法开发者
1. [02-记忆分类与标签体系](./02-classification-tags.md) - 理解标签设计
2. [03-记忆检索机制](./03-retrieval.md) - 掌握检索算法
3. [04-图注意力机制](./04-attention.md) - 优化注意力评分

### 应用开发者
1. [06-快速入门指南](./06-quickstart.md) - 快速上手编码
2. [01-图记忆架构概述](./01-architecture.md) - 理解核心概念
3. [07-FAQ 与故障排查](./07-faq.md) - 解决实际问题

### 产品经理/设计师
1. [01-图记忆架构概述](./01-architecture.md) - 理解产品能力
2. [05-复杂任务编排](./05-task-orchestration.md) - 设计用户流程
3. [02-记忆分类与标签体系](./02-classification-tags.md) - 理解记忆管理

---

## 📚 相关文档

- [三级记忆检索架构设计](../memory_architecture_design.md) - 旧版记忆架构文档
- [记忆图谱架构升级规划](../../.qoder/specs/memory-graph-architecture-upgrade.md) - 架构设计原文档
- [记忆架构改造任务文档](../记忆架构改造任务文档.md) - 实施任务清单

---

## 🤝 贡献指南

如果您发现文档中的错误或有改进建议，请：

1. 在 GitHub 上提交 Issue
2. 提交 Pull Request 修改文档
3. 联系核心开发团队讨论

---

## 📄 许可证

本文档采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) 许可证。

---

**最后更新**: 2026-04-19  
**维护者**: 祖龙系统核心开发团队
