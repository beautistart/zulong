# 三级记忆检索架构设计文档

## 📋 问题背景

### 原有架构的缺陷

1. **❌ 超过 3 轮的对话历史如何处理？**
   - 原实现：只保留最近 2 轮在 `conversation_history`
   - 问题：其他对话被丢弃（除非已持久化到向量库）
   - 缺陷：**没有临时对话的检索机制**

2. **❌ 需要前 10 轮内容怎么办？**
   - 原实现：无法处理跨越多轮的长程依赖
   - 问题：向量检索只能找到"语义相似"的记忆，无法找到"时间序列上靠前"的记忆

3. **❌ L1-B 如何检索临时对话？**
   - 原实现：L1-B **没有记忆检索功能**
   - 问题：完全依赖 L2 的 `conversation_history`（只有 2 轮）

4. **❌ 检索的是摘要还是全文？**
   - 原实现：检索的是**完整对话**（user + ai）
   - 缺陷：**没有摘要机制**，检索效率低
   - 缺陷：**没有分级读取机制**

## ✅ 三级记忆架构设计

### 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    L2 推理引擎                           │
├─────────────────────────────────────────────────────────┤
│ 1. 工作记忆 (Working Memory)                             │
│    - 最近 2 轮对话（完整内容）                            │
│    - 直接从 conversation_history 读取                    │
│    - 用途：保持即时对话连贯性                            │
│                                                         │
│ 2. 临时记忆 (Episodic Memory) ← 新增                    │
│    - 最近 20-50 轮对话（摘要 + 索引）                     │
│    - 基于摘要检索，支持详细内容读取                      │
│    - 存储在 SharedMemoryPool                            │
│    - 用途：支持中等时间跨度的上下文依赖                  │
│                                                         │
│ 3. 长期记忆 (Semantic Memory)                           │
│    - 向量化的知识和经验                                 │
│    - 通过 RAG 检索                                       │
│    - 用途：提供背景知识和事实信息                        │
└─────────────────────────────────────────────────────────┘
```

### 记忆分层对比

| 层级 | 存储内容 | 存储位置 | 检索方式 | 用途 | 保留时间 |
|------|---------|---------|---------|------|---------|
| **工作记忆** | 完整对话 | 内存 (conversation_history) | 时间序列（最近 2 轮） | 即时上下文 | 会话期间 |
| **临时记忆** | 摘要 + 索引 | SharedMemoryPool | 语义检索（基于摘要） | 中等跨度依赖 | 2 小时 |
| **长期记忆** | 向量化知识 | ChromaDB | 向量相似度 | 背景知识 | 永久 |

## 🔧 核心组件实现

### 1. EpisodicMemory（临时记忆管理器）

**文件**: `zulong/memory/episodic_memory.py`

**核心功能**:
- ✅ **对话摘要**：为每轮对话生成简短摘要（50-100 字）
- ✅ **基于摘要检索**：快速找到相关对话
- ✅ **分级读取**：
  - Level 1: 摘要（快速浏览）
  - Level 2: 完整对话（按需读取）
- ✅ **时间窗口管理**：支持按时间范围检索

**数据结构**:
```python
{
    "episode_id": 42,
    "summary": "询问定义：AI MAX 395 是什么 → AI MAX 395 是一款高性能处理器...",
    "user_preview": "AI MAX 395 是什么？",
    "ai_preview": "AI MAX 395 是一款高性能处理器，具有...",
    "trace_id": "mem_abc123",  # 指向完整对话
    "timestamp": 1234567890,
    "ttl": 7200  # 2 小时
}
```

**关键方法**:
```python
# 存储对话并生成摘要
await episodic_memory.store_episode(user_input, ai_response)

# 基于摘要检索
episodes = await episodic_memory.search_by_summary(query, top_k=5, time_window=7200)

# 分级读取：读取完整对话
full_dialogue = await episodic_memory.get_full_dialogue(episode_id)
```

### 2. L2 推理引擎增强

**文件**: `zulong/l2/inference_engine.py`

**记忆注入策略**:
```python
# 1. 工作记忆：最近 2 轮对话
recent_history = conversation_history[-4:]  # 2 轮 = 4 条消息

# 2. 临时记忆：基于摘要检索 Top-5
relevant_episodes = await episodic_memory.search_by_summary(
    query=user_input,
    top_k=5,
    time_window=7200  # 2 小时内
)

# 3. 长期记忆：从 ShortTermMemory 检索
relevant_memories = await short_term_memory.search_similar(user_input, top_k=3)
```

**上下文注入格式**:
```
System Prompt:
- 角色定义
- 时间信息
- 人称规则
- 工具描述
- 视觉观察（如有）
- RAG 知识（如有）
- 【相关记忆】(基于摘要检索)  ← 新增
  1. [摘要] 询问定义：AI MAX 395 是什么 → ...
     [详情] 说'读取第 1 条详情'可查看完整对话
  💡 提示：如需查看某条记忆的完整内容，请说'读取第 X 条详情'。

Messages:
- System
- User/Ali 最近 2 轮对话
- User 当前输入
```

### 3. 记忆读取工具

**工具名称**: `read_memory_detail`

**用途**: 当用户说"读取第 X 条详情"时，模型调用此工具读取完整对话

**工具定义**:
```json
{
    "name": "read_memory_detail",
    "description": "读取临时记忆的详细内容。当用户说'读取第 X 条详情'时使用此工具",
    "parameters": {
        "type": "object",
        "properties": {
            "episode_index": {
                "type": "integer",
                "description": "要读取的记忆编号（对应注入上下文中的序号）"
            }
        },
        "required": ["episode_index"]
    }
}
```

**工具响应**:
```
【第 1 条记忆详情】
用户问：AI MAX 395 是什么？

AI 答：AI MAX 395 是一款高性能处理器，具有以下特性：
- 采用 5nm 工艺
- 集成 128 核 GPU
- 支持 DDR5 内存
...
```

## 📊 工作流程

### 1. 记忆存储流程

```
用户输入 → L2 生成回复 → 同时存储到两个地方：
                            ├─→ ShortTermMemory（完整对话）
                            └─→ EpisodicMemory（生成摘要 + 索引）
```

### 2. 记忆检索流程

```
用户新输入 → L2 推理引擎
              ├─→ 工作记忆：最近 2 轮（保持连贯性）
              ├─→ 临时记忆：基于摘要检索 Top-5（中等跨度）
              └─→ 长期记忆：基于语义检索 Top-3（历史相似对话）
```

### 3. 分级读取流程

```
用户："读取第 2 条详情"
  ↓
L2 调用 read_memory_detail 工具
  ↓
从 SharedMemoryPool 读取完整对话
  ↓
返回完整内容到对话
```

## 🎯 使用场景示例

### 场景 1：跨越多轮的指代消解

```
第 1 轮：用户："AI MAX 395 是什么？"
        AI："AI MAX 395 是一款高性能处理器..."

第 2-10 轮：讨论其他话题...

第 11 轮：用户："它多少钱？"
        AI：[检索临时记忆] → 找到第 1 轮关于 AI MAX 395 的讨论
            → "AI MAX 395 的价格约为..."
```

### 场景 2：主动查看历史详情

```
用户："我之前问过什么关于处理器的问题？"
  ↓
AI：[检索临时记忆] → 显示摘要列表
    1. [摘要] 询问定义：AI MAX 395 是什么 → ...
    2. [摘要] 询问方法：如何安装 CPU 散热器 → ...
    3. [摘要] 询问价格：AI MAX 395 多少钱 → ...
    💡 提示：如需查看某条记忆的完整内容，请说'读取第 X 条详情'。

用户："读取第 1 条详情"
  ↓
AI：[调用 read_memory_detail 工具] → 显示完整对话
```

### 场景 3：长时间跨度的上下文依赖

```
第 1 轮：用户："我想了解量子计算"
        AI："量子计算是基于量子力学原理的新型计算范式..."

第 5 轮：用户："刚才说的量子比特是什么？"
        AI：[检索临时记忆] → 找到第 1 轮的讨论
            → "我之前提到，量子比特（qubit）是..."
```

## 🔍 检索策略对比

### 原有策略 vs 新策略

| 维度 | 原有策略 | 新策略（三级记忆） |
|------|---------|------------------|
| **检索范围** | 仅 2 轮工作记忆 + 向量库 | 工作记忆 + 临时记忆 + 长期记忆 |
| **检索依据** | 语义相似度 | 语义相似度 + 时间窗口 + 摘要 |
| **检索效率** | 低（需要读取完整对话） | 高（先检索摘要，再按需读取） |
| **长程依赖** | ❌ 不支持 | ✅ 支持（通过临时记忆） |
| **模型主动性** | ❌ 被动接受注入 | ✅ 主动读取详情 |

## 📈 性能优化

### 1. 摘要生成优化

**当前实现**：基于规则的简单摘要
```python
# 判断问题类型
if "是什么" in user_input:
    question_type = "询问定义"
elif "怎么" in user_input:
    question_type = "询问方法"
...

summary = f"{question_type}: {user_input[:30]} → {ai_response[:30]}"
```

**未来优化**：使用轻量级模型生成摘要
```python
# 使用 L1-B 生成高质量摘要
summary = await summary_model.generate(
    prompt=f"请用 50 字总结以下对话：\n用户：{user_input}\nAI: {ai_response}"
)
```

### 2. 并发读取优化

```python
# 并发读取所有候选记忆
tasks = [get_turn_by_id(turn_id) for turn_id in turn_ids]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. 缓存优化

```python
# 缓存最近检索结果
@lru_cache(maxsize=100)
def search_by_summary_cached(query_hash, top_k, time_window):
    ...
```

## 🚀 未来扩展

### 1. L1-B 记忆检索增强

**当前**：L1-B 没有记忆检索功能

**未来**：
```python
# L1-B 在路由前检索相关记忆
async def on_user_text(self, event):
    # 1. 检索相关记忆
    relevant_memories = await episodic_memory.search_by_summary(
        query=event.text, top_k=3
    )
    
    # 2. 将记忆注入到 L2 输入
    event.metadata["relevant_memories"] = relevant_memories
    
    # 3. 路由到 L2
    await route_to_l2(event)
```

### 2. 记忆重要性评分

```python
# 为每轮对话生成重要性评分（0-1）
importance_score = await calculate_importance(user_input, ai_response)

# 高重要性记忆保留更长时间
if importance_score > 0.8:
    ttl = 86400  # 24 小时
elif importance_score > 0.5:
    ttl = 7200   # 2 小时
else:
    ttl = 1800   # 30 分钟
```

### 3. 记忆关联图

```python
# 建立记忆之间的关联
# 例如：第 5 轮对话引用了第 1 轮的概念
# 创建双向链接，支持导航
memory_graph.add_link(
    from_episode=5,
    to_episode=1,
    relation="references"
)
```

## ✅ 验收标准

### 功能验收

- [ ] 工作记忆：正确保留最近 2 轮对话
- [ ] 临时记忆：正确生成摘要并存储
- [ ] 记忆检索：基于摘要检索 Top-K 相关记忆
- [ ] 分级读取：支持"读取第 X 条详情"指令
- [ ] 时间窗口：正确过滤过期记忆

### 性能验收

- [ ] 检索延迟：< 500ms（基于摘要检索）
- [ ] 存储延迟：< 1s（包含摘要生成）
- [ ] 读取延迟：< 200ms（从共享池读取）

### 质量验收

- [ ] 摘要准确性：能准确反映对话核心内容
- [ ] 检索相关性：Top-3 检索结果相关性 > 80%
- [ ] 上下文连贯性：跨越多轮对话时保持连贯

## 📝 总结

通过引入**三级记忆架构**，我们解决了以下核心问题：

1. ✅ **超过 3 轮的对话历史**：通过临时记忆管理，保留最近 20-50 轮
2. ✅ **长程依赖需求**：支持检索前 10 轮甚至更早的内容
3. ✅ **L1-B 记忆检索**：为未来 L1-B 集成记忆检索提供接口
4. ✅ **分级读取机制**：先检索摘要，再按需读取详情

这套架构参考了人类记忆系统的运作方式：
- **工作记忆** ≈ 短期记忆（7±2 个组块）
- **临时记忆** ≈ 情景记忆（特定时间地点的事件）
- **长期记忆** ≈ 语义记忆（事实和知识）

通过分层管理，实现了**效率**与**完整性**的平衡。
