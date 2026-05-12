# 祖龙系统 - 经验向量搜索与注入完整流程

**文档版本**: 2026-03-29  
**对应代码**: `zulong/l1b/async_scheduler.py`, `zulong/memory/three_libraries.py`

---

## 📊 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           用户输入 (User Input)                              │
│                     "网络慢怎么办？"                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        L1-B 异步调度器 (AsyncL1BScheduler)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 1. handle_request() - 接收用户请求                                   │   │
│  │    - 创建 TaskItem (task_id, raw_input, status=PENDING)             │   │
│  │    - 将任务放入 task_queue                                          │   │
│  │    - 启动异步检索任务 _retrieve_and_inject()                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 2. rewrite_query() - 查询重写                                        │   │
│  │    - "切蛋糕" → "任务分片"                                            │   │
│  │    - "滚雪球" → "迭代积累"                                            │   │
│  │    - "网络慢" → "网络慢" (无需重写)                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 3. _retrieve_and_inject() - 后台检索与注入                          │   │
│  │    - 调用 library_manager.retrieve_all()                            │   │
│  │    - 并行检索三个库（技能 + 经验 + 知识）                            │   │
│  │    - 构建超级 Prompt                                                 │   │
│  │    - 发布 SYSTEM_L2_COMMAND 事件                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ThreeLibraryManager (三库管理器)                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ retrieve_all() - 并行检索三个库                                      │   │
│  │    ↓                                                                 │   │
│  │    1. skill_store.get_all() → 技能（内存直读，0ms）                  │   │
│  │    2. experience_store.search_by_text() → 经验（向量检索）           │   │
│  │    3. knowledge_store.search_by_text_async() → 知识（异步检索）      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                         经验库向量检索流程                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ search_by_text(query, filter_type="logic", limit=5)                 │   │
│  │    ↓                                                                 │   │
│  │    1. _get_embedding(query) → query_vector (768 维)                   │   │
│  │    2. search(query_vector, filter_type, limit)                       │   │
│  │       - 遍历所有经验                                                 │   │
│  │       - 过滤：exp.experience_type == filter_type                     │   │
│  │       - 计算余弦相似度：cosine_similarity(query_vector, exp.embedding)│   │
│  │       - 排序：按相似度降序                                           │   │
│  │       - 返回：Top-K 经验列表                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                      构建超级 Prompt (Prompt Assembly)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ build_super_prompt(query, skills, experiences, knowledge)           │   │
│  │    ↓                                                                 │   │
│  │    1. skills.to_prompt_context() → "## 系统能力"                     │   │
│  │    2. experiences.to_prompt_context() → "## 相关经验"                │   │
│  │    3. knowledge.to_prompt_context() → "## 相关知识"                  │   │
│  │    4. 添加用户请求 → "## 用户请求"                                   │   │
│  │    5. 合并所有部分                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                        发布事件到 L2 (Event Publishing)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ event_bus.publish(SYSTEM_L2_COMMAND)                                │   │
│  │    payload: {                                                        │   │
│  │      "task_id": "abc12345",                                          │   │
│  │      "prompt": final_prompt,                                         │   │
│  │      "status": "READY",                                              │   │
│  │      "retrieval_time_ms": 150.5                                      │   │
│  │    }                                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                          L2 推理引擎 (Inference Engine)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 接收 SYSTEM_L2_COMMAND 事件                                           │   │
│  │    - 提取 prompt                                                     │   │
│  │    - 调用 Model Engine 进行推理                                       │   │
│  │    - 生成回复                                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 详细流程分解

### 阶段 1️⃣：用户输入与任务创建

**代码位置**: [`async_scheduler.py:169-201`](file://d:\AI\project\zulong_beta4\zulong\l1b\async_scheduler.py#L169-L201)

```python
def handle_request(self, user_input: str, 
                   context: Optional[Dict] = None) -> str:
    """主入口：接收用户请求（非阻塞）"""
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    # 创建任务项
    task = TaskItem(
        task_id=task_id,
        prompt="",
        status="PENDING",
        raw_input=user_input,
        context=context or {}
    )
    
    # 放入队列
    self.task_queue.put(task)
    
    # 启动异步检索
    if self._loop and self._running:
        asyncio.run_coroutine_threadsafe(
            self._retrieve_and_inject(task),
            self._loop
        )
    
    return task_id
```

**关键数据结构**:
```python
@dataclass
class TaskItem:
    task_id: str              # 任务 ID（8 位短 UUID）
    prompt: str               # 组装后的 Prompt（初始为空）
    status: str               # "PENDING" → "READY" → "EXECUTING" → "COMPLETED"
    raw_input: str            # 用户原始输入
    context: Dict[str, Any]   # 附加上下文（如 messages）
    created_at: float         # 创建时间戳
    ready_at: Optional[float] # 就绪时间戳
```

---

### 阶段 2️⃣：查询重写（可选）

**代码位置**: [`async_scheduler.py:153-168`](file://d:\AI\project\zulong_beta4\zulong\l1b\async_scheduler.py#L153-L168)

```python
def rewrite_query(self, raw_input: str) -> str:
    """查询重写：将口语转化为技术术语"""
    tech_query = raw_input
    
    # 预定义的重写规则
    rewrite_rules = {
        "切蛋糕": "任务分片",
        "滚雪球": "迭代积累",
        "搭积木": "模块化构建",
        "剥洋葱": "逐层深入",
        # ... 更多规则
    }
    
    for colloquial, technical in rewrite_rules.items():
        if colloquial in raw_input:
            tech_query = tech_query.replace(colloquial, technical)
            logger.info(f"查询重写：'{colloquial}' -> '{technical}'")
    
    return tech_query
```

**示例**:
- 输入："像切蛋糕一样分任务"
- 输出："像任务分片一样分任务"

---

### 阶段 3️⃣：并行检索三个库

**代码位置**: [`async_scheduler.py:240-260`](file://d:\AI\project\zulong_beta4\zulong\l1b\async_scheduler.py#L240-L260)  
**三库管理器**: [`three_libraries.py:786-821`](file://d:\AI\project\zulong_beta4\zulong\memory\three_libraries.py#L786-L821)

```python
async def _retrieve_and_inject(self, task: TaskItem):
    """后台任务：主动记忆注入"""
    
    # 1. 重写查询
    tech_query = self.rewrite_query(task.raw_input)
    
    # 2. 并行检索三个库
    results = await self.library_manager.retrieve_all(
        query=tech_query,
        experience_type="logic",      # 只检索逻辑类经验
        experience_limit=5,           # 最多 5 条经验
        knowledge_limit=10            # 最多 10 条知识
    )
    
    # 3. 构建超级 Prompt
    memory_prompt = self.library_manager.build_super_prompt(
        query=task.raw_input,
        skills=results.get("skills"),
        experiences=results.get("experiences"),
        knowledge=results.get("knowledge")
    )
    
    # 4. 合并上下文
    context_prompt = self._build_context_from_messages(messages)
    final_prompt = context_prompt + "\n\n" + memory_prompt
    
    # 5. 标记任务就绪
    task.prompt = final_prompt
    task.status = "READY"
```

**并行检索实现**:
```python
async def retrieve_all(self, query: str, ...) -> Dict[str, Any]:
    """并行检索三个库"""
    # 1. 技能库（内存直读，0ms 延迟）
    skills = self.skill_store.get_all()
    
    # 2. 经验库（向量检索，需要 Embedding）
    experiences = self.experience_store.search_by_text(
        query, experience_type, experience_limit
    )
    
    # 3. 知识库（异步检索，避免阻塞）
    knowledge = await self.knowledge_store.search_by_text_async(
        query, knowledge_domain, knowledge_limit
    )
    
    return {
        "skills": skills,
        "experiences": experiences,
        "knowledge": knowledge
    }
```

---

### 阶段 4️⃣：经验库向量检索详解

**代码位置**: [`three_libraries.py:360-374`](file://d:\AI\project\zulong_beta4\zulong\memory\three_libraries.py#L360-L374)

#### 4.1 向量化查询文本

```python
def search_by_text(self, query: str,
                   filter_type: Optional[str] = "logic",
                   limit: int = 5) -> List[Experience]:
    """通过文本查询"""
    # 1. 将查询文本转换为向量
    query_vector = self._get_embedding(query)
    
    # 2. 执行向量检索
    return self.search(query_vector, filter_type, limit)
```

**Embedding 获取**:
```python
def _get_embedding(self, text: str) -> np.ndarray:
    """获取文本的向量表示"""
    if self._embedding_model is None:
        # 当前使用模拟向量（768 维）
        return np.random.rand(768).astype(np.float32)
    
    try:
        # 未来集成真实 Embedding 模型
        if hasattr(self._embedding_model, 'encode'):
            return self._embedding_model.encode([text])[0]
    except Exception as e:
        logger.error(f"Embedding 失败：{e}")
        return np.random.rand(768).astype(np.float32)
```

#### 4.2 余弦相似度计算与排序

```python
def search(self, query_vector: np.ndarray,
           filter_type: Optional[str] = "logic",
           limit: int = 5) -> List[Experience]:
    """语义检索 + 强过滤"""
    results = []
    
    # 1. 遍历所有经验
    for exp in self._experiences.values():
        # 2. 类型过滤（硬过滤）
        if filter_type and exp.experience_type != filter_type:
            continue
        
        # 3. 计算余弦相似度（软排序）
        if exp.embedding is not None:
            similarity = np.dot(query_vector, exp.embedding) / (
                np.linalg.norm(query_vector) * 
                np.linalg.norm(exp.embedding) + 1e-8
            )
            results.append((similarity, exp))
    
    # 4. 按相似度降序排序
    results.sort(key=lambda x: x[0], reverse=True)
    
    # 5. 返回 Top-K
    return [exp for _, exp in results[:limit]]
```

**余弦相似度公式**:
```
similarity = cos(θ) = (A · B) / (||A|| × ||B||)
```

**示例计算**:
```
查询："网络慢怎么办"
query_vector = [0.02, 0.85, -0.11, ...] (768 维)

经验 1: "检查路由器是否过热"
exp1_vector = [0.01, 0.82, -0.15, ...]
similarity1 = 0.92  ← 高度相关

经验 2: "重启路由器可以解决很多问题"
exp2_vector = [0.05, 0.75, -0.08, ...]
similarity2 = 0.85  ← 中等相关

经验 3: "清理桌面垃圾"
exp3_vector = [-0.12, 0.03, 0.95, ...]
similarity3 = 0.15  ← 不相关

返回：[exp1, exp2] (Top-2)
```

---

### 阶段 5️⃣：构建超级 Prompt

**代码位置**: [`three_libraries.py:823-854`](file://d:\AI\project\zulong_beta4\zulong\memory\three_libraries.py#L823-L854)

```python
def build_super_prompt(self, query: str,
                       skills: Optional[Dict] = None,
                       experiences: Optional[List] = None,
                       knowledge: Optional[List] = None) -> str:
    """构建超级 Prompt"""
    prompt_parts = []
    
    # 1. 添加技能部分
    if skills:
        prompt_parts.append(
            self.skill_store.to_prompt_context(list(skills.keys()))
        )
    
    # 2. 添加经验部分
    if experiences:
        prompt_parts.append(
            self.experience_store.to_prompt_context(experiences)
        )
    
    # 3. 添加知识部分
    if knowledge:
        prompt_parts.append(
            self.knowledge_store.to_prompt_context(knowledge)
        )
    
    # 4. 添加用户请求
    prompt_parts.append(f"\n## 用户请求\n{query}")
    
    # 5. 合并所有部分
    return "\n\n".join(prompt_parts)
```

**经验 Prompt 格式化**:
```python
def to_prompt_context(self, experiences: List[Experience]) -> str:
    """将经验转换为 Prompt 上下文"""
    if not experiences:
        return ""
    
    context_parts = ["## 相关经验"]
    for exp in experiences:
        status = "✅" if exp.success else "❌"
        context_parts.append(
            f"\n{status} [{exp.experience_type}] {exp.content}"
        )
    
    return "\n".join(context_parts)
```

**最终 Prompt 示例**:
```
## 系统能力

### navigation (导航与路径规划能力)
指令:
- 规划安全、高效的路径
- 实时避障
- 动态调整路线
安全规则:
- 禁止进入禁区
- 保持安全距离

### conversation (对话交互能力)
指令:
- 理解用户意图
- 提供有用的回答
- 保持礼貌和专业

## 相关经验

✅ [logic] 当用户抱怨网络慢时，应引导其检查路由器是否过热
✅ [logic] 网络设置优化：调整 DNS 服务器可以提升网速
❌ [failure] 路由器故障导致网络中断，需要更换新路由器

## 相关知识

[network] WiFi 2.4GHz 频段穿墙能力强，但速度较慢
  来源：网络优化手册

[network] 5GHz 频段速度快，但穿墙能力弱
  来源：网络优化手册

## 用户请求
网络慢怎么办？
```

---

### 阶段 6️⃣：发布事件到 L2

**代码位置**: [`async_scheduler.py:287-302`](file://d:\AI\project\zulong_beta4\zulong\l1b\async_scheduler.py#L287-L302)

```python
# 发布 SYSTEM_L2_COMMAND 事件
event_bus.publish(ZulongEvent(
    type=EventType.SYSTEM_L2_COMMAND,
    priority=EventPriority.NORMAL,
    source="L1-B-Async",
    payload={
        "task_id": task.task_id,
        "prompt": final_prompt,
        "status": "READY",
        "retrieval_time_ms": elapsed
    }
))
```

**事件总线传递**:
```
L1-B AsyncScheduler → EventBus → L2 InferenceEngine
```

---

## 📊 数据流与状态流转

### TaskItem 状态流转

```
PENDING (创建任务)
    ↓
    [放入 task_queue]
    ↓
RETRIEVING (开始检索)
    ↓
    [retrieve_all() 完成]
    ↓
READY (Prompt 构建完成)
    ↓
    [发布 SYSTEM_L2_COMMAND]
    ↓
EXECUTING (L2 正在推理)
    ↓
    [L2 完成推理]
    ↓
COMPLETED (任务完成)
```

### 向量检索数据流

```
用户查询："网络慢怎么办"
    ↓
rewrite_query() → "网络慢怎么办" (无需重写)
    ↓
_get_embedding() → query_vector (768 维)
    ↓
search(query_vector, filter_type="logic", limit=5)
    ↓
[遍历经验库]
    ├─ exp1: type="logic" ✓ → 计算相似度 0.92 → 保留
    ├─ exp2: type="logic" ✓ → 计算相似度 0.85 → 保留
    ├─ exp3: type="failure" ✗ → 过滤跳过
    ├─ exp4: type="logic" ✓ → 计算相似度 0.78 → 保留
    └─ exp5: type="success" ✗ → 过滤跳过
    ↓
排序：[exp1(0.92), exp4(0.78), exp2(0.65)]
    ↓
返回 Top-5: [exp1, exp4, exp2]
    ↓
to_prompt_context() → 格式化 Prompt
```

---

## 🎯 关键设计特点

### 1. 异步非阻塞

```python
# 主线程不等待检索完成
def handle_request(self, user_input: str) -> str:
    task = TaskItem(...)
    self.task_queue.put(task)
    
    # 异步执行，立即返回 task_id
    asyncio.run_coroutine_threadsafe(
        self._retrieve_and_inject(task),
        self._loop
    )
    
    return task_id  # 立即返回，不阻塞
```

### 2. 并行检索

```python
async def retrieve_all(self, query: str, ...) -> Dict[str, Any]:
    # 三个库并行检索（虽然当前是顺序调用，但框架支持并行）
    skills = self.skill_store.get_all()  # 0ms
    experiences = self.experience_store.search_by_text(...)  # ~50ms
    knowledge = await self.knowledge_store.search_by_text_async(...)  # ~100ms
    
    return {"skills": skills, "experiences": experiences, "knowledge": knowledge}
```

### 3. 类型过滤 + 向量相似度

```python
# 双重保障：先过滤，后排序
for exp in self._experiences.values():
    # 硬过滤：类型必须匹配
    if filter_type and exp.experience_type != filter_type:
        continue
    
    # 软排序：按相似度排序
    similarity = cosine_similarity(query_vector, exp.embedding)
    results.append((similarity, exp))

results.sort(key=lambda x: x[0], reverse=True)
```

### 4. 元数据完整传递

```python
@dataclass
class Experience:
    id: str
    content: str
    experience_type: str  # logic/failure/success/preference
    task_id: Optional[str]
    success: bool
    metadata: Dict[str, Any]  # tags, source, importance, etc.
    embedding: Optional[np.ndarray]
    timestamp: float
```

---

## 🔧 当前实现状态

### ✅ 已实现功能

1. **任务创建与异步检索**
   - TaskItem 数据结构
   - 异步非阻塞检索
   - 任务状态管理

2. **查询重写**
   - 预定义重写规则
   - 口语 → 技术术语转换

3. **三库并行检索**
   - 技能库（内存直读）
   - 经验库（向量检索）
   - 知识库（异步检索）

4. **向量检索**
   - 查询向量化
   - 余弦相似度计算
   - 类型过滤
   - Top-K 排序

5. **Prompt 构建**
   - 技能上下文
   - 经验上下文（带成功/失败标记）
   - 知识上下文（带来源）
   - 用户请求

6. **事件发布**
   - SYSTEM_L2_COMMAND 事件
   - 包含完整 Prompt 和任务信息

### ⚠️ 待完善功能

1. **真实 Embedding 模型**
   - 当前使用模拟向量（`np.random.rand(768)`）
   - 需要集成 BAAI/bge-small-zh-v1.5 等模型

2. **混合检索**
   - 当前仅支持向量检索
   - 可添加关键词 BM25 检索

3. **时间衰减因子**
   - 新经验权重更高
   - 老经验权重衰减

4. **多标签过滤**
   - 当前仅支持单类型过滤
   - 可支持多标签组合过滤

---

## 📝 完整调用链路示例

### 用户输入："网络慢怎么解决"

**1. 创建任务**
```python
task = TaskItem(
    task_id="a1b2c3d4",
    prompt="",
    status="PENDING",
    raw_input="网络慢怎么解决"
)
```

**2. 查询重写**
```python
tech_query = rewrite_query("网络慢怎么解决")
# 输出："网络慢怎么解决" (无需重写)
```

**3. 并行检索**
```python
results = await retrieve_all(
    query="网络慢怎么解决",
    experience_type="logic",
    experience_limit=5,
    knowledge_limit=10
)

# 返回:
{
    "skills": {coding, navigation, conversation, ...},
    "experiences": [
        Experience(content="检查路由器过热", similarity=0.92),
        Experience(content="调整 DNS 设置", similarity=0.85),
        Experience(content="重启路由器", similarity=0.78)
    ],
    "knowledge": [
        Knowledge(content="WiFi 2.4GHz vs 5GHz", source="网络手册")
    ]
}
```

**4. 构建 Prompt**
```python
final_prompt = build_super_prompt(
    query="网络慢怎么解决",
    skills=results["skills"],
    experiences=results["experiences"],
    knowledge=results["knowledge"]
)

# 输出:
"""
## 系统能力
### navigation (导航与路径规划能力)
指令:
- 规划安全、高效的路径
...

## 相关经验
✅ [logic] 检查路由器是否过热
✅ [logic] 调整 DNS 服务器可以提升网速
✅ [logic] 重启路由器可以解决很多问题

## 相关知识
[network] WiFi 2.4GHz 频段穿墙能力强，但速度较慢
  来源：网络优化手册

## 用户请求
网络慢怎么解决
"""
```

**5. 发布事件**
```python
event_bus.publish(ZulongEvent(
    type=EventType.SYSTEM_L2_COMMAND,
    payload={
        "task_id": "a1b2c3d4",
        "prompt": final_prompt,
        "status": "READY",
        "retrieval_time_ms": 150.5
    }
))
```

**6. L2 推理**
```python
# L2 接收事件，调用 Model Engine
response = model.generate(final_prompt)
# 输出："建议您检查路由器是否过热，可以尝试重启路由器..."
```

---

## 🎯 总结

### 核心流程

```
用户输入 → 任务创建 → 查询重写 → 三库检索 → 向量排序 → 
Prompt 构建 → 事件发布 → L2 推理 → 生成回复
```

### 关键组件

| 组件 | 职责 | 延迟 |
|------|------|------|
| AsyncL1BScheduler | 任务调度、异步检索 | ~150ms |
| ThreeLibraryManager | 三库并行检索 | ~100ms |
| ExperienceStore | 向量检索、类型过滤 | ~50ms |
| SkillStore | 内存技能读取 | 0ms |
| KnowledgeStore | 异步知识检索 | ~100ms |

### 向量作用

1. **语义理解**: 不依赖关键词匹配，理解"网络慢"和"网速卡"的语义相似性
2. **相似度排序**: 按余弦相似度排序，确保最相关的经验在前
3. **类型过滤**: 先过滤类型，再计算相似度，提高效率

### 下一步优化

1. 集成真实 Embedding 模型
2. 实现时间衰减因子
3. 支持多标签组合过滤
4. 添加混合检索（向量 + 关键词）

---

**文档完成时间**: 2026-03-29  
**测试脚本**: [`tests/test_l1b_vector_injection.py`](file://d:\AI\project\zulong_beta4\tests\test_l1b_vector_injection.py)
