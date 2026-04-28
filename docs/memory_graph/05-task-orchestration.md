# 05-复杂任务编排

> **阅读时间**: 15 分钟  
> **前置知识**: [01-图记忆架构概述](./01-architecture.md), [04-图注意力机制](./04-attention.md)  
> **相关文档**: [task_graph.py](../../zulong/memory/task_graph.py), [agent_orchestrator.py](../../zulong/agent/agent_orchestrator.py)

---

## 📋 目录

1. [复杂任务编排概述](#复杂任务编排概述)
2. [TaskGraph 与 MemoryGraph 集成](#taskgraph-与-memorygraph-集成)
3. [任务创建流程](#任务创建流程)
4. [任务执行与监控](#任务执行与监控)
5. [Agent 协作机制](#agent-协作机制)
6. [经验沉淀与回流](#经验沉淀与回流)
7. [任务编排示例](#任务编排示例)
8. [最佳实践](#最佳实践)

---

## 🎯 复杂任务编排概述

### 核心挑战

复杂任务面临的核心挑战：

1. **任务分解**: 如何将复杂任务拆解为可执行的子任务？
2. **上下文管理**: 如何在任务执行过程中保持相关上下文的注入？
3. **Agent 协作**: 如何协调多个 Agent 完成不同子任务？
4. **经验沉淀**: 如何将任务执行过程中的经验转化为可复用的知识？

### MemoryGraph 的解决方案

```
复杂任务编排架构:

用户："帮我写一个 Python 脚本，爬取天气数据并保存到 Excel"
         ↓
L2 推理引擎
├─ 创建 TaskGraph
│  ├─ 根任务：o1_1 "爬取天气数据并保存到 Excel"
│  ├─ 子任务：o1_1_1 "分析需求"
│  ├─ 子任务：o1_1_2 "编写爬虫代码"
│  ├─ 子任务：o1_1_3 "编写 Excel 导出代码"
│  └─ 子任务：o1_1_4 "测试运行"
│
└─ 同步到 MemoryGraph
   ├─ 添加 TASK 节点 (o1_1, o1_1_1, ...)
   ├─ 添加 HIERARCHY 边 (父子关系)
   └─ 设置 backend_ref (指向原始 TaskGraph)
         ↓
任务执行过程中
├─ 每个子任务完成 → 更新节点 activation
├─ 引用知识文档 → 添加 REFERENCE 边
└─ 遇到问题 → 创建 EXPERIENCE 节点（经验沉淀）
         ↓
任务完成后
└─ 经验回流到 MemoryGraph（形成 ASSOCIATION 边）
```

**核心优势**:
- ✅ **结构化分解**: 复杂任务自动拆解为可执行子任务
- ✅ **上下文关联**: 任务、对话、知识、文件自动关联
- ✅ **经验沉淀**: 执行过程中的经验自动沉淀为可复用知识
- ✅ **Agent 协作**: 多个 Agent 分工协作，高效完成复杂任务

---

## 🔗 TaskGraph 与 MemoryGraph 集成

### 架构关系

```
┌─────────────────────────────────────────────────────────┐
│  TaskGraph (任务图)                                      │
├─────────────────────────────────────────────────────────┤
│  - 负责任务分解                                          │
│  - 管理子任务状态                                        │
│  - 协调 Agent 执行                                       │
│  - 存储任务结果                                          │
│                                                          │
│  数据结构：                                              │
│  TaskNode {                                              │
│    task_id: str                                          │
│    description: str                                      │
│    status: TaskStatus                                    │
│    subtasks: List[TaskNode]                              │
│    files: List[str]                                      │
│    result: Optional[str]                                 │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
         ↓ TaskGraphAdapter (只读适配器)
┌─────────────────────────────────────────────────────────┐
│  MemoryGraph (记忆图谱)                                  │
├─────────────────────────────────────────────────────────┤
│  - 负责任务索引                                          │
│  - 关联任务与对话/知识/文件                              │
│  - 管理任务经验沉淀                                      │
│  - 提供任务检索                                          │
│                                                          │
│  数据结构：                                              │
│  GraphNode {                                             │
│    node_id: "task:o1_1"                                  │
│    node_type: NodeType.TASK                              │
│    backend_ref: "taskgraph:o1_1"                         │
│    metadata: {...}                                       │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
```

### TaskGraphAdapter 实现

```python
# task_graph_adapter.py
class TaskGraphAdapter:
    """
    TaskGraph → MemoryGraph 的只读适配器
    
    职责:
    1. 监听 TaskGraph 变化
    2. 同步到 MemoryGraph（添加/更新 TASK 节点）
    3. 建立任务与其他节点的关联
    """
    
    def __init__(self, memory_graph: MemoryGraph):
        self.memory_graph = memory_graph
        self._task_graph: Optional[TaskGraph] = None
    
    def attach_task_graph(self, task_graph: TaskGraph):
        """绑定 TaskGraph 实例"""
        self._task_graph = task_graph
        
        # 同步现有任务
        for task_node in task_graph.all_tasks():
            self._sync_task_node(task_node)
    
    def _sync_task_node(self, task_node: TaskNode):
        """
        同步任务节点到 MemoryGraph
        
        策略:
        1. 添加 TASK 节点
        2. 添加 HIERARCHY 边（父子关系）
        3. 添加 REFERENCE 边（任务 → 相关文件）
        """
        # 1. 添加 TASK 节点
        node = GraphNode(
            node_id=f"task:{task_node.task_id}",
            node_type=NodeType.TASK,
            label=task_node.description[:50],
            backend_ref=f"taskgraph:{task_node.task_id}",
            metadata={
                "task_id": task_node.task_id,
                "status": task_node.status.value,
                "importance": "important",  # 任务默认重要度
            }
        )
        self.memory_graph.add_node(node)
        
        # 2. 添加 HIERARCHY 边（父子关系）
        if task_node.parent_task_id:
            self.memory_graph.add_edge(
                f"task:{task_node.parent_task_id}",
                f"task:{task_node.task_id}",
                EdgeType.HIERARCHY,
                protected=True  # 结构性边
            )
        
        # 3. 添加 REFERENCE 边（任务 → 相关文件）
        for file_path in task_node.files:
            file_node_id = f"file:{file_path}"
            if file_node_id in self.memory_graph._nodes:
                self.memory_graph.add_edge(
                    f"task:{task_node.task_id}",
                    file_node_id,
                    EdgeType.REFERENCE,
                    weight=0.8
                )
```

---

## 📝 任务创建流程

### 完整流程

```
用户输入："帮我写一个 Python 脚本，爬取天气数据并保存到 Excel"
         ↓
1. L2 推理引擎分析意图
   - 识别为复杂任务
   - 决定创建 TaskGraph
         ↓
2. TaskGraph 创建
   - 根任务：o1_1 "爬取天气数据并保存到 Excel"
   - 子任务自动分解:
     ├─ o1_1_1 "分析需求"
     ├─ o1_1_2 "编写爬虫代码"
     ├─ o1_1_3 "编写 Excel 导出代码"
     └─ o1_1_4 "测试运行"
         ↓
3. TaskGraphAdapter 同步到 MemoryGraph
   - 添加 TASK 节点 (o1_1, o1_1_1, ...)
   - 添加 HIERARCHY 边 (父子关系)
   - 设置 backend_ref (指向原始 TaskGraph)
         ↓
4. MemoryGraph 建立关联
   - 从当前对话追溯相关知识
   - 添加 REFERENCE 边 (任务 → 知识)
   - 添加 REFERENCE 边 (任务 → 相关文件)
         ↓
5. 注入上下文
   - 任务结构
   - 相关知识文档
   - 相关文件
         ↓
6. Agent 执行子任务
```

### 代码实现

```python
# task_creator.py
class TaskCreator:
    """任务创建器"""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        memory_graph: MemoryGraph,
        task_adapter: TaskGraphAdapter,
    ):
        self.task_graph = task_graph
        self.memory_graph = memory_graph
        self.task_adapter = task_adapter
    
    async def create_task(self, user_input: str) -> str:
        """
        创建复杂任务
        
        Args:
            user_input: 用户输入
        
        Returns:
            任务 ID
        """
        # 1. 使用 LLM 分解任务
        task_decomposition = await self._decompose_task(user_input)
        
        # 2. 创建 TaskGraph
        root_task = self.task_graph.add_task(
            description=task_decomposition["root_description"],
            parent_task_id=None
        )
        
        # 3. 创建子任务
        for subtask_desc in task_decomposition["subtasks"]:
            self.task_graph.add_task(
                description=subtask_desc,
                parent_task_id=root_task.task_id
            )
        
        # 4. 同步到 MemoryGraph
        self.task_adapter.attach_task_graph(self.task_graph)
        
        # 5. 建立知识关联
        await self._link_knowledge(root_task)
        
        return root_task.task_id
    
    async def _decompose_task(self, user_input: str) -> Dict:
        """使用 LLM 分解任务"""
        prompt = f"""
请将以下复杂任务分解为可执行的子任务：

用户输入：{user_input}

返回格式：
{{
    "root_description": "根任务描述",
    "subtasks": [
        "子任务 1 描述",
        "子任务 2 描述",
        ...
    ]
}}
"""
        # 调用 LLM...
        pass
    
    async def _link_knowledge(self, root_task: TaskNode):
        """建立任务与知识的关联"""
        # 1. 检索相关知识
        knowledge_nodes = await self.memory_graph.retrieve_context(
            query_text=root_task.description,
            top_k=5
        )
        
        # 2. 添加 REFERENCE 边
        for knowledge in knowledge_nodes:
            if knowledge["node_type"] == "knowledge":
                self.memory_graph.add_edge(
                    f"task:{root_task.task_id}",
                    knowledge["node_id"],
                    EdgeType.REFERENCE,
                    weight=0.7
                )
```

---

## 🔄 任务执行与监控

### 任务执行流程

```python
# task_executor.py
class TaskExecutor:
    """任务执行器"""
    
    def __init__(
        self,
        task_graph: TaskGraph,
        memory_graph: MemoryGraph,
        agent_orchestrator: AgentOrchestrator,
    ):
        self.task_graph = task_graph
        self.memory_graph = memory_graph
        self.agent_orchestrator = agent_orchestrator
    
    async def execute_task(self, task_id: str):
        """
        执行任务
        
        流程:
        1. 获取任务节点
        2. 获取子任务列表
        3. 依次执行子任务
        4. 更新任务状态
        5. 更新 MemoryGraph 激活值
        """
        task_node = self.task_graph.get_task(task_id)
        if not task_node:
            raise ValueError(f"Task {task_id} not found")
        
        subtasks = self.task_graph.get_subtasks(task_id)
        
        # 依次执行子任务
        for subtask in subtasks:
            # 1. 更新子任务状态为进行中
            self.task_graph.update_task_status(
                subtask.task_id,
                TaskStatus.IN_PROGRESS
            )
            
            # 2. 更新 MemoryGraph 激活值
            self.memory_graph.update_node_activation(
                f"task:{subtask.task_id}",
                1.0
            )
            
            # 3. 执行子任务
            result = await self.agent_orchestrator.execute_subtask(subtask)
            
            # 4. 更新子任务状态为完成
            self.task_graph.update_task_status(
                subtask.task_id,
                TaskStatus.COMPLETED
            )
            self.task_graph.set_task_result(
                subtask.task_id,
                result
            )
            
            # 5. 创建经验节点（如果有价值）
            if result.has_valuable_experience:
                await self._create_experience_node(subtask, result)
        
        # 6. 更新根任务状态
        self.task_graph.update_task_status(
            task_id,
            TaskStatus.COMPLETED
        )
    
    async def _create_experience_node(self, subtask: TaskNode, result: Any):
        """创建经验节点"""
        experience = GraphNode(
            node_id=f"experience:{subtask.task_id}",
            node_type=NodeType.EXPERIENCE,
            label=f"经验：{subtask.description}",
            metadata={
                "task_id": subtask.task_id,
                "content": result.experience_summary,
                "importance": "important",
            }
        )
        self.memory_graph.add_node(experience)
        
        # 建立关联
        self.memory_graph.add_edge(
            f"task:{subtask.task_id}",
            f"experience:{subtask.task_id}",
            EdgeType.CAUSAL,
            weight=0.8
        )
```

---

## 🤖 Agent 协作机制

### 多 Agent 协作架构

```
┌─────────────────────────────────────────────────────────┐
│  L2 推理引擎 (主实例)                                    │
├─────────────────────────────────────────────────────────┤
│  当前聚焦：task:o1_1                                     │
│  种子节点：[task:o1_1, dialogue:42, dialogue:41]         │
│  BFS 扩散激活 → 发现相关节点                             │
│  ├─ task:o1_1_1 (子任务，HIERARCHY 边)                   │
│  ├─ file:weather_api.py (文件引用，REFERENCE 边)         │
│  ├─ knowledge:python_requests (知识，SEMANTIC 边)        │
│  └─ experience:crawl_error_handling (经验，ASSOCIATION 边)│
│  注入上下文 → LLM 推理                                   │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  L1-B 调度器                                             │
├─────────────────────────────────────────────────────────┤
│  - 监控 L2 状态                                          │
│  - 空闲时触发后台复盘                                    │
│  - 紧急情况下切换 KV Cache                               │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│  L0-A 执行引擎                                           │
├─────────────────────────────────────────────────────────┤
│  - 执行具体工具调用                                      │
│  - 文件操作                                              │
│  - 网络请求                                              │
└─────────────────────────────────────────────────────────┘
```

### Agent 角色分工

```python
# agent_orchestrator.py
class AgentOrchestrator:
    """
    Agent 编排器
    
    角色分工:
    - L2 推理引擎：负责任务分解、上下文管理
    - L1-B 调度器：负责资源调度、后台复盘
    - L0-A 执行引擎：负责具体执行
    """
    
    def __init__(
        self,
        l2_engine: L2InferenceEngine,
        l1b_scheduler: L1BScheduler,
        l0a_executor: L0AExecutor,
    ):
        self.l2 = l2_engine
        self.l1b = l1b_scheduler
        self.l0a = l0a_executor
    
    async def execute_subtask(self, subtask: TaskNode) -> Any:
        """
        执行子任务
        
        流程:
        1. L2 分析子任务，确定所需工具和知识
        2. L1-B 调度资源（如需要）
        3. L0-A 执行具体操作
        4. L2 汇总结果
        """
        # 1. L2 分析
        analysis = await self.l2.analyze_subtask(subtask)
        
        # 2. 准备上下文
        context = await self._prepare_context(analysis)
        
        # 3. L0-A 执行
        execution_result = await self.l0a.execute(
            tools=analysis.required_tools,
            context=context
        )
        
        # 4. L2 汇总
        summary = await self.l2.summarize_result(execution_result)
        
        return summary
    
    async def _prepare_context(self, analysis: Any) -> Dict:
        """准备上下文"""
        # 从 MemoryGraph 检索相关知识、经验、文件
        context_nodes = await self.l2.memory_graph.retrieve_context(
            query_text=analysis.query,
            top_k=10
        )
        
        # 注入上下文
        return self.l2.build_context(context_nodes)
```

---

## 💎 经验沉淀与回流

### 经验沉淀流程

```python
# experience_extractor.py
class ExperienceExtractor:
    """经验提取器"""
    
    def __init__(self, memory_graph: MemoryGraph):
        self.memory_graph = memory_graph
    
    async def extract_from_task(self, task_id: str):
        """
        从任务执行过程中提取经验
        
        流程:
        1. 分析任务执行日志
        2. 识别有价值的经验
        3. 创建 EXPERIENCE 节点
        4. 建立 ASSOCIATION 边
        """
        # 1. 获取任务执行日志
        logs = await self._get_task_logs(task_id)
        
        # 2. 使用 LLM 提取经验
        experience_summary = await self._llm_extract_experience(logs)
        
        # 3. 创建 EXPERIENCE 节点
        experience_node = GraphNode(
            node_id=f"experience:{task_id}",
            node_type=NodeType.EXPERIENCE,
            label=f"经验：{task_id}",
            metadata={
                "task_id": task_id,
                "content": experience_summary,
                "importance": "important",
            }
        )
        self.memory_graph.add_node(experience_node)
        
        # 4. 建立关联
        self.memory_graph.add_edge(
            f"task:{task_id}",
            f"experience:{task_id}",
            EdgeType.CAUSAL,
            weight=0.8
        )
        
        # 5. 与其他经验建立 ASSOCIATION 边（赫布学习）
        similar_experiences = await self._find_similar_experiences(experience_summary)
        for similar_exp in similar_experiences:
            self.memory_graph.add_edge(
                f"experience:{task_id}",
                similar_exp.node_id,
                EdgeType.ASSOCIATION,
                weight=0.5
            )
```

### 经验回流机制

```
任务执行过程:
1. 子任务 o1_1_2 "编写爬虫代码" 遇到问题
   - 错误：SSL 证书验证失败
   
2. L2 推理引擎解决问题
   - 解决方案：添加 verify=False 参数
   
3. 创建经验节点
   - experience:o1_1_2: "爬虫 SSL 证书问题处理"
   
4. 经验回流到 MemoryGraph
   - 添加 EXPERIENCE 节点
   - 与 task:o1_1_2 建立 CAUSAL 边
   - 与其他 SSL 相关经验建立 ASSOCIATION 边
   
5. 未来任务遇到类似问题
   - BFS 扩散激活 → 发现 experience:o1_1_2
   - 自动注入上下文
   - AI 快速解决问题
```

---

## 💡 任务编排示例

### 示例 1：数据分析任务

```
用户："帮我分析这个 Excel 文件中的销售数据，找出趋势"
     ↓
1. TaskGraph 创建
   ├─ 根任务：o1_1 "分析 Excel 销售数据"
   ├─ 子任务：o1_1_1 "读取 Excel 文件"
   ├─ 子任务：o1_1_2 "数据清洗"
   ├─ 子任务：o1_1_3 "数据分析"
   └─ 子任务：o1_1_4 "生成可视化图表"
     ↓
2. MemoryGraph 同步
   - 添加 TASK 节点
   - 添加 HIERARCHY 边
   - 添加 REFERENCE 边 (任务 → sales_data.xlsx)
     ↓
3. 执行过程
   - o1_1_1: 使用 pandas 读取 Excel
   - o1_1_2: 处理缺失值、异常值
   - o1_1_3: 计算月度增长率、同比环比
   - o1_1_4: 生成折线图、柱状图
     ↓
4. 经验沉淀
   - experience:o1_1_2: "Excel 缺失值处理技巧"
   - experience:o1_1_3: "销售数据趋势分析方法"
     ↓
5. 结果交付
   - 分析报告
   - 可视化图表
   - 经验文档
```

### 示例 2：Web 开发任务

```
用户："帮我创建一个简单的博客网站，支持文章发布和评论"
     ↓
1. TaskGraph 创建
   ├─ 根任务：o1_1 "创建博客网站"
   ├─ 子任务：o1_1_1 "需求分析"
   ├─ 子任务：o1_1_2 "技术选型"
   ├─ 子任务：o1_1_3 "数据库设计"
   ├─ 子任务：o1_1_4 "后端 API 开发"
   ├─ 子任务：o1_1_5 "前端页面开发"
   └─ 子任务：o1_1_6 "测试部署"
     ↓
2. MemoryGraph 同步
   - 添加 TASK 节点
   - 添加 HIERARCHY 边
   - 添加 REFERENCE 边 (任务 → 相关知识/文件)
     ↓
3. 执行过程
   - o1_1_1: 确定功能需求（文章发布、评论、用户系统）
   - o1_1_2: 选择 Flask + SQLite + Vue.js
   - o1_1_3: 设计 users/posts/comments 表
   - o1_1_4: 开发 RESTful API
   - o1_1_5: 开发前端页面
   - o1_1_6: 单元测试、部署到服务器
     ↓
4. 经验沉淀
   - experience:o1_1_3: "博客数据库设计最佳实践"
   - experience:o1_1_4: "Flask RESTful API 开发技巧"
   - experience:o1_1_5: "Vue.js 组件化开发经验"
     ↓
5. 结果交付
   - 完整的博客网站源码
   - 部署文档
   - 经验文档集合
```

---

## 📚 最佳实践

### 1. 任务分解原则

```python
# ✅ 好的任务分解
subtasks = [
    "读取 Excel 文件",           # 单一职责
    "数据清洗",                 # 边界清晰
    "数据分析",                 # 可独立验证
    "生成可视化图表"            # 结果明确
]

# ❌ 不好的任务分解
subtasks = [
    "处理数据并生成报告",       # 职责不清
    "分析和可视化",             # 边界模糊
]
```

### 2. 经验沉淀时机

```python
# 应该沉淀经验的场景:
- ✅ 遇到并解决了新问题
- ✅ 发现了最佳实践
- ✅ 性能优化有显著效果
- ✅ 踩坑并找到解决方案

# 不需要沉淀经验的场景:
- ❌ 常规操作，无特殊技巧
- ❌ 直接调用库函数，无额外处理
- ❌ 简单重复性工作
```

### 3. 关联建立策略

```python
# 强关联（weight >= 0.7）
- HIERARCHY 边（父子关系）
- CAUSAL 边（因果关系）
- REFERENCE 边（直接引用）

# 弱关联（weight < 0.7）
- SEMANTIC 边（语义相似）
- ASSOCIATION 边（赫布学习产生）
```

### 4. 任务监控指标

```python
# 关键指标
metrics = {
    "task_completion_rate": 0.95,      # 任务完成率
    "avg_subtask_duration": 120,       # 平均子任务耗时（秒）
    "experience_extraction_rate": 0.3, # 经验提取率
    "agent_collaboration_efficiency": 0.88,  # Agent 协作效率
}
```

---

## 🎯 总结

MemoryGraph 的**复杂任务编排机制**通过 TaskGraph 集成实现了：

1. ✅ **结构化分解**: 复杂任务自动拆解为可执行子任务
2. ✅ **上下文关联**: 任务、对话、知识、文件自动关联
3. ✅ **经验沉淀**: 执行过程中的经验自动沉淀为可复用知识
4. ✅ **Agent 协作**: 多个 Agent 分工协作，高效完成复杂任务
5. ✅ **经验回流**: 经验通过 ASSOCIATION 边形成知识网络

**核心优势**:
- ✅ **可追溯**: 任务执行过程完整记录，可复盘优化
- ✅ **可复用**: 经验沉淀为知识，未来任务可直接复用
- ✅ **可扩展**: 支持新增任务类型、Agent 角色
- ✅ **自组织**: 赫布学习自动增强常用经验关联

**下一步**:
- [06-快速入门指南](./06-quickstart.md) - 快速上手编码
- [07-FAQ 与故障排查](./07-faq.md) - 解决实际问题

---

**最后更新**: 2026-04-19  
**维护者**: 祖龙系统核心开发团队
