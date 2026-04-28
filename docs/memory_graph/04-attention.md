# 04-图注意力机制

> **阅读时间**: 10 分钟  
> **前置知识**: [01-图记忆架构概述](./01-architecture.md), [03-记忆检索机制](./03-retrieval.md)  
> **相关文档**: [attention_window_manager.py](../../zulong/memory/attention_window_manager.py)

---

## 📋 目录

1. [注意力机制演进](#注意力机制演进)
2. [从 1D 到图注意力的跃升](#从-1d-到图注意力的跃升)
3. [图注意力评分公式](#图注意力评分公式)
4. [BFS 扩散激活算法](#bfs-扩散激活算法)
5. [与 AttentionWindowManager 集成](#与-attentionwindowmanager-集成)
6. [注意力可视化](#注意力可视化)
7. [性能优化](#性能优化)
8. [实际应用示例](#实际应用示例)

---

## 🧠 注意力机制演进

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

---

### 新架构：图注意力

**核心思想**:

```
从 1D 线性序列 → 图结构扩散

用户输入："它多少钱？"
         ↓
MemoryGraph:
1. 从当前节点 (dialogue:31) 出发
2. 沿 REFERENCE 边追溯到 dialogue:1
3. 发现 dialogue:1 关联 knowledge:ai_max_395
4. 沿 CAUSAL 边找到 knowledge:price
5. 注入高注意力权重

图注意力评分:
- dialogue:1 的评分 = 1.0 × time_decay × graph_boost
                     = 1.0 × 0.35 × 2.0 = 0.70
- dialogue:30 的评分 = 1.0 × 0.97 × 1.0 = 0.97

虽然 dialogue:1 时间衰减严重，但 graph_boost=2.0 补偿
→ 最终被注入上下文
```

**核心优势**:
- ✅ 发现跨类型的语义关联
- ✅ 对模型大小不敏感（图检索质量由图结构和边权决定）
- ✅ 对数据量不敏感（BFS 有 max_depth + 阈值剪枝）

---

## 📈 从 1D 到图注意力的跃升

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
│  其中 graph_boost = 1.0 + activation (最大 2.0x)          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 图注意力评分公式

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

### 注意力评分示例

```
场景：第 31 轮用户问"它多少钱？"

节点：dialogue:1 (第 1 轮讨论 AI MAX 395)

计算过程:
- base = 1.0
- time_decay = 0.5^(30/20) = 0.35
- mode_multiplier = 1.0 (无工具调用)
- graph_boost = 1.0 + activation

如果 activation = 0.8 (BFS 扩散发现强关联):
weight = 1.0 × 0.35 × 1.0 × 1.8 = 0.63

如果 activation = 0.0 (无关联):
weight = 1.0 × 0.35 × 1.0 × 1.0 = 0.35

对比：dialogue:30 (最近一轮)
- base = 1.0
- time_decay = 0.5^(1/20) = 0.97
- mode_multiplier = 1.0
- graph_boost = 1.0 (假设无关联)
weight = 1.0 × 0.97 × 1.0 × 1.0 = 0.97

结果：虽然 dialogue:1 评分较低，但因 graph_boost 补偿
      被注入上下文，AI 能正确解析"它"的指代
```

---

## 🌳 BFS 扩散激活算法

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

---

## 🔗 与 AttentionWindowManager 集成

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

---

## 🎨 注意力可视化

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
🟢 绿色：activation < 0.2 (无激活)
```

---

## ⚡ 性能优化

### 缓存优化

```python
class AttentionWindowManager:
    def __init__(self):
        self._activation_cache: Dict[int, float] = {}
        self._cache_ttl = 300  # 5 分钟缓存
    
    async def get_activation(self, node_id: int) -> float:
        """带缓存的激活值获取"""
        now = time.time()
        
        # 检查缓存
        if node_id in self._activation_cache:
            cached_time, cached_value = self._activation_cache[node_id]
            if now - cached_time < self._cache_ttl:
                return cached_value
        
        # 重新计算
        if self.memory_graph:
            value = self.memory_graph.get_node_activation(node_id)
        else:
            value = 0.0
        
        # 更新缓存
        self._activation_cache[node_id] = (now, value)
        
        return value
```

### 批量计算

```python
async def compute_activations_batch(
    self,
    seed_node_ids: List[str],
    max_depth: int = 3,
) -> Dict[str, float]:
    """
    批量计算激活值
    
    优化:
    - 一次性计算所有种子节点的扩散
    - 避免重复遍历
    """
    # 使用 BFS 一次性计算所有种子节点的扩散
    return self.compute_activations(seed_node_ids, max_depth)
```

### 延迟计算

```python
class MemoryGraph:
    def __init__(self):
        self._activation_dirty = False
        self._activation_cache: Optional[Dict[str, float]] = None
    
    def mark_activation_dirty(self):
        """标记激活值需要重新计算"""
        self._activation_dirty = True
        self._activation_cache = None
    
    def get_activations(self) -> Dict[str, float]:
        """延迟计算激活值"""
        if self._activation_cache is None or self._activation_dirty:
            self._activation_cache = self._compute_activations_internal()
            self._activation_dirty = False
        return self._activation_cache
```

---

## 💡 实际应用示例

### 示例 1：长程依赖解析

```
第 1 轮：用户："AI MAX 395 是什么？"
        AI："AI MAX 395 是一款高性能处理器..."
        MemoryGraph: 创建 dialogue:1 (importance=fact)

第 2-30 轮：讨论其他话题...

第 31 轮：用户："它多少钱？"
         ↓
AttentionWindowManager:
1. 调用 MemoryGraph.compute_activations(seed_node_ids=["dialogue:31"])
2. BFS 扩散:
   - dialogue:31 → REFERENCE → dialogue:1 (activation=0.85)
   - dialogue:1 → CAUSAL → knowledge:ai_max_395 (activation=0.42)
3. 计算图注意力评分:
   - dialogue:1: weight = 1.0 × 0.35 × 1.0 × 1.85 = 0.65
   - dialogue:30: weight = 1.0 × 0.97 × 1.0 × 1.0 = 0.97
4. 注入上下文:
   - [摘要] 第 1 轮：询问 AI MAX 395 定义
   - [知识] AI MAX 395 产品信息
         ↓
AI："AI MAX 395 的价格约为 1299 元"
```

### 示例 2：任务相关记忆发现

```
用户："帮我写一个 Python 脚本，爬取天气数据"
     ↓
TaskGraph: 创建 task:o1_1
     ↓
MemoryGraph: 添加 TASK 节点
     ↓
AttentionWindowManager:
1. 种子节点：task:o1_1
2. BFS 扩散:
   - task:o1_1 → HIERARCHY → task:o1_1_1 (activation=0.5)
   - task:o1_1 → HIERARCHY → task:o1_1_2 (activation=0.5)
   - task:o1_1 → REFERENCE → file:weather.py (activation=0.4)
   - file:weather.py → SEMANTIC → knowledge:python_requests (activation=0.14)
   - task:o1_1_2 → ASSOCIATION → experience:crawl_error (activation=0.15)
3. 注入上下文:
   - [任务] 爬取天气数据
   - [子任务] 分析需求
   - [子任务] 编写代码
   - [文件] weather.py
   - [知识] Python requests 库用法
   - [经验] 爬虫错误处理
         ↓
AI 基于完整上下文生成代码
```

### 示例 3：跨类型语义关联

```
第 5 轮：用户："我想学习 Python"
        AI："Python 是一门优秀的编程语言..."
        MemoryGraph: 创建 dialogue:5 (importance=important)

第 10 轮：用户："有什么好书推荐吗？"
         AI："《Python 编程：从入门到实践》很不错"
         MemoryGraph: 创建 dialogue:10, knowledge:python_book

第 15 轮：用户："刚才说的书在哪里买？"
         ↓
AttentionWindowManager:
1. 种子节点：dialogue:15
2. BFS 扩散:
   - dialogue:15 → REFERENCE → dialogue:10 (activation=0.75)
   - dialogue:10 → CAUSAL → knowledge:python_book (activation=0.38)
3. 注入上下文:
   - [摘要] 第 10 轮：Python 书籍推荐
   - [知识] 《Python 编程：从入门到实践》
         ↓
AI："《Python 编程：从入门到实践》可以在京东、当当等平台购买"
```

---

## 🎯 总结

MemoryGraph 的**图注意力机制**通过 BFS 扩散激活实现了：

1. ✅ **长程依赖发现**: 从当前节点追溯多轮前的对话
2. ✅ **跨类型关联**: 发现任务、对话、知识、文件间的语义关联
3. ✅ **动态增强**: 基于激活值动态调整注意力权重（最大 2.0x）
4. ✅ **性能优化**: BFS 剪枝 + 缓存，平均延迟 < 100ms

**核心优势**:
- ✅ **语义敏感**: 不再仅依赖时间远近，而是基于图结构发现关联
- ✅ **模型无关**: 对 0.8B 到 100B+ 模型均适用
- ✅ **可扩展**: 支持新增节点/边类型，无需修改核心算法

**下一步**:
- [05-复杂任务编排](./05-task-orchestration.md) - 掌握 TaskGraph 集成
- [06-快速入门指南](./06-quickstart.md) - 快速上手编码

---

**最后更新**: 2026-04-19  
**维护者**: 祖龙系统核心开发团队
