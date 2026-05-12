# 02-记忆分类与标签体系

> **阅读时间**: 10 分钟  
> **前置知识**: [01-图记忆架构概述](./01-architecture.md)  
> **相关文档**: [memory_graph.py](../../zulong/memory/memory_graph.py)

---

## 📋 目录

1. [多维标签体系概述](#多维标签体系概述)
2. [温度标签 (Temperature)](#温度标签-temperature)
3. [重要度标签 (Importance)](#重要度标签-importance)
4. [时间段标签 (Time Scope)](#时间段标签-time-scope)
5. [标签的正交性](#标签的正交性)
6. [重要度动态提升](#重要度动态提升)
7. [差异化衰减机制](#差异化衰减机制)
8. [实际应用示例](#实际应用示例)

---

## 🏷️ 多维标签体系概述

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

---

## 🌡️ 温度标签 (Temperature)

### 温度定义

温度标签反映节点的**访问频率和最近访问时间**，用于检索路由。

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

### 温度缓存（前端展示用）

虽然温度是动态计算的，但为了前端展示性能，会在 metadata 中缓存：

```python
def update_temperature(self, node_id: str) -> Optional[Temperature]:
    """
    动态计算温度并同步写入 metadata 缓存（前端展示用）
    
    Returns:
        Temperature 枚举值
    """
    temp = self.get_temperature(node_id)
    if temp is not None:
        node = self._nodes[node_id]
        node.metadata["temperature"] = temp.value
    return temp
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

# 2. 可视化展示（前端用不同颜色标记温度）
# HOT:   红色 🔴
# WARM:  黄色 🟡
# COLD:  蓝色 🔵
```

---

## ⭐ 重要度标签 (Importance)

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
# 重要度有序映射（值越大越重要）
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
# graph_adapters.py
class DialogueAdapter:
    # 重要度检测规则表
    _IMPORTANCE_RULES = [
        (r"我叫.*", Importance.IDENTITY),              # 姓名
        (r"我今年.*岁", Importance.IDENTITY),          # 年龄
        (r"帮我记住.*", Importance.MUST_REMEMBER),     # 显式要求
        (r"我的 (电话 | 手机号 | 地址|生日).*", Importance.FACT),  # 事实信息
        (r"明天.*记得提醒我", Importance.IMPORTANT),   # 承诺/提醒
        (r"嗯 | 好 | 哦 | 好的 | 知道了", Importance.TRIVIAL),  # 闲聊
    ]
    
    def _detect_importance(self, text: str) -> Tuple[Importance, List[str]]:
        """
        根据文本内容自动检测重要度
        
        Returns:
            (importance, matched_reasons)
        """
        matched_reasons = []
        max_importance = Importance.NORMAL
        
        for pattern, importance in self._IMPORTANCE_RULES:
            if re.search(pattern, text, re.IGNORECASE):
                matched_reasons.append(f"匹配规则：{pattern}")
                if _IMPORTANCE_ORDER[importance] > _IMPORTANCE_ORDER[max_importance]:
                    max_importance = importance
        
        return max_importance, matched_reasons
    
    async def add_round(self, user_input, ai_response):
        # 创建节点
        node = GraphNode(...)
        
        # 自动检测重要度
        importance, reasons = self._detect_importance(user_input)
        node.metadata["importance"] = importance.value
        if reasons:
            node.metadata["detected_entities"] = reasons
        
        self.memory_graph.add_node(node)
```

### 重要度半衰期

不同重要度的节点使用不同的衰减半衰期：

```python
# 重要度 → 衰减半衰期（小时）
_IMPORTANCE_HALF_LIFE = {
    Importance.TRIVIAL: 6.0,            # 6 小时（快速遗忘）
    Importance.NORMAL: 24.0,            # 24 小时（默认）
    Importance.IDENTITY: 720.0,         # 30 天（长期记忆）
    Importance.FACT: 360.0,             # 15 天（中期记忆）
    Importance.IMPORTANT: 168.0,        # 7 天（周级别）
    Importance.MUST_REMEMBER: float('inf'),  # 永不衰减
}
```

---

## ⏰ 时间段标签 (Time Scope)

### 时间段定义

时间段标签是**查询时动态计算**的，用于并行检索的互斥过滤。

```python
# 不存储在 metadata 中，查询时计算
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

**关键设计**: 时间段标签用于**互斥过滤**，避免热数据被 FAISS 重复检索。

---

## 🔀 标签的正交性

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

---

## 📈 重要度动态提升

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

### LLM 审查提升

```python
# llm_memory_reviewer.py
class LLMMemoryReviewer:
    async def review_before_evict(self, memories, usage_ratio, target_free):
        """
        LLM 审查记忆重要度
        
        审查维度:
        - 是否包含身份信息
        - 是否包含事实信息
        - 是否是用户显式要求
        - 是否与当前任务强相关
        
        审查决策:
        - KEEP: 保留并提升重要度
        - DISCARD: 允许删除
        - COMPRESS: 压缩内容但保留
        - PROMOTE: 提升重要度
        - MERGE: 合并到相关记忆
        """
```

---

## ⏳ 差异化衰减机制

### 边权衰减公式

```python
def decay_and_prune(self):
    """衰减非结构性边权，移除弱连接和孤立节点"""
    now = time.time()
    ln2 = math.log(2)
    
    for src, tgt, data in self._graph.edges(data=True):
        if data.get("protected"):
            continue
        
        # 获取两端节点的重要度，取更高的（决定半衰期）
        imp_src = self.get_importance(src) or Importance.NORMAL
        imp_tgt = self.get_importance(tgt) or Importance.NORMAL
        higher_imp = max(imp_src, imp_tgt, key=lambda x: _IMPORTANCE_ORDER.get(x, 1))
        
        # 获取半衰期
        half_life = _IMPORTANCE_HALF_LIFE.get(higher_imp, 24.0)
        if half_life == float('inf'):  # MUST_REMEMBER 永不衰减
            data["protected"] = True
            continue
        
        # 艾宾浩斯衰减公式
        elapsed_hours = (now - data["last_activated"]) / 3600
        decayed = data["weight"] * math.exp(-elapsed_hours * ln2 / half_life)
        
        if decayed < 0.05:
            # 太弱，移除
            self._graph.remove_edge(src, tgt)
        else:
            data["weight"] = decayed
```

### 孤立节点清理

```python
def _decay_and_prune(self):
    """...边衰减逻辑..."""
    
    # 移除孤立节点（度为 0），按重要度分级容忍时间
    nodes_to_remove = []
    for node_id in list(self._nodes.keys()):
        node = self._nodes[node_id]
        
        # TASK 类型节点永不因孤立而删除
        if node.node_type == NodeType.TASK:
            continue
        
        if self._graph.degree(node_id) == 0:
            imp = self.get_importance(node_id) or Importance.NORMAL
            imp_order = _IMPORTANCE_ORDER.get(imp, 1)
            
            # importance >= identity (order >= 2) → 永不因孤立删除（沉睡记忆）
            if imp_order >= _IMPORTANCE_ORDER[Importance.IDENTITY]:
                continue
            
            # 按重要度选择容忍时间
            if imp == Importance.TRIVIAL:
                orphan_age = 21600   # 6h
            elif imp_order >= _IMPORTANCE_ORDER[Importance.IMPORTANT]:
                orphan_age = 604800  # 7 天
            else:
                orphan_age = 86400   # 24h (normal 默认)
            
            if (time.time() - node.last_accessed) > orphan_age:
                nodes_to_remove.append(node_id)
    
    for nid in nodes_to_remove:
        self.remove_node(nid)
```

---

## 💡 实际应用示例

### 示例 1：身份信息长期保存

```python
# 第 1 轮：用户告知姓名
用户："我叫小明"
AI："你好小明，很高兴认识你！"

# DialogueAdapter 自动检测
_detect_importance("我叫小明")
  → 匹配规则：r"我叫.*"
  → 返回：Importance.IDENTITY

# 节点创建
node = GraphNode(
    node_id="dialogue:1",
    metadata={
        "importance": "identity",
        "content": "我叫小明",
        "detected_entities": ["匹配规则：我叫.*"]
    }
)

# 衰减特性
- 半衰期：720 小时（30 天）
- 30 天后边权衰减到 0.5
- 即使成为孤立节点（度为 0），也永不因孤立删除
```

### 示例 2：琐碎闲聊快速遗忘

```python
# 第 5 轮：用户简单回应
用户："好的"
AI："那我们继续吧"

# DialogueAdapter 自动检测
_detect_importance("好的")
  → 匹配规则：r"嗯 | 好 | 哦 | 好的"
  → 返回：Importance.TRIVIAL

# 节点创建
node = GraphNode(
    node_id="dialogue:5",
    metadata={"importance": "trivial"}
)

# 衰减特性
- 半衰期：6 小时
- 6 小时后边权衰减到 0.5
- 如果成为孤立节点，6 小时后被清理
```

### 示例 3：任务指令自动提升

```python
# 第 10 轮：用户提出任务
用户："帮我写一个 Python 脚本"
AI："好的，我来帮你..."

# 初始检测
_detect_importance("帮我写一个 Python 脚本")
  → 返回：Importance.NORMAL（无匹配规则）

# 节点创建
node = GraphNode(
    node_id="task:001",
    metadata={"importance": "normal"}
)

# 访问 3 次后（用户多次追问）
access_count = 3
run_importance_review()
  → 检测到 access_count >= 3 且 importance == NORMAL
  → 自动提升：NORMAL → IMPORTANT

# 提升后特性
- 半衰期从 24h 延长到 168h（7 天）
- 孤立容忍时间从 24h 延长到 7 天
```

### 示例 4：用户显式要求

```python
# 第 20 轮：用户显式要求
用户："帮我记住，我明天要参加重要会议"
AI："好的，我已经记住了"

# DialogueAdapter 自动检测
_detect_importance("帮我记住，我明天要参加重要会议")
  → 匹配规则：r"帮我记住.*"
  → 返回：Importance.MUST_REMEMBER

# 节点创建
node = GraphNode(
    node_id="dialogue:20",
    metadata={
        "importance": "must_remember",
        "content": "我明天要参加重要会议"
    }
)

# 特殊处理
if importance == Importance.MUST_REMEMBER:
    # 所有关联边自动设为 protected
    for edge in connected_edges:
        edge["protected"] = True

# 衰减特性
- 半衰期：inf（永不衰减）
- 边权始终保持初始值
- 永不因孤立删除
```

---

## 🎯 总结

MemoryGraph 的**多维标签体系**通过三组正交标签实现了灵活的记忆分类：

1. ✅ **温度标签** (HOT/WARM/COLD): 基于访问时间实时计算，用于检索路由
2. ✅ **重要度标签** (6 级分级): 写入时检测 + 动态提升，决定衰减半衰期
3. ✅ **时间段标签** (RECENT/NON_RECENT): 查询时计算，用于并行检索的互斥过滤

**核心优势**:
- ✅ **灵活性**: 标签正交，可自由组合（如 `HOT + MUST_REMEMBER + RECENT`）
- ✅ **自适应性**: 重要度动态提升，访问频繁的记忆自动变重要
- ✅ **差异化寿命**: 重要信息永久保存，琐碎信息快速遗忘（模拟人脑）
- ✅ **检索优化**: 热数据直接遍历，冷数据 FAISS 检索（性能最优）

**下一步**:
- [03-记忆检索机制](./03-retrieval.md) - 深入理解并行检索算法
- [04-图注意力机制](./04-attention.md) - 掌握图注意力评分公式

---

**最后更新**: 2026-04-19  
**维护者**: 祖龙系统核心开发团队
