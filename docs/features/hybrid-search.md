# 混合检索配置指南

## 概述

本指南介绍如何在祖龙 (ZULONG) 系统中配置和优化混合检索（向量 + BM25）。

**对应 TSD**: v2.3 第 14.2 节  
**版本**: 1.0  
**日期**: 2026-03-29

---

## 快速开始

### 1. 基本使用

```python
from zulong.memory import (
    EnhancedExperienceStore,
    get_embedding_manager,
    get_balanced_config
)

# 初始化
store = EnhancedExperienceStore()
manager = get_embedding_manager()

# 获取默认配置（平衡）
config = get_balanced_config()

# 编码查询
query = "网络问题"
query_emb = manager.encode_query(query)

# 混合检索
results = store.search(
    query_vector=query_emb,
    query_text=query,  # BM25 使用
    config=config,
    top_k=5
)
```

---

## 配置模板

### 1. 平衡配置（默认）

**适用场景**: 通用检索，兼顾语义和相关性

```python
from zulong.memory import get_balanced_config

config = get_balanced_config()

# 参数
# vector_weight=0.7        # 向量权重 70%
# bm25_weight=0.3          # BM25 权重 30%
# time_decay_factor=0.1    # 每天衰减 10%
# half_life_days=7         # 半衰期 7 天
```

**推荐指数**: ⭐⭐⭐⭐⭐

**使用场景**:
- 日常经验检索
- 通用问答
- 故障排查

---

### 2. 向量优先配置

**适用场景**: 语义搜索，模糊匹配

```python
from zulong.memory import get_vector_focused_config

config = get_vector_focused_config()

# 参数
# vector_weight=0.9        # 向量权重 90%
# bm25_weight=0.1          # BM25 权重 10%
# time_decay_factor=0.05   # 每天衰减 5%
# half_life_days=14        # 半衰期 14 天
```

**推荐指数**: ⭐⭐⭐⭐

**使用场景**:
- 模糊查询（"那个...怎么弄来着？"）
- 语义相似问题
- 跨语言检索

**示例**:
```python
# 用户问："那个连不上网的"
# 实际匹配："WiFi 连接失败"、"网络无法访问"
```

---

### 3. 关键词优先配置

**适用场景**: 精准关键词匹配

```python
from zulong.memory import get_keyword_focused_config

config = get_keyword_focused_config()

# 参数
# vector_weight=0.5        # 向量权重 50%
# bm25_weight=0.5          # BM25 权重 50%
# time_decay_factor=0.15   # 每天衰减 15%
# half_life_days=5         # 半衰期 5 天
```

**推荐指数**: ⭐⭐⭐⭐

**使用场景**:
- 技术术语检索
- 错误代码匹配
- 精准查询

**示例**:
```python
# 用户问："Error 404"
# 精准匹配："HTTP 404 错误"、"404 Not Found"
```

---

### 4. 新鲜内容优先配置

**适用场景**: 实时数据，新闻检索

```python
from zulong.memory import get_fresh_content_config

config = get_fresh_content_config()

# 参数
# vector_weight=0.7        # 向量权重 70%
# bm25_weight=0.3          # BM25 权重 30%
# time_decay_factor=0.2    # 每天衰减 20%
# half_life_days=3         # 半衰期 3 天
```

**推荐指数**: ⭐⭐⭐

**使用场景**:
- 实时新闻
- 最新经验
- 时效性内容

**示例**:
```python
# 优先返回最近 3 天的经验
# 7 天前的经验权重衰减到 50%
```

---

## 自定义配置

### 1. 基础自定义

```python
from zulong.memory import HybridSearchConfig

config = HybridSearchConfig(
    vector_weight=0.8,         # 向量权重
    bm25_weight=0.2,           # BM25 权重
    time_decay_factor=0.12,    # 时间衰减因子
    half_life_days=6           # 半衰期（天）
)
```

### 2. 权重自动归一化

```python
# 权重和不为 1 时自动归一化
config = HybridSearchConfig(
    vector_weight=0.9,
    bm25_weight=0.3  # 总和 1.2
)

# 自动调整为
# vector_weight = 0.9 / 1.2 = 0.75
# bm25_weight = 0.3 / 1.2 = 0.25
```

---

## 自适应优化

### 1. 优化器使用

```python
from zulong.memory import HybridSearchOptimizer, get_balanced_config

# 初始化
config = get_balanced_config()
optimizer = HybridSearchOptimizer(config)

# 记录搜索历史
for i in range(50):
    results = store.search(query=f"查询{i}", top_k=10)
    
    optimizer.record_search(
        query=f"查询{i}",
        results=results,
        latency_ms=45,
        user_clicked=0  # 用户点击了第 1 个结果
    )
    
    # 记录反馈
    optimizer.record_feedback(
        query=f"查询{i}",
        result_id=results[0]['id'],
        relevance_score=0.9  # 相关度评分
    )
```

### 2. 性能分析

```python
# 生成性能报告
report = optimizer.analyze_performance()

print(f"总搜索次数：{report['total_searches']}")
print(f"平均延迟：{report['avg_latency_ms']:.2f}ms")
print(f"点击率：{report['click_through_rate']:.2%}")
print(f"用户满意度：{report['user_satisfaction']:.2f}")
print(f"当前配置：{report['current_config']}")
```

### 3. 获取优化建议

```python
# 获取优化建议
suggestions = optimizer.suggest_optimization()

print(f"建议向量权重：{suggestions['vector_weight']}")
print(f"建议 BM25 权重：{suggestions['bm25_weight']}")
```

### 4. 应用优化

```python
# 应用优化建议
optimizer.apply_optimization(suggestions)

# 验证新配置
print(f"新配置：vector={optimizer.config.vector_weight}, "
      f"bm25={optimizer.config.bm25_weight}")
```

---

## 混合检索公式

### 1. 得分计算

```python
import numpy as np

def calculate_final_score(vector_score, bm25_score, config):
    """计算最终得分"""
    # 加权平均
    final_score = (
        config.vector_weight * vector_score +
        config.bm25_weight * bm25_score
    )
    
    return final_score

def calculate_time_weight(created_at, config):
    """计算时间权重"""
    from datetime import datetime
    
    age_days = (datetime.now() - created_at).days
    time_weight = np.exp(-config.time_decay_factor * age_days)
    
    return time_weight

def calculate_ranked_score(final_score, time_weight):
    """计算排名得分"""
    return final_score * time_weight
```

### 2. 示例计算

```python
from datetime import datetime, timedelta

config = get_balanced_config()

# 经验 A: 向量 0.9, BM25 0.6, 3 天前
exp_a_vector = 0.9
exp_a_bm25 = 0.6
exp_a_age = 3

final_a = calculate_final_score(exp_a_vector, exp_a_bm25, config)
time_a = calculate_time_weight(
    datetime.now() - timedelta(days=exp_a_age),
    config
)
ranked_a = calculate_ranked_score(final_a, time_a)

print(f"经验 A: final={final_a:.3f}, time={time_a:.3f}, ranked={ranked_a:.3f}")

# 经验 B: 向量 0.7, BM25 0.8, 1 天前
exp_b_vector = 0.7
exp_b_bm25 = 0.8
exp_b_age = 1

final_b = calculate_final_score(exp_b_vector, exp_b_bm25, config)
time_b = calculate_time_weight(
    datetime.now() - timedelta(days=exp_b_age),
    config
)
ranked_b = calculate_ranked_score(final_b, time_b)

print(f"经验 B: final={final_b:.3f}, time={time_b:.3f}, ranked={ranked_b:.3f}")
```

---

## 性能监控

### 1. 统计信息

```python
# 获取统计信息
stats = optimizer.get_stats()

print(f"搜索历史：{stats['history_size']}")
print(f"反馈历史：{stats['feedback_size']}")
print(f"配置：{stats['config']}")
```

### 2. 性能指标

**关键指标**:
- **平均延迟**: `< 100ms` ✅
- **点击率**: `> 30%` ✅
- **用户满意度**: `> 0.7` ✅

**优化建议**:
- 延迟过高 → 减小 top_k 或使用 CPU 模式
- 点击率低 → 调整权重配置
- 满意度低 → 收集更多反馈数据

---

## 最佳实践

### 1. 配置选择

**通用场景**:
```python
config = get_balanced_config()  # 默认选择
```

**语义搜索**:
```python
config = get_vector_focused_config()
```

**精准匹配**:
```python
config = get_keyword_focused_config()
```

**实时内容**:
```python
config = get_fresh_content_config()
```

### 2. 权重调优

**逐步调整**:
```python
# 初始配置
config = HybridSearchConfig(vector_weight=0.7)

# 收集 100 次搜索数据
for i in range(100):
    # ... 记录搜索和反馈

# 分析并调整
suggestions = optimizer.suggest_optimization()
optimizer.apply_optimization(suggestions)

# 验证效果
report = optimizer.analyze_performance()
```

### 3. 时间衰减

**半衰期选择**:
- **短半衰期** (3-5 天): 新闻、实时数据
- **中半衰期** (7-10 天): 通用经验
- **长半衰期** (14-30 天): 技术文档、稳定知识

---

## 故障排查

### 1. 权重不生效

**现象**: 检索结果不符合预期权重

**原因**: 权重未归一化

**解决方案**:
```python
# 检查权重和
total = config.vector_weight + config.bm25_weight
print(f"权重和：{total}")  # 应为 1.0

# 如果不为 1，系统会自动归一化
# 建议手动归一化
config.vector_weight /= total
config.bm25_weight /= total
```

### 2. 时间衰减过快

**现象**: 旧经验完全不被检索到

**原因**: 时间衰减因子过大

**解决方案**:
```python
# 减小衰减因子
config = HybridSearchConfig(
    time_decay_factor=0.05,  # 从 0.2 降到 0.05
    half_life_days=14        # 增加半衰期
)
```

### 3. 优化建议不合理

**现象**: 优化后效果变差

**原因**: 反馈数据不足或有偏

**解决方案**:
```python
# 检查数据量
stats = optimizer.get_stats()
if stats['feedback_size'] < 50:
    print("反馈数据不足，继续收集")
else:
    # 检查数据质量
    suggestions = optimizer.suggest_optimization()
    # 人工审核建议
    print(f"建议：{suggestions}")
```

---

## 示例代码

### 1. 完整检索流程

```python
from zulong.memory import (
    EnhancedExperienceStore,
    get_embedding_manager,
    get_balanced_config,
    HybridSearchOptimizer
)

# 初始化
store = EnhancedExperienceStore()
manager = get_embedding_manager()
config = get_balanced_config()
optimizer = HybridSearchOptimizer(config)

# 用户查询
query = "如何连接网络"
query_emb = manager.encode_query(query)

# 执行检索
results = store.search(
    query_vector=query_emb,
    query_text=query,
    config=config,
    top_k=5
)

# 记录搜索
optimizer.record_search(
    query=query,
    results=results,
    latency_ms=45,
    user_clicked=0
)

# 用户反馈（假设点击了第 1 个结果）
optimizer.record_feedback(
    query=query,
    result_id=results[0]['id'],
    relevance_score=0.9
)

# 定期优化
if optimizer.get_stats()['feedback_size'] > 100:
    suggestions = optimizer.suggest_optimization()
    optimizer.apply_optimization(suggestions)
```

### 2. A/B 测试配置

```python
from zulong.memory import HybridSearchConfig

# 配置 A: 向量优先
config_a = HybridSearchConfig(vector_weight=0.9)

# 配置 B: 平衡
config_b = HybridSearchConfig(vector_weight=0.7)

# 交替测试
for i, query in enumerate(queries):
    config = config_a if i % 2 == 0 else config_b
    results = store.search(query=query, config=config)
    # 记录点击和反馈
```

---

## 参考资料

- TSD v2.3 第 14.2 节：混合检索配置
- [混合检索最佳实践](https://example.com/hybrid-search)
- [BM25 算法详解](https://example.com/bm25)
