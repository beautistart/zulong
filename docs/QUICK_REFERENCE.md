# 增强版经验库 - 快速参考

## 🚀 快速开始

### 1. 安装依赖（推荐）

```bash
# 中文分词（提升 BM25 精度）
pip install jieba

# 真实 Embedding 模型（提升语义检索）
pip install sentence-transformers
```

### 2. 基本使用

```python
from zulong.memory.three_libraries import ExperienceStore

# 创建经验库
store = ExperienceStore(
    db_path="data/experience_db",
    enable_persistence=True  # 启用持久化
)

# 添加经验
exp_id = store.add_experience(
    content="当用户抱怨网络慢时，应引导其检查路由器是否过热",
    experience_type="logic",
    tags=["network", "troubleshooting"],
    importance_score=0.9
)

# 检索经验
results = store.search_by_text(
    query="网络慢怎么办",
    filter_types=["logic", "success"],
    filter_tags=["network"],
    limit=5
)

# 生成 Prompt 上下文
prompt_context = store.to_prompt_context(results)
```

---

## 📚 API 参考

### 创建与配置

```python
# 创建经验库
store = ExperienceStore(
    db_path="data/experience_db",      # 数据库路径
    enable_persistence=True            # 是否启用持久化
)

# 配置混合检索
store.configure_hybrid_search(
    alpha=0.7,          # 向量权重（0-1）
    time_decay=0.05,    # 时间衰减因子（每天）
    max_age_days=30     # 最大保留天数
)

# 设置 Embedding 模型
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
store.set_embedding_model(model)
```

### 添加经验

```python
# 基础用法
exp_id = store.add_experience(
    content="经验内容",
    experience_type="logic",  # logic/failure/success/preference
    success=True
)

# 高级用法
exp_id = store.add_experience(
    content="网络优化建议",
    experience_type="success",
    task_id="task_123",
    success=True,
    tags=["network", "optimization"],
    importance_score=0.9,
    metadata={"source": "user_feedback"}
)
```

### 检索经验

```python
# 文本检索
results = store.search_by_text(
    query="网络慢怎么办",
    filter_types=["logic", "success"],    # 类型过滤
    filter_tags=["network"],              # 标签过滤
    tag_logic="OR",                       # OR/AND 逻辑
    use_hybrid=True,                      # 混合检索
    apply_time_decay=True,                # 时间衰减
    limit=5                               # 返回数量
)

# 向量检索
query_vector = store._get_embedding("网络问题")
results = store.search(
    query_vector=query_vector,
    filter_types=["logic"],
    limit=5
)

# 纯关键词检索（BM25）
store.hybrid_alpha = 0.0  # 100% BM25
results = store.search_by_text("路由器重启", use_hybrid=True)

# 纯向量检索
store.hybrid_alpha = 1.0  # 100% 向量
results = store.search_by_text("网络问题", use_hybrid=True)
```

### 标签过滤

```python
# OR 逻辑（宽松）- 包含任一标签
results = store.search_by_text(
    query="技术问题",
    filter_tags=["network", "hardware"],
    tag_logic="OR"
)

# AND 逻辑（严格）- 包含所有标签
results = store.search_by_text(
    query="网络硬件问题",
    filter_tags=["network", "hardware"],
    tag_logic="AND"
)

# 组合过滤（类型 + 标签）
results = store.search_by_text(
    query="网络优化",
    filter_types=["logic", "success"],
    filter_tags=["network"],
    tag_logic="OR"
)
```

### 持久化操作

```python
# 手动保存
store.save()

# 关闭并保存
store.close()

# 重新加载
store = ExperienceStore(
    db_path="data/experience_db",
    enable_persistence=True
)
# 自动从磁盘加载所有数据
```

### 统计信息

```python
stats = store.get_statistics()

print(f"总经验数：{stats['total']}")
print(f"类型分布：{stats['type_distribution']}")
print(f"Top 标签：{stats['top_tags']}")
print(f"年龄分布：{stats['age_distribution']}")
print(f"平均访问次数：{stats['avg_access_count']:.2f}")
```

### Prompt 上下文

```python
# 生成 Prompt 上下文
results = store.search_by_text("网络问题", limit=3)
prompt_context = store.to_prompt_context(results)

# 输出示例：
"""
## 相关经验

✅ [logic] 当用户抱怨网络慢时，应引导其检查路由器 (标签：network, troubleshooting) [新，0.5 天]

✅ [success] 网络设置优化：调整 DNS 服务器 (标签：network, optimization) [新，2.1 天]

✅ [logic] 导航路径规划时，应避开动态障碍物 (标签：navigation, safety) [新，3.2 天]
"""
```

---

## 🎯 参数说明

### 经验类型 (experience_type)

| 类型 | 说明 | 使用场景 |
|------|------|---------|
| `logic` | 逻辑规则 | 通用知识、规则、方法 |
| `failure` | 失败经验 | 错误案例、避坑指南 |
| `success` | 成功经验 | 成功案例、最佳实践 |
| `preference` | 用户偏好 | 用户习惯、个性化设置 |

### 检索参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | str | - | 查询文本 |
| `filter_types` | List[str] | None | 经验类型过滤列表 |
| `filter_tags` | List[str] | None | 标签过滤列表 |
| `tag_logic` | str | "OR" | 标签过滤逻辑（OR/AND） |
| `use_hybrid` | bool | True | 是否使用混合检索 |
| `apply_time_decay` | bool | True | 是否应用时间衰减 |
| `limit` | int | 5 | 返回数量限制 |

### 混合检索配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alpha` | float | 0.7 | 向量权重（0-1） |
| `time_decay` | float | 0.05 | 时间衰减因子（每天） |
| `max_age_days` | int | 30 | 最大保留天数 |

**alpha 值建议**:
- `0.0` = 100% BM25（关键词精确匹配）
- `0.5` = 50% 向量 + 50% BM25（平衡）
- `0.7` = 70% 向量 + 30% BM25（推荐）
- `1.0` = 100% 向量（语义理解）

---

## 📁 文件结构

```
data/experience_db/
├── experiences.db          # SQLite 数据库（元数据）
│   └── 表：experiences
│       - id, content, experience_type
│       - task_id, success, metadata
│       - timestamp, keywords, tags
│       - importance_score, access_count
│       └── last_accessed
│
└── experiences_data.pkl    # Pickle 文件（二进制数据）
    ├── embeddings          # 经验向量
    └── bm25_index          # BM25 关键词索引
```

---

## 🔧 常见问题

### Q1: 如何迁移旧版数据？

```python
from zulong.memory.three_libraries import ExperienceStore

# 创建新版经验库（会自动加载旧数据）
store = ExperienceStore(
    db_path="data/experience_db",
    enable_persistence=True
)

# 如果有旧版数据，可以手动迁移
old_experiences = [...]  # 旧版数据
for exp in old_experiences:
    store.add_experience(
        content=exp['content'],
        experience_type=exp['type'],
        success=exp.get('success', True)
    )
```

### Q2: 如何备份数据？

```python
# 1. 手动保存
store.save()

# 2. 复制整个数据库目录
import shutil
shutil.copytree("data/experience_db", "data/experience_db_backup")

# 3. 恢复时复制回来
shutil.rmtree("data/experience_db")
shutil.copytree("data/experience_db_backup", "data/experience_db")
```

### Q3: 如何优化检索性能？

```python
# 1. 安装 jieba（提升 BM25 精度）
pip install jieba

# 2. 使用真实 Embedding 模型（提升语义检索）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
store.set_embedding_model(model)

# 3. 调整 alpha 值（根据场景）
store.configure_hybrid_search(alpha=0.7)  # 通用场景

# 4. 缓存热门查询
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str):
    return store.search_by_text(query, limit=5)
```

### Q4: 如何处理大量数据？

```python
# 1. 批量添加（减少保存次数）
for i, exp in enumerate(experiences):
    store.add_experience(exp)
    if i % 100 == 0:
        store.save()  # 每 100 条保存一次

# 2. 定期清理老经验
def cleanup_old_experiences():
    stats = store.get_statistics()
    if stats['total'] > 10000:
        # 清理 30 天以上的经验
        # （需要实现 delete_by_age 方法）
        pass

# 3. 使用索引优化查询
# SQLite 已自动创建索引
# - idx_type: experience_type
# - idx_timestamp: timestamp
```

---

## 📊 最佳实践

### 1. 标签命名规范

```python
# 推荐：使用英文小写，下划线分隔
tags = ["network", "troubleshooting", "hardware"]

# 避免：中文、大写、空格
tags = ["网络", "Troubleshooting", "hardware fix"]  # ❌
```

### 2. 经验内容格式

```python
# 推荐：清晰、具体、可操作
content = "当用户抱怨网络慢时，应引导其检查路由器是否过热，并建议重启路由器"

# 避免：模糊、抽象、不可操作
content = "网络问题需要解决"  # ❌
```

### 3. 重要性分数设置

```python
# 关键经验（系统核心逻辑）
importance_score = 1.0

# 重要经验（常用方法）
importance_score = 0.8-0.9

# 一般经验（参考信息）
importance_score = 0.5-0.7

# 次要经验（补充信息）
importance_score = 0.3-0.4
```

### 4. 检索策略选择

```python
# 场景 1: 精确匹配特定术语
results = store.search_by_text(
    query="DNS 服务器配置",
    use_hybrid=False,  # 纯 BM25
    filter_tags=["network"]
)

# 场景 2: 语义理解（同义词）
results = store.search_by_text(
    query="网速卡",  # 实际想找"网络慢"
    use_hybrid=True,
    alpha=0.8  # 高向量权重
)

# 场景 3: 综合检索（推荐）
results = store.search_by_text(
    query="网络慢怎么办",
    use_hybrid=True,
    alpha=0.7,  # 平衡语义和关键词
    apply_time_decay=True,  # 新经验优先
    limit=5
)
```

---

## 🎉 完整示例

```python
"""
祖龙系统经验库 - 完整使用示例
"""

from zulong.memory.three_libraries import ExperienceStore
from sentence_transformers import SentenceTransformer

# ========== 1. 初始化 ==========
print("📦 初始化经验库...")

store = ExperienceStore(
    db_path="data/experience_db",
    enable_persistence=True
)

# 配置混合检索
store.configure_hybrid_search(
    alpha=0.7,
    time_decay=0.05,
    max_age_days=30
)

# 设置真实 Embedding 模型
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
store.set_embedding_model(model)

print(f"✅ 经验库已初始化：{len(store._experiences)} 条经验")

# ========== 2. 添加经验 ==========
print("\n📝 添加经验...")

experiences = [
    {
        "content": "当用户抱怨网络慢时，应引导其检查路由器是否过热，并建议重启路由器",
        "type": "logic",
        "tags": ["network", "troubleshooting", "hardware"],
        "importance": 0.9
    },
    {
        "content": "网络设置优化：调整 DNS 服务器为 8.8.8.8 可以提升网速",
        "type": "success",
        "tags": ["network", "optimization", "dns"],
        "importance": 0.85
    },
    {
        "content": "机械臂抓取物体时，应先校准坐标系，确保抓取精度",
        "type": "logic",
        "tags": ["manipulation", "calibration", "robot"],
        "importance": 0.95
    }
]

for exp_data in experiences:
    exp_id = store.add_experience(
        content=exp_data["content"],
        experience_type=exp_data["type"],
        success=True,
        tags=exp_data["tags"],
        importance_score=exp_data["importance"]
    )
    print(f"   ✅ 添加经验：{exp_id[:8]}...")

print(f"✅ 共添加 {len(experiences)} 条经验")

# ========== 3. 检索经验 ==========
print("\n🔍 检索经验...")

query = "网络慢怎么办"
print(f"   查询：{query}")

results = store.search_by_text(
    query=query,
    filter_types=["logic", "success"],
    filter_tags=["network"],
    tag_logic="OR",
    use_hybrid=True,
    apply_time_decay=True,
    limit=3
)

print(f"   返回 {len(results)} 条结果:")
for i, exp in enumerate(results, 1):
    print(f"   {i}. [{exp.experience_type}] {exp.content[:50]}...")
    print(f"      标签：{exp.tags[:3]}")

# ========== 4. 生成 Prompt ==========
print("\n📋 生成 Prompt 上下文...")

prompt_context = store.to_prompt_context(results)
print(prompt_context)

# ========== 5. 统计信息 ==========
print("\n📊 统计信息...")

stats = store.get_statistics()
print(f"   总经验数：{stats['total']}")
print(f"   类型分布：{stats['type_distribution']}")
print(f"   Top 标签：{stats['top_tags'][:3]}")

# ========== 6. 保存并关闭 ==========
print("\n💾 保存并关闭...")

store.save()
store.close()

print("✅ 完成！")
```

---

## 📞 支持与反馈

如有问题或建议，请查阅以下文档：

- **架构设计**: [`ENHANCED_EXPERIENCE_STORE_ARCHITECTURE.md`](file://d:\AI\project\zulong_beta4\docs\ENHANCED_EXPERIENCE_STORE_ARCHITECTURE.md)
- **实现总结**: [`IMPLEMENTATION_SUMMARY.md`](file://d:\AI\project\zulong_beta4\docs\IMPLEMENTATION_SUMMARY.md)
- **三大功能**: [`THREE_ENHANCEMENTS_SUMMARY.md`](file://d:\AI\project\zulong_beta4\docs\THREE_ENHANCEMENTS_SUMMARY.md)

---

**最后更新**: 2026-03-29  
**版本**: v1.0  
**状态**: ✅ 生产就绪
