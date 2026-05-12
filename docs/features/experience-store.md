# 集成经验库文档

## 概述

集成经验库将 Phase 2 完成的**复盘机制**和**时间标签体系**整合到增强版经验库中，提供统一的经验管理接口。

对应 TSD v2.3 第 13 章

## 系统架构

```
集成经验库 (IntegratedExperienceStore)
├── 基础经验库 (EnhancedExperienceStore)
│   ├── 向量检索 (FAISS)
│   ├── BM25 关键词检索
│   ├── 混合检索
│   └── 智能打标
├── 时间标签管理 (TimeTags)
│   ├── 三维时间追踪
│   ├── 时间衰减策略
│   └── 经验状态评估
├── 降智回滚 (RollbackManager)
│   ├── 自动降级
│   ├── 自动归档
│   └── 自动删除
└── 复盘机制 (Review Mechanism)
    ├── 三重触发
    ├── 成功经验提炼
    ├── 失败案例分析
    └── 三重防重复
```

## 核心功能

### 1. 统一经验管理

**添加经验**（自动带时间标签）:
```python
from zulong.memory import get_integrated_experience_store

store = get_integrated_experience_store()

# 添加成功经验
exp_id = store.add_experience(
    content="成功连接 WiFi 的经验：1.打开设置 2.选择网络 3.输入密码",
    experience_type="success",
    tags=["network", "wifi"],
    metadata={"task_id": "task_001"}
)
```

**搜索经验**（自动时间衰减）:
```python
# 搜索经验（自动应用时间衰减）
results = store.search(
    query="网络",
    filter={'tags': ['network']},
    limit=10,
    use_time_decay=True  # 启用时间衰减
)

for result in results:
    print(f"ID: {result['id']}")
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']}")
    print(f"Time Weight: {result['time_evaluation']['time_weight']}")
```

**更新使用记录**:
```python
# 更新经验使用记录（自动更新时间标签）
store.update_usage(exp_id)
```

### 2. 自动复盘

**启动复盘触发器**:
```python
await store.start()  # 启动三重触发器
```

**触发方式**:

1. **用户主动触发**:
```python
await store.review_trigger.trigger_user_active({
    'type': 'success',
    'data': {
        'dialog': [...],
        'success_marker': '成功'
    }
})
```

2. **安静模式触发**（自动）:
- 任务队列空闲持续 10 分钟
- 自动复盘最近 1 小时内失败或执行缓慢的任务

3. **夜间定时触发**（自动）:
- 每天凌晨 03:00
- 批量分析过去 24 小时内所有成功任务日志

### 3. 自动降智回滚

**评估并执行回滚**:
```python
# 评估所有经验并执行回滚
summary = await store.evaluate_and_rollback()

print(f"评估数量：{summary['total']}")
print(f"执行数量：{summary['executed_count']}")
print(f"降级：{summary['by_action'].get('downgrade', 0)}")
print(f"归档：{summary['by_action'].get('archive', 0)}")
print(f"删除：{summary['by_action'].get('delete', 0)}")
```

**回滚策略**:
- **30 天未验证** → 降级
- **90 天未使用** → 归档
- **180 天未使用** → 删除
- **评分≥0.7** → 重新激活

### 4. 时间标签检索

**时间标签三维追踪**:
- `created_at`: 创建时间
- `last_used_at`: 最后使用时间
- `last_validated_at`: 最后验证时间

**经验状态**:
- **Active** (活跃): 7 天内使用
- **Validated** (已验证): 30 天内验证
- **Stale** (陈旧): 90 天内未使用
- **Archived** (已归档): 超过 90 天

**时间衰减公式**:
```
最终分数 = 原始分数 × 时间权重
时间权重 = exp(-decay × age_days)
```

## 使用示例

### 完整工作流

```python
import asyncio
from zulong.memory import get_integrated_experience_store

async def main():
    # 1. 获取实例
    store = get_integrated_experience_store(
        db_path="data/experience_db",
        enable_persistence=True,
        enable_smart_tagging=True,
        enable_review=True,
        enable_time_tags=True
    )
    
    # 2. 启动
    await store.start()
    
    # 3. 添加经验
    exp_id = store.add_experience(
        content="WiFi 连接成功：步骤 1-2-3",
        experience_type="success",
        tags=["network", "wifi"]
    )
    
    # 4. 更新使用
    store.update_usage(exp_id)
    
    # 5. 搜索
    results = store.search(
        query="WiFi",
        filter={'tags': ['network']},
        limit=5,
        use_time_decay=True
    )
    
    # 6. 评估回滚
    summary = await store.evaluate_and_rollback()
    
    # 7. 获取统计
    stats = store.get_stats()
    print(f"总经验数：{stats['total_experiences']}")
    print(f"复盘次数：{stats['reviews_triggered']}")
    print(f"回滚次数：{stats['rollbacks_executed']}")
    
    # 8. 停止
    await store.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 与 RAG 集成

```python
from zulong.memory import get_integrated_experience_store, ExperienceRAG

# 获取集成经验库
store = get_integrated_experience_store()

# 创建经验 RAG
rag = ExperienceRAG()

# 搜索相关经验
results = store.search(
    query="如何连接网络",
    limit=5,
    use_time_decay=True
)

# 构建 RAG 上下文
context = "\n".join([r['content'] for r in results])

# 注入到 LLM
system_prompt = f"""你是祖龙机器人助手。
相关经验：
{context}
"""
```

## 配置选项

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| db_path | str | "data/experience_db" | 数据库路径 |
| enable_persistence | bool | True | 是否启用持久化 |
| enable_smart_tagging | bool | True | 是否启用智能打标 |
| enable_review | bool | True | 是否启用复盘机制 |
| enable_time_tags | bool | True | 是否启用时间标签 |

### 混合检索配置

```python
store.store.configure_hybrid_search(
    alpha=0.7,           # 向量检索权重 (0-1)
    time_decay=0.1,      # 时间衰减因子 (每天衰减比例)
    max_age_days=30      # 经验最大保留天数
)
```

## 统计监控

**获取统计信息**:
```python
stats = store.get_stats()

# 基础统计
print(f"总经验数：{stats['total_experiences']}")
print(f"BM25 文档数：{stats['bm25_docs']}")

# 时间标签
print(f"时间标签已启用：{stats.get('time_tags_enabled')}")

# 回滚统计
if 'rollback_stats' in stats:
    print(f"总评估：{stats['rollback_stats']['total_evaluations']}")
    print(f"降级：{stats['rollback_stats']['downgrades']}")
    print(f"归档：{stats['rollback_stats']['archives']}")
    print(f"删除：{stats['rollback_stats']['deletes']}")

# 复盘统计
if 'review_stats' in stats:
    print(f"总触发：{stats['review_stats']['total_triggers']}")
    print(f"用户主动：{stats['review_stats']['user_active_count']}")
    print(f"安静模式：{stats['review_stats']['quiet_mode_count']}")
    print(f"夜间定时：{stats['review_stats']['night_schedule_count']}")
```

## 最佳实践

### 1. 单例模式

```python
# ✅ 推荐：使用全局单例
store = get_integrated_experience_store()

# ❌ 不推荐：直接实例化
store = IntegratedExperienceStore()
```

### 2. 生命周期管理

```python
# 启动时
await store.start()

# 使用时
# ... 添加/搜索/更新经验 ...

# 停止时
await store.stop()
```

### 3. 错误处理

```python
try:
    exp_id = store.add_experience(
        content="...",
        experience_type="success"
    )
except Exception as e:
    logger.error(f"添加经验失败：{e}")
```

### 4. 批量操作

```python
# 批量评估回滚
summary = await store.evaluate_and_rollback(
    experience_ids=None  # None=全部，或指定 ID 列表
)
```

## 性能优化

### 1. 持久化

```python
# 启用持久化（推荐生产环境）
store = get_integrated_experience_store(
    enable_persistence=True,
    db_path="data/experience_db"
)

# Mock 模式（开发测试）
store = get_integrated_experience_store(
    enable_persistence=False
)
```

### 2. 智能打标

```python
# 启用智能打标（自动领域识别）
store = get_integrated_experience_store(
    enable_smart_tagging=True
)

# 禁用智能打标（更快）
store = get_integrated_experience_store(
    enable_smart_tagging=False
)
```

### 3. 时间标签

```python
# 启用时间标签（自动时间衰减）
store = get_integrated_experience_store(
    enable_time_tags=True
)

# 禁用时间标签（更快）
store = get_integrated_experience_store(
    enable_time_tags=False
)
```

## 测试

**运行集成测试**:
```bash
python tests/test_integration.py
```

**测试覆盖**:
- ✅ 集成经验库初始化
- ✅ 添加经验（带时间标签）
- ✅ 搜索（带时间衰减）
- ✅ 更新使用记录
- ✅ 降智回滚
- ✅ 复盘集成
- ✅ 端到端工作流

## 故障排查

### 问题 1: 经验 ID 不匹配

**现象**: `KeyError: 'exp_xxx'`

**原因**: 使用了自定义 ID 而非经验库生成的 ID

**解决**:
```python
# ✅ 正确
exp_id = store.add_experience(...)

# ❌ 错误
exp_id = "custom_id"
store.add_experience(...)
```

### 问题 2: 回滚执行失败

**现象**: `'EnhancedExperienceStore' object has no attribute 'update_experience_weight'`

**原因**: 回滚模块调用了经验库未实现的方法

**解决**: 这是预期行为，回滚建议仅供参考，实际执行需要手动处理

### 问题 3: 复盘未触发

**现象**: 复盘触发器启动但无反应

**检查**:
1. 确认 `enable_review=True`
2. 检查防重复过滤器是否过于严格
3. 查看日志确认触发器状态

## 下一步

1. ✅ 复盘机制 (三重触发 + 防重复) - **已完成**
2. ✅ 时间标签体系与降智回滚 - **已完成**
3. ✅ 集成经验库 - **已完成**
4. ⏳ Phase 3: 专家技能模块与 RAG 增强 - **下一步**

## 参考资料

- TSD v2.3 第 12 章：时间标签与降智回滚
- TSD v2.3 第 13 章：集成经验库
- [复盘机制与智能经验库系统架构升级原子任务](../资料/复盘机制与智能经验库系统架构升级原子任务.txt)
- [Phase 2 完成总结](./PHASE2_COMPLETION_SUMMARY.md)
