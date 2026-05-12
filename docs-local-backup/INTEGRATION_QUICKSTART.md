# 集成经验库快速参考

## 快速开始

### 1. 导入模块

```python
from zulong.memory import get_integrated_experience_store
```

### 2. 获取实例

```python
store = get_integrated_experience_store(
    db_path="data/experience_db",
    enable_persistence=True,
    enable_smart_tagging=True,
    enable_review=True,
    enable_time_tags=True
)
```

### 3. 启动服务

```python
await store.start()
```

### 4. 添加经验

```python
exp_id = store.add_experience(
    content="成功连接 WiFi 的经验：1.打开设置 2.选择网络 3.输入密码",
    experience_type="success",
    tags=["network", "wifi"],
    metadata={"task_id": "task_001"}
)
```

### 5. 搜索经验

```python
results = store.search(
    query="网络",
    filter={'tags': ['network']},
    limit=10,
    use_time_decay=True
)

for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['content']}")
```

### 6. 更新使用

```python
store.update_usage(exp_id)
```

### 7. 评估回滚

```python
summary = await store.evaluate_and_rollback()
print(f"执行回滚：{summary['executed_count']}")
```

### 8. 停止服务

```python
await store.stop()
```

## 常用配置

### 混合检索配置

```python
store.store.configure_hybrid_search(
    alpha=0.7,        # 向量权重
    time_decay=0.1,   # 时间衰减
    max_age_days=30   # 最大保留天数
)
```

### 复盘触发配置

```python
# 用户主动触发
await store.review_trigger.trigger_user_active({
    'type': 'success',
    'data': {'dialog': [...], 'success_marker': '成功'}
})

# 安静模式（自动）
# 夜间模式（自动，每天 03:00）
```

## 统计监控

```python
stats = store.get_stats()

print(f"总经验数：{stats['total_experiences']}")
print(f"复盘次数：{stats['reviews_triggered']}")
print(f"回滚次数：{stats['rollbacks_executed']}")
```

## 测试命令

```bash
python tests/test_integration.py
```

## 常见问题

### Q: 如何禁用时间标签？

```python
store = get_integrated_experience_store(
    enable_time_tags=False
)
```

### Q: 如何禁用复盘？

```python
store = get_integrated_experience_store(
    enable_review=False
)
```

### Q: 如何禁用持久化？

```python
store = get_integrated_experience_store(
    enable_persistence=False
)
```

### Q: 如何手动触发复盘？

```python
await store.review_trigger.trigger_user_active({
    'type': 'failure',
    'data': {'error': '...', 'task': '...'}
})
```

## 完整示例

```python
import asyncio
from zulong.memory import get_integrated_experience_store

async def main():
    # 1. 获取实例
    store = get_integrated_experience_store()
    
    # 2. 启动
    await store.start()
    
    # 3. 添加经验
    exp_id = store.add_experience(
        content="WiFi 连接成功：步骤 1-2-3",
        experience_type="success",
        tags=["network", "wifi"]
    )
    
    # 4. 搜索
    results = store.search(
        query="WiFi",
        filter={'tags': ['network']},
        limit=5
    )
    
    # 5. 更新使用
    store.update_usage(exp_id)
    
    # 6. 评估回滚
    summary = await store.evaluate_and_rollback()
    
    # 7. 获取统计
    stats = store.get_stats()
    print(f"总经验数：{stats['total_experiences']}")
    
    # 8. 停止
    await store.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 参考资料

- [完整文档](./INTEGRATED_EXPERIENCE_STORE.md)
- [Phase 2.5 总结](./PHASE2_5_COMPLETION_SUMMARY.md)
- TSD v2.3 第 13 章
