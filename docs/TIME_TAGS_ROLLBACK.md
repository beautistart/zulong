# 时间标签体系与降智回滚文档

## 概述

时间标签体系为祖龙系统的经验记忆提供三维时间追踪能力，配合降智回滚机制实现经验的自动降级、归档和清理，保持记忆库的活力和准确性。

对应 TSD v2.3 第 12 章

## 系统架构

```
时间标签与回滚系统
├── 时间标签体系 (time_tags.py)
│   ├── TimeTags (三维时间追踪)
│   ├── TimeDecayStrategy (时间衰减策略)
│   └── TimeTagManager (时间标签管理器)
└── 降智回滚机制 (rollback.py)
    ├── RollbackStrategy (回滚策略)
    └── RollbackManager (回滚管理器)
```

## 核心模块

### 1. 三维时间标签 (TimeTags)

**三个时间维度**:
- **created_at**: 经验创建时间
- **last_used_at**: 最后使用时间
- **last_validated_at**: 最后验证时间

**经验状态**:
```python
ExperienceStatus.ACTIVE       # 活跃 (7 天内使用)
ExperienceStatus.VALIDATED    # 已验证 (30 天内验证)
ExperienceStatus.STALE        # 过期 (30-90 天未使用)
ExperienceStatus.ARCHIVED     # 归档 (90 天以上未使用)
```

**使用示例**:
```python
from zulong.memory import TimeTags, ExperienceStatus
from datetime import datetime, timedelta

# 创建时间标签
tags = TimeTags()

# 获取状态
status = tags.get_status()
print(f"状态：{status.value}")

# 获取年龄
age = tags.get_age_days()
print(f"年龄：{age} 天")

# 模拟旧经验
old_tags = TimeTags(
    created_at=datetime.utcnow() - timedelta(days=100),
    last_used_at=datetime.utcnow() - timedelta(days=100)
)

old_status = old_tags.get_status()
print(f"旧经验状态：{old_status.value}")  # archived
```

### 2. 时间衰减策略 (TimeDecayStrategy)

**衰减参数**:
- **decay_rate**: 衰减率 (每天)
- **half_life_days**: 半衰期 (天，默认 30 天)
- **min_weight**: 最小权重 (默认 0.1)

**计算指标**:
1. **时间权重**: 基于经验年龄的指数衰减
2. **新鲜度分数**: 基于最后使用/验证时间
3. **使用频率分数**: 每天平均使用次数

**使用示例**:
```python
from zulong.memory import TimeTags, TimeDecayStrategy, TimeTagManager

# 创建衰减策略
strategy = TimeDecayStrategy(
    decay_rate=0.1,
    half_life_days=30,
    min_weight=0.1
)

# 创建时间标签
tags = TimeTags(
    created_at=datetime.utcnow() - timedelta(days=60)
)

# 计算权重
weight = strategy.calculate_weight(tags, base_weight=1.0)
print(f"60 天后的权重：{weight:.3f}")

# 计算新鲜度
recency = strategy.calculate_recency_score(tags)
print(f"新鲜度分数：{recency:.3f}")

# 使用管理器进行综合评估
manager = TimeTagManager(strategy)
evaluation = manager.evaluate_experience(tags, usage_count=10)

print(f"综合评分：{evaluation['overall_score']:.3f}")
print(f"建议：{evaluation['recommendation']}")
```

### 3. 降智回滚策略 (RollbackStrategy)

**回滚阈值**:
- **downgrade_days**: 30 天未验证 -> 降级
- **archive_days**: 90 天未使用 -> 归档
- **delete_days**: 180 天未使用 -> 删除
- **reactivation_threshold**: 0.7 (重新激活阈值)

**回滚动作**:
```python
RollbackAction.NONE         # 无需操作
RollbackAction.DOWNGRADE    # 降级 (降低权重)
RollbackAction.ARCHIVE      # 归档 (移动到归档库)
RollbackAction.DELETE       # 删除 (从库中移除)
RollbackAction.REACTIVATE   # 重新激活 (提升权重)
```

**经验等级**:
```python
ExperienceLevel.LEVEL_1     # 新经验 (< 7 天)
ExperienceLevel.LEVEL_2     # 成熟经验 (7-30 天)
ExperienceLevel.LEVEL_3     # 稳定经验 (30-90 天)
ExperienceLevel.LEVEL_4     # 过期经验 (> 90 天)
```

**使用示例**:
```python
from zulong.memory import RollbackStrategy, RollbackAction

# 创建回滚策略
strategy = RollbackStrategy(
    downgrade_days=30,
    archive_days=90,
    delete_days=180
)

# 评估回滚动作
from zulong.memory import TimeTags

old_tags = TimeTags(
    created_at=datetime.utcnow() - timedelta(days=100),
    last_used_at=datetime.utcnow() - timedelta(days=100)
)

action = strategy.evaluate_rollback(old_tags, overall_score=0.2)
print(f"回滚动作：{action.value}")  # archive
```

### 4. 回滚管理器 (RollbackManager)

**功能**:
- 单个经验评估
- 批量经验评估
- 回滚执行
- 回调通知
- 统计监控

**使用示例**:
```python
from zulong.memory import get_rollback_manager, RollbackResult

# 获取管理器
manager = get_rollback_manager()

# 注册回调
async def rollback_callback(result: RollbackResult):
    print(f"回滚：{result.experience_id} -> {result.action.value}")

manager.register_callback(rollback_callback)

# 评估单个经验
exp = {
    'id': 'exp_001',
    'time_tags': {
        'created_at': (datetime.utcnow() - timedelta(days=100)).isoformat(),
        'last_used_at': (datetime.utcnow() - timedelta(days=100)).isoformat(),
        'last_validated_at': None
    },
    'usage_count': 2
}

result = await manager.evaluate_experience(exp)
print(f"回滚动作：{result.action.value}")

# 批量评估
experiences = [exp1, exp2, exp3]
results = await manager.evaluate_batch(experiences)

# 获取统计
stats = manager.get_stats()
print(f"统计：{stats}")
```

## 完整工作流

### 经验生命周期管理

```python
from zulong.memory import (
    get_time_tag_manager,
    get_rollback_manager,
    TimeTags
)
from datetime import datetime, timedelta

# 1. 创建经验时间标签
time_manager = get_time_tag_manager()
rollback_manager = get_rollback_manager(time_manager)

# 2. 新经验创建
exp = {
    'id': 'exp_new',
    'time_tags': TimeTags().to_dict(),
    'usage_count': 0
}

# 3. 评估 (新经验)
result = await rollback_manager.evaluate_experience(exp)
# -> action: REACTIVATE (新经验，评分高)

# 4. 使用经验后更新
exp['time_tags'] = TimeTags(
    created_at=datetime.utcnow() - timedelta(days=10),
    last_used_at=datetime.utcnow() - timedelta(days=1),
    last_validated_at=datetime.utcnow() - timedelta(days=5)
).to_dict()
exp['usage_count'] = 15

# 5. 再次评估 (活跃经验)
result = await rollback_manager.evaluate_experience(exp)
# -> action: REACTIVATE (活跃，保持高权重)

# 6. 长期未使用 (30 天)
exp['time_tags'] = TimeTags(
    created_at=datetime.utcnow() - timedelta(days=60),
    last_used_at=datetime.utcnow() - timedelta(days=60)
).to_dict()

result = await rollback_manager.evaluate_experience(exp)
# -> action: DOWNGRADE (降级，降低权重)

# 7. 长期未使用 (90 天+)
exp['time_tags'] = TimeTags(
    created_at=datetime.utcnow() - timedelta(days=120),
    last_used_at=datetime.utcnow() - timedelta(days=120)
).to_dict()

result = await rollback_manager.evaluate_experience(exp)
# -> action: ARCHIVE (归档)

# 8. 执行回滚
success = rollback_manager.execute_rollback(result, experience_store)
```

## 统计监控

### 时间标签管理器统计
```python
stats = time_manager.get_stats()
```

### 回滚管理器统计
```python
stats = rollback_manager.get_stats()
# {
#     'total_evaluations': 100,
#     'downgrades': 20,
#     'archives': 10,
#     'deletes': 5,
#     'reactivations': 15,
#     'no_action': 50
# }
```

### 回滚摘要
```python
summary = rollback_manager.get_rollback_summary(results)
# {
#     'total': 100,
#     'by_action': {
#         'downgrade': 20,
#         'archive': 10,
#         'delete': 5,
#         'reactivate': 15,
#         'none': 50
#     },
#     'recommendations': [...]
# }
```

## 最佳实践

### 1. 配置衰减策略
```python
# 高频使用场景：短半衰期
strategy = TimeDecayStrategy(
    half_life_days=15,  # 15 天半衰期
    min_weight=0.2
)

# 低频使用场景：长半衰期
strategy = TimeDecayStrategy(
    half_life_days=60,  # 60 天半衰期
    min_weight=0.05
)
```

### 2. 配置回滚阈值
```python
# 保守策略：延迟降级/归档
strategy = RollbackStrategy(
    downgrade_days=60,
    archive_days=180,
    delete_days=365
)

# 激进策略：快速降级/归档
strategy = RollbackStrategy(
    downgrade_days=15,
    archive_days=45,
    delete_days=90
)
```

### 3. 批量处理
```python
# 分批处理大量经验
experiences = [...]  # 1000 个经验

# 每批 100 个
results = await rollback_manager.evaluate_batch(
    experiences,
    batch_size=100
)
```

## 故障排查

### 问题 1: 经验未正确降级
**检查**:
1. 时间标签是否正确更新
2. 最后使用时间是否准确
3. 回滚策略阈值配置

### 问题 2: 回调未触发
**检查**:
1. 回调是否正确注册
2. 回调函数是否为 async
3. 检查异常日志

### 问题 3: 批量评估失败
**检查**:
1. 经验数据格式是否正确
2. time_tags 字段是否存在
3. 批次大小是否合理

## 测试

运行测试脚本:
```bash
python tests/test_time_tags_rollback.py
```

测试覆盖:
- ✅ Time Tags System
- ✅ Time Decay Strategy
- ✅ Rollback Strategy
- ✅ Rollback Manager
- ✅ Time Tag Manager
- ✅ Integration (生命周期模拟)

Total: 6/6 passed

## 文件结构

```
zulong/memory/
├── time_tags.py             # 时间标签体系
├── rollback.py              # 降智回滚机制
└── __init__.py              # 模块导出

tests/
└── test_time_tags_rollback.py  # 测试脚本
```

## 与复盘机制集成

```python
from zulong.review import get_review_trigger, TriggerType
from zulong.memory import get_rollback_manager

# 1. 获取组件
trigger = get_review_trigger()
rollback_manager = get_rollback_manager()

# 2. 注册复盘回调
async def review_and_rollback(request):
    # 复盘触发后，执行回滚评估
    experiences = get_all_experiences()
    results = await rollback_manager.evaluate_batch(experiences)
    
    # 执行回滚
    for result in results:
        if result.action != RollbackAction.NONE:
            rollback_manager.execute_rollback(result, experience_store)

trigger.register_callback(TriggerType.NIGHT_SCHEDULE, review_and_rollback)

# 3. 启动
await trigger.start()
```

## 下一步

1. ✅ 复盘机制 (三重触发 + 防重复) - **已完成**
2. ✅ 时间标签体系与降智回滚 - **已完成**

## 参考资料

- TSD v2.3 第 12 章：时间标签与降智回滚
- [复盘机制与智能经验库系统架构升级原子任务](../资料/复盘机制与智能经验库系统架构升级原子任务.txt)
