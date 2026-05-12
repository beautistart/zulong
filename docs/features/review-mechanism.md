# 复盘机制系统文档

## 概述

复盘机制是祖龙系统自我进化的核心引擎，通过三重触发、经验分类处理和三重防重复机制，实现系统能力的持续提升。

对应 TSD v2.3 第 11 章

## 系统架构

```
复盘机制系统
├── 三重触发器 (trigger.py)
│   ├── 用户主动触发 (高优先级)
│   ├── 安静模式触发 (中优先级)
│   └── 夜间定时触发 (低优先级)
├── 经验分类处理
│   ├── 成功经验提炼器 (success_extractor.py)
│   └── 失败案例分析器 (failure_analyzer.py)
└── 三重防重复机制 (deduplication.py)
    ├── 事件级过滤
    ├── 内容级过滤
    └── 时间级过滤
```

## 核心模块

### 1. 三重触发器 (ReviewTrigger)

**功能**:
- 用户主动触发：用户明确说"复盘"、"总结一下"等
- 安静模式触发：30 分钟无交互自动触发
- 夜间定时触发：每天凌晨 2 点自动触发

**优先级调度**:
```python
TriggerPriority.HIGH      # 用户主动 (立即执行)
TriggerPriority.MEDIUM    # 安静模式 (延迟 5 分钟)
TriggerPriority.LOW       # 夜间定时 (延迟 30 分钟)
```

**使用示例**:
```python
from zulong.review import get_review_trigger, TriggerType

trigger = get_review_trigger(
    quiet_mode_timeout_minutes=30,
    night_trigger_hour=2,
    night_trigger_minute=0
)

# 注册回调
async def review_callback(request):
    print(f"复盘触发：{request['type']}")
    
trigger.register_callback(TriggerType.USER_ACTIVE, review_callback)

# 启动触发器
await trigger.start()
```

### 2. 成功经验提炼器 (SuccessExperienceExtractor)

**功能**:
- 任务描述提取
- 关键步骤识别
- 成功因素分析
- 结构化经验生成

**提取流程**:
1. 识别用户任务描述
2. 提取 assistant 的关键步骤
3. 识别成功确认标记
4. 生成结构化经验

**使用示例**:
```python
from zulong.review import get_success_extractor

extractor = get_success_extractor()

dialog_history = [
    {'role': 'user', 'content': '如何连接 WiFi?'},
    {'role': 'assistant', 'content': '1. 打开设置\n2. 点击网络\n3. 选择 WiFi'},
    {'role': 'user', 'content': '成功了!'}
]

experience = extractor.extract_from_dialog(
    dialog_history=dialog_history,
    success_marker='成功'
)
```

### 3. 失败案例分析器 (FailureAnalyzer)

**功能**:
- 错误归因分析 (能力不足/环境限制/指令错误)
- 避坑指南生成
- 权重策略 (失败案例权重 1.5 倍)
- 失败模式识别

**错误类型**:
```python
'capability_limit'    # 能力不足
'environment_limit'   # 环境限制
'instruction_error'   # 指令错误
'timeout'             # 超时
'resource_error'      # 资源错误
'unknown'             # 未知
```

**使用示例**:
```python
from zulong.review import get_failure_analyzer

analyzer = get_failure_analyzer()

case = analyzer.analyze_from_error(
    error_message='网络错误：无法连接到服务器',
    task_description='下载文件'
)

print(f"错误类型：{case.error_type}")
print(f"根本原因：{case.root_cause}")
print(f"避坑指南：{case.avoidance_guide}")
```

### 4. 三重防重复机制 (DeduplicationFilter)

**功能**:
- 事件级过滤：失败必复盘、成功抽样 (10%)
- 内容级过滤：向量查重 (相似度>0.95)
- 时间级过滤：1 小时窗口聚合

**过滤规则**:
1. **事件级**: 失败事件必须复盘，成功事件抽样
2. **内容级**: MD5 哈希查重 + 向量相似度
3. **时间级**: 同一任务 1 小时内最多 5 次复盘

**使用示例**:
```python
from zulong.review import get_dedup_filter

dedup = get_dedup_filter(
    similarity_threshold=0.95,
    time_window_minutes=60,
    success_sampling_rate=0.1
)

# 判断是否需要复盘
should_review, reason = dedup.should_review(
    event_data={'task_description': '连接 WiFi'},
    event_type='failure'
)

print(f"是否复盘：{should_review}")
print(f"原因：{reason}")
```

## 完整工作流

```python
from zulong.review import (
    get_review_trigger,
    get_success_extractor,
    get_failure_analyzer,
    get_dedup_filter,
    TriggerType
)

# 1. 初始化组件
trigger = get_review_trigger()
extractor = get_success_extractor()
analyzer = get_failure_analyzer()
dedup = get_dedup_filter()

# 2. 注册复盘回调
async def review_handler(request):
    event_data = request.get('context', {})
    event_type = event_data.get('type', 'failure')
    
    # 3. 防重复检查
    should_review, reason = dedup.should_review(event_data, event_type)
    
    if not should_review:
        print(f"跳过复盘：{reason}")
        return
    
    # 4. 经验处理
    if event_type == 'success':
        experience = extractor.extract_from_dialog(
            dialog_history=event_data.get('dialog', [])
        )
        if experience:
            extractor.save_to_experience_store(experience)
    else:
        case = analyzer.analyze_from_error(
            error_message=event_data.get('error', ''),
            task_description=event_data.get('task', '')
        )
        if case:
            analyzer.save_to_experience_store(case)

trigger.register_callback(TriggerType.USER_ACTIVE, review_handler)

# 5. 启动
await trigger.start()
```

## 统计监控

### 触发器统计
```python
stats = trigger.get_stats()
# {
#     'total_triggers': 10,
#     'user_active_count': 5,
#     'quiet_mode_count': 3,
#     'night_schedule_count': 2,
#     'failed_triggers': 0
# }
```

### 防重复统计
```python
stats = dedup.get_stats()
# {
#     'total_events': 100,
#     'filtered_by_event': 60,
#     'filtered_by_content': 20,
#     'filtered_by_time': 10,
#     'passed': 10
# }
```

## 最佳实践

### 1. 触发器配置
```python
# 开发环境：短超时，便于测试
trigger = get_review_trigger(
    quiet_mode_timeout_minutes=5,
    night_trigger_hour=3,
    night_trigger_minute=0
)

# 生产环境：标准配置
trigger = get_review_trigger(
    quiet_mode_timeout_minutes=30,
    night_trigger_hour=2,
    night_trigger_minute=0
)
```

### 2. 抽样率调整
```python
# 高频使用场景：降低抽样率
dedup = get_dedup_filter(
    success_sampling_rate=0.05  # 5%
)

# 低频使用场景：提高抽样率
dedup = get_dedup_filter(
    success_sampling_rate=0.2  # 20%
)
```

### 3. 时间窗口调整
```python
# 快速迭代：短时间窗口
dedup = get_dedup_filter(
    time_window_minutes=30
)

# 稳定运行：长时间窗口
dedup = get_dedup_filter(
    time_window_minutes=120
)
```

## 故障排查

### 问题 1: 复盘未触发
**检查**:
1. 触发器是否启动：`trigger.is_running()`
2. 回调是否注册：检查 `register_callback`
3. 优先级是否正确：HIGH > MEDIUM > LOW

### 问题 2: 重复复盘
**检查**:
1. 防重复过滤器是否启用
2. 时间窗口配置是否合理
3. 内容哈希是否正确计算

### 问题 3: 经验未保存
**检查**:
1. 经验库是否初始化
2. experience_store 是否传入
3. 检查异常日志

## 测试

运行测试脚本:
```bash
python tests/test_review_mechanism.py
```

测试覆盖:
- ✅ 三重触发机制
- ✅ 成功经验提炼
- ✅ 失败案例分析
- ✅ 三重防重复
- ✅ 集成测试

## 文件结构

```
zulong/review/
├── __init__.py              # 模块导出
├── trigger.py               # 三重触发器
├── success_extractor.py     # 成功经验提炼器
├── failure_analyzer.py      # 失败案例分析器
└── deduplication.py         # 防重复过滤器

tests/
└── test_review_mechanism.py # 测试脚本
```

## 下一步

1. ✅ 复盘机制 (三重触发 + 防重复) - **已完成**
2. ⏳ 时间标签体系与降智回滚 - 进行中

## 参考资料

- TSD v2.3 第 11 章：复盘机制
- [复盘机制与智能经验库系统架构升级原子任务](../资料/复盘机制与智能经验库系统架构升级原子任务.txt)
