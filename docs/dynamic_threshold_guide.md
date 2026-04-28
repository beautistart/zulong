# TSD v2.4 动态阈值管理使用指南

**版本**: 1.0  
**日期**: 2026-04-10  
**对应架构**: TSD v2.4 资源自适应规范

---

## 📋 概述

动态阈值管理器（DynamicThresholdManager）是 TSD v2.4 架构的核心组件，用于根据模型配置和系统负载**实时计算**记忆复盘触发阈值。

### 核心优势

1. **资源自适应**：根据模型上下文窗口自动调整阈值
2. **智能熔断**：显存超限（>95%）时强制触发复盘
3. **混合触发**：支持 Token 数、轮次数、时间衰减、长文本多种触发方式
4. **零侵入性**：与现有记忆系统完全兼容

---

## 🚀 快速开始

### 1. 系统启动时初始化

在 `bootstrap.py` 或系统启动脚本中：

```python
from zulong.l1b.memory_config_initializer import initialize_memory_config
from zulong.models.config import ModelID

# 初始化动态阈值（从 L2_PRIME 模型配置读取）
initialize_memory_config(ModelID.L2_PRIME)

# 或使用异步版本
# await async_initialize_memory_config(ModelID.L2_PRIME)
```

### 2. 自动集成（无需额外代码）

动态阈值已自动集成到以下组件：

- ✅ `ShortTermMemory` - 短期记忆
- ✅ `EpisodicMemory` - 情景记忆

**无需修改现有代码**，系统会自动使用动态阈值！

---

## 📊 动态阈值计算逻辑

### 硬上限（Token 数）

```python
硬上限 = ContextWindow_max × SafetyFactor

# SafetyFactor 根据量化级别自动调整：
# - INT4 量化：0.85（显存宽裕）
# - INT8 量化：0.80
# - FP16/FP32: 0.75（标准）
```

**示例**：

| 模型 | 上下文窗口 | 量化 | 硬上限 |
|------|-----------|------|--------|
| Qwen-0.8B | 4096 | INT4 | **3481 tokens** |
| Llama-3-8B | 8192 | INT4 | **6963 tokens** |
| Qwen-72B | 32768 | FP16 | **24576 tokens** |

### 软上限（轮次数）

```python
软上限 = BaseTurns × SpeedFactor

# SpeedFactor 根据模型大小调整：
# - >70B: 0.5（超大模型，尽早复盘）
# - >30B: 0.7（大模型）
# - >14B: 0.9（中型模型）
# - <14B: 1.0（小模型，多保留轮次）
```

**示例**：

| 模型大小 | SpeedFactor | 软上限 |
|---------|------------|--------|
| 0.8B | 1.0 | **10 轮** |
| 8B | 1.0 | **10 轮** |
| 14B | 0.9 | **9 轮** |
| 72B | 0.5 | **5 轮** |

---

## 🔍 触发条件详解

### 1. Token 容量超限（硬触发）

```python
当前 Token 数 >= 硬上限
→ 立即触发复盘
```

### 2. 轮次数超限（软触发）

```python
当前轮数 >= 软上限
→ 触发复盘
```

### 3. 时间衰减触发

```python
无活动时间 > 180 秒（3 分钟）
→ 触发复盘（话题结束）
```

### 4. 长文本输入检测

```python
用户输入 Token 数 > 1000
→ 立即触发复盘（为长文本腾出空间）
```

### 5. 显存紧急熔断

```python
显存使用率 > 95%
→ 进入紧急模式，强制下调阈值 20%
→ 立即触发复盘
```

---

## 📈 监控与调试

### 查看当前阈值

```python
from zulong.l1b.dynamic_threshold_manager import get_dynamic_threshold_manager

manager = get_dynamic_threshold_manager()
thresholds = manager.get_thresholds()

print(f"硬上限：{thresholds.hard_token_limit} tokens")
print(f"软上限：{thresholds.soft_turn_limit} 轮")
print(f"显存使用：{thresholds.vram_usage*100:.1f}%")
print(f"紧急模式：{thresholds.is_emergency_mode}")
```

### 查看记忆统计

```python
from zulong.memory.short_term_memory import ShortTermMemory

stm = ShortTermMemory()
stats = stm.get_stats()

print(f"当前轮数：{stats['current_turn']}")
print(f"Token 计数：{stats['token_counter']}")
print(f"硬上限：{stats['hard_token_limit']}")
print(f"软上限：{stats['soft_turn_limit']}")
```

### 日志监控

系统会自动输出详细日志：

```
[动态阈值] 检测到长时间无活动 (185.3 秒)，触发复盘
🚨 [动态阈值] 触发复盘：token_limit_exceeded
  - 当前 Token: 3520 / 3481
  - 当前轮数：12 / 10
🚨 [DynamicThresholdManager] 进入紧急模式！显存使用率：96.5%
  - 硬上限已下调：3481 → 2784
  - 软上限已下调：10 → 6
```

---

## 🛠️ 高级配置

### 自定义阈值参数

```python
from zulong.l1b.dynamic_threshold_manager import get_dynamic_threshold_manager

manager = get_dynamic_threshold_manager()

# 调整长文本检测阈值
manager.long_text_threshold = 1500  # 默认 1000 tokens

# 调整时间衰减阈值
from zulong.memory.short_term_memory import ShortTermMemory
stm = ShortTermMemory()
stm.inactivity_threshold = 300  # 默认 180 秒（改为 5 分钟）
```

### 注册回调函数

```python
def on_emergency_trigger(reason: str):
    """紧急复盘触发时的回调"""
    print(f"🚨 紧急触发：{reason}")

manager = get_dynamic_threshold_manager()
manager.register_emergency_trigger_callback(on_emergency_trigger)
```

---

## 📊 性能优化建议

### 小模型场景（<10B）

```yaml
特点：
  - 推理速度快
  - 显存占用低
  - 可多保留轮次

建议配置：
  - 软上限：12-15 轮
  - 硬上限：6000-8000 tokens
  - 时间衰减：300 秒（5 分钟）
```

### 大模型场景（>30B）

```yaml
特点：
  - 推理速度慢
  - 显存极贵
  - 应尽早复盘

建议配置：
  - 软上限：5-8 轮
  - 硬上限：20000-24000 tokens
  - 时间衰减：120 秒（2 分钟）
```

---

## 🎯 最佳实践

### 1. 系统启动时初始化

```python
# ✅ 推荐：在 bootstrap.py 中初始化
initialize_memory_config(ModelID.L2_PRIME)

# ❌ 不推荐：在使用时懒加载
# 会导致首次计算延迟
```

### 2. 监控显存状态

```python
# ✅ 推荐：启动显存监控线程
from zulong.l1b.memory_config_initializer import memory_config_initializer
memory_config_initializer._start_vram_monitor()

# ❌ 不推荐：忽略显存监控
# 可能导致 OOM
```

### 3. 定期检查阈值

```python
# ✅ 推荐：在关键节点检查阈值
if manager.is_emergency_mode:
    logger.warning("当前处于紧急模式，建议减少上下文")

# ❌ 不推荐：忽略紧急模式
```

---

## 🔧 故障排查

### 问题 1：阈值未生效

**症状**：系统仍使用固定阈值（如 3500 tokens）

**解决方案**：
```python
# 1. 检查是否已初始化
from zulong.l1b.memory_config_initializer import memory_config_initializer
if not hasattr(memory_config_initializer, '_initialized'):
    initialize_memory_config()

# 2. 检查模型配置是否正确
from zulong.models.config import get_model_config
config = get_model_config(ModelID.L2_PRIME)
print(config)
```

### 问题 2：显存监控不工作

**症状**：`vram_usage` 始终为 0.5（默认值）

**解决方案**：
```python
# 1. 检查是否安装 pynvml
pip install pynvml

# 2. 或检查是否安装 torch
pip install torch

# 3. 修改 _get_gpu_vram_usage() 使用可用的监控库
```

### 问题 3：频繁触发复盘

**症状**：每轮对话都触发复盘

**解决方案**：
```python
# 1. 检查模型配置是否过小
thresholds = manager.get_thresholds()
print(f"软上限：{thresholds.soft_turn_limit}")  # 如果<5，说明模型配置有问题

# 2. 调整安全系数
manager.safety_factor = 0.85  # 提高安全系数
```

---

## 📚 相关文档

- [TSD v2.4 架构规范](../docs/TSD_v2.4.md)
- [记忆系统架构分析](./memory_architecture_analysis.md)
- [DynamicThresholdManager API 文档](../api/l1b/dynamic_threshold_manager.md)

---

## 🎉 总结

动态阈值管理器是 TSD v2.4"资源自适应"理念的核心实现，它确保祖龙系统能够在**任何模型配置**下都能找到**性能与成本的最优平衡点**。

**核心优势**：
- ✅ 自动适配不同规模的模型
- ✅ 智能应对显存危机
- ✅ 零侵入性集成
- ✅ 完善的监控与调试支持

**下一步**：
1. 在测试环境验证动态阈值效果
2. 根据实际负载微调参数
3. 集成到生产环境

---

**文档版本**: 1.0  
**最后更新**: 2026-04-10  
**维护者**: ZULONG Team
