# LLM自主注意力模式选择 - 实现总结报告

**生成时间**: 2026-05-17  
**实现状态**: ✅ 核心模块完成  
**后续工作**: 系统集成、测试验证

---

## 一、实现概览

### 已完成模块

| 阶段 | 模块 | 状态 | 文件 |
|------|------|------|------|
| 阶段1 | 数据模型与配置管理 | ✅ 完成 | `attention_types.py`, `attention_config.py` |
| 阶段2 | 压力检测模块 | ✅ 完成 | `pressure_detector.py` |
| 阶段3 | LLM决策模块 | ✅ 完成 | `attention_mode_selector.py` |
| 阶段4 | 控制机制模块 | ✅ 完成 | `mode_switch_controller.py` |
| 阶段5 | 系统集成 | ⏳ 待实施 | 需扩展`attention_window.py` |
| 阶段6 | 测试验证 | ⏳ 待实施 | 需编写测试用例 |

---

## 二、核心模块详解

### 2.1 数据模型 (`attention_types.py`)

**枚举类型**:
- `PressureTrend`: RISING/STABLE/FALLING (压力趋势)
- `TriggerType`: TOOL_DRIVEN/LLM_AUTONOMOUS/FALLBACK (触发类型)
- `OscillationLevel`: NONE/SLIGHT/OBVIOUS/SEVERE (震荡级别)

**数据类**:
- `PressureMetrics`: 压力指标 (current_pressure, trend, velocity, predicted)
- `DecisionRequest`: 决策请求 (pressure_metrics, current_mode, task_context)
- `DecisionResponse`: 决策响应 (mode, reason, confidence, is_fallback)
- `SwitchRecord`: 切换记录 (old_mode, new_mode, trigger_type, timestamp)
- `OscillationState`: 震荡状态 (is_oscillating, level, pattern)
- `ThresholdCheckResult`: 阈值检查结果
- `CooldownCheckResult`: 冷却检查结果
- `ModeSwitchResult`: 模式切换结果

---

### 2.2 配置管理 (`attention_config.py`)

**AttentionConfig配置类**:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| enabled | True | 功能开关 |
| pressure_threshold_high | 0.9 | 高压阈值 |
| pressure_threshold_medium | 0.75 | 中压阈值 |
| cooldown_base_seconds | 30.0 | 基础冷却时间 |
| fallback_mode | FOCUS | Fallback模式 |
| decision_timeout_ms | 500 | 决策超时 |
| oscillation_detection_window | 10 | 震荡检测窗口 |

**核心方法**:
- `load_from_yaml()`: 从YAML配置文件加载
- `validate()`: 参数验证与自动修正
- `to_dict()`: 导出为字典

---

### 2.3 压力检测 (`pressure_detector.py`)

**PressureDetector类**:

```python
class PressureDetector:
    def calculate_pressure() -> PressureMetrics
    def check_threshold(metrics) -> ThresholdCheckResult
    def _calculate_velocity() -> float
    def _predict_pressure() -> float
    def _determine_trend() -> PressureTrend
```

**压力计算公式**:
```
pressure_value = total_tokens / current_budget
velocity = (pressure_new - pressure_old) / time_diff
predicted = current + velocity * 5.0
```

**阈值判断逻辑**:
- 压力 ≥ 高压阈值 → 触发LLM选择
- 压力 ≥ 中压阈值 且 趋势上升 → 触发LLM选择
- 预测压力 ≥ 高压阈值 → 触发LLM选择

---

### 2.4 LLM决策 (`attention_mode_selector.py`)

**AttentionModeSelector类**:

```python
class AttentionModeSelector:
    def build_decision_request() -> DecisionRequest
    async def call_llm_decision() -> DecisionResponse
    def parse_decision_response() -> DecisionResponse
```

**Prompt模板结构**:
```
1. 当前压力指标 (压力值、趋势、预算使用率、消息数量)
2. 当前状态 (当前模式、任务上下文)
3. 可选注意力模式 (GLOBAL/FOCUS/SINGLE_CHAIN的说明)
4. 近期切换历史
5. 决策要求 (返回JSON格式)
```

**决策解析策略**:
1. 优先JSON解析 (`json.loads()`)
2. Fallback正则匹配 (`\bGLOBAL\b`, `\bFOCUS\b`, `\bSINGLE.?CHAIN\b`)
3. 最终Fallback返回默认模式

---

### 2.5 控制机制 (`mode_switch_controller.py`)

**CooldownManager冷却管理器**:
- 检查是否处于冷却期
- 动态调整冷却因子 (1.0 ~ 5.0)
- 震荡时延长冷却时间

**OscillationDetector震荡检测器**:
- 检测ABA震荡模式 → 冷却因子 × 1.5
- 检测ABAB震荡模式 → 冷却因子 × 2.0
- 检测频繁切换 → 冷却因子 × 1.2

**ModeSwitchController切换控制器**:
- 协调所有组件执行切换
- 记录切换历史
- 应用震荡缓解策略

---

## 三、集成方案

### 3.1 AttentionWindowManager扩展

需要在`attention_window.py`的`AttentionWindowManager`类中添加:

```python
class AttentionWindowManager:
    def __init__(self, ...):
        # 新增组件初始化
        self._config = AttentionConfig.load_from_yaml()
        self._pressure_detector = PressureDetector(self, self._config)
        self._mode_selector = AttentionModeSelector(self._config, llm_client)
        self._cooldown_manager = CooldownManager(self._config)
        self._oscillation_detector = OscillationDetector(self._config)
        self._mode_controller = ModeSwitchController(
            self._config, 
            self._cooldown_manager, 
            self._oscillation_detector
        )
    
    def apply_window(self) -> List[Dict]:
        # 新增：LLM自主选择检查
        if self._should_try_llm_selection():
            self._try_llm_mode_selection()
        
        # 原有裁剪逻辑...
```

### 3.2 新增方法

```python
def _should_try_llm_selection(self) -> bool:
    """检查是否应尝试LLM自主选择"""
    if not self._config.enabled:
        return False
    
    cooldown_result = self._mode_controller.can_switch()
    if not cooldown_result.is_allowed:
        return False
    
    return True

async def _try_llm_mode_selection(self):
    """尝试LLM自主选择注意力模式"""
    # 1. 计算压力
    metrics = self._pressure_detector.calculate_pressure()
    
    # 2. 检查阈值
    threshold_result = self._pressure_detector.check_threshold(metrics)
    if not threshold_result.should_trigger:
        return
    
    # 3. 构建决策请求
    task_context = self._get_task_context_summary()
    request = self._mode_selector.build_decision_request(
        metrics, self.mode.name, task_context,
        self._mode_controller.get_switch_history()
    )
    
    # 4. 调用LLM决策
    decision = await self._mode_selector.call_llm_decision(request)
    
    # 5. 应用决策
    if decision.mode != self.mode.name:
        result = self._mode_controller.apply_llm_decision(
            self.mode.name, decision, metrics.current_pressure
        )
        if result.switched:
            self.mode = AttentionMode[result.new_mode]
```

---

## 四、关键设计亮点

### 4.1 多维度压力评估

```
压力值 + 压力趋势 + 压力变化速率 + 预测压力 = 综合判断
```

### 4.2 结构化Prompt

- 明确的任务和约束
- 三种模式的详细说明
- JSON格式响应要求
- 历史决策参考

### 4.3 多层防护机制

```
冷却时间 → 震荡检测 → Fallback机制 → 默认模式
```

### 4.4 向后兼容

- 工具驱动模式切换不受影响
- 功能可配置启用/禁用
- 不改变现有推理流程

---

## 五、性能指标

| 指标 | 目标值 | 实现情况 |
|------|--------|----------|
| 压力检测延迟 | < 5ms | ✅ 实现，有性能日志监控 |
| LLM决策超时 | 500ms | ✅ 实现，asyncio.wait_for() |
| 冷却时间 | 30秒起 | ✅ 实现，动态调整 |
| 震荡检测 | 实时 | ✅ 实现，ABA/ABAB模式检测 |

---

## 六、后续工作

### 阶段5：系统集成 (预估3天)

- [ ] 扩展`AttentionWindowManager`类
- [ ] 修改`apply_window()`集成点
- [ ] 配置文件扩展 (`config/zulong_config.yaml`)
- [ ] LLM客户端注入

### 阶段6：测试验证 (预估2天)

- [ ] 单元测试 (PressureDetector, AttentionModeSelector等)
- [ ] 集成测试 (完整流程测试)
- [ ] 性能测试 (延迟验证)
- [ ] 震荡场景测试

---

## 七、文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `zulong/l2/attention_types.py` | ~230 | 数据模型定义 |
| `zulong/l2/attention_config.py` | ~150 | 配置管理 |
| `zulong/l2/pressure_detector.py` | ~200 | 压力检测 |
| `zulong/l2/attention_mode_selector.py` | ~280 | LLM决策 |
| `zulong/l2/mode_switch_controller.py` | ~260 | 控制机制 |

**总计**: ~1120行代码

---

**实现完成时间**: 2026-05-17  
**核心功能**: ✅ 已实现  
**集成状态**: ⏳ 待集成到AttentionWindowManager
