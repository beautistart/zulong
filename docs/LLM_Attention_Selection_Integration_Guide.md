# LLM自主注意力模式选择 - 集成指南

**版本**: v1.0  
**日期**: 2026-05-17  
**状态**: ✅ 核心模块已完成，待集成到AttentionWindowManager

---

## 一、集成概览

### 已完成工作

| 模块 | 文件 | 状态 |
|------|------|------|
| 数据模型 | `zulong/l2/attention_types.py` | ✅ 完成 |
| 配置管理 | `zulong/l2/attention_config.py` | ✅ 完成 |
| 压力检测 | `zulong/l2/pressure_detector.py` | ✅ 完成 |
| LLM决策 | `zulong/l2/attention_mode_selector.py` | ✅ 完成 |
| 控制机制 | `zulong/l2/mode_switch_controller.py` | ✅ 完成 |
| 集成扩展 | `zulong/l2/attention_integration.py` | ✅ 完成 |
| 配置文件 | `config/zulong_config.yaml` | ✅ 已添加 |

### 待集成工作

- [ ] 修改`zulong/l2/attention_window.py`，添加LLM选择支持
- [ ] 在`InferenceEngine`中设置LLM客户端
- [ ] 测试验证

---

## 二、集成步骤

### 步骤1：添加导入语句

在`zulong/l2/attention_window.py`文件顶部添加：

```python
# LLM自主注意力选择导入
from .attention_types import (
    PressureMetrics, DecisionRequest, DecisionResponse,
    SwitchRecord, OscillationState, TriggerType,
    PressureTrend, OscillationLevel,
)
from .attention_config import AttentionConfig
from .pressure_detector import PressureDetector
from .attention_mode_selector import AttentionModeSelector
from .mode_switch_controller import (
    ModeSwitchController, CooldownManager, OscillationDetector
)
```

---

### 步骤2：扩展__init__()方法

在`AttentionWindowManager.__init__()`方法末尾添加：

```python
# ── LLM自主注意力选择组件初始化 ──
try:
    self._llm_config = AttentionConfig.load_from_yaml()
    self._llm_config.validate()
    
    if self._llm_config.enabled:
        self._pressure_detector = PressureDetector(self, self._llm_config)
        self._mode_selector = AttentionModeSelector(self._llm_config)
        
        self._cooldown_manager = CooldownManager(self._llm_config)
        self._oscillation_detector = OscillationDetector(self._llm_config)
        self._mode_controller = ModeSwitchController(
            self._llm_config,
            self._cooldown_manager,
            self._oscillation_detector
        )
        
        self._llm_client = None  # 后续通过set_llm_client()设置
        self._last_llm_selection_time = None
        
        logger.info(
            f"[AttentionWindow] LLM自主注意力选择已启用: "
            f"pressure_threshold={self._llm_config.pressure_threshold_high}"
        )
    else:
        self._llm_config = None
        logger.info("[AttentionWindow] LLM自主注意力选择已禁用")
        
except Exception as e:
    logger.error(f"[AttentionWindow] LLM注意力选择初始化失败: {e}")
    self._llm_config = None
```

---

### 步骤3：修改apply_window()方法

在`apply_window()`方法开头添加LLM选择检查：

```python
def apply_window(self) -> List[Dict]:
    # ── 新增：LLM自主注意力选择检查 ──
    if self._should_try_llm_selection():
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._try_llm_mode_selection_async())
            else:
                loop.run_until_complete(self._try_llm_mode_selection_async())
        except Exception as e:
            logger.warning(f"[AttentionWindow] LLM注意力选择执行失败: {e}")
    
    # ── 原有裁剪逻辑继续执行 ──
    if not self.envelopes:
        return []
    # ... (保持原有代码不变)
```

---

### 步骤4：添加辅助方法

在`AttentionWindowManager`类中添加以下方法：

#### 4.1 set_llm_client()

```python
def set_llm_client(self, llm_client):
    """设置LLM客户端实例
    
    Args:
        llm_client: LLM客户端实例（需支持chat或generate方法）
    """
    if self._llm_config and self._llm_config.enabled:
        self._llm_client = llm_client
        self._mode_selector.set_llm_client(llm_client)
        logger.info("[AttentionWindow] LLM客户端已设置")
```

#### 4.2 _should_try_llm_selection()

```python
def _should_try_llm_selection(self) -> bool:
    """检查是否应尝试LLM自主选择
    
    Returns:
        是否应尝试
    """
    if not self._llm_config or not self._llm_config.enabled:
        return False
    
    if not self._llm_client:
        return False
    
    cooldown_result = self._mode_controller.can_switch()
    if not cooldown_result.is_allowed:
        logger.debug(f"[AttentionWindow] 冷却期未过: {cooldown_result.message}")
        return False
    
    return True
```

#### 4.3 _try_llm_mode_selection_async()

```python
async def _try_llm_mode_selection_async(self):
    """异步执行LLM自主注意力模式选择"""
    try:
        # 1. 计算压力
        metrics = self._pressure_detector.calculate_pressure()
        
        # 2. 检查阈值
        threshold_result = self._pressure_detector.check_threshold(metrics)
        if not threshold_result.should_trigger:
            logger.debug(f"[AttentionWindow] 压力未超阈值: {threshold_result.message}")
            return
        
        logger.info(
            f"[AttentionWindow] 触发LLM注意力选择: {threshold_result.trigger_type} "
            f"(压力={metrics.current_pressure:.3f})"
        )
        
        # 3. 构建决策请求
        task_context = self._get_task_context_summary()
        current_mode = self.mode.name
        switch_history = self._mode_controller.get_switch_history(5)
        
        request = self._mode_selector.build_decision_request(
            pressure_metrics=metrics,
            current_mode=current_mode,
            task_context=task_context,
            switch_history=switch_history,
        )
        
        # 4. 调用LLM决策
        decision = await self._mode_selector.call_llm_decision(request)
        
        logger.info(
            f"[AttentionWindow] LLM决策结果: mode={decision.mode}, "
            f"confidence={decision.confidence:.2f}, fallback={decision.is_fallback}"
        )
        
        # 5. 应用决策
        if decision.mode != current_mode:
            result = self._mode_controller.apply_llm_decision(
                current_mode=current_mode,
                decision=decision,
                pressure_value=metrics.current_pressure,
            )
            
            if result.switched:
                try:
                    new_mode = AttentionMode[decision.mode]
                    self.mode = new_mode
                    logger.info(
                        f"[AttentionWindow] 注意力模式已切换: "
                        f"{current_mode} → {decision.mode}"
                    )
                except KeyError:
                    logger.error(f"[AttentionWindow] 无效的模式名称: {decision.mode}")
        else:
            logger.info(f"[AttentionWindow] LLM选择保持当前模式: {current_mode}")
            
    except Exception as e:
        logger.error(f"[AttentionWindow] LLM注意力选择流程异常: {e}")
```

#### 4.4 _get_task_context_summary()

```python
def _get_task_context_summary(self) -> str:
    """获取任务上下文摘要（供LLM决策参考）
    
    Returns:
        任务上下文摘要文本（≤500字符）
    """
    summary_parts = []
    
    if self.task_graph:
        try:
            node_count = self.task_graph.node_count if hasattr(self.task_graph, 'node_count') else 0
            summary_parts.append(f"任务图节点数: {node_count}")
            
            if self._current_node_id:
                summary_parts.append(f"当前节点: {self._current_node_id}")
        except Exception:
            pass
    
    message_count = len(self.envelopes)
    total_tokens = sum(e.tokens for e in self.envelopes)
    summary_parts.append(f"消息数量: {message_count}")
    summary_parts.append(f"总tokens: {total_tokens}")
    
    recent_tools = []
    for env in self.envelopes[-5:]:
        if env.tool_name:
            recent_tools.append(env.tool_name)
    if recent_tools:
        summary_parts.append(f"近期工具: {', '.join(recent_tools)}")
    
    summary = " | ".join(summary_parts)
    return summary[:500]
```

#### 4.5 get_llm_selection_stats()

```python
def get_llm_selection_stats(self) -> Dict:
    """获取LLM注意力选择的统计信息
    
    Returns:
        统计信息字典
    """
    if not self._llm_config:
        return {"enabled": False}
    
    history = self._mode_controller.get_switch_history(10)
    llm_switches = [r for r in history if r.trigger_type == TriggerType.LLM_AUTONOMOUS]
    fallback_switches = [r for r in history if r.trigger_type == TriggerType.FALLBACK]
    
    return {
        "enabled": self._llm_config.enabled,
        "current_mode": self.mode.name,
        "total_switches": len(history),
        "llm_autonomous_switches": len(llm_switches),
        "fallback_switches": len(fallback_switches),
        "cooldown_seconds": self._cooldown_manager.get_current_cooldown_seconds(),
        "switch_history": [r.to_dict() for r in history[-5:]],
    }
```

---

### 步骤5：在InferenceEngine中集成

在`zulong/l2/inference_engine.py`中：

```python
class InferenceEngine:
    def __init__(self, ...):
        # ... 原有初始化代码 ...
        
        # 初始化AttentionWindowManager
        self._attn_window = AttentionWindowManager(
            context_window_size=128000,
            task_graph=self.task_graph,
            memory_graph=self.memory_graph,
        )
        
        # 设置LLM客户端（使能LLM自主选择）
        if hasattr(self, '_llm_client'):
            self._attn_window.set_llm_client(self._llm_client)
```

---

## 三、配置说明

配置文件位置：`config/zulong_config.yaml`

```yaml
attention_selection:
  # 功能开关
  enabled: true                      # 启用/禁用LLM自主选择
  
  # 压力阈值配置
  pressure_threshold_high: 0.9       # 高压阈值 (推荐0.8-1.0)
  pressure_threshold_medium: 0.75    # 中压阈值 (推荐0.6-0.8)
  
  # 冷却时间配置
  cooldown_base_seconds: 30.0        # 基础冷却时间(秒)
  
  # Fallback配置
  fallback_mode: "FOCUS"             # Fallback默认模式
  
  # 性能配置
  decision_timeout_ms: 500           # LLM决策超时(毫秒)
  
  # 震荡检测配置
  oscillation_detection_window: 10   # 震荡检测窗口大小
  max_switch_history: 50             # 最大切换历史记录数
  min_confidence_threshold: 0.3      # 最低置信度阈值
```

---

## 四、验证方法

### 4.1 单元测试

```python
# tests/l2/test_attention_selection.py

def test_pressure_detection():
    """测试压力检测"""
    config = AttentionConfig()
    detector = PressureDetector(mock_awm, config)
    metrics = detector.calculate_pressure()
    assert metrics.current_pressure >= 0
    
def test_threshold_check():
    """测试阈值判断"""
    config = AttentionConfig(pressure_threshold_high=0.9)
    detector = PressureDetector(mock_awm, config)
    metrics = PressureMetrics(current_pressure=1.0, ...)
    result = detector.check_threshold(metrics)
    assert result.should_trigger == True
    
def test_mode_switch():
    """测试模式切换"""
    config = AttentionConfig()
    controller = ModeSwitchController(...)
    decision = DecisionResponse(mode="FOCUS", reason="test", confidence=0.8)
    result = controller.apply_llm_decision("GLOBAL", decision, 0.95)
    assert result.switched == True
```

### 4.2 集成测试

```python
def test_full_integration():
    """测试完整集成流程"""
    awm = AttentionWindowManager(context_window_size=128000)
    awm.set_llm_client(mock_llm_client)
    
    # 注册大量消息触发压力
    for i in range(100):
        awm.register_message({"role": "user", "content": "test" * 100})
    
    # 执行apply_window，应触发LLM选择
    messages = awm.apply_window()
    
    # 检查统计信息
    stats = awm.get_llm_selection_stats()
    assert stats["enabled"] == True
```

---

## 五、注意事项

### 5.1 向后兼容

- 工具驱动的模式切换不受影响，优先级高于LLM选择
- 如果配置`enabled: false`，系统行为与原来完全一致

### 5.2 性能考虑

- 压力检测在`apply_window()`中执行，目标延迟 < 5ms
- LLM决策异步执行，不阻塞主流程
- 冷却时间防止频繁触发

### 5.3 错误处理

- LLM决策失败时自动Fallback到默认模式
- 所有异常都有日志记录，不会中断推理流程
- 震荡检测防止模式快速切换

---

## 六、文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `zulong/l2/attention_types.py` | ~230 | 数据模型定义 |
| `zulong/l2/attention_config.py` | ~150 | 配置管理 |
| `zulong/l2/pressure_detector.py` | ~200 | 压力检测 |
| `zulong/l2/attention_mode_selector.py` | ~280 | LLM决策 |
| `zulong/l2/mode_switch_controller.py` | ~260 | 控制机制 |
| `zulong/l2/attention_integration.py` | ~300 | 集成扩展代码片段 |
| `config/zulong_config.yaml` | +23 | 配置扩展 |

---

**集成完成后，系统将具备以下能力**：

1. ✅ 实时监测上下文压力
2. ✅ 压力超限触发LLM决策
3. ✅ LLM自主选择注意力模式
4. ✅ 冷却时间防震荡
5. ✅ 向后兼容工具驱动模式
