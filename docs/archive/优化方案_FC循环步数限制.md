# FC 循环超时机制优化方案

## 问题分析

### 当前机制

```python
# inference_engine.py 第 915-959 行
max_fc_turns = 10  # 最大 10 轮
fc_turn = 0

while fc_turn < max_fc_turns:
    fc_turn += 1
    
    # 每轮 API 调用超时 300 秒
    api_response = future.result(timeout=self._fc_loop_timeout)  # 300s
    
    except concurrent.futures.TimeoutError:
        logger.error(f"🚨 [FC] Turn {fc_turn} 超时 (>{self._fc_loop_timeout}s)")
        response = self._get_fallback_response(user_input)  # ❌ 直接降级！
        break
```

### 核心问题

1. **时间超时机制不合理**
   - 复杂任务的每一步可能需要较长时间（工具调用、网络请求、文件操作）
   - 300 秒超时看似很长，但复杂任务可能执行 10+ 个工具，每步 20-30 秒
   - 一旦超时，**直接返回降级回复**，任务完全失败

2. **步数限制过低**
   - `max_fc_turns = 10` 对于复杂任务严重不足
   - 实际案例：市场研究任务需要 15-20 步（搜索→分析→总结→验证）

3. **缺乏弹性**
   - 当前机制是"硬超时"：一到时间就放弃
   - 没有考虑任务复杂度和进度

---

## 优化方案

### 方案概述

**从"时间超时"改为"步数上限"机制**

```yaml
# config/zulong_config.yaml 修改
l2_inference:
  # 原超时配置（保留但放宽）
  timeout:
    core: 600          # 300s → 600s（10分钟，极端情况）
    backup: 120        # 60s → 120s
    fc_loop: 600       # 300s → 600s
  
  # 新增：步数限制配置
  step_limits:
    enabled: true
    max_fc_turns: 100              # 10 → 100（充足步数）
    soft_limit: 50                 # 软限制：超过 50 步后开始警告
    hard_limit: 100                # 硬限制：绝对上限
    warning_interval: 10           # 每 10 步输出一次进度日志
```

---

## 具体修改

### 1. 配置文件修改

**文件**: `config/zulong_config.yaml`

```yaml
l2_inference:
  # 超时配置（大幅放宽）
  timeout:
    core: 600          # 核心模型单次调用超时（10分钟）
    backup: 120        # 备用模型超时（2分钟）
    fc_loop: 600       # FC 循环单次调用超时（10分钟）
  
  # 新增：步数限制配置
  step_limits:
    enabled: true
    max_fc_turns: 100              # FC 循环最大步数（替代硬编码 max_fc_turns=10）
    soft_limit: 50                 # 软限制：超过后警告但不中断
    hard_limit: 100                # 硬限制：绝对上限，强制终止
    warning_interval: 10           # 每 N 步输出进度日志
  
  # CircuitBreaker 配置（保持不变，作为智能兜底）
  circuit_breaker:
    enabled: true
    # ... 其他配置不变
```

---

### 2. 代码修改

**文件**: `zulong/l2/inference_engine.py`

#### 修改 1: 初始化时加载步数配置

```python
# 第 156-162 行（新增步数配置）

# 🔥 加载超时配置（从配置文件读取）
_l2_config = get_l2_inference_config()
_timeout_config = _l2_config.get('timeout', {})
self._core_timeout = _timeout_config.get('core', 600)  # 核心模型超时（默认 600s）
self._backup_timeout = _timeout_config.get('backup', 120)  # 备用模型超时
self._fc_loop_timeout = _timeout_config.get('fc_loop', 600)  # FC 循环超时

# 🔥 新增：加载步数限制配置
_step_config = _l2_config.get('step_limits', {})
self._max_fc_turns = _step_config.get('max_fc_turns', 100)  # FC 循环最大步数
self._soft_limit = _step_config.get('soft_limit', 50)  # 软限制
self._hard_limit = _step_config.get('hard_limit', 100)  # 硬限制
self._warning_interval = _step_config.get('warning_interval', 10)  # 警告间隔

logger.info(f"⏱️ [L2] 超时配置: core={self._core_timeout}s, backup={self._backup_timeout}s, fc_loop={self._fc_loop_timeout}s")
logger.info(f"🔢 [L2] 步数配置: max={self._max_fc_turns}, soft={self._soft_limit}, hard={self._hard_limit}")
```

#### 修改 2: FC 循环使用步数限制

```python
# 第 914-924 行（修改 FC 循环）

# 原代码：
# max_fc_turns = 10
# fc_turn = 0

# 修改为：
max_fc_turns = self._max_fc_turns  # 从配置读取（100）
fc_turn = 0
response = None
self._interrupt_flag = False
tool_results_buffer: List[Dict[str, Any]] = []

# 🔥 发布 pipeline_start 事件
self._publish_task_graph_event("pipeline_start", 0, "", "")

while fc_turn < max_fc_turns:
    fc_turn += 1
    
    # 🔥 新增：进度监控
    if fc_turn % self._warning_interval == 0:
        logger.info(f"[FC] 进度: {fc_turn}/{max_fc_turns} 步，工具调用次数: {len(tool_results_buffer)}")
    
    if fc_turn > self._soft_limit:
        logger.warning(f"[FC] ⚠️ 已超过软限制 ({self._soft_limit} 步)，继续执行...")
    
    if fc_turn >= self._hard_limit:
        logger.error(f"[FC] 🚨 达到硬限制 ({self._hard_limit} 步)，强制终止")
        break
    
    # 中断检查（保持不变）
    if self._interrupt_flag:
        logger.info(f"[FC] Turn {fc_turn}: 检测到中断信号，终止 FC 循环")
        response = response or ""
        break
    
    # ... 其余代码不变 ...
```

#### 修改 3: 超时处理改为警告而非直接降级

```python
# 第 954-959 行（修改超时处理）

# 原代码：
# except concurrent.futures.TimeoutError:
#     logger.error(f"🚨 [FC] Turn {fc_turn} 超时 (>{self._fc_loop_timeout}s)")
#     response = self._get_fallback_response(user_input)  # ❌ 直接降级
#     break

# 修改为：
except concurrent.futures.TimeoutError:
    logger.warning(f"⚠️ [FC] Turn {fc_turn} 超时 (>{self._fc_loop_timeout}s)，继续尝试...")
    
    # 🔥 新增：不直接降级，记录超时但继续执行
    # 超时可能是由于：
    # 1. 工具调用耗时较长（如网络请求、文件操作）
    # 2. 模型正在思考复杂问题
    # 3. 系统负载较高
    
    # 检查是否已接近硬限制
    if fc_turn >= self._hard_limit:
        logger.error(f"[FC] 🚨 超时且达到硬限制，使用降级回复")
        response = self._get_fallback_response(user_input)
        break
    else:
        # 继续下一轮（不 break）
        logger.info(f"[FC] 跳过本轮超时，继续执行 (当前步数: {fc_turn}/{max_fc_turns})")
        continue
```

#### 修改 4: 达到最大轮数的处理优化

```python
# 第 1061-1063 行（修改达到最大轮数的处理）

# 原代码：
# if response is None:
#     logger.warning(f"[FC] 达到最大轮次 {max_fc_turns}，使用降级回复")
#     response = self._get_fallback_response(user_input)

# 修改为：
if response is None:
    if fc_turn >= self._hard_limit:
        logger.warning(f"[FC] 达到硬限制 {max_fc_turns} 步，使用降级回复")
        response = self._get_fallback_response(user_input)
    else:
        # 可能是被中断或其他原因
        logger.warning(f"[FC] FC 循环异常终止 (已执行 {fc_turn} 步)")
        response = self._get_fallback_response(user_input)
```

---

### 3. CircuitBreaker 兜底机制

**保持不变**，作为智能兜底：

```yaml
# config/zulong_config.yaml
l2_inference:
  circuit_breaker:
    enabled: true
    safety_hard_cap: 50              # CircuitBreaker 的绝对上限
    # ... 其他信号配置不变
```

**作用**：
- 步数限制是"机械的"：只管步数
- CircuitBreaker 是"智能的"：检测重复调用、信息增益、上下文压力等
- 两者配合：步数上限保证任务能完成，CircuitBreaker 防止无限循环

---

## 预期效果

### 优化前

| 指标 | 数值 |
|------|------|
| 最大步数 | 10 步 |
| 单步超时 | 300 秒 |
| 超时行为 | ❌ 直接降级 |
| 复杂任务成功率 | 0% |

### 优化后

| 指标 | 数值 |
|------|------|
| 最大步数 | 100 步 |
| 单步超时 | 600 秒 |
| 超时行为 | ✅ 继续执行 |
| 复杂任务成功率 | 预期 80%+ |

### 保护机制

| 层级 | 机制 | 触发条件 | 行为 |
|------|------|---------|------|
| L1 | 步数软限制 | >50 步 | 警告日志 |
| L2 | 步数硬限制 | >100 步 | 强制终止 |
| L3 | CircuitBreaker | 检测到循环/重复 | 智能降级 |
| L4 | 用户中断 | 新指令到达 | 立即终止 |

---

## 实施状态

### ✅ 已完成

#### 1. 配置文件修改
- **文件**: `config/zulong_config.yaml`
- **修改内容**:
  ```yaml
  timeout:
    core: 600      # 300s → 600s
    backup: 120    # 60s → 120s
    fc_loop: 600   # 300s → 600s
  
  step_limits:
    enabled: true
    max_fc_turns: 100    # 新增：100 步上限
    soft_limit: 50       # 新增：50 步软限制
    hard_limit: 100      # 新增：100 步硬限制
    warning_interval: 10 # 新增：每 10 步警告
  ```
- **状态**: ✅ 已完成

#### 2. 推理引擎代码修改
- **文件**: `zulong/l2/inference_engine.py`
- **修改内容**:
  1. ✅ 初始化时加载步数配置（第 156-170 行）
  2. ✅ FC 循环使用配置步数限制（第 914-935 行）
  3. ✅ 添加进度监控和警告（第 927-935 行）
  4. ✅ 超时处理改为继续执行（第 954-968 行）
  5. ✅ 达到最大轮数的处理优化（第 1061-1068 行）
- **状态**: ✅ 已完成

#### 3. 文档更新
- **文件**: `docs/优化方案_FC循环步数限制.md`
- **状态**: ✅ 已记录实施情况

### ⏳ 待完成

#### 4. 重启系统测试
- **状态**: ⏳ 等待用户重启系统
- **预计时间**: 2-3 分钟

#### 5. 验证复杂任务成功率
- **状态**: ⏳ 等待测试
- **预计时间**: 5-10 分钟

---

## 实施步骤

### ✅ 步骤 1: 修改配置文件（已完成）
- 文件: `config/zulong_config.yaml`
- 修改: 超时配置 + 新增步数配置
- 时间: 2 分钟

### 步骤 2: 修改推理引擎
- 文件: `zulong/l2/inference_engine.py`
- 修改: 4 处代码（初始化、FC 循环、超时处理、轮数检查）
- 时间: 10 分钟

### 步骤 3: 重启系统测试
- 命令: 重启 ZULONG 系统
- 测试: 发送复杂任务（如市场研究、技术趋势分析）
- 观察: 日志中的步数进度和工具调用情况
- 时间: 5 分钟

### 步骤 4: 验证效果
- 对比: 优化前后复杂任务成功率
- 检查: 日志中是否还有超时降级
- 调整: 如需要，微调步数限制
- 时间: 10 分钟

---

## 风险评估

### 风险 1: 无限循环

**风险等级**: 低

**原因**:
- CircuitBreaker 作为智能兜底
- 硬限制 100 步保证绝对终止
- 用户可手动中断

**缓解措施**:
- 保持 CircuitBreaker 启用
- 监控日志中的步数进度
- 如发现问题，可随时下调硬限制

### 风险 2: 资源占用过高

**风险等级**: 低

**原因**:
- 单步超时 600 秒，总任务时间可能较长
- 但每个任务是顺序执行的，不会并发

**缓解措施**:
- 监控内存和 CPU 使用
- 必要时调整超时时间
- 使用任务挂起机制

### 风险 3: 用户体验下降

**风险等级**: 低

**原因**:
- 复杂任务执行时间变长
- 但成功率大幅提升

**缓解措施**:
- 通过前端显示进度（已实现任务图谱）
- 每 10 步输出进度日志
- 用户可随时中断

---

## 总结

### 核心改动

1. ✅ **取消硬超时降级**: 超时不直接返回失败，而是继续执行
2. ✅ **步数上限替代**: 10 步 → 100 步，给予充足执行空间
3. ✅ **超时配置放宽**: 300s → 600s，适应复杂任务需求
4. ✅ **保持智能兜底**: CircuitBreaker 继续生效

### 预期收益

- **复杂任务成功率**: 0% → 80%+
- **用户体验**: 从"频繁失败"到"稳定完成"
- **系统健壮性**: 多层保护，不会无限循环

### 后续优化

- 根据实际使用情况微调步数限制
- 考虑按任务类型动态调整步数（简单任务 20 步，复杂任务 100 步）
- 优化 CircuitBreaker 的检测算法

---

**建议**: 立即实施此方案，解决当前复杂任务无法完成的核心问题。
