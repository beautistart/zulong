# FC循环期间禁止节点审查提交 - 实现总结

> 实现时间: 2026-05-18
> 目标: 防止FC循环运行期间，节点审查任务竞争LLM资源

---

## 一、问题分析

### 1.1 问题描述

在FC循环运行期间，以下审查提交点可能提交任务到L2-BACKUP，与主对话竞争LLM资源：

| 提交点 | 文件位置 | 触发时机 |
|--------|---------|---------|
| **节点重要性晋升审查** | `memory_graph.py:1711-1791` | 系统空闲5分钟 + 候选积压 |
| **边衰减审查** | `memory_graph.py:1876-1957` | 修剪循环每30分钟 |
| **短期记忆淘汰审查** | `short_term_memory.py:550-583` | 容量超限时 |

### 1.2 竞争场景

```
用户发起任务 → FC循环启动 → LLM推理（占用L2-CORE）
                          │
                          ├─ 工具执行 → MemoryGraph.add_node()
                          │              └─ _pending_llm_candidates.append()
                          │
                          └─ 同时期：
                              _idle_review_loop() 检测到候选
                              └─ 提交审查任务到L2-BACKUP
                                  └─ 与主对话竞争LLM资源 ❌
```

---

## 二、解决方案

### 2.1 核心机制

在`StateManager`中新增`fc_loop_running`标志，所有审查提交点在提交前检查该标志。

### 2.2 数据流

```
FC循环启动
    │
    ▼
state_manager.set_fc_loop_running(True)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼                                     ▼
节点审查循环                        边审查提交
    │                                     │
    ├─ is_fc_loop_running()?              ├─ is_fc_loop_running()?
    │   └─ True → 跳过提交                │   └─ True → 跳过提交
    │                                     │
    ▼                                     ▼
等待FC结束                           等待FC结束
    │                                     │
    └──────────────┬──────────────────────┘
                   │
                   ▼
FC循环结束
    │
    ▼
state_manager.set_fc_loop_running(False)
    │
    ▼
审查任务可以提交
```

---

## 三、修改清单

### 3.1 StateManager 新增方法

**文件**: `zulong/core/state_manager.py`

```python
# 新增属性（行38）
self._fc_loop_running = False

# 新增方法（行200-222）
def is_fc_loop_running(self) -> bool:
    """判断FC循环是否正在运行"""
    with self._lock:
        return self._fc_loop_running

def set_fc_loop_running(self, running: bool):
    """设置FC循环运行状态"""
    with self._lock:
        old_state = self._fc_loop_running
        self._fc_loop_running = running
        if old_state != running:
            logger.info(f"FC loop running state changed: {old_state} -> {running}")
```

---

### 3.2 IDEFCRunner 设置标志

**文件**: `zulong/ide/ide_fc_runner.py`

#### 同步模式 `_run_loop()` (行1792-1872)

```python
def _run_loop(self, state: IDEFCState) -> IDEFCResult:
    # ✅ 进入时设置标志
    try:
        from zulong.core.state_manager import state_manager
        state_manager.set_fc_loop_running(True)
    except Exception:
        pass
    
    while True:
        tr = self._check(state)
        if tr:
            # ✅ 守卫终止时清除标志
            state_manager.set_fc_loop_running(False)
            return self._finalize(state, tr)
        
        try:
            # ... FC循环逻辑
            
            if verdict == "done":
                # ✅ 正常完成时清除标志
                state_manager.set_fc_loop_running(False)
                return IDEFCResult(phase="done", ...)
                
        except Exception as loop_err:
            if state.loop_error_count >= 3:
                # ✅ 异常终止时清除标志
                state_manager.set_fc_loop_running(False)
                return self._finalize(state, "loop_error")
```

#### 异步模式 `run_or_resume_async()` (行319-615)

```python
async def run_or_resume_async(...):
    # ✅ 进入时设置标志（行326-331）
    try:
        from zulong.core.state_manager import state_manager
        state_manager.set_fc_loop_running(True)
    except Exception:
        pass
    
    while True:
        try:
            if verdict == "done":
                # ✅ 正常完成时清除标志（行532-537）
                state_manager.set_fc_loop_running(False)
                return IDEFCResult(phase="done", ...)
                
        except asyncio.CancelledError:
            # ✅ 取消时清除标志（行593-598）
            state_manager.set_fc_loop_running(False)
            return await loop.run_in_executor(...)
            
        except Exception as loop_err:
            if state.loop_error_count >= 3:
                # ✅ 异常终止时清除标志（行607-612）
                state_manager.set_fc_loop_running(False)
                return await loop.run_in_executor(...)
```

---

### 3.3 MemoryGraph 审查提交点检查

#### 节点重要性晋升审查 (行1711-1717)

**文件**: `zulong/memory/memory_graph.py`

```python
async def _idle_review_loop(self):
    while self._running:
        # ... 空闲检测
        
        # ✅ 新增：检查FC循环状态
        try:
            if state_manager.is_fc_loop_running():
                logger.debug("[MemoryGraph] FC循环运行中，跳过节点审查提交")
                continue
        except Exception:
            pass
        
        # ... 原有审查提交逻辑
```

#### 边衰减审查 (行1937-1944)

```python
async def submit_prune_review(self):
    # ... 构建审查数据
    
    # ✅ 新增：检查FC循环状态
    try:
        from zulong.core.state_manager import state_manager
        if state_manager.is_fc_loop_running():
            logger.debug("[MemoryGraph] FC循环运行中，跳过边审查提交")
            return
    except Exception:
        pass
    
    # ... 原有审查提交逻辑
```

---

### 3.4 ShortTermMemory 审查提交点检查

**文件**: `zulong/memory/short_term_memory.py` (行552-582)

```python
async def _evict_low_value_memories(...):
    if num_to_delete > 0:
        # ✅ 新增：检查FC循环状态
        fc_running = False
        try:
            from zulong.core.state_manager import state_manager
            fc_running = state_manager.is_fc_loop_running()
            if fc_running:
                logger.debug("[ShortTermMemory] FC循环运行中，跳过淘汰前审查提交")
        except Exception:
            pass
        
        if not fc_running:
            # ... 原有审查提交逻辑
```

---

## 四、修改文件汇总

| 文件 | 修改类型 | 行号范围 |
|------|---------|---------|
| `zulong/core/state_manager.py` | 新增属性和方法 | 38, 200-222 |
| `zulong/ide/ide_fc_runner.py` | 设置/清除标志 | 326-331, 532-537, 593-598, 607-612, 1793-1872 |
| `zulong/memory/memory_graph.py` | 检查标志 | 1711-1717, 1937-1944 |
| `zulong/memory/short_term_memory.py` | 检查标志 | 552-582 |

---

## 五、验证测试

### 5.1 StateManager 验证

```python
from zulong.core.state_manager import state_manager

# 初始状态
assert state_manager.is_fc_loop_running() == False

# 设置运行
state_manager.set_fc_loop_running(True)
assert state_manager.is_fc_loop_running() == True

# 清除运行
state_manager.set_fc_loop_running(False)
assert state_manager.is_fc_loop_running() == False
```

### 5.2 预期行为

| 场景 | FC状态 | 审查提交 | 日志输出 |
|------|--------|---------|---------|
| 系统空闲，无FC运行 | False | ✅ 正常提交 | 无 |
| FC循环运行中 | True | ❌ 跳过提交 | `[MemoryGraph] FC循环运行中，跳过节点审查提交` |
| FC循环刚结束 | False | ✅ 正常提交 | 无 |

---

## 六、注意事项

### 6.1 暂停不清除标志

在同步模式的`_pause_for_remote()`中，FC循环暂停等待远程工具结果时，**不清除标志**：

```python
if remote:
    # 暂停等待远程工具，不清除标志
    # 恢复后继续运行，仍然是FC循环
    return self._pause_for_remote(state, remote)
```

**原因**: 暂停只是等待，恢复后仍然是FC循环的一部分。

### 6.2 异常处理

所有标志设置/检查都包裹在`try-except`中，确保：
- StateManager导入失败时不影响主流程
- 避免因标志管理失败导致FC循环异常

### 6.3 线程安全

`fc_loop_running`使用`self._lock`保护，确保多线程访问安全。

---

## 七、后续优化建议

| 优化项 | 说明 | 优先级 |
|--------|------|--------|
| **审查积压通知** | FC结束时检查积压数量，超过阈值时记录日志 | 低 |
| **动态调整间隔** | FC运行时延长审查检查间隔 | 低 |
| **审查队列** | 使用无锁队列替代`_pending_llm_candidates` | 中 |

---

**文档结束**
