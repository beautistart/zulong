# ReviewStateManager 使用指南

**创建日期**: 2026-04-07  
**版本**: TSD v2.3 (祖龙 β4)  
**作者**: AI Assistant

---

## 📋 概述

`ReviewStateManager` 是一个**单例模式**的状态管理器，用于统一管理复盘相关的所有状态。它解决了之前分散状态管理导致的同步问题和竞态条件。

---

## 🎯 核心功能

### 1. 统一管理复盘状态
- 复盘模式（快速/深度）
- 复盘阶段（选择/分析/生成/确认/完成）
- 会话 ID 和时间信息
- 经验数据和确认状态

### 2. 线程安全的防重入锁
- 防止重复触发复盘流程
- 支持并发场景下的状态一致性
- 自动同步到全局 `state_manager`

### 3. 状态变更回调
- 支持注册状态变更监听器
- 实时通知状态变化
- 便于 UI 更新和日志记录

### 4. 强制退出机制
- 异常情况下强制重置所有状态
- 确保 L2 状态正确恢复
- 防止系统卡死

---

## 🚀 快速开始

### 导入模块

```python
from zulong.review.state_manager import (
    get_review_state_manager,
    ReviewMode,
    ReviewStage
)
```

### 获取单例实例

```python
# 获取全局单例
state_manager = get_review_state_manager()
```

---

## 📖 API 参考

### 核心状态管理

#### `enter_review_mode(mode: ReviewMode, session_id: str)`
进入复盘模式

**参数**:
- `mode`: 复盘模式（`ReviewMode.QUICK` 或 `ReviewMode.DEEP`）
- `session_id`: 会话 ID

**示例**:
```python
state_manager.enter_review_mode(ReviewMode.QUICK, "session_123")
```

#### `exit_review_mode(reason: str = 'completed')`
退出复盘模式

**参数**:
- `reason`: 退出原因（`'completed'` | `'cancelled'` | `'failed'`）

**返回**:
- `float`: 复盘耗时（秒）

**示例**:
```python
duration = state_manager.exit_review_mode('completed')
print(f"复盘耗时：{duration:.2f}秒")
```

#### `force_exit()`
强制退出复盘模式（异常情况下使用）

**示例**:
```python
try:
    # 执行复盘逻辑
    pass
except Exception as e:
    state_manager.force_exit()
```

---

### 阶段管理

#### `update_stage(stage: ReviewStage, message: str = "")`
更新当前阶段

**参数**:
- `stage`: 阶段枚举（`ReviewStage`）
- `message`: 状态消息

**示例**:
```python
state_manager.update_stage(ReviewStage.ANALYZING, '正在分析对话...')
```

#### 便捷方法

```python
state_manager.set_analyzing()              # 设置为分析中
state_manager.set_generating()             # 设置为生成经验中
state_manager.set_waiting_confirmation(5)  # 设置为等待确认（5 条经验）
state_manager.set_confirming()             # 设置为确认中
state_manager.set_completed()              # 设置为已完成
```

---

### 防重入锁管理

#### `acquire_processing_lock() -> bool`
获取处理锁

**返回**:
- `True`: 成功获取锁
- `False`: 锁已被占用

**示例**:
```python
if not state_manager.acquire_processing_lock():
    logger.warning("复盘正在处理中，忽略重复请求")
    return

try:
    # 执行复盘逻辑
    pass
finally:
    state_manager.release_processing_lock()
```

#### `release_processing_lock()`
释放处理锁

---

### 数据管理

#### `set_pending_experiences(experiences: list, summary: str = "", tags: list = None)`
设置待确认的经验数据

**示例**:
```python
state_manager.set_pending_experiences(
    experiences=[{'title': '经验 1', 'content': '...'}],
    summary='本次复盘总结',
    tags=['沟通', '协作']
)
```

#### `get_pending_experiences() -> Optional[list]`
获取待确认的经验数据

#### `confirm_experiences()`
确认经验

---

### 状态查询

#### `is_active() -> bool`
是否处于复盘模式

#### `get_mode() -> Optional[ReviewMode]`
获取复盘模式

#### `get_stage() -> Optional[ReviewStage]`
获取当前阶段

#### `get_session_id() -> Optional[str]`
获取会话 ID

#### `is_processing() -> bool`
是否正在处理中

#### `get_status_message() -> str`
获取状态消息

#### `get_session_info() -> Dict[str, Any]`
获取会话信息字典

**返回示例**:
```python
{
    'is_active': True,
    'mode': 'quick',
    'stage': 'analyzing',
    'session_id': 'session_123',
    'is_processing': False,
    'experience_count': 5,
    'confirmed': False,
    'status_message': '正在检索记忆库和分析对话...',
    'duration': 12.34,
    'start_time': '2026-04-07T10:30:00'
}
```

#### `get_status_indicator() -> str`
获取状态指示器（用于 UI 显示）

**返回示例**:
- `'🤔 选择模式'`
- `'🔍 分析中...'`
- `'💡 提炼经验...'`
- `'✅ 等待确认'`
- `'✨ 已完成'`

---

### 便捷方法

#### `can_accept_input() -> bool`
是否可以接受用户输入

**逻辑**:
- 非复盘模式：`True`
- 处理中：`False`
- 选择阶段或等待确认阶段：`True`
- 其他阶段：`False`

#### `should_forward_to_replay() -> bool`
是否应该转发到 ReplayIntegration

**逻辑**:
- 复盘模式下且未处理中：`True`
- 其他情况：`False`

---

## 🔧 实际使用示例

### 示例 1：Gatekeeper 中的使用

```python
from zulong.review.state_manager import get_review_state_manager, ReviewMode

class Gatekeeper:
    def __init__(self):
        self._review_state_manager = get_review_state_manager()
    
    def on_user_voice(self, event):
        text = event.payload.get("text", "")
        
        # 检测复盘关键词
        if '快速复盘' in text:
            # 🔥 使用 ReviewStateManager 的检查锁机制
            if not self._review_state_manager.acquire_processing_lock():
                logger.debug("复盘正在处理中，忽略重复指令")
                return
            
            try:
                # 进入复盘模式
                session_id = str(uuid.uuid4())[:8]
                self._review_state_manager.enter_review_mode(
                    ReviewMode.QUICK, 
                    session_id
                )
                
                # 转发到 ReplayIntegration
                self._forward_to_replay_integration(text, 'quick_review')
                
            finally:
                # 释放处理锁
                self._review_state_manager.release_processing_lock()
```

### 示例 2：ReplayIntegration 中的使用

```python
from zulong.review.state_manager import get_review_state_manager, ReviewMode

class ReplayIntegration:
    def __init__(self):
        self._state_manager = get_review_state_manager()
    
    async def _handle_user_active_review(self, context):
        try:
            # 🔥 检查是否已有处理锁
            if not self._state_manager.acquire_processing_lock():
                logger.warning("无法获取处理锁，忽略重复请求")
                return
            
            # 进入复盘模式
            session_id = str(uuid.uuid4())[:8]
            mode = ReviewMode.QUICK if review_type == 'quick' else ReviewMode.DEEP
            self._state_manager.enter_review_mode(mode, session_id)
            
            # 执行复盘逻辑
            await self._handle_quick_review_async(recent_data, context)
            
        except Exception as e:
            logger.error(f"用户主动复盘失败：{e}", exc_info=True)
            # 🔥 异常时强制退出
            self._state_manager.force_exit()
        finally:
            # 🔥 释放处理锁
            try:
                self._state_manager.release_processing_lock()
            except Exception as e:
                logger.error(f"释放处理锁失败：{e}")
    
    async def _handle_quick_review_async(self, recent_data, context):
        try:
            # 更新阶段
            self._state_manager.set_analyzing()
            
            # 分析对话
            analysis_result = await self._analyze_conversation_with_l2_async(recent_data)
            
            # 更新阶段
            self._state_manager.set_generating()
            
            # 生成经验
            experiences = await self._generate_experiences_async(analysis_result)
            
            # 更新阶段
            self._state_manager.set_waiting_confirmation(len(experiences))
            
            # 完成复盘
            self._state_manager.exit_review_mode('completed')
            
        except Exception as e:
            # 异常时强制退出
            self._state_manager.force_exit()
            raise
```

### 示例 3：状态监控和 UI 更新

```python
from zulong.review.state_manager import get_review_state_manager

class ReviewMonitor:
    def __init__(self):
        self.state_manager = get_review_state_manager()
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """设置状态监控"""
        # 注册状态变更回调
        self.state_manager.register_state_change_callback(self._on_state_change)
    
    def _on_state_change(self, action: str):
        """状态变更回调"""
        logger.info(f"状态变更：{action}")
        
        # 更新 UI
        if action == 'update_stage':
            stage = self.state_manager.get_stage()
            message = self.state_manager.get_status_message()
            self.update_ui_status(stage, message)
        
        elif action == 'exit_review_mode':
            info = self.state_manager.get_session_info()
            logger.info(f"复盘完成，耗时：{info['duration']:.2f}秒")
    
    def update_ui_status(self, stage, message):
        """更新 UI 状态显示"""
        # 实现 UI 更新逻辑
        indicator = self.state_manager.get_status_indicator()
        print(f"当前状态：{indicator} - {message}")
```

---

## 🎯 最佳实践

### 1. 始终使用 `try...finally` 确保锁释放

```python
# ✅ 正确做法
if state_manager.acquire_processing_lock():
    try:
        # 执行复盘逻辑
        pass
    finally:
        state_manager.release_processing_lock()

# ❌ 错误做法
state_manager.acquire_processing_lock()
# 执行复盘逻辑（如果异常，锁永远不会释放）
```

### 2. 使用枚举类型而非字符串

```python
# ✅ 正确做法
state_manager.enter_review_mode(ReviewMode.QUICK, session_id)
state_manager.update_stage(ReviewStage.ANALYZING, "分析中")

# ❌ 错误做法
state_manager.enter_review_mode("quick", session_id)
state_manager.update_stage("analyzing", "分析中")
```

### 3. 强制退出仅在异常情况下使用

```python
try:
    # 正常复盘流程
    pass
except Exception as e:
    # ✅ 正确：异常时强制退出
    state_manager.force_exit()

# ❌ 错误：正常流程中使用强制退出
state_manager.force_exit()  # 会跳过清理逻辑
```

### 4. 使用便捷方法提高可读性

```python
# ✅ 推荐：使用便捷方法
state_manager.set_analyzing()
state_manager.set_generating()
state_manager.set_waiting_confirmation(count)

# 不推荐：直接调用 update_stage
state_manager.update_stage(ReviewStage.ANALYZING, "正在检索记忆库和分析对话...")
```

### 5. 检查状态后再执行操作

```python
# ✅ 推荐：先检查状态
if state_manager.can_accept_input():
    # 处理用户输入
    pass

if state_manager.should_forward_to_replay():
    # 转发到 ReplayIntegration
    pass

# ❌ 不推荐：直接执行
# 处理用户输入（可能在错误的阶段）
```

---

## 🔍 调试技巧

### 获取完整会话信息

```python
info = state_manager.get_session_info()
print(f"复盘模式：{info['mode']}")
print(f"当前阶段：{info['stage']}")
print(f"会话 ID: {info['session_id']}")
print(f"经验数量：{info['experience_count']}")
print(f"耗时：{info['duration']:.2f}秒")
```

### 监控状态变更

```python
def log_state_change(action):
    info = state_manager.get_session_info()
    logger.info(f"[{action}] 状态变更：{info}")

state_manager.register_state_change_callback(log_state_change)
```

### 检查锁状态

```python
if state_manager.is_processing():
    logger.warning("处理锁已被占用")
else:
    logger.info("处理锁可用")
```

---

## ⚠️ 注意事项

### 1. 单例模式

`ReviewStateManager` 是单例模式，所有模块共享同一个实例。

```python
# 所有获取的实例都是同一个
manager1 = get_review_state_manager()
manager2 = get_review_state_manager()
assert manager1 is manager2  # True
```

### 2. 线程安全

所有公共方法都是线程安全的，内部使用锁保护状态。

```python
# 可以在多线程环境下安全使用
thread1: state_manager.acquire_processing_lock()
thread2: state_manager.acquire_processing_lock()  # 会返回 False
```

### 3. 状态同步

`ReviewStateManager` 会自动同步状态到全局 `state_manager`。

```python
# 进入复盘模式时自动同步
state_manager.enter_review_mode(ReviewMode.QUICK, session_id)
# 自动设置：
# - state_manager.set_context('review_mode', True)
# - state_manager.set_context('review_session_id', session_id)
# - state_manager.set_context('review_type', 'quick')
```

### 4. 异常处理

异常情况下务必调用 `force_exit()`。

```python
try:
    # 复盘逻辑
    pass
except Exception:
    state_manager.force_exit()  # 确保状态重置
    raise
```

---

## 📊 状态流转图

```
[非复盘模式]
    ↓ 用户触发复盘
[SELECTING] ←→ [WAITING_CONFIRMATION]
    ↓                    ↓
[ANALYZING]              ↓
    ↓                    ↓
[GENERATING]             ↓
    ↓                    ↓
[CONFIRMING]             ↓
    ↓                    ↓
[COMPLETED/CANCELLED/FAILED]
    ↓
[非复盘模式]
```

---

## 📚 相关文档

- [复盘机制状态同步问题修复报告](复盘机制状态同步问题修复报告.md)
- [TSD v2.3 第 11.1 节 - 复盘机制](TSD_v2.3.md#111-复盘机制)
- [状态管理器设计文档](state_manager_design.md)

---

**版本**: 1.0  
**最后更新**: 2026-04-07  
**维护者**: AI Assistant
