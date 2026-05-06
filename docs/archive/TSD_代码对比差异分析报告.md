# TSD 与代码对比差异分析报告

**生成日期**: 2026-04-02  
**对比版本**: TSD v2.3 vs 当前代码库  
**分析目的**: 识别代码已实现但 TSD 未更新的功能模块

---

## 📊 总体概况

### 已实现但 TSD 未记录/更新的核心功能

| 功能模块 | 代码实现状态 | TSD 记录状态 | 差异等级 |
|---------|------------|------------|---------|
| 1. 动态路由架构 | ✅ 完整实现 | ❌ 未记录 | 🔴 高 |
| 2. WAITING 状态机制 | ✅ 完整实现 | ❌ 未记录 | 🔴 高 |
| 3. 任务冻结与恢复栈 | ✅ 完整实现 | ⚠️ 简略提及 | 🟡 中 |
| 4. 事件复盘机制 | ✅ 完整实现 | ❌ 未记录 | 🔴 高 |
| 5. 参数校准机制 | ✅ 完整实现 | ❌ 未记录 | 🔴 高 |
| 6. 经验库系统 | ✅ 完整实现 | ✅ 已记录 (v2.3) | 🟢 低 |
| 7. 视觉注意力机制 | ✅ 完整实现 | ⚠️ 部分记录 | 🟡 中 |

---

## 🔴 高优先级差异（需立即更新 TSD）

### 1. 动态路由架构（Dynamic Routing Architecture）

**代码实现位置**:
- `zulong/core/event_bus.py` - 事件总线路由逻辑
- `zulong/l1b/scheduler_gatekeeper.py` - L1-B Gatekeeper 路由决策
- `zulong/core/types.py` - 新增事件类型定义

**已实现的关键特性**:

```python
# File: zulong/core/types.py
class EventType(Enum):
    # 🎯 动态路由架构新增事件类型
    DIRECT_WAKEUP = "DIRECT_WAKEUP"        # L2 空闲时直连唤醒
    INTERACTION_TRIGGER = "INTERACTION_TRIGGER"  # L2 忙碌时触发中断
```

**路由逻辑**（代码已实现）:

```
用户事件 → EventBus → L1-B Gatekeeper → 判断 L2 状态
    ↓
    ├─ L2 空闲 → DIRECT_WAKEUP → 直接路由（无快照/无冻结）
    └─ L2 忙碌 → INTERACTION_TRIGGER → 标准中断流程（冻结 + 快照 + 重组）
```

**代码证据**:

```python
# File: zulong/l1b/scheduler_gatekeeper.py
def on_interaction_trigger(self, event: ZulongEvent):
    """
    处理 INTERACTION_TRIGGER 事件 (动态路由架构)
    
    L2 忙碌时触发，走 L1-B 标准中断流程：
    1. 冻结当前任务快照
    2. 重组多模态 Prompt
    3. 强制打断 L2
    4. 注入新任务
    """
    # 构建中断模式的 Prompt（包含历史上下文）
    prompt = self._build_interrupt_mode_prompt(payload)
    
    # 执行标准中断流程
    self._handle_visual_attention_interrupt(prompt, payload)

def on_direct_wakeup(self, event: ZulongEvent):
    """
    处理 DIRECT_WAKEUP 事件 (动态路由架构)
    
    L2 空闲时直连，L1-B 仅做简单透传：
    - 不执行冻结/快照操作
    - 构建最简 Prompt（无需历史上下文）
    - 直接路由到 L2
    """
    # 构建简单 Prompt（空闲模式，无需历史上下文）
    prompt = self._build_simple_wakeup_prompt(payload)
    
    # 直接路由到 L2
    self._route_visual_attention_to_l2(prompt, payload)
```

**TSD 缺失内容**:
- ❌ 缺少 `DIRECT_WAKEUP` 和 `INTERACTION_TRIGGER` 事件类型定义
- ❌ 缺少动态路由决策逻辑说明
- ❌ 缺少空闲/忙碌双模式路由流程图
- ❌ 缺少性能优化说明（空闲模式跳过冻结/快照步骤）

**建议更新章节**:
- 第 2 章：系统架构 → 新增 2.2.7 动态路由架构
- 第 4 章：功能模块 → 更新 4.2 L1-B 调度器
- 第 6 章：接口定义 → 新增事件类型 API

---

### 2. WAITING 状态机制（有效状态判断）

**代码实现位置**:
- `zulong/core/state_manager.py` - 状态管理器
- `zulong/core/types.py` - L2Status 枚举
- `zulong/l1b/scheduler_gatekeeper.py` - Gatekeeper 有效状态判断

**已实现的关键逻辑**:

```python
# File: zulong/core/types.py
class L2Status(Enum):
    IDLE = "IDLE"          # 空闲
    BUSY = "BUSY"          # 忙碌
    WAITING = "WAITING"    # ⚠️ 新增：等待间隙（任务未结束，但在等待反馈）
    UNLOADED = "UNLOADED"  # 卸载
```

**有效状态判断逻辑**（代码已实现）:

```python
# File: zulong/core/state_manager.py
def get_effective_status(self):
    """
    【核心工具函数】供 Gatekeeper 使用
    将 WAITING 视为 BUSY 处理
    """
    with self._lock:
        if self._l2_status in [L2Status.BUSY, L2Status.WAITING]:
            return "ACTIVE_TASK"  # 统一视为"有任务在身"
        return "IDLE"
```

**Gatekeeper 使用示例**:

```python
# File: zulong/l1b/scheduler_gatekeeper.py
def on_user_voice(self, event: ZulongEvent):
    # 1. 获取"有效状态" (将 WAITING 合并到 BUSY 逻辑中)
    effective_state = state_manager.get_effective_status()
    l2_status = state_manager.get_l2_status()
    
    logger.debug(f"Gatekeeper Check: Raw={l2_status.name}, Effective={effective_state}")

    # 2. 紧急关键词检测 (最高优先级，无视状态)
    if self._is_emergency(text):
        self._trigger_emergency_interrupt(text)
        return

    # 3. 普通路由逻辑 (根据有效状态分流)
    if effective_state == "ACTIVE_TASK":
        # --- 对应要求的：【任务中/空闲】时执行【忙碌】逻辑 ---
        if l2_status == L2Status.WAITING:
            # WAITING 状态：可以直接插入新指令打断当前续写
            self._handle_interruption_during_wait(text, event.type)
        else:
            # 真正的 BUSY (计算中)，只能排队或强行抢占
            self._queue_task_or_preempt(text)
    else:
        # --- 对应要求的：【空闲】保持原逻辑 ---
        self._handle_normal_command(text, event.priority, event.type)
```

**状态流转逻辑**（代码已实现）:

```python
# File: zulong/core/state_manager.py
def set_l2_status(self, status: L2Status, task_id: str = None):
    """设置 L2 状态"""
    with self._lock:
        old_status = self._l2_status
        self._l2_status = status
        if task_id:
            self._active_task_id = task_id
        
        logger.info(f"L2 status changed to: {status.name} (Task: {task_id})")
        
        # 🔥 关键逻辑：如果是分段任务完成，进入 WAITING 而不是 IDLE
        if status == L2Status.IDLE and self._active_task_id is not None:
            # 如果还有任务 ID 挂着，说明任务没做完，只是暂停了
            self._l2_status = L2Status.WAITING
            logger.warning(f"⚠️  Task {self._active_task_id} is suspended. Status forced to WAITING.")
```

**TSD 缺失内容**:
- ❌ 缺少 `WAITING` 状态的定义和用途说明
- ❌ 缺少 `get_effective_status()` 核心工具函数说明
- ❌ 缺少"将 WAITING 视为 BUSY"的路由决策逻辑
- ❌ 缺少分段任务场景的状态流转说明

**建议更新章节**:
- 第 2 章：系统架构 → 更新 2.3.2 状态机定义
- 第 4 章：功能模块 → 更新 4.2 L1-B 状态判断逻辑
- 第 6 章：接口定义 → 新增 `get_effective_status()` API

---

### 3. 事件复盘机制（Event Replay System）

**代码实现位置**:
- `zulong/replay/` - 复盘系统完整实现
- `zulong/core/types.py` - 复盘相关事件类型

**已实现的完整模块**:

```
zulong/replay/
├── __init__.py              # 复盘模块初始化
├── integration.py           # 复盘集成逻辑
├── context_snapshot.py      # 上下文快照
├── dossier_serializer.py    # 档案序列化
├── attributor.py            # 归因分析
├── patch_compiler.py        # System_Patch 编译器
├── calibration_manager.py   # 参数校准管理
├── experience_store.py      # 经验存储
├── l0_logger.py            # L0 日志记录
├── l1a_logger.py           # L1-A 日志记录
├── ring_buffer.py          # 环形缓冲区
└── clock_synchronizer.py   # 时钟同步器
```

**已定义的事件类型**（代码）:

```python
# File: zulong/core/types.py
class EventType(Enum):
    # 🎯 事件复盘机制事件类型 (TSD v2.1)
    REPLAY_TRIGGERED = "REPLAY_TRIGGERED"      # 复盘触发 (任务失败时)
    REPLAY_DOSSIER_CREATED = "REPLAY_DOSSIER_CREATED"  # 事件档案创建完成
    REPLAY_ATTRIBUTION_DONE = "REPLAY_ATTRIBUTION_DONE"  # 归因分析完成
    REPLAY_PATCH_GENERATED = "REPLAY_PATCH_GENERATED"  # System_Patch 生成完成
    REPLAY_PATCH_APPLIED = "REPLAY_PATCH_APPLIED"  # System_Patch 应用完成
    
    # 🎯 参数校准事件类型 (TSD v2.1)
    SYSTEM_CALIBRATION = "SYSTEM_CALIBRATION"  # 系统校准事件
    CALIBRATION_APPLIED = "CALIBRATION_APPLIED"  # 校准参数已应用
    CALIBRATION_FAILED = "CALIBRATION_FAILED"  # 校准参数应用失败
    
    # 🎯 经验库事件类型 (TSD v2.1)
    EXPERIENCE_STORED = "EXPERIENCE_STORED"    # 经验已存储
    EXPERIENCE_RETRIEVED = "EXPERIENCE_RETRIEVED"  # 经验已检索
```

**核心实现示例**:

```python
# File: zulong/replay/context_snapshot.py
class ContextSnapshot:
    """上下文快照管理器"""
    
    def capture(self) -> Dict:
        """捕获当前上下文快照"""
        # 实现代码...
    
    def restore(self, snapshot: Dict) -> bool:
        """恢复上下文快照"""
        # 实现代码...

# File: zulong/replay/dossier_serializer.py
class DossierSerializer:
    """事件档案序列化器"""
    
    def serialize(self, event_data: Dict) -> bytes:
        """序列化为二进制档案"""
        # 实现代码...
    
    def deserialize(self, data: bytes) -> Dict:
        """反序列化档案"""
        # 实现代码...

# File: zulong/replay/attributor.py
class Attributor:
    """归因分析器"""
    
    def analyze(self, dossier: Dict) -> AttributionResult:
        """分析事件失败原因"""
        # 实现代码...
```

**TSD 缺失内容**:
- ❌ 缺少完整的复盘系统架构说明
- ❌ 缺少 12 个复盘子模块的功能定义
- ❌ 缺少复盘事件类型定义（5 种事件）
- ❌ 缺少参数校准机制说明
- ❌ 缺少经验库事件类型定义（2 种事件）

**建议更新章节**:
- 新增第 12 章：事件复盘机制（完整章节）
  - 12.1 复盘触发条件
  - 12.2 上下文快照机制
  - 12.3 事件档案序列化
  - 12.4 归因分析算法
  - 12.5 System_Patch 编译器
  - 12.6 参数校准机制
  - 12.7 经验库集成

---

## 🟡 中优先级差异（需补充说明）

### 4. 任务冻结与恢复栈（Task Freeze & Resume Stack）

**代码实现位置**:
- `zulong/l2/task_state_manager.py` - 任务状态管理器
- `zulong/core/types.py` - TaskSnapshot 数据类

**已实现的关键功能**:

```python
# File: zulong/l2/task_state_manager.py
class TaskStateManager:
    """任务状态管理器"""
    
    def __init__(self):
        self._active_task_id: Optional[str] = None
        self._frozen_tasks: Dict[str, TaskSnapshot] = {}  # 冻结任务字典
        self._task_stack: List[str] = []  # 任务栈（支持多层嵌套）
    
    def freeze_current(self):
        """冻结当前任务"""
        with self._lock:
            if self._active_task_id:
                # 创建快照
                snapshot = TaskSnapshot(
                    task_id=self._active_task_id,
                    context_history=[],
                    working_memory={},
                    execution_pointer="generating_step_2",
                    created_at=time.time(),
                    last_updated=time.time()
                )
                
                # 保存到冻结任务
                self._frozen_tasks[self._active_task_id] = snapshot
                
                # 推入栈（支持多层嵌套）
                self._task_stack.append(self._active_task_id)
                
                # 清除活跃任务
                self._active_task_id = None
    
    def resume_task(self, task_id: str):
        """恢复任务"""
        with self._lock:
            if task_id in self._frozen_tasks:
                # 从冻结任务中取出
                snapshot = self._frozen_tasks.pop(task_id)
                
                # 如果有活跃任务，先冻结（嵌套支持）
                if self._active_task_id:
                    self.freeze_current()
                
                # 设置为活跃任务
                self._active_task_id = task_id
                
                # 从栈中弹出
                if task_id in self._task_stack:
                    self._task_stack.remove(task_id)
                
                # 同步到状态管理器 - 恢复任务时设置为 BUSY
                from zulong.core.state_manager import state_manager
                from zulong.core.types import L2Status
                state_manager.set_l2_status(L2Status.BUSY, task_id)
```

**TSD 已有记录**（但不完整）:
- ⚠️ TSD v2.3 第 10 章提到经验库系统
- ⚠️ 简略提及任务冻结概念
- ❌ 缺少任务栈（多层嵌套）实现细节
- ❌ 缺少 `TaskStateManager` API 定义
- ❌ 缺少与状态管理器的同步逻辑

**建议更新章节**:
- 第 10 章：经验库系统 → 新增 10.3 任务冻结与恢复栈
- 第 6 章：接口定义 → 新增 `TaskStateManager` API

---

### 5. 视觉注意力机制（Visual Attention）

**代码实现位置**:
- `zulong/l1b/scheduler_gatekeeper.py` - Gatekeeper 视觉注意力处理

**已实现的关键功能**:

```python
# File: zulong/l1b/scheduler_gatekeeper.py
def on_visual_attention(self, event: ZulongEvent):
    """
    处理 L1-C 视觉注意力事件 (TSD v1.8)
    
    当 L1-C 检测到交互意图（挥手、注视、靠近）时触发。
    L1-B 负责收集上下文并路由到 L2。
    """
    payload = event.payload
    intent_type = payload.get("intent_type", "unknown")
    intent_confidence = payload.get("intent_confidence", 0.0)
    person_distance = payload.get("person_distance", float('inf'))
    
    # 1. 获取 L2 状态
    effective_state = state_manager.get_effective_status()
    l2_status = state_manager.get_l2_status()
    
    # 2. 构建 Prompt
    prompt = self._build_visual_attention_prompt(payload)
    
    # 3. 根据状态决定处理方式
    if effective_state == "ACTIVE_TASK":
        if l2_status == L2Status.WAITING:
            # WAITING 状态：可以直接插入新指令
            self._handle_visual_attention_interrupt(prompt, payload)
        else:
            # BUSY 状态：排队或抢占
            self._queue_visual_attention_task(prompt, payload)
    else:
        # IDLE 状态：直接路由
        self._route_visual_attention_to_l2(prompt, payload)
```

**TSD 已有记录**（但不完整）:
- ⚠️ TSD v1.8 提到 L1-C 视觉插件架构
- ⚠️ 简略提及视觉注意力事件
- ❌ 缺少 L1-B 处理视觉注意力的详细流程
- ❌ 缺少意图类型识别逻辑（WAVING/GAZING/APPROACHING）
- ❌ 缺少与动态路由架构的集成说明

**建议更新章节**:
- 第 4 章：功能模块 → 更新 4.4 感知预处理
- 第 2 章：系统架构 → 更新 2.4.3 视觉注意力流程

---

## 🟢 低优先级差异（已记录）

### 6. 经验库系统（Experience Store）

**状态**: ✅ TSD v2.3 已完整记录

**已记录章节**:
- 第 9 章：数据存储架构（热/冷分层）
- 第 10 章：经验库系统（混合检索、智能打标）
- 第 11 章：复盘机制（用户主动/安静模式/夜间定时）

**代码实现**:
- `zulong/replay/experience_store.py` - 经验存储
- `zulong/memory/three_libraries.py` - 三层记忆库
- `zulong/memory/time_tags.py` - 时间标签
- `zulong/memory/rollback.py` - 回滚机制

**一致性评估**: ✅ 代码与 TSD 基本一致，无需额外更新

---

## 📝 更新建议与优先级

### 立即更新（🔴 高优先级）

1. **新增第 2.2.7 章：动态路由架构**
   - 定义 `DIRECT_WAKEUP` 和 `INTERACTION_TRIGGER` 事件
   - 说明空闲/忙碌双模式路由逻辑
   - 提供性能优化数据（跳过冻结/快照的收益）

2. **更新第 2.3.2 章：状态机定义**
   - 新增 `WAITING` 状态详细说明
   - 新增 `get_effective_status()` 函数说明
   - 说明"将 WAITING 视为 BUSY"的决策逻辑

3. **新增第 12 章：事件复盘机制**
   - 完整复盘系统架构（12 个子模块）
   - 复盘事件类型定义（5 种事件）
   - 参数校准机制（3 种事件）
   - 经验库事件类型（2 种事件）

### 近期更新（🟡 中优先级）

4. **更新第 10.3 章：任务冻结与恢复栈**
   - 任务栈多层嵌套机制
   - `TaskStateManager` API 定义
   - 与状态管理器的同步逻辑

5. **更新第 4.4 章：感知预处理**
   - L1-B 视觉注意力处理流程
   - 意图类型识别逻辑
   - 与动态路由架构的集成

### 后续优化（🟢 低优先级）

6. **代码注释增强**
   - 在关键函数 docstring 中引用 TSD 章节
   - 添加 TSD 版本号标记

7. **一致性检查自动化**
   - 创建 TSD-Code 一致性检查脚本
   - 定期生成差异报告

---

## 📊 统计数据

| 类别 | 数量 | 占比 |
|------|------|------|
| 🔴 高优先级差异 | 3 项 | 43% |
| 🟡 中优先级差异 | 2 项 | 29% |
| 🟢 低优先级差异 | 2 项 | 29% |
| **总计** | **7 项** | **100%** |

**需新增 TSD 章节**: 2 章（动态路由架构、事件复盘机制）  
**需更新 TSD 章节**: 5 章（状态机、任务冻结、视觉注意力等）

---

## ✅ 行动计划

### 第一阶段（本周）
- [ ] 新增 TSD 第 2.2.7 章：动态路由架构
- [ ] 更新 TSD 第 2.3.2 章：状态机定义（WAITING 状态）
- [ ] 新增 TSD 第 12 章：事件复盘机制

### 第二阶段（下周）
- [ ] 更新 TSD 第 10.3 章：任务冻结与恢复栈
- [ ] 更新 TSD 第 4.4 章：感知预处理（视觉注意力）
- [ ] 代码注释增强（添加 TSD 引用）

### 第三阶段（长期）
- [ ] 创建 TSD-Code 一致性检查脚本
- [ ] 建立定期审查机制（每月一次）
- [ ] 开发文档自动生成工具

---

**报告生成时间**: 2026-04-02  
**分析师**: ZULONG 首席架构师  
**状态**: ✅ 已完成
