# 祖龙 (ZULONG) 系统自主学习逻辑完整梳理

**分析日期**: 2026-03-30  
**分析版本**: v2.4（事件驱动重构版）  
**分析人员**: 系统架构组  

---

## 📋 核心问题解答

### 问题 1: 任务失败时，是否立即触发复盘机制？

**答案**: ❌ **不是立即触发，而是由复盘触发器调度触发**

**完整流程**:

```
任务失败
  ↓
L2 捕获异常
  ↓
【未自动触发复盘】❌
  ↓
等待复盘触发器调度
  ↓
复盘触发器 (trigger.py) 三种触发方式:
  1. 用户主动触发 (高优先级)
  2. 安静模式触发 (中优先级)
  3. 夜间定时触发 (低优先级)
  ↓
触发失败案例分析器 (failure_analyzer.py)
  ↓
生成失败案例对象 (FailureCase)
  ↓
保存到经验库 (add_experience)
  ↓
【此时才触发事件驱动】✅
  ↓
热更新引擎检测新经验
  ↓
生成补丁
  ↓
应用补丁
```

**关键代码**:

**文件**: [`trigger.py`](file:///d:/AI/project/zulong_beta4/zulong/review/trigger.py#L169-L194)

```python
async def trigger_user_active(self, context: Optional[Dict] = None) -> bool:
    """用户主动触发复盘"""
    logger.info(f"[ReviewTrigger] 用户主动触发复盘")
    
    return await self._queue_trigger(
        trigger_type=TriggerType.USER_ACTIVE,
        priority=TriggerPriority.HIGH,
        context=context
    )
```

**文件**: [`failure_analyzer.py`](file:///d:/AI/project/zulong_beta4/zulong/review/failure_analyzer.py#L307-L351)

```python
def save_to_experience_store(self, case: FailureCase, weight_multiplier: float = 1.5):
    """保存案例到经验库 (失败案例权重 1.5 倍)"""
    
    # 构建经验内容
    content = f"""
任务：{case.task_description}
错误类型：{case.error_type}
根本原因：{case.root_cause}
避坑指南：{case.avoidance_guide}
"""
    
    # 添加到经验库
    self.experience_store.add_experience(
        content=content,
        experience_type="failure_case",
        tags=["failure", "case_study", case.error_type],
        metadata={...}
    )
```

**结论**: 
- ❌ **任务失败不会立即触发复盘**
- ✅ **复盘由触发器调度（用户主动/安静模式/夜间定时）**
- ✅ **只有经验库写入时才会触发事件驱动的热更新**

---

### 问题 2: 补丁是什么？跟经验向量是否同一个东西？

**答案**: ❌ **不是同一个东西，有本质区别**

#### 经验向量 (Experience Vector)

**定义**: 经验内容的向量表示，用于语义检索

**生成时机**: 经验写入时

**生成者**: [`enhanced_experience_store.py`](file:///d:/AI/project/zulong_beta4/zulong/memory/enhanced_experience_store.py#L420-L426) 的 `_get_embedding()` 方法

**保存位置**: 经验对象的 `embedding` 字段

**用途**: 用于混合检索（向量相似度搜索）

**数据结构**:
```python
@dataclass
class Experience:
    id: str
    content: str
    experience_type: str
    embedding: Optional[np.ndarray] = None  # 512 维向量
    keywords: List[str] = None  # BM25 关键词
    tags: List[str] = None  # 多标签
    metadata: Dict[str, Any] = None
```

---

#### 补丁 (SystemPatch)

**定义**: 参数/规则/策略的调整指令

**生成时机**: 热更新引擎检测到新经验时

**生成者**: [`hot_update_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/memory/hot_update_engine.py#L200-L290) 的 `_generate_patch_from_experience()` 方法

**保存位置**: [`HotUpdateEngine._patches`](file:///d:/AI/project/zulong_beta4/zulong/memory/hot_update_engine.py#L108-L109) 字典

**用途**: 指导 L0/L1-A/L1-B 层参数调整

**数据结构**:
```python
@dataclass
class SystemPatch:
    patch_id: str
    patch_type: PatchType  # PARAMETER/RULE/STRATEGY/THRESHOLD
    target_layer: str  # "l0"/"l1a"/"l1b"/"l2"
    condition: str     # 触发条件
    adjustment: Dict[str, Any]  # 调整内容
    priority: int
    status: PatchStatus
```

---

#### 本质区别对比

| 维度 | 经验向量 | 补丁 |
|------|---------|------|
| **本质** | 知识的向量表示 | 参数调整指令 |
| **形态** | 512 维浮点数组 | 结构化数据对象 |
| **生成者** | Embedding 模型 | 热更新引擎 |
| **生成时机** | 经验写入时 | 经验触发事件后 |
| **保存位置** | 经验库 (Qdrant/SQLite) | 热更新引擎内存 |
| **用途** | 语义检索 | 参数调整 |
| **生命周期** | 长期存储 | 临时应用（可回滚） |
| **作用对象** | 无（只读） | L0/L1-A/L1-B 层参数 |

---

### 问题 3: 补丁怎么判断生成的？由谁生成的？

**答案**: 由热更新引擎根据经验类型判断并生成

#### 判断逻辑

**文件**: [`hot_update_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/memory/hot_update_engine.py#L200-L230)

```python
async def _generate_patch_from_experience(self, experience: Any) -> Optional[SystemPatch]:
    """从经验生成补丁"""
    
    # 1. 分析经验类型
    exp_type = getattr(experience, 'experience_type', 'unknown')
    content = getattr(experience, 'content', '')
    metadata = getattr(experience, 'metadata', {})
    
    # 2. 根据经验类型生成补丁
    if exp_type == 'failure':
        # 失败案例 → 参数调整补丁
        patch = await self._generate_failure_patch(experience)
    elif exp_type == 'success':
        # 成功案例 → 策略优化补丁
        patch = await self._generate_success_patch(experience)
    elif exp_type == 'preference':
        # 用户偏好 → 规则更新补丁
        patch = await self._generate_preference_patch(experience)
    else:
        return None  # 其他类型不生成补丁
    
    return patch
```

#### 生成规则

**失败案例** (`experience_type='failure'`):
```python
async def _generate_failure_patch(self, experience: Any):
    metadata = getattr(experience, 'metadata', {})
    
    # 关键：从元数据中提取参数调整建议
    if 'parameter_adjustment' in metadata:
        adjustment = metadata['parameter_adjustment']
        
        return SystemPatch(
            patch_id=f"patch_{timestamp}_{hash}",
            patch_type=PatchType.PARAMETER,
            target_layer="l0",  # 执行层
            condition=experience.content,
            adjustment=adjustment,  # 如：{"GRIP_FORCE": 0.7}
            priority=8  # 高优先级
        )
```

**成功案例** (`experience_type='success'`):
```python
async def _generate_success_patch(self, experience: Any):
    metadata = getattr(experience, 'metadata', {})
    
    # 从元数据中提取策略优化
    if 'strategy_optimization' in metadata:
        optimization = metadata['strategy_optimization']
        
        return SystemPatch(
            patch_type=PatchType.STRATEGY,
            target_layer="l1b",  # 调度层
            adjustment=optimization,  # 如：{"GRIP_STRATEGY": "gentle"}
            priority=5  # 中优先级
        )
```

**生成者**: [`HotUpdateEngine`](file:///d:/AI/project/zulong_beta4/zulong/memory/hot_update_engine.py#L93-L147) 单例

---

### 问题 4: 补丁注入到哪里？保存到哪里？由谁来注入？

#### 注入到哪里？

**注入目标**: L0/L1-A/L1-B 层的参数/规则/策略

**文件**: [`patch_applier.py`](file:///d:/AI/project/zulong_beta4/zulong/memory/patch_applier.py#L127-L181)

```python
async def apply_to_l0(self, patch: SystemPatch) -> bool:
    """应用补丁到 L0 执行器"""
    
    # 验证调整内容
    for param_name, new_value in patch.adjustment.items():
        if not self._validate_parameter(param_name, new_value):
            return False
    
    # 应用调整
    for param_name, new_value in patch.adjustment.items():
        if param_name in self._l0_parameters:
            old_value = self._l0_parameters[param_name].value
            self._l0_parameters[param_name].value = new_value  # ✅ 注入到参数
            
            logger.info(f"[PatchApplier] L0 参数更新：{param_name} {old_value} → {new_value}")
```

**注入位置**:
- **L0 层**: [`PatchApplier._l0_parameters`](file:///d:/AI/project/zulong_beta4/zulong/memory/patch_applier.py#L50) 字典
- **L1-A 层**: [`PatchApplier._l1a_rules`](file:///d:/AI/project/zulong_beta4/zulong/memory/patch_applier.py#L53) 字典
- **L1-B 层**: [`PatchApplier._l1b_strategies`](file:///d:/AI/project/zulong_beta4/zulong/memory/patch_applier.py#L56) 字典

---

#### 保存到哪里？

**补丁保存位置**: [`HotUpdateEngine._patches`](file:///d:/AI/project/zulong_beta4/zulong/memory/hot_update_engine.py#L108-L109) 字典

```python
class HotUpdateEngine:
    def __init__(self):
        # 补丁存储
        self._patches: Dict[str, SystemPatch] = {}
        self._active_patches: Dict[str, List[SystemPatch]] = {}  # layer -> patches
```

**经验保存位置**: [`EnhancedExperienceStore._experiences`](file:///d:/AI/project/zulong_beta4/zulong/memory/enhanced_experience_store.py#L213-L214) 字典 + 向量数据库

```python
class EnhancedExperienceStore:
    def __init__(self):
        self._experiences: Dict[str, Experience] = {}  # 内存
        # 持久化：SQLite + Qdrant 向量数据库
```

---

#### 由谁来注入？

**注入执行者**: [`PatchApplier`](file:///d:/AI/project/zulong_beta4/zulong/memory/patch_applier.py#L47-L63)

**注入流程**:

```
HotUpdateEngine.apply_patch(patch)
  ↓
查找对应层的应用器
  ↓
PatchApplier.apply_to_l0/apply_to_l1a/apply_to_l1b(patch)
  ↓
验证参数范围
  ↓
更新参数值
  ↓
记录变更历史
```

**关键代码**:

**文件**: [`hot_update_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/memory/hot_update_engine.py#L292-L335)

```python
async def apply_patch(self, patch: SystemPatch) -> bool:
    """应用补丁"""
    
    # 1. 更新状态
    patch.status = PatchStatus.APPLYING
    self._patches[patch.patch_id] = patch
    
    # 2. 查找应用器
    if patch.target_layer not in self._patch_appliers:
        raise ValueError(f"未找到 {patch.target_layer} 层的应用器")
    
    applier = self._patch_appliers[patch.target_layer]
    
    # 3. 应用补丁
    success = await applier(patch)  # ✅ 调用 PatchApplier
    
    if success:
        patch.status = PatchStatus.APPLIED
        patch.applied_at = datetime.utcnow()
        
        # 4. 添加到活跃补丁
        self._active_patches[patch.target_layer].append(patch)
        
        return True
```

---

## 📊 完整数据流图

### 自主学习完整流程

```
┌─────────────────────────────────────────────────────────────┐
│                    任务执行阶段                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │   任务成功/失败          │
              └─────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐
    │  L2 捕获异常     │         │  用户说"成功了"  │
    └─────────────────┘         └─────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │ ❌ 不会立即触发复盘！    │
              │    等待触发器调度        │
              └─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    复盘触发阶段                              │
│  (trigger.py - 三种触发方式)                                │
│  1. 用户主动触发 (高优先级)                                  │
│  2. 安静模式触发 (中优先级)                                  │
│  3. 夜间定时触发 (低优先级)                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  失败案例分析器          │
              │  (failure_analyzer.py)  │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  生成 FailureCase 对象   │
              │  - case_id              │
              │  - error_type           │
              │  - root_cause           │
              │  - avoidance_guide      │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  保存到经验库            │
              │  (add_experience)       │
              └─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    经验存储阶段                              │
│  (enhanced_experience_store.py)                             │
│                                                             │
│  1. 创建 Experience 对象                                     │
│     - id, content, type, metadata                          │
│  2. 生成向量 (Embedding 模型)                                │
│     - embedding: [0.1, 0.2, ..., 0.9] (512 维)              │
│  3. 提取关键词 (BM25)                                        │
│     - keywords: ["抓取", "力度", "滑落"]                    │
│  4. 提取标签 (智能打标)                                      │
│     - tags: ["failure", "manipulation"]                    │
│  5. 存储到内存 + SQLite + Qdrant                            │
│  6. 【关键】事件触发！                                       │
│     asyncio.create_task(                                    │
│       hot_update_engine.on_experience_added(experience)     │
│     )                                                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    事件驱动触发                              │
│  (hot_update_engine.py)                                     │
│                                                             │
│  async def on_experience_added(self, experience):          │
│      # 检测到新经验                                         │
│      patch = await self._generate_patch_from_experience()  │
│      await self.apply_patch(patch)                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  判断经验类型            │
              │  - failure → 参数补丁    │
              │  - success → 策略补丁    │
              │  - preference → 规则补丁 │
              └─────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  生成 SystemPatch 对象    │
              │  - patch_id             │
              │  - patch_type           │
              │  - target_layer         │
              │  - adjustment           │
              └─────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    补丁应用阶段                              │
│  (patch_applier.py)                                         │
│                                                             │
│  1. 验证补丁类型                                            │
│  2. 验证参数范围                                            │
│  3. 应用到对应层                                            │
│     - L0: _l0_parameters[param].value = new_value          │
│     - L1-A: _l1a_rules[rule_id].update(data)               │
│     - L1-B: _l1b_strategies[strategy_id].update(data)      │
│  4. 记录变更历史                                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  参数已更新 ✅            │
              │  下次任务执行改进后行为  │
              └─────────────────────────┘
```

---

## 🔍 关键概念对比表

| 概念 | 定义 | 生成者 | 保存位置 | 用途 |
|------|------|--------|---------|------|
| **经验 (Experience)** | 任务成功/失败的结构化记录 | 复盘机制 | 经验库 (内存+SQLite+Qdrant) | 知识存储、检索 |
| **经验向量 (Embedding)** | 经验的向量表示 | Embedding 模型 | Experience.embedding | 语义检索 |
| **补丁 (SystemPatch)** | 参数调整指令 | 热更新引擎 | HotUpdateEngine._patches | 指导参数调整 |
| **参数 (Parameter)** | L0/L1 层的配置值 | 系统初始化 | PatchApplier._l0_parameters | 执行器配置 |

---

## ⚠️ 关键发现

### 发现 1: 复盘机制并非自动触发

**问题**: 当前系统中，任务失败后**不会立即触发复盘**，而是等待触发器调度。

**影响**: 
- ✅ 避免频繁复盘消耗资源
- ❌ 可能导致失败经验延迟注入
- ⚠️ 如果触发器未启动，失败经验永远不会被分析

**建议**: 
- 考虑在 L2 捕获异常时自动触发轻量级复盘
- 或者确保复盘触发器始终运行

---

### 发现 2: 经验和补丁是完全不同的概念

**混淆点**: 容易将"经验向量"和"补丁"混为一谈。

**澄清**:
- **经验向量** = 知识的向量表示（用于检索）
- **补丁** = 参数调整指令（用于执行）

**关系**: 
```
经验 (Experience)
  ↓ (被热更新引擎检测)
补丁 (SystemPatch)
  ↓ (被 PatchApplier 应用)
参数调整 (Parameter Update)
```

---

### 发现 3: 事件驱动只在经验写入时触发

**关键代码**:

**文件**: [`enhanced_experience_store.py`](file:///d:/AI/project/zulong_beta4/zulong/memory/enhanced_experience_store.py#L450-L462)

```python
# 【关键】写入成功后，立即触发补丁生成（事件驱动）
if self.hot_update_engine:
    try:
        # 异步调用，不阻塞主流程
        asyncio.create_task(
            self.hot_update_engine.on_experience_added(experience)
        )
        logger.debug(f"[EnhancedExperienceStore] 已触发补丁生成：{exp_id[:8]}")
    except Exception as e:
        logger.error(f"[EnhancedExperienceStore] 触发补丁生成失败：{e}")
```

**流程**:
```
经验库写入 (add_experience)
  ↓
asyncio.create_task(on_experience_added)
  ↓
热更新引擎处理
  ↓
生成补丁
  ↓
应用补丁
```

---

### 发现 4: L2 推理引擎不直接参与经验生成

**发现**: [`inference_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/inference_engine.py) 中**没有直接调用**经验库或复盘机制的代码。

**实际流程**:
```
L2 推理失败
  ↓
抛出异常
  ↓
【由上层捕获并决定是否触发复盘】
  ↓
复盘触发器 (trigger.py)
  ↓
失败案例分析器 (failure_analyzer.py)
```

**结论**: L2 只负责推理，复盘由独立的复盘模块负责。

---

## 📝 总结

### 问题 1 答案
❌ **任务失败不会立即触发复盘**，而是由复盘触发器（trigger.py）调度触发（用户主动/安静模式/夜间定时）。

### 问题 2 答案
❌ **补丁和经验向量不是同一个东西**：
- **经验向量**: 512 维浮点数组，用于语义检索
- **补丁**: 结构化调整指令，用于参数更新

### 问题 3 答案
**补丁由热更新引擎生成**，根据经验类型判断：
- `failure` → 参数调整补丁 (L0)
- `success` → 策略优化补丁 (L1-B)
- `preference` → 规则更新补丁 (L1-A)

### 问题 4 答案
**补丁注入到 PatchApplier 的参数/规则/策略字典中**，由 PatchApplier 负责执行注入。

---

**维护者**: 祖龙 (ZULONG) 系统架构组  
**文档版本**: v1.0  
**创建日期**: 2026-03-30
