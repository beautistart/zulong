# 任务恢复流程重设计方案

## 背景 (Context)

用户审查了 `记忆系统架构文档.md` 后提出 5 项关键修正：
1. **注意力层级表（493-499行）描述错误** -- 当前描述的是视觉处理注意力，不是图记忆动态注意力
2. **禁止硬编码语义筛选** -- `InferenceEngine 检测到恢复关键词` 这种方式违反设计理念
3. **恢复流程不是基于图记忆的** -- 当前流程是硬编码关键词链，不是模型自主决策
4. **严厉禁止硬编码语义识别** -- 所有"是否恢复"的判断应由 L2 模型自己完成

**核心原则：3D 记忆城市中，L2 模型拥有最大自由度，通过 FC 工具自主导航记忆图、自主发现旧任务、自主决定是否恢复。系统层不做意图预判。**

---

## 一、架构文档修正

### 修改文件: `docs/记忆系统架构文档.md`

### 1.1 注意力层级表（第 493-499 行）

**当前（错误）：** 表格描述了 L0_SENSOR ~ L3_COGNITIVE 四层，是视觉处理的注意力层级。

**改为：** 基于图记忆的动态注意力分两个板块：

**板块一：思维深度导航**
- L2 模型通过 `navigate_attention` FC 工具主动控制注意力焦点
- 三种导航操作：`deeper`（深入子节点）、`broader`（回退父节点）、`jump`（跳转指定节点）
- 对应代码：`attention_tool.py` NavigateAttentionTool

**板块二：基于 BFS 扩散的三类注意力**
| 类型 | 机制 | 代码对应 |
|------|------|---------|
| 全局注意力 | BFS 从种子节点沿加权边扩散，decay 衰减，覆盖整张图 | `memory_graph.py` compute_activations() |
| 单链注意力 | 沿 HIERARCHY 父子链聚焦，从当前焦点到根的纵向路径 | `attention_tool.py` deeper/broader |
| 局部注意力 | 当前焦点节点的直接邻域（1-hop neighbors） | `memory_graph_tools.py` discover_related |

保留原有 `AttentionLayer` 枚举（L0_SENSOR ~ L3_COGNITIVE）的说明，但明确它是**事件路由层级**，不是注意力机制本身。

### 1.2 恢复流程（第 1066-1090 行）

**当前（错误）：** 硬编码流程 `用户说"继续" → InferenceEngine检测关键词 → find_by_description`

**改为：** 基于图记忆的自主恢复流程（见下方第二节详细设计）

### 1.3 对话流程图 Gatekeeper 部分（第 1173-1179 行）

**当前（错误）：** `意图识别: 复杂任务`

**改为：** Gatekeeper 只做物理层打包（语音模式检测、上下文快照、注意力层级判断），不做语义意图识别。

---

## 二、代码修改方案

### 总体思路

**删除方向：** 移除所有 `is_resume` 标志位及其传播链路
**保留方向：** L2 模型通过现有 FC 工具自主完成恢复判断

**现有可用的 FC 工具（无需新增）：**
- `task_list_suspended` -- 列出/查询挂起任务
- `resume_task` -- 恢复指定任务
- `recall_memory` -- 从记忆图检索相关记忆
- `read_memory_node` -- 读取指定记忆节点
- `discover_related` -- 发现关联节点
- `navigate_attention` -- 导航思维焦点

**新的恢复流程（L2 自主决策）：**
```
用户输入（任意内容）
  |
  v
L1-B Gatekeeper（物理层打包，不判断意图）
  |-- 语音模式检测
  |-- 上下文快照收集
  |-- 注意力层级路由
  |-- 打包发送给 L2（无 is_resume 字段）
  |
  v
L2 InferenceEngine（模型自主 FC 循环）
  |-- 模型看到用户说"继续上次的" → 自己决定调 task_list_suspended
  |-- 模型看到用户说"帮我做XXX" → 自己决定是新任务还是旧任务
  |-- 模型通过 recall_memory / discover_related 在记忆图中搜索
  |-- 模型自主决定是否调 resume_task
  |
  v
Session 绑定（事后绑定）
  |-- 若恢复了旧任务 → 通过 task_id 查找原 session 并绑定
  |-- 若是新任务 → 正常的 embedding 相似度分配
```

---

### 2.1 文件: `zulong/l1b/scheduler_gatekeeper.py`

**改动范围：** 两处几乎相同的硬编码块

**位置 A（第 1347-1406 行）：** `_package_normal_task()` 方法中
- 删除第 1347-1396 行：整个 `_resume_keywords` 定义、`is_resume` 检测、`TaskSuspensionManager` 调用、兜底时间排序逻辑
- 修改第 1399-1406 行的 `packaged_task` 字典：移除 `is_resume` 和 `resume_task_id` 两个字段

**位置 B（第 1790-1849 行）：** 另一个打包方法中（重复代码）
- 同样删除 `_resume_keywords` 定义和整个 `if is_resume:` 块
- 同样从 `packaged_task` 字典中移除 `is_resume` 和 `resume_task_id`

### 2.2 文件: `zulong/l2/inference_engine.py`

**改动范围：** 5 处 `is_resume` 传播点 + 1 处恢复引导注入

| 行号 | 当前代码 | 改动 |
|------|---------|------|
| 620 | `is_resume = event.payload.get("is_resume", False)` | 删除此行 |
| 625-626 | `self._is_resume = is_resume` / `self._resume_task_id = ...` | 删除这两行 |
| 679 | `..., is_resume=is_resume)` | 移除 `is_resume` 参数 |
| 810 | `def _process_with_memory(..., is_resume: bool = False):` | 移除 `is_resume` 参数 |
| 889-899 | 硬编码注入 `【任务恢复模式】` 引导消息 | **删除整个 if is_resume 块** |
| 1516-1521 | `is_resume = getattr(self, '_is_resume', False)` + 传参 | 删除，改为 `is_resume=False`（或直接移除参数） |
| 2201-2206 | 同上（第二处 session 分配） | 同上处理 |

**关键删除：第 889-899 行的恢复引导注入块**
```python
# 删除以下整块（这是硬编码恢复引导）：
if is_resume:
    resume_task_id = getattr(self, '_resume_task_id', "")
    resume_guidance = (
        "【任务恢复模式】用户正在恢复一个之前挂起的任务。..."
    )
    messages.append({"role": "system", "content": resume_guidance})
```

### 2.3 文件: `zulong/memory/graph_adapters.py`

**改动范围：** 2 个方法的 `is_resume` 参数

**ensure_session()（第 486-550 行）：**
- 移除 `is_resume: bool = False` 参数
- 移除第 514-527 行的 `if is_resume` 特殊分支
- 保留正常的 session 查找逻辑

**assign_session_by_similarity()（第 556-638 行）：**
- 移除 `is_resume: bool = False` 参数
- 移除第 585 行的 `if is_resume and resume_task_id` 特殊分支
- 移除第 638 行的 `bound_task_id=resume_task_id if is_resume else None` 条件赋值
- 保留 embedding 相似度匹配和新建 session 的正常逻辑

**新增事后绑定机制：**
- 当 L2 模型通过 `resume_task` 工具恢复任务后，在 `TaskResumeTool.execute()` 的返回路径中调用 `bind_session_to_task()` 把当前 session 绑定到恢复的任务
- 这是事后绑定（post-hoc），而非事前预判（pre-judgment）

### 2.4 文件: `zulong/l2/task_suspension.py`

**改动范围：** `find_by_description()` 方法（第 228-296 行）

- 删除第 253-258 行的硬编码 `resume_words` 列表和清洗逻辑：
  ```python
  # 删除以下代码：
  resume_words = ['继续', '接着', '上次', '恢复', '回到', '之前', '的', '任务', '那个']
  query_clean = query_lower
  for w in resume_words:
      query_clean = query_clean.replace(w, '')
  query_clean = query_clean.strip()
  ```
- 改为直接使用原始 query 进行匹配（L2 模型会传入有意义的描述，不需要系统层清洗）
- 如果 query 为空或仅含停用词，按时间排序返回最近任务（保留现有兜底逻辑）

### 2.5 文件: `zulong/tools/task_tools.py`

**改动范围：** TaskListSuspendedTool 的 description（第 771-775 行）

**当前：**
```python
"列出所有已挂起的任务。当用户说'继续'、'接着做'、'上次那个任务'等，
先调用此工具查看有哪些挂起的任务，然后决定恢复哪个。
也可以传入 query 参数来按描述模糊匹配。"
```

**改为（中性描述）：**
```python
"列出所有已挂起的任务，返回任务 ID、描述、挂起时间等摘要信息。
可传入 query 参数按描述模糊匹配特定任务。"
```

移除对用户措辞的硬编码假设，让模型自主判断何时调用此工具。

### 2.6 文件: `zulong/l2/recovery_notifier.py`

**改动范围：** 通知文本（第 157 行）

**当前：**
```python
"说「继续」恢复最近的任务，或描述新需求开始新任务。"
```

**改为：**
```python
"有未完成的任务可供恢复，你可以告诉我需要继续哪个任务，或直接开始新的任务。"
```

这是面向用户的友好提示，不涉及系统内部的语义检测逻辑，属于合理的引导文案（非硬编码关键词检测）。

---

## 三、修改顺序

1. `scheduler_gatekeeper.py` -- 切断硬编码检测的源头
2. `inference_engine.py` -- 移除 is_resume 传播和恢复引导注入
3. `graph_adapters.py` -- 移除 is_resume 参数，改为事后绑定
4. `task_suspension.py` -- 移除 resume_words 硬编码清洗
5. `task_tools.py` -- 更新工具描述为中性文案
6. `recovery_notifier.py` -- 更新用户通知文案
7. `记忆系统架构文档.md` -- 同步修正注意力层级表、恢复流程、对话流程图

---

## 四、不变的部分（明确保留）

- `TaskSuspensionManager` 的核心功能（挂起/恢复/持久化）完全保留
- `TaskListSuspendedTool` 和 `TaskResumeTool` 作为 FC 工具完全保留
- `RecoveryNotifier` 的启动扫描逻辑保留（只改通知文案）
- `MemoryGraph` 的 BFS 扩散激活、search、get_neighbors 全部保留
- `NavigateAttentionTool` 的 deeper/broader/jump 全部保留
- `AttentionLayer` 枚举（L0~L3）保留（它是事件路由层级，不是注意力机制）
- `graph_adapters.py` 的 `bind_session_to_task()`、`_propagate_address_to_tasks()` 保留

---

## 五、验证方法

1. **语法验证：** 对所有修改的 .py 文件运行 `python -m py_compile`
2. **接口一致性：** Grep 确认所有 `is_resume` 引用已被移除或更新
3. **功能验证思路：**
   - 用户说"继续上次的任务" → Gatekeeper 不做特殊处理，正常打包发给 L2
   - L2 模型看到消息后，自主决定调用 `task_list_suspended` 查看挂起任务
   - L2 模型自主决定调用 `resume_task` 恢复特定任务
   - Session 在恢复后通过 `bind_session_to_task()` 事后绑定

---

## 六、影响评估

- **删除代码量：** ~150 行硬编码恢复检测逻辑
- **新增代码量：** ~5 行（事后绑定调用）
- **破坏性：** 无。所有 FC 工具保持不变，L2 模型原有的 FC 循环能力完全支持自主恢复
- **向后兼容：** `packaged_task` 字典移除 `is_resume` 字段，但 InferenceEngine 已经不再读取该字段，不影响其他消费者
