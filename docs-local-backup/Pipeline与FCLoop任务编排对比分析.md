# Pipeline 与 FC Loop 任务编排对比分析

## 背景

祖龙系统经历了一次重大架构升级：从 **Pipeline 五阶段流水线架构**（`pipeline/orchestrator.py`）迁移到 **统一 FC Loop 架构**（`inference_engine.py`）。迁移目标是消除双路径、统一消息处理，但在迁移过程中，**部分关键模型约束机制被遗漏**，导致小参数模型（如 qwen3.5:4b）在"已有任务修改"场景下无法可靠调用工具。

本文档记录完整对比分析和修复方案。

---

## 一、架构演化路径

```
旧架构（双路径）:
  用户 → Gatekeeper → [硬编码路由]
    ├─ 普通消息 → L2（无工具） → 单轮回复
    └─ 恢复关键词 → Orchestrator（200轮硬循环 + TaskGraph + 护栏 + 检查点）

新架构（统一路径）:
  用户 → Gatekeeper → L2（两阶段 FC）
    ├─ Round 1: start_session（tool_choice=required）→ 意图分类
    └─ Round 2: 场景过滤工具 + FC 自主循环（tool_choice=auto）
```

**参考文件**：
- 旧版: `safetensorsvers/架构重大升级前一版/pipeline/orchestrator.py` (958行)
- 旧版: `safetensorsvers/架构重大升级前一版/pipeline/agent_prompt.py` (134行)
- 旧版: `safetensorsvers/架构重大升级前一版/pipeline/tools.py` (36.7KB)
- 新版: `zulong/l2/inference_engine.py` (核心 FC Loop 在 1282-1572 行)
- 新版: `zulong/l2/intent_prompt_builder.py` (520行)

---

## 二、模型约束机制对比

### 2.1 总览对比表

| 约束机制 | 旧版 Pipeline Orchestrator | 当前 FC Loop | 缺失影响 |
|----------|--------------------------|-------------|---------|
| 连续无工具调用容忍 | `MAX_NO_TOOL_TURNS = 3`，连续3次纯文本后注入提醒并 continue | **无** — 第一次纯文本就 break 退出 | **致命** |
| 系统提示词详细度 | 111行完整提示词，10条工作守则，每个工具详细说明 | ~30行精简版，仅列工具名称 | **重要** |
| 周期性状态检查 | `STATE_CHECK_INTERVAL=10`，每10轮护栏引擎检查 | **无**（仅 Rule A 检查过早完成声明） | 中等 |
| 护栏引擎(GuardrailEngine) | 每次工具调用后验证，不合规时追加修正消息 | **无** | 中等 |
| 依赖审视提示 | 4+叶子节点无依赖时注入提醒 | **无** | 低 |
| 超时预警注入 | 80%时间后注入"请尽快用submit提交" | 仅日志警告，不注入消息 | 低 |
| tool_choice 设置 | `"auto"` | `"auto"`（RESUME首轮 `required`） | 相同 |

### 2.2 核心缺失：MAX_NO_TOOL_TURNS 回环机制

#### 旧版实现（orchestrator.py 第 355-378 行）

```python
MAX_NO_TOOL_TURNS = 3  # 连续纯文本输出的最大容忍次数

# --- 在主循环中 ---
if not tool_calls:
    no_tool_count += 1
    logger.info(f"[Agent] Turn {turn}: 纯文本输出 (连续 {no_tool_count} 次)")

    # 检查是否已通过 submit 结束
    if self.tool_registry.final_answer is not None:
        break

    # 超过容忍次数 → 提醒模型使用工具
    if no_tool_count >= MAX_NO_TOOL_TURNS:
        hint_msg = {
            "role": "user",
            "content": (
                "[系统提示] 请通过工具操作任务图谱来完成任务。"
                "如果已完成所有任务，请调用 submit_final_answer 提交最终答案。"
            ),
        }
        attn_window.register_message(hint_msg, turn=turn)
        no_tool_count = 0
    continue  # ← 关键！不 break，继续循环让模型再试
```

**效果**：模型连续输出纯文本时，系统会注入提醒消息引导其使用工具，最多容忍3次。

#### 当前实现（inference_engine.py 第 1469-1572 行）

```python
else:
    # 模型直接返回内容，无工具调用
    response = msg.content or ""
    
    # Rule A: 过早完成声明拦截
    # ... (检查后可能 continue)
    
    # InfoGap: 信息缺口检测
    # ... (检查后可能 continue)
    
    if _should_continue:
        continue
    break  # ← 第一次纯文本就直接退出！
```

**问题**：当模型选择用文本回答（而非调用工具），只要不触发 Rule A 或 InfoGap，FC Loop 立即终止。**没有给模型第二次尝试的机会**。

---

### 2.3 系统提示词详细度差异

#### 旧版 agent_prompt.py（111行完整版）

```
六大阶段工作流（分析→确认→规划→执行→审查→提交）
10条明确的工作守则：
  0. 智能判断：简单问题不创建图谱
  1. 先确认：模糊需求先 ask_user
  2. 先分析：第一步标记 analysis 为 in_progress
  3. 先规划后执行：至少 2 个大纲
  3.5. 建立依赖
  4. 按序执行
  5. 代码必写
  6. 结果必记
  7. 文件必关联
  8. 善用记忆
  9. 灵活调整
  10. 完整交付

每个工具的详细说明：
  - plan_add_node(parent_id, label, desc): 具体参数说明
  - plan_mark_status(node_id, status, result?): 完成时必须提供 result
  - view_focused_context(node_id): 执行前了解上下文
  - ...共 15 个工具的完整说明
```

#### 当前 intent_prompt_builder.py（~30行任务规则）

```
【任务管理规则】
当前已进入任务规划模式。系统已自动创建任务图骨架。
你需要做的：
1. 用 task_add_node 向任务图添加子任务节点
2. 用 task_add_dependency 声明节点间的先后顺序
3. 用 task_view_overview 查看任务概览确认结构
4. 按依赖顺序逐个执行子任务，用 task_mark_status 更新节点状态
5. 所有子任务完成后用 submit_final_answer 提交最终答案
```

**差异**：当前版本缺少对工具参数的详细说明、缺少明确的"不要纯文本输出"约束、缺少"先查看再操作"的引导。对大模型（671B）来说这已够用，但小模型（4B）需要更明确的指令。

---

### 2.4 护栏引擎缺失

旧版 `GuardrailEngine` 提供两个能力：

1. **check_tool_call()** — 每次工具调用后校验：
   - 节点状态合法性（不允许跳过中间状态）
   - 结果完整性（completed 必须有 result）
   - 文件关联一致性

2. **check_state()** — 每 10 轮整体检查：
   - 是否有 in_progress 超时的节点
   - 是否有依赖死锁
   - 是否有孤立节点

当前 FC Loop 仅有 Rule A（过早完成声明拦截）和 InfoGapDetector。

---

## 三、问题复现路径

### 场景：小模型修改已有任务

1. 用户发送 "把之前未完成的住宿建议和预算规划也完成掉，另外再加一个购物推荐的模块"
2. Round 1 意图分类 → COMPLEX（正确）
3. `StartSessionTool._handle_complex()` 检测到已有任务图 → 返回 `already_exists=True`
4. `_build_complex_prompt()` 注入 `already_exists` 提示（告知用 task_view_overview 查看）
5. Round 2 FC Loop 启动，tool_choice="auto"
6. **qwen3.5:4b 选择直接生成文本**（关于住宿、预算、购物的详细内容），不调用任何工具
7. FC Loop 检测到纯文本输出 → Rule A 未触发（模型没声称"已完成"） → InfoGap 未触发
8. **`break` — FC Loop 立即终止**
9. 文本回复发送给前端，但任务图未更新

### 如果有 MAX_NO_TOOL_TURNS 机制：

步骤 7 后：
- `no_tool_count = 1`（< 3），**不 break，而是 continue**
- 注入系统消息引导使用工具
- 模型第二次尝试，可能开始调用 task_view_overview
- 最多给 3 次机会

---

## 四、修复方案

### 方案 A：工具调用回环机制（核心修复）

在 `inference_engine.py` 的 FC Loop 中，模型输出纯文本且未触发 Rule A / InfoGap 时，检查是否有活跃任务图且有未完成节点。如果有，不 break 而是注入工具使用提醒后 continue。

**插入位置**：第 1570 行（`break` 之前）

```python
# ── 工具调用回环（借鉴旧版 Pipeline 的 MAX_NO_TOOL_TURNS 机制）──
_MAX_NO_TOOL_CONTINUES = 3  # 最多容忍 3 次纯文本输出
_has_active_incomplete_tasks = False
try:
    from zulong.tools.task_tools import get_active_task_graph
    _check_tg = get_active_task_graph()
    if _check_tg:
        _leaf_nodes = _check_tg.get_leaf_nodes()
        _uncompleted = [n for n in _leaf_nodes if n.status != "completed"]
        _has_active_incomplete_tasks = len(_uncompleted) > 0
except Exception:
    pass

if _has_active_incomplete_tasks and _no_tool_count < _MAX_NO_TOOL_CONTINUES:
    _no_tool_count += 1
    _nudge_msg = {
        "role": "system",
        "content": (
            f"[系统提示] 当前任务图仍有未完成的子任务。"
            f"请不要直接用文字回答，而是通过工具操作任务图谱：\n"
            f"1. 先用 task_view_overview 查看当前任务进度\n"
            f"2. 用 task_mark_status 完成未完成的任务\n"
            f"3. 用 task_add_node 添加新的子任务\n"
            f"（连续纯文本输出 {_no_tool_count}/{_MAX_NO_TOOL_CONTINUES}）"
        ),
    }
    messages.append({"role": "assistant", "content": response})
    messages.append(_nudge_msg)
    if self._attn_window:
        self._attn_window.register_message(
            {"role": "assistant", "content": response}, turn=fc_turn,
        )
        self._attn_window.register_message(_nudge_msg, turn=fc_turn)
    response = None
    _should_continue = True
    logger.info(
        f"[FC] 工具调用回环: 纯文本输出 {_no_tool_count}/{_MAX_NO_TOOL_CONTINUES}，"
        f"注入工具使用提醒"
    )
```

需要初始化计数器（在 FC Loop 开始前）：
```python
_no_tool_count = 0  # 连续纯文本输出计数
```

以及在有工具调用时重置：
```python
if msg.tool_calls:
    _no_tool_count = 0  # 重置计数
    # ... 原有工具调用处理 ...
```

### 方案 B：增强 already_exists 提示词

在 `intent_prompt_builder.py` 中，当 `already_exists=True` 时，增加更强的工具使用约束：

```python
elif scaffold_data.get("already_exists") and graph_id:
    system_parts.append(
        f"\n【当前任务图（已有）】\n"
        f"图谱ID: {graph_id}\n"
        f"任务: {title}\n"
        f"这是一个已经存在的任务图，用户正在此基础上提出新的要求。\n"
        f"\n"
        f"⚠️ 重要规则：你必须通过工具来操作任务图，不能只用文字回答。\n"
        f"请严格按以下步骤操作：\n"
        f"第一步：调用 task_view_overview 查看当前任务图的完整状态\n"
        f"第二步：根据用户的新需求，用工具操作任务图：\n"
        f"  - task_mark_status(node_id, status='completed', result='...') 完成未完成的任务\n"
        f"  - task_add_node(parent_id='req', label='...', desc='...') 添加新的子任务\n"
        f"第三步：所有操作完成后，用 submit_final_answer 提交汇总\n"
        f"\n"
        f"禁止：直接用文字回答用户的问题而不调用任何工具。\n"
    )
```

### 方案 C：已有任务修改时首轮强制工具（可选增强）

类似 RESUME 场景的 `_force_first_tool` 机制，当 `already_exists=True` 时，Round 2 第一轮也强制调用 `task_view_overview`：

```python
# inference_engine.py 中，FC Loop 的 tool_choice 设置
if _force_first_tool and fc_turn == 1:
    api_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": "task_view_overview"}
    }
elif _force_already_exists_first_tool and fc_turn == 1:
    # already_exists 场景也强制首轮查看
    api_kwargs["tool_choice"] = {
        "type": "function",
        "function": {"name": "task_view_overview"}
    }
else:
    api_kwargs["tool_choice"] = "auto"
```

---

## 五、优先级建议

| 方案 | 优先级 | 改动范围 | 预期效果 |
|------|--------|---------|---------|
| A: 工具调用回环 | **P0** | inference_engine.py (~30行) | 给小模型 3 次重试机会，直接解决"一次纯文本就终止"的问题 |
| B: 提示词增强 | **P1** | intent_prompt_builder.py (~15行) | 更强的指令约束，配合方案A效果更好 |
| C: 首轮强制工具 | P2 | inference_engine.py (~10行) | 确保小模型至少先看到任务图状态 |

建议实施顺序：A → B → C，每步完成后测试验证。

---

## 六、旧版 Pipeline 的三层控制哲学

旧版 `orchestrator.py` 开头注释明确说明了三层控制机制：

```
三层控制机制：
  - 第一层 Prompt：系统提示词引导方向（agent_prompt.py）
  - 第二层 FC Schema：结构化约束（tools.py 的参数类型和 enum）
  - 第三层 Guardrails：质量保障（guardrails.py）
```

当前 FC Loop 保留了第一层（Prompt）和第二层（FC Schema），但**第三层（Guardrails）被完全移除**。方案 A 的回环机制实质上是在恢复第三层的部分能力——当模型行为偏离预期时，系统主动干预纠正。

---

## 附录：测试验证记录

### 测试环境
- 模型: qwen3.5:4b (4.7B 参数, Q4_K_M 量化)
- Ollama 本地推理
- 系统版本: zulong_beta4

### 测试一：复杂任务创建（成功）
- 输入: "帮我制定一个为期三天的北京旅游攻略，包括景点推荐、美食推荐和交通安排"
- 结果: 创建任务图 tg_1777098582，7个子任务，3个完成，17轮工具调用
- 结论: 4B 模型在初始任务创建时工具调用能力正常

### 测试二：已有任务修改（失败）
- 输入: "把之前未完成的住宿建议和预算规划也完成掉，另外再加一个购物推荐的模块"
- 结果: 模型生成文本但未调用任何工具，任务图未更新
- 原因: FC Loop 第一次纯文本输出就 break，缺少回环机制
- Rule C 修复验证通过: 任务图引用未被清空
- already_exists 分支生效: StartSessionTool 正确返回 already_exists=True

### 已验证的修复项
| 修复项 | 状态 |
|--------|------|
| Fix 1: _classify_intent() 429 备用模型回退 | 代码已就位，未触发验证 |
| Fix 2: Rule C 不清空活跃图谱引用 | ✅ 已验证通过 |
| Fix 3: already_exists 提示分支 | ✅ 已生效（但小模型未遵循） |
