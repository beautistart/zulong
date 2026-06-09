# 祖龙经验自动提炼与应用机制设计纪要

日期：2026-05-31

本文记录“祖龙系统如何自动提炼经验，并让经验作用于 LLM 决策”的讨论结论。本文必须服从 `TSD/祖龙 (ZULONG) 机器人系统技术规格说明书 (TSD)1.7.txt`，尤其对齐以下主线：

- L1-B 只做工具预判、上下文检索、记忆打包与输出模态辅助。
- L2/FC 负责真实工具决策与执行循环。
- 任务执行过程必须进入 `InteractionStore -> MemoryGraph -> ExperienceStore` 闭环。
- MemoryGraph 与 TaskGraph 共同构成图记忆系统，是经验提炼的语义权威来源。

## 1. 核心原则

经验不是日志，不是完整对话，也不是一次性工具结果。经验必须是可复用、可泛化、可检索、可追溯的结构化规则。

经验库只保存方法论和摘要索引，不保存完整原始细节。完整证据、执行链、失败节点、成功节点和上下文关系由 MemoryGraph/TaskGraph 承载。

用户指令优先级最高，可以覆盖经验建议。但当用户指令与已召回经验明显冲突时，系统必须弹窗提示建议方式、风险和覆盖后果，并等待用户确认后再继续。

## 2. 经验与记忆的边界

### 2.1 入经验库

以下内容可以成为经验：

- 任务执行成功流程。
- 任务执行失败教训。
- 用户纠正后形成的可复用规则。
- 错误工具链及其规避方式。
- 结构化 `IF/THEN` 执行规则。
- “失败后采用某种做法成功”的纠正链经验。

### 2.2 入图记忆但不入经验库

以下内容应作为记忆保存，不作为经验库条目：

- 用户偏好。
- 审批偏好。
- 对某个会话、某个项目、某个人的局部要求。
- 用户覆盖经验建议的决策记录。

这些内容可以影响后续召回，但不应污染 ExperienceStore 的通用经验集合。

### 2.3 不应长期保存为经验

以下内容不得进入经验库：

- 不可泛化的一次性内容。
- 完整命令。
- 绝对路径。
- 用户隐私。
- 临时代码片段。
- 一次性日志。
- 未经证据支持的模型幻觉判断。

如确需用于复盘，应只保留在原始事件或图节点的证据链中，并做摘要化、脱敏和衰减。

## 3. 权威数据源

经验提炼的数据源以 `MemoryGraph + TaskGraph` 为准。

`InteractionStore` 仍保留为原始事件流水和可审计账本，但不作为经验判断的最终权威。推荐分工如下：

| 层 | 职责 |
|---|---|
| InteractionStore | 原始事件、卡片回放、审计流水 |
| MemoryGraph | 工具、审批、失败、成功、纠正、任务摘要等语义节点和边 |
| TaskGraph | 任务目标、子任务状态、失败节点、成功节点、执行归属 |
| TaskExecutionTrace | 从图中聚合出的任务执行摘要 |
| ExperienceStore | 经验摘要索引、标签、置信度、图节点地址 |

## 4. LLM 生成与审查

经验候选生成交给 LLM，经验审查也交给 LLM，形成“候选生成 LLM + 审查 LLM”两段式机制。

### 4.1 候选生成 LLM

候选生成 LLM 的职责：

- 从 MemoryGraph/TaskGraph 聚合上下文。
- 判断是否值得沉淀经验。
- 生成结构化 `IF/THEN` 经验候选。
- 给出类型、置信度、适用条件、证据图节点地址。

推荐候选格式：

```text
TYPE: failure | procedure | correction
IF: 适用场景条件
THEN: 建议动作
BECAUSE: 证据摘要
AVOID: 应避免动作
SOURCE_GRAPH_NODES: [...]
SOURCE_TASK_NODES: [...]
CONFIDENCE: 0.0-1.0
```

### 4.2 审查 LLM

审查 LLM 的职责：

- 判断候选是否可泛化。
- 判断是否包含路径、隐私、完整命令等不该入库内容。
- 判断是否与已有经验重复或冲突。
- 决定入库、挂起、合并、降权或丢弃。

审查不得阻塞主任务。审查结果可异步写回 MemoryGraph 与 ExperienceStore。

## 5. 触发机制

以下事件都可以触发经验候选生成：

- 任务完成。
- 任务失败。
- 用户说“不是这样”或语义等价纠正。
- 用户显式要求复盘。
- 任务恢复失败。
- LLM 在执行中主动意识到“这里值得沉淀经验”。

### 5.1 LLM 主动提议经验

可以提供工具 `propose_experience_candidate`。L2/LLM 在执行中发现可沉淀经验时调用该工具提交候选，但不得直接写入正式经验库。

工具调用只创建候选或挂起节点，最终入库仍经过审查 LLM。

## 6. 实时失败经验与成功经验节奏

实时失败经验需要立即总结并应用，帮助当前任务后续执行避坑。

成功经验暂时挂起，任务完成后再做完整入库总结、审查、去重和摘要化。

用户纠正时，系统应自动把用户语义锚定的错误点挂起或标记；等任务成功完成后，再把失败错误节点与成功节点建立关联，生成“失败后如何成功”的经验链。

所有经验提炼都不能打断主任务、不能阻塞任务执行。失败经验的快速回注只允许注入摘要提示或临时热经验。

## 7. 长期经验框架与实时经验微步

祖龙的经验应用分为两种颗粒度：

### 7.1 长期经验：任务级常驻框架

长期经验是大框架和方向，来自 ExperienceStore 摘要索引与 MemoryGraph 经验节点。

它的作用是给当前任务建立稳定方向，例如：

```text
IF 新建 Web 项目并写代码
THEN 先建立 TaskGraph/workspace，再逐文件写入并验证
AVOID 直接用单个写文件工具替代项目生命周期
```

长期经验不应在每个 FC 微步反复检索，也不应整段塞入上下文。推荐流程：

```text
用户任务开始 / TaskGraph 创建
-> L1-B 工具预判 + 冷经验检索
-> 形成 ExperienceFrame
-> ExperienceFrame 在当前任务内常驻
```

`ExperienceFrame` 只保存少量高相关长期经验摘要、图节点地址和适用边界。它是任务级方向，不负责每一步的细节调整。

### 7.2 实时经验：FC 微步动态细节

实时经验是当前任务、当前节点、当前工具链里的现场调整。它来自 LLM 每一步要做的事、工具调用、工具结果、用户纠正、审批结果和当前 TaskGraph 状态。

每个 FC 微步中，LLM 在执行下一步任务前，必须先产生结构化动作意图：

```text
NEXT_ACTION: 下一步要干什么
TOOL_CALL: 准备调用什么工具
TASK_NODE: 当前对应的任务节点
WHY: 为什么做这一步
```

系统拿这一步的 `NEXT_ACTION + TOOL_CALL + TASK_NODE + ExperienceFrame` 做 MemoryGraph 扩散和经验检索，只返回当前步需要的实时经验数据。

实时经验返回的是 `RuntimeExperienceObservation`，而不是长期经验全文：

```text
RUNTIME_EXPERIENCE_OBSERVATION:
- 当前步骤命中的实时经验摘要
- 应避免的重复失败动作
- 可替代动作
- 关联的失败/成功节点地址
- 是否来自用户纠正、工具失败或审批拒绝
```

实时经验只服务下一步或当前任务局部上下文，可频繁更新、覆盖和失效。它不自动进入 ExperienceStore；只有任务结束后，经过候选生成 LLM 和审查 LLM，才可能沉淀为长期经验或纯经验节点。

### 7.3 FC 微步不是完整用户轮次

一次用户任务可以包含多个 FC 微步。LLM 输出“下一步要做什么 + 工具调用”只表示完成了一次局部推理，不表示整个任务完成。

任务状态继续存在于：

- TaskGraph 当前节点。
- FC state。
- 工具 action/observation 账本。
- ExperienceFrame。
- RuntimeExperienceObservation。
- MemoryGraph 激活路径。

因此，实时经验必须进入下一次 FC 微步的结构化 observation，让 LLM 能够读取与注意到；但它不需要作为普通用户对话文本展示。

## 8. 经验节点化与经验链

经验应首先作为 MemoryGraph 中的 EXPERIENCE 节点存在，而不是只存在于 ExperienceStore 文档中。

ExperienceStore/RAG 保存：

- 经验摘要。
- `experience_graph_node_id`。
- 标签。
- 置信度。
- 适用场景。
- 时间/温度/重要度元数据。

完整经验数据由 MemoryGraph 节点和边承载。召回 ExperienceStore 摘要后，应通过图节点地址回跳 MemoryGraph 扩散检索完整上下文。

### 8.1 必要边类型

经验链推荐至少支持以下边类型：

| 边类型 | 含义 |
|---|---|
| DERIVED_FROM | 经验来源于某任务、某轮对话或某个执行 trace |
| APPLIES_TO | 经验适用于某类任务、工具、节点或场景 |
| SIMILAR_TO | 经验之间语义相似 |
| CORRECTS | 新经验修正旧经验或修正错误节点 |
| DEPENDS_ON | 某经验依赖另一经验或前置条件 |
| CONTRADICTS | 经验之间存在冲突 |
| FAILED_THEN_SUCCEEDED | 从失败节点指向后续成功节点，表示“这次失败后采用该做法成功” |

`FAILED_THEN_SUCCEEDED` 是必需边，用于表达失败到成功的可学习路径。它应连接：

```text
failure_node -> success_node
failure_node -> correction_experience_node
correction_experience_node -> success_node
```

这样后续遇到相似失败时，MemoryGraph 可以沿失败节点扩散到成功做法，而不只是召回一条孤立经验文本。

## 9. 纯经验节点

当图节点进入淘汰阶段时，如果其普通记忆内容不再重要，但节点承载的重要经验仍有价值，则不应删除整个节点。

系统应清除普通记忆内容，保留经验结构和必要摘要，将该节点降级为“纯经验节点”。

纯经验节点特点：

- 不保留完整原始上下文。
- 保留经验摘要、标签、置信度和关键边。
- 保留来源图地址或可审计摘要。
- 可单独导出。
- 可与另一个 MemoryGraph 合并。

合并早期允许完整记忆节点、经验节点、纯经验节点混合存在。后续纯经验节点被复用并在新图谱中建立完整上下文后，旧的冗余纯经验节点可逐步修剪。

## 10. 经验应用方式

经验应用与工具调用保持同一逻辑：

```text
L1-B 工具预判
-> 任务级冷经验检索形成 ExperienceFrame
-> 每个 FC 微步由 LLM 输出 NEXT_ACTION + TOOL_CALL
-> 系统基于 NEXT_ACTION + TOOL_CALL + TASK_NODE 做实时经验扩散
-> 返回 RuntimeExperienceObservation
-> L2/FC 按 ExperienceFrame 和 RuntimeExperienceObservation 调整下一步工具选择和执行顺序
```

系统不应把长期经验固定注入每轮 prompt，以免上下文污染。长期经验以 ExperienceFrame 的摘要形式常驻，实时经验以 RuntimeExperienceObservation 的结构化 observation 进入下一次 FC 微步。

当 LLM 需要长期经验细节时，再调用经验或图记忆工具读取：

```text
search_experience -> 返回摘要和 graph_node_id
read_memory_node / discover_related -> 通过 graph_node_id 回跳 MemoryGraph
```

实时经验检索返回的必须是当前步相关数据，不返回大段长期经验全文。

Web 端需要展示经验使用情况，但默认折叠。展示内容包括：

- 任务级 ExperienceFrame 是否启用。
- 当前 FC 微步是否产生 RuntimeExperienceObservation。
- 命中的实时经验摘要。
- 是否应用实时经验。
- 是否被用户覆盖。
- 经验来源图节点地址。

## 11. 错误工具链的约束层级

错误工具链经验默认作为提示词软约束，而不是直接升级为系统硬约束。

建议分级：

| 级别 | 含义 | 生效条件 |
|---|---|---|
| 经验软约束 | LLM 召回后参考 | 默认 |
| 会话级临时硬约束 | 当前任务内阻止重复踩坑 | 同一任务重复失败或实时失败经验触发 |
| 系统级硬约束 | 写入工具边界或执行守卫 | 多次任务重复出现、TSD 支持、人工确认 |

这样既能约束 LLM，又避免把某个具体任务的局部规则误写成全局硬约束。

## 12. 用户覆盖经验

用户指令最高优先级，可以覆盖经验建议。

当覆盖发生时：

1. 系统弹窗提示历史经验建议和风险。
2. 用户确认后继续执行用户指定方式。
3. 覆盖行为保存为图记忆事件。
4. 覆盖行为不默认进入 ExperienceStore。
5. 如果覆盖后成功，可在任务结束后由审查 LLM 判断是否形成新经验。

## 13. 强化、去重与淘汰

经验强化和淘汰暂定沿用 MemoryGraph 的分级方式：

- HOT/WARM/COLD 温度。
- 重要度。
- 访问次数。
- 成功复用次数。
- 失败复用次数。
- 时间衰减。

去重优先级高于强化策略。相同或高度相似的经验应合并为经验链或更新旧经验，而不是重复写入 ExperienceStore。

后续待讨论问题：

- 相似经验的合并阈值。
- 冲突经验的优先级。
- 纯经验节点被复用后的修剪条件。
- 失败经验实时生效的具体有效期。

## 14. 与现有实现的关系

当前已有或正在使用的落点：

- `zulong/launcher/web_chat_router.py`：交互事件持久化与任务状态。
- `zulong/launcher/memory_mirror.py`：Web/工具事件镜像到 MemoryGraph。
- `zulong/review/task_execution_extractor.py`：从任务执行事件聚合 trace。
- `zulong/review/task_experience_generator.py`：从 trace 生成经验候选。
- `zulong/memory/enhanced_experience_store.py`：增强经验库。
- `zulong/tools/experience_tool.py`：`search_experience` FC 工具。
- `zulong/tools/tool_bag.py`：L1-B 工具预判与经验检索工具注入。

后续实现应围绕以上模块演进，不改变 TSD 的 L1-B -> L2 -> FC -> MemoryGraph 主链。
