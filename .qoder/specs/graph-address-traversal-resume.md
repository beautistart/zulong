# MemoryGraph 地址遍历恢复任务

## Context

当前任务恢复依赖重量级路径：挂起 -> JSON序列化 -> 磁盘文件 -> 加载 -> 反序列化 -> 设置活跃图 -> 恢复。但 MemoryGraph 中已经存储了所有任务节点的完整信息（状态、描述、结果、依赖关系），LLM 可以通过 `resolve_address()`、BFS 遍历、`compute_activations()` 等机制直接访问。

**核心设计思想**：LLM 拥有节点地址后，通过注意力模式和 BFS 机制可以调阅任何节点的任何信息。无需依赖文件系统来"恢复"任务——从 MemoryGraph 直接重建即可。

**目标链路**：
```
用户引用节点 @[节点名#tg:abc/task:o3]
    -> resolve_address() 找到 GraphNode
    -> 提取 graph_id
    -> BFS 遍历 HIERARCHY/DEPENDENCY 边收集所有节点
    -> 重建 TaskGraph 运行时对象
    -> set_active_task_graph()
    -> LLM 继续执行（未完成的继续，已完成的可修改）
```

---

## 实现方案

### Phase 1: 从 MemoryGraph 重建 TaskGraph

**文件**: `zulong/memory/graph_adapters.py`

**新增函数**: `rebuild_task_graph_from_memory(mg, graph_id) -> Optional[TaskGraph]`

核心逻辑：
1. 过滤 `mg` 中所有 `node_type == TASK` 且 `metadata["graph_id"] == graph_id` 的节点
2. 识别根节点（`sub_type == "task_root"` 或本地 ID 为 `req`）
3. 从 metadata 还原 TaskNode 字段（status, desc, label, result, task_domain, semantic_summary, analysis_content）
4. 从 HIERARCHY 边重建 h_edges
5. 从 DEPENDENCY 边重建 d_edges
6. 构建 `TaskGraph(title, graph_id)` 并填充
7. **降级兜底**：如果 MemoryGraph 中节点不足 2 个，尝试从 `data/graph_backups/{graph_id}.json` 加载

### Phase 2: IDE 意图检测增强

**文件**: `zulong/ide/ide_fc_runner.py` — `_detect_ide_intent()` 方法

新增规则（在现有规则之前）：
1. 正则检测用户输入中的 `@[label#address]` 引用
2. 调用 `mg.resolve_address(address)` 确认是 TASK 类型节点
3. 提取 `graph_id`，调用 Phase 1 的重建函数
4. 返回 `intent="resume"`，附带 `referenced_node_id` 供注意力窗口聚焦

### Phase 3: LLM 工具 `task_resume_by_address`

**文件**: `zulong/tools/task_tools.py`

新增 `TaskResumeByAddressTool(BaseTool)`:
- 参数：`address`（必需）, `focus_node_id`（可选）
- 逻辑：解析地址 -> 重建图谱 -> 设为活跃 -> 返回概览
- 降级顺序：MemoryGraph 重建 -> 磁盘备份 -> 挂起任务目录
- 注册到 `zulong/tools/tool_engine.py` 的工具列表中

### Phase 4: 已完成节点可修改（Revise Mode）

**文件**: `zulong/tools/task_tools.py`

新增 `TaskReviseNodeTool(BaseTool)`:
- 参数：`node_id`（必需）, `reason`（必需，说明修改原因）
- 逻辑：将 `completed` 节点状态改为 `in_progress`，记录修改原因到 metadata，`content_version++`
- 返回节点之前的 result 供参考

同时扩展 `TaskMarkStatusTool`：允许 `completed -> in_progress` 转换（标记为 revision）

---

## 关键文件

| 文件 | 修改内容 |
|------|---------|
| `zulong/memory/graph_adapters.py` | 新增 `rebuild_task_graph_from_memory()` |
| `zulong/ide/ide_fc_runner.py` | `_detect_ide_intent()` 增加 @引用检测 |
| `zulong/tools/task_tools.py` | 新增 `TaskResumeByAddressTool` + `TaskReviseNodeTool` |
| `zulong/tools/tool_engine.py` | 注册新工具 |

---

## 验证方法

1. **单元测试**：构造含多个 TASK 节点的 MemoryGraph，调用 `rebuild_task_graph_from_memory()`，验证重建的 TaskGraph 结构完整
2. **集成测试**：用户消息包含 `@[任务名#tg:xxx/task:o1]`，验证 FC Runner 自动激活图谱并以 resume 模式运行
3. **端到端**：从 Web 前端引用一个已完成任务的节点，发送消息，验证 LLM 能看到完整任务上下文并继续工作
4. **Revise 验证**：对已完成节点调用 `task_revise_node`，验证状态正确转换且历史结果可访问
