# Code Anchor (代码锚点) 功能实现方案

## Context

用户提出 Web 监控端缺少对话历史、实时节点消息推送、以及任务节点建立过程可视化。核心需求是实现规划文档 Section 13 中描述的 **Code Anchor (代码锚点)** 功能 —— 在记忆/任务节点与具体代码位置之间建立双向关联，使系统能够回答：
- "这段代码为什么这样写？"（代码 -> 记忆/任务）
- "这个记忆/任务涉及哪些代码？"（记忆/任务 -> 代码）

## 实现方案概述

新增 `CodeAnchor` 数据模型和 `CodeAnchorStore` 存储层，3 个新 FC 工具，并在 Web 前端展示锚点信息。

---

## 步骤 1：创建 `zulong/memory/code_anchor.py`（新文件）

**核心数据结构：**

```python
@dataclass
class CodeAnchor:
    id: str                    # UUID
    file_path: str             # 相对项目根路径
    symbol: Optional[str]      # 函数/类/变量名（最稳定标识符）
    line_start: Optional[int]  
    line_end: Optional[int]    
    commit_sha: Optional[str]  # 关联时的 commit
    content_hash: Optional[str] # 代码片段 hash
    anchor_type: str           # implementation / affected / created / deleted
    snippet_preview: str       # 代码预览（前2-3行）
    owner_ref: str             # "mg:{node_id}" 或 "tg:{graph_id}/{task_node_id}"
    created_at: float          # timestamp
```

**CodeAnchorStore（单例）：**
- `_anchors: Dict[str, CodeAnchor]` — anchor_id -> CodeAnchor
- `_file_index: Dict[str, Set[str]]` — file_path -> anchor_ids（反向索引）
- `_owner_index: Dict[str, Set[str]]` — owner_ref -> anchor_ids（反向索引）
- 持久化到 `data/memory_graph/code_anchors.json`
- `_pending_changes: List[Dict]` 用于 delta 广播
- 方法：`add_anchor()`, `remove_anchor()`, `get_anchors_by_file()`, `get_anchors_by_owner()`, `save()`, `_load()`
- 使用 `threading.Lock()` 保证线程安全
- `get_code_anchor_store()` 工厂函数（懒加载单例）

---

## 步骤 2：创建 `zulong/tools/code_anchor_tools.py`（新文件）

### 工具 1: `zulong_memory_write_with_code`

保存记忆并关联代码位置（原子操作）。

参数：
- `content` (string, required) — 记忆内容
- `label` (string, optional) — 短标签
- `importance` (string, optional) — 重要度
- `code_refs` (array, required) — 每项含 file_path, symbol?, line_start?, line_end?, anchor_type?, snippet_preview?

执行逻辑：
1. 创建 MemoryGraph KNOWLEDGE 节点
2. 为每个 code_ref 创建 CodeAnchor（owner_ref = `"mg:{node_id}"`）
3. 在 node.metadata 中存储 `code_anchors` (anchor_id 列表) 和 `code_ref_summary` (一行摘要)

### 工具 2: `zulong_code_query`

查询某段代码相关的记忆、任务和经验。

参数：
- `file_path` (string, required)
- `symbol` (string, optional)
- `line_range` (object, optional) — `{start, end}`

执行逻辑：
1. 通过 `CodeAnchorStore.get_anchors_by_file(file_path)` 获取文件锚点
2. 按 symbol / line_range 过滤
3. 解析 owner_ref -> 加载关联的 MemoryGraph 节点 / TaskGraph 节点
4. 返回结构化结果（memories, tasks, experiences）

### 工具 3: `zulong_task_link_code`

将任务节点关联到实现代码。

参数：
- `task_node_id` (string, required)
- `code_refs` (array, required) — 同上格式

执行逻辑：
1. 查找 TaskGraph 中的节点
2. 创建 CodeAnchor（owner_ref = `"tg:{graph_id}/{task_node_id}"`）
3. 存储 anchor_id 到 `TaskGraph.metadata["code_anchors"][task_node_id]`
4. 同步到 MemoryGraph 对应的 TASK 节点
5. 触发 on_change_callback 广播

---

## 步骤 3：修改 `zulong/tools/tool_engine.py`

在 `_register_builtin_tools()` 中添加注册块：

```python
try:
    from zulong.tools.code_anchor_tools import (
        MemoryWriteWithCodeTool, CodeQueryTool, TaskLinkCodeTool,
    )
    for tool_cls in [MemoryWriteWithCodeTool, CodeQueryTool, TaskLinkCodeTool]:
        tool_inst = tool_cls()
        if self.register_tool(tool_inst):
            logger.info(f"[ToolEngine] 注册 {tool_inst.name}")
except ImportError:
    logger.debug("[ToolEngine] code_anchor_tools 模块未找到")
```

---

## 步骤 4：修改 `zulong/tools/memory_graph_tools.py`

- `RecallMemoryTool.execute()`：结果项中添加 `"code_ref": metadata.get("code_ref_summary", "")`
- `ReadMemoryNodeTool.execute()`：如有 `code_anchors` metadata，加载完整锚点数据返回

---

## 步骤 5：修改 `zulong/cline/cline_ide_server.py`

添加 `_inject_code_anchor_monitor_hook()`：
- 与 `_inject_memory_graph_monitor_hook()` 同模式
- 钩入 `CodeAnchorStore._mark_dirty()` 广播 `CODE_ANCHOR_UPDATE` 事件
- 在 `_run_fc_loop()` 中调用（紧跟 memory graph hook 之后）
- WELCOME 消息中添加 code_anchor_stats

---

## 步骤 6：修改 `openclaw_bridge/web/static/index.html`

### 6.1 WebSocket 事件处理
在 `handleMessage()` switch 中添加 `CODE_ANCHOR_UPDATE` case，维护 `codeAnchorCache` (Map)。

### 6.2 Task Graph 详情面板
`renderGraphDetailView()` 中，"关联文件" 区域之后添加 "代码锚点" 区域：
- 图标 + 锚点类型 badge
- Symbol 名称（粗体）+ 文件路径
- 行范围标签 `L42-87`
- snippet_preview monospace 代码框
- 点击跳转 `openFileInIDE()`

### 6.3 Memory Graph 详情面板
`showMgNodeDetail()` 中为含 `code_anchors` 的节点添加同样的锚点展示区域。

### 6.4 SVG 节点标记
Task Graph 节点有 code_anchors 时显示 `📌` badge。

---

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| CodeAnchor 存储位置 | 独立文件 `code_anchors.json` | 不影响现有 memory_graph.json 体积 |
| GraphNode 集成方式 | `metadata["code_anchors"]` 存 ID 列表 | 零 schema 变更，完全向后兼容 |
| TaskNode 集成方式 | `TaskGraph.metadata["code_anchors"]` 字典 | 不修改 TaskNode dataclass |
| 普通检索返回格式 | 仅 `code_ref_summary` 一行 | 控制 LLM token 消耗（+12%~20%） |
| 完整锚点返回 | 仅 `zulong_code_query` 工具 | 按需加载，避免上下文膨胀 |

## 向后兼容性

- GraphNode.metadata 是 `Dict[str, Any]`，新增 key 不影响旧数据
- TaskGraph.metadata 同理
- `code_anchors.json` 不存在时 store 启动为空
- 所有工具对空 store 优雅降级（返回空结果）

## 验证方法

1. **单元测试**：为 CodeAnchorStore 编写 CRUD + 持久化测试
2. **集成测试**：通过 FC loop 调用 3 个新工具，验证锚点创建和查询
3. **Web 前端**：启动 IDE server，连接 monitor WebSocket，验证 CODE_ANCHOR_UPDATE 事件和前端渲染
4. **向后兼容**：加载旧数据（无 code_anchors 字段），确认系统正常运行

## 涉及文件清单

| 文件 | 操作 |
|------|------|
| `zulong/memory/code_anchor.py` | 新建 |
| `zulong/tools/code_anchor_tools.py` | 新建 |
| `zulong/tools/tool_engine.py` | 修改（添加注册） |
| `zulong/tools/memory_graph_tools.py` | 修改（recall_memory + read_memory_node） |
| `zulong/cline/cline_ide_server.py` | 修改（hook + broadcast） |
| `openclaw_bridge/web/static/index.html` | 修改（前端展示） |
