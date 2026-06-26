# 祖龙 IDE 写文件工具新增 edits 模式设计方案

> 状态：设计文档，暂不改业务代码。
> 对齐 TSD 1.7 §23（IDE 工具体系）。

---

## 1. 背景

### 1.1 问题

当前 `ide_write_file` 只有 `overwrite` 和 `append` 两种模式，LLM 写长文件时面临：

1. **JSON 截断**：文件内容在 tool_call JSON 中，受 `max_tokens` 限制，长内容 JSON 会被截断导致解析失败
2. **内容冗余**：LLM 做小修改（修 bug、加函数）也必须输出完整文件内容，浪费 token 和网络带宽
3. **用户体验差**：用户看不到增量的代码变更，只能看到"文件被覆写/追加"

### 1.2 Cline 现有能力（已审计）

Cline 插件（`VscodeExecutionBridge.ts`）已有完整的编辑器操作能力：

| 操作 | 方法 | 实现方式 |
|---|---|---|
| `write_to_file` | `writeFile()` (L222-246) | `fs.writeFile` 完整写入，走 `confirmFileChange` 审批 |
| `replace_in_file` | `replaceInFile()` (L267-287) | SEARCH/REPLACE 格式 diff，`applySearchReplaceDiff` |
| `create_directory` | `createDirectory()` (L248-261) | `fs.mkdir` |
| `delete_file` | `deleteFile()` (L314+) | `fs.unlink` |

`replace_in_file` 已经提供了 diff 编辑能力。**但 Python 工具层没有暴露这个功能。**

### 1.3 当前 Python→Cline 通信路径

```
ide_write_file.execute() (Python)
  → _run_async_request("ide:execute_tool", payload) (ide_bridge_tools.py:822)
    → ide_server.request_ide_action("ide_execute_tool", payload) (ide_server.py:843)
      → WebSocket → Cline executeTool(toolName, args)
        → switch(toolName) { case "write_to_file": ... } (VscodeExecutionBridge.ts:143-148)
```

---

## 2. 方案决策

### 2.1 是否直接改造 ide_write_file？
**否，使用"扩展 mode 参数"方式，而非替换。**

`ide_write_file` 的 `overwrite` 和 `append` 模式仍然需要，不可删除：

| 场景 | 必需模式 | 原因 |
|---|---|---|
| 创建新文件 | overwrite | 从头建文件，无法 diff |
| 追加内容 | append | 增量追加，适合日志/报告 |
| 创建目录 | create_directory | 非文件操作 |
| 修改现有文件 | edit（新增） | 精准编辑，不传输完整内容 |

### 2.2 edits vs diff 模式对比

| 维度 | SEARCH/REPLACE diff | Position-based edits |
|---|---|---|
| 格式 | `<<<<<<< SEARCH\n原文\n=======\n替换\n>>>>>>> REPLACE` | `[{op, line/range, text}]` |
| LLM 生成难度 | 中等（需精确匹配原文字符） | 简单（只需行号+内容） |
| 可靠性 | 差（空白/编码不一致→匹配失败） | 好（行号定位，容错性好） |
| Cline 现成实现 | ✅ `replaceInFile` | ❌ 需新增 |
| 实时编辑器更新 | ❌ 全文写入后 reload | ✅ WorkspaceEdit API |
| 多块编辑 | 支持（可串多块 SEARCH/REPLACE） | 支持（操作列表） |

**推荐**：先复用 Cline 现有的 `replace_in_file`（最小投入），同时设计 `edit` 模式协议（中长期优化）。

### 2.3 方案

**Phase 1（立即可做）**：在 Python 工具层新增 `ide_replace_file` 工具，透传 Cline 的 `replace_in_file` 能力。**零 Cline 端改动。**

**Phase 2（后续优化）**：新增 `ide_write_file(mode="edit")`，支持基于位置的操作列表，需要在 Cline 端新增 handler。

---

## 3. Phase 1 设计：ide_replace_file（复用 Cline replace_in_file）

### 3.1 新增工具

```python
# zulong/tools/ide_bridge_tools.py 新增

class IdeReplaceFileTool(BaseTool):
    """通过 SEARCH/REPLACE diff 替换文件内容（复用 Cline replace_in_file）。"""

    def __init__(self):
        super().__init__(name="ide_replace_file", category=ToolCategory.CODE)
        self.description = (
            "使用 SEARCH/REPLACE 格式差异替换文件中的指定内容。"
            "适合对已有文件做精准修改（修 bug、加函数、改配置），无需传输完整文件内容。"
            "diff 格式：第一块以 <<<<<<< SEARCH 开头，然后是原标题，"
            "======= 分隔，然后是替换内容，>>>>>>> REPLACE 结尾。"
            "可串联多块 diff。"
            "如果修改内容较多建议直接用 ide_write_file(mode='overwrite') 重写整个文件。"
        )
```

### 3.2 WebSocket 协议（无改动）

复用现有 `ide:execute_tool` 协议，Cline 已有 handler：

```json
// Python→Cline WebSocket 消息（现有协议，无需改动）
{
  "type": "ide_execute_tool",
  "tool_name": "replace_in_file",
  "arguments": {
    "path": "/workspace/src/main.py",
    "diff": "<<<<<<< SEARCH\ndef old_func():\n    pass\n=======\ndef old_func():\n    return 42\n>>>>>>> REPLACE"
  }
}
```

### 3.3 与 ide_write_file 的联动

`ide_replace_file` 适合 LLM 自主选择使用。当 LLM 调用 `ide_write_file` 且文件已存在时，可在 prompt 中引导优先使用 `ide_replace_file` 做轻量修改。

---

## 4. Phase 2 设计：ide_write_file edits 模式

### 4.1 新增 mode 值

`ide_write_file` 新增 `mode="edit"`，接受 `edits` 参数：

```python
# ide_write_file 参数扩展
{
    "mode": "edit",                              # 新增值
    "file_path": "/workspace/src/main.py",
    "edits": [                                    # 新增参数
        {"op": "insert",   "line": 5, "text": "import os\n"},
        {"op": "delete",   "line": 3, "count": 2},
        {"op": "replace",  "start_line": 10, "end_line": 12, "text": "def new_func():\n    pass\n"},
        {"op": "append",   "text": "\n# New section\n"}
    ]
}
```

### 4.2 编辑操作类型

| op | 必需参数 | 语义 |
|---|---|---|
| `insert` | `line`, `text` | 在第 `line` 行前插入文本 |
| `delete` | `line`, `count` | 从第 `line` 行开始删除 `count` 行 |
| `replace` | `start_line`, `end_line`, `text` | 替换第 `start_line` 到 `end_line`（含）的行为 `text` |
| `append` | `text` | 追加到文件末尾 |

### 4.3 WebSocket 协议

```json
// Python→Cline 新消息类型
{
  "type": "ide_execute_tool",
  "tool_name": "apply_edits",
  "arguments": {
    "path": "/workspace/src/main.py",
    "edits": [
      {"op": "insert", "line": 5, "text": "import os\n"},
      {"op": "delete", "line": 3, "count": 2}
    ]
  }
}
```

### 4.4 Cline 端实现（伪代码）

```typescript
// VscodeExecutionBridge.ts 新增
case "apply_edits":
    return this.applyEdits(args)

private async applyEdits(args: Record<string, any>): Promise<string> {
    const filePath = this.resolvePath(args.path)
    this.ensureInsideWorkspace(filePath)
    const original = await fs.readFile(filePath, "utf-8")
    const lines = original.split("\n")
    const edits: EditOp[] = args.edits || []

    for (const edit of edits) {
        switch (edit.op) {
            case "insert":  lines.splice(edit.line - 1, 0, ...edit.text.split("\n")); break
            case "delete":  lines.splice(edit.line - 1, edit.count); break
            case "replace": lines.splice(edit.start_line - 1, edit.end_line - edit.start_line + 1, ...edit.text.split("\n")); break
            case "append":  lines.push(...edit.text.split("\n")); break
        }
    }
    const next = lines.join("\n")

    const approved = await this.confirmFileChange({
        filePath, original, next,
        operation: "modify",
        summary: `应用 ${edits.length} 个编辑操作`,
    })
    if (!approved) return `用户未应用编辑: ${filePath}`

    await fs.writeFile(filePath, next, "utf-8")
    this.sendFileChanged(filePath, "edited")
    return `已应用 ${edits.length} 个编辑操作: ${filePath}`
}
```

### 4.5 为什么 edits 优于 diff

| 场景 | diff（SEARCH/REPLACE） | edits（行操作） |
|---|---|---|
| 换空白字符 | ❌ 匹配失败 | ✅ 不受影响 |
| 文件已被人修改 | ❌ search 文本不匹配 | ✅ 行号可能偏移但容错 |
| LLM 生成难度 | 高（需输出原文字符） | 低（只需行号 + 新内容） |
| 多块编辑 | 支持 | 支持 |
| 传输量 | 含原文字符，较大 | 只含新内容，小 |

---

## 5. Phase 2 备选方案：直接用 VS Code WorkspaceEdit API

Cline 插件可以直接调 `vscode.WorkspaceEdit`，在编辑器里实时应用修改，不需要 `fs.readFile`+`fs.writeFile`（当前 `writeFile` 和 `replaceInFile` 的实现方式）。

```typescript
// 更优实现：直接用 WorkspaceEdit（免 fs 读写）
const uri = vscode.Uri.file(filePath)
const edit = new vscode.WorkspaceEdit()
for (const op of edits) {
    const pos = new vscode.Position(op.line - 1, 0)
    switch (op.op) {
        case "insert":  edit.insert(uri, pos, op.text); break
        case "delete":  edit.delete(uri, new vscode.Range(pos, pos.translate(op.count, 0))); break
        case "replace": edit.replace(uri, new vscode.Range(
            new vscode.Position(op.start_line - 1, 0),
            new vscode.Position(op.end_line - 1, Number.MAX_SAFE_INTEGER)
        ), op.text); break
    }
}
await vscode.workspace.applyEdit(edit)
// → 编辑器实时更新，合并到 undo 栈，用户看得见光标变化
```

**优势**：
- 编辑器内实时更新，光标跟随
- 修改进 VS Code 的 undo stack（Ctrl+Z 可撤）
- 不触发文件系统 IO，更快

**当前不推荐推进**：因为 Cline 的 `confirmFileChange` 审批流程依赖"原内容 vs 新内容"的 diff 对比，`WorkspaceEdit` 直接操作编辑器缓冲区不走这个审批流程，需要重构审批机制。**后续可做，当前先走 `fs.readFile`+`fs.writeFile` 方式。**

---

## 6. 实施路径

| 阶段 | 内容 | 改动文件 | 工作量 |
|---|---|---|---|
| **Phase 1** | 新增 `ide_replace_file` 工具（复用 Cline `replace_in_file`） | `ide_bridge_tools.py`（新增 1 个类） | ~50 行 Python |
| **Phase 1** | 工具注册 + LLM prompt 提示 | `tool_registry.py`（注册新工具） | ~5 行 |
| **Phase 2** | `ide_write_file` 新增 `mode="edit"` + `edits` 参数 | `ide_bridge_tools.py`（扩展 IdeWriteFileTool.execute） | ~80 行 Python |
| **Phase 2** | Cline 新增 `apply_edits` handler | `VscodeExecutionBridge.ts`（新增 case） | ~60 行 TS |
| Phase 2（可选） | 切到 `WorkspaceEdit` API | `VscodeExecutionBridge.ts` | 需重构审批流 |

---

## 7. LLM Prompt 引导

在 system prompt 中新增：

```
## 文件编辑策略

- 创建新文件：使用 ide_write_file(mode="overwrite", content="完整内容")
- 追加内容：使用 ide_write_file(mode="append", content="追加内容")
- 修改已有文件（轻量）：优先使用 ide_replace_file(diff="SEARCH/REPLACE")
- 修改已有文件（多处修改）：使用 ide_write_file(mode="edit", edits=[...])
- 完全重写文件：使用 ide_write_file(mode="overwrite")
```

---

## 8. 与现有 ide_write_file 的兼容性

- `overwrite` 和 `append` 模式行为不变，向后兼容
- `mode="edit"` 是新增值，老代码不受影响
- `ide_replace_file` 是完全独立的新工具
- 所有现有审批流程（`confirmFileChange`、`require_folder_access_authorization`、路径白名单）保持不变

---

## 9. 后续工作

- Phase 1 实施后，`_try_repair_truncated_json`（之前加的截断修复）可以降级为兜底——因为 LLM 用 `ide_replace_file` 时不再输出长 content，JSON 截断概率大幅降低
- 长期考虑在 Cline 端用 `vscode.WorkspaceEdit` 实现编辑，给用户实时编辑体验
- edits 模式稳定后，考虑自动降级：当 LLM 的 `ide_write_file` 内容超 2000 字符时，工具端自动拒绝并建议切换 edits 模式
