# 祖龙 LLM 对 VS Code 完整控制体系设计方案

> **对齐 TSD v2.7 第 23 章：工具袋 + L1-B 预判 + L2 自主补充**

---

## 一、背景与目标

### 1.1 现状

祖龙的 LLM 目前仅能通过 `VscodeExecutionBridge` 访问 VS Code 的 **~30%** 能力：

| 能力域 | 暴露度 | 详情 |
|--------|--------|------|
| 文件 I/O | 90% | `read_file`/`write_to_file`/`replace_in_file`/`delete_file` ✅ |
| 终端执行 | 100% | `execute_command` ✅ |
| 代码分析 | 30% | 仅有正则扫描 `list_code_definition_names`，无 linter 诊断 |
| VS Code 命令 | **0%** | LLM 无法调用 `commands.executeCommand` |
| 用户交互 | 20% | `ask_followup_question` 实际是 no-op |
| 设置/插件 | **0%** | 无法管理扩展、修改配置 |

核心缺陷：LLM 只能读写文件和执行终端命令，无法利用 VS Code 的 **200+ 内置命令、扩展系统、诊断管道、Git 集成** 等完整能力。

### 1.2 目标

让 LLM 拥有对 VS Code 的**完整程序化控制能力**：
- 代码编写 → 格式化、组织导入、重构
- 依赖安装 → 通过终端执行 `npm install`/`pip install`
- 插件扩展 → 安装/禁用/配置 VS Code 扩展
- MCP 配置 → 通过 VS Code 的 MCP 管理能力
- 诊断检查 → 实时获取 linter/编译器错误
- 用户交互 → 输入框、文件选择器、进度提示

### 1.3 TSD 约束

**必须严格遵循 TSD v2.7 第 23 章的工具袋三层控制体系**：

```
L1-B 工具预判 (关键词+规则) → 建议工具集合
    ↓
工具袋名单 (TOOL_BAG_FULL，扁平全量清单，17 → 扩充)
    ↓
L2 LLM 从工具包中选择 + 自主补充 (request_tool_supplement)
    ↓
_intent_prompt_builder 意图工具白名单 (CHAT/RESUME/COMPLEX)
    ↓
_collect_named_tool_definitions() 物理过滤
```

关键设计原则：
- **工具袋不分类传入 LLM** — 扁平全量清单，每工具一行描述
- **risk 字段不传给 LLM** — 仅用于前端审批逻辑
- **L1-B 只是预判建议** — L2 可自主补充
- **request_tool_supplement 常驻** — 安全阀，确保 L2 永远不会被锁死

---

## 二、新增工具清单及工具袋注册

### 2.1 新增工具总览

| 工具名 | 类别 | 风险 | 执行器 | 需审批 | 来源 |
|--------|------|------|--------|--------|------|
| `vscode_run_command` | vscode_command | HIGH | vscode_bridge | 是 | **新增（核心）** |
| `get_diagnostics` | vscode_diagnostic | LOW | vscode_bridge | 否 | HostBridge 已有 |
| `ask_user_input` | vscode_interaction | LOW | vscode_bridge | 否 | HostBridge 已有 |
| `ask_user_select_file` | vscode_interaction | LOW | vscode_bridge | 否 | HostBridge 已有 |
| `vscode_manage_extension` | vscode_extension | HIGH | vscode_bridge | 是 | **新增** |
| `open_settings` | vscode_ui | LOW | vscode_bridge | 否 | HostBridge 已有 |
| `open_problems` | vscode_ui | LOW | vscode_bridge | 否 | HostBridge 已有 |

### 2.2 工具袋名单扩充 (`l1b/tool_predictor.py` → `TOOL_BAG_FULL`)

在现有 17 个工具的基础上追加以下条目：

```python
# ===== VS Code 命令与诊断 =====
{
    "name": "vscode_run_command",
    "desc": "执行 VS Code 内置/扩展命令（格式化、重构、Git 操作等）",
    "risk": "HIGH",
},
{
    "name": "get_diagnostics",
    "desc": "获取工作区所有文件 linter/编译器诊断（Error/Warning/Info/Hint）",
    "risk": "LOW",
},
# ===== VS Code 用户交互 =====
{
    "name": "ask_user_input",
    "desc": "弹出 VS Code 输入框向用户提问",
    "risk": "LOW",
},
{
    "name": "ask_user_select_file",
    "desc": "弹出系统文件选择对话框，让用户选择文件/文件夹",
    "risk": "LOW",
},
# ===== VS Code 扩展与设置 =====
{
    "name": "vscode_manage_extension",
    "desc": "安装/卸载/启用/禁用 VS Code 扩展",
    "risk": "HIGH",
},
{
    "name": "open_settings",
    "desc": "打开 VS Code 设置面板",
    "risk": "LOW",
},
{
    "name": "open_problems",
    "desc": "打开 VS Code 问题面板（展示诊断结果）",
    "risk": "LOW",
},
```

### 2.3 工具集名称扩充 (`tool_bag.py`)

```python
# ===== 新增：VS Code 命令工具集 =====
VSCODE_COMMAND_TOOL_NAMES = {
    "vscode_run_command",
}

VSCODE_DIAGNOSTIC_TOOL_NAMES = {
    "get_diagnostics",
}

VSCODE_INTERACTION_TOOL_NAMES = {
    "ask_user_input",
    "ask_user_select_file",
}

VSCODE_EXTENSION_TOOL_NAMES = {
    "vscode_manage_extension",
}

VSCODE_UI_TOOL_NAMES = {
    "open_settings",
    "open_problems",
}

# ===== 扩充：IDE 桥接工具集 =====
IDE_TOOL_NAMES = {
    "ide_open_workspace",
    "ide_write_file",
    # 新增 VS Code 桥接工具
    "vscode_run_command",
    "get_diagnostics",
    "ask_user_input",
    "ask_user_select_file",
    "vscode_manage_extension",
    "open_settings",
    "open_problems",
}
```

### 2.4 分类/风险/执行器映射更新 (`tool_bag.py`)

```python
def _category_for(name: str, category: ToolCategory) -> str:
    # ... 现有映射保持不变 ...
    if name in VSCODE_COMMAND_TOOL_NAMES:
        return "vscode_command"
    if name in VSCODE_DIAGNOSTIC_TOOL_NAMES:
        return "vscode_diagnostic"
    if name in VSCODE_INTERACTION_TOOL_NAMES:
        return "vscode_interaction"
    if name in VSCODE_EXTENSION_TOOL_NAMES:
        return "vscode_extension"
    if name in VSCODE_UI_TOOL_NAMES:
        return "vscode_ui"
    # ... 其余保持不变 ...

def _risk_for(name: str, category: ToolCategory) -> str:
    # ... 现有映射保持不变 ...
    if name in VSCODE_COMMAND_TOOL_NAMES or name in VSCODE_EXTENSION_TOOL_NAMES:
        return "high"
    if name in VSCODE_INTERACTION_TOOL_NAMES or name in VSCODE_DIAGNOSTIC_TOOL_NAMES or name in VSCODE_UI_TOOL_NAMES:
        return "low"
    # ... 其余保持不变 ...

def _executor_for(name: str, category: ToolCategory) -> str:
    # ... 现有映射 ...
    if name in IDE_TOOL_NAMES:  # IDE_TOOL_NAMES 已经扩充
        return "vscode_bridge"
    # ... 其余保持不变 ...

def _examples_for(name: str, category: str) -> List[str]:
    examples = {
        # ... 现有映射保持不变 ...
        "vscode_command": ["格式化代码", "整理导入", "运行测试任务", "Git 提交"],
        "vscode_diagnostic": ["检查 lint 错误", "查看编译警告", "获取所有文件诊断"],
        "vscode_interaction": ["弹出输入框收集信息", "让用户选择文件路径"],
        "vscode_extension": ["安装扩展", "卸载扩展", "查看已安装扩展"],
        "vscode_ui": ["打开设置面板", "打开问题面板"],
    }
    return examples.get(category, [f"使用 {name} 完成相关操作"])
```

---

## 三、L1-B 工具预判扩充

### 3.1 关键词映射更新 (`l1b/tool_predictor.py` → `L1BToolPredictor.KEYWORDS_MAP`)

```python
KEYWORDS_MAP: Dict[str, List[str]] = {
    # ... 现有映射保持不变 ...
    "写代码|修改|改|创建|新建": [
        "read_file", "search_files", "write_to_file", "replace_in_file",
    ],
    "查|搜|找|在哪|搜索": [
        "search_files", "search_knowledge", "web_search", "read_file",
    ],
    "运行|执行|跑|测试": [
        "execute_command", "read_file",
    ],
    "回忆|之前|上次|记得": [
        "recall_memory", "read_memory_node", "discover_related",
    ],
    "知识|关系|谁|什么|哪里": [
        "search_knowledge", "discover_related",
    ],
    # ===== 新增：VS Code 相关预判 =====
    "格式化|格式化代码|整理导入|format|lint|prettier": [
        "vscode_run_command", "read_file",
    ],
    "重构|重命名|改名|rename": [
        "vscode_run_command", "read_file", "replace_in_file",
    ],
    "git|提交|commit|推送|push|分支|branch|merge": [
        "vscode_run_command", "execute_command",
    ],
    "扩展|插件|extension|install extension|卸载扩展": [
        "vscode_manage_extension",
    ],
    "设置|setting|配置|preference": [
        "open_settings", "read_file",
    ],
    "错误|报错|error|warning|诊断|diagnostic|lint错误": [
        "get_diagnostics", "open_problems", "read_file",
    ],
    "选择文件|打开文件|browse|选择目录": [
        "ask_user_select_file", "list_files",
    ],
}
```

### 3.2 预判逻辑更新 (`tool_bag.py` → `predict_tools_for_turn()`)

在 `_needs_write(lower)` 块之后添加：

```python
# ===== 新增：VS Code 命令预判 =====
if _needs_vscode_command(lower):
    add(["vscode_run_command"], "用户要求执行 VS Code 命令（格式化/重构/Git/测试运行）")
    context_bundle["needs_vscode_command"] = True
    risk_notes.append("vscode_run_command 可执行任意 VS Code 命令，高风险命令需审批。")

# ===== 新增：诊断预判 =====
if _needs_diagnostics(lower):
    add(["get_diagnostics", "open_problems"], "用户询问代码错误或需要检查 lint/编译状态")
    context_bundle["needs_diagnostics"] = True

# ===== 新增：扩展管理预判 =====
if _needs_extension_management(lower):
    add(["vscode_manage_extension"], "用户要求管理 VS Code 扩展（安装/卸载/查看）")
    risk_notes.append("扩展安装/卸载涉及系统权限，需用户审批。")
```

### 3.3 新增需求检测函数 (`tool_bag.py`)

```python
def _needs_vscode_command(text: str) -> bool:
    """检测是否需要 VS Code 命令能力"""
    indicators = [
        "格式化", "format", "整理导入", "organize", "prettier", "lint",
        "重构", "重命名", "rename", "跳转", "go to definition",
        "git", "提交", "commit", "推送", "push", "拉取", "pull",
        "分支", "branch", "merge", "合并", "暂存", "stage",
        "测试任务", "test task", "build task", "构建任务",
        "运行测试", "run test", "调试", "debug",
        "快捷", "command palette", "命令面板",
    ]
    lowered = text.lower()
    return any(indicator.lower() in lowered for indicator in indicators)

def _needs_diagnostics(text: str) -> bool:
    """检测是否需要诊断信息"""
    indicators = [
        "错误", "报错", "error", "warning", "警告",
        "诊断", "diagnostic", "lint", "类型错误", "type error",
        "编译", "compile", "问题", "problem", "issue",
    ]
    lowered = text.lower()
    return any(indicator.lower() in lowered for indicator in indicators)

def _needs_extension_management(text: str) -> bool:
    """检测是否需要扩展管理"""
    indicators = [
        "扩展", "extension", "插件", "plugin",
        "安装扩展", "install extension", "卸载扩展", "uninstall",
        "禁用", "disable", "启用", "enable",
    ]
    lowered = text.lower()
    return any(indicator.lower() in lowered for indicator in indicators)
```

### 3.4 优先级排序更新 (`tool_bag.py` → `_finalize_prediction()`)

```python
priority = {
    "request_tool_supplement": 0,
    "search_tools": 1,
    "web_search": 2,
    "ide_write_file": 3,
    "exec_write_file": 3,
    "exec_run_command": 4,
    "vscode_run_command": 5,      # 新增
    "get_diagnostics": 6,          # 新增
    "ask_user_input": 7,           # 新增
    "ask_user_select_file": 8,     # 新增
    "vscode_manage_extension": 9,  # 新增
    "open_settings": 10,           # 新增
    "open_problems": 10,           # 新增
    "task_view_overview": 15,
    "task_create_plan": 16,
    "task_mark_status": 17,
}
```

### 3.5 `summarize_tool_bundle()` 更新

```python
def summarize_tool_bundle(prediction: Dict[str, Any], *, limit: int = 1600) -> str:
    # ... 现有代码 ...
    if ctx.get("needs_vscode_command"):
        lines.append(
            "- VS Code 命令提示: 可调用 vscode_run_command 执行格式化、重构、Git 等。"
        )
    if ctx.get("needs_diagnostics"):
        lines.append(
            "- 诊断提示: 调用 get_diagnostics 检查代码错误，open_problems 打开问题面板。"
        )
    # ... 其余保持不变 ...
```

---

## 四、工具 Schema 定义

### 4.1 `vscode_run_command`（核心工具）

```json
{
    "name": "vscode_run_command",
    "description": "执行 VS Code 内置或扩展命令。可用于格式化代码、重构、运行测试、Git 操作、打开面板等。部分高风险命令需要用户审批。常用命令: editor.action.formatDocument, editor.action.organizeImports, workbench.action.tasks.build, git.stage 等。",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "VS Code 命令 ID"
            },
            "args": {
                "type": "array",
                "description": "命令参数（可选）",
                "items": { "type": "string" }
            }
        },
        "required": ["command"]
    }
}
```

### 4.2 `get_diagnostics`

```json
{
    "name": "get_diagnostics",
    "description": "获取工作区文件的 linter/编译器诊断信息（Error/Warning/Info/Hint）。修改代码后用于检查是否引入错误。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件绝对路径（可选，不传返回所有文件）"
            }
        },
        "required": []
    }
}
```

### 4.3 `ask_user_input`

```json
{
    "name": "ask_user_input",
    "description": "弹出 VS Code 输入框向用户询问信息。",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": { "type": "string", "description": "提示文字" },
            "placeholder": { "type": "string", "description": "占位文字（可选）" },
            "default_value": { "type": "string", "description": "默认值（可选）" }
        },
        "required": ["prompt"]
    }
}
```

### 4.4 `ask_user_select_file`

```json
{
    "name": "ask_user_select_file",
    "description": "弹出系统文件/文件夹选择对话框，让用户选择路径。",
    "parameters": {
        "type": "object",
        "properties": {
            "title": { "type": "string", "description": "对话框标题" },
            "type": {
                "type": "string",
                "enum": ["file", "folder"],
                "description": "选择文件还是文件夹"
            }
        },
        "required": ["title", "type"]
    }
}
```

### 4.5 `vscode_manage_extension`

```json
{
    "name": "vscode_manage_extension",
    "description": "管理 VS Code 扩展（安装/卸载/启用/禁用/查询列表）。高风险操作需要用户审批。",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["install", "uninstall", "enable", "disable", "list"],
                "description": "操作类型"
            },
            "extension_id": {
                "type": "string",
                "description": "扩展 ID（install/uninstall/enable/disable 时必填）"
            }
        },
        "required": ["action"]
    }
}
```

### 4.6 `open_settings`

```json
{
    "name": "open_settings",
    "description": "打开 VS Code 设置面板。",
    "parameters": {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "enum": ["user", "workspace"],
                "description": "用户设置或工作区设置"
            }
        },
        "required": []
    }
}
```

### 4.7 `open_problems`

```json
{
    "name": "open_problems",
    "description": "打开 VS Code 问题面板，展示诊断结果。",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}
```

---

## 五、IDE 工具注册表更新

### 5.1 `IDE_REMOTE_TOOLS` 扩充 (`ide_tool_registry.py`)

```python
IDE_REMOTE_TOOLS = {
    # ... 现有 11 个工具 ...
    "read_file",
    "write_to_file",
    "replace_in_file",
    "delete_file",
    "execute_command",
    "search_files",
    "list_files",
    "list_code_definition_names",
    "browser_action",
    "ask_followup_question",
    "attempt_completion",
    # ===== 新增 =====
    "vscode_run_command",
    "get_diagnostics",
    "ask_user_input",
    "ask_user_select_file",
    "vscode_manage_extension",
    "open_settings",
    "open_problems",
    # ===== 修复 =====
    "create_directory",  # TS 侧已实现，仅缺注册
}
```

### 5.2 同时修复 `_IDE_TOOL_SCHEMAS`

添加上述 8 个工具（含 `create_directory`）的 OpenAI FC schema。

---

## 六、`vscode_run_command` 安全设计

### 6.1 安全分层

严格按照 TSD 的"risk 字段不传给 LLM"原则，在 TypeScript 侧实现：

```
LLM 调用 vscode_run_command(command="editor.action.formatDocument")
    ↓
TypeScript classifyCommand(command):
    ├── safe ──────► 直接执行 ✅
    ├── high_risk ──► showWarningMessage → 用户确认 → 执行/拒绝
    └── blocked ────► throw Error ❌
```

### 6.2 命令分级

#### safe（免审批）

`editor.action.formatDocument`, `editor.action.organizeImports`, `editor.action.goToDefinition`, `editor.action.peekDefinition`, `editor.action.referenceSearch.trigger`, `editor.action.rename`, `workbench.action.problems.focus`, `workbench.action.output.focus`, `workbench.action.terminal.focus`, `workbench.action.tasks.build`, `workbench.action.tasks.test`, `vscode.executeDocumentSymbolProvider`, `vscode.executeCompletionItemProvider`, `vscode.executeHoverProvider`, `vscode.executeDefinitionProvider`

#### high_risk（需审批）

`git.stage`, `git.stageAll`, `git.commit`, `git.commitAll`, `git.push`, `git.pull`, `git.clean`, `editor.action.clipboardCutAction`, `deleteFile`, `workbench.files.action.deleteFile`

#### blocked（硬阻止）

`workbench.action.reloadWindow`, `workbench.action.closeWindow`, `workbench.action.quit`, `workbench.extensions.installExtension`, `workbench.extensions.uninstallExtension`

### 6.3 命令策略配置文件 (`config/vscode_command_policy.yaml`)

```yaml
vscode_command_policy:
  safe:
    - "editor.action.formatDocument"
    - "editor.action.organizeImports"
    - "workbench.action.tasks.build"
    - "workbench.action.tasks.test"
    - "vscode.executeDocumentSymbolProvider"
    - "vscode.executeCompletionItemProvider"
    - "vscode.executeHoverProvider"
    - "vscode.executeDefinitionProvider"
    # ... 更多

  high_risk:
    - "git.stage"
    - "git.commit"
    - "git.push"

  blocked:
    - "workbench.action.reloadWindow"
    - "workbench.action.closeWindow"
    - "workbench.action.quit"
```

---

## 七、实现阶段（对齐工具袋规范）

### Phase 1: 工具袋注册 + vscode_run_command

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 1.1 | `zulong/tools/tool_bag.py` | 新增 5 个工具集 + 映射函数 + 预判逻辑 + 需求检测函数 | ~100 行 |
| 1.2 | `zulong/l1b/tool_predictor.py` | `TOOL_BAG_FULL` 追加 7 条目 + `KEYWORDS_MAP` 新增 7 组映射 | ~60 行 |
| 1.3 | `zulong/ide/ide_tool_registry.py` | `IDE_REMOTE_TOOLS` +8 + `_IDE_TOOL_SCHEMAS` +8 | ~120 行 |
| 1.4 | `zulong-ide/src/hosts/vscode/VscodeExecutionBridge.ts` | `executeTool()` switch +8 case + `vscodeRunCommand()` + 安全分级 | ~100 行 |
| 1.5 | `config/vscode_command_policy.yaml` | 新建安全策略配置 | ~30 行 |

**Phase 1 合计**: ~410 行，5 个文件

### Phase 2: HostBridge 工具实现

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 2.1 | `VscodeExecutionBridge.ts` | `getDiagnostics()` → 调已有 `getDiagnostics()` 函数 | ~10 行 |
| 2.2 | `VscodeExecutionBridge.ts` | `askUserInput()` → `vscode.window.showInputBox()` | ~10 行 |
| 2.3 | `VscodeExecutionBridge.ts` | `askUserSelectFile()` → `vscode.window.showOpenDialog()` | ~10 行 |
| 2.4 | `VscodeExecutionBridge.ts` | `vscodeManageExtension()` → 扩展 API | ~50 行 |
| 2.5 | `VscodeExecutionBridge.ts` | `openSettings()` / `openProblems()` → HostBridge | ~15 行 |

**Phase 2 合计**: ~95 行，1 个文件

### Phase 3: create_directory 修复 + 系统提示更新

| # | 文件 | 改动 | 说明 |
|---|------|------|------|
| 3.1 | `ide_tool_registry.py` | 注册 `create_directory` schema | ~10 行 |
| 3.2 | `ide_prompt_handler.py` | 更新工具使用说明 | ~20 行 |

**Phase 3 合计**: ~30 行，2 个文件

### 总改动量

| Phase | 文件数 | 行数 |
|-------|--------|------|
| 1 | 5 | ~410 |
| 2 | 1 | ~95 |
| 3 | 2 | ~30 |
| **合计** | **6 个文件（2 个共用）** | **~535** |

---

## 八、与 TSD 各条款的对齐清单

| TSD 要求 | 对齐方式 |
|----------|---------|
| 工具袋扁平全量清单 | `TOOL_BAG_FULL` 追加 7 个新工具，保持扁平结构 |
| risk 不传 LLM | 仅在 `_risk_for()` 中使用，用于前端审批，tool_bag 传给 LLM 的 describe 不含 risk |
| L1-B 关键词+规则预判 | `KEYWORDS_MAP` 新增 7 组映射 + `predict_tools_for_turn()` 新增 3 个检测函数 |
| L2 自主补充 | `request_tool_supplement` 始终可用，支持 `list_all_tools=True` 查看全量 |
| 三级工具暴露控制 | 新工具注册 → 意图白名单 (COMPLEX=None 全放) → 物理过滤 |
| 5W 透明汇报 | 新工具的 thought 字段复用现有框架，每次调用附带选择原因 |
| 最大化权限 + 超高可控性 | 允许 L2 通过 `vscode_run_command` 调用任何 VS Code 命令，但 blocked 命令硬阻止 |
