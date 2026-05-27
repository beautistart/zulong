# 祖龙系统架构变更总结 (v2.9-草案)
# 基于实际代码与文档对比，供 TSD 更新审核

> 生成日期: 2026-05-26
> 基准 TSD: v2.8 (2026-05-24)
> 范围: 自 TSD v2.8 以来所有已实现但未并入 TSD 的代码变更

---

## 一、变更总览

| 类别 | 变更项 | 影响文件数 | 新增代码行 |
|------|--------|-----------|-----------|
| **A. VS Code 完整控制** | 8 个新工具 + 安全策略 + 执行桥 | 4 | ~1,500 |
| **B. 工具袋机制落地** | 完整 ToolBag + 关键字预判引擎 | 2 | ~1,100 |
| **C. Web 端对话恢复** | 刷新后自动恢复对话 | 1 | ~30 |
| **D. IDE 提示增强** | 终端检测 + 工具使用说明扩充 | 1 | ~200 |
| **E. 工具注册表** | 完整 IDE 工具注册 + Schema 缓存 | 1 | ~580 |

---

## 二、A 类: VS Code 完整控制体系 (最重要变更)

### 2.1 新增 8 个 IDE 远程工具

**代码位置**: `zulong/ide/ide_tool_registry.py` (IDE_REMOTE_TOOLS)

| # | 工具名 | 功能 | 风险等级 |
|---|--------|------|---------|
| 1 | `vscode_run_command` | 执行 VS Code 命令 (200+ commands) | 动态(安全/高风险/阻止) |
| 2 | `get_diagnostics` | 获取 linter/编译器诊断信息 | LOW |
| 3 | `ask_user_input` | VS Code 输入框 | LOW |
| 4 | `ask_user_select_file` | 文件/文件夹选择对话框 | LOW |
| 5 | `vscode_manage_extension` | 扩展管理(安装/卸载/启用/禁用) | HIGH |
| 6 | `open_settings` | 打开 VS Code 设置 | LOW |
| 7 | `open_problems` | 打开 VS Code 问题面板 | LOW |
| 8 | `create_directory` | 创建目录(修复已有缺失) | LOW |

IDE_REMOTE_TOOLS 从 11 个扩展至 **19 个**。

### 2.2 VS Code 命令安全策略

**代码位置**: `config/vscode_command_policy.yaml`

三梯级安全模型:

| 级别 | 数量 | 行为 | 代表命令 |
|------|------|------|---------|
| **safe** | 48 个 | 自动执行 | `editor.action.formatDocument`, `editor.action.organizeImports`, `tasks.build`, `vscode.execute*Provider` |
| **high_risk** | 16 个 | 用户审批弹窗 | `git.stage`, `git.commit`, `git.push`, `deleteFile` |
| **blocked** | 5 个 | 硬拒绝 | `workbench.action.reloadWindow`, `workbench.action.closeWindow`, `workbench.action.quit` |

### 2.3 VscodeExecutionBridge (TypeScript)

**代码位置**: `zulong-ide/src/hosts/vscode/VscodeExecutionBridge.ts` (983 行)

VS Code 插件内的 TypeScript 执行桥，是 LLM 操作 VS Code 的唯一入口:

- 18 个工具 switch-case 分发
- `classifyCommand()`: 三梯级命令安全分类
- `confirmCommandExecution()`: 模态审批弹窗
- `getDiagnostics()`: `vscode.languages.getDiagnostics()` 分组按严重度返回
- `askUserInput()`: `vscode.window.showInputBox()`
- `askUserSelectFile()`: `vscode.window.showOpenDialog()`
- `vscodeManageExtension()`: 扩展列表/安装/卸载/启用/禁用
- `requestWebApproval()`: Web 端审批等待 (120s 超时)
- `sendBridgeInteraction()`: plan/action/observation/progress/approval/summary 结构事件
- `createCheckpoint()`: Git 风格文件变更检查点

---

## 三、B 类: 工具袋机制完整落地

### 3.1 tool_bag.py (新文件)

**代码位置**: `zulong/tools/tool_bag.py` (722 行)

核心数据结构和函数:

| 符号 | 类型 | 用途 |
|------|------|------|
| `ToolBagEntry` | dataclass | 单工具条目 (name/category/description/risk/executor/examples) |
| `ToolPrediction` | dataclass | 预测结果 (predicted_tools/context_bundle/reasons/risk_notes/task_graph_policy) |
| `VSCODE_COMMAND_TOOL_NAMES` | set | `{vscode_run_command}` |
| `VSCODE_DIAGNOSTIC_TOOL_NAMES` | set | `{get_diagnostics}` |
| `VSCODE_INTERACTION_TOOL_NAMES` | set | `{ask_user_input, ask_user_select_file}` |
| `VSCODE_EXTENSION_TOOL_NAMES` | set | `{vscode_manage_extension}` |
| `VSCODE_UI_TOOL_NAMES` | set | `{open_settings, open_problems}` |
| `IDE_TOOL_NAMES` | set | 原有 2 工具 + 7 个新 VS Code 工具 |

核心函数:

| 函数 | 用途 |
|------|------|
| `predict_tools_for_turn(text, registry, intent_result, referenced_nodes, has_task_graph)` | **11 步检测流水线**: 社交问候→实时信息→记忆→节点引用→项目分析→IDE工作区→任务图→文件写→终端→VS Code命令→诊断 |
| `supplement_tools(registry, missing_capability, reason, suggested_tools)` | L2 自主补充工具时返回匹配的工具条目和 schema |
| `build_tool_bag(registry)` | 从 ToolRegistry 构建工具目录 |
| `summarize_tool_bundle(prediction)` | 将预测结果转为人类可读文本 |
| `_category_for(name, category)` | 工具分类归类 (含 5 个 VS Code 新类别) |
| `_risk_for(name, category)` | 风险评估 (VS Code 命令/扩展=high, 诊断/交互/UI=low) |
| `_executor_for(name, category)` | 执行器归属 (IDE_TOOL_NAMES→ide) |
| `_examples_for(name, category)` | 示例生成 (含 5 个 VS Code 新类别) |
| `_needs_vscode_command(text)` | VS Code 命令需求检测 (格式化/git/提交/重构等) |
| `_needs_diagnostics(text)` | 诊断需求检测 (错误/报错/error/warning/诊断等) |
| `_needs_extension_management(text)` | 扩展管理需求检测 (扩展/插件/extension/plugin 等) |

### 3.2 tool_predictor.py (更新)

**代码位置**: `zulong/l1b/tool_predictor.py` (355 行)

**TOOL_BAG_FULL**: 从 17 个扩展至 **24 个** 工具 (新增 7 个 VS Code 工具)。

**KEYWORDS_MAP**: 从 5 组扩展至 **12 组**，新增 7 组 VS Code 关键字:

| 关键字模式 | 预测工具 |
|-----------|---------|
| `格式化\|format\|lint\|prettier\|重命名\|rename` | `vscode_run_command, read_file` |
| `git\|提交\|commit\|推送\|push\|分支\|branch\|merge\|合并\|暂存` | `vscode_run_command, execute_command` |
| `扩展\|插件\|extension\|plugin\|安装扩展\|卸载扩展` | `vscode_manage_extension` |
| `设置\|setting\|配置\|preference` | `open_settings, read_file` |
| `错误\|报错\|error\|warning\|诊断\|diagnostic\|lint错误` | `get_diagnostics, open_problems, read_file` |
| `选择文件\|打开文件\|browse\|选择目录` | `ask_user_select_file, list_files` |
| `输入\|填入\|回答` | `ask_user_input` |

---

## 四、C 类: Web 端刷新对话恢复

**代码位置**: `openclaw_bridge/web/static/index.html`

**问题**: 页面刷新后，`switchToSession()` 因 WebSocket 未连接而失败 → 聊天区显示"无法加载对话记录"。WS 连接后 `syncBackendState()` 只刷侧栏，不加载消息。

**修复**:

| 变更 | 行号 | 内容 |
|------|------|------|
| 新增标志位 | L2532 | `let _messagesReloadNeeded = false` |
| switchToSession 修改 | L2625-2632 | WS 未连接时设标志 + 显示"正在重连后将自动恢复对话..." |
| syncBackendState 修改 | L7159-7168 | WS 重连后检测标志 → 自动发送 `GET_SESSION_MESSAGES` |

**数据流**: `InteractionStore (SQLite)` ← 每消息持久化 → 刷新后 `GET /api/chat/sessions` → `GET_SESSION_MESSAGES` → 前端渲染

---

## 五、D 类: IDE Prompt 增强

**代码位置**: `zulong/ide/ide_prompt_handler.py`

### 5.1 终端环境自动检测

`_build_zulong_system_prompt()` 新增逻辑:

```
检测 SHELL/TERM 环境变量 → Git Bash / PowerShell / CMD
注入平台特定命令语法提示
```

### 5.2 VS Code 工具使用说明扩充

COMPLEX 系统提示中新增 5 行 VS Code 工具使用指令:

```
- VS Code 命令 → vscode_run_command(command='editor.action.formatDocument')
- 检查代码错误 → get_diagnostics()
- 用户输入 → ask_user_input(prompt='提示') / ask_user_select_file
- 管理扩展 → vscode_manage_extension(action='list')
```

### 5.3 RESUME 模式增强

新增 `has_only_root` 检测: 恢复时若任务图仅有根节点，提示 LLM"创建结构"而非"仅使用已有节点"。

### 5.4 XML 工具残余检测

`extract_ide_tool_block()` 新增诊断日志: 工具区剥离失败时报警含哪些 XML 标签残留。

---

## 六、E 类: IDE 工具注册表

**代码位置**: `zulong/ide/ide_tool_registry.py` (581 行)

### 6.1 新增类

| 类 | 用途 |
|---|------|
| `CachedSchema` | 带 SHA-256 哈希的工具 Schema 缓存条目 |
| `SchemaCache` | 工具 Schema 缓存管理器 (命中率统计/失效检测) |
| `IDEToolRegistry` | IDE 工具注册表 (合并内部+远程工具, 意图过滤, 动态更新) |

### 6.2 _IDE_TOOL_SCHEMAS 扩展

从原 11 个扩展至 **19 个** OpenAI function-calling schema，覆盖全部 IDE_REMOTE_TOOLS。

### 6.3 _ZULONG_TOOLS_DISABLED_IN_IDE_MODE

新增 `{exec_write_file, exec_run_command, exec_read_file}` — IDE 模式下禁用与远程工具重叠的内部工具。

### 6.4 _RESUME_EXCLUDED_INTERNAL_TOOLS

新增 `{task_create_plan, task_add_node}` — 任务恢复时不暴露可能破坏已恢复任务图结构的工具。

---

## 七、文件清单

### 新建文件 (未提交到 git)

| 文件 | 行数 | 说明 |
|------|------|------|
| `zulong/tools/tool_bag.py` | 722 | 工具袋预测引擎 |
| `zulong/l1b/tool_predictor.py` | 355 | L1-B 关键字工具预判器 |
| `zulong/ide/ide_tool_registry.py` | 581 | IDE 工具注册表 + Schema 缓存 |
| `zulong/ide/ide_prompt_handler.py` | 518 | IDE 系统提示处理器 |
| `zulong-ide/src/hosts/vscode/VscodeExecutionBridge.ts` | 983 | VS Code TypeScript 执行桥 |
| `config/vscode_command_policy.yaml` | 85 | VS Code 命令安全策略 |

### 修改文件 (未提交到 git)

| 文件 | 变更行数 | 说明 |
|------|---------|------|
| `openclaw_bridge/web/static/index.html` | ~30 | Web 刷新对话恢复修复 |
| `zulong/ide/ide_server.py` | ~200 | REST API 扩展 (LLM配置/模型层/会话CRUD/任务图谱) + PreloadManager |
| `zulong/launcher/web_chat_router.py` | ~200 | 消息持久化 + 统一协议 + 对话编排 + LLM/模型/会话 API |

---

## 八、建议 TSD 更新内容

### 8.1 新增章节

| 建议编号 | 标题 | 内容 |
|---------|------|------|
| §23.X | **VS Code 完整控制体系** | VS Code 执行桥 + 8 工具 + 三梯级安全策略 + 命令白名单 |
| §23.X+1 | **IDE 工具注册表** | IDEToolRegistry + SchemaCache + 内部/远程工具分类 + 意图过滤 |
| §23.X+2 | **Web 会话持久化与恢复** | InteractionStore + 页面刷新消息恢复机制 |

### 8.2 更新章节

| 章节 | 更新内容 |
|------|---------|
| §23.2.2 工具袋结构 | TOOL_BAG_FULL 从 17 扩至 24; 新增 VS Code 工具分类 (VSCODE_COMMAND/DIAGNOSTIC/INTERACTION/EXTENSION/UI) |
| §23.2.3 L1-B 工具预判 | KEYWORDS_MAP 从 5 组扩展到 12 组; 新增 3 个检测函数 (_needs_vscode_command/_needs_diagnostics/_needs_extension_management) |
| §23.2.4 预测流水线 | predict_tools_for_turn 从 ~8 步扩至 11 步 |

### 8.3 修订历史条目

```
| v2.9 | 2026-05-26 | VS Code 完整控制 + 工具袋落地 + Web 刷新恢复:
- VS Code 完整控制体系: 8 个新 IDE 工具(vscode_run_command/get_diagnostics/ask_user_input/ask_user_select_file/vscode_manage_extension/open_settings/open_problems/create_directory) + VscodeExecutionBridge(983行) + 三梯级命令安全策略
- 工具袋机制完整落地: tool_bag.py(722行) + L1BToolPredictor 关键字扩展(5组→12组)
- Web 端刷新对话恢复: 竞态条件修复, InteractionStore 持久化 + WS 重连自动恢复
- IDE Prompt 增强: 终端环境自动检测 + VS Code 工具指令扩充 + RESUME 模式 has_only_root 检测
- IDE 工具注册表: IDEToolRegistry + SchemaCache + IDE_REMOTE_TOOLS(11→19)
```

---

## 九、验证清单

| 验证项 | 状态 |
|--------|------|
| IDE_REMOTE_TOOLS = 19 个 (含 8 个新增) | ✅ 已验证 |
| _IDE_TOOL_SCHEMAS = 19 个 (含 8 个新增) | ✅ 已验证 |
| TOOL_BAG_FULL = 24 个 (含 7 个新增) | ✅ 已验证 |
| KEYWORDS_MAP = 12 组 (含 7 组新增) | ✅ 已验证 |
| L1-B 预测测试 (格式化/git/扩展/报错) 全部匹配 | ✅ 已验证 |
| TypeScript 编译 0 错误 | ✅ 已验证 |
| Python 语法检查全部通过 | ✅ 已验证 |
| WebSocket GET_SESSION_MESSAGES 正确返回消息 | ✅ 已验证 |
| 服务器运行 19/19 模块 | ✅ 已验证 |
