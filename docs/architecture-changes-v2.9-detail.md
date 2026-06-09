# 祖龙系统 v2.9 详细架构逻辑文档

> 生成日期: 2026-05-26
> 基准 TSD: v2.8
> 状态: 完整架构逻辑分析，基于实时代码深度阅读

---

## 目录

1. [工具袋 (Tool Bag) 完整架构逻辑](#1-工具袋-tool-bag-完整架构逻辑)
2. [L1-B 预判引擎 (Tool Predictor) 完整架构逻辑](#2-l1-b-预判引擎-tool-predictor-完整架构逻辑)
3. [IDE 工具注册与 Schema 缓存](#3-ide-工具注册与-schema-缓存)
4. [VS Code 执行桥与安全模型](#4-vs-code-执行桥与安全模型)
5. [IDE Prompt 构建流水线](#5-ide-prompt-构建流水线)
6. [Web 会话持久化与恢复机制](#6-web-会话持久化与恢复机制)

---

## 1. 工具袋 (Tool Bag) 完整架构逻辑

**代码位置**: `zulong/tools/tool_bag.py` (722 行)

### 1.1 核心设计理念

`tool_bag.py` 是工具袋机制的核心实现，**完全确定性**（非 ML 驱动）。它不进行完整的对话轮次分类（chat/complex/resume），仅预测下一 L2 步骤可能需要的具体工具和上下文提示。这确保 L2 始终拥有最佳的可用工具组合，同时保留完全的自主选择权。

### 1.2 核心数据结构

#### 1.2.1 ToolBagEntry — 单工具描述条目

```python
@dataclass
class ToolBagEntry:
    name: str                # 工具名称
    category: str            # 分类: memory/task_graph/project_code/file_write/terminal/
                             #       vscode_command/vscode_diagnostic/vscode_interaction/
                             #       vscode_extension/vscode_ui/ide_bridge/network/robot/
                             #       system/utility
    description: str         # 人类可读描述
    inputs: List[str]        # 参数列表 (从 OpenAI schema 提取)
    risk: str               # 风险级别: low/medium/high
    executor: str           # 执行器: backend/vscode_bridge/external_service
    requires_approval: bool # 是否需要审批 (risk=high → True)
    examples: List[str]     # 使用示例 (中文化)
```

#### 1.2.2 ToolPrediction — 预测结果

```python
@dataclass
class ToolPrediction:
    predicted_tools: List[str]      # 预判工具列表 (按优先级排序)
    context_bundle: Dict[str, Any]  # 上下文标记包:
                                    #   turn_shape: "simple_social"/None
                                    #   needs_realtime: bool
                                    #   needs_memory: bool
                                    #   referenced_nodes: list
                                    #   needs_project_context: bool
                                    #   needs_ide_workspace: bool
                                    #   needs_ide_file_write: bool
                                    #   needs_directory_create: bool
                                    #   needs_vscode_command: bool
                                    #   needs_diagnostics: bool
    reasons: List[str]              # 每个预判的推理理由
    risk_notes: List[str]           # 风险提示
    task_graph_policy: str          # 任务图策略:
                                    #   "none" — 不需要任务图
                                    #   "inspect" — 需要检视现有节点
                                    #   "reuse" — 复用现有任务图
                                    #   "inspect_or_create" — 检视或创建
```

### 1.3 11 步预测流水线 (`predict_tools_for_turn`)

每一步都检查 `lower(text)` 中是否存在关键指标词，按顺序执行，不短路：

| 步骤 | 函数 | 检测内容 | 触发词示例 | 添加工具 |
|------|------|---------|-----------|---------|
| 0 | — | 始终可用工具 | — | `request_tool_supplement`, `search_tools` |
| 1 | `_is_simple_social()` | 社交问候（≤18字符） | 你好/hello/谢谢/早上好 | **清空全部工具**，返回 `turn_shape=simple_social` |
| 2 | `_needs_realtime()` | 实时信息需求 | 天气/新闻/热搜/股价/汇率 | `web_search` |
| 3 | `_needs_memory()` | 记忆查询 | 记得/回忆/之前/上次/历史 | `recall_memory`, `read_memory_node`, `discover_related`, `search_experience` |
| 4 | `referenced_nodes` | 节点引用（参数检查） | 传入 `referenced_nodes` 非空 | `read_memory_node`, `task_get_detail`, `task_view_overview` |
| 5 | `_needs_project_read()` | 代码/项目分析 | 代码/项目/文件/模块/函数/类/bug + 文件扩展名正则 | `zulong_code_query`, `search_code_symbols`, `get_symbol_context`, `get_impact_analysis`, `index_*`, `analyze_module` |
| 6 | `_needs_ide_workspace_open()` | IDE工作区打开 | 打开+IDE/项目/工作区+路径正则 | `ide_open_workspace` |
| 7 | `_needs_task_graph()` | 任务管理 | 任务/计划/步骤/继续/恢复/实现/开发/重构 + intent检查 | `task_view_overview`, `task_get_detail`, `task_create_plan`, `task_add_node`, `task_mark_status`, `task_update_node`, `task_suspend`, `submit_final_answer` |
| 8 | `_needs_write()` | 文件写入 | 写/创建/新建/修改/修复/实现/删除/替换/重构 → 但排除否定形式 | `exec_write_file`, `task_attach_file`, `zulong_memory_write_with_code` → 如有显式宿主路径加 `ide_write_file` |
| 9 | `_needs_terminal()` | 命令执行 | 运行/执行/测试/编译/构建/安装/启动/npm/pip | `exec_run_command` |
| 10 | `_needs_vscode_command()` | VS Code命令 | 格式化/git/提交/重构/lint/重命名 | `vscode_run_command` |
| 11 | `_needs_diagnostics()` | 诊断信息 | 错误/报错/error/warning/诊断/编译 | `get_diagnostics`, `open_problems` |
| 12 | `_needs_extension_management()` | 扩展管理 | 扩展/插件/extension/plugin | `vscode_manage_extension` |

### 1.4 工具分类体系

共 15 个工具分类，其中 5 个为新增 VS Code 分类：

```
memory          — 记忆工具 (7个)
task_graph      — 任务图工具 (15个)
project_code    — 代码图谱工具 (9个)
file_write      — 文件写入工具 (8个)
terminal        — 终端工具 (3个)
vscode_command  — VS Code 命令 (1个) ← 新
vscode_diagnostic — 诊断工具 (1个) ← 新
vscode_interaction — 用户交互 (2个) ← 新
vscode_extension — 扩展管理 (1个) ← 新
vscode_ui       — UI面板 (2个) ← 新
ide_bridge      — IDE桥接 (2个)
network         — 网络 (2个)
robot           — 机器人控制
system          — 系统工具
utility         — 通用工具
```

分类函数 `_category_for()` 按优先级链依次匹配：`MEMORY > TASK > CODE > WRITE > TERMINAL > VSCODE_COMMAND > VSCODE_DIAGNOSTIC > VSCODE_INTERACTION > VSCODE_EXTENSION > VSCODE_UI > IDE_BRIDGE > NETWORK > ROBOT > SYSTEM > utility`。

### 1.5 风险分级

| 风险 | 触发条件 | 行为 |
|------|---------|------|
| **high** | 文件写入工具、终端工具、VS Code 命令、扩展管理 | `requires_approval=True` |
| **medium** | IDE 桥接工具、任务图修改、机器人控制 | `requires_approval=False` |
| **low** | 诊断、用户交互、UI、网络搜索、记忆 | `requires_approval=False` |

### 1.6 优先级排序 (`_finalize_prediction`)

工具按以下优先级排序后输出（数值越小越优先）：

```
0: request_tool_supplement
1: search_tools
2: web_search
3: ide_write_file / exec_write_file
4: exec_run_command
5: vscode_run_command
6: get_diagnostics
7: ask_user_input
8: ask_user_select_file
9: vscode_manage_extension
10: open_settings / open_problems
15: task_view_overview
16: task_create_plan
17: task_mark_status
50: 其余工具 (默认)
```

### 1.7 工具补充机制 (`supplement_tools`)

L2 在现有工具不足时可以调用 `request_tool_supplement`，触发 `supplement_tools()` 函数：

1. **`list_all_tools=True`**: 返回工具袋中全部工具
2. **指定工具名**: 直接从 `suggested_tools` 中匹配
3. **关键词查询**: 对 `name + category + description + inputs + examples` 做 token 匹配 + `_keyword_score()` 加权评分（匹配权重 +4，分类匹配 +2）

### 1.8 否定检测 (`_needs_write`)

特殊处理否定表达式，防止误判写入需求：
- 排除: "不要修改" / "不修改" / "无需修改" / "别改" / "不要动文件" / "只读" / "仅分析" 等
- 检测到否定词时直接返回 `False`

---

## 2. L1-B 预判引擎 (Tool Predictor) 完整架构逻辑

**代码位置**: `zulong/l1b/tool_predictor.py` (355 行)

### 2.1 架构定位

**L1BToolPredictor** 是 `tool_bag.py` 的薄包装层，提供 TSD 定义的标准接口。它替代了原 ALBERT "闲聊/复杂任务/任务恢复" 三类意图分类中的**两类**（闲聊/复杂任务），仅用于预判策略。ALBERT 其余 12 类不受影响。

```
调用链: L1-B Gatekeeper → L1BToolPredictor.predict_tools() →
        (可选) tool_bag.predict_tools_for_turn() →
        L2 InferenceEngine 消费预判结果
```

### 2.2 TOOL_BAG_FULL — 工具全量清单

从 17 个扩展至 **24 个**，新增 7 个 VS Code 完整控制工具：

| 序号 | 工具名 | 分类 | 风险 |
|------|--------|------|------|
| 1-6 | `read_file`, `write_to_file`, `replace_in_file`, `delete_file`, `list_files`, `search_files` | 文件操作 | LOW ~ CRITICAL |
| 7 | `execute_command` | 命令执行 | CRITICAL |
| 8-10 | `web_fetch`, `web_search`, `use_skill` | 网络与外部 | LOW ~ MEDIUM |
| 11-13 | `recall_memory`, `read_memory_node`, `save_memory_note` | 记忆与图谱 | LOW |
| 14-15 | `search_knowledge`, `discover_related` | 知识图谱 | LOW |
| 16-17 | `open_file`, `show_diff` | IDE 桥接 | LOW |
| **18** | **`vscode_run_command`** ← 新 | VS Code 命令 | HIGH |
| **19** | **`get_diagnostics`** ← 新 | VS Code 诊断 | LOW |
| **20** | **`ask_user_input`** ← 新 | VS Code 用户交互 | LOW |
| **21** | **`ask_user_select_file`** ← 新 | VS Code 文件选择 | LOW |
| **22** | **`vscode_manage_extension`** ← 新 | VS Code 扩展管理 | HIGH |
| **23** | **`open_settings`** ← 新 | VS Code 设置 | LOW |
| **24** | **`open_problems`** ← 新 | VS Code 问题面板 | LOW |

### 2.3 KEYWORDS_MAP — 关键词映射

从 5 组扩展至 **12 组**，新增 7 组 VS Code 关键词：

| # | 正则模式 | 预测工具 | 状态 |
|---|---------|---------|------|
| 1 | `写代码\|修改\|改\|创建\|新建` | `read_file, search_files, write_to_file, replace_in_file` | 原有 |
| 2 | `查\|搜\|找\|在哪\|搜索` | `search_files, search_knowledge, web_search, read_file` | 原有 |
| 3 | `运行\|执行\|跑\|测试` | `execute_command, read_file` | 原有 |
| 4 | `回忆\|之前\|上次\|记得` | `recall_memory, read_memory_node, discover_related` | 原有 |
| 5 | `知识\|关系\|谁\|什么\|哪里` | `search_knowledge, discover_related` | 原有 |
| 6 | `格式化\|format\|lint\|prettier\|重命名\|rename` | `vscode_run_command, read_file` | **新增** |
| 7 | `git\|提交\|commit\|推送\|push\|分支\|branch\|merge\|合并\|暂存` | `vscode_run_command, execute_command` | **新增** |
| 8 | `扩展\|插件\|extension\|plugin\|安装扩展\|卸载扩展` | `vscode_manage_extension` | **新增** |
| 9 | `设置\|setting\|配置\|preference` | `open_settings, read_file` | **新增** |
| 10 | `错误\|报错\|error\|warning\|诊断\|diagnostic\|lint错误` | `get_diagnostics, open_problems, read_file` | **新增** |
| 11 | `选择文件\|打开文件\|browse\|选择目录` | `ask_user_select_file, list_files` | **新增** |
| 12 | `输入\|填入\|回答` | `ask_user_input` | **新增** |

### 2.4 predict_tools 流程

```
1. 关键词+规则快速预判
   → 遍历 KEYWORDS_MAP 的 12 组正则
   → 对每组匹配到的 prompt，将对应工具加入 suggested set

2. 任务类型分类 (_classify_task_type)
   → COMPLEX_INDICATORS 命中 → COMPLEX_TASK
   → len(prompt) ≤ 18 → SIMPLE_CHAT
   → 否则 → COMPLEX_TASK

3. 简单对话补足
   → SIMPLE_CHAT 时也能加 web_search 和 recall_memory

4. 始终包含基础工具
   → read_file, search_files

5. 置信度计算 (_calc_confidence)
   → suggested ≤ 2 → 0.5
   → suggested ≤ 4 → 0.7
   → 否则 → min(0.95, 0.7 + count * 0.05)

6. 返回结果
   → { suggested_tools, tool_bag(全量), confidence, reason, task_type }
```

### 2.5 predict_from_tool_bag 桥接

当 `registry` 可用时，`predict_from_tool_bag()` 调用 `tool_bag.predict_tools_for_turn()` 获取详细预测结果，合并入 L1BToolPredictor 的简化结果中，添加 `risk_notes` 和 `task_graph_policy`。

---

## 3. IDE 工具注册与 Schema 缓存

**代码位置**: `zulong/ide/ide_tool_registry.py` (581 行)

### 3.1 工具分类架构

```
┌─────────────────────────────────────────┐
│          IDEToolRegistry                 │
│                                          │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ 内部工具       │  │ 远程工具           │ │
│  │ (internal)   │  │ (remote)         │ │
│  │              │  │                  │ │
│  │ task/memory/ │  │ read_file        │ │
│  │ attention/   │  │ write_to_file    │ │
│  │ code_graph/  │  │ execute_command  │ │
│  │ 等           │  │ ... (19个)       │ │
│  │              │  │                  │ │
│  │ 服务端直接执行  │  │ XML→IDE插件执行   │ │
│  └──────────────┘  └──────────────────┘ │
│                                          │
│  SchemaCache ──── SHA-256 哈希失效机制    │
│  命中率统计 + 自动失效                    │
└─────────────────────────────────────────┘
```

### 3.2 IDE_REMOTE_TOOLS — 远程工具集合

从 11 个扩展至 **19 个**：

| # | 工具名 | 说明 | 状态 |
|---|--------|------|------|
| 1-11 | `read_file`, `write_to_file`, `replace_in_file`, `delete_file`, `execute_command`, `search_files`, `list_files`, `list_code_definition_names`, `browser_action`, `ask_followup_question`, `attempt_completion` | 原有 | 原有 |
| 12 | `vscode_run_command` | 执行 VS Code 命令 | **新增** |
| 13 | `get_diagnostics` | 获取诊断信息 | **新增** |
| 14 | `ask_user_input` | 弹出输入框 | **新增** |
| 15 | `ask_user_select_file` | 弹出文件选择 | **新增** |
| 16 | `vscode_manage_extension` | 扩展管理 | **新增** |
| 17 | `open_settings` | 打开设置 | **新增** |
| 18 | `open_problems` | 打开问题面板 | **新增** |
| 19 | `create_directory` | 创建目录 | **新增** (修复已有缺失) |

### 3.3 工具过滤规则

#### _ZULONG_TOOLS_DISABLED_IN_IDE_MODE

IDE 模式下禁用与远程工具功能重叠的内部工具：

```
exec_write_file  ← 与 write_to_file 重叠
exec_run_command ← 与 execute_command 重叠
exec_read_file   ← 与 read_file 重叠
```

#### _RESUME_EXCLUDED_INTERNAL_TOOLS

任务恢复时不暴露可能破坏已恢复任务图结构的工具：

```
task_create_plan  ← 会创建全新图谱
task_add_node     ← 节点已在恢复的图谱中
```

### 3.4 SchemaCache — 缓存机制

```
数据结构:
  Dict[str, CachedSchema]  # tool_name → {schema, definition_hash, created_at, hit_count}

流程:
  get(name, definition):
    1. 从 _cache 取 CachedSchema
    2. 计算当前 definition 的 SHA-256 (前16位)
    3. 哈希不匹配 → 删除缓存条目，返回 None (自动失效)
    4. 匹配 → hit_count++，_cache_hits++，返回缓存的 schema

  set(name, schema, definition):
    1. 计算 definition 的 SHA-256
    2. 创建 CachedSchema(schema, hash)
    3. 写入 _cache

统计: get_hit_rate() → _cache_hits / _total_requests
```

**失效场景**: 工具定义（parameters/description 等）有任何变更时，SHA-256 不匹配自动失效，无需手动清除。

### 3.5 _IDE_TOOL_SCHEMAS — 19 个 OpenAI FC Schema

从 11 个扩展至 19 个，每个 schema 包含完整的 `type: "function"` + `function.name` + `function.description` + `function.parameters` (含 required 字段)。

### 3.6 get_combined_tool_definitions_for_intent 流程

```
1. intent 参数取值: "complex" | "resume" | "chat" (默认 "complex")

2. _get_filtered_internal_tools():
   a. 遍历 self.tool_engine.registry.tools
   b. 跳过 disabled 工具
   c. CHAT模式 → 仅加载 extra_include 中的工具 (极简工具集)
   d. 非CHAT模式:
      - 跳过 _ZULONG_TOOLS_DISABLED_IN_IDE_MODE (3个执行工具)
      - 跳过 IDE_REMOTE_TOOLS (由 _IDE_TOOL_SCHEMAS 单独提供)
      - RESUME模式 → 额外跳过 task_create_plan + task_add_node
   e. 尝试 SchemaCache.get(name, tool):
      - 命中 → 直接用缓存 schema
      - 未命中 → tool.get_function_schema() → SchemaCache.set()
   f. (非CHAT) 追加 TaskGraph CRUD 工具 schema

3. 合并: internal(过滤后) + remote(_IDE_TOOL_SCHEMAS 全量)
```

---

## 4. VS Code 执行桥与安全模型

**代码位置**: `zulong-ide/src/hosts/vscode/VscodeExecutionBridge.ts` (983 行)
**安全策略**: `config/vscode_command_policy.yaml` (85 行)

### 4.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│              VscodeExecutionBridge                   │
│              (VS Code 扩展内 TypeScript)              │
│                                                      │
│  WebSocket ←→ ZulongWebSocket transport              │
│  ws://127.0.0.1:8090/ide                            │
│                                                      │
│  事件监听 (registerHandlers):                         │
│  ├─ tool_request → handleToolRequest()               │
│  ├─ ide_open_workspace → openWorkspace()             │
│  ├─ ide_open_file → openFile()                       │
│  ├─ ide_open_terminal → openTerminal()                │
│  ├─ ide_show_diff → showDiff()                       │
│  ├─ ide_get_context → sendIdeContext()                │
│  ├─ connected → sendIdeContext()                      │
│  └─ ide_approval_result → handleApprovalResult()      │
│                                                      │
│  18个工具执行方法 switch-case                          │
│  + 安全模型 (三梯级)                                  │
│  + 审批系统 (Web端/VS Code弹窗双通道)                   │
│  + Checkpoint 检查点                                  │
└──────────────────────────────────────────────────────┘
```

### 4.2 18 个工具执行方法

| # | 方法 | 工具名 | 关键行为 |
|---|------|--------|---------|
| 1 | `readFile()` | `read_file` | 路径解析 → 工作区边界检查 → 支持行范围 |
| 2 | `writeFile()` | `write_to_file` | 路径解析 → diff展示 → Web审批 → mkdir → 写入 → Checkpoint → 打开文件 |
| 3 | `createDirectory()` | `create_directory` | 路径解析 → Web审批 → mkdir → Checkpoint |
| 4 | `replaceInFile()` | `replace_in_file` | 路径解析 → SEARCH/REPLACE diff应用 → 审批 → 写入 → Checkpoint |
| 5 | `deleteFile()` | `delete_file` | 工作区边界检查 → Web审批(high) → 删除 → Checkpoint |
| 6 | `listFiles()` | `list_files` | 递归文件列表 (排除 node_modules/.git) |
| 7 | `searchFiles()` | `search_files` | 正则搜索 + 文件过滤 (最多500条) |
| 8 | `executeCommand()` | `execute_command` | 命令风险分级 → 审批 → terminal.sendText() |
| 9 | `listCodeDefinitionNames()` | `list_code_definition_names` | 复用 searchFiles 搜索定义模式 |
| 10 | `attempt_completion` | `attempt_completion` | 直接返回 result/response |
| 11 | `ask_followup_question` | `ask_followup_question` | 返回 "[Web端统一交互]" 提示 |
| **12** | **`vscodeRunCommand()`** | **`vscode_run_command`** | ← **新**: 三级安全分类 → blocked抛出/高风险审批 → executeCommand |
| **13** | **`getDiagnostics()`** | **`get_diagnostics`** | ← **新**: getDiagnostics() 按Error/Warning/Other分组 |
| **14** | **`askUserInput()`** | **`ask_user_input`** | ← **新**: showInputBox |
| **15** | **`askUserSelectFile()`** | **`ask_user_select_file`** | ← **新**: showOpenDialog |
| **16** | **`vscodeManageExtension()`** | **`vscode_manage_extension`** | ← **新**: list/install/uninstall/enable/disable |
| **17** | **`openSettings()`** | **`open_settings`** | ← **新**: workbench.action.openSettings |
| **18** | **`openProblems()`** | **`open_problems`** | ← **新**: workbench.actions.view.problems |

### 4.3 三梯级安全模型

#### 4.3.1 命令安全分级 (`classifyCommand()`)

```
blocked (阻止):
  ├─ workbench.action.reloadWindow    — 破坏 VS Code 稳定性
  ├─ workbench.action.closeWindow
  ├─ workbench.action.quit
  ├─ workbench.extensions.installExtension
  └─ workbench.extensions.uninstallExtension

high_risk (审批):
  ├─ git.stage / git.stageAll / git.stageSelectedRanges
  ├─ git.unstage / git.unstageAll / git.unstageSelectedRanges
  ├─ git.commit / git.commitAll / git.commitStaged
  ├─ git.push / git.pull
  ├─ git.clean / git.revert / git.revertSelectedRanges
  ├─ deleteFile / workbench.files.action.deleteFile
  └─ editor.action.clipboardCutAction

safe (免审批):
  ├─ 编辑器操作: formatDocument, organizeImports, rename, commentLine 等
  ├─ 任务运行: tasks.build, tasks.test, tasks.runTask
  ├─ 面板操作: problems.focus, output.focus, terminal.focus
  ├─ Provider: executeDocumentSymbolProvider, executeCompletionItemProvider 等
  ├─ Notebook: notebook.execute, notebook.cell.execute
  └─ 当前代码中未在 blocked/high_risk 的任意其他命令
```

#### 4.3.2 YAML 配置文件策略

`config/vscode_command_policy.yaml` 显式声明了:
- **safe**: 48 个命令 (编辑器+任务+面板+Provider+Notebook)
- **high_risk**: 16 个命令 (Git操作+破坏性文件操作+剪切)
- **blocked**: 5 个命令 (窗口破坏+扩展管理走专门工具)

> **注意**: 代码中的 `classifyCommand()` 实现（基于前缀匹配）是 YAML 策略的实时执行版本，两者需保持一致。

### 4.4 审批系统

#### 4.4.1 双通道审批

| 通道 | 触发方式 | 等待时间 | 适用场景 |
|------|---------|---------|---------|
| **Web端审批** | `requestWebApproval()` → `sendApprovalRequired()` → WebSocket → 等待 `ide_approval_result` | 120s 超时 | 文件写/删、命令执行、目录创建 |
| **VS Code弹窗** | `confirmCommandExecution()` → `showWarningMessage({modal: true})` | 用户点击 | VS Code 高风险命令 (git操作等) |

#### 4.4.2 Web 审批流程

```
LLM 请求工具
  → Python后端: 暂停FC循环, 发送 XML tool_use 至 IDE
  → VscodeExecutionBridge: executeTool()
  → 需要审批时:
     1. sendApprovalRequired() → WebSocket 发送审批请求
     2. 生成 approval_id = "approval_{timestamp}_{random}"
     3. 传入 pendingApprovals Map (key=approval_id, value=resolver)
     4. 启动 120s 超时定时器
     5. 等待 ide_approval_result 事件
     6. handleApprovalResult() → 找到 resolver → 调用 resolver(approved)
     7. 超时 → 自动拒绝
```

#### 4.4.3 文件变更审批

`confirmFileChange()` 特殊流程：
1. 工作区边界检查
2. `openDiffPreview()` — 在 VS Code 中展示差异视图 (base64 URI)
3. `waitForWebApproval()` — 等待 Web 端确认
4. 审批通过 → 发送 `diff_ready(status="approved")` → 写入
5. 审批拒绝 → 发送 `diff_ready(status="rejected")` → 不写入

### 4.5 Checkpoint 检查点系统

每次文件写入/删除/目录创建/文件替换后自动创建检查点：

```
createCheckpoint(summary)
  → getCheckpointTracker() → CheckpointTracker.create() → tracker.commit()
  → sendCheckpointStatus(summary, checkpointId, status, error?)
  → checkpoint 失败不阻断操作 (降级为 skipped/failed)
```

### 4.6 Bridge Interaction 事件

`sendBridgeInteraction()` 发送结构化事件，包含 7 种类型：

```
kind: "plan" | "action" | "observation" | "progress" | "approval" | "summary" | "user_interject"
```

每个事件包含: `pair_id`, `kind`, `status`, `title`, `detail`, `tool_name`

### 4.7 IDE 上下文上报

`sendIdeContext()` 在连接建立和每次请求时发送：
```
{
  workspace_path,
  active_file,
  active_selection: { start_line, end_line },
  open_tabs: [fsPath, ...]
}
```

---

## 5. IDE Prompt 构建流水线

**代码位置**: `zulong/ide/ide_prompt_handler.py` (518 行)

### 5.1 整体流程

```
用户消息到达
  → IDEPromptHandler.process_system_prompt()
    1. 找到 messages 中的 system 消息
    2. extract_ide_tool_block() 提取/移除 IDE XML 工具定义区
    3. 保留非工具定义部分 (角色设定、规则等)
    4. 检测终端环境 (SHELL/TERM 环境变量)
    5. 检索祖龙上下文 (记忆+任务+经验)
    6. 根据 intent 选择 COMPLEX 或 RESUME 模板
    7. 注入增强内容 → 替换原始 system prompt
```

### 5.2 XML 工具定义区提取 (`extract_ide_tool_block`)

分两个阶段：

#### 阶段一：模式匹配（4 种模式）

| 模式 | 正则 | 示例 |
|------|------|------|
| IDE标准 | `={3,}\s*\n\s*TOOL USE\s*\n.*?(?=\n={3,}\|\Z)` | `====\nTOOL USE\n...\n====` |
| 同行 | `={3,}\s*TOOL USE\s*={0,}\s*\n.*?` | `==== TOOL USE ====` |
| Markdown H1 | `# Tool(?:s\| Use)\s*\n.*?(?=\n# [A-Z]\|\Z)` | `# Tools` |
| Markdown H2 | `## Tool(?:s\| Use)\s*\n.*?(?=\n## [A-Z]\|\Z)` | `## Tool Use` |

#### 阶段二：XML 标签密度检测（fallback）

当模式匹配未命中，但 system prompt 中包含 ≥ 3 个 XML 工具标签时：
1. 找到第一个标签位置
2. 向前搜索分隔线 (`====`) 或标题 (`# `)
3. 向后搜索最后一个标签的闭合标签
4. 再向后搜索段落分隔
5. 截取工具区 → 返回 (tool_block, remaining)

#### 残留检测

当两个阶段都失败时，检查是否有 XML 工具标签残留，如有则生成警告日志：
```
"[IDEPromptHandler] 未能剥离工具定义区！system prompt 含 N 个 XML 工具标签: [...]"
```

### 5.3 终端环境自动检测

`_build_zulong_system_prompt()` 中读取 `os.environ['SHELL']` 和 `os.environ['TERM']`：

| 检测条件 | 终端类型 | 注入提示 |
|---------|---------|---------|
| `bash` in SHELL or `git` in SHELL | Git Bash (Unix-like) | 使用 Git Bash 语法: ls, grep, find, chmod |
| `powershell` or `pwsh` in SHELL | PowerShell | 使用 PowerShell 语法: Get-ChildItem, Select-String |
| `cmd` in SHELL or SHELL='' | CMD (Windows) | 使用 CMD 语法: dir, findstr, type, 路径使用反斜杠 |
| 其他 | Unknown | 根据环境变量判断 |

### 5.4 COMPLEX 模板规则

```markdown
【任务管理规则】(~100 行详细指令)
核心原则:
  - ⚠️ 不要反问用户！直接根据已有信息开始规划和执行
  - ⚠️ 文本叙述不等于执行！必须调用工具
  - ⚠️ 每次回复必须包含至少一个工具调用

任务图结构规则:
  - 先创建顶层模块节点 (parent_id='req')
  - 对于复杂模块，再创建子步骤节点
  - 目标深度: 至少 2 层，复杂任务可达 3 层
  - 先搭建完整大纲再执行

代码图谱强制规则:
  - 分析新项目 → 第一步必须调用 index_project
  - 分析单文件 → 调用 index_code_file
  - 查找函数/类 → search_code_symbols
  - 了解调用关系 → get_symbol_context
  - 评估修改影响 → get_impact_analysis

VS Code 工具指令 (新增5行):
  - VS Code 命令 → vscode_run_command
  - 检查代码错误 → get_diagnostics()
  - 用户输入 → ask_user_input / ask_user_select_file
  - 管理扩展 → vscode_manage_extension
```

### 5.5 RESUME 模板规则

```markdown
【任务恢复模式】
系统已自动恢复之前挂起的任务。任务图已加载到内存中。

动态注入:
  - 任务进度表 (_build_progress_table)
  - 第一个未完成任务标记

has_only_root 检测:
  ✅ 只有根节点 → 提示 LLM "创建任务结构"：
     - 使用 task_add_node 添加子任务节点
     - 创建至少2层结构
     - 添加完成后调用 task_view_overview
  ❌ 有完整结构 → 提示 LLM "使用现有节点"：
     - ✗ 禁止调用 task_create_plan
     - ✗ 禁止调用 task_add_node
     - ✓ 只使用 task_mark_status 更新现有节点状态
```

### 5.6 上下文检索 (`retrieve_zulong_context`)

三个检索通道：

| 通道 | 来源 | top_k | 输出 |
|------|------|-------|------|
| 记忆检索 | `MemoryGraph.retrieve_context()` | 5 | `memory_context`: "[{node_type}] {label}: {content[:200]}" |
| 任务图状态 | `TaskGraph.to_focused_planning_table()` | — | `task_context`: 表格文本 (截断至500字符) |
| 经验检索 | `RAGManager.search_all()` | 2 | `experience_hints`: "[{lib_name}] {content[:150]}" |

---

## 6. Web 会话持久化与恢复机制

### 6.1 问题根因

**问题**: Web端 (`index.html`) 页面刷新后，聊天区丢失 LLM 对话内容。

**根因**: `switchToSession()` 在 WebSocket 未连接时执行 REST API 获取会话元数据，但因 WS 未连无法发送 `GET_SESSION_MESSAGES` 获取消息。WS 连接成功后 `syncBackendState()` 只刷新侧栏会话列表，不会重新加载当前会话的消息。

### 6.2 修复方案

**文件**: `zulong_web/static/index.html`

三个变更点：

```javascript
// 1. 新增标志位 (L2532)
let _messagesReloadNeeded = false;

// 2. switchToSession 修改 (L2625-2632)
//    原: WS 未连接时报错"无法加载对话记录"
//    新: 设置 _messagesReloadNeeded = true，显示"正在重连后将自动恢复对话..."
if (!_ws || _ws.readyState !== WebSocket.OPEN) {
    _messagesReloadNeeded = true;
    _pendingSessionId = sessionId;
    renderSystemHint("正在重连后将自动恢复对话...");
    return;  // 不报错
}

// 3. syncBackendState 修改 (L7159-7168)
//    原: 只刷新侧栏
//    新: WS 重连后检测标志 + 自动恢复
if (_messagesReloadNeeded && _pendingSessionId) {
    _messagesReloadNeeded = false;
    _ws.send(JSON.stringify({
        type: "GET_SESSION_MESSAGES",
        session_id: _pendingSessionId
    }));
}
```

### 6.3 数据流

```
┌─ 消息写入时 ─────────────────────────────────────┐
│  LLM reply → web_chat_router.py                 │
│    → ConversationOrchestrator                   │
│      → InteractionStore.save_interaction()      │
│        → SQLite (interactions 表)               │
└─────────────────────────────────────────────────┘

┌─ 页面刷新后恢复时 ────────────────────────────────┐
│  1. initSessionManager() → REST API             │
│     GET /api/chat/sessions                      │
│     → 填充侧栏会话列表                            │
│                                                 │
│  2. connect() → WebSocket 建立                   │
│     → syncBackendState()                        │
│       → _messagesReloadNeeded 检查               │
│       → GET_SESSION_MESSAGES                    │
│         → InteractionStore.get_session_messages()│
│         → CHAT_MESSAGE 逐个下发                  │
│                                                 │
│  3. normalizeIncomingMessage() 解码嵌套 payload  │
│  4. renderSessionMessages() 按 role 渲染消息      │
└─────────────────────────────────────────────────┘
```

### 6.4 消息持久化路径

| 路径 | 触发条件 | 持久化内容 | 方式 |
|------|---------|-----------|------|
| 用户消息 | WebSocket `CHAT_MESSAGE` | user message | `InteractionStore.save_interaction()` |
| LLM 回复 | `ConversationOrchestrator` 完成 | assistant reply | `InteractionStore.save_interaction()` |
| 工具调用 | FC 循环中工具执行 | tool_call / tool_result | `InteractionStore.save_interaction()` |
| 系统消息 | 会话状态变更 | system event | `InteractionStore.save_interaction()` |

### 6.5 消息恢复通道

| 通道 | 协议 | 触发时机 | 数据来源 |
|------|------|---------|---------|
| REST API | `GET /api/chat/sessions` | 页面初始化 | `InteractionStore.list_sessions()` |
| WebSocket | `GET_SESSION_MESSAGES` | WS 重连 + _messagesReloadNeeded | `InteractionStore.get_session_messages()` |
| MemoryMirror | 内部检索 | 切换会话 | `MemoryMirror.retrieve_related()` 3层层级 |

---

## 附录: 完整调用链路图

```
用户输入 (IDE 前端)
    │
    ▼
┌─────────────────────────────────────────┐
│  L1-B Gatekeeper (2652行)               │
│  ├─ ALBERT-tiny 15类意图分类            │
│  └─ L1BToolPredictor.predict_tools()    │
│      ├─ KEYWORDS_MAP 12组关键词匹配     │
│      └─ (可选) tool_bag 详细预测         │
│          ├─ predict_tools_for_turn()    │
│          │   └─ 11步检测流水线           │
│          └─ supplement_tools() (按需)    │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  IDEToolRegistry                        │
│  ├─ classify(tool_name): internal/remote │
│  ├─ get_combined_tool_definitions()     │
│  │   ├─ _get_filtered_internal_tools()  │
│  │   │   ├─ SchemaCache.get() (SHA-256) │
│  │   │   ├─ 禁用重叠工具                │
│  │   │   └─ RESUME 排除工具             │
│  │   └─ _IDE_TOOL_SCHEMAS (19个远程)    │
│  └─ SchemaCache 命中率统计              │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  IDEPromptHandler                       │
│  ├─ extract_ide_tool_block()            │
│  │   ├─ 4种模式匹配                     │
│  │   └─ XML标签密度fallback             │
│  ├─ 终端环境检测 (SHELL/TERM)           │
│  ├─ 上下文检索 (记忆+任务+经验)          │
│  └─ 模板注入 (COMPLEX/RESUME)           │
│      ├─ has_only_root 检测              │
│      ├─ _build_progress_table()         │
│      └─ VS Code 工具指令 (5行新增)      │
└──────────────┬──────────────────────────┘
               ▼
┌─────────────────────────────────────────┐
│  L2 InferenceEngine (3934行)            │
│  FC循环 → LLM 推理                      │
│  ├─ 内部工具: 服务端直接执行             │
│  └─ 远程工具: WebSocket → IDE            │
│      └─ VscodeExecutionBridge           │
│          ├─ 18个工具 switch-case        │
│          ├─ 三梯级安全模型               │
│          ├─ 双通道审批 (Web/弹窗)        │
│          ├─ Diff展示 + 文件变更审批      │
│          └─ Checkpoint 检查点            │
└─────────────────────────────────────────┘
```

---

> 本文档基于实际代码深度阅读生成，所有行号、函数名、数据结构均与源代码完全一致。
> 基于 TSD v2.8 的变更已在上文草案 (`architecture-changes-v2.9-draft.md`) 中汇总，本文档为详细的架构逻辑阐述。

🤖 Generated with [Qoder](https://qoder.com)
