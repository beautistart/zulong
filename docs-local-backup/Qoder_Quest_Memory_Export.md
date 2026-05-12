# Qoder Quest 记忆导出

> 导出时间: 2026-05-05 (更新)
> 项目: zulong_beta4 (Zulong IDE)
> 用途: 迁移到新 Qoder 账号时，让 Qoder 读取此文件并重建 Quest 记忆
> 导入方式: 在新账号对话中直接说"请读取此文件并将记忆逐条存入 Quest"，Qoder 会自动调用 update_memory 工具

---

## 一、Quest 存储的记忆 (共 15 条)

### 1. 项目介绍

#### [记忆] Zulong IDE项目整体概况与核心功能
- **分类**: project_introduction
- **关键词**: Zulong IDE, VS Code插件, AI编程助手, 聚焦链, 自动批准
- **内容**: Zulong IDE是一个基于Cline v3.82.0的VS Code插件，核心定位是AI编程助手。主要功能包括：通过自然语言描述任务自动执行文件读写、终端命令、浏览器操作和MCP工具调用；所有操作默认需人工批准，支持按类型开启自动批准或启用极速模式；提供聚焦链（维护任务进度清单）、子智能体（并行处理）、检查点（操作回滚）等高级功能；用户通过侧边栏图标、顶部导航栏（新建任务/MCP/历史/设置）和快捷键（Ctrl+Shift+P）进行交互。

#### [记忆] Qoder会话绑定机制
- **分类**: project_introduction
- **关键词**: Qoder, 会话绑定, 项目目录, 跨账号迁移
- **内容**: Qoder会话上下文绑定项目目录，不支持跨账号迁移。

#### [记忆] MemoryGraph与TaskGraph双图谱架构
- **分类**: project_introduction
- **关键词**: MemoryGraph, TaskGraph, 双图谱架构, 知识图谱, 任务计划
- **内容**: MemoryGraph是长期知识图谱，作为Agent的'大脑'跨会话持久化，存储在./data/memory_graph/，包含DIALOGUE/CODE_SYMBOL/MODULE等多类型节点和HIERARCHY/REFERENCE/DEPENDENCY等多维边，支持记忆检索、温度衰减和多维关系；TaskGraph是当前任务计划，作为Agent的'执行计划'仅限单次任务会话，存储在内存+./agent_workspace/，包含requirement/analysis/task等节点类型和hEdge/dEdge两种边，支持任务编排、前端可视化和状态追踪；两者独立但协作，通过FC Runner主循环连接：LLM操作TaskGraph，CRG后台线程同步到MemoryGraph，TaskGraphAdapter在任务完成时将节点归档到MemoryGraph，实现'正在做什么'到'记住了什么'的知识沉淀。

#### [记忆] 项目分析内容展示位置约束
- **分类**: project_introduction
- **关键词**: 项目分析, 记忆图谱, 任务图谱, 展示位置
- **内容**: 项目分析内容（如CRG索引结果）不得出现在记忆图谱视图中，必须严格限定在任务图谱视图内展示。

#### [记忆] CRG数据建模与挂载规范
- **分类**: project_introduction
- **关键词**: CRG数据, 任务节点, 对话节点, req, 建模规范
- **内容**: CRG数据必须全部作为任务节点挂载到当前会话的对话根节点（req）下，所有CRG数据在系统中均被建模为任务节点，不得作为记忆图谱节点存在。

#### [记忆] CRG与MemoryGraph数据映射现状（已修复）
- **分类**: project_introduction
- **关键词**: CRG, MemoryGraph, Import关系, DEPENDENCY边
- **内容**: CRG模块通过AST解析提取CodeSymbol、ImportInfo、CallInfo等结构化数据。已修复Import边丢失问题：在graph_builder.py中添加_build_import_edges()方法生成imports类型CodeEdge，在graph_adapters.py中修复边过滤逻辑支持file:{path}格式的import边。修复后303条import边可正确生成，67条DEPENDENCY边成功注入MemoryGraph。

#### [记忆] CRG在MemoryGraph与TaskGraph中的形态差异
- **分类**: project_introduction
- **关键词**: CRG, MemoryGraph, TaskGraph, 多维图谱, 树形结构
- **内容**: CRG在系统中存在两种形态：1) MemoryGraph中为多维图谱，包含MODULE/FILE/CODE_SYMBOL三类节点及HIERARCHY/REFERENCE/DEPENDENCY五种边关系；2) TaskGraph中为简化树形结构，仅保留HIERARCHY维度的目录包含关系（D0-D4层级），缺少CALLS/INHERITS等依赖边。

### 2. 构建配置

#### [记忆] webview-ui构建配置
- **分类**: project_build_configuration
- **关键词**: webview-ui, npm, 构建
- **内容**: 项目前端UI（webview-ui）使用npm进行构建，需先安装其依赖，然后执行构建命令。

#### [记忆] npm构建脚本配置
- **分类**: project_build_configuration
- **关键词**: npm, 构建脚本, webview, proto
- **内容**: 项目使用npm脚本进行构建，包括'build:webview'用于构建前端UI，'protos'用于生成TypeScript proto文件。

#### [记忆] 主项目esbuild构建配置
- **分类**: project_build_configuration
- **关键词**: esbuild, 主构建, production
- **内容**: 项目主构建使用esbuild.mjs脚本，通过'node esbuild.mjs --production'命令执行生产环境构建。

#### [记忆] VSIX打包配置
- **分类**: project_build_configuration
- **关键词**: VSIX, vsce, 打包
- **内容**: 项目使用vsce工具打包为VSIX扩展包，命令为'npx @vscode/vsce package'，支持--no-dependencies等参数。

#### [记忆] VS Code扩展安装配置
- **分类**: project_build_configuration
- **关键词**: VS Code, 安装, VSIX
- **内容**: 构建完成的VSIX扩展包可通过'code --install-extension'命令安装到VS Code中。

#### [记忆] proto文件生成配置
- **分类**: project_build_configuration
- **关键词**: proto, grpc-tools, protoc
- **内容**: 项目通过运行'npm run protos'命令生成TypeScript proto文件，依赖grpc-tools提供的protoc二进制。

### 3. 环境配置

#### [记忆] Zulong任务与记忆数据存储与迁移方式
- **分类**: project_environment_configuration
- **关键词**: 任务图, 记忆图, 本地存储, 数据迁移
- **内容**: Zulong系统中的任务图和记忆图数据存储在Python后端的本地数据库或文件中，迁移时可通过拷贝对应数据文件至新环境完成。

### 4. 技术栈

#### [记忆] CRG核心数据结构定义
- **分类**: project_tech_stack
- **关键词**: CodeGraph, CodeSymbol, CodeEdge, FileParseResult, dataclass
- **内容**: CRG项目核心数据结构包括：1) CodeGraph（代码图谱主容器），含symbols、edges、file_results等字段；2) CodeSymbol（代码符号），含name、kind、qualified_name、parameters、docstring等11个字段；3) CodeEdge（代码关系边），含source_id、target_id、edge_type；4) FileParseResult（文件解析结果）；5) ImportInfo和CallInfo（导入与调用详情）。所有结构均使用Python dataclass实现。

---

## 二、会话积累的开发知识（未存入 Quest 记忆）

### 1. 完整构建流水线

```bash
# 1. 生成 proto
cd zulong-ide && npm run protos

# 2. 构建前端 webview
cd webview-ui && npm install && npm run build && cd ..

# 3. 构建主扩展
node esbuild.mjs --production

# 4. 打包 VSIX
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license

# 5. 安装到 VS Code
code --install-extension zulong-ide-0.1.0.vsix --force
```

### 2. 项目架构

- **基础框架**: Cline v3.82.0 (VS Code 扩展)
- **前端**: React + Vite (webview-ui/)
- **扩展主体**: TypeScript + esbuild (zulong-ide/src/)
- **后端**: Python FastAPI + WebSocket (zulong/cline/)
- **通信协议**: WebSocket JSON (ws://127.0.0.1:8090/ide)

### 3. 关键文件映射

| 文件 | 作用 |
|------|------|
| `zulong-ide/src/core/api/providers/zulong.ts` | ZulongHandler — WebSocket API 提供者 |
| `zulong-ide/src/core/api/transport/zulong-websocket.ts` | WebSocket 传输层 |
| `zulong-ide/webview-ui/src/components/settings/providers/ZulongProvider.tsx` | 祖龙设置 UI |
| `zulong-ide/webview-ui/src/components/settings/ApiOptions.tsx` | API 提供者选择和条件渲染 |
| `zulong-ide/src/shared/providers/providers.json` | 提供者列表（祖龙在第一位） |
| `zulong/cline/cline_ide_server.py` | Python 后端入口 (FastAPI + WebSocket) |
| `zulong/cline/cline_fc_runner.py` | FC 循环执行器 |
| `zulong/cline/cline_tool_registry.py` | 工具注册和智能分流 |
| `config/zulong_config.yaml` | 全局配置（端口 8090） |

### 4. 中文本地化状态

已完成 6 个批次的中文本地化：
- Batch 1-5: 设置、通用组件、聊天界面等
- Batch 6: 历史记录、MCP 服务器、Kanban 模态框（已完成，~100 个字符串）

已知注意事项：
- **组件函数名不能翻译**：之前误将 `RefreshButton` 改为 `刷新Button` 导致 TS2304 错误
- **多行 JSX 需要精确匹配缩进**：批量替换脚本无法处理跨行 JSX，需手动修复
- `replace_all` 参数适合全局替换但要注意区分 UI 文本和代码标识符

### 5. 已删除的功能

- **Account 页面**: 移除了 Cline 的账户注册页面、Navbar 图标、App.tsx 路由
- **Kanban 模态框**: 移除了 Cline 的 Kanban 推广弹窗及其相关状态/效果
- 现有 4 个导航标签: 聊天 / MCP / 历史 / 设置

### 6. 已知未解决问题

**Provider 选择 Bug**: 在设置中选择 "Zulong (祖龙)" 后，设置面板显示 Anthropic 的配置而非祖龙的 WebSocket URL 配置。
- `ApiOptions.tsx` 中的条件渲染逻辑 (`selectedProvider === "zulong"` → `<ZulongProvider>`) 看起来正确
- 需要进一步调查 `handleProviderChange` 函数和 `selectedProvider` 状态管理流程

### 7. 祖龙系统核心概念

| 概念 | 说明 |
|------|------|
| MemoryGraph | 9 种节点类型 + 7 种边类型，BFS 扩散激活 + FAISS 双路径检索，赫布学习 + 艾宾浩斯衰减 |
| TaskGraph | 树形任务分解，依赖追踪，中断冻结与恢复 |
| AttentionWindow | GLOBAL / FOCUS / SINGLE_CHAIN 三种模式注意力窗口 |
| CircuitBreaker | 6 信号熔断器（重复/循环/增益递减/上下文压力/时间/空转） |
| Focus Chain | Cline 原生功能，与祖龙的 AttentionWindow/MemoryGraph 不冲突 |
| 工具智能分流 | 远程工具(IDE端执行) vs 内部工具(Python后端执行) |

---

## 三、本地会话文件位置

会话 JSONL 文件（完整对话记录）位于：
```
C:\Users\HiWin11\AppData\Roaming\Qoder\SharedClientCache\cli\projects\d--AI-project-zulong-beta4\
```

包含约 10 个 session 文件（`task-*.session.execution.jsonl`）。

---

## 四、迁移操作指南

在新 Qoder 账号中打开此项目后：

1. 让 Qoder 读取本文件并导入记忆：
   ```
   请读取 docs/Qoder_Quest_Memory_Export.md，将其中"一、Quest 存储的记忆"部分的每条记忆
   使用 update_memory 工具逐条重建到 Quest 中（注意保留分类和关键词）
   ```

2. AGENTS.md 会被自动读取，无需手动操作

3. 补充项目理解：
   ```
   请读取 docs/Zulong_IDE使用指南.md 了解项目架构
   ```

### 关于 update_memory 工具

`update_memory` 是 Quest MCP 服务器提供的工具，在 Qoder CLI 环境中可用。
如果新账号中该工具不可用，替代方案：
- 在对话中逐条复述记忆内容，Qoder 会自动将其纳入对话上下文
- 或直接将本文件内容粘贴到对话中作为背景知识
