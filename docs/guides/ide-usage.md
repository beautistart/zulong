# Zulong IDE 使用指南

> **版本**: v0.1.0
> **基础框架**: Cline v3.82.0
> **最后更新**: 2026-05-03

---

## 1. 概述

Zulong IDE 是一个 VS Code 插件，作为祖龙多层自适应智能体系统的前端交互界面。它既可以作为独立的 AI 编程助手使用（直接对接 LLM API），也可以连接祖龙 Python 后端，使用完整的多层推理、记忆图谱和任务编排能力。

---

## 2. 安装

### 2.1 从 VSIX 安装

```bash
code --install-extension zulong-ide-0.1.0.vsix --force
```

安装后重新加载 VS Code 窗口（`Ctrl+Shift+P` → "Reload Window"）。

### 2.2 从源码构建

```bash
cd zulong-ide

# 安装依赖
npm install
cd webview-ui && npm install && cd ..

# 构建
npm run build:webview
node esbuild.mjs --production

# 打包 VSIX
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license
```

---

## 3. 界面导航

### 3.1 侧边栏入口

安装后，VS Code 左侧活动栏出现 Zulong IDE 图标，点击打开主面板。

### 3.2 顶部导航栏（4 个图标）

| 图标 | 功能 | 说明 |
|------|------|------|
| **+** | 新建任务 | 清除当前任务，创建新对话 |
| **服务器** | MCP 服务器 | 管理 MCP 工具扩展（市场/远程/配置） |
| **时钟** | 历史记录 | 查看、搜索、筛选、删除过去的任务 |
| **齿轮** | 设置 | API 配置、功能、浏览器、终端、通用设置 |

### 3.3 快捷键

- `Ctrl+Shift+P` → 搜索 "Zulong" 可看到所有可用命令

---

## 4. 两种使用模式

### 4.1 独立模式（直接对接 LLM API）

无需启动 Python 后端，直接使用第三方 LLM：

1. 打开设置 → API 配置
2. 选择提供商（OpenAI / Anthropic / OpenRouter / 其他兼容 API）
3. 填入 API Key 和模型名称
4. 在聊天框中输入任务即可

此模式下使用 Cline 原生的工具调用、聚焦链等功能，不涉及祖龙记忆图谱和多层推理。

### 4.2 祖龙模式（连接 Python 后端）

使用祖龙完整的多层推理引擎、记忆图谱和任务编排：

#### 步骤 1：启动祖龙 Python 后端

```bash
cd d:\AI\project\zulong_beta4

# 方式一
python -m zulong.cline.cline_ide_server

# 方式二
uvicorn zulong.cline.cline_ide_server:app --host 127.0.0.1 --port 8090
```

验证后端是否就绪：
```bash
curl http://127.0.0.1:8090/health
# 应返回: {"status":"ok","active_sessions":0,"engine_ready":true}
```

#### 步骤 2：配置 IDE

1. 打开设置 → API 配置
2. 选择提供商为 **Zulong**
3. 服务器 URL 默认为 `ws://127.0.0.1:8090`（可自定义）

#### 步骤 3：正常使用

发送任务后，IDE 通过 WebSocket 连接到 Python 后端，祖龙会启动完整的 FC 循环。

---

## 5. 集成架构

```
┌──────────────────────────────┐      WebSocket (JSON)       ┌──────────────────────────────┐
│    Zulong IDE (VS Code)      │  ◄═══════════════════════►  │    祖龙 Python 后端           │
│                              │   ws://127.0.0.1:8090/ide   │                              │
│  用户界面 & 远程工具执行       │                             │  L2 推理引擎核心              │
│  ├─ 文件读写                  │                             │  ├─ InferenceEngine          │
│  ├─ 终端命令执行              │                             │  ├─ MemoryGraph (记忆图谱)    │
│  ├─ 浏览器操作                │                             │  ├─ TaskGraph (任务编排)       │
│  └─ MCP 工具调用              │                             │  ├─ AttentionWindow (注意力)   │
│                              │                             │  ├─ CircuitBreaker (熔断器)    │
│  核心文件:                    │                             │  └─ RuleGuardian (规则守卫)    │
│  ├─ zulong-websocket.ts      │                             │                              │
│  ├─ zulong.ts (Handler)      │                             │  核心文件:                    │
│  └─ ZulongProvider.tsx       │                             │  ├─ cline_ide_server.py      │
│                              │                             │  ├─ cline_fc_runner.py       │
│                              │                             │  └─ cline_tool_registry.py   │
└──────────────────────────────┘                             └──────────────────────────────┘
```

### 5.1 工具智能分流

| 工具类型 | 执行位置 | 示例 |
|---------|---------|------|
| **远程工具** | IDE 侧 | read_file, write_to_file, execute_command, browser_action |
| **内部工具** | Python 后端 | recall_memory, read_memory_node, save_memory_note, discover_related |

### 5.2 消息协议

**IDE → 后端:**
- `session_start` — 启动新任务
- `session_resume` — 恢复暂停的任务
- `tool_result` — 返回工具执行结果
- `user_cancel` — 取消正在运行的任务

**后端 → IDE:**
- `tool_request` — 请求 IDE 执行远程工具
- `display_text` — 流式输出文本
- `display_reasoning` — 流式输出推理过程
- `task_complete` — 任务完成
- `task_error` — 任务失败
- `status_update` — 进度状态更新
- `session_ack` — 确认连接

---

## 6. 功能设置说明

### 6.1 核心功能

| 设置项 | 说明 |
|--------|------|
| 计划模式 / 执行模式 | 可为规划和执行分配不同的 LLM 模型 |
| 子智能体 | 允许创建子任务并行处理 |
| 原生工具调用 | 使用 LLM 原生的 function calling |
| 并行工具调用 | 允许同时执行多个工具 |
| 严格计划模式 | 计划模式下不可执行工具 |

### 6.2 编辑器功能

| 设置项 | 说明 |
|--------|------|
| 自动压缩 | 对话过长时自动压缩历史 |
| 聚焦链 | 跨交互维护任务进度清单（提醒间隔 1-10） |
| 功能提示 | 在对话中显示功能提示 |
| 后台编辑 | 允许后台编辑文件 |
| 检查点 | 操作回滚点 |

### 6.3 高级功能

| 设置项 | 说明 |
|--------|------|
| 极速模式 | 全自动执行，跳过人工审批（谨慎使用） |
| 摸鱼模式 | 空闲时自动执行任务 |
| 完成双重检查 | 完成任务前进行二次验证 |
| 工作树 | 使用 Git worktree 隔离修改 |

### 6.4 自动批准

聊天框下方有自动批准控制条，可按操作类型分别开启：
- 读取项目文件
- 编辑项目文件
- 执行安全命令
- 使用浏览器
- 使用 MCP 服务器
- 启用通知

---

## 7. MCP 服务器

MCP (Model Context Protocol) 允许扩展智能体的工具能力：

- **市场** — 浏览和安装社区 MCP 服务器
- **远程服务器** — 添加远程 MCP 服务器（Streamable HTTP / SSE）
- **配置** — 管理已安装的 MCP 服务器（启用/禁用/重启/删除/超时设置）

每个 MCP 服务器可以提供：
- **工具** — 可调用的功能（支持单独设置自动批准）
- **资源** — 可访问的数据源
- **提示** — 预定义的提示模板

---

## 8. 历史记录

- 支持模糊搜索
- 排序方式：最新 / 最早 / 最贵 / 最多 Token / 最相关
- 筛选：仅当前工作区 / 仅收藏
- 支持收藏、批量删除、导出
- 日期分组：今天 / 更早

---

## 9. 祖龙模式下的独有能力

连接 Python 后端后，相比独立模式额外获得：

| 能力 | 说明 |
|------|------|
| **MemoryGraph 记忆图谱** | 9 种节点类型 + 7 种边类型，BFS 扩散激活 + FAISS 双路径检索，赫布学习 + 艾宾浩斯衰减 |
| **TaskGraph 任务编排** | 树形任务分解，依赖追踪，中断冻结与恢复 |
| **三模式注意力窗口** | GLOBAL / FOCUS / SINGLE_CHAIN 三种模式，自动 Token 预算管理 |
| **6 信号熔断器** | 重复检测、模式循环、信息增益递减、上下文压力、经过时间、无进度空转 |
| **5 层 FC 保护链** | Rule → Circuit → Retry → Timeout → Quota 五层防护 |
| **信息缺口检测** | 自动识别 NEED_USER_INPUT / NEED_SUBTASK_RESULT |

---

## 10. 故障排查

| 问题 | 解决方案 |
|------|---------|
| 侧边栏空白 | 重新加载 VS Code 窗口 |
| 连接祖龙后端失败 | 确认 Python 后端已启动，访问 `http://127.0.0.1:8090/health` 检查 |
| WebSocket 断连 | IDE 自动重连（最多 5 次，指数退避），也可手动重新发送任务 |
| 工具执行超时 | 在 MCP 设置中调整请求超时时间 |
