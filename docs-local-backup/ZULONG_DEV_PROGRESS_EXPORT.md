# Zulong (祖龙) 项目开发进度导出

> 导出时间: 2025-05-06 (最后更新)
> 用途: 账号迁移参考文档
> 配套文档: docs/TASK_确定性任务恢复机制重构.md (待实施的下一个重要任务)

---

## 一、项目概述

Zulong IDE 是基于 Cline v3.82.0 的 VS Code 扩展，作为祖龙多层自适应智能体系统的前端。
后端为 Python FastAPI + WebSocket，核心是多层架构的 FC (Function Calling) 循环系统。

### 关键技术栈
- **前端**: React + Vite (webview-ui/) + TypeScript esbuild 扩展
- **后端**: Python FastAPI, WebSocket (ws://127.0.0.1:8090/ide)
- **LLM**: SiliconFlow 平台, model_id 格式如 `Qwen/Qwen3.6-27B`
- **记忆**: MemoryGraph (NetworkX 图 + 地址索引 + BFS 激活扩散)
- **任务**: TaskGraph (树形结构 + 层次边 + 依赖边)
- **Web 仪表盘**: openclaw_bridge/web/static/index.html (独立 Web 监控面板)

### 配置
- 上下文窗口: 4K ~ 1M tokens 可调 (Web UI 滑块)
- 端口: 8090 (config/zulong_config.yaml)
- 磁盘备份目录: data/graph_backups/

---

## 二、核心架构文件映射

| 文件 | 角色 |
|------|------|
| zulong/ide/ide_server.py | Python 后端入口 (FastAPI + WebSocket) |
| zulong/ide/ide_fc_runner.py | FC 循环执行器 (while 循环, 工具分流, 意图检测) |
| zulong/ide/ide_tool_registry.py | 工具注册 & 智能路由 |
| zulong/ide/ide_session.py | IDEFCState, AgentSession 定义 |
| zulong/tools/task_tools.py | 任务管理 FC 工具集 (13+ 工具) |
| zulong/tools/tool_engine.py | ToolEngine 工具引擎 (统一注册) |
| zulong/memory/memory_graph.py | MemoryGraph 核心 (节点/边/BFS/注意力) |
| zulong/memory/graph_adapters.py | 适配器 (TaskGraphAdapter, DialogueAdapter, KnowledgeGraphAdapter) |
| zulong/l2/task_graph.py | TaskGraph 数据结构 |
| zulong/l2/attention_window.py | 注意力窗口管理 (消息评分/裁剪) |
| zulong/l2/circuit_breaker.py | CircuitBreaker 安全网 |
| zulong/l2/rule_guardian.py | 规则守护 |
| zulong/launcher/web_chat_router.py | Web 聊天路由 (WebSocket) |
| openclaw_bridge/web/static/index.html | Web 仪表盘前端 |
| zulong-ide/src/core/api/providers/zulong.ts | IDE 扩展 WebSocket API provider |
| zulong-ide/src/core/api/transport/zulong-websocket.ts | WebSocket 传输层 |

---

## 三、已完成的修复与功能 (按时间)

### 第一批 (早期 session, 已合入)
1. 机器人图标删除修复
2. 不相关回复问题修复
3. model_config 持久化修复
4. 损坏的 model_id 修复
5. Checkpoint 超时修复
6. XML 清理
7. 上下文窗口控制
8. CircuitBreaker 修复
9. 规划模式修复
10. 模式检测豁免
11. 移除 _ultimate_hard_limit
12. 进度报告机制修复
13. 进度报告后继续检查
14. 注意力窗口注册 bug 修复
15. group_id=None 驱逐 bug 修复
16. 任务恢复
17. 取消机制
18. 周期性进度报告

### 第二批 (本 session)

#### 2.1 PROGRESS_REPORT 前端处理器
- **文件**: openclaw_bridge/web/static/index.html
- **内容**: 添加 `handleProgressReport()` 函数 + WebSocket case

#### 2.2 _call_model 备用失败回退修复
- **文件**: zulong/ide/ide_fc_runner.py
- **问题**: 主+备 API 都失败时返回 fallback 文本，evaluator 误判为"完成"
- **修复**: 返回 `(None, None)` 触发 api_error 终结

#### 2.3 Web 停止链修复
- **文件**: zulong/launcher/web_chat_router.py
- **问题**: STOP_GENERATION 通过 event_bus 发布但无法到达 IDE FC runner 的 cancel_event
- **修复**: `_handle_stop_generation` 直接设置 `sess.cancel_event.set()` + `_engine_instance._interrupt_flag`
- **前端**: `stopGeneration()` 同时发送 STOP_GENERATION 和 STOP_TASK

#### 2.4 右键删除对话消息 (前端+后端)
- **后端 API**: `DELETE /api/chat/sessions/{session_id}/messages/{message_id}`
- **前端**: #msgContextMenu + showMsgContextMenu() + deleteChatMessage()
- **Bug 修复**: addMessage 新增 existingId 参数, 恢复历史消息时传入 msg.id, deleteChatMessage 增加 DOM 索引回退

#### 2.5 删除任务图谱 (前端+后端)
- **后端 API**: `DELETE /api/task-graph/{graph_id}`
- **功能**: 清除活跃图 + 删除磁盘备份 + 广播 TASK_GRAPH_DELETED
- **前端**: deleteGraph() 调用后端 API, handleTaskGraphDeleted() 处理器

#### 2.6 MemoryGraph 地址遍历恢复任务 (4 Phase)
- **Phase 1**: `rebuild_task_graph_from_memory(mg, graph_id)` in graph_adapters.py
  - 从 MemoryGraph TASK 节点 + HIERARCHY/DEPENDENCY 边重建 TaskGraph
- **Phase 2**: `_try_activate_from_reference(user_input)` in ide_fc_runner.py
  - 正则匹配 @[label#tg:xxx/task:yyy] → resolve_address → rebuild → activate
  - _detect_ide_intent 规则 1.5 调用
- **Phase 3**: `TaskResumeByAddressTool` in task_tools.py
  - LLM FC 工具, 参数 address, 从 MemoryGraph 恢复历史图谱
- **Phase 4**: `TaskReviseNodeTool` in task_tools.py
  - 将 completed/skipped 节点重置为 in_progress, 支持已完成节点修订
- **工具注册**: tool_engine.py 中注册 TaskResumeByAddressTool + TaskReviseNodeTool

#### 2.7 IDE Resume 新建图谱问题修复 (系统性)
- **根因**: 
  1. _active_task_graph 是内存变量, 进程重启丢失
  2. _detect_ide_intent Rule 2 要求 has_active_tg=True 才触发 RESUME
  3. _auto_create_task_plan 对"已全部完成"的图强制新建
  4. bind_session_to_task() 从未在 IDE 流程中被调用
- **修复**:
  - ide_server.py: _handle_session_resume 启动前调用 load_latest_backup() 恢复
  - ide_fc_runner.py Rule 2: 无活跃 TG 时调用 load_latest_backup() (含已完成图)
  - ide_fc_runner.py _auto_create_task_plan: is_resume=True 时已完成图也复用
  - ide_fc_runner.py _init_dialogue_tracking: 调用 bind_session_to_task() 绑定 session→task
  - task_tools.py: 新增 load_latest_backup() (不限状态, 跳过单节点骨架图)

---

## 四、环境配置记忆

- **SiliconFlow model_id**: 必须用标准格式 `Qwen/Qwen3.6-27B`, 不能混入 Ollama 格式
- **上下文窗口**: 4K~1M tokens 可调
- **Checkpoints**: 基于 Shadow Git 的代码快照系统, 15秒初始化超时 (Windows 下 globby 扫描慢)

---

## 五、已知架构问题 & 改进方向

### 任务恢复不稳定 (核心痛点)
- **现状**: 启发式规则链 (关键词 + 活跃 TG + 未完成节点) 脆弱
- **建议方向**: 
  1. IDE resume payload 应携带 graph_id (确定性恢复)
  2. TaskGraph 持久化为独立目录 (不依赖 MemoryGraph 重建)
  3. MemoryGraph 只做语义索引层 (找历史任务用), 不参与恢复主路径
  4. 恢复 = load(graph_id) → 设为活跃, 一行代码零启发式

### Provider selection bug
- 在 IDE Settings 中选择 "Zulong (祖龙)" 时可能显示 Anthropic 配置
- 需排查 ApiOptions.tsx 的 handleProviderChange

---

## 六、构建命令

```bash
# Full build (在 zulong-ide/ 目录):
npm run protos
cd webview-ui && npm install && npm run build && cd ..
node esbuild.mjs --production
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license
code --install-extension zulong-ide-0.1.0.vsix --force

# TypeScript 检查:
cd zulong-ide && npx tsc --noEmit

# Python 语法检查:
python -m py_compile zulong/ide/ide_fc_runner.py
```

---

## 七、关键数据结构

### MemoryGraph 节点类型
TASK, DIALOGUE, KNOWLEDGE, EXPERIENCE, EPISODE, FILE, CONCEPT, PERSON, DOCUMENT, CODE_SYMBOL, MODULE

### MemoryGraph 边类型
HIERARCHY (结构), DEPENDENCY (结构), REFERENCE, TEMPORAL (结构), SEMANTIC, CAUSAL, ASSOCIATION

### TaskGraph 节点地址格式
`tg:{graph_id}/task:{node_id}` (如 `tg:tg_1234567890/task:o1`)

### 前端节点引用格式
`@[label#address]` (如 `@[配置nginx#tg:tg_123/task:o1]`)

---

## 八、FC 工具列表 (task_tools.py)

1. task_create_plan - 创建任务规划图
2. task_add_node - 添加节点
3. task_mark_status - 更新状态
4. task_view_overview - 查看概览
5. task_suspend - 挂起任务
6. task_list_suspended - 列出挂起任务
7. task_add_dependency - 添加依赖边
8. task_get_detail - 获取节点详情
9. task_update_node - 更新节点
10. task_remove_node - 删除节点
11. task_update_content - 更新内容
12. task_attach_file - 附加文件
13. submit_final_answer - 提交最终答案
14. **task_resume_by_address** - 通过地址恢复 (新增)
15. **task_revise_node** - 修订已完成节点 (新增)

---

## 九、Web 仪表盘功能

- 实时消息流 (WebSocket)
- 任务图谱可视化面板
- 思考过程面板 (Thinking Steps)
- 右键删除消息 (单条/以下全部)
- 右键删除图谱
- 停止生成按钮 (STOP_GENERATION + STOP_TASK)
- PROGRESS_REPORT 进度展示
- CRG 代码索引完成通知
- 会话管理 (创建/切换/删除)

---

## 十、待实施任务 (下一优先级)

### 10.1 确定性任务恢复机制重构 (HIGH)

**详细文档**: `docs/TASK_确定性任务恢复机制重构.md`

**核心思想**: 用 graph_id 做确定性恢复锚点，MemoryGraph 退为辅助发现层

**现状问题**: 
- 启发式规则链(5+条件AND)极不稳定
- 进程重启/任务完成/新窗口 → 都可能导致恢复失败
- 已做了多处补丁(load_latest_backup/is_resume分支等)但本质是打补丁

**方案**:
```
IDE resume payload 增加 graph_id
  → 后端 _deterministic_load_graph(graph_id):
      Level 1: 内存匹配
      Level 2: 磁盘 data/graph_backups/{graph_id}.json
      Level 3: MemoryGraph rebuild_task_graph_from_memory()
  → 跳过所有启发式规则
  → 强制 intent=resume + force_first_tool
```

**涉及文件**:
1. `zulong-ide/src/core/api/transport/zulong-websocket.ts` — sendSessionResume 加 graphId
2. `zulong-ide/src/core/api/providers/zulong.ts` — 从历史消息提取 graph_id
3. `zulong/ide/ide_server.py` — _handle_session_resume + _deterministic_load_graph
4. `zulong/ide/ide_fc_runner.py` — _init_state 有 graph_id 跳过启发式

### 10.2 IDE Provider Selection Bug (MEDIUM)

选择"Zulong (祖龙)"时可能显示 Anthropic 配置面板。
排查 `zulong-ide/webview-ui/src/components/settings/ApiOptions.tsx` 的 `handleProviderChange`。

---

## 十一、系统运行命令

```bash
# 启动后端 (FastAPI + WebSocket):
cd zulong_beta4
python -m zulong.ide.ide_server

# 或通过 launcher:
python -m zulong.launcher.main

# Web 仪表盘访问:
# http://127.0.0.1:8090/static/index.html
```

---

## 十二、调试技巧

- 所有日志前缀: `[IDEFCRunner]`, `[ZulongIDE]`, `[GraphBackup]`, `[TaskTools]`
- 意图检测日志: `意图检测: RESUME/COMPLEX`
- 图谱恢复日志: `确定性恢复` / `从备份恢复` / `从 MemoryGraph 重建`
- 语法检查: `python -m py_compile zulong/ide/ide_fc_runner.py`
- 导入验证: `python -c "from zulong.tools.task_tools import TaskResumeByAddressTool"`
