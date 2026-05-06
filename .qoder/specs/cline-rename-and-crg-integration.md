# Plan: 祖龙系统统一架构升级 (7 大任务合并执行)

## Context

祖龙系统经过深度源码审查（7 个核心模块 10,000+ 行代码），发现以下系统性问题需要统一解决：

1. **Cline 命名残留**: 180+ 处 "cline" 引用分布在 15+ 文件中，系统已完全独立演进，命名有误导性
2. **搜索工具断裂**: `openclaw_search` 依赖不稳定的 OpenClaw API 中间层，SearXNG Docker 已运行但未直连
3. **幽灵工具**: `submit_final_answer` 被 5 个核心模块引用但从未实现；`ask_user` 仅 IDE 可用
4. **L4/L5 硬限制**: `total_fc_turns <= 100` 对超复杂任务不合理，需替换为检查点+进度报告+自动继续（不采用上下文压缩——AttentionWindow 已更高级）
5. **两套 FC 循环**: `fc_graph.py` (LangGraph) 和 `ClineFCRunner` (while-loop) 并行，后者拥有 6 项独有认知能力
6. **任务类型盲区**: `TaskNode` 无 `task_domain` 字段，Execute 阶段工具集固定
7. **代码感知缺失**: 缺少代码结构图谱能力，code-review-graph (MIT) 需内化

本计划合并了 `FC循环统一与通用任务架构分析报告` 中的所有任务与之前的 Cline/SearXNG/CRG 任务，统一规划执行。

## 依赖拓扑

```
P0: A(Cline重命名)
    |
    |---> P1-A: B(SearXNG直连)      -- 并行
    |---> P1-B: C(幽灵工具修复)     -- 并行
    |---> P1-C: E(任务域感知)       -- 并行
    |
    +---> P2: D(L4/L5软限制)        -- 依赖 P0
         |
         +---> P3: F(FC循环统一)    -- 依赖 P0+P1-B+P2
              |
              +---> P4: G(CRG内化)  -- 依赖 P0+P1-C
```

---

## P0: Cline 命名清理 (任务 A) -- 180+ 引用, 15+ 文件

**必须最先执行**: 所有后续任务的 import 路径和类名都取决于此步。

### P0-1: 目录和文件重命名

| 旧路径 | 新路径 |
|--------|--------|
| `zulong/cline/` | `zulong/ide/` |
| `cline_fc_runner.py` | `ide_fc_runner.py` |
| `cline_ide_server.py` | `ide_server.py` |
| `cline_tool_registry.py` | `ide_tool_registry.py` |
| `cline_format_translator.py` | `ide_format_translator.py` |
| `cline_prompt_handler.py` | `ide_prompt_handler.py` |
| `cline_session.py` | `ide_session.py` |
| `skill_packs/packs/cline_coder/` | `skill_packs/packs/ide_coder/` |

### P0-2: 符号映射表

**类名 (9个):**

| 旧名 | 新名 | 文件 |
|------|------|------|
| `ClineFCRunner` | `IDEFCRunner` | `ide_fc_runner.py` |
| `ClineFCResult` | `IDEFCResult` | `ide_fc_runner.py` |
| `ClineFCState` | `IDEFCState` | `ide_session.py` |
| `ClineSession` | `AgentSession` | `ide_session.py` (注: `ide_server.py:57` 已有 `IDESession`) |
| `ClineSessionStore` | `AgentSessionStore` | `ide_session.py` |
| `ClineToolRegistry` | `IDEToolRegistry` | `ide_tool_registry.py` |
| `ClineFormatTranslator` | `IDEFormatTranslator` | `ide_format_translator.py` |
| `ClinePromptHandler` | `IDEPromptHandler` | `ide_prompt_handler.py` |
| `ClineCoderPack` | `IDECoderPack` | `skill_packs/packs/ide_coder/__init__.py` |

**常量 (3个):**

| 旧名 | 新名 |
|------|------|
| `CLINE_REMOTE_TOOLS` | `IDE_REMOTE_TOOLS` |
| `_ZULONG_TOOLS_DISABLED_IN_CLINE_MODE` | `_ZULONG_TOOLS_DISABLED_IN_IDE_MODE` |
| `_CLINE_TOOL_SCHEMAS` | `_IDE_TOOL_SCHEMAS` |

**状态值 + 变量 + 方法:**

| 旧 | 新 |
|------|------|
| `"waiting_cline"` (6处) | `"waiting_remote"` |
| `"pause_for_cline"` (1处) | `"pause_for_remote"` |
| `cline_intent` 字段 | `ide_intent` |
| `cline_session` 局部变量 | `agent_session` |
| `cline_system_prompt` 参数 | `ide_system_prompt` |
| `thread_name_prefix="cline_fc_model"` | `"ide_fc_model"` |
| `_detect_cline_intent()` | `_detect_ide_intent()` |
| `_pause_for_cline()` | `_pause_for_remote()` |
| `parse_cline_tool_results()` | `parse_ide_tool_results()` |
| `extract_cline_tool_block()` | `extract_ide_tool_block()` |

**日志标签 (107+ 处):** `[ClineFCRunner]->[IDEFCRunner]`, `[ClineSessionStore]->[AgentSessionStore]`, etc.

**uvicorn 入口:** `"zulong.cline.cline_ide_server:app"` -> `"zulong.ide.ide_server:app"`

### P0-3: 外部 import 更新 (7处)

| 文件 | 行号 |
|------|------|
| `launcher/modules/ide_server_module.py` | L31 |
| `launcher/web_chat_router.py` | L200, L404, L417, L492 |
| `tools/task_tools.py` | L245 |
| `l2/attention_window.py` | 注释引用 |

所有 `from zulong.cline.cline_*` -> `from zulong.ide.ide_*`

### P0-4: 配置/文档更新

- `config/zulong_config.yaml`: `cline:` section -> `ide:`, `"cline_coder"` -> `"ide_coder"`
- `AGENTS.md`: Critical File Map 中 `zulong/cline/` -> `zulong/ide/`
- `docs/FC循环统一与通用任务架构分析报告.md`: 更新 cline 引用 + 补写 L4/L5 替代方案
- `zulong/skill_packs/packs/__init__.py`: `"cline_coder"` -> `"ide_coder"`

### P0 执行顺序
1. 创建 `zulong/ide/` 目录，复制+重命名 7 个文件
2. 包内应用所有符号重命名 (类名、常量、状态值、变量、方法、日志标签)
3. 更新包内 cross-import (`zulong.cline.X` -> `zulong.ide.X`)
4. 更新 7 处外部 import
5. 迁移 skill_packs/packs/cline_coder/ -> ide_coder/
6. 更新配置和文档
7. 删除 `zulong/cline/` 和 `skill_packs/packs/cline_coder/`

### P0 验证
```bash
python -c "from zulong.ide import IDEFCRunner, IDEToolRegistry, AgentSessionStore"
python -c "from zulong.ide.ide_session import IDEFCState, AgentSession"
python -c "from zulong.ide.ide_server import app, _get_engine"
# grep 确认无 zulong.cline 残留 (排除 __pycache__)
```

---

## P1-A: SearXNG 直连搜索 (任务 B) -- 与 P1-B/P1-C 并行

### 改动范围

| 操作 | 文件 |
|------|------|
| **新建** | `zulong/tools/web_search.py` |
| **修改** | `zulong/tools/tool_engine.py` L487-498 (替换注册) |
| **修改** | `zulong/tools/core_tool_manager.py` L20 (`"openclaw_search"` -> `"web_search"`) |
| **修改** | `zulong/l2/circuit_breaker.py` L50 (删除 `"openclaw_search"`, `"web_search"` 已在) |
| **修改** | `zulong/l2/intent_prompt_builder.py` (工具名引用) |
| **删除** | `zulong/tools/openclaw_search.py` |

### WebSearchTool 设计

```python
class WebSearchTool(BaseTool):
    """直连 SearXNG 的搜索工具"""
    name = "web_search"
    # 直连 SearXNG: GET http://localhost:8101/search?q={query}&format=json&language=zh-CN
    # 参考 openclaw_bridge/api_server.py:159-194 的 _search_with_searxng() 实现
    # action: "search" | "fetch_webpage"
    # fetch_webpage: requests.get(url) + html.parser 提取正文
    # config 路径: tools.web_search.searxng_url (默认 http://localhost:8101)
    # 初始化时 health check SearXNG 可达性
```

### P1-A 验证
```bash
curl "http://localhost:8101/search?q=test&format=json"  # SearXNG 可达
python -c "from zulong.tools.web_search import WebSearchTool; print(WebSearchTool().name)"
```

---

## P1-B: 幽灵工具修复 (任务 C) -- 与 P1-A/P1-C 并行

### submit_final_answer 实现

**已有引用** (5处, 预期此工具存在但从未注册):
- `circuit_breaker.py:61` -- `CB_RETAINED_NAMES` 白名单
- `circuit_breaker.py:74` -- `ACTION_TOOLS` 终结类工具
- `attention_window.py:89` -- `GLOBAL_TRIGGER_TOOLS`
- `attention_window.py:519` -- 特殊处理分支 (调用时清除焦点)
- `task_graph.py:129` -- 注释文档

**设计**: 在 `zulong/tools/task_tools.py` 新增 `SubmitFinalAnswerTool`:
- 参数: `answer: str` (最终回复文本)
- 执行: 将所有 `in_progress` 叶节点标记为 `completed` + 设置终止信号
- FC Runner 集成: `_exec_internal()` 检测到此工具后，设 `state.last_response_content = answer` 并终止循环
- 注册到 `tool_engine.py` 的 `_register_builtin_tools()`

### ask_user 实现

**设计**: 在 `zulong/tools/session_tools.py` (新文件) 新增 `AskUserTool`:
- 参数: `question: str`
- IDE/WebSocket 模式: 推送 `ask_user` 消息到前端，暂停等待回复
- Web/EventBus 模式: 通过 EventBus 发布事件
- 降级: IDE 连接时转发为 `ask_followup_question` 远程调用
- 注册到 `tool_engine.py`

### P1-B 验证
```bash
python -c "from zulong.tools.task_tools import SubmitFinalAnswerTool; print(SubmitFinalAnswerTool().name)"
# 模拟 FC 循环中调用 submit_final_answer -> 循环正确终止
```

---

## P1-C: 任务域感知 (任务 E) -- 与 P1-A/P1-B 并行

### P1-C-1: TaskNode 新增 task_domain 字段

`zulong/l2/task_graph.py:35-99` 的 `TaskNode` dataclass:
```python
task_domain: str = "general"  # "coding" | "general" | "hybrid"
```
- `to_dict()` / `from_dict()` 中序列化/反序列化
- `_sync_node_to_memory_graph()` 中将 `task_domain` 写入 metadata

### P1-C-2: Execute 阶段动态工具集

`orchestrator_graph.py:79-86` 的 `_EXECUTE_TOOLS` 改为按 `task_domain` 动态选择:

```python
_GENERAL_EXECUTE_TOOLS = _EXECUTE_TOOLS | {"web_search", "ask_user", "submit_final_answer"}
_CODING_EXECUTE_TOOLS  = _EXECUTE_TOOLS | {"web_search", "submit_final_answer",
                                           "search_symbols", "get_symbol_context"}
```

`execute_node()` (L311) 中根据当前 subtask 的 `task_domain` 选择工具集。

### P1-C-3: OrchestratorState nesting_depth 追踪

- 新增 `nesting_depth: int`, `max_nesting_depth: int` (默认 5, 可配置)
- `schedule_node()` 中根据 TaskGraph 层级计算

### P1-C-4: 子任务语义指纹去重

`task_tools.py` 的 `TaskAddNodeTool.execute()` 中:
- `SequenceMatcher(label+desc)` 与已存在节点对比，> 0.85 阻止创建

### P1-C 验证
```bash
python -c "from zulong.l2.task_graph import TaskNode; n = TaskNode('t1','test','task','pending','desc',task_domain='coding'); print(n.task_domain)"
```

---

## P2: L4/L5 检查点+进度报告+自动继续 (任务 D)

**核心原则**: 不采用上下文压缩 -- AttentionWindow (三模式+原子淘汰+BFS扩散+MemoryGraph持久化+recall_memory恢复) 已比简单分级压缩更高级。

### P2-1: OrchestratorState 扩展

`orchestrator_graph.py:49-64` 新增:
```python
progress_report_interval: int    # 每 N 步生成报告 (默认 30)
last_report_turn: int            # 上次报告时的 turn 数
progress_reports: List[Dict]     # 报告历史
auto_continue: bool              # 自动继续 (默认 True)
```

### P2-2: schedule_node 软限制改造

`orchestrator_graph.py:262-270` **现有行为**: `total_fc_turns >= max_turns` -> 强制终止

**新行为**:
1. 达到限制时 -> 调用 `_generate_progress_report()` 生成结构化报告
2. 通过 EventBus/WebSocket 推送给用户
3. `auto_continue == True` -> 自动将 `max_total_fc_turns += interval` (弹性预算)
4. `auto_continue == False` -> 进入 synthesize
5. **安全阀**: 连续 3 次报告无进展（内容相同）-> 强制终止
6. **终极硬限制**: `max_total_fc_turns * 3` 作为绝对上限（防止无限执行）

### P2-3: 进度报告函数

新增 `_generate_progress_report(state, tg)`:
```python
{
    "turn": total_fc_turns,
    "completed": [{"id", "label", "result_preview"}],
    "in_progress": [{"id", "label"}],
    "pending": [{"id", "label"}],
    "elapsed_time": ...,
    "fc_turns_used": ...,
}
```

### P2-4: 利用现有机制

- `_save_checkpoint()` (L625-654) 已保存关键状态 -> 扩展加入进度报告字段
- `_pause_for_remote()` (原 `_pause_for_cline`, ide_fc_runner L1598) -> 扩展支持步数阈值触发
- CircuitBreaker 保留作为死循环防护（不是步数限制）

### P2-5: 配置更新

```yaml
# config/zulong_config.yaml
orchestrator:
  progress_report_interval: 30
  auto_continue: true
  max_reports_before_force_stop: 5
```

### P2-6: 更新 FC分析报告文档

将本方案写入 `docs/FC循环统一与通用任务架构分析报告.md` 的 10.5 节，替换原有的五层硬限制方案。明确记录：不采用分级上下文压缩，因为祖龙的 AttentionWindow 已经是更高级的上下文管理方案。

### P2 验证
```bash
# 设置小间隔验证: max_total_fc_turns=10, progress_report_interval=5
# 验证第 5/10 步生成进度报告
# 验证 auto_continue=true 时系统继续; false 时暂停
```

---

## P3: FC 循环统一 (任务 F)

**依赖**: P0 (路径稳定) + P1-B (submit_final_answer 已实现) + P2 (软限制机制)

### P3-1: 创建 UnifiedFCRunner

新文件: `zulong/l2/unified_fc_runner.py`

以 IDEFCRunner (原 ClineFCRunner, 1900+ 行) 为基座，保留 6 项独有认知能力:
1. **语义漂移检测** (SemanticDriftDetector)
2. **对话轮次跟踪** (DialogueAdapter + DIALOGUE 节点)
3. **会话记忆自动保存** (`_auto_save_session_memory()`)
4. **注意力窗口持久化/恢复** (serialize / from_serialized)
5. **安全网组件 per-runner 隔离** (CB/RG 独立实例)
6. **任务自动完成** (`_auto_complete_task()`)

支持 3 种运行模式:
- **同步 HTTP**: `run_or_resume()` -- 原 ClineFCRunner 模式
- **异步 WebSocket**: `run_loop_async()` -- IDE 模式
- **编排器内嵌**: `run_for_orchestrator()` -- 替换 `fc_graph.run_fc_loop()`

### P3-2: 创建 ToolExecutor

新文件: `zulong/l2/tool_executor.py`

统一路由:
- `internal` -> `tool_engine.call_tool()` 直接执行
- `remote` -> WebSocket 推送 (IDE) 或 `exec_tools` 降级 (Web)
- `submit_final_answer` -> 终止路径
- `ask_user` -> 暂停等待

从 IDEFCRunner 的 `_exec_tools()` (L1101-1161) 和 `_exec_tools_async()` (L335-499) 抽取分流逻辑。

### P3-3: 替换调用点

| 文件 | 改动 |
|------|------|
| `zulong/l2/inference_engine.py` | `run_fc_loop()` -> `UnifiedFCRunner.run_for_orchestrator()` |
| `zulong/l2/orchestrator_graph.py` L171/L317/L498/L576 | 4 处 `run_fc_loop()` 调用替换 |
| `zulong/launcher/web_chat_router.py` | `_chat_via_engine()` 改用 UnifiedFCRunner |

### P3-4: 废弃 fc_graph.py

- 保留文件但添加 DeprecationWarning
- `run_fc_loop()` 转发到 UnifiedFCRunner (shim)
- 保留 1 个版本周期后删除

### P3 验证
```bash
python -c "from zulong.l2.unified_fc_runner import UnifiedFCRunner"
# InferenceEngine 通过 UnifiedFCRunner 执行，7 道安全网全部生效
# Orchestrator plan/execute/synthesize 通过 UnifiedFCRunner 执行
# IDE WebSocket 模式正常推送
# fc_graph.run_fc_loop() shim 调用成功
```

### P3 风险
- **2000+ 行迁移**: 逐安全网验证 (漂移检测/RuleGuardian/InfoGap/AutoMark/Backfill/短回复拦截/CB)
- **per-runner 隔离必须保持**: `_attn_window`, `_rule_guardian`, `_circuit_breaker` 是 per-runner 实例
- **编排器兼容**: 需提供 `(response, fc_turn)` 返回格式的适配方法

---

## P4: CRG 内化 (任务 G) -- code-review-graph 核心能力

**依赖**: P0 (路径稳定) + P1-C (task_domain 支持 "coding" 类型)

### P4-1: 新建子包

```
zulong/memory/code_graph/
  __init__.py
  parser.py          -- Tree-sitter AST 解析 (Python/JS/TS)
  graph_builder.py   -- 符号图构建 + SQLite 缓存
  memory_sync.py     -- CodeGraphAdapter (BaseGraphAdapter 实现)

zulong/tools/code_graph_tools.py  -- FC 工具
```

### P4-2: CRG -> MemoryGraph 映射

**节点映射:**

| 代码符号 | NodeType | Importance | metadata |
|----------|----------|------------|----------|
| 文件 | `FILE` | `NORMAL` | `{path, language, line_count}` |
| 公开函数/方法 | `KNOWLEDGE` | `IMPORTANT` | `{signature, line_start, line_end}` |
| 类 | `KNOWLEDGE` | `IMPORTANT` | `{signature, method_count}` |
| 接口/协议 | `CONCEPT` | `IMPORTANT` | `{signature, implementations}` |

**边映射:**

| 关系 | EdgeType | Protected? |
|------|----------|------------|
| imports | `DEPENDENCY` | Yes |
| inherits | `HIERARCHY` | Yes |
| calls | `REFERENCE` | No |
| contains | `HIERARCHY` | Yes |

**node_id 格式**: `code:{file_path}#{symbol_name}`

### P4-3: FC 工具 (5个 action)

| action | 功能 |
|--------|------|
| `index_project` | 全量解析 -> 符号图 -> 同步 MemoryGraph |
| `index_file` | 增量解析单文件 |
| `search_symbols` | 按名称/类型搜索 (走 MemoryGraph) |
| `get_symbol_context` | 调用链 + 依赖关系 |
| `get_impact_analysis` | BFS 扩散影响分析 |

### P4-4: 依赖

- `tree-sitter` + `tree-sitter-python` + `tree-sitter-javascript` + `tree-sitter-typescript`
- `sqlite3` (stdlib) -- 文件 hash 变更检测缓存
- 评估 `zulong/memory/code_anchor.py` 和 `zulong/tools/code_anchor_tools.py` 是否可复用/扩展

### P4 验证
```bash
python -c "from zulong.memory.code_graph.parser import CodeParser; p = CodeParser(); print(len(p.parse_file('zulong/memory/memory_graph.py')), 'symbols')"
python -c "from zulong.memory.code_graph.memory_sync import CodeGraphAdapter"
python -c "from zulong.tools.code_graph_tools import CodeGraphTool"
# 对 zulong/ 执行 index_project -> MemoryGraph 出现 FILE/KNOWLEDGE/CONCEPT 节点
```

---

## 跨阶段文件修改矩阵

| 文件 | P0 | P1-A | P1-B | P1-C | P2 | P3 | P4 |
|------|:--:|:----:|:----:|:----:|:--:|:--:|:--:|
| `zulong/ide/ide_fc_runner.py` | R | | W | | W | W | |
| `zulong/l2/orchestrator_graph.py` | | | | W | W | W | |
| `zulong/l2/task_graph.py` | | | | W | | | |
| `zulong/tools/tool_engine.py` | | W | W | | | | W |
| `zulong/tools/core_tool_manager.py` | | W | | | | | |
| `zulong/tools/task_tools.py` | R | | W | W | | | |
| `zulong/l2/circuit_breaker.py` | | W | | | | | |
| `zulong/l2/attention_window.py` | R | | | | | | |
| `zulong/memory/graph_adapters.py` | | | | | | | W |
| `config/zulong_config.yaml` | W | | | | W | | W |
| `zulong/launcher/web_chat_router.py` | R | | | | | W | |
| `zulong/l2/inference_engine.py` | | | | | | W | |
| `AGENTS.md` | W | | | | | | |
| `docs/FC循环统一与通用任务架构分析报告.md` | | | | | W | | |

R = 仅路径/引用更新, W = 功能性修改

---

## 全局验证 (所有阶段完成后)

```bash
# 1. 无 cline 残留
grep -rn "zulong\.cline\|ClineFCRunner\|ClineSession\|ClineToolRegistry" zulong/ --include="*.py" | grep -v __pycache__

# 2. 无 openclaw_search 残留
grep -rn "openclaw_search" zulong/ --include="*.py" | grep -v __pycache__

# 3. 核心 import 链
python -c "
from zulong.ide import IDEFCRunner, IDEToolRegistry, AgentSessionStore
from zulong.ide.ide_session import IDEFCState, AgentSession
from zulong.ide.ide_server import app, _get_engine
from zulong.tools.web_search import WebSearchTool
from zulong.tools.task_tools import SubmitFinalAnswerTool
from zulong.l2.task_graph import TaskNode
from zulong.memory.code_graph.parser import CodeParser
from zulong.memory.code_graph.memory_sync import CodeGraphAdapter
print('All imports OK')
"

# 4. SearXNG 搜索测试
curl "http://localhost:8101/search?q=python&format=json"

# 5. IDE WebSocket 握手
# 启动 IDE server, 验证 ws://127.0.0.1:8090/ide 连接正常

# 6. FC 循环端到端
# InferenceEngine -> UnifiedFCRunner -> 工具执行 -> 安全网生效 -> 正常终止
```
