# 祖龙记忆 MCP — 未来规划文档

> 版本: v1.0  
> 日期: 2026-04-29  
> 状态: 规划阶段

---

## 1. 战略定位

### 1.1 双轨并行策略

| 产品线 | 定位 | 目标 |
|--------|------|------|
| **祖龙主线** | 机器人/LLM 操作系统 | 长期独立研发，完整 AI 系统 |
| **祖龙记忆 MCP** | 编程 Agent 的跨会话记忆插件 | 快速落地、验证市场、拉取投资 |

### 1.2 核心洞察

当前主流编程 Agent（Claude Code、Qoder、Cursor、Windsurf、Devin）的能力本质：

```
编程 Agent = 强模型 + 工具集 + 系统提示词
```

它们的**共同缺陷**：无跨会话长期记忆。每次新会话从零开始，不记得项目的架构决策、技术选型、踩过的坑。

### 1.3 产品定位

**祖龙记忆 MCP 不是编程 Agent，而是编程 Agent 的记忆层。**

- 不跑 LLM、不做推理 — 纯状态管理服务
- 不替代任何编程 Agent — 增强它们
- 编程 Agent 自己的模型同时作为祖龙的 LLM — 深度融合

### 1.4 竞争分化

| 维度 | Claude Code / Qoder / Cursor | 祖龙记忆 MCP 增强后 |
|------|------------------------------|---------------------|
| 编程能力 | 强 | 不变（继承宿主） |
| 单次会话 | 优秀 | 不变 |
| **跨会话记忆** | 无/弱 | **MemoryGraph** |
| **宏观任务管理** | 无（仅 TodoList） | **TaskGraph 有向图** |
| **经验累积** | 无 | **ExperienceStore** |
| **跨工具记忆共享** | 不可能 | **统一 MCP Server** |
| **任务中断恢复** | 弱 | **图状态持久化** |

---

## 2. 技术架构

### 2.1 整体架构

```
┌────────────────────────────────────────────┐
│  编程 Agent（Claude Code / Qoder / Cursor） │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  模型（Claude Sonnet / GPT-4o）       │  │
│  │                                      │  │
│  │  同一个模型同时承担：                   │  │
│  │  - 编程（读写文件、bash、git）         │  │
│  │  - 记忆（调 zulong MCP 管理图记忆）    │  │
│  │  - 规划（调 zulong MCP 管理任务图）    │  │
│  └───────────────┬──────────────────────┘  │
│                  │ tool calls               │
│  ┌───────────────▼──────────────────────┐  │
│  │  工具层                               │  │
│  │  ┌─────────────┐  ┌───────────────┐  │  │
│  │  │ 内置工具     │  │ MCP: 祖龙     │  │  │
│  │  │ - read_file │  │ - memory_*    │  │  │
│  │  │ - edit_file │  │ - task_*      │  │  │
│  │  │ - bash      │  │ - experience_*│  │  │
│  │  │ - glob/grep │  │               │  │  │
│  │  └─────────────┘  └───────┬───────┘  │  │
│  └────────────────────────────┼──────────┘  │
└───────────────────────────────┼─────────────┘
                                │ MCP 协议
                    ┌───────────▼───────────┐
                    │   祖龙记忆 MCP Server  │
                    │                       │
                    │  - MemoryGraph 引擎    │
                    │  - TaskGraph 状态机    │
                    │  - ExperienceStore    │
                    │  - 持久化存储          │
                    │                       │
                    │  （无 LLM、无推理）     │
                    └───────────────────────┘
```

### 2.2 设计原则

1. **零 LLM 依赖** — 所有"智能"由宿主编程 Agent 的模型提供
2. **纯状态服务** — 只做持久化状态管理（增删改查 + 图关系）
3. **协议标准** — 严格遵循 MCP 协议规范
4. **轻量部署** — pip install 即用，无需 Docker/GPU/外部数据库
5. **跨 Agent 通用** — 同一份记忆被多个编程 Agent 共享

### 2.3 记忆完整性保障

记忆创建采用**双层保障机制**：

| 层级 | 机制 | 覆盖率 | 记忆质量 |
|------|------|--------|---------|
| **主动层** | 编程 Agent 模型通过 MCP 工具主动调用 | ~60-80% | 高（有原因、上下文、决策推理） |
| **被动层** | Git Observer 后台进程自动捕获代码变更 | ~100% | 基础（只有"改了什么"） |

主动层提供高质量过程性记忆，被动层作为兜底确保零遗漏。

---

## 3. MCP 工具设计

### 3.1 第一版（MVP）

```yaml
记忆工具:
  zulong_memory_search:
    description: "搜索项目记忆（架构决策、代码规范、踩过的坑、经验教训）"
    params:
      query: string        # 搜索关键词或自然语言描述
      category?: string    # 可选分类过滤
    returns: 相关记忆列表（按相关度排序）

  zulong_memory_write:
    description: "记录重要信息到长期记忆"
    params:
      content: string      # 记忆内容
      type: enum           # "decision" | "lesson" | "pattern" | "todo" | "context"
    returns: 记忆节点 ID

  zulong_memory_list:
    description: "列出最近的记忆条目"
    params:
      limit?: number       # 返回数量（默认 10）
      type?: string        # 按类型过滤
    returns: 最近记忆列表

任务工具:
  zulong_task_view:
    description: "查看当前项目的任务全景（宏观目标、子任务状态、依赖关系）"
    returns: 任务图文本可视化

  zulong_task_create:
    description: "创建新的任务节点"
    params:
      label: string        # 任务描述
      parent_id?: string   # 父任务 ID
      depends_on?: string[] # 依赖的任务 ID 列表
    returns: 任务节点 ID

  zulong_task_update:
    description: "更新任务状态"
    params:
      task_id: string      # 任务 ID
      status: enum         # "pending" | "in_progress" | "completed" | "blocked"
      note?: string        # 状态更新备注
    returns: 更新确认

上下文工具:
  zulong_context_inject:
    description: "一键获取当前项目的关键记忆摘要（用于会话开始时）"
    returns: 项目关键记忆 + 活跃任务 + 近期决策的摘要
```

### 3.2 第二版（差异化增强）

```yaml
经验工具:
  zulong_experience_query:
    description: "查询相关的历史经验"
    params:
      context: string      # 当前工作上下文
    returns: 相关经验列表（含成功/失败案例）

图操作工具:
  zulong_memory_link:
    description: "在两个记忆节点之间建立关联"
    params:
      from_id: string
      to_id: string
      relation: string     # 关系描述
    returns: 关联 ID

  zulong_memory_graph_view:
    description: "查看记忆图的局部结构"
    params:
      center_id?: string   # 中心节点
      depth?: number       # 展开深度
    returns: 图结构可视化

任务增强:
  zulong_task_decompose:
    description: "获取宏观任务的分解建议（基于历史经验）"
    params:
      goal: string         # 宏观目标描述
    returns: 建议的子任务分解方案
```

---

## 4. 项目规则注入

### 4.1 各编程 Agent 的规则文件

| 编程 Agent | 规则文件 | 注入方式 |
|-----------|---------|---------|
| Claude Code | `CLAUDE.md` | 追加记忆使用规范段落 |
| Qoder | `AGENTS.md` | 追加记忆使用规范段落 |
| Cursor | `.cursorrules` | 追加记忆使用规范段落 |
| Windsurf | `.windsurfrules` | 追加记忆使用规范段落 |
| Trae | `.trae/rules` | 追加记忆使用规范段落 |

### 4.2 通用规则模板

```markdown
## 祖龙记忆系统使用规范

你已连接祖龙记忆 MCP Server。请在以下时机使用记忆工具：

### 开始工作时
- 调用 zulong_context_inject() 获取项目关键记忆
- 修改某模块前调用 zulong_memory_search(模块名) 查询相关记忆

### 工作过程中
- 做出架构/技术选型决策时 → zulong_memory_write(content, type="decision")
- 发现坑/注意事项时 → zulong_memory_write(content, type="lesson")
- 识别出可复用的模式时 → zulong_memory_write(content, type="pattern")

### 任务管理
- 收到宏观任务时 → zulong_task_create 分解子任务
- 开始/完成子任务时 → zulong_task_update 更新状态
- 被阻塞时 → zulong_task_update(status="blocked", note="原因")

### 结束工作时
- 有未完成的待办 → zulong_memory_write(content, type="todo")
```

---

## 5. 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| MCP 框架 | `mcp` Python SDK（官方） | 标准协议，原生支持 |
| 语言 | Python 3.10+ | 与祖龙主线一致，便于代码共享 |
| 存储 | SQLite + JSON 文件 | 零外部依赖，单文件数据库 |
| 图引擎 | 简化版内存图 + SQLite 持久化 | 轻量，无需 Neo4j |
| 搜索 | BM25 + sqlite-vss（可选向量） | 关键词搜索为主，向量为辅 |
| 分发 | PyPI (`pip install zulong-memory`) | 最低使用门槛 |
| 配置 | TOML/YAML + CLI | 统一配置管理 |

---

## 6. 代码共享方案

### 6.1 问题

祖龙主线和祖龙记忆 MCP 共享核心记忆逻辑代码。需要保证：
- 两个产品的记忆逻辑一致
- 修复/改进能同步到两边
- 各自可以独立开发非共享部分

### 6.2 方案对比

#### 方案 A：Git Submodule

```
zulong_beta4/                    # 祖龙主线
├── zulong/
│   ├── memory/                  # 主线特有的完整记忆系统
│   └── ...
├── zulong-memory-core/          # ← Git Submodule
│   ├── memory_graph.py
│   ├── task_graph.py
│   ├── experience_store.py
│   └── storage/
└── ...

zulong-memory-mcp/               # 祖龙记忆 MCP 独立仓库
├── zulong-memory-core/          # ← 同一个 Git Submodule
│   ├── memory_graph.py
│   ├── task_graph.py
│   ├── experience_store.py
│   └── storage/
├── server/
│   └── mcp_server.py           # MCP 协议层（MCP 独有）
├── cli/
│   └── zulong_memory           # CLI 工具（MCP 独有）
└── templates/                   # 规则模板（MCP 独有）
```

**优势**：
- 共享代码有独立仓库，版本明确
- 两边可以 pin 到不同版本（主线用最新开发版，MCP 用稳定版）
- Git 原生支持，无额外工具

**劣势**：
- Submodule 操作对团队有学习成本
- 版本同步需要手动 update
- CI/CD 需要配置递归 clone

#### 方案 B：内部 Python Package（Private PyPI / 本地路径）

```
zulong-memory-core/              # 独立仓库，发布为 Python 包
├── src/
│   └── zulong_memory_core/
│       ├── __init__.py
│       ├── memory_graph.py
│       ├── task_graph.py
│       ├── experience_store.py
│       └── storage/
├── pyproject.toml
└── tests/

# 祖龙主线的 requirements.txt:
zulong-memory-core @ git+https://github.com/xxx/zulong-memory-core.git@v1.2.0

# 祖龙记忆 MCP 的 requirements.txt:
zulong-memory-core @ git+https://github.com/xxx/zulong-memory-core.git@v1.2.0
```

**优势**：
- Python 生态标准方式，pip 自动管理
- 版本语义化（semver），依赖清晰
- 两边可以独立声明版本约束
- 可发布到 Private PyPI（团队协作）或直接 git+url

**劣势**：
- 发布流程稍重（需要打 tag/release）
- 开发期间频繁修改时需要 `pip install -e .`
- 跨仓库调试不如 submodule 直观

#### 方案 C：Monorepo + Workspace

```
zulong-workspace/                # 单仓库
├── packages/
│   ├── memory-core/            # 共享核心
│   │   ├── src/zulong_memory_core/
│   │   └── pyproject.toml
│   ├── zulong-main/            # 祖龙主线
│   │   ├── zulong/
│   │   └── pyproject.toml      # depends on memory-core
│   └── zulong-memory-mcp/      # 祖龙记忆 MCP
│       ├── server/
│       └── pyproject.toml      # depends on memory-core
└── pyproject.toml              # workspace root
```

**优势**：
- 原子提交：共享代码改动和使用方同时更新
- 无版本同步问题（同仓库同分支）
- Refactoring 友好（全局搜索替换）
- CI/CD 统一

**劣势**：
- 仓库变大
- 权限控制粒度降低（如果未来需要开源 MCP 但不开源主线）
- 需要 Python workspace 工具支持（如 uv workspace, hatch）

### 6.3 推荐方案

**短期（MVP 阶段）：方案 B — 内部 Python Package**

理由：
- 祖龙主线已是独立仓库，不宜立即重组为 monorepo
- MCP 分支需要独立仓库（将来开源/发布需要）
- Python Package 方式对 pip install 用户最友好
- 开发期间用 `pip install -e ../zulong-memory-core` 链接本地路径

**中期（团队扩大后）：评估是否迁移到方案 C**

如果团队人数 >3，频繁跨仓库改动成为瓶颈，再考虑 Monorepo。

### 6.4 共享代码边界

```python
# zulong-memory-core 包含（纯逻辑，无外部依赖）：
zulong_memory_core/
├── __init__.py
├── memory_graph.py        # 图记忆引擎：节点增删改查、图遍历、关系管理
├── task_graph.py          # 任务图：状态机、依赖管理、状态聚合
├── experience_store.py    # 经验存储：索引、检索、去重
├── search/
│   ├── bm25.py           # BM25 文本搜索
│   └── vector.py         # 向量搜索（可选）
├── storage/
│   ├── base.py           # 存储抽象接口
│   ├── sqlite_backend.py # SQLite 实现
│   └── json_backend.py   # JSON 文件实现（调试用）
├── models.py             # 数据模型（MemoryNode, TaskNode, Experience）
└── utils.py              # 通用工具函数
```

```python
# 祖龙主线独有（不共享）：
zulong/
├── l2/inference_engine.py  # LLM 推理（MCP 版不需要）
├── l2/fc_graph.py          # FC 循环（MCP 版不需要）
├── l1/                     # 视觉/语音/硬件层
├── models/container.py     # 模型容器
└── ...

# 祖龙记忆 MCP 独有（不共享）：
zulong-memory-mcp/
├── server/mcp_server.py    # MCP 协议实现
├── server/git_observer.py  # Git 被动观测器
├── cli/                    # CLI 工具
└── templates/              # 项目规则模板
```

### 6.5 版本同步策略

```
zulong-memory-core 版本规范：
├── MAJOR: 接口不兼容变更（MemoryNode 结构变化等）
├── MINOR: 新增功能（新增搜索算法、新存储后端等）
└── PATCH: Bug 修复

同步规则：
├── 祖龙主线：跟踪 latest（开发版）
├── 祖龙记忆 MCP：pin 到稳定 release 版
└── 发布前：MCP 升级 core 版本 → 完整测试 → 发布
```

---

## 7. 跨编程 Agent 记忆一致性

### 7.1 设计目标

用户上午用 Cursor，下午用 Claude Code，记忆无缝流转：

```
上午 Cursor 会话:
  模型 → zulong_memory_write("auth 模块改为 JWT，refresh token 待实现")
  模型 → zulong_task_create("实现 refresh token 机制")

下午 Claude Code 会话:
  模型 → zulong_context_inject()
       → 返回: "auth 模块已改为 JWT，待实现 refresh token"
  模型 → zulong_memory_search("auth")
       → 返回: 上午 Cursor 中记录的所有 auth 相关记忆
```

### 7.2 实现方式

所有编程 Agent 连接**同一个 zulong-memory MCP Server 实例**：

```
                    ┌──────────────────┐
    Cursor ────────→│                  │
                    │  zulong-memory   │
    Claude Code ──→│  MCP Server      │← 单实例，统一存储
                    │                  │
    Qoder ────────→│  .zulong/        │
                    │  (SQLite 数据库)  │
    Trae ─────────→│                  │
                    └──────────────────┘
```

### 7.3 冲突处理

当多个 Agent 同时连接（极少见但可能）：
- 记忆写入：追加模式，不存在冲突
- 任务更新：Last-Write-Wins + 变更日志
- SQLite WAL 模式支持并发读、串行写

---

## 8. 项目交付物

```
zulong-memory/                   # 独立仓库
├── pyproject.toml              # 项目元数据 + 依赖
├── src/
│   └── zulong_memory/
│       ├── __init__.py
│       ├── server.py           # MCP Server 主入口
│       ├── tools/              # MCP 工具实现
│       │   ├── memory_tools.py
│       │   ├── task_tools.py
│       │   └── experience_tools.py
│       ├── git_observer.py     # Git 被动观测器
│       └── cli.py              # 命令行入口
├── templates/                  # 项目规则模板
│   ├── claude.md.j2
│   ├── agents.md.j2
│   ├── cursorrules.j2
│   ├── windsurfrules.j2
│   └── trae_rules.j2
├── tests/
├── docs/
└── README.md
```

---

## 9. 用户体验流程

```bash
# 1. 安装
pip install zulong-memory

# 2. 初始化项目
cd my-project
zulong-memory init
  → 检测已存在的规则文件（CLAUDE.md / .cursorrules 等）
  → 智能追加记忆使用规范（不覆盖已有内容）
  → 创建 .zulong/ 目录
  → 输出 MCP 配置信息（供用户添加到 Agent 配置）

# 3. 配置编程 Agent 的 MCP 连接
# Claude Code: ~/Library/Application Support/Claude/claude_desktop_config.json
# Qoder: .qoder/mcp.json
# Cursor: .cursor/mcp.json
zulong-memory config --agent claude   # 自动配置
zulong-memory config --agent qoder
zulong-memory config --agent cursor

# 4. 启动（自动后台运行，或由编程 Agent 按需启动）
zulong-memory serve

# 5. 正常使用编程 Agent —— 记忆自动积累

# 6. 查看记忆状态
zulong-memory status              # 记忆节点数、任务进度
zulong-memory search "auth"       # CLI 搜索记忆
zulong-memory tasks               # 查看任务图
zulong-memory export              # 导出记忆（备份/迁移）
```

---

## 10. 投资者价值主张

### 10.1 市场机会

- 编程 Agent 市场 2026 年预计 $50B+
- 所有主流编程 Agent 都缺少跨会话记忆（公认痛点）
- MCP 协议成为行业标准（Anthropic 主导，各厂跟进）

### 10.2 商业壁垒

1. **数据网络效应** — 记忆积累越多，价值越高，用户越难迁走
2. **协议先发** — 率先覆盖主流编程 Agent 的 MCP 适配
3. **图记忆技术** — 祖龙主线积累的图记忆引擎是核心 IP

### 10.3 变现路径

| 层级 | 功能 | 定价 |
|------|------|------|
| 免费版 | 本地存储、单项目、基础记忆 | $0 |
| Pro 版 | 云同步、多项目、高级搜索、经验系统 | $9-19/月 |
| Team 版 | 团队记忆共享、权限管理、审计日志 | $29-49/人/月 |
| Enterprise | 私有部署、自定义集成、SLA | 定制 |

### 10.4 关键指标

MVP 阶段验证：
- 安装量 > 1000（30 天内）
- DAU/MAU > 40%
- 平均每个项目记忆节点 > 50
- NPS > 50

---

## 11. 实施路径

```
Phase 1: MVP（目标：2-3 周）
├── 抽取 zulong-memory-core 独立包
├── 实现 MCP Server（7 个核心工具）
├── CLI 工具（init / serve / status）
├── Claude Code + Qoder 规则模板
├── 基础文档 + README
└── 验证：在真实项目中使用 2 周

Phase 2: 打磨（目标：之后 2-3 周）
├── Git Observer 被动记忆采集
├── Cursor / Windsurf / Trae 适配
├── 向量搜索支持
├── 记忆可视化 CLI
├── 单元测试覆盖 >80%
└── PyPI 正式发布

Phase 3: 增长（目标：之后）
├── 云同步方案
├── 团队记忆共享
├── Web Dashboard
├── 开源社区运营
└── 投资材料准备
```

---

## 12. 与祖龙主线的长期关系

```
祖龙主线 (zulong_beta4)
│
├── 独立完整的机器人/LLM 操作系统
├── 包含视觉、语音、硬件控制层
├── 自带 LLM 推理、FC 循环
├── 使用 zulong-memory-core 作为记忆引擎
│         ↕ 共享
│   zulong-memory-core (独立包)
│         ↕ 共享
└── 祖龙记忆 MCP (zulong-memory)
    ├── 面向开发者的轻量记忆服务
    ├── 无 LLM、无推理
    ├── 快速验证记忆系统的市场价值
    └── 为祖龙主线提供：
        - 用户反馈 → 改进记忆引擎
        - 市场验证 → 投资说服力
        - 收入 → 支撑主线研发
```

**双向反哺**：
- MCP 产品验证记忆架构 → 反馈改进主线记忆引擎
- 主线的图记忆研究成果 → 增强 MCP 产品的技术深度
- MCP 产品的收入 → 支撑主线长期研发

---

## 13. 代码锚点系统（Code Anchor）

### 13.1 核心概念

代码锚点实现**记忆/任务 → 代码片段**的双向追踪：

```
┌─────────────────┐         ┌──────────────────────┐
│  MemoryNode     │         │  CodeAnchor          │
│                 │  1:N    │                      │
│  "auth 改用 JWT" ├────────→│  src/auth/jwt.py     │
│                 │         │  symbol: JWTHandler   │
│                 │         │  lines: 45-78        │
│                 │         │  commit: a3f2b1c     │
└─────────────────┘         └──────────────────────┘

┌─────────────────┐         ┌──────────────────────┐
│  TaskNode       │         │  CodeAnchor          │
│                 │  1:N    │                      │
│  "实现refresh   ├────────→│  src/auth/refresh.py │
│   token机制"    │         │  symbol: RefreshFlow  │
│                 │         │  status: created      │
└─────────────────┘         └──────────────────────┘
```

**双向查询能力：**
- 从记忆/任务 → 找到相关代码："这个决策影响了哪些代码？"
- 从代码 → 找到相关记忆/任务："这段代码为什么这样写？属于哪个任务？"

### 13.2 数据模型

```python
@dataclass
class CodeAnchor:
    """代码锚点 — 将记忆/任务关联到具体代码位置"""
    id: str
    # 定位信息（多层保障，从稳定到不稳定）
    file_path: str              # 相对于项目根的文件路径
    symbol: str | None          # 函数/类/变量名（最稳定）
    line_start: int | None      # 起始行（会随编辑变化）
    line_end: int | None        # 结束行
    # 版本追踪
    commit_sha: str | None      # 关联时的 commit
    content_hash: str | None    # 代码片段的 hash（检测变化）
    # 元数据
    anchor_type: str            # "implementation" | "affected" | "created" | "deleted"
    snippet_preview: str        # 代码片段预览（前2-3行，用于快速确认）
    created_at: datetime
```

### 13.3 锚点稳定性策略

行号随编辑变化，需要多层定位保障：

```
定位优先级（从稳定到不稳定）：
1. symbol（函数名/类名） — 最稳定，重命名时可跟踪
2. content_hash         — 内容不变时精确匹配
3. file_path + line_range — 快速定位，最易过时
4. commit_sha            — 用于回溯历史状态
```

**Git Observer 自动更新锚点**：

```python
class AnchorUpdater:
    """监听 git 变更，自动更新代码锚点的行号和路径"""
    
    def on_commit(self, commit):
        diff = parse_diff(commit)
        for changed_file in diff.files:
            anchors = self.db.find_anchors_by_file(changed_file.path)
            for anchor in anchors:
                if changed_file.renamed:
                    anchor.file_path = changed_file.new_path
                elif anchor.symbol:
                    # 按 symbol 重新定位（最可靠）
                    new_loc = find_symbol(changed_file.new_path, anchor.symbol)
                    if new_loc:
                        anchor.line_start, anchor.line_end = new_loc
                    else:
                        anchor.status = "stale"  # 标记为过期
                else:
                    # 按行偏移调整
                    anchor.line_start += diff.line_offset_at(anchor.line_start)
```

### 13.4 新增 MCP 工具

```yaml
zulong_memory_write_with_code:
  description: "记录记忆并关联到具体代码位置"
  params:
    content: string
    type: string
    code_refs:
      - file: string
        symbol?: string
        lines?: [int, int]
        relation: "implements" | "affects" | "caused_by" | "replaced"

zulong_code_query:
  description: "查询某段代码相关的记忆、任务和经验"
  params:
    file: string
    symbol?: string
    lines?: [int, int]
  returns:
    memories: [...]
    tasks: [...]
    experiences: [...]

zulong_task_link_code:
  description: "将任务关联到实现它的代码"
  params:
    task_id: string
    code_refs:
      - file: string
        symbol?: string
        relation: "created" | "modified" | "tested"
```

### 13.5 数据膨胀与上下文负担分析

#### 数据膨胀评估

**单个 CodeAnchor 的存储开销：**

```
字段             典型大小
─────────────────────────
file_path        ~60 bytes   ("src/modules/auth/jwt_handler.py")
symbol           ~30 bytes   ("JWTHandler.verify_token")
line_start/end   ~8 bytes    (int)
commit_sha       ~40 bytes   (SHA-1)
content_hash     ~32 bytes   (MD5/SHA-256 前缀)
anchor_type      ~15 bytes
snippet_preview  ~120 bytes  (代码前2行)
─────────────────────────
单个锚点合计     ~305 bytes
```

**项目级数据增长预估：**

| 项目规模 | 记忆节点数 | 平均锚点/节点 | 锚点总量 | 额外存储 |
|---------|-----------|-------------|---------|---------|
| 小型（个人项目） | ~200 | 1.5 | ~300 | ~90 KB |
| 中型（团队项目） | ~2,000 | 2.0 | ~4,000 | ~1.2 MB |
| 大型（6个月+） | ~10,000 | 2.5 | ~25,000 | ~7.5 MB |

**结论：存储开销可忽略。** 即使大型项目，锚点数据也不到 10 MB，远小于 Git 仓库本身。SQLite 轻松处理 10 万级记录。

**与纯记忆数据的对比：**
```
不含锚点的记忆节点：~500 bytes（content + metadata）
含锚点的记忆节点：  ~500 + 305×2 = ~1,110 bytes
数据膨胀率：        约 2.2 倍
```

膨胀主要在存储层（SQLite），不在内存层 — 锚点数据按需加载，不常驻内存。

#### LLM 上下文负担分析

**核心问题：锚点数据是否会占满 LLM 的上下文窗口？**

**答案：不会，因为锚点数据不进入 LLM 上下文。**

关键设计原则：

```
LLM 上下文中出现的内容：
✓ 记忆的 content 文本（"auth 改用 JWT，原因是..."）
✓ 任务的 label（"实现 refresh token 机制"）
✓ 经验的 lesson（"改 auth 时注意时钟偏差"）

LLM 上下文中 **不** 出现的内容：
✗ CodeAnchor 的 file_path、line_start、commit_sha 等元数据
✗ snippet_preview（除非用户明确查询）
✗ 锚点更新日志
```

**MCP 工具返回策略 — 分层精简：**

```python
# zulong_memory_search 返回格式（进入 LLM 上下文的部分）
{
    "memories": [
        {
            "content": "auth 模块改为 JWT，需要支持微服务间认证",
            "type": "decision",
            "code_ref_summary": "→ src/auth/jwt_handler.py:JWTHandler"
            # ↑ 只返回一行摘要，不返回完整锚点数据
        }
    ]
}

# zulong_code_query 返回格式（用户明确按代码查询时才展开）
{
    "file": "src/auth/middleware.py",
    "related_memories": [
        {"content": "...", "relation": "affects"},
        {"content": "...", "relation": "caused_by"}
    ],
    "related_tasks": [
        {"label": "...", "status": "completed"}
    ]
}
```

**上下文 token 消耗对比：**

| 查询类型 | 不含锚点 | 含锚点摘要 | 增量 |
|---------|---------|-----------|------|
| memory_search（返回5条） | ~500 tokens | ~600 tokens | +20% |
| code_query（返回相关信息） | N/A | ~400 tokens | 新增场景 |
| context_inject（会话开始） | ~800 tokens | ~900 tokens | +12% |

**结论：上下文负担极小（+12%~20%），完全可控。**

#### 控制策略

即便如此，仍提供以下控制机制防止极端情况：

**1. 锚点摘要策略（Summary Mode）**

```yaml
# 默认：返回记忆时只附带一行代码引用摘要
memory_search 返回:
  content: "auth 改为 JWT"
  code_hint: "→ jwt_handler.py:JWTHandler"   # 一行，~30 tokens

# 仅当用户调用 zulong_code_query 时才展开完整锚点
code_query 返回:
  file: "src/auth/jwt_handler.py"
  lines: [45, 78]
  snippet: "class JWTHandler:\n    def verify_token(self, token)..."
```

**2. 记忆衰减与归档**

```python
# 超过阈值时自动归档旧记忆（锚点随之归档）
MAX_ACTIVE_MEMORIES = 500         # 活跃记忆上限
MAX_ANCHORS_PER_MEMORY = 5       # 单条记忆最多关联5个锚点
STALE_ANCHOR_TTL_DAYS = 90       # 过期锚点90天后清理
```

**3. 按需加载（Lazy Loading）**

```python
class MemoryNode:
    # content 始终加载（轻量）
    content: str
    
    # 锚点按需加载（不在搜索结果中展开）
    @property
    def code_anchors(self) -> list[CodeAnchor]:
        if self._anchors is None:
            self._anchors = self.db.load_anchors(self.id)
        return self._anchors
```

**4. 上下文预算管理**

```python
# MCP Server 内部限制返回给 LLM 的总 token 数
MAX_RESPONSE_TOKENS = 2000  # 单次工具调用返回上限

def format_search_results(memories, token_budget=MAX_RESPONSE_TOKENS):
    results = []
    used_tokens = 0
    for mem in memories:
        # 基础信息：content + type + 时间
        entry_tokens = count_tokens(mem.content) + 20
        # 代码摘要：仅一行 hint
        if mem.code_anchors:
            hint = f"→ {mem.code_anchors[0].file_path}:{mem.code_anchors[0].symbol}"
            entry_tokens += count_tokens(hint)
        
        if used_tokens + entry_tokens > token_budget:
            break  # 超预算则截断
        results.append(format_entry(mem, include_hint=True))
        used_tokens += entry_tokens
    return results
```

### 13.6 竞品对比

| 产品 | 记忆→代码关联 | 代码→记忆反查 | 锚点自动更新 |
|------|:---:|:---:|:---:|
| Mem0 | 无 | 无 | 无 |
| Cognee | 无（图节点是实体，非代码位置） | 无 | 无 |
| agentmemory | 可选附带文件路径（非结构化） | 无 | 无 |
| Neo4j Agent Memory | 无 | 无 | 无 |
| codebase-memory-mcp | 代码语义缓存（无记忆关联） | 无 | 无 |
| **祖龙记忆 MCP** | **结构化锚点（symbol+行号+commit）** | **zulong_code_query** | **Git Observer 自动维护** |

### 13.7 独特体验场景

**1. 代码考古 — "这段代码为什么这样写？"**
```
Agent 调用: zulong_code_query(file="src/auth/jwt.py", symbol="JWTHandler")
返回: 决策记忆 → "选择 PyJWT 而非 python-jose，原因是 jose 依赖编译困难"
```

**2. 智能提醒 — "改这里要注意什么"**
```
Agent 准备修改 auth/middleware.py
Agent 调用: zulong_code_query(file="src/auth/middleware.py")
返回: 经验教训 → "token 校验需处理时钟偏差（leeway=30s），之前出过 bug"
```

**3. 影响分析 — "改这个函数会影响什么？"**
```
Agent 调用: zulong_code_query(file="src/core/config.py", symbol="load_config")
返回: 
  - 任务关联: "配置重构" (in_progress)
  - 记忆关联: "config 改为 YAML 后，5个模块的导入方式都要改"(affects 5 files)
```

**4. 任务进度可视化 — 代码级粒度**
```
Agent 调用: zulong_task_view()
返回:
  任务: "实现用户认证系统"
  ├── [completed] JWT 签发 → jwt_handler.py:JWTHandler ✓
  ├── [completed] 中间件集成 → middleware.py:auth_middleware ✓
  ├── [in_progress] Refresh Token → refresh.py (50% 实现)
  └── [pending] Rate Limiting → (无代码关联)
```

---

## 附录 A：MCP Server 配置示例

### Claude Code

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "zulong-memory": {
      "command": "zulong-memory",
      "args": ["serve", "--stdio"],
      "env": {
        "ZULONG_PROJECT": "/path/to/project"
      }
    }
  }
}
```

### Qoder

```json
// .qoder/mcp.json
{
  "servers": {
    "zulong-memory": {
      "command": "zulong-memory",
      "args": ["serve", "--stdio"]
    }
  }
}
```

### Cursor

```json
// .cursor/mcp.json
{
  "mcpServers": {
    "zulong-memory": {
      "command": "zulong-memory",
      "args": ["serve", "--stdio"]
    }
  }
}
```

---

## 附录 B：核心数据模型

```python
@dataclass
class CodeAnchor:
    """代码锚点 — 将记忆/任务关联到具体代码位置"""
    id: str                    # UUID
    file_path: str             # 相对于项目根的文件路径
    symbol: str | None         # 函数/类/变量名（最稳定的标识符）
    line_start: int | None     # 起始行
    line_end: int | None       # 结束行
    commit_sha: str | None     # 关联时的 commit
    content_hash: str | None   # 代码片段 hash（检测内容变化）
    anchor_type: str           # implementation / affected / created / deleted
    snippet_preview: str       # 代码预览（前2-3行）
    created_at: datetime

@dataclass
class MemoryNode:
    id: str                    # UUID
    content: str               # 记忆内容
    type: str                  # decision / lesson / pattern / todo / context
    created_at: datetime       # 创建时间
    source_agent: str          # 来源 Agent（claude-code / qoder / cursor）
    project: str               # 所属项目
    tags: list[str]            # 标签
    links: list[str]           # 关联的其他节点 ID
    code_anchors: list[CodeAnchor]  # 关联的代码锚点

@dataclass
class TaskNode:
    id: str                    # UUID
    label: str                 # 任务描述
    status: str                # pending / in_progress / completed / blocked
    parent_id: str | None      # 父任务
    depends_on: list[str]      # 依赖任务
    created_at: datetime
    updated_at: datetime
    notes: list[str]           # 状态变更备注
    code_anchors: list[CodeAnchor]  # 关联的代码锚点

@dataclass
class Experience:
    id: str
    context: str               # 触发上下文
    outcome: str               # 结果（成功/失败）
    lesson: str                # 经验教训
    related_memories: list[str] # 关联记忆节点
    code_anchors: list[CodeAnchor]  # 关联的代码锚点
    created_at: datetime
```
