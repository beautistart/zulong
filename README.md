<div align="center">

# 🐉 祖龙 (ZULONG)

### 多层自适应智能体认知操作系统

**具有生物学记忆机制的 AI Agent 框架**

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![VS Code](https://img.shields.io/badge/VS%20Code-Extension-green.svg)](https://code.visualstudio.com/)
[![GitHub release](https://img.shields.io/github/release/beautistart/zulong.svg)](https://github.com/beautistart/zulong/releases)
[![Stars](https://img.shields.io/github/stars/beautistart/zulong?style=social)](https://github.com/beautistart/zulong)

[English](./docs/README_EN.md) | 简体中文

</div>

---

## 🎯 一句话介绍

**祖龙是一个具有统一记忆图谱并采用赫布学习、艾宾浩斯衰减等算法，在动态注意力控制的系统下，实现了年级别完整记忆的AI代理认知系统，并且在系统层面实现了无限上下文，在消费级硬件（AI MAX 395 128G）上就可以运行完整的记忆+推理+多模态能力等。**

### ✨ 祖龙记忆系统的核心特点

**在深入了解技术之前，先了解你能体验到的差异化能力：**

#### 1. 无限上下文

祖龙通过自主动态注意力机制，让AI突破模型上下文窗口。即使在执行超复杂长程任务它仍然不会丢失全局注意，子任务的执行不会偏离方向。

#### 2. 记忆关联发现

祖龙通过全图记忆节点BFS扩散+依赖关联边发现，实现无关键词，无语义关系，但是有其他关联的记忆发现，模拟人脑的"联想"能力，实现举一反三。

#### 3. 跨年级的完整记忆

祖龙系统的记忆是基于图记忆实现的，所有记忆都存储在记忆节点里，并且连成了一张大网，记忆节点存有完整未压缩的记忆，无论多久的记忆都可以回忆起来。

---

## 📖 祖龙的故事

我是一名室内设计师，用2个月的时间，独立开发了祖龙82000+行代码。

不用感到不可思议，因为设计师的素养就是作为项目的"总工程师"规划蓝图，让各个专业板块的人士去落地蓝图。

我设计了祖龙的架构蓝图，让智能编程IDE帮我实现的代码：

- **千问桌面端**：项目顾问
- **trae**：前期后端代码工程师
- **qoder**：后期项目纠偏+功能板块实现
- **codearts**：后期项目纠偏+代码审查+功能板块实现

---

> **📢 v1.0.0 正式发布（2026-05-12）**
>
> 记忆板块经过重大架构升级后，这次是祖龙的首次正式发布，包含完整的记忆图谱、死循环检测、跨年级完整记忆等核心能力。
>
> **核心更新**：
> - ✅ **MemoryGraph 记忆图谱** - 9节点+7边+赫布学习+艾宾浩斯衰减
> - ✅ **CircuitBreaker 6信号检测** - 信息增益检测等6种信号
> - ✅ **跨年级完整记忆** - 完整状态序列化
> - ✅ **5层防护链** - 基于qwen3.6-27B模型
> - ✅ **完整的 VS Code 扩展+ TTS/ASR 语音交互**
>
> 详见 [CHANGELOG.md](./CHANGELOG.md)

---

## ✨ 核心亮点

### 1. 🔮 统一记忆图谱 (MemoryGraph)

```
记忆不是平面文本，而是相互关联的知识网络
┌─────────────────────────────────────────┐
│  MemoryGraph (NetworkX DiGraph)          │
│  ├─ 9 种节点：TASK/DIALOGUE/KNOWLEDGE...  │
│  ├─ 7 种边：HIERARCHY/SEMANTIC/CAUSAL... │
│  ├─ 赫布学习：共激活边自动增强权重        │
│  ├─ 艾宾浩斯衰减：exp(-age/half_life)    │
│  ├─ 持久化：LMDB + GraphML               │
│  └─ 双路径检索：BFS热遍历 + FAISS冷检索   │
└─────────────────────────────────────────┘
```

**技术优势**：
- ✅ **持久化统一记忆图谱**（LMDB + GraphML 存储）
- ✅ **赫布学习引擎**：共激活计数 ≥ 3 自动创建 ASSOCIATION 边
- ✅ **艾宾浩斯遗忘曲线**：6 级重要度半衰期（TRIVIAL 6h → MUST_REMEMBER ∞）
- ✅ **双路径检索**：热路径 BFS（<50ms）+ 冷路径 FAISS（<200ms）
- ✅ **语义边自动发现**：后台余弦相似度 > 0.7 自动创建 SEMANTIC 边
- ✅ **三维标签系统**：Temperature × Importance × TimeScope 正交组合

### 2. 🛡️ CircuitBreaker 死循环检测器 - **业界最完善**

```python
# 6 信号综合熔断机制
Signal 1: 相同调用重复检测（name + params_hash）
Signal 2: 模式循环检测（工具频次 + Jaccard 相似度）
Signal 3: 信息增益递减检测（result hash 重叠率）
Signal 4: 上下文压力检测（token 估算 / 窗口比）
Signal 5: 经过时间检测（已禁用，步数为主控）
Signal 6: 无进度空转检测（连续信息检索无行动）

# 状态机：GREEN → YELLOW(注入警告) → RED(强制停止)
```

**对比优势**：
| 项目 | 检测方式 |
|------|----------|
| **祖龙** | 6 信号综合熔断 + 信息增益检测 |
| LangChain | max_iterations 硬限制 |
| CrewAI | max_iterations 硬限制 |
| OpenDevin | 时间/步数限制 |

### 3. ⏸️ 跨天级任务挂起/恢复 - **竞品无此能力**

```python
# 完整状态序列化
SuspendableTaskState:
  - messages: List[Dict]              # 对话历史
  - task_graph: TaskGraph             # 任务树快照
  - circuit_breaker_state: Dict       # CB 状态
  - memory_seeds: List[str]           # 记忆种子节点
  - environment_snapshot: Dict        # 工作目录文件状态

# 支持：暂停 → 关机 → 第二天开机 → 恢复继续执行
```

**适用场景**：
- 🤖 24 小时陪伴式机器人
- 📋 超长程项目管理（跨周/跨月）
- 🔄 中断后环境变化自动重评估

### 4. 🧠 两阶段意图分类 + FC 循环

```
Round 1: 意图分类
  └─ 强制 tool_choice → CHAT/COMPLEX/RESUME

Round 2: 场景化执行
  ├─ CHAT: 直接对话
  ├─ COMPLEX: 启动 FC 循环 + TaskGraph 自动规划
  └─ RESUME: 从快照恢复 + 继续执行
```

**配套 5 层防护链**（~500 行）：
1. CB 强制收敛检查
2. RuleGuardian 过早完成拦截
3. InfoGap 信息缺口检测
4. RESUME AutoMark 安全网
5. COMPLEX Backfill 节点回填

### 5. 🎙️ 语音交互能力（TTS + ASR）

```yaml
TTS (Kokoro-82M):
  - 参数：82M
  - 推理：<0.3s (CPU)
  - 音色：zf_xiaoxiao (中文女声)

ASR (SenseVoice-Small):
  - 参数：244M (ONNX INT8 量化)
  - 能力：中/英/日/韩/粤语 + 情感识别 + 事件检测
  - 推理：0.5-1s (5秒音频)

**整体延迟**：3-4s (端到端，云端API调用)
```

---

## 🏗️ 系统架构

### 四层推理模型

```
┌─────────────────────────────────────────────┐
│            L3 专家层 (Expert Layer)           │
│  7 种专家模型池：GENERAL/LOGIC/CREATIVE/...   │
│  热切换 < 10ms                               │
├─────────────────────────────────────────────┤
│         L2 认知层 (Cognitive Layer)           │
│  InferenceEngine (5700+ 行)                  │
│  ├─ 两阶段推理 + FC 循环                      │
│  ├─ MemoryGraph 记忆检索                      │
│  ├─ TaskGraph 任务编排                        │
│  ├─ CircuitBreaker 熔断                       │
│  └─ 注意力窗口三模式                          │
├─────────────────────────────────────────────┤
│        L1-B 调度层 (Scheduler Layer)          │
│  Gatekeeper + AttentionController            │
│  ├─ 事件优先级路由 (CRITICAL>HIGH>NORMAL>LOW)│
│  └─ 中断处理 (冻结→重组→注入)                 │
├─────────────────────────────────────────────┤
│      L1-A/C 感知层 (Perception Layer)         │
│  L1-A: 音频融合 + 自反应机制                  │
│  L1-C: YOLOv10 人体检测 + MediaPipe 姿态      │
├─────────────────────────────────────────────┤
│          L0 设备层 (Device Layer)             │
│  USB 摄像头/麦克风/扬声器驱动                 │
│  运动检测 (帧差分 + 光流法)                   │
└─────────────────────────────────────────────┘
```

### 前后端分离架构

```
┌──────────────────────────────┐
│   VS Code Extension (前端)    │
│   ├─ React + Vite Webview     │
│   ├─ TypeScript + esbuild     │
│   └─ 工具执行 + UI 渲染        │
└──────────┬───────────────────┘
           │ WebSocket
           │ ws://127.0.0.1:8090/ide
           ↓
┌──────────────────────────────┐
│   Python Backend (后端)       │
│   ├─ FastAPI + WebSocket      │
│   ├─ L2 推理引擎              │
│   ├─ MemoryGraph 记忆系统     │
│   └─ TTS/ASR 语音交互         │
└──────────────────────────────┘
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+
- VS Code
- 推荐设备：AI MAX 395 128G（可纯 CPU 运行）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/beautistart/zulong.git
cd zulong

# 2. 安装 Python 后端依赖
python -m venv zulong_env
source zulong_env/bin/activate  # Windows: zulong_env\Scripts\activate
pip install -r requirements.txt

# 3. 安装前端依赖
cd zulong-ide
npm install
cd webview-ui && npm install && cd ..

# 4. 构建 VS Code 扩展
npm run protos                      # 生成 TypeScript proto 文件
node esbuild.mjs --production       # esbuild 打包
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license

# 5. 安装扩展到 VS Code
code --install-extension zulong-ide-0.1.0.vsix --force
```

### 启动服务

祖龙系统统一使用start.py启动（在项目根目录启动终端并运行start.py）：

```bash
# 统一启动脚本（位于项目根目录）
python start.py

# 或使用带传感器模拟模式
python start.py --mock-sensors

# 打开 VS Code，点击祖龙图标启动会话
```

### 配置文件

编辑 `config/zulong_config.yaml`：

```yaml
# LLM 配置
llm:
  backend: "vllm"  # 可选: ollama, lm_studio, openai
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  
# WebSocket 端口
ide_server:
  port: 8090
  host: "127.0.0.1"
  
# 语音配置
audio:
  tts:
    backend: kokoro
    voice: zf_xiaoxiao
  asr:
    backend: sensevoice
    language: zh
```

---

## 🆚 竞品对比

| 维度 | 祖龙 | LangChain | CrewAI | MemGPT/Letta | AutoGPT |
|------|------|-----------|--------|--------------|---------|
| **统一记忆图谱** | ✅ LMDB + GraphML | ❌ 内存 DAG | ❌ | ❌ 单路向量 | ❌ 基于文件 |
| **赫布学习** | ✅ 共激活增强 | ❌ | ❌ | ❌ | ❌ |
| **艾宾浩斯衰减** | ✅ exp 衰减 | ❌ | ❌ | ❌ | ❌ 基于年龄 |
| **双路径检索** | ✅ BFS + FAISS | ❌ | ❌ | ❌ 单路 | ❌ 单路 |
| **死循环检测** | ✅ 6 信号熔断 | ❌ 硬限制 | ❌ 硬限制 | ❌ | ❌ 硬限制 |
| **任务挂起/恢复** | ✅ 跨天级 | ❌ | ❌ | ❌ | ❌ |
| **语音交互** | ✅ TTS + ASR | ❌ | ❌ | ❌ | ❌ |

---

## 📂 项目结构

```
zulong_beta4/
├── zulong-ide/                 # VS Code 扩展前端
│   ├── src/                    # TypeScript 源码
│   │   ├── core/api/           # API 提供者
│   │   └── webview-ui/         # React Webview
│   └── package.json
├── zulong/                     # Python 后端
│   ├── ide/                    # IDE 模式专用
│   │   ├── ide_server.py       # WebSocket 服务端
│   │   ├── ide_fc_runner.py    # FC 循环执行器
│   │   └── ide_tool_registry.py
│   ├── l2/                     # L2 推理引擎核心
│   │   ├── inference_engine.py
│   │   ├── fc_graph.py
│   │   ├── circuit_breaker.py
│   │   ├── task_graph.py
│   │   └── attention_window.py
│   ├── memory/                 # 记忆系统
│   │   ├── memory_graph.py     # 统一记忆图谱 (LMDB + GraphML)
│   │   ├── rag_manager.py
│   │   └── episodic_memory.py
│   ├── l1a/                    # L1-A 音频融合层
│   ├── l1b/                    # L1-B 事件调度层
│   ├── l1c/                    # L1-C 视觉感知层
│   ├── l3/                     # L3 多专家模型层
│   └── mcp/                    # MCP 协议支持
├── config/                     # 配置文件
│   └── zulong_config.yaml
├── docs/                       # 文档
│   ├── TSD_v2.4.md             # 技术规格说明书
│   ├── 祖龙系统深度技术分析报告.md
│   └── Zulong_IDE使用指南.md
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

---

## 🔧 核心模块说明

### 1. MemoryGraph 统一记忆图谱

**文件**: `zulong/memory/memory_graph.py` (2784 行)

**关键方法**:
```python
# 双路径检索
retrieve_context(query, session_id, top_k=10) -> List[MemoryNode]

# 赫布学习
hebbian_strengthen(node_a, node_b) -> None

# 艾宾浩斯衰减
decay_and_prune() -> None

# BFS 扩散激活
compute_activations(seed_nodes, max_depth=3) -> Dict[str, float]
```

### 2. CircuitBreaker 死循环检测器

**文件**: `zulong/l2/circuit_breaker.py` (800+ 行)

**关键属性**:
```python
# 6 个检测信号
signal_repeated_call: bool      # 相同调用重复
signal_pattern_loop: bool       # 模式循环
signal_info_gain_decay: bool    # 信息增益递减
signal_context_pressure: bool   # 上下文压力
signal_time_elapsed: bool       # 经过时间
signal_no_progress: bool        # 无进度空转

# 状态机
state: "GREEN" | "YELLOW" | "RED"
```

### 3. TaskGraph 任务图谱

**文件**: `zulong/l2/task_graph.py` (1500+ 行)

**关键能力**:
- 无限深度递归树
- 模板节点 + 动态生成节点
- 叶子节点执行 + 非叶子节点状态聚合
- 任务依赖管理

### 4. InferenceEngine 推理引擎

**文件**: `zulong/l2/inference_engine.py` (5700+ 行)

**关键流程**:
1. 意图分类（CHAT/COMPLEX/RESUME）
2. 记忆检索（MemoryGraph + RAG）
3. 注意力窗口裁剪
4. FC 循环执行
5. 工具调用分流
6. 5 层防护链评估

---

## 🛠️ 工具系统

### 内部工具（后端直接执行）

| 工具名 | 职责 |
|--------|------|
| `task_create_plan` | 创建任务计划树 |
| `task_add_node` | 添加任务节点 |
| `task_mark_status` | 标记任务状态 |
| `recall_memory` | 记忆检索 |
| `read_memory_node` | 读取记忆节点 |
| `save_memory_note` | 保存记忆笔记 |
| `discover_related` | 发现关联节点 |
| `focus_on_chain` | 切换注意力焦点 |

### 远程工具（前端执行）

| 工具名 | 职责 |
|--------|------|
| `read_file` | 读取文件 |
| `write_to_file` | 写入文件 |
| `execute_command` | 执行命令 |
| `search_files` | 搜索文件 |
| `browser_action` | 浏览器操作 |

---

## 📖 MCP 协议支持

祖龙提供独立的 MCP Server，可在其他 IDE 中使用祖龙记忆能力：

```python
# mcp_server.py
Server("zulong-memory")

# 7 个 MCP 工具
- zulong_memory_search    # 项目级记忆搜索
- zulong_memory_save      # 保存项目记忆
- zulong_task_search      # 历史任务搜索
- zulong_experience_search # 经验库搜索
- zulong_knowledge_query  # 知识库查询
- zulong_graph_query      # 记忆图谱查询
- zulong_entity_link      # 实体关联
```

---

## 📝 文档

**快速导航**: [文档索引页](./docs/index.md) - 快速查找你需要的文档

### 技术文档
- [技术规格说明书 (TSD)](./docs/architecture/technical-spec-v3.md) - 完整系统架构设计
- [深度技术分析报告](./docs/architecture/system-overview.md) - 代码审查与竞品对比
- [异构图记忆系统详解](./docs/memory_graph/) - MemoryGraph 设计与实现
- [熔断器设计文档](./docs/CircuitBreaker_Design.md) - 6信号死循环检测机制

### 使用指南
- [IDE 使用指南](./docs/Zulong_IDE使用指南.md) - 用户操作手册
- [快速启动指南](./docs/guides/quick-start.md) - 3步安装与启动
- [配置指南](./docs/guides/configuration.md) - 系统配置说明
- [Docker 部署指南](./docs/guides/docker-deployment.md) - 容器化部署
- [文档索引](./docs/index.md) - 完整文档导航

### 开发文档
- [贡献指南](./CONTRIBUTING.md) - 如何贡献代码
- [更新日志](./CHANGELOG.md) - 版本更新记录

---

## 🤝 贡献指南

欢迎贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解详情。

### 开发路线图

- [ ] 扩展 MCP 工具集（7 → 30+）
- [ ] 添加基准测试数据
- [ ] TaskGraph UI 可视化
- [ ] 多 Agent 协作支持
- [ ] 性能优化（关键路径 Rust/Cython 重写）

---

## 📄 许可证

本仓库采用分层许可：

- **核心代码** (`zulong/`): AGPL-3.0
- **VS Code 扩展** (`zulong-ide/`): MIT
- **文档** (`docs/`): CC BY-NC-SA 4.0

详见 [LICENSE](./LICENSE) 文件。

---

## 👨‍💻 作者

**一个室内设计师，用 AI 助手在 2 个月内独立开发了祖龙系统**

- 知乎: [@zhiaoimn](https://www.zhihu.com/people/zhiaoimn)
- GitHub: [@beautistart](https://github.com/beautistart)

---

##  致谢

祖龙系统的开发离不开众多优秀的开源项目与社区贡献。在此向以下项目团队表示诚挚感谢：

### 核心基础框架

- **[Cline](https://github.com/cline/cline)** v3.82.0 - 祖龙 IDE 基于 Cline 框架开发，感谢 Cline 团队的优秀工作
- **[PyTorch](https://github.com/pytorch/pytorch)** - 深度学习框架，提供模型推理与计算核心
- **[FastAPI](https://github.com/fastapi/fastapi)** + **[Uvicorn](https://github.com/encode/uvicorn)** - 高性能异步 Web 服务框架
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - 预训练模型加载与推理

### 记忆系统与向量计算

- **[NetworkX](https://github.com/networkx/networkx)** - 记忆图谱核心图计算引擎
- **[LMDB](https://lmdb.readthedocs.io/)** - 高性能嵌入式键值数据库（图谱持久化）
- **[FAISS](https://github.com/facebookresearch/faiss)** - Facebook AI 向量相似度搜索引擎
- **[Qdrant](https://github.com/qdrant/qdrant)** - 向量数据库与语义检索

### 语音交互能力

- **[FunASR / SenseVoice](https://github.com/modelscope/FunASR)** - 阿里巴巴达摩院开源语音识别引擎（ASR 主引擎）
- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** - 轻量级文本转语音模型（TTS 主引擎，82M 参数）
- **[edge-tts](https://github.com/rany2/edge-tts)** - 微软云端 TTS 备选方案
- **[Whisper](https://github.com/openai/whisper)** - OpenAI 开源多语言语音识别模型（备选 fallback）

### 前端与 UI

- **[React](https://github.com/facebook/react)** - 前端 UI 框架
- **[Tailwind CSS](https://tailwindcss.com/)** - 原子类 CSS 框架
- **[Radix UI](https://www.radix-ui.com/)** - 无头 UI 组件库
- **[Vite](https://vitejs.dev/)** - 前端构建工具
- **[esbuild](https://esbuild.github.io/)** - 极速 JavaScript/TypeScript 打包器

### 开发工具与基础设施

- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** - 模型上下文协议 SDK
- **[OpenTelemetry](https://opentelemetry.io/)** - 可观测性与链路追踪
- **[Playwright](https://playwright.dev/)** - 浏览器自动化测试
- **[Mermaid](https://mermaid.js.org/)** - 图表与流程图渲染

### 模型与权重

- **[Qwen 系列模型](https://github.com/QwenLM/Qwen)** - 阿里巴巴通义千问（祖龙核心推理模型）
- **[CosyVoice3-0.5B](https://github.com/FunAudioLLM/CosyVoice)** - 阿里通义实验室开源 TTS 模型
- **[BAAI BGE 系列](https://github.com/FlagOpen/FlagEmbedding)** - 北京智源研究院文本嵌入模型
- **[MediaPipe](https://google.github.io/mediapipe/)** - Google 开源跨平台机器学习管道（人脸/手部/姿态检测）

### Agent 框架与编排

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - 图式 AI Agent 编排框架
- **[LangChain](https://github.com/langchain-ai/langchain)** - 多 LLM 应用开发框架
- **[VLLM](https://github.com/vllm-project/vllm)** - 高性能 LLM 推理与服务引擎

### 视觉与多模态

- **[OpenCV](https://github.com/opencv/opencv)** - 计算机视觉库（摄像头模块与运动检测）

### 自然语言处理

- **[NLTK](https://www.nltk.org/)** - 自然语言处理工具包
- **[jieba](https://github.com/fxsjy/jieba)** - 中文分词工具

### 数值计算与科学计算

- **[NumPy](https://numpy.org/)** - 数值计算与多维数组
- **[SciPy](https://scipy.org/)** - 科学计算与线性代数

### 安全与解析

- **[PyJWT](https://github.com/jpadilla/pyjwt)** - JSON Web Token 创建与验证
- **[Tree-sitter](https://tree-sitter.github.io/)** - 增量代码解析器生成器

### AI 编程工具

感谢以下 AI 编程工具在祖龙开发过程中的帮助：

- **千问桌面端** - 项目顾问
- **trae** - 前期后端代码工程师
- **qoder** - 后期项目纠偏+功能板块实现
- **codearts** - 后期项目纠偏+代码审查+功能板块实现

---

## ⭐ Star History

如果这个项目对你有帮助，请给一个 ⭐ Star 支持一下！

[![Star History Chart](https://api.star-history.com/svg?repos=beautistart/zulong&type=Date)](https://star-history.com/#beautistart/zulong&Date)

---

<div align="center">

**Made with ❤️ by a Interior Designer turned AI Developer**

**祖龙 - 让 AI 拥有真正的记忆**

</div>
