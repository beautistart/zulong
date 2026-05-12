<div align="center">

# 🐉 祖龙 (ZULONG)

### 多层自适应智能体认知操作系统

**82K+ 行 Python | 消费级硬件可运行 | 一个设计师用 AI 搓出来的**

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![VS Code](https://img.shields.io/badge/VS%20Code-Extension-green.svg)](https://code.visualstudio.com/)
[![GitHub release](https://img.shields.io/github/release/beautistart/zulong.svg)](https://github.com/beautistart/zulong/releases)
[![Stars](https://img.shields.io/github/stars/beautistart/zulong?style=social)](https://github.com/beautistart/zulong)

[English](./docs/README_EN.md) | 简体中文

</div>

---

## 为何一个室内设计师，敢去造 AI 大脑？

我是一名室内设计师，用 **2 个月的时间**，独立开发了祖龙 **82000+ 行代码**。

不用感到不可思议，因为设计师的素养就是作为项目的"总工程师"规划蓝图，让各个专业板块的人士去落地蓝图。

我设计了祖龙的架构蓝图，让智能编程 IDE 帮我实现的代码：

- **千问桌面端**：项目顾问
- **trae**：前期后端代码工程师
- **qoder**：后期项目纠偏 + 功能板块实现
- **codearts**：后期项目纠偏 + 代码审查 + 功能板块实现

**祖龙是什么？**

> 祖龙是一个具有统一记忆图谱并采用赫布学习、艾宾浩斯衰减等算法，在动态注意力控制的系统下，实现了年级别完整记忆的 AI 代理认知系统，并且在系统层面实现了无限上下文，在消费级硬件（AI MAX 395 128G）上就可以运行完整的记忆 + 推理 + 多模态能力等。

**🎬 视频演示**

<p align="center">祖龙系统三层注意力机制和超复杂项目记忆文件展示</p>

<p align="center">
  <a href="https://youtu.be/-W-WYg_eQz4" target="_blank">
    <img src="https://img.youtube.com/vi/-W-WYg_eQz4/maxresdefault.jpg" alt="1分钟看懂祖龙" width="100%" />
  </a>
</p>

<p align="center"><b>👆 点击图片观看视频</b></p>

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

- **持久化统一记忆图谱**（LMDB + GraphML 存储）
- **赫布学习引擎**：共激活计数 ≥ 3 自动创建 ASSOCIATION 边
- **艾宾浩斯遗忘曲线**：6 级重要度半衰期（TRIVIAL 6h → MUST_REMEMBER ∞）
- **双路径检索**：热路径 BFS（<50ms）+ 冷路径 FAISS（<200ms）
- **语义边自动发现**：后台余弦相似度 > 0.7 自动创建 SEMANTIC 边

### 2. 🛡️ CircuitBreaker 死循环检测器

**6 信号综合熔断机制**：相同调用重复、模式循环、信息增益递减、上下文压力、经过时间、无进度空转

状态机：`GREEN → YELLOW(注入警告) → RED(强制停止)`

| 项目 | 检测方式 |
|------|----------|
| **祖龙** | 6 信号综合熔断 + 信息增益检测 |
| LangChain | max_iterations 硬限制 |
| CrewAI | max_iterations 硬限制 |
| OpenDevin | 时间/步数限制 |

### 3. ⏸️ 跨天级任务挂起/恢复

支持：`暂停 → 关机 → 第二天开机 → 恢复继续执行`

适用场景：24 小时陪伴式机器人、超长程项目管理（跨周/跨月）、中断后环境变化自动重评估

### 4. 🧠 两阶段意图分类 + FC 循环

```
Round 1: 意图分类 → CHAT/COMPLEX/RESUME
Round 2: 场景化执行
  ├─ CHAT: 直接对话
  ├─ COMPLEX: 启动 FC 循环 + TaskGraph 自动规划
  └─ RESUME: 从快照恢复 + 继续执行
```

配套 5 层防护链（CB 强制收敛、RuleGuardian 过早完成拦截、InfoGap 信息缺口检测等）

### 5. 🎙️ 语音交互能力（TTS + ASR）

- **TTS (Kokoro-82M)**：82M 参数，<0.3s (CPU)，zf_xiaoxiao 中文女声
- **ASR (SenseVoice-Small)**：244M (ONNX INT8 量化)，中/英/日/韩/粤语 + 情感识别 + 事件检测
- **整体延迟**：3-4s (端到端，云端API调用)

---

## 🏗️ 系统架构

### 四层推理模型

```
L3 专家层 (Expert Layer)           - 7 种专家模型池，热切换 < 10ms
  ↓
L2 认知层 (Cognitive Layer)        - InferenceEngine (5700+ 行)，两阶段推理 + FC 循环
  ↓
L1-B 调度层 (Scheduler Layer)      - Gatekeeper + AttentionController，事件优先级路由
  ↓
L1-A/C 感知层 (Perception Layer)   - 音频融合 + YOLOv10 人体检测 + MediaPipe 姿态
  ↓
L0 设备层 (Device Layer)           - USB 摄像头/麦克风/扬声器驱动，运动检测
```

### 前后端分离架构

```
VS Code Extension (前端)  ←WebSocket→  Python Backend (后端)
  ├─ React + Vite Webview                ├─ FastAPI + WebSocket
  ├─ TypeScript + esbuild                ├─ L2 推理引擎
  └─ 工具执行 + UI 渲染                  ├─ MemoryGraph 记忆系统
                                         └─ TTS/ASR 语音交互
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
├── zulong-ide/                 # VS Code 扩展前端 (React + TypeScript)
├── zulong/                     # Python 后端核心
│   ├── ide/                    # IDE 模式 (WebSocket 服务 + 工具注册)
│   ├── l2/                     # L2 推理引擎 (推理 + 记忆 + 熔断 + 任务图)
│   ├── memory/                 # 记忆系统 (MemoryGraph + RAG)
│   ├── l1a/ / l1b/ / l1c/     # 感知层 (音频/调度/视觉)
│   └── l3/                     # L3 多专家模型层
├── config/                     # 配置文件 (zulong_config.yaml)
├── docs/                       # 技术文档与使用指南
└── requirements.txt            # Python 依赖
```

---

## 🔧 核心模块

| 模块 | 文件 | 行数 | 核心能力 |
|------|------|------|----------|
| **MemoryGraph** | `zulong/memory/memory_graph.py` | 2784 行 | 双路径检索、赫布学习、艾宾浩斯衰减、BFS 扩散激活 |
| **CircuitBreaker** | `zulong/l2/circuit_breaker.py` | 800+ 行 | 6 信号检测、状态机 (GREEN→YELLOW→RED) |
| **TaskGraph** | `zulong/l2/task_graph.py` | 1500+ 行 | 无限深度递归树、模板节点、任务依赖管理 |
| **InferenceEngine** | `zulong/l2/inference_engine.py` | 5700+ 行 | 两阶段推理、记忆检索、注意力窗口、FC 循环、5 层防护 |

---

## 🛠️ 工具系统

**内部工具**（后端执行）：`task_create_plan` | `task_add_node` | `task_mark_status` | `recall_memory` | `read_memory_node` | `save_memory_note` | `discover_related` | `focus_on_chain`

**远程工具**（前端执行）：`read_file` | `write_to_file` | `execute_command` | `search_files` | `browser_action`

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

## 📚 详细文档

**快速导航**：[文档索引页](./docs/index.md) - 快速查找你需要的文档

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

- GitHub: [@beautistart](https://github.com/beautistart)
- 邮箱: beautistart@qq.com

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

#### 意图与推理模型

- **[Qwen 系列模型](https://github.com/QwenLM/Qwen)** - 阿里巴巴通义千问（祖龙核心推理模型）
- **[ALBERT-tiny-chinese](https://github.com/ymcui/Chinese-ALBERT)** - 哈尔滨工业大学中文轻量级意图识别模型（15类意图分类）

#### 语音交互模型

- **[SenseVoice-Small](https://github.com/modelscope/FunASR)** - 阿里巴巴达摩院开源语音识别引擎（ASR 主引擎，244M 参数，ONNX INT8 量化，支持中/英/日/韩/粤语 + 情感识别 + 事件检测）
- **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** - 语音识别 ONNX 推理引擎
- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** - 轻量级文本转语音模型（TTS 主引擎，82M 参数，CPU 实时推理 <0.3s）
- **[Whisper](https://github.com/openai/whisper)** - OpenAI 开源多语言语音识别模型（ASR 备选 fallback）
- **[edge-tts](https://github.com/rany2/edge-tts)** - 微软云端 TTS 备选方案

#### 向量与嵌入模型

- **[BAAI BGE 系列](https://github.com/FlagOpen/FlagEmbedding)** - 北京智源研究院文本嵌入模型（记忆图谱向量检索）

#### 视觉与多模态模型

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
