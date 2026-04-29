# 祖龙 (ZULONG) AI System v2.0

祖龙是一个本地优先的自主 AI Agent 系统，基于三层分离架构构建。它通过 Function Calling 驱动任务执行，结合图记忆网络实现经验沉淀与复用，能够自主规划、拆解并完成复杂的多步骤任务。

## 核心架构

```
┌─────────────────────────────────────────────────────┐
│                   用户界面层                          │
│         Web UI (WebSocket) / 语音 / 视觉              │
├─────────────────────────────────────────────────────┤
│                 OpenClaw Bridge                       │
│       WebAdapter · EventBusBridge · Listeners         │
├──────────┬──────────────────────┬───────────────────┤
│  TaskGraph │    InferenceEngine   │    MemoryGraph    │
│  任务规划   │    FC Loop (L2)     │    图记忆网络      │
│  执行追踪   │    意图识别 · 工具链   │    经验萃取       │
├──────────┴──────────────────────┴───────────────────┤
│                    工具层 (Tools)                      │
│   exec_write_file · exec_run_command · web_search     │
│   task_create_plan · task_add_node · task_mark_status  │
└─────────────────────────────────────────────────────┘
```

### 三层分离

| 层 | 职责 | 说明 |
|---|---|---|
| **TaskGraph** | 任务规划与执行追踪 | 节点完成即冻结，保证任务状态确定性 |
| **Git** | 文件版本管理 | 直接复用 Git 能力，不重复实现 |
| **MemoryGraph** | 经验沉淀 | 通过"同一文件再修改""对话结束自动萃取""Git commit 感知"三大触发点，将踩坑与解决方案提炼为 experience 节点 |

## 项目结构

```
zulong_beta4/
├── zulong/                     # 核心引擎
│   ├── l2/                     # L2 推理层
│   │   ├── inference_engine.py # 推理引擎 (FC Loop 入口)
│   │   ├── fc_graph.py         # LangGraph FC 循环图
│   │   ├── task_graph.py       # TaskGraph 数据结构
│   │   ├── attention_window.py # 注意力窗口管理
│   │   ├── circuit_breaker.py  # 智能迭代控制
│   │   ├── task_archive.py     # 已完成任务归档
│   │   └── task_suspension.py  # 任务挂起/恢复
│   ├── memory/                 # 记忆系统
│   │   ├── memory_graph.py     # MemoryGraph (NetworkX 图)
│   │   ├── graph_adapters.py   # TaskGraph ↔ MemoryGraph 投射
│   │   ├── embedding_manager.py# 向量嵌入
│   │   ├── rag_manager.py      # RAG 检索增强
│   │   └── experience_generator.py # 经验自动萃取
│   ├── tools/                  # FC 工具集
│   │   ├── exec_tools.py       # 文件写入 / 命令执行
│   │   ├── task_tools.py       # 任务图管理 (6个工具)
│   │   ├── session_tool.py     # 会话启动 / 意图分类
│   │   └── tool_engine.py      # 工具注册与调用引擎
│   ├── l1b/                    # L1b 调度层
│   │   └── scheduler_gatekeeper.py
│   ├── config/                 # 配置管理
│   ├── skill_packs/            # 可插拔技能包
│   └── ...
├── openclaw_bridge/            # 前端桥接层
│   ├── adapters/
│   │   └── web_adapter.py      # WebSocket/HTTP 服务
│   ├── listeners/              # 事件监听器
│   └── web/static/
│       └── index.html          # 单页 Web UI
├── config/
│   └── zulong_config.yaml      # 统一配置文件
├── agent_workspace/            # 任务工作目录 (每任务独立子文件夹)
├── data/                       # 持久化数据
│   ├── graph_backups/          # TaskGraph 备份
│   ├── completed_tasks/        # 已完成任务归档
│   ├── suspended_tasks/        # 挂起任务
│   └── rag/                    # RAG 向量库
├── tests/                      # 测试
└── logs/                       # 运行日志
```

## 关键特性

- **FC 自主循环**: LangGraph 驱动的 Function Calling 循环，模型自主决定调用工具或直接回复，支持最多 100 步多轮工具链
- **任务图谱 (TaskGraph)**: 树形任务拆解，节点状态自动级联，支持挂起/恢复/归档
- **图记忆网络 (MemoryGraph)**: NetworkX 有向图，9 种节点类型 + 7 种边类型，BFS 激活传播
- **思维可视化**: D3.js 力导向浮动窗口，实时展示活跃记忆节点及其邻居
- **任务独立工作目录**: 每个复杂任务自动在 `agent_workspace/` 下创建隔离文件夹
- **智能迭代控制 (Circuit Breaker)**: 5 维信号检测（重复调用、模式循环、信息增益、上下文压力、时间），防止死循环
- **多后端 LLM 支持**: Ollama / vLLM / SGLang / llama.cpp / LM Studio / OpenAI
- **RAG 检索增强**: BGE 向量嵌入 + FAISS 索引 + 混合搜索
- **视觉感知**: YOLO + MediaPipe 摄像头分析（可选）
- **语音交互**: CosyVoice TTS + 麦克风输入（可选）

## 快速开始

### 环境要求

- Python 3.10+
- CUDA GPU（推荐，CPU 也可运行）
- 本地 LLM 后端（默认使用 Ollama）

### 安装

```bash
# 克隆项目
git clone https://github.com/beautistart/zulong_beta4.git
cd zulong_beta4

# 安装依赖
pip install -r requirements.txt

# 安装 Ollama 并拉取模型 (默认后端)
# https://ollama.com
ollama pull gpt-oss:20b-cloud
```

### 配置

编辑 `config/zulong_config.yaml`：

```yaml
# 选择 LLM 后端
llm:
  backend: "ollama"           # ollama / vllm / sglang / openai / ...

# Ollama 配置
  ollama:
    base_url: "http://localhost:11434/v1"
    model_id: "gpt-oss:20b-cloud"
    num_ctx: 16384            # 上下文窗口，根据显存调整
```

### 运行

```bash
python -m zulong.bootstrap
```

Web UI 默认地址: `http://localhost:5555`

## 配置说明

所有配置集中在 `config/zulong_config.yaml`，主要模块：

| 配置区块 | 说明 |
|----------|------|
| `system` | 基础配置（日志级别、数据目录等） |
| `llm` | LLM 后端选择与参数（支持 7 种后端） |
| `l2_inference` | 推理引擎（步数限制、Circuit Breaker、超时等） |
| `vision` | 视觉系统（YOLO、MediaPipe、摄像头） |
| `audio` | 音频系统（麦克风、TTS） |
| `memory` | 记忆系统（短期记忆、RAG、经验库） |
| `tools` | 工具系统（工作目录、搜索引擎、技能包） |
| `event_bus` | WebSocket 事件总线 |

## 测试

```bash
# 运行核心测试
python -m pytest tests/test_v3_fixes_regression.py -q
python -m pytest tests/test_memory_graph.py -q
python -m pytest tests/test_task_graph_fix.py -q
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 任务编排 | LangGraph |
| 图数据结构 | NetworkX |
| 向量嵌入 | BAAI/bge-small-zh-v1.5 + FAISS |
| 前端可视化 | D3.js (力导向图) |
| WebSocket | 自建事件总线 |
| LLM 接口 | OpenAI-compatible API |
| 视觉 | YOLOv10 + MediaPipe |
| TTS | CosyVoice 2 |

## License

Private - All rights reserved.
