<div align="center">

#  Zulong (祖龙)

### Multi-Layer Adaptive Cognitive Agent Operating System

**An AI Agent Framework with Biological Memory Mechanisms**

[![License](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![VS Code](https://img.shields.io/badge/VS%20Code-Extension-green.svg)](https://code.visualstudio.com/)
[![GitHub release](https://img.shields.io/github/release/beautistart/zulong.svg)](https://github.com/beautistart/zulong/releases)
[![Stars](https://img.shields.io/github/stars/beautistart/zulong?style=social)](https://github.com/beautistart/zulong)

[English](./docs/README_EN.md) | [简体中文](../README.md)

</div>

---

## 🎯 At a Glance

**Zulong is an AI Agent cognitive system with a unified memory graph, employing Hebbian learning and Ebbinghaus decay algorithms, achieving year-level complete memory under dynamic attention control. It implements infinite context at the system level, and can run full memory + inference + multimodal capabilities on consumer-grade hardware (AI MAX 395 128G).**

### ✨ Core Features of Zulong Memory System

**Before diving into the technical details, here are the differentiated capabilities you can experience:**

#### 1. Infinite Context

Zulong enables AI to break through model context window limits through autonomous dynamic attention mechanism. Even when executing ultra-complex long-running tasks, it never loses global attention, and subtask execution stays on track.

#### 2. Memory Association Discovery

Zulong discovers memory associations through BFS diffusion across all graph memory nodes + dependency association edges, enabling discovery of memories without keywords or semantic relationships but with other associations, simulating the human brain's "associative" capability for举一反三 (inferring other cases from one instance).

#### 3. Year-Level Complete Memory

Zulong's memory system is built on graph memory, with all memories stored in memory nodes connected as a large network. Memory nodes contain complete, uncompressed memories, allowing recall of memories no matter how old they are.

---

##  The Story Behind Zulong

I'm an interior designer who independently developed Zulong's 82,000+ lines of code in 2 months.

Don't be surprised—because a designer's expertise is being the "chief engineer" of a project, planning blueprints and letting professionals in various fields implement them.

I designed Zulong's architectural blueprint and used AI-powered coding IDEs to help me implement the code:

- **Qwen Desktop** - Project Advisor
- **trae** - Early-stage backend code engineer
- **qoder** - Late-stage project correction + feature implementation
- **codearts** - Late-stage project correction + code review + feature implementation

---

> **📢 v1.0.0 Official Release (2026-05-12)**
>
> After major architecture upgrades to the memory system, this is Zulong's first official release, including complete memory graph, infinite loop detection, year-level complete memory, and other core capabilities.
>
> **Core Updates**:
> - ✅ **MemoryGraph** - 9 node types + 7 edge types + Hebbian learning + Ebbinghaus decay
> - ✅ **CircuitBreaker 6-signal detection** - Information gain detection and 6 signal types
> - ✅ **Year-level complete memory** - Full state serialization
> - ✅ **5-layer protection chain** - Based on qwen3.6-27B model
> - ✅ **Complete VS Code Extension + TTS/ASR voice interaction**
>
> See [CHANGELOG.md](./CHANGELOG.md) for details

---

## ✨ Key Highlights

### 1. 🔮 Unified Memory Graph (MemoryGraph)

```
Memory is not flat text, but an interconnected knowledge network
┌─────────────────────────────────────────┐
│  MemoryGraph (NetworkX DiGraph)          │
│  ├─ 9 Node Types: TASK/DIALOGUE/KNOWLEDGE... │
│  ├─ 7 Edge Types: HIERARCHY/SEMANTIC/CAUSAL... │
│  ├─ Hebbian Learning: Co-activated edges auto-strengthen │
│  ├─ Ebbinghaus Decay: exp(-age/half_life)    │
│  ├─ Persistence: LMDB + GraphML               │
│  └─ Dual-Path Retrieval: BFS hot + FAISS cold   │
└─────────────────────────────────────────┘
```

**Technical Advantages**:
- ✅ **Persistent unified memory graph** (LMDB + GraphML storage)
- ✅ **Hebbian learning engine**: Co-activation count ≥ 3 auto-creates ASSOCIATION edges
- ✅ **Ebbinghaus forgetting curve**: 6-level importance half-life (TRIVIAL 6h → MUST_REMEMBER ∞)
- ✅ **Dual-path retrieval**: Hot path BFS (<50ms) + Cold path FAISS (<200ms)
- ✅ **Semantic edge auto-discovery**: Background cosine similarity > 0.7 auto-creates SEMANTIC edges
- ✅ **3D tagging system**: Temperature × Importance × TimeScope orthogonal combination

### 2. 🛡️ CircuitBreaker Infinite Loop Detector - **Most Comprehensive in Industry**

```python
# 6-signal comprehensive circuit breaking mechanism
Signal 1: Repeated call detection (name + params_hash)
Signal 2: Pattern loop detection (tool frequency + Jaccard similarity)
Signal 3: Information gain decay detection (result hash overlap rate)
Signal 4: Context pressure detection (token estimate / window ratio)
Signal 5: Time elapsed detection (disabled, steps as main control)
Signal 6: No-progress idle detection (continuous info retrieval without action)

# State machine: GREEN → YELLOW(inject warning) → RED(force stop)
```

**Comparison**:
| Project | Detection Method |
|---------|------------------|
| **Zulong** | 6-signal comprehensive circuit breaking + information gain detection |
| LangChain | max_iterations hard limit |
| CrewAI | max_iterations hard limit |
| OpenDevin | Time/step limits |

### 3. ⏸️ Cross-Day Task Suspend/Resume - **No Competitor Has This Capability**

```python
# Complete state serialization
SuspendableTaskState:
  - messages: List[Dict]              # Conversation history
  - task_graph: TaskGraph             # Task tree snapshot
  - circuit_breaker_state: Dict       # CB state
  - memory_seeds: List[str]           # Memory seed nodes
  - environment_snapshot: Dict        # Working directory file state

# Supports: Suspend → Shutdown → Boot next day → Resume and continue
```

**Use Cases**:
- 🤖 24-hour companion robots
- 📋 Ultra-long project management (cross-week/cross-month)
- 🔄 Auto-re-evaluation after environment changes from interruption

### 4. 🧠 Two-Stage Intent Classification + FC Loop

```
Round 1: Intent Classification
  └─ Forced tool_choice → CHAT/COMPLEX/RESUME

Round 2: Scenario Execution
  ├─ CHAT: Direct conversation
  ├─ COMPLEX: Start FC loop + TaskGraph auto-planning
  └─ RESUME: Restore from snapshot + continue execution
```

**5-Layer Protection Chain** (~500 lines):
1. CB forced convergence check
2. RuleGuardian premature completion interception
3. InfoGap information gap detection
4. RESUME AutoMark safety net
5. COMPLEX Backfill node backfill

### 5. ️ Voice Interaction (TTS + ASR)

```yaml
TTS (Kokoro-82M):
  - Parameters: 82M
  - Inference: <0.3s (CPU)
  - Voice: zf_xiaoxiao (Chinese female)

ASR (SenseVoice-Small):
  - Parameters: 244M (ONNX INT8 quantized)
  - Capabilities: Chinese/English/Japanese/Korean/Cantonese + emotion recognition + event detection
  - Inference: 0.5-1s (5s audio)

**Overall Latency**: 3-4s (end-to-end, cloud API calls)
```

---

## 🏗️ System Architecture

### Four-Layer Inference Model

```
┌─────────────────────────────────────────────┐
│            L3 Expert Layer                    │
│  7 expert model pools: GENERAL/LOGIC/CREATIVE/... │
│  Hot-swap < 10ms                               │
├─────────────────────────────────────────────┤
│         L2 Cognitive Layer                    │
│  InferenceEngine (5700+ lines)                  │
│  ├─ Two-stage inference + FC loop               │
│  ├─ MemoryGraph memory retrieval                │
│  ├─ TaskGraph task orchestration                │
│  ├─ CircuitBreaker circuit breaking             │
│  └─ Attention window three modes                │
├─────────────────────────────────────────────┤
│        L1-B Scheduler Layer                     │
│  Gatekeeper + AttentionController               │
│  ├─ Event priority routing (CRITICAL>HIGH>NORMAL>LOW)│
│  └─ Interruption handling (freeze→reorganize→inject) │
├─────────────────────────────────────────────┤
│      L1-A/C Perception Layer                    │
│  L1-A: Audio fusion + self-reflection mechanism   │
│  L1-C: YOLOv10 human detection + MediaPipe pose   │
├─────────────────────────────────────────────┤
│          L0 Device Layer                        │
│  USB camera/microphone/speaker drivers            │
│  Motion detection (frame diff + optical flow)     │
└─────────────────────────────────────────────┘
```

### Frontend-Backend Separated Architecture

```
┌──────────────────────────────┐
│   VS Code Extension (Frontend) │
│   ├─ React + Vite Webview      │
│   ├─ TypeScript + esbuild      │
│   └─ Tool execution + UI rendering │
└──────────┬───────────────────┘
           │ WebSocket
           │ ws://127.0.0.1:8090/ide
           ↓
┌──────────────────────────────┐
│   Python Backend               │
│   ├─ FastAPI + WebSocket       │
│   ├─ L2 Inference Engine       │
│   ├─ MemoryGraph Memory System │
│   └─ TTS/ASR Voice Interaction │
└──────────────────────────────┘
```

---

## 🚀 Quick Start

### Requirements

- Python 3.10+
- Node.js 18+
- VS Code
- Recommended hardware: AI MAX 395 128G (can run on pure CPU)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/beautistart/zulong.git
cd zulong

# 2. Install Python backend dependencies
python -m venv zulong_env
source zulong_env/bin/activate  # Windows: zulong_env\Scripts\activate
pip install -r requirements.txt

# 3. Install frontend dependencies
cd zulong-ide
npm install
cd webview-ui && npm install && cd ..

# 4. Build VS Code extension
npm run protos                      # Generate TypeScript proto files
node esbuild.mjs --production       # esbuild bundling
npx @vscode/vsce package --no-dependencies --allow-missing-repository --skip-license

# 5. Install extension to VS Code
code --install-extension zulong-ide-0.1.0.vsix --force
```

### Start Service

Zulong uses start.py for unified startup (open terminal in project root directory and run start.py):

```bash
# Unified startup script (in project root directory)
python start.py

# Or with sensor simulation mode
python start.py --mock-sensors

# Open VS Code, click the Zulong icon to start session
```

### Configuration

Edit `config/zulong_config.yaml`:

```yaml
# LLM Configuration
llm:
  backend: "vllm"  # Options: ollama, lm_studio, openai
  model_id: "Qwen/Qwen2.5-7B-Instruct"
  
# WebSocket port
ide_server:
  port: 8090
  host: "127.0.0.1"
  
# Voice Configuration
audio:
  tts:
    backend: kokoro
    voice: zf_xiaoxiao
  asr:
    backend: sensevoice
    language: zh
```

---

## 🆚 Competitor Comparison

| Feature | Zulong | LangChain | CrewAI | MemGPT/Letta | AutoGPT |
|---------|--------|-----------|--------|--------------|---------|
| **Unified Memory Graph** | ✅ LMDB + GraphML | ❌ In-memory DAG | ❌ | ❌ Single-path vector | ❌ File-based |
| **Hebbian Learning** | ✅ Co-activation strengthening | ❌ | ❌ |  | ❌ |
| **Ebbinghaus Decay** | ✅ exp decay | ❌ | ❌ | ❌ | ❌ Age-based |
| **Dual-Path Retrieval** | ✅ BFS + FAISS | ❌ | ❌ | ❌ Single-path | ❌ Single-path |
| **Infinite Loop Detection** | ✅ 6-signal circuit breaking |  Hard limit | ❌ Hard limit | ❌ | ❌ Hard limit |
| **Task Suspend/Resume** | ✅ Cross-day | ❌ | ❌ | ❌ | ❌ |
| **Voice Interaction** | ✅ TTS + ASR | ❌ | ❌ | ❌ | ❌ |

---

## 📂 Project Structure

```
zulong_beta4/
├── zulong-ide/                 # VS Code extension frontend
│   ├── src/                    # TypeScript source code
│   │   ├── core/api/           # API providers
│   │   └── webview-ui/         # React Webview
│   └── package.json
├── zulong/                     # Python backend
│   ├── ide/                    # IDE mode specific
│   │   ├── ide_server.py       # WebSocket server
│   │   ├── ide_fc_runner.py    # FC loop executor
│   │   └── ide_tool_registry.py
│   ├── l2/                     # L2 inference engine core
│   │   ├── inference_engine.py
│   │   ├── fc_graph.py
│   │   ├── circuit_breaker.py
│   │   ├── task_graph.py
│   │   └── attention_window.py
│   ├── memory/                 # Memory system
│   │   ├── memory_graph.py     # Unified memory graph (LMDB + GraphML)
│   │   ├── rag_manager.py
│   │   ── episodic_memory.py
│   ├── l1a/                    # L1-A audio fusion layer
│   ├── l1b/                    # L1-B event scheduling layer
│   ├── l1c/                    # L1-C visual perception layer
│   ├── l3/                     # L3 multi-expert model layer
│   └── mcp/                    # MCP protocol support
├── config/                     # Configuration files
│   └── zulong_config.yaml
├── docs/                       # Documentation
│   ├── TSD_v2.4.md             # Technical Specification Document
│   ├── system-overview.md
│   └── ide-usage.md
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🔧 Core Modules

### 1. MemoryGraph Unified Memory Graph

**File**: `zulong/memory/memory_graph.py` (2784 lines)

**Key Methods**:
```python
# Dual-path retrieval
retrieve_context(query, session_id, top_k=10) -> List[MemoryNode]

# Hebbian learning
hebbian_strengthen(node_a, node_b) -> None

# Ebbinghaus decay
decay_and_prune() -> None

# BFS diffusion activation
compute_activations(seed_nodes, max_depth=3) -> Dict[str, float]
```

### 2. CircuitBreaker Infinite Loop Detector

**File**: `zulong/l2/circuit_breaker.py` (800+ lines)

**Key Attributes**:
```python
# 6 detection signals
signal_repeated_call: bool      # Repeated same call
signal_pattern_loop: bool       # Pattern loop
signal_info_gain_decay: bool    # Information gain decay
signal_context_pressure: bool   # Context pressure
signal_time_elapsed: bool       # Time elapsed
signal_no_progress: bool        # No progress idle

# State machine
state: "GREEN" | "YELLOW" | "RED"
```

### 3. TaskGraph Task Graph

**File**: `zulong/l2/task_graph.py` (1500+ lines)

**Key Capabilities**:
- Infinite depth recursive tree
- Template nodes + dynamic generation nodes
- Leaf node execution + non-leaf node state aggregation
- Task dependency management

### 4. InferenceEngine Inference Engine

**File**: `zulong/l2/inference_engine.py` (5700+ lines)

**Key Process**:
1. Intent classification (CHAT/COMPLEX/RESUME)
2. Memory retrieval (MemoryGraph + RAG)
3. Attention window clipping
4. FC loop execution
5. Tool call dispatch
6. 5-layer protection chain evaluation

---

## 🛠️ Tool System

### Internal Tools (Executed by Backend)

| Tool Name | Responsibility |
|-----------|----------------|
| `task_create_plan` | Create task plan tree |
| `task_add_node` | Add task node |
| `task_mark_status` | Mark task status |
| `recall_memory` | Memory retrieval |
| `read_memory_node` | Read memory node |
| `save_memory_note` | Save memory note |
| `discover_related` | Discover related nodes |
| `focus_on_chain` | Switch attention focus |

### Remote Tools (Executed by Frontend)

| Tool Name | Responsibility |
|-----------|----------------|
| `read_file` | Read file |
| `write_to_file` | Write to file |
| `execute_command` | Execute command |
| `search_files` | Search files |
| `browser_action` | Browser action |

---

##  MCP Protocol Support

Zulong provides an independent MCP Server that can be used in other IDEs to leverage Zulong's memory capabilities:

```python
# mcp_server.py
Server("zulong-memory")

# 7 MCP tools
- zulong_memory_search    # Project-level memory search
- zulong_memory_save      # Save project memory
- zulong_task_search      # Historical task search
- zulong_experience_search # Experience store search
- zulong_knowledge_query  # Knowledge base query
- zulong_graph_query      # Memory graph query
- zulong_entity_link      # Entity linking
```

---

## 📝 Documentation

**Quick Navigation**: [Documentation Index](./docs/index.md) - Quickly find the documentation you need

### Technical Documentation
- [Technical Specification Document (TSD)](./docs/architecture/technical-spec-v3.md) - Complete system architecture design
- [In-Depth Technical Analysis Report](./docs/architecture/system-overview.md) - Code review and competitor comparison
- [Heterogeneous Graph Memory System](./docs/memory_graph/) - MemoryGraph design and implementation
- [CircuitBreaker Design Document](./docs/CircuitBreaker_Design.md) - 6-signal infinite loop detection mechanism

### User Guides
- [IDE Usage Guide](./docs/Zulong_IDE使用指南.md) - User operation manual
- [Quick Start Guide](./docs/guides/quick-start.md) - 3-step installation and startup
- [Configuration Guide](./docs/guides/configuration.md) - System configuration instructions
- [Docker Deployment Guide](./docs/guides/docker-deployment.md) - Containerized deployment
- [Documentation Index](./docs/index.md) - Complete documentation navigation

### Development Documentation
- [Contributing Guide](./CONTRIBUTING.md) - How to contribute code
- [Changelog](./CHANGELOG.md) - Version update records

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

### Roadmap

- [ ] Expand MCP tool set (7 → 30+)
- [ ] Add benchmark data
- [ ] TaskGraph UI visualization
- [ ] Multi-Agent collaboration support
- [ ] Performance optimization (critical path Rust/Cython rewrite)

---

## 📄 License

This repository uses a layered license:

- **Core Code** (`zulong/`): AGPL-3.0
- **VS Code Extension** (`zulong-ide/`): MIT
- **Documentation** (`docs/`): CC BY-NC-SA 4.0

See the [LICENSE](./LICENSE) file for details.

---

## 👨‍💻 Author

**An interior designer who independently developed Zulong in 2 months with AI assistants**

- GitHub: [@beautistart](https://github.com/beautistart)
- Email: beautistart@qq.com

---

##  Acknowledgments

Zulong's development relies on numerous excellent open-source projects and community contributions. Sincere thanks to the following project teams:

### Core Infrastructure Framework

- **[Cline](https://github.com/cline/cline)** v3.82.0 - Zulong IDE is developed based on Cline framework. Thanks to the Cline team for their excellent work
- **[PyTorch](https://github.com/pytorch/pytorch)** - Deep learning framework providing model inference and computation core
- **[FastAPI](https://github.com/fastapi/fastapi)** + **[Uvicorn](https://github.com/encode/uvicorn)** - High-performance async web service framework
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** - Pre-trained model loading and inference

### Memory System & Vector Computation

- **[NetworkX](https://github.com/networkx/networkx)** - Core graph computation engine for memory graph
- **[LMDB](https://lmdb.readthedocs.io/)** - High-performance embedded key-value database (graph persistence)
- **[FAISS](https://github.com/facebookresearch/faiss)** - Facebook AI vector similarity search engine
- **[Qdrant](https://github.com/qdrant/qdrant)** - Vector database and semantic retrieval

### Voice Interaction

- **[FunASR / SenseVoice](https://github.com/modelscope/FunASR)** - Alibaba DAMO Academy open-source speech recognition engine (ASR main engine)
- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** - Lightweight text-to-speech model (TTS main engine, 82M parameters)
- **[edge-tts](https://github.com/rany2/edge-tts)** - Microsoft cloud TTS alternative
- **[Whisper](https://github.com/openai/whisper)** - OpenAI open-source multilingual speech recognition model (fallback)

### Frontend & UI

- **[React](https://github.com/facebook/react)** - Frontend UI framework
- **[Tailwind CSS](https://tailwindcss.com/)** - Atomic CSS framework
- **[Radix UI](https://www.radix-ui.com/)** - Headless UI component library
- **[Vite](https://vitejs.dev/)** - Frontend build tool
- **[esbuild](https://esbuild.github.io/)** - Ultra-fast JavaScript/TypeScript bundler

### Development Tools & Infrastructure

- **[Model Context Protocol (MCP)](https://modelcontextprotocol.io/)** - Model Context Protocol SDK
- **[OpenTelemetry](https://opentelemetry.io/)** - Observability and distributed tracing
- **[Playwright](https://playwright.dev/)** - Browser automation testing
- **[Mermaid](https://mermaid.js.org/)** - Chart and flowchart rendering

### Models & Weights

#### Intent & Inference Models

- **[Qwen Series](https://github.com/QwenLM/Qwen)** - Alibaba Tongyi Qianwen (Zulong core inference model)
- **[ALBERT-tiny-chinese](https://github.com/ymcui/Chinese-ALBERT)** - Harbin Institute of Technology Chinese lightweight intent recognition model (15-class intent classification)

#### Voice Interaction Models

- **[SenseVoice-Small](https://github.com/modelscope/FunASR)** - Alibaba DAMO Academy open-source speech recognition engine (ASR main engine, 244M parameters, ONNX INT8 quantized, supports Chinese/English/Japanese/Korean/Cantonese + emotion recognition + event detection)
- **[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)** - Speech recognition ONNX inference engine
- **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** - Lightweight text-to-speech model (TTS main engine, 82M parameters, CPU real-time inference <0.3s)
- **[Whisper](https://github.com/openai/whisper)** - OpenAI open-source multilingual speech recognition model (ASR fallback)
- **[edge-tts](https://github.com/rany2/edge-tts)** - Microsoft cloud TTS alternative

#### Vector & Embedding Models

- **[BAAI BGE Series](https://github.com/FlagOpen/FlagEmbedding)** - Beijing Academy of Artificial Intelligence text embedding model (memory graph vector retrieval)

#### Vision & Multimodal Models

- **[MediaPipe](https://google.github.io/mediapipe/)** - Google open-source cross-platform ML pipeline (face/hand/pose detection)

### Agent Framework & Orchestration

- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Graph-based AI Agent orchestration framework
- **[LangChain](https://github.com/langchain-ai/langchain)** - Multi-LLM application development framework
- **[VLLM](https://github.com/vllm-project/vllm)** - High-performance LLM inference and serving engine

### Vision & Multimodal

- **[OpenCV](https://github.com/opencv/opencv)** - Computer vision library (camera module and motion detection)

### Natural Language Processing

- **[NLTK](https://www.nltk.org/)** - Natural Language Toolkit
- **[jieba](https://github.com/fxsjy/jieba)** - Chinese word segmentation tool

### Numerical & Scientific Computing

- **[NumPy](https://numpy.org/)** - Numerical computing and multidimensional arrays
- **[SciPy](https://scipy.org/)** - Scientific computing and linear algebra

### Security & Parsing

- **[PyJWT](https://github.com/jpadilla/pyjwt)** - JSON Web Token creation and verification
- **[Tree-sitter](https://tree-sitter.github.io/)** - Incremental code parser generator

### AI Coding Tools

Thanks to the following AI coding tools for their help in Zulong's development:

- **Qwen Desktop** - Project Advisor
- **trae** - Early-stage backend code engineer
- **qoder** - Late-stage project correction + feature implementation
- **codearts** - Late-stage project correction + code review + feature implementation

---

## ⭐ Star History

If this project helps you, please give it a ⭐ Star to support us!

[![Star History Chart](https://api.star-history.com/svg?repos=beautistart/zulong&type=Date)](https://star-history.com/#beautistart/zulong&Date)

---

<div align="center">

**Made with ❤️ by an Interior Designer turned AI Developer**

**Zulong - Giving AI True Memory**

</div>
