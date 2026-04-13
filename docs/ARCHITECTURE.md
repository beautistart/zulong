# ZULONG Architecture Guide

## Overview

ZULONG uses a four-layer architecture inspired by the human nervous system. Each layer has clear responsibilities and communicates through an event-driven bus.

## Layer Architecture

### L0: Core Services (Brain Stem)

The foundational layer providing system-wide services.

**Components:**
- **EventBus** (`core/event_bus.py`) - Central async message bus with priority-based dispatching
- **StateManager** (`core/state_manager.py`) - Global state tracking (power mode, L2 status, active modules)
- **PowerManager** (`core/power_manager.py`) - Power state management (ACTIVE / SILENT modes)
- **Types** (`core/types.py`) - Event types, power states, L2 status enums

**Key Design:**
- All inter-layer communication goes through EventBus
- Events have priorities: CRITICAL > HIGH > NORMAL > LOW
- CRITICAL events (e.g., emergency stop) bypass all scheduling

### L1: Reflex Layer (Spinal Cord)

Fast-response layer handling sensory input and scheduling.

**L1-A: Reflex Engine** (`l1a/`)
- Processes raw sensor data (vision, audio)
- Generates reflex events for urgent stimuli
- Runs lightweight models for quick classification

**L1-B: Scheduler & Gatekeeper** (`l1b/`)
- Attention control and intent filtering
- Power-aware scheduling (unload L2 in SILENT mode)
- Document parsing and context packaging

**L1-C: Silent Visual Attention** (`l1c/`)
- Continuous background vision processing
- Optical flow detection, gesture recognition
- Only generates events on state transitions (silent by default)

### L2: Expert Reasoning (Cortex)

The reasoning layer powered by LangGraph state graphs.

**Components:**
- **ExpertInvoker** (`l2/expert_invoker.py`) - Orchestrates expert model calls
- **EventHandler** (`l2/event_handler.py`) - Translates EventBus events into expert tasks
- **RAGIntegrationNode** (`l2/rag_node.py`) - Retrieves relevant context from memory
- **IntentRecognition** (`l2/intent_recognition_node.py`) - Classifies user intent

**Key Design:**
- LangGraph StateGraph for structured reasoning flow
- Expert calls are async with timeout and fallback
- RAG integration injects relevant memory context before reasoning

### L3: Expert Skill Pool (Specialized Areas)

Specialized skill execution layer.

**Components:**
- **BaseExpertNode** (`l3/base_expert_node.py`) - Abstract base for all expert nodes
- **TTSExpertNode** (`l3/tts_expert_node.py`) - CosyVoice text-to-speech (CPU)
- **VisionExpertNode** (`l3/vision_expert_node.py`) - Visual understanding
- **NavExpertNode** (`l3/nav_expert_node.py`) - Navigation planning
- **ModelSwitcher** (`l3/model_switcher.py`) - Dynamic model loading/unloading
- **ExpertContainer** (`l3/expert_container.py`) - Expert lifecycle management

## Memory System

### Community Edition

The community edition includes the RAG pipeline:

```
User Input -> Embedding (bge-small-zh) -> Hybrid Search -> Context Injection -> LLM
                                             |
                                     BM25 + Vector Search
```

**Components:**
- **RAGManager** (`memory/rag_manager.py`) - Central RAG orchestrator
- **EmbeddingManager** (`memory/embedding_manager.py`) - Vector encoding
- **TaggingEngine** (`memory/tagging_engine.py`) - Automatic content tagging
- **HybridSearch** (`memory/hybrid_search_config.py`) - BM25 + vector fusion
- **VectorCache** (`memory/vector_cache.py`) - Embedding result caching

### Enterprise Edition (not included)

Enterprise adds advanced memory features:
- Memory consolidation and evolution
- Knowledge graph construction
- Person profile tracking
- Episodic memory with temporal indexing

## Plugin System

### Architecture

```
PluginManager
    |
    +-- registers --> IL1Module implementations
    |
    +-- dispatches events via EventBus
    |
    +-- manages lifecycle (init/shutdown)
```

### Interface Contract

All plugins implement `IL1Module` (defined in `modules/l1/core/interface.py`):

| Method | Purpose |
|--------|---------|
| `module_id` | Unique identifier |
| `priority` | Event processing priority |
| `on_event(event)` | Handle incoming events |
| `initialize()` | Setup resources |
| `shutdown()` | Cleanup resources |

### Built-in Plugins

| Plugin | Path | Function |
|--------|------|----------|
| Vision | `plugins/vision/` | Camera processing, gesture detection |
| Voice | `plugins/voice/` | Audio input/output |
| Motor | `plugins/motor/` | Actuator control |
| Gas | `plugins/gas/` | Gas sensor monitoring |

## Event System

### Event Priority

| Priority | Value | Use Case |
|----------|-------|----------|
| CRITICAL | 4 | Emergency (fall detection, fire) |
| HIGH | 3 | Obstacle detection, motion trigger |
| NORMAL | 2 | User speech, routine events |
| LOW | 1 | Background updates, logging |

### Event Flow

1. Sensor/plugin generates event
2. EventBus dispatches by priority
3. L1-A checks for reflex triggers
4. L1-B schedules for L2 processing
5. L2 invokes expert reasoning with RAG context
6. L3 executes specialized skills
7. Results flow back through EventBus

## Hardware Optimization

### GPU Memory Budget (RTX 3060 6GB)

| Component | VRAM | Strategy |
|-----------|------|----------|
| L1 Model (0.8B INT4) | ~400MB | Always loaded |
| L2 Model (2B) | ~1.5GB | Load on demand |
| L3 Expert Models | ~2GB | Hot-swap pool |
| Framework overhead | ~500MB | Fixed |
| **Total** | **~4.4GB** | Within 6GB limit |

### Key Optimizations
- INT4 quantization for all LLM models
- TTS runs on CPU (zero GPU memory)
- Embedding model on CPU
- Model hot-swap: only one expert active at a time
- Shared KV Cache across model switches

## Deployment

### Docker

```bash
cp docker-compose.yml.example docker-compose.yml
# Edit docker-compose.yml with your settings
docker-compose up -d
```

### Monitoring

The system exposes metrics via `zulong/utils/metrics.py`:
- Event processing latency
- Model inference time
- Memory retrieval hit rate
- GPU memory usage

Optional Grafana dashboard available via `docker-compose.yml.example`.
