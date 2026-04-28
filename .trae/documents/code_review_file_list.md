# 代码审查文件清单

## 核心代码文件

### 1. 主系统代码 (zulong/)
- `zulong/__init__.py`
- `zulong/bootstrap.py`
- `zulong/state.py`

#### 核心层 (L0)
- `zulong/core/__init__.py`
- `zulong/core/event_bus.py`
- `zulong/core/state.py`
- `zulong/core/state_manager.py`
- `zulong/core/types.py`
- `zulong/core/websocket_server.py`
- `zulong/core/power_manager.py`
- `zulong/core/attention_atoms.py`
- `zulong/core/graph.py`

#### 反射层 (L1)
- `zulong/l1/__init__.py`
- `zulong/l1/config.py`
- `zulong/l1/graph.py`
- `zulong/l1/nodes.py`
- `zulong/l1/scheduler.py`
- `zulong/l1/sensors.py`

#### L1-A 反射层
- `zulong/l1a/__init__.py`
- `zulong/l1a/l1a_config.py`
- `zulong/l1a/reflex_controller.py`
- `zulong/l1a/audio_preprocessor.py`
- `zulong/l1a/vision_processor.py`
- `zulong/l1a/fusion_controller.py`
- `zulong/l1a/context_tracker.py`
- `zulong/l1a/reflex/vision_node.py`
- `zulong/l1a/reflex/vl_audio_node.py`

#### L1-B 调度层
- `zulong/l1b/__init__.py`
- `zulong/l1b/l1b_config.py`
- `zulong/l1b/scheduler_gatekeeper.py`
- `zulong/l1b/async_scheduler.py`
- `zulong/l1b/optimized_scheduler.py`
- `zulong/l1b/hotswap_scheduler.py`
- `zulong/l1b/attention_controller.py`
- `zulong/l1b/intent_filter.py`
- `zulong/l1b/document_parser.py`
- `zulong/l1b/review_trigger_node.py`
- `zulong/l1b