# 祖龙 (ZULONG) 系统 L2 层模型定义分析报告

**分析日期**: 2026-03-30  
**分析版本**: v2.4  
**分析目的**: 检查 L2 层的两个模型被定义为什么名字

---

## 📋 L2 层模型定义

L2 层共有 **2 个模型**，在 [`ModelID`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py#L9-L18) 枚举中定义：

### 1. **L2_CORE** - L2 核心推理模型 ⭐⭐⭐⭐⭐

**定义位置**: [`models/config.py`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py#L13)

```python
L2_CORE = "L2_CORE"  # L2: 推理决策 (Qwen3.5-0.8B, GPU)
```

**详细配置**: [`models/config.py`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py#L58-L65)

```python
ModelID.L2_CORE: ModelConfig(
    model_id=ModelID.L2_CORE,
    repo_id="models/Qwen/Qwen3___5-0___8B",  # 使用 Qwen3.5-0.8B 基础模型
    estimated_vram_gb=1.8,  # float16 精度显存占用
    is_expert=False,  # 基础模型（常驻）
    device="cuda",  # L2 运行在 GPU
    enabled=True  # 启用加载
)
```

**关键特性**:
- ✅ **模型**: Qwen3.5-0.8B（基础模型，float16 精度）
- ✅ **设备**: GPU (CUDA)
- ✅ **显存占用**: ~1.8 GB
- ✅ **类型**: 常驻模型（`is_expert=False`）
- ✅ **状态**: 启用（`enabled=True`）
- ✅ **本地路径**: `d:\AI\project\zulong_beta4\zulong\models\Qwen\Qwen3___5-0___8B`

**使用位置**:
- [`l2/inference_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/inference_engine.py#L56) - 推理引擎加载
- [`l2/vlm_agent.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/vlm_agent.py#L73) - VLM 智能体加载

---

### 2. **L2_BACKUP** - L2 备用热交换模型 ⭐⭐⭐⭐

**定义位置**: [`models/config.py`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py#L14)

```python
L2_BACKUP = "L2_BACKUP"  # L2 备用：热交换备用实例 (Qwen3.5-0.8B-INT4, GPU)
```

**详细配置**: [`models/config.py`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py#L67-L74)

```python
ModelID.L2_BACKUP: ModelConfig(
    model_id=ModelID.L2_BACKUP,
    repo_id="models/unsloth/Qwen3.5-0.8B",  # unsloth INT4 量化版本
    estimated_vram_gb=0.5,  # INT4 量化显存占用
    is_expert=False,  # 基础模型（常驻）
    device="cuda",  # L2_BACKUP 运行在 GPU
    enabled=True  # 启用加载
)
```

**关键特性**:
- ✅ **模型**: Qwen3.5-0.8B（unsloth INT4 量化版本）
- ✅ **设备**: GPU (CUDA)
- ✅ **显存占用**: ~0.5 GB（INT4 量化）
- ✅ **类型**: 常驻模型（`is_expert=False`）
- ✅ **状态**: 启用（`enabled=True`）
- ✅ **本地路径**: `d:\AI\project\zulong_beta4\zulong\models\unsloth\Qwen3.5-0.8B`
- ✅ **用途**: 热交换时的备用实例，显存占用更小

---

## 📊 L2 模型对比

| 特性 | L2_CORE | L2_BACKUP |
|------|---------|-----------|
| **模型 ID** | `L2_CORE` | `L2_BACKUP` |
| **模型名称** | Qwen3.5-0.8B | Qwen3.5-0.8B (unsloth INT4) |
| **精度** | float16 | INT4 量化 |
| **显存占用** | 1.8 GB | 0.5 GB |
| **设备** | GPU (CUDA) | GPU (CUDA) |
| **类型** | 常驻模型 | 常驻模型 |
| **状态** | 启用 | 启用 |
| **本地路径** | `models/Qwen/Qwen3___5-0___8B` | `models/unsloth/Qwen3.5-0.8B` |
| **主要用途** | 主推理模型 | 热交换备用 |

---

## 🔍 模型加载逻辑

### 加载位置

**文件**: [`models/container.py`](file:///d:/AI/project/zulong_beta4/zulong/models/container.py#L45-L93)

```python
def _load_resident_models(self):
    """加载所有常驻模型（is_expert=False 且 enabled=True）"""
    for model_id, config in MODEL_CONFIGS.items():
        # 跳过专家模型
        if config.is_expert:
            continue
        
        # 检查是否启用
        if not config.enabled:
            continue
        
        # 根据模型 ID 选择正确的本地模型路径
        if model_id == ModelID.L1_SCHEDULER:
            # L1-B: Qwen3.5-0.8B-Base (CPU)
            model_name = os.path.join(base_dir, "models", "Qwen", "Qwen3___5-0___8B-Base")
        elif model_id == ModelID.L2_CORE:
            # L2: Qwen3.5-0.8B (GPU)
            model_name = os.path.join(base_dir, "models", "Qwen", "Qwen3___5-0___8B")
        elif model_id == ModelID.L2_BACKUP:
            # L2_BACKUP: unsloth/Qwen3.5-0.8B (GPU)
            model_name = os.path.join(base_dir, "models", "unsloth", "Qwen3.5-0.8B")
        elif model_id == ModelID.EMBEDDING:
            # Embedding: BAAI/bge-small-zh-v1.5 (CPU)
            model_name = os.path.join(base_dir, "models", "BAAI", "bge-small-zh-v1.5")
        
        # 加载模型
        loader = RealModelLoader(model_name=model_name, device=device)
        if loader.load_model():
            self.resident_models[model_id] = loader
            self.current_vram_usage += config.estimated_vram_gb
```

---

### 使用位置

#### L2_CORE 使用

**文件**: [`l2/inference_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/inference_engine.py#L54-L58)

```python
def __init__(self):
    """初始化推理引擎"""
    self.model_container = ModelContainer()
    self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
    logger.info("L2 Inference Engine initialized with L2_CORE model")
```

**文件**: [`l2/inference_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/inference_engine.py#L101-L105)

```python
async def reload_l2_model(self):
    """重新加载 L2 模型（用于热交换后恢复）"""
    logger.info("Reloading L2 model...")
    self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
    logger.info("L2 model reloaded")
```

#### L2_BACKUP 使用

L2_BACKUP 模型主要用于**热交换场景**，当 L2_CORE 需要卸载时，可以使用 L2_BACKUP 作为临时替代。

---

## 📝 完整模型列表

系统中所有模型定义（[`models/config.py`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py#L9-L18)）：

```python
class ModelID(Enum):
    """模型 ID 枚举"""
    L1_SCHEDULER = "L1_SCHEDULER"  # L1-B: 调度与音频理解 (Qwen3.5-0.8B-Base, CPU)
    L2_CORE = "L2_CORE"  # L2: 推理决策 (Qwen3.5-0.8B, GPU)
    L2_BACKUP = "L2_BACKUP"  # L2 备用：热交换备用实例 (Qwen3.5-0.8B-INT4, GPU)
    EMBEDDING = "EMBEDDING"  # Embedding: RAG 向量生成 (CPU 运行)
    TTS_SYNTHESIS = "TTS_SYNTHESIS"  # TTS: 语音合成 (CPU 运行)
    EXPERT_NAV = "EXPERT_NAV"
    EXPERT_MANIPULATION = "EXPERT_MANIPULATION"
    EXPERT_VISION = "EXPERT_VISION"
```

---

## 🎯 总结

### L2 层两个模型的定义名称：

1. **主模型**: `L2_CORE`
   - 模型 ID: `ModelID.L2_CORE`
   - 模型名称：Qwen3.5-0.8B (float16)
   - 显存占用：1.8 GB
   - 用途：L2 核心推理

2. **备用模型**: `L2_BACKUP`
   - 模型 ID: `ModelID.L2_BACKUP`
   - 模型名称：Qwen3.5-0.8B (unsloth INT4 量化)
   - 显存占用：0.5 GB
   - 用途：热交换备用实例

---

## 🔍 相关文件

- [`models/config.py`](file:///d:/AI/project/zulong_beta4/zulong/models/config.py) - 模型配置定义
- [`models/container.py`](file:///d:/AI/project/zulong_beta4/zulong/models/container.py) - 模型单例容器
- [`models/engine.py`](file:///d:/AI/project/zulong_beta4/zulong/models/engine.py) - 模型加载引擎
- [`l2/inference_engine.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/inference_engine.py) - L2 推理引擎
- [`l2/l2_config.py`](file:///d:/AI/project/zulong_beta4/zulong/l2/l2_config.py) - L2 本地配置

---

**维护者**: 祖龙 (ZULONG) 系统架构组  
**文档版本**: v1.0  
**创建日期**: 2026-03-30
