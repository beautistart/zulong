# 祖龙 (ZULONG) 统一配置系统使用指南

## 📋 概述

祖龙系统现已实现统一的配置管理系统，将所有模块的硬编码配置迁移到集中管理的配置文件中，并通过环境变量提供灵活的配置覆盖机制。

## 🎯 核心特性

### 1. 统一配置文件
- **位置**: `config/zulong_config.yaml`
- **支持**: 多环境配置 (development/staging/production)
- **格式**: YAML，支持注释和结构化数据
- **功能**: 集中管理所有模块的配置参数

### 2. 环境变量覆盖
- **优先级**: 环境变量 > 配置文件 > 默认值
- **格式**: `ZULONG_<MODULE>_<PARAMETER>`
- **示例**: `ZULONG_LLM_BACKEND=ollama`

### 3. 多后端支持
支持以下 LLM 推理后端:
- ✅ **Ollama** (默认) - 本地轻量级部署
- ✅ **vLLM** - 高性能推理引擎
- ✅ **SGLang** - 高效推理框架
- ✅ **llama.cpp** - CPU/GPU 混合推理
- ✅ **LM Studio** - 桌面应用集成
- ✅ **OpenAI** - 云端 API 服务

## 📁 文件结构

```
zulong_beta4/
├── config/
│   ├── zulong_config.yaml    # 主配置文件
│   ├── .env.example          # 环境变量示例
│   └── load_env.bat          # Windows 环境变量加载脚本
├── zulong/
│   ├── config/
│   │   └── config_manager.py # 配置管理器
│   ├── bootstrap.py          # 系统引导 (已更新)
│   └── models/
│       └── container.py      # 模型容器 (已更新)
└── docs/
    └── CONFIGURATION_GUIDE.md # 本文档
```

## 🚀 快速开始

### 方式 1: 使用配置文件 (推荐)

1. **编辑配置文件** `config/zulong_config.yaml`:

```yaml
llm:
  backend: "ollama"  # 或 vllm/sglang/llamaccp/openai
  ollama:
    base_url: "http://localhost:11434/v1"
    model_id: "deepseek-v3.1:671b-cloud"
    backup_model_id: "qwen3.5:4b"
```

2. **启动系统**:
```bash
python -m zulong.bootstrap
```

### 方式 2: 使用环境变量

1. **加载环境变量** (Windows):
```powershell
# 使用默认配置
call config\load_env.bat

# 或手动设置
set ZULONG_LLM_BACKEND=ollama
set ZULONG_OLLAMA_MODEL_ID=deepseek-v3.1:671b-cloud
```

2. **启动系统**:
```bash
python -m zulong.bootstrap
```

### 方式 3: 混合模式

配置文件 + 环境变量覆盖:

```bash
# 基础配置来自配置文件
# 特定参数通过环境变量覆盖
set ZULONG_LLM_BACKEND=vllm
set ZULONG_VLLM_BASE_URL=http://localhost:8000/v1
python -m zulong.bootstrap
```

## ⚙️ 配置详解

### 1. LLM 后端配置

#### Ollama 配置
```yaml
llm:
  backend: "ollama"
  ollama:
    base_url: "http://localhost:11434/v1"
    model_id: "deepseek-v3.1:671b-cloud"  # 主模型
    backup_model_id: "qwen3.5:4b"         # 备用模型
```

#### vLLM 配置
```yaml
llm:
  backend: "vllm"
  vllm:
    base_url: "http://localhost:8000/v1"
    model_id: "/path/to/Qwen3___5-0.8B-AWQ"
    gpu_memory_utilization: 0.5
    max_model_len: 4096
```

#### SGLang 配置
```yaml
llm:
  backend: "sglang"
  sglang:
    base_url: "http://localhost:30000/v1"
    model_id: "/path/to/Qwen3___5-0.8B-AWQ"
    mem_fraction_static: 0.8
```

#### llama.cpp 配置
```yaml
llm:
  backend: "llamaccp"
  llamaccp:
    base_url: "http://localhost:8080/v1"
    model_id: "qwen3.5-4b"
    n_ctx: 4096
    n_threads: 8
```

#### OpenAI 云端配置
```yaml
llm:
  backend: "openai"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"  # 从环境变量读取
    model_id: "gpt-4o-mini"
```

### 2. L2 推理引擎配置

```yaml
l2_inference:
  # 模型引用 (从 llm 配置继承)
  core_model: "${llm.ollama.model_id}"
  backup_model: "${llm.ollama.backup_model_id}"
  
  # 生成参数
  generation:
    max_tokens: 1024
    temperature: 0.3
    top_p: 0.85
    repetition_penalty: 1.2
  
  # 超时配置 (秒)
  timeout:
    core: 30
    backup: 30
  
  # 重试配置
  retry:
    max_attempts: 3
    delay: 1.0
```

### 3. 视觉系统配置

```yaml
vision:
  camera:
    enabled: false
    device_index: 1
    resolution: [1280, 720]
  
  yolo:
    model_path: "yolov10n.pt"
    device: "cuda"
    confidence_threshold: 0.25
```

### 4. 音频系统配置

```yaml
audio:
  microphone:
    enabled: true
    sample_rate: 16000
  
  speaker:
    enabled: true
    sample_rate: 44100
  
  tts:
    backend: "cosyvoice"
    model_path: "iic/CosyVoice2-0.5B"
```

### 5. 记忆系统配置

```yaml
memory:
  short_term:
    max_turns: 100
    ttl: 3600  # 秒
  
  rag:
    enabled: true
    data_dir: "./data/rag"
    embedding_model: "BAAI/bge-small-zh-v1.5"
    top_k: 5
  
  experience:
    enabled: true
    db_path: "./data/experience_db/experiences.db"
```

## 🔧 环境变量列表

### 系统基础
```bash
ZULONG_ENV=production              # 运行环境
ZULONG_LOG_LEVEL=INFO              # 日志级别
ZULONG_DEBUG_MODE=false            # 调试模式
ZULONG_DATA_DIR=./data             # 数据目录
ZULONG_MODELS_DIR=./models         # 模型目录
```

### LLM 后端
```bash
# 通用配置
ZULONG_LLM_BACKEND=ollama

# Ollama 配置
ZULONG_OLLAMA_BASE_URL=http://localhost:11434/v1
ZULONG_OLLAMA_MODEL_ID=deepseek-v3.1:671b-cloud
ZULONG_OLLAMA_BACKUP_MODEL_ID=qwen3.5:4b

# vLLM 配置
ZULONG_VLLM_BASE_URL=http://localhost:8000/v1
ZULONG_VLLM_MODEL_ID=/path/to/model
ZULONG_VLLM_GPU_MEMORY_UTILIZATION=0.5

# SGLang 配置
ZULONG_SGLANG_BASE_URL=http://localhost:30000/v1
ZULONG_SGLANG_MODEL_ID=/path/to/model

# llama.cpp 配置
ZULONG_LLAMACCP_BASE_URL=http://localhost:8080/v1
ZULONG_LLAMACCP_MODEL_ID=qwen3.5-4b

# OpenAI 配置
ZULONG_OPENAI_BASE_URL=https://api.openai.com/v1
ZULONG_OPENAI_MODEL_ID=gpt-4o-mini
ZULONG_OPENAI_API_KEY=sk-xxx
```

### L2 推理
```bash
ZULONG_L2_CORE_MODEL=deepseek-v3.1:671b-cloud
ZULONG_L2_BACKUP_MODEL=qwen3.5:4b
ZULONG_L2_MAX_TOKENS=1024
ZULONG_L2_TEMPERATURE=0.3
ZULONG_L2_CORE_TIMEOUT=30
```

### 视觉系统
```bash
ZULONG_CAMERA_ENABLED=false
ZULONG_YOLO_MODEL_PATH=yolov10n.pt
```

### 音频系统
```bash
ZULONG_MICROPHONE_ENABLED=true
ZULONG_SPEAKER_ENABLED=true
ZULONG_TTS_BACKEND=cosyvoice
```

### 记忆系统
```bash
ZULONG_RAG_ENABLED=true
ZULONG_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

### 工具系统
```bash
ZULONG_OPENCLAW_ENABLED=true
ZULONG_OPENCLAW_API_URL=http://localhost:3000
ZULONG_WEB_SEARCH_ENABLED=true
```

### Web 服务
```bash
ZULONG_API_HOST=localhost
ZULONG_API_PORT=3000
ZULONG_WEBSOCKET_HOST=localhost
ZULONG_WEBSOCKET_PORT=5555
```

### 安全
```bash
ZULONG_API_KEY=your-secret-key
OPENAI_API_KEY=sk-your-openai-key
```

## 📊 配置优先级

配置加载优先级 (从高到低):

1. **环境变量** - 最高优先级，用于临时覆盖
2. **配置文件** - 主要配置来源
3. **默认值** - 内置默认配置

示例:
```bash
# 配置文件中设置:
# llm.backend = "ollama"

# 环境变量覆盖:
set ZULONG_LLM_BACKEND=vllm

# 结果：系统使用 vllm 后端
```

## 🔍 配置验证

### 1. 检查配置文件语法
```bash
python -c "import yaml; yaml.safe_load(open('config/zulong_config.yaml'))"
```

### 2. 查看当前配置
在系统启动日志中查看:
```
🔧 [CONFIG] LLM 后端：ollama
🤖 [CONFIG] LLM 后端：ollama
   API 地址：http://localhost:11434/v1
   模型 ID: deepseek-v3.1:671b-cloud
🧠 [CONFIG] L2 核心模型：deepseek-v3.1:671b-cloud
🧠 [CONFIG] L2 备用模型：qwen3.5:4b
```

### 3. 测试配置加载
```python
from zulong.config.config_manager import get_config, get_llm_config

# 获取任意配置值
backend = get_config('llm.backend')
print(f"LLM 后端：{backend}")

# 获取 LLM 完整配置
llm_config = get_llm_config()
print(f"API 地址：{llm_config['base_url']}")
```

## 🎛️ 多环境配置

### 开发环境
```yaml
# config/zulong_config.yaml
environments:
  development:
    system:
      debug_mode: true
      log_level: "DEBUG"
    llm:
      backend: "ollama"
      ollama:
        model_id: "qwen3.5:0.8b"  # 使用小模型加速开发
```

### 生产环境
```yaml
environments:
  production:
    system:
      debug_mode: false
      log_level: "INFO"
    llm:
      backend: "vllm"  # 使用高性能后端
      vllm:
        model_id: "/path/to/Qwen3___5-0.8B-AWQ"
```

### 切换环境
```bash
# 方式 1: 环境变量
set ZULONG_ENV=production

# 方式 2: 配置文件
system:
  environment: "production"
```

## 🔧 故障排查

### 问题 1: 配置文件未找到
**现象**: 系统使用默认配置，忽略自定义设置

**解决**:
1. 检查 `config/zulong_config.yaml` 是否存在
2. 设置环境变量 `ZULONG_CONFIG` 指向配置文件:
   ```bash
   set ZULONG_CONFIG=D:\AI\project\zulong_beta4\config\zulong_config.yaml
   ```

### 问题 2: 环境变量未生效
**现象**: 配置值与预期不符

**解决**:
1. 确认环境变量名称正确 (全部大写，下划线分隔)
2. 检查环境变量是否在启动前设置
3. 使用 `echo %VARIABLE%` 验证环境变量值

### 问题 3: 配置冲突
**现象**: 多个配置源产生冲突

**解决**:
1. 遵循优先级规则：环境变量 > 配置文件 > 默认值
2. 避免在多处设置同一参数
3. 使用统一配置管理系统

## 📝 最佳实践

### 1. 版本控制
- ✅ 将 `zulong_config.yaml` 纳入版本控制
- ✅ 使用 `.env.example` 作为模板
- ❌ 不要将包含敏感信息的 `.env` 文件提交

### 2. 环境隔离
- 为不同环境创建独立配置文件:
  - `config/development.yaml`
  - `config/production.yaml`
- 通过 `ZULONG_CONFIG` 环境变量切换

### 3. 敏感信息管理
```yaml
# 配置文件中使用环境变量占位符
llm:
  openai:
    api_key: "${OPENAI_API_KEY}"  # 从系统环境变量读取
```

```bash
# 在操作系统中设置
set OPENAI_API_KEY=sk-your-secret-key
```

### 4. 配置文档化
- 在配置文件中添加详细注释
- 维护配置参数说明文档
- 记录配置变更历史

## 🔄 从旧版本迁移

### 迁移步骤

1. **备份现有配置**:
   ```bash
   cp zulong/models/container.py zulong/models/container.py.backup
   ```

2. **创建新配置文件**:
   ```bash
   cp config/zulong_config.yaml.example config/zulong_config.yaml
   ```

3. **迁移参数**:
   - 将硬编码值复制到 YAML 配置文件
   - 更新模块导入配置管理器

4. **测试验证**:
   ```bash
   python -m zulong.bootstrap
   ```

### 兼容性保证

- ✅ 完全向后兼容旧的环境变量
- ✅ 自动检测并使用新配置系统
- ✅ 配置加载失败时回退到默认值

## 📚 API 参考

### ConfigManager 类

```python
from zulong.config.config_manager import ConfigManager

# 获取实例
config = ConfigManager()

# 获取配置值
value = config.get('llm.ollama.model_id')
value = config.get('system.debug_mode', default=False)

# 类型安全的方法
int_val = config.get_int('l2_inference.max_tokens')
float_val = config.get_float('llm.vllm.gpu_memory_utilization')
bool_val = config.get_bool('system.debug_mode')
list_val = config.get_list('l2_inference.visual_keywords')
dict_val = config.get_dict('memory.rag')

# 重新加载配置
config.reload()
```

### 便捷函数

```python
from zulong.config.config_manager import (
    get_config,
    get_llm_config,
    get_l2_inference_config,
    get_memory_config,
    get_vision_config,
    get_audio_config,
)

# 获取 LLM 配置
llm_config = get_llm_config()  # 自动使用配置的 backend
llm_config = get_llm_config('vllm')  # 指定 backend

# 获取其他模块配置
l2_config = get_l2_inference_config()
memory_config = get_memory_config()
vision_config = get_vision_config()
audio_config = get_audio_config()
```

## 🎓 示例场景

### 场景 1: 切换到 vLLM 后端

**配置文件方式**:
```yaml
llm:
  backend: "vllm"
  vllm:
    base_url: "http://localhost:8000/v1"
    model_id: "/path/to/Qwen3___5-0.8B-AWQ"
```

**环境变量方式**:
```bash
set ZULONG_LLM_BACKEND=vllm
set ZULONG_VLLM_BASE_URL=http://localhost:8000/v1
set ZULONG_VLLM_MODEL_ID=/path/to/Qwen3___5-0.8B-AWQ
```

### 场景 2: 使用云端 DeepSeek

**配置文件方式**:
```yaml
llm:
  backend: "openai"
  openai:
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model_id: "deepseek-chat"
```

**环境变量方式**:
```bash
set ZULONG_LLM_BACKEND=openai
set ZULONG_OPENAI_BASE_URL=https://api.deepseek.com/v1
set DEEPSEEK_API_KEY=your-api-key
```

### 场景 3: 开发模式调试

**配置文件方式**:
```yaml
system:
  environment: "development"
  debug_mode: true
  log_level: "DEBUG"

llm:
  backend: "ollama"
  ollama:
    model_id: "qwen3.5:0.8b"  # 小模型加速开发
```

**环境变量方式**:
```bash
set ZULONG_ENV=development
set ZULONG_DEBUG_MODE=true
set ZULONG_LOG_LEVEL=DEBUG
set ZULONG_LLM_BACKEND=ollama
set ZULONG_OLLAMA_MODEL_ID=qwen3.5:0.8b
```

## 📞 支持与反馈

如有问题或建议，请:
1. 查看系统启动日志中的配置信息
2. 检查配置文件语法是否正确
3. 验证环境变量是否设置正确
4. 参考本文档的配置示例

---

**文档版本**: v2.0  
**最后更新**: 2026-04-16  
**适用版本**: ZULONG v2.0+
