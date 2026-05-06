# 祖龙 (ZULONG) 统一配置系统

## 📋 概述

祖龙系统现已实现**统一的配置管理系统**,将所有模块的硬编码配置迁移到集中管理的配置文件中，并通过环境变量提供灵活的配置覆盖机制。

## ✨ 核心特性

### 1. 🎯 统一配置文件
- **位置**: [`config/zulong_config.yaml`](config/zulong_config.yaml)
- **支持**: 多环境配置 (development/staging/production)
- **格式**: YAML，支持注释和结构化数据
- **功能**: 集中管理所有模块的配置参数

### 2. 🔧 环境变量覆盖
- **优先级**: 环境变量 > 配置文件 > 默认值
- **格式**: `ZULONG_<MODULE>_<PARAMETER>`
- **示例**: `ZULONG_LLM_BACKEND=ollama`

### 3. 🚀 多后端支持
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
│   ├── zulong_config.yaml    # 主配置文件 ⭐
│   ├── .env.example          # 环境变量示例
│   └── load_env.bat          # Windows 环境变量加载脚本
├── zulong/
│   ├── config/
│   │   └── config_manager.py # 配置管理器核心
│   ├── bootstrap.py          # 系统引导 (已更新)
│   └── models/
│       └── container.py      # 模型容器 (已更新)
├── tests/
│   └── test_config_system.py # 配置系统测试
└── docs/
    └── CONFIGURATION_GUIDE.md # 详细使用指南
```

## 🚀 快速开始

### 方式 1: 使用配置文件 (推荐)

1. **编辑配置文件** [`config/zulong_config.yaml`](config/zulong_config.yaml):

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

## 📊 配置示例

### LLM 后端配置

#### Ollama (默认)
```yaml
llm:
  backend: "ollama"
  ollama:
    base_url: "http://localhost:11434/v1"
    model_id: "deepseek-v3.1:671b-cloud"
    backup_model_id: "qwen3.5:4b"
```

#### vLLM
```yaml
llm:
  backend: "vllm"
  vllm:
    base_url: "http://localhost:8000/v1"
    model_id: "/path/to/Qwen3___5-0.8B-AWQ"
    gpu_memory_utilization: 0.5
    max_model_len: 4096
```

#### SGLang
```yaml
llm:
  backend: "sglang"
  sglang:
    base_url: "http://localhost:30000/v1"
    model_id: "/path/to/Qwen3___5-0.8B-AWQ"
    mem_fraction_static: 0.8
```

#### OpenAI 云端
```yaml
llm:
  backend: "openai"
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model_id: "gpt-4o-mini"
```

### L2 推理引擎配置
```yaml
l2_inference:
  core_model: "${llm.ollama.model_id}"
  backup_model: "${llm.ollama.backup_model_id}"
  
  generation:
    max_tokens: 1024
    temperature: 0.3
    top_p: 0.85
  
  timeout:
    core: 30
    backup: 30
```

### 视觉系统配置
```yaml
vision:
  camera:
    enabled: false
    device_index: 1
  
  yolo:
    model_path: "models/yolov10n.pt"
    device: "cuda"
```

### 音频系统配置
```yaml
audio:
  microphone:
    enabled: true
    sample_rate: 16000
  
  speaker:
    enabled: true
  
  tts:
    backend: "cosyvoice"
    model_path: "iic/CosyVoice2-0.5B"
```

### 记忆系统配置
```yaml
memory:
  short_term:
    max_turns: 100
    ttl: 3600
  
  rag:
    enabled: true
    embedding_model: "BAAI/bge-small-zh-v1.5"
    top_k: 5
  
  experience:
    enabled: true
    db_path: "./data/experience_db/experiences.db"
```

## 🔧 常用环境变量

### 系统基础
```bash
ZULONG_ENV=production              # 运行环境
ZULONG_LOG_LEVEL=INFO              # 日志级别
ZULONG_DEBUG_MODE=false            # 调试模式
```

### LLM 后端
```bash
ZULONG_LLM_BACKEND=ollama
ZULONG_OLLAMA_BASE_URL=http://localhost:11434/v1
ZULONG_OLLAMA_MODEL_ID=deepseek-v3.1:671b-cloud
ZULONG_OLLAMA_BACKUP_MODEL_ID=qwen3.5:4b
```

### vLLM (如使用)
```bash
ZULONG_VLLM_BASE_URL=http://localhost:8000/v1
ZULONG_VLLM_MODEL_ID=/path/to/model
ZULONG_VLLM_GPU_MEMORY_UTILIZATION=0.5
```

### L2 推理
```bash
ZULONG_L2_CORE_MODEL=deepseek-v3.1:671b-cloud
ZULONG_L2_BACKUP_MODEL=qwen3.5:4b
ZULONG_L2_TEMPERATURE=0.3
```

### 其他模块
```bash
ZULONG_CAMERA_ENABLED=false
ZULONG_MICROPHONE_ENABLED=true
ZULONG_RAG_ENABLED=true
ZULONG_OPENCLAW_ENABLED=true
```

## 🧪 测试配置系统

运行测试脚本验证配置系统:

```bash
python tests\test_config_system.py
```

**预期输出**:
```
======================================================================
  祖龙配置系统测试
======================================================================

[测试 1] 配置管理器初始化...
✅ 配置管理器初始化成功

[测试 2] 获取系统配置...
✅ 系统配置读取成功

[测试 3] 获取 LLM 配置...
✅ LLM 配置读取成功
   后端：ollama
   API 地址：http://localhost:11434/v1
   模型 ID: deepseek-v3.1:671b-cloud

...

✨ 所有测试通过！配置系统工作正常。
```

## 📚 详细文档

完整的使用指南请参考:
- 📖 [配置系统详细指南](docs/CONFIGURATION_GUIDE.md)
- 📋 [配置文件示例](config/zulong_config.yaml)
- 🔧 [环境变量模板](config/.env.example)

## 🔍 配置验证

### 检查配置文件语法
```bash
python -c "import yaml; yaml.safe_load(open('config/zulong_config.yaml'))"
```

### 查看当前配置
系统启动时会输出配置信息:
```
🔧 [CONFIG] LLM 后端：ollama
🤖 [CONFIG] LLM 后端：ollama
   API 地址：http://localhost:11434/v1
   模型 ID: deepseek-v3.1:671b-cloud
🧠 [CONFIG] L2 核心模型：deepseek-v3.1:671b-cloud
🧠 [CONFIG] L2 备用模型：qwen3.5:4b
```

### 测试配置加载
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
```

### 切换环境
```bash
# 方式 1: 环境变量
set ZULONG_ENV=production

# 方式 2: 配置文件
system:
  environment: "production"
```

## 📝 最佳实践

### 1. 版本控制
- ✅ 将 `zulong_config.yaml` 纳入版本控制
- ✅ 使用 `.env.example` 作为模板
- ❌ 不要将包含敏感信息的 `.env` 文件提交

### 2. 敏感信息管理
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

### 3. 配置文档化
- 在配置文件中添加详细注释
- 维护配置参数说明文档
- 记录配置变更历史

## 🔄 从旧版本迁移

### 迁移步骤

1. **备份现有配置**
2. **创建新配置文件**: 已创建 [`config/zulong_config.yaml`](config/zulong_config.yaml)
3. **迁移参数**: 将硬编码值复制到 YAML 配置文件
4. **测试验证**: `python tests\test_config_system.py`

### 兼容性保证

- ✅ 完全向后兼容旧的环境变量
- ✅ 自动检测并使用新配置系统
- ✅ 配置加载失败时回退到默认值

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
    model_id: "qwen3.5:0.8b"
```

**环境变量方式**:
```bash
set ZULONG_ENV=development
set ZULONG_DEBUG_MODE=true
set ZULONG_LOG_LEVEL=DEBUG
```

## 📞 支持与反馈

如有问题或建议:
1. 查看系统启动日志中的配置信息
2. 检查配置文件语法是否正确
3. 验证环境变量是否设置正确
4. 参考 [详细配置指南](docs/CONFIGURATION_GUIDE.md)

---

**文档版本**: v2.0  
**最后更新**: 2026-04-16  
**适用版本**: ZULONG v2.0+
