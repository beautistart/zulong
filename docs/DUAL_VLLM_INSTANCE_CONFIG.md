# 双实例 vLLM 配置指南

## 📦 模型配置

### 模型文件结构
```
models/Qwen/
├── Qwen3___5-2B-AWQ/          ← L2_CORE 使用（主实例）
└── Qwen3___5-2B-AWQ-backup/   ← L2_BACKUP 使用（备份实例）
```

### 配置说明

| 模型实例 | 用途 | 服务端口 | 量化格式 | 显存占用 |
|---------|------|---------|---------|---------|
| **L2_CORE** | 主推理引擎 | 8000 | AWQ 4bit | ~1.5-2.0 GB |
| **L2_BACKUP** | 备份推理引擎 | 8001 | AWQ 4bit | ~1.5-2.0 GB |

## 🚀 启动步骤

### 方案 A：同时启动两个 vLLM 实例（推荐）

#### 1. 启动 L2_CORE 的 vLLM 服务

**方式 1：使用启动脚本**
```batch
scripts\start_vllm_wsl2_2b_awq.bat
```

**方式 2：手动启动**
```bash
wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization awq \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --trust-remote-code \
  --dtype auto
"
```

#### 2. 启动 L2_BACKUP 的 vLLM 服务

**方式 1：使用启动脚本**
```batch
scripts\start_vllm_wsl2_2b_awq_backup.bat
```

**方式 2：手动启动**
```bash
wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ-backup \
  --port 8001 \
  --tensor-parallel-size 1 \
  --quantization awq \
  --gpu-memory-utilization 0.4 \
  --max-model-len 4096 \
  --trust-remote-code \
  --dtype auto
"
```

#### 3. 启动祖龙系统

```bash
python -m zulong.main
```

## 🔧 配置参数说明

### L2_CORE 配置
- `--port 8000`: 服务端口
- `--gpu-memory-utilization 0.8`: GPU 内存利用率 80%
- `--quantization awq`: AWQ 4bit 量化
- `--max-model-len 4096`: 最大上下文长度

### L2_BACKUP 配置
- `--port 8001`: 服务端口（与 L2_CORE 隔离）
- `--gpu-memory-utilization 0.4`: GPU 内存利用率 40%（节省显存）
- `--quantization awq`: AWQ 4bit 量化
- `--max-model-len 4096`: 最大上下文长度

## 🎯 优势

### 1. 完全隔离
- ✅ 两个实例使用不同的模型文件
- ✅ 运行在不同的端口
- ✅ 互不影响，独立管理

### 2. 冗余备份
- ✅ L2_CORE 出问题时 L2_BACKUP 可立即接管
- ✅ 支持热切换
- ✅ 提高系统可靠性

### 3. 性能一致
- ✅ 相同的模型架构（Qwen3.5-2B）
- ✅ 相同的量化格式（AWQ 4bit）
- ✅ 相同的推理性能

### 4. 灵活配置
- ✅ 可单独启动 L2_CORE 或 L2_BACKUP
- ✅ 可同时运行两个实例
- ✅ 可根据需求调整显存分配

## ⚠️ 注意事项

### 显存要求
- **最小显存**: 3.0 GB（两个实例同时运行）
- **推荐显存**: 4.0 GB 以上
- **单个实例**: 1.5-2.0 GB

### 降级方案
如果显存不足，可以只启动 L2_CORE，L2_BACKUP 会自动降级为占位符模式。

## 🔍 验证连接

### 测试 L2_CORE
```bash
curl http://localhost:8000/v1/models
```

### 测试 L2_BACKUP
```bash
curl http://localhost:8001/v1/models
```

## 📝 环境变量配置

在 `.env` 文件中配置：
```bash
# 启用 vLLM for L2_CORE
USE_VLLM_FOR_L2=true

# 启用 vLLM for L2_BACKUP
USE_VLLM_FOR_L2_BACKUP=true

# vLLM 基础 URL（L2_CORE）
VLLM_BASE_URL=http://localhost:8000/v1

# L2_BACKUP 专用 URL
VLLM_BACKUP_BASE_URL=http://localhost:8001/v1
```

## 🛠️ 故障排除

### 问题 1: 端口已被占用
**解决**: 修改启动脚本中的 `--port` 参数

### 问题 2: 显存不足
**解决**: 
- 降低 `--gpu-memory-utilization` 参数
- 只启动一个 vLLM 实例
- 关闭其他 GPU 应用

### 问题 3: 模型加载失败
**解决**: 
- 检查模型路径是否正确
- 确认 WSL2 可以访问 Windows 文件系统
- 重新下载模型文件
