# Qwen3.5-2B-AWQ-4bit 快速部署指南

## 🎯 目标

使用 **AWQ 4bit 量化**版本加载 Qwen3.5-2B，配合 Qwen3.5-0.8B（全量），实现：
- ✅ 显存占用：~1.5-2.0 GB（减少 75%）
- ✅ 推理性能：吞吐量提升 40-50%
- ✅ 精度损失：控制在 2% 以内
- ✅ vLLM 原生支持

## 📦 模型信息

**模型来源**：
- 仓库：[cyankiwi/Qwen3.5-2B-AWQ-4bit](https://modelscope.cn/models/cyankiwi/Qwen3.5-2B-AWQ-4bit)
- 平台：ModelScope
- 量化方法：AWQ 4bit
- 文件大小：约 1.5 GB
- 兼容性：vLLM, SGLang, Transformers

## 🚀 快速开始（3 步）

### 步骤 1：下载 AWQ 量化模型

**方法 A：使用下载脚本（推荐）**

```batch
cd d:\AI\project\zulong_beta4
scripts\download_qwen3_5_2b_awq.bat
```

**方法 B：手动下载**

```bash
# 在 WSL2 中执行
source ~/vllm-env/bin/activate
modelscope download cyankiwi/Qwen3.5-2B-AWQ-4bit \
  --local_dir /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ
```

**方法 C：使用 modelscope-cli**

```batch
# Windows PowerShell
modelscope download cyankiwi/Qwen3.5-2B-AWQ-4bit ^
  --local_dir models\Qwen\Qwen3___5-2B-AWQ
```

### 步骤 2：启动 vLLM AWQ 服务

**运行启动脚本**：

```batch
cd d:\AI\project\zulong_beta4
scripts\start_vllm_wsl2_2b_awq.bat
```

**预期输出**：
```
==================================================
vLLM AWQ 量化配置
  - quantization: awq
  - gpu-memory-utilization: 0.8
  - max-model-len: 4096
==================================================

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 步骤 3：配置祖龙系统使用 vLLM

**设置环境变量**：

```batch
set USE_VLLM_FOR_L2=true
python -m zulong.main
```

**预期输出**：
```
[ModelContainer] [vLLM] L2_CORE 将使用 vLLM OpenAI API
[ModelContainer] [OK] L2_CORE vLLM 占位符注册成功
[ModelContainer] 初始化完成，当前显存使用：1.80/5.8GB
```

## ✅ 验证部署

### 1. 测试 vLLM API

```bash
curl http://localhost:8000/v1/models
```

**预期响应**：
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.5-2B-AWQ",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

### 2. 测试推理

```bash
curl http://localhost:8000/v1/completions ^
  -H "Content-Type: application/json" ^
  -d "{\"model\": \"Qwen3.5-2B-AWQ\", \"prompt\": \"你好\", \"max_tokens\": 50}"
```

### 3. 检查显存占用

**在 WSL2 中执行**：
```bash
nvidia-smi
```

**预期结果**：
- vLLM 进程显存占用：~1.5 GB
- 总显存占用：~1.8-2.0 GB（包括 0.8B 模型）

## 🔧 高级配置

### 1. 调整显存利用率

如果显存充足，可以提高利用率以获得更好性能：

```batch
wsl bash -c "
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ ^
  --port 8000 ^
  --tensor-parallel-size 1 ^
  --quantization awq ^
  --gpu-memory-utilization 0.9 ^
  --max-model-len 4096
"
```

### 2. 增加上下文长度

如果需要更长上下文：

```batch
wsl bash -c "
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ ^
  --port 8000 ^
  --tensor-parallel-size 1 ^
  --quantization awq ^
  --gpu-memory-utilization 0.85 ^
  --max-model-len 8192 ^
  --trust-remote-code
"
```

### 3. 启用 KV Cache 量化（进一步优化）

```batch
wsl bash -c "
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ ^
  --port 8000 ^
  --tensor-parallel-size 1 ^
  --quantization awq ^
  --kv-cache-dtype fp8_e4m3 ^
  --gpu-memory-utilization 0.9 ^
  --max-model-len 8192
"
```

## 📊 性能对比

| 配置 | 显存占用 | 推理速度 | 精度 | 推荐度 |
|------|---------|---------|------|--------|
| **AWQ 4bit** | ~1.5 GB | ~40-50 tokens/s | 98%+ | ⭐⭐⭐⭐⭐ |
| **Transformer INT4** | ~2.0 GB | ~20 tokens/s | 98%+ | ⭐⭐⭐⭐ |
| **vLLM FP16** | ~4.25 GB | ~50 tokens/s | 100% | ⭐⭐（显存不足） |
| **GGUF Q4_K_M** | ~1.3 GB | ~15 tokens/s | 95%+ | ⭐⭐⭐ |

## 🔍 故障排查

### 问题 1：模型文件不存在

**错误**：
```
ValueError: Model path does not exist: /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ
```

**解决**：
```bash
# 检查模型目录
ls /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ

# 重新下载
modelscope download cyankiwi/Qwen3.5-2B-AWQ-4bit \
  --local_dir /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ
```

### 问题 2：vLLM 不支持 AWQ

**错误**：
```
ValueError: Unknown quantization method: awq
```

**解决**：
```bash
# 升级 vLLM 到最新版本
source ~/vllm-env/bin/activate
pip install --upgrade vllm

# 确保版本 >= 0.3.0
pip show vllm
```

### 问题 3：显存不足

**错误**：
```
RuntimeError: CUDA out of memory
```

**解决**：
1. 降低 `--gpu-memory-utilization`（尝试 0.7）
2. 减少 `--max-model-len`（尝试 2048）
3. 关闭其他占用显存的程序

### 问题 4：模型加载失败

**错误**：
```
ValueError: Cannot find config.json in the model path
```

**解决**：
```bash
# 检查模型文件完整性
ls /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ/

# 应该包含：
# - config.json
# - quantize_config.json
# - *.safetensors
# - tokenizer.json
```

如果文件不完整，重新下载。

## 📝 总结

### 优势

1. ✅ **显存占用极低**：仅 ~1.5 GB
2. ✅ **推理速度快**：~40-50 tokens/s
3. ✅ **vLLM 原生支持**：无需额外配置
4. ✅ **精度损失小**：< 2%
5. ✅ **立即可用**：下载即用，无需自己量化

### 推荐配置

**日常使用**：
```bash
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization awq \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --trust-remote-code
```

**高性能需求**：
```bash
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-AWQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization awq \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

## 🔗 参考资料

- 模型页面：https://modelscope.cn/models/cyankiwi/Qwen3.5-2B-AWQ-4bit
- vLLM 量化文档：https://docs.vllm.ai/en/latest/quantization.html
- AWQ 论文：https://arxiv.org/abs/2306.00978
- 量化技术详解：`资料/vLLM 不仅支持量化加载.txt`
