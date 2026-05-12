# vLLM 配置总结与 GGUF 格式说明

## 📋 配置测试结果

### ✅ 成功配置：Qwen3.5-0.8B

**启动命令**:
```bash
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8
```

**性能指标**:
- ✅ 模型加载内存：1.72 GiB
- ✅ 编译时间：~35 秒
- ✅ KV cache 容量：23,392 tokens
- ✅ 最大并发：15.82x (4096 tokens/request)
- ✅ 状态：**可稳定运行**

### ❌ 失败配置：Qwen3.5-2B

**尝试的配置**:
```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.45 \
  --dtype float16 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8
```

**失败原因**:
- ❌ 模型加载内存：4.25 GiB（已接近显存上限）
- ❌ 编译时间：~306 秒（是 0.8B 的 9 倍）
- ❌ 错误：`RuntimeError: Engine core initialization failed`
- ❌ 失败阶段：性能分析后，KV cache 初始化前
- ❌ 根本原因：**显存不足**

## 🔍 根本原因分析

### 显存需求计算

对于 Qwen3.5-2B:
1. **模型权重**: 4.25 GiB (float16)
2. **KV cache**: 需要为 2048 tokens 预留空间
3. **CUDA graphs**: 需要额外的显存用于编译后的图
4. **激活值**: 推理过程中的中间变量

**总需求**: 约 6-7 GiB
**可用显存**: 6.0 GiB × 0.45 = 2.7 GiB（严重不足）

即使设置 `--gpu-memory-utilization 0.45`，实际可用显存也远低于模型权重需求。

### 为什么 0.8B 可以运行？

- 模型权重：1.72 GiB
- 可用显存：6.0 GiB × 0.7 = 4.2 GiB
- 剩余显存：4.2 - 1.72 = 2.48 GiB（足够用于 KV cache 和 CUDA graphs）

## 💡 解决方案

### 方案 A：使用 0.8B 模型（强烈推荐）

**优势**:
- ✅ 已验证可稳定运行
- ✅ 编译速度快（35 秒 vs 300 秒）
- ✅ 显存占用低（1.72GB vs 4.25GB）
- ✅ 支持更长上下文（8192 tokens）
- ✅ 支持工具调用（auto-tool-choice）

**劣势**:
- ⚠️ 模型能力相对较弱（但在简单任务上表现良好）

**推荐场景**:
- 日常对话
- 简单问答
- 工具调用（网络搜索等）
- 文本生成

### 方案 B：使用量化版本的 2B 模型

如果 ModelScope 有 **AWQ** 或 **GPTQ** 量化版本：

```bash
# AWQ 量化版本（如果存在）
vllm serve Qwen/Qwen3.5-2B-AWQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --quantization awq \
  --gpu-memory-utilization 0.6

# GPTQ 量化版本（如果存在）
vllm serve Qwen/Qwen3.5-2B-GPTQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --quantization gptq \
  --gpu-memory-utilization 0.6
```

**优势**:
- 量化后显存需求降低 50-75%
- 推理速度更快

**劣势**:
- 精度略有损失
- 需要查找量化版本模型

### 方案 C：使用 CPU offload（实验性）

```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.3 \
  --enable-chunked-prefill \
  --cpu-offload
```

**优势**:
- 可以将部分模型参数卸载到 CPU 内存
- 降低 GPU 显存压力

**劣势**:
- 推理速度显著下降
- vLLM 的 CPU offload 支持有限

### 方案 D：升级硬件

**推荐配置**:
- GPU: RTX 3060 12GB / RTX 4060 Ti 16GB
- 内存：16GB+
- 存储：NVMe SSD（加速模型加载）

## ❓ vLLM 是否支持 GGUF 格式？

### 答案：**不支持**

vLLM **不直接支持 GGUF 格式**的模型。

### vLLM 支持的格式

1. **Safetensors**（推荐）
   - Hugging Face 标准格式
   - 加载速度快，安全性高
   - ModelScope/HuggingFace 上的模型主要格式

2. **PyTorch (.bin / .pt)**
   - 传统 PyTorch 格式
   - 兼容性最好
   - 加载速度较慢

3. **AWQ / GPTQ / FP8**（量化格式）
   - 需要特定的 `--quantization` 参数
   - 显存占用更低
   - 推理速度更快

### GGUF 是什么？

**GGUF (GGML Unified Format)** 是 **llama.cpp** 项目专用的模型格式：
- 专为 CPU 推理优化
- 支持多种量化精度（Q4_K_M, Q5_K_M, Q8_0 等）
- 可以在纯 CPU 环境下运行大模型
- 与 vLLM 的 GPU 优化路线不同

### 如果需要使用 GGUF 格式

**选项 1: 使用 llama.cpp**
```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 运行 GGUF 模型
./server -m models/Qwen3.5-2B-Q4_K_M.gguf \
  --port 8080 \
  --ctx-size 4096 \
  --n-gpu-layers 35
```

**优势**:
- 支持 GGUF 格式
- 可以在 CPU 上运行
- 支持 GPU 卸载（部分层）

**劣势**:
- 推理速度比 vLLM 慢
- 不支持 vLLM 的高级特性（PagedAttention、连续批处理等）
- API 接口不同

**选项 2: 转换 GGUF 为 Safetensors**
```bash
# 使用转换工具（如果存在）
python convert_gguf_to_safetensors.py \
  --input model.gguf \
  --output model_safetensors/
```

**注意**: 这种转换可能不可行，因为 GGUF 是量化格式，而 Safetensors 通常是全精度格式。

**选项 3: 直接使用 HuggingFace/ModelScope 的原始模型**
- 下载 Safetensors 版本
- 这正是我们目前正在使用的方式

## 📝 推荐配置清单

### 日常使用（推荐）

**模型**: Qwen3.5-0.8B
**启动脚本**: `scripts/start_vllm_wsl2.bat`
**用途**: 日常对话、工具调用、简单问答

### 需要更强能力时

**方案 1**: 使用在线 API（如 Qwen API、DeepSeek API）
**方案 2**: 升级 GPU 到 12GB+ 显存
**方案 3**: 使用 llama.cpp + GGUF 量化模型（CPU 推理）

## 🔧 快速启动指南

### 启动 0.8B 模型（推荐）

```batch
# Windows 批处理
scripts\start_vllm_wsl2.bat
```

或在 WSL2 中直接运行：
```bash
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8
```

### 测试连接

```bash
# 测试 API
curl http://localhost:8000/v1/models

# 测试对话
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 📊 性能对比

| 模型 | 显存占用 | 编译时间 | KV cache 容量 | 稳定性 | 推荐度 |
|------|---------|---------|--------------|--------|--------|
| Qwen3.5-0.8B | 1.72 GB | 35 秒 | 23,392 tokens | ✅ 稳定 | ⭐⭐⭐⭐⭐ |
| Qwen3.5-2B | 4.25 GB | 306 秒 | 无法初始化 | ❌ 崩溃 | ⭐ |

## 🎯 结论

**强烈建议使用 Qwen3.5-0.8B**，原因：
1. ✅ 已验证可稳定运行
2. ✅ 性能足够应对日常任务
3. ✅ 支持工具调用和网络搜索
4. ✅ 编译速度快，启动迅速
5. ✅ 显存占用低，不会崩溃

**不建议使用 Qwen3.5-2B**，原因：
1. ❌ 显存不足，无法稳定运行
2. ❌ 编译时间过长（5 分钟）
3. ❌ 即使成功启动，性能提升有限
4. ❌ 可能频繁崩溃

**如果确实需要 2B 的能力**，建议：
- 使用在线 API
- 升级硬件（12GB+ GPU）
- 使用 llama.cpp + GGUF 量化模型（CPU 推理）
