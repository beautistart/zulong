# vLLM Qwen3.5-2B 配置指南

## 问题总结

在 WSL2 环境中尝试启动 Qwen3.5-2B 模型时遇到以下问题：

1. **显存不足**: 2B 模型需要约 4.25 GiB 显存加载权重，加上 KV cache 和 CUDA graphs，总需求超过 6GB
2. **Engine Core 初始化失败**: 在 torch.compile 完成后，初始化 KV cache 时失败
3. **CUDA graph 内存分配失败**: 性能分析阶段后崩溃

## 已测试的配置组合

### ✅ 成功配置：Qwen3.5-0.8B
```bash
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.7 \
  --trust-remote-code
```
- 模型加载内存：1.72 GiB
- 编译时间：~35 秒
- KV cache tokens: 23,392
- 状态：**可稳定运行**

### ❌ 失败配置 1: Qwen3.5-2B (标准)
```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --dtype float16 \
  --trust-remote-code
```
- 模型加载内存：4.25 GiB
- 编译时间：~274 秒
- 失败原因：GPU memory utilization 设置过高

### ❌ 失败配置 2: Qwen3.5-2B (低显存)
```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.5 \
  --dtype float16 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16
```
- 模型加载内存：4.25 GiB
- 编译时间：~305 秒
- 失败原因：Engine core initialization failed

### ❌ 失败配置 3: Qwen3.5-2B (极简)
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
- 模型加载内存：4.25 GiB
- 编译时间：~306 秒
- 失败原因：Engine core initialization failed (性能分析后崩溃)

## 推荐解决方案

### 方案 A：使用 0.8B 模型（推荐）
**优点**：
- 稳定运行，无崩溃
- 编译时间短（35 秒 vs 300 秒）
- 显存占用低（1.72GB vs 4.25GB）
- 可支持更长的上下文（23K tokens）

**缺点**：
- 模型能力较弱（0.8B vs 2B）

### 方案 B：使用 AWQ 量化版本的 2B 模型
如果 ModelScope 有 AWQ 量化版本的 Qwen3.5-2B，可以尝试：
```bash
vllm serve Qwen/Qwen3.5-2B-AWQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --quantization awq \
  --gpu-memory-utilization 0.6
```

### 方案 C：使用 CPU offload
尝试将部分模型参数加载到 CPU：
```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.4 \
  --enable-chunked-prefill \
  --cpu-offload
```

### 方案 D：使用更小的上下文窗口
```bash
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.4 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 4
```

## 启动脚本

### 0.8B 模型启动脚本（推荐）
文件：`scripts/start_vllm_wsl2.bat`
```batch
@echo off
wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve Qwen/Qwen3.5-0.8B ^
  --port 8000 ^
  --tensor-parallel-size 1 ^
  --max-model-len 8192 ^
  --enable-auto-tool-choice ^
  --tool-call-parser qwen3_coder ^
  --dtype bfloat16 ^
  --gpu-memory-utilization 0.8
"
```

### 2B 模型启动脚本（实验性）
文件：`scripts/start_vllm_wsl2_2b.bat`
```batch
@echo off
wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve Qwen/Qwen3.5-2B ^
  --port 8000 ^
  --tensor-parallel-size 1 ^
  --max-model-len 4096 ^
  --enable-auto-tool-choice ^
  --tool-call-parser hermes ^
  --dtype float16 ^
  --gpu-memory-utilization 0.5 ^
  --max-num-batched-tokens 4096 ^
  --max-num-seqs 16
"
```

## 关键参数说明

### `--gpu-memory-utilization`
- **作用**: 控制 GPU 显存使用比例
- **范围**: 0.0 - 1.0
- **推荐值**: 
  - 0.8B 模型：0.7-0.8
  - 2B 模型：0.4-0.5（需要更多测试）

### `--max-model-len`
- **作用**: 最大上下文长度
- **影响**: 直接影响 KV cache 大小
- **推荐值**:
  - 0.8B 模型：4096-8192
  - 2B 模型：1024-2048（保守设置）

### `--dtype`
- **作用**: 模型精度
- **选项**: float16, bfloat16, float32
- **推荐**: 
  - float16: 兼容性最好
  - bfloat16: 需要 Ampere 架构 GPU

### `--max-num-seqs`
- **作用**: 最大并发序列数
- **影响**: 影响并发性能和显存占用
- **推荐值**: 8-16（保守设置）

## 下一步建议

1. **优先使用 0.8B 模型**: 已经验证可以稳定运行
2. **查找 AWQ 量化版本**: 在 ModelScope 搜索 Qwen3.5-2B 的量化版本
3. **测试 CPU offload**: 尝试将部分计算卸载到 CPU
4. **考虑升级硬件**: 如果需要运行 2B 模型，建议使用 8GB+ 显存的 GPU

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- Qwen3.5 ModelScope: https://modelscope.cn/models/Qwen/Qwen3.5-2B
- vLLM GitHub: https://github.com/vllm-project/vllm
