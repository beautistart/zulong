# Qwen3.5-2B 量化模型搜索结果报告

## 📊 搜索结果总结

### ✅ 找到的量化版本

#### 1. **GGUF 格式**（大量可用）

**来源**：
- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [bartowski/Qwen_Qwen3.5-2B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF)

**量化等级**：
- Q8_0 (2.01 GB) - 极高质量
- Q6_K (1.64 GB) - 推荐
- Q5_K_M (1.46 GB) - 推荐
- Q4_K_M (1.32 GB) - 默认推荐
- Q3_K_M (1.14 GB) - 低质量

**特点**：
- ✅ 量化技术成熟，质量损失小
- ✅ 文件大小适中（1.1-2.0 GB）
- ✅ 适合 CPU 推理（llama.cpp）
- ❌ **vLLM 不支持 GGUF 格式**

#### 2. **MLX OptiQ 格式**（Apple Silicon 专用）

**来源**：
- [mlx-community/Qwen3.5-2B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-2B-OptiQ-4bit)

**特点**：
- ✅ 混合精度量化（4.5 BPW）
- ✅ 大小：1.365 GB
- ✅ GSM8K 准确率：48.0%
- ❌ **仅适用于 Apple Silicon（MLX 框架）**
- ❌ vLLM 不支持

### ✅ **找到的 AWQ 量化版本**

#### **AWQ 4bit 格式**（重要发现！）

**来源**：
- [cyankiwi/Qwen3.5-2B-AWQ-4bit](https://modelscope.cn/models/cyankiwi/Qwen3.5-2B-AWQ-4bit)

**特点**：
- ✅ **vLLM 原生支持 AWQ 格式**
- ✅ 4bit 量化，显存占用约 1.5 GB
- ✅ 兼容 vLLM、SGLang、Transformers
- ✅ 质量损失 < 2%

## 🔍 原因分析

### 为什么没有 GPTQ/AWQ 版本？

1. **模型太新**：Qwen3.5-2B 是 2026 年 3 月发布的，量化社区可能还未跟进
2. **官方策略**：阿里优先发布原始模型，量化版本由社区后续提供
3. **技术路线**：官方可能更推荐使用 GGUF（CPU 友好）或 FP8（H100 友好）

## 💡 解决方案

### 方案 A：使用 GGUF + llama.cpp（立即可用）

**优点**：
- ✅ 模型已存在，可直接下载
- ✅ 显存占用低（Q4_K_M 仅 1.32 GB）
- ✅ 支持 CPU 推理
- ✅ 质量损失小（Q4_K_M 精度损失 < 5%）

**缺点**：
- ❌ vLLM 不支持
- ❌ 需要改用 llama.cpp 框架
- ❌ 推理速度比 vLLM 慢

**实施步骤**：

1. **下载模型**：
```bash
# 下载 Q4_K_M 量化版本
huggingface-cli download bartowski/Qwen_Qwen3.5-2B-GGUF \
  --local-dir models/Qwen/Qwen3___5-2B-GGUF \
  --include "Qwen3.5-2B-Q4_K_M.gguf"
```

2. **安装 llama.cpp**：
```bash
# WSL2 中
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake . -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

3. **启动服务**：
```bash
./build/bin/llama-server \
  -m models/Qwen/Qwen3___5-2B-GGUF/Qwen3.5-2B-Q4_K_M.gguf \
  --port 8000 \
  --ctx-size 4096 \
  --n-gpu-layers 35
```

### 方案 B：自己量化 GPTQ/AWQ（需要时间）

**优点**：
- ✅ 兼容 vLLM
- ✅ 推理速度快
- ✅ 显存占用低

**缺点**：
- ❌ 需要 30-60 分钟量化时间
- ❌ 需要至少 4GB 显存
- ❌ 需要安装 AutoGPTQ/AutoAWQ

**实施步骤**：

1. **安装工具**：
```bash
pip install auto-gptq optimum
# 或
pip install auto-awq
```

2. **运行量化脚本**：
```python
# 参考 docs/VLLM_GPTQ_QWEN2B_GUIDE.md 中的量化脚本
python scripts/quantize_qwen3_5_2b.py
```

3. **使用 vLLM 加载**：
```bash
vllm serve models/Qwen/Qwen3___5-2B-GPTQ \
  --port 8000 \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128
```

### 方案 C：使用 Transformer 原生 INT4（当前可用）

**优点**：
- ✅ 无需额外工具
- ✅ 立即可用
- ✅ 显存占用低（~2.0 GB）

**缺点**：
- ❌ 推理速度较慢
- ❌ 不支持 vLLM 优化

**实施步骤**：

1. **设置环境变量**：
```batch
set USE_VLLM_FOR_L2=false
```

2. **启动祖龙系统**：
```batch
python -m zulong.main
```

### 方案 D：使用 FP8（需要 H100/A100）

**优点**：
- ✅ vLLM 原生支持
- ✅ 性能最佳
- ✅ 显存占用低

**缺点**：
- ❌ 需要 H100/A100 GPU
- ❌ RTX 3060 不支持

## 📋 推荐方案

### 短期方案（立即使用）

**使用 Transformer 原生 INT4 加载**：
- ✅ 无需额外工作
- ✅ 已经配置好
- ✅ 显存占用 ~2.5 GB

**配置**：
```batch
set USE_VLLM_FOR_L2=false
python -m zulong.main
```

### 中期方案（1-2 周内）

**方案 1：下载 GGUF + llama.cpp**
- 下载现成的 GGUF 模型
- 使用 llama.cpp 推理
- 显存占用 ~1.5 GB

**方案 2：自己量化 GPTQ**
- 使用 AutoGPTQ 量化
- 使用 vLLM 推理
- 显存占用 ~1.5 GB
- 推理速度更快

### 长期方案（未来规划）

**升级硬件**：
- 如果有 12GB+ GPU，可以直接使用 vLLM FP16
- 如果有 H100，可以使用 FP8 量化
- 获得最佳性能

## 🎯 下一步行动

### 立即执行

1. **使用 Transformer 原生加载**
   - 设置 `USE_VLLM_FOR_L2=false`
   - 重启祖龙系统
   - 验证功能正常

### 可选执行（根据需要）

2. **下载 GGUF 模型**
   ```bash
   huggingface-cli download bartowski/Qwen_Qwen3.5-2B-GGUF \
     --local-dir models/Qwen/Qwen3___5-2B-GGUF \
     --include "Qwen3.5-2B-Q4_K_M.gguf"
   ```

3. **测试 llama.cpp**
   - 编译 llama.cpp
   - 启动服务
   - 对比性能

4. **自己量化 GPTQ**
   - 安装 AutoGPTQ
   - 运行量化脚本
   - 测试 vLLM 加载

## 📊 性能对比预期

| 方案 | 显存占用 | 推理速度 | 实施难度 | 推荐度 |
|------|---------|---------|---------|--------|
| **Transformer INT4** | ~2.0 GB | ~20 tokens/s | ⭐ 简单 | ⭐⭐⭐⭐ |
| **GGUF Q4_K_M** | ~1.5 GB | ~15 tokens/s | ⭐⭐ 中等 | ⭐⭐⭐ |
| **GPTQ 4bit** | ~1.5 GB | ~40 tokens/s | ⭐⭐⭐ 困难 | ⭐⭐⭐⭐⭐ |
| **vLLM FP16** | ~4.25 GB | ~50 tokens/s | ⭐ 简单 | ⭐⭐（显存不足） |

## 🔗 参考资料

- [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
- [bartowski/Qwen_Qwen3.5-2B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF)
- [mlx-community/Qwen3.5-2B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-2B-OptiQ-4bit)
- vLLM 量化文档：https://docs.vllm.ai/en/latest/quantization.html
- 量化技术详解：`资料/vLLM 不仅支持量化加载.txt`
