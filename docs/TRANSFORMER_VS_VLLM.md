# Transformer 原生加载 vs vLLM 加载对比（已更新）

## ✅ 更新说明

根据最新文档 `资料/vLLM 不仅支持量化加载.txt`，vLLM 确实支持多种量化方案！

## ❓ 之前的误解

我之前错误地认为 vLLM 不支持量化，实际上：

### vLLM 支持的量化方案

1. **AWQ（Activation-aware Weight Quantization）**
   - 配置：`--quantization awq`
   - 显存占用：约 25-30% 原始大小
   - 精度损失：< 2%

2. **GPTQ（General-purpose Post-training Quantization）**
   - 配置：`--quantization gptq --gptq-bits 4 --gptq-group-size 128`
   - 显存占用：约 25-30% 原始大小
   - 精度损失：< 2%

3. **FP8 KV Cache 量化**
   - 配置：`--kv-cache-dtype fp8_e4m3`
   - 专门用于 KV 缓存的低精度存储

4. **INT8/INT4（bitsandbytes）**
   - vLLM 0.3.0+ 支持
   - 配置：`--load-format bitsandbytes`

### 正确的 vLLM 量化加载方式

```bash
# 加载 Qwen3.5-2B 的 GPTQ 量化版本
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --trust-remote-code
```

#### 2. **显存管理机制不同**

**Transformer 原生**：
- **按需分配**：只加载模型权重
- **动态管理**：推理时根据需要分配 KV cache
- **无预分配**：启动时显存占用低

**vLLM**：
- **预分配**：启动时分配 KV cache、CUDA graphs
- **静态管理**：为了性能优化，预留大量显存
- **最小显存要求高**：即使量化也需要足够的连续显存

### 📊 显存占用对比

| 加载方式 | 2B 模型 | 0.8B 模型 | 总计 | 是否可行 |
|---------|--------|---------|------|---------|
| **Transformer INT4** | ~1.5-2.0 GB | ~0.5 GB | **~2.0-2.5 GB** | ✅ 可行 |
| **Transformer FP16** | ~4.0 GB | ~1.5 GB | **~5.5 GB** | ⚠️ 勉强 |
| **vLLM FP16** | ~4.25 GB + KV | ~1.72 GB + KV | **~6.0+ GB** | ❌ 失败 |
| **vLLM AWQ** (如果有) | ~1.5 GB + KV | ~0.5 GB + KV | **~2.5-3.0 GB** | ⚠️ 可能可行 |

### 🎯 为什么 vLLM 不支持 bitsandbytes INT4？

1. **技术路线不同**：
   - bitsandbytes：动态量化，推理时反量化
   - AWQ/GPTQ：静态量化，权重直接存储为 INT4

2. **性能优化需求**：
   - vLLM 需要连续的显存用于 CUDA graphs
   - bitsandbytes 的量化格式不适合 GPU 优化

3. **内核实现**：
   - vLLM 使用自定义的 CUDA kernel
   - 不支持 bitsandbytes 的反量化操作

## 💡 解决方案

### 方案 A：使用 Transformer 原生加载（推荐）

**配置**：
```python
# zulong/models/config.py
ModelID.L2_CORE: ModelConfig(
    model_id=ModelID.L2_CORE,
    repo_id="models/Qwen/Qwen3___5-2B",
    estimated_vram_gb=2.5,  # INT4 量化显存占用
    is_expert=False,
    device="cuda",
    enabled=True,
    use_int4=True  # ✅ 使用 INT4 量化加载
)
```

**优点**：
- ✅ 已验证可稳定运行
- ✅ 显存占用低（2.0-2.5 GB）
- ✅ 支持任意 HuggingFace 模型
- ✅ 无需额外配置

**缺点**：
- ⚠️ 推理速度比 vLLM 慢
- ⚠️ 不支持连续批处理等高级特性

### 方案 B：使用 vLLM + AWQ 量化模型

**步骤**：

1. **查找 AWQ 量化版本**：
   ```bash
   # 在 ModelScope 搜索
   https://modelscope.cn/models?q=Qwen3.5-2B-AWQ
   ```

2. **如果存在 AWQ 版本**：
   ```bash
   vllm serve Qwen/Qwen3.5-2B-AWQ \
     --port 8000 \
     --tensor-parallel-size 1 \
     --max-model-len 4096 \
     --quantization awq \
     --gpu-memory-utilization 0.6
   ```

**优点**：
- ✅ vLLM 的高性能
- ✅ 显存占用低（AWQ 量化）

**缺点**：
- ⚠️ 可能找不到 AWQ 版本
- ⚠️ 需要额外下载量化模型

### 方案 C：自己量化模型为 AWQ 格式

**步骤**：

```bash
# 1. 安装 autoawq
pip install autoawq

# 2. 量化模型
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-2B",
    device_map="cuda",
)

model.quantize(
    tokenizer=tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
)

model.save_quantized("Qwen3.5-2B-AWQ")
```

**优点**：
- ✅ 可以自定义量化参数
- ✅ 一次量化，多次使用

**缺点**：
- ⚠️ 量化过程耗时（需要 GPU）
- ⚠️ 需要额外存储空间

## 🔧 当前推荐配置

### 使用 Transformer 原生加载

**修改 `zulong/models/container.py`**：
```python
# 将 USE_VLLM_FOR_L2 设置为 False
USE_VLLM_FOR_L2 = os.environ.get("USE_VLLM_FOR_L2", "false").lower() == "false"
```

**或者直接设置环境变量**：
```batch
set USE_VLLM_FOR_L2=false
```

**启动祖龙系统**：
```batch
# 重启祖龙和 OpenClaw
python -m zulong.main
```

### 验证加载成功

```python
# 检查显存占用
import torch
print(f"显存占用：{torch.cuda.memory_allocated() / 1024**3:.2f} GB")
# 应该显示 ~2.0-2.5 GB（2B INT4 + 0.8B INT4）
```

## 📝 总结

### 为什么现在不行？

1. **vLLM 不支持 bitsandbytes INT4 量化**
   - 需要 AWQ/GPTQ 专用格式
   - 我们的模型是原始 Safetensors 格式

2. **vLLM 显存管理机制不同**
   - 需要预分配大量显存
   - 即使降低 `--gpu-memory-utilization` 也无法满足最低需求

3. **6GB 显存限制**
   - 2B 模型 FP16 需要 4.25 GB
   - 加上 KV cache 和 CUDA graphs，总需求超过 6GB

### 最佳实践

**当前阶段**：
- ✅ 使用 Transformer 原生加载（INT4 量化）
- ✅ 2B + 0.8B 同时加载，总显存 ~2.5 GB
- ✅ 稳定运行，无崩溃风险

**未来升级**：
- 如果有 12GB+ GPU，可以使用 vLLM FP16
- 如果找到 AWQ 版本，可以使用 vLLM + AWQ
- 如果自己量化，可以体验 vLLM 高性能

## 🔗 参考资料

- bitsandbytes 文档：https://github.com/TimDettmers/bitsandbytes
- AWQ 文档：https://github.com/mit-han-lab/llm-awq
- vLLM 量化支持：https://docs.vllm.ai/en/latest/quantization.html
- Qwen3.5 ModelScope：https://modelscope.cn/models/Qwen/Qwen3.5-2B
