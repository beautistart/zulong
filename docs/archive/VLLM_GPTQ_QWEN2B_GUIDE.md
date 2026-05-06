# vLLM GPTQ 量化加载 Qwen3.5-2B 配置指南

## 🎯 目标

使用 vLLM 的 **GPTQ 量化**加载 Qwen3.5-2B（4bit）+ Qwen3.5-0.8B（全量），实现低显存占用和高性能推理。

## 📊 预期效果

根据文档数据：
- **显存占用**：减少 75% 以上（2B 模型从 4.25GB → ~1.5GB）
- **推理性能**：吞吐量提升 40-50%
- **精度损失**：控制在 2% 以内
- **总显存需求**：2B(GPTQ) + 0.8B(FP16) ≈ **2.0-2.5 GB**

## 📋 方案选择

### 方案 A：使用 ModelScope 上的 GPTQ 量化版本（推荐）

查找现成的 GPTQ 量化模型：
```bash
# 在 ModelScope 搜索
https://modelscope.cn/models?q=Qwen3.5-2B-GPTQ
```

### 方案 B：自己量化（如果找不到现成的）

使用 AutoGPTQ 工具将原始模型转换为 GPTQ 格式。

## 🚀 实施步骤

### 步骤 1：查找 GPTQ 量化模型

#### 在 ModelScope 搜索

访问以下链接搜索 GPTQ 版本：
- https://modelscope.cn/models?q=Qwen3.5-2B-GPTQ
- https://modelscope.cn/models?q=Qwen3.5-2B-AWQ

#### 可能的模型仓库

如果找到以下任一模型，即可使用：
- `Qwen/Qwen3.5-2B-GPTQ`
- `Qwen/Qwen3.5-2B-AWQ`
- `modelscope/Qwen3.5-2B-GPTQ-Int4`

### 步骤 2：配置 vLLM 启动脚本

#### 创建 GPTQ 专用启动脚本

创建 `scripts/start_vllm_wsl2_2b_gptq.bat`：

```batch
@echo off
REM ========================================
REM WSL2 vLLM Server 启动脚本 (Qwen3.5-2B GPTQ 量化版)
REM ========================================

echo ================================================================================
echo                   WSL2 vLLM Server 启动脚本 (GPTQ 量化)
echo ================================================================================
echo.
echo 模型配置:
echo   - L2_CORE: Qwen3.5-2B (GPTQ 4bit 量化)
echo   - L2_BACKUP: Qwen3.5-0.8B (FP16 全量)
echo   - 预计显存占用：~2.0-2.5 GB
echo.
echo 按 Ctrl+C 停止服务
echo ================================================================================
echo.

REM 检查 WSL 是否可用
wsl --status >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL 未安装或不可用
    pause
    exit /b 1
)

echo [OK] WSL 已就绪
echo.

REM 启动 vLLM Server (GPTQ 量化)
echo [START] 启动 vLLM Server (Qwen3.5-2B GPTQ)...
echo.

wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve Qwen/Qwen3.5-2B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --trust-remote-code \
  --dtype auto
"

pause
```

### 步骤 3：如果找不到 GPTQ 版本，自己量化

#### 安装 AutoGPTQ

```bash
# 在 WSL2 中执行
source ~/vllm-env/bin/activate
pip install auto-gptq optimum
```

#### 量化脚本

创建 `scripts/quantize_qwen3_5_2b.py`：

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import torch

# 模型路径
model_path = "/mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B"
quantized_path = "/mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-GPTQ"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,              # 4bit 量化
    group_size=128,      # 分组大小
    damp_percent=0.01,   # 阻尼系数
    desc_act=False,      # 是否启用描述性激活
    max_length=4096,     # 最大序列长度
)

# 加载模型（使用 FP16）
model = AutoGPTQForCausalLM.from_pretrained(
    model_path,
    quantize_config=quantize_config,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 准备校准数据（使用少量样本即可）
calibration_data = [
    "你好，请介绍一下你自己",
    "什么是人工智能？",
    "今天天气怎么样？",
    "请解释一下机器学习的基本原理",
]

# 量化模型
print("开始量化模型...")
model.quantize(calibration_data, batch_size=1)

# 保存量化后的模型
print(f"保存量化模型到：{quantized_path}")
model.save_quantized(quantized_path)
tokenizer.save_pretrained(quantized_path)

print("量化完成！")
```

#### 运行量化

```bash
cd /mnt/d/AI/project/zulong_beta4
source ~/vllm-env/bin/activate
python scripts/quantize_qwen3_5_2b.py
```

**注意**：量化过程需要约 30-60 分钟，需要足够的 GPU 显存（至少 4GB）。

### 步骤 4：使用量化后的模型

#### 修改 vLLM 启动命令

```bash
vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-2B-GPTQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --trust-remote-code
```

## 🔧 高级配置

### 1. KV Cache 量化（进一步优化显存）

```bash
vllm serve Qwen/Qwen3.5-2B \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --kv-cache-dtype fp8_e4m3 \  # FP8 KV Cache
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

**效果**：
- KV Cache 显存占用再减少 50%
- 总显存需求降至 ~1.8-2.0 GB
- 适合更长上下文（8192+ tokens）

### 2. 混合精度策略

```bash
vllm serve Qwen/Qwen3.5-2B \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --dtype float16 \  # 保持部分层为 FP16
  --gpu-memory-utilization 0.85
```

### 3. 性能优化参数

```bash
vllm serve Qwen/Qwen3.5-2B \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --gpu-memory-utilization 0.9 \  # 提高 GPU 利用率
  --max-num-batched-tokens 8192 \  # 增加批处理 tokens
  --max-num-seqs 32 \  # 增加并发序列数
  --enable-prefix-caching  # 启用前缀缓存
```

## 📊 性能预期

### 显存占用对比

| 配置 | 2B 模型 | 0.8B 模型 | 总计 | 是否可行 |
|------|--------|---------|------|---------|
| **FP16 全量** | 4.25 GB | 1.72 GB | ~6.0 GB | ❌ 失败 |
| **GPTQ 4bit** | ~1.5 GB | 1.72 GB | ~3.2 GB | ⚠️ 可能可行 |
| **GPTQ + FP8 KV** | ~1.2 GB | 1.72 GB | ~2.9 GB | ✅ 可行 |
| **GPTQ + 0.8B INT4** | ~1.5 GB | ~0.5 GB | ~2.0 GB | ✅ 最佳 |

### 推理性能对比

| 指标 | Transformer INT4 | vLLM GPTQ | 提升 |
|------|----------------|-----------|------|
| **吞吐量** | ~20 tokens/s | ~40-50 tokens/s | **+100-150%** |
| **延迟** | ~100ms | ~50ms | **-50%** |
| **并发能力** | 低 | 高 | **显著提升** |

## ✅ 推荐配置组合

### 最佳配置（推荐）

**2B 模型（GPTQ 4bit） + 0.8B 模型（INT4 量化）**

```bash
# vLLM 启动 2B GPTQ
vllm serve Qwen/Qwen3.5-2B-GPTQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096

# Transformer 加载 0.8B INT4（在祖龙系统中）
# config.py 中配置：
ModelID.L2_BACKUP: ModelConfig(
    use_int4=True  # INT4 量化
)
```

**总显存占用**：~2.0 GB
**推理性能**：~40-50 tokens/s
**精度损失**：< 2%

## 🔍 故障排查

### 问题 1：找不到 GPTQ 模型

**解决方案**：
1. 在 ModelScope/HuggingFace 搜索 GPTQ 版本
2. 如果找不到，使用 AutoGPTQ 自己量化
3. 或者使用 AWQ 量化（效果类似）

### 问题 2：量化后精度下降明显

**解决方案**：
1. 尝试 INT8 量化（`--gptq-bits 8`）
2. 增加校准数据量（使用 100+ 样本）
3. 调整 `group_size`（尝试 64 或 256）

### 问题 3：vLLM 版本不支持 GPTQ

**解决方案**：
```bash
# 升级 vLLM 到最新版本
pip install --upgrade vllm

# 确保版本 >= 0.3.0
pip show vllm
```

### 问题 4：量化模型加载失败

**检查清单**：
- [ ] 模型文件完整性（检查 `config.json` 和 `*.safetensors`）
- [ ] GPTQ 配置文件存在（`quantize_config.json`）
- [ ] vLLM 版本支持 GPTQ（>= 0.3.0）
- [ ] CUDA 版本兼容（>= 11.7）

## 📝 总结

### 关键要点

1. **vLLM 确实支持量化**：AWQ、GPTQ、FP8 等多种方案
2. **GPTQ 是最佳选择**：显存占用低（25-30%），精度损失小（<2%）
3. **需要量化版本模型**：ModelScope 搜索或自己量化
4. **性能提升显著**：吞吐量提升 40-50%，延迟降低 50%

### 推荐实施路径

**阶段 1**：查找现成的 GPTQ 模型
- 在 ModelScope 搜索 `Qwen3.5-2B-GPTQ`
- 如果找到，直接下载并使用

**阶段 2**：自己量化（如果阶段 1 失败）
- 使用 AutoGPTQ 工具
- 量化时间：30-60 分钟
- 需要 GPU 显存：至少 4GB

**阶段 3**：优化配置
- 尝试 KV Cache 量化
- 调整并发参数
- 测试性能和精度

### 下一步行动

1. ✅ 在 ModelScope 搜索 GPTQ 模型
2. ⚠️ 如果找到，下载并测试
3. ⚠️ 如果找不到，使用 AutoGPTQ 量化
4. ✅ 配置 vLLM 启动脚本
5. ✅ 测试性能和显存占用
6. ✅ 根据结果优化参数

## 🔗 参考资料

- vLLM 量化文档：https://docs.vllm.ai/en/latest/quantization.html
- AutoGPTQ GitHub：https://github.com/AutoGPTQ/AutoGPTQ
- ModelScope Qwen 系列：https://modelscope.cn/models?q=Qwen
- 量化技术详解：`资料/vLLM 不仅支持量化加载.txt`
