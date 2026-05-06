# vLLM GPTQ 量化实施计划

## 🎯 目标

使用 vLLM 的 **GPTQ 4bit 量化**技术加载 Qwen3.5-2B，配合 Qwen3.5-0.8B（全量），实现：
- ✅ 显存占用：~2.0-2.5 GB（减少 75%）
- ✅ 推理性能：吞吐量提升 40-50%
- ✅ 精度损失：控制在 2% 以内

## 📋 实施步骤

### 阶段 1：查找现成的 GPTQ 量化模型

#### 1.1 在 ModelScope 搜索

**搜索关键词**：
- `Qwen3.5-2B-GPTQ`
- `Qwen3.5-2B-AWQ`
- `Qwen3.5-2B-Int4`

**搜索链接**：
- https://modelscope.cn/models?q=Qwen3.5-2B-GPTQ
- https://modelscope.cn/models?q=Qwen3.5-2B-AWQ
- https://huggingface.co/models?search=Qwen3.5-2B-GPTQ

#### 1.2 检查模型文件

如果找到候选模型，检查是否包含以下文件：
- ✅ `config.json` - 模型配置文件
- ✅ `quantize_config.json` - GPTQ 量化配置
- ✅ `*.safetensors` 或 `*.bin` - 模型权重文件
- ✅ `tokenizer.json` - Tokenizer 文件

#### 1.3 下载模型

**如果找到合适的模型**：
```bash
# 使用 modelscope-cli 下载
modelscope download Qwen/Qwen3.5-2B-GPTQ --local_dir models/Qwen/Qwen3___5-2B-GPTQ
```

**或者使用 Git**：
```bash
cd models/Qwen
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3.5-2B-GPTQ.git Qwen3___5-2B-GPTQ
```

### 阶段 2：自己量化（如果阶段 1 失败）

#### 2.1 安装 AutoGPTQ

**在 WSL2 中执行**：
```bash
source ~/vllm-env/bin/activate
pip install auto-gptq optimum
```

#### 2.2 准备量化脚本

**创建脚本**：`scripts/quantize_qwen3_5_2b.py`

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
    "Python 和 Java 有什么区别？",
    "如何学习编程？",
    "什么是深度学习？",
    "推荐几本好书",
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

#### 2.3 运行量化

**在 WSL2 中执行**：
```bash
cd /mnt/d/AI/project/zulong_beta4
source ~/vllm-env/bin/activate
python scripts/quantize_qwen3_5_2b.py
```

**注意事项**：
- ⏱️ 量化时间：约 30-60 分钟
- 💾 需要 GPU 显存：至少 4GB
- 📊 建议使用 100+ 校准样本以获得更好效果

### 阶段 3：配置 vLLM 启动脚本

#### 3.1 使用 GPTQ 模型启动

**脚本**：`scripts/start_vllm_wsl2_2b_gptq.bat`

```batch
wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve models/Qwen/Qwen3___5-2B-GPTQ \
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
```

#### 3.2 使用在线 GPTQ 模型

如果 ModelScope 上有 GPTQ 版本，可以直接使用：

```batch
wsl bash -c "
source ~/vllm-env/bin/activate
export VLLM_USE_MODELSCOPE=true
vllm serve Qwen/Qwen3.5-2B-GPTQ \
  --port 8000 \
  --tensor-parallel-size 1 \
  --quantization gptq \
  --gptq-bits 4 \
  --gptq-group-size 128 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096
"
```

### 阶段 4：配置祖龙系统

#### 4.1 修改模型容器配置

**文件**：`zulong/models/container.py`

确保 `USE_VLLM_FOR_L2` 设置为 `true`：

```python
USE_VLLM_FOR_L2 = os.environ.get("USE_VLLM_FOR_L2", "true").lower() == "true"
```

#### 4.2 设置环境变量

**方法 1**：在启动脚本中设置
```batch
set USE_VLLM_FOR_L2=true
python -m zulong.main
```

**方法 2**：在系统环境变量中设置
```batch
setx USE_VLLM_FOR_L2 true
```

### 阶段 5：测试和验证

#### 5.1 启动 vLLM Server

```bash
cd d:\AI\project\zulong_beta4
scripts\start_vllm_wsl2_2b_gptq.bat
```

**预期输出**：
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### 5.2 测试 vLLM API

```bash
curl http://localhost:8000/v1/models
```

**预期响应**：
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3.5-2B-GPTQ",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

#### 5.3 测试推理

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-2B-GPTQ",
    "prompt": "你好，请介绍一下你自己",
    "max_tokens": 100
  }'
```

#### 5.4 启动祖龙系统

```bash
cd d:\AI\project\zulong_beta4
set USE_VLLM_FOR_L2=true
python -m zulong.main
```

**预期输出**：
```
[ModelContainer] [vLLM] L2_CORE 将使用 vLLM OpenAI API
[ModelContainer] [OK] L2_CORE vLLM 占位符注册成功
[ModelContainer] 初始化完成，当前显存使用：2.00/5.8GB
```

#### 5.5 检查显存占用

**在 WSL2 中执行**：
```bash
nvidia-smi
```

**预期结果**：
- vLLM 进程显存占用：~1.5-2.0 GB
- 总显存占用：~2.0-2.5 GB（包括 0.8B 模型）

## 🔧 故障排查

### 问题 1：找不到 GPTQ 模型

**解决方案**：
1. 在 ModelScope/HuggingFace 搜索
2. 如果找不到，使用 AutoGPTQ 自己量化
3. 或者尝试 AWQ 量化（效果类似）

### 问题 2：vLLM 不支持 GPTQ 参数

**解决方案**：
```bash
# 检查 vLLM 版本
pip show vllm

# 升级到最新版本（>= 0.3.0）
pip install --upgrade vllm
```

### 问题 3：量化模型加载失败

**检查清单**：
- [ ] 模型文件完整性
- [ ] `quantize_config.json` 存在
- [ ] vLLM 版本支持 GPTQ
- [ ] CUDA 版本兼容（>= 11.7）

### 问题 4：显存仍然不足

**解决方案**：
1. 降低 `--gpu-memory-utilization`（尝试 0.7）
2. 减少 `--max-model-len`（尝试 2048）
3. 启用 KV Cache 量化（`--kv-cache-dtype fp8_e4m3`）

## 📊 性能测试

### 测试脚本

创建 `scripts/test_vllm_performance.py`：

```python
import requests
import time

# vLLM API 端点
url = "http://localhost:8000/v1/completions"

# 测试提示
prompts = [
    "你好，请介绍一下你自己",
    "什么是人工智能？",
    "请解释一下机器学习的基本原理",
]

# 测试参数
test_config = {
    "model": "Qwen/Qwen3.5-2B-GPTQ",
    "max_tokens": 100,
    "temperature": 0.7,
}

# 性能测试
results = []
for prompt in prompts:
    payload = {**test_config, "prompt": prompt}
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    result = response.json()
    output = result["choices"][0]["text"]
    
    latency = (end_time - start_time) * 1000  # ms
    tokens = len(output.split())
    tokens_per_second = tokens / (end_time - start_time)
    
    results.append({
        "prompt": prompt,
        "output": output,
        "latency_ms": latency,
        "tokens": tokens,
        "tokens_per_second": tokens_per_second,
    })
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print(f"Latency: {latency:.2f} ms")
    print(f"Tokens/s: {tokens_per_second:.2f}")
    print("-" * 60)

# 平均性能
avg_latency = sum(r["latency_ms"] for r in results) / len(results)
avg_tokens_per_second = sum(r["tokens_per_second"] for r in results) / len(results)

print(f"\n平均延迟：{avg_latency:.2f} ms")
print(f"平均吞吐量：{avg_tokens_per_second:.2f} tokens/s")
```

### 性能对比

运行测试后，对比以下指标：

| 指标 | Transformer INT4 | vLLM GPTQ | 提升 |
|------|----------------|-----------|------|
| **延迟** | ~100ms | ~50ms | -50% |
| **吞吐量** | ~20 tokens/s | ~40-50 tokens/s | +100-150% |
| **显存占用** | ~2.5 GB | ~2.0 GB | -20% |

## ✅ 验收标准

### 功能验收

- [ ] vLLM Server 成功启动
- [ ] GPTQ 模型正常加载
- [ ] API 接口可访问
- [ ] 推理功能正常

### 性能验收

- [ ] 显存占用 ≤ 2.5 GB
- [ ] 推理延迟 ≤ 100ms
- [ ] 吞吐量 ≥ 30 tokens/s
- [ ] 精度损失 ≤ 2%

### 稳定性验收

- [ ] 连续运行 1 小时无崩溃
- [ ] 并发请求处理正常
- [ ] 显存无泄漏

## 📝 总结

### 关键成功因素

1. ✅ **找到或创建 GPTQ 模型**：阶段 1 或阶段 2
2. ✅ **正确配置 vLLM 参数**：阶段 3
3. ✅ **祖龙系统集成**：阶段 4
4. ✅ **性能和稳定性测试**：阶段 5

### 预期收益

- 💾 **显存占用**：减少 75%（6.0 GB → 2.0-2.5 GB）
- ⚡ **推理性能**：提升 40-50%
- 🎯 **精度保持**：损失 < 2%
- 🚀 **并发能力**：显著提升

### 下一步计划

1. ✅ 执行阶段 1：查找 GPTQ 模型
2. ⚠️ 如果失败，执行阶段 2：自己量化
3. ✅ 执行阶段 3-5：配置、集成、测试
4. 📊 根据测试结果优化参数
5. 🎉 正式部署使用

## 🔗 参考资料

- 详细配置指南：`docs/VLLM_GPTQ_QWEN2B_GUIDE.md`
- 量化技术文档：`资料/vLLM 不仅支持量化加载.txt`
- vLLM 官方文档：https://docs.vllm.ai
- AutoGPTQ GitHub：https://github.com/AutoGPTQ/AutoGPTQ
