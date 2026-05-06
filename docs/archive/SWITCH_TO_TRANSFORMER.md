# 快速切换回 Transformer 原生加载指南

## 🎯 目标

从 vLLM 切换回 **Transformer 原生加载（INT4 量化）**，实现 2B + 0.8B 同时稳定运行。

## 📋 当前状态检查

### 1. 检查配置文件

打开 `zulong/models/config.py`，确认以下配置：

```python
# L2: Qwen3.5-2B (本地模型，GPU 运行，INT4 量化) -> ~2.5 GB
ModelID.L2_CORE: ModelConfig(
    model_id=ModelID.L2_CORE,
    repo_id="models/Qwen/Qwen3___5-2B",
    estimated_vram_gb=2.5,  # ✅ INT4 量化显存占用
    is_expert=False,
    device="cuda",
    enabled=True,
    use_int4=True  # ✅ 使用 INT4 量化加载
)

# L2_BACKUP: Qwen3.5-0.8B (本地模型，GPU 运行，INT4 量化) -> ~0.5 GB
ModelID.L2_BACKUP: ModelConfig(
    model_id=ModelID.L2_BACKUP,
    repo_id="models/unsloth/Qwen3.5-0.8B",
    estimated_vram_gb=0.5,  # ✅ INT4 量化显存占用
    is_expert=False,
    device="cuda",
    enabled=True,
    use_int4=True  # ✅ 使用 INT4 量化加载
)
```

### 2. 检查 vLLM 开关

打开 `zulong/models/container.py`，确认：

```python
# 🔥 vLLM 配置：是否使用 vLLM 代替本地模型加载
# 如果设置为 True，L2_CORE 将不会本地加载，而是通过 OpenAI API 调用 vLLM
USE_VLLM_FOR_L2 = os.environ.get("USE_VLLM_FOR_L2", "false").lower() == "true"
```

**默认值是 `"false"`**，所以只要不设置环境变量就会使用 Transformer 原生加载。

## 🚀 切换步骤

### 方法 1：设置环境变量（推荐）

**Windows PowerShell**：
```powershell
# 设置环境变量为 false
$env:USE_VLLM_FOR_L2 = "false"

# 启动祖龙系统
python -m zulong.main
```

**Windows CMD**：
```batch
set USE_VLLM_FOR_L2=false
python -m zulong.main
```

**WSL2 Bash**：
```bash
export USE_VLLM_FOR_L2=false
python -m zulong.main
```

### 方法 2：修改代码（永久切换）

打开 `zulong/models/container.py`，修改第 14 行：

```python
# 修改前
USE_VLLM_FOR_L2 = os.environ.get("USE_VLLM_FOR_L2", "false").lower() == "true"

# 修改后（强制使用 Transformer）
USE_VLLM_FOR_L2 = False
```

### 方法 3：使用启动脚本

创建 `scripts/start_zulong_transformer.bat`：

```batch
@echo off
echo ================================================================================
echo                    祖龙系统启动脚本 (Transformer 原生加载)
echo ================================================================================
echo.
echo 此脚本将使用 Transformer 原生加载方式启动祖龙系统
echo 模型配置:
echo   - L2_CORE: Qwen3.5-2B (INT4 量化，GPU)
echo   - L2_BACKUP: Qwen3.5-0.8B (INT4 量化，GPU)
echo   - 总显存占用：~2.5-3.0 GB
echo.
echo 按 Ctrl+C 停止服务
echo ================================================================================
echo.

REM 设置环境变量
set USE_VLLM_FOR_L2=false

REM 启动祖龙系统
cd /d "%~dp0.."
python -m zulong.main

pause
```

## ✅ 验证加载成功

### 1. 检查启动日志

启动后应该看到类似输出：

```
[ModelContainer] 初始化模型容器...
[ModelContainer] 加载常驻模型...
[ModelContainer] 加载常驻模型：L2_CORE
[ModelContainer] [INFO] 使用 Transformer 原生加载（INT4 量化）
[ModelContainer] 目标设备：CUDA
[ModelContainer] 使用 INT4 量化：True
[ModelContainer] [OK] 加载完成：L2_CORE
[ModelContainer] 加载常驻模型：L2_BACKUP
[ModelContainer] [OK] 加载完成：L2_BACKUP
[ModelContainer] 初始化完成，当前显存使用：2.50/5.8GB
```

### 2. 检查显存占用

在 Python 中运行：

```python
import torch

# 检查显存占用
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"已分配显存：{allocated:.2f} GB")
print(f"预留显存：{reserved:.2f} GB")
print(f"可用显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3 - allocated:.2f} GB")
```

**预期结果**：
- 已分配显存：~2.0-2.5 GB
- 预留显存：~2.5-3.0 GB
- 可用显存：~3.0-3.5 GB

### 3. 测试模型推理

```python
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

# 获取模型容器
container = ModelContainer()

# 测试 L2_CORE 模型
l2_model = container.resident_models[ModelID.L2_CORE]

# 测试推理
prompt = "你好，请介绍一下你自己"
response = l2_model.generate(prompt, max_new_tokens=100)
print(response)
```

## 📊 性能对比

| 指标 | Transformer INT4 | vLLM FP16 |
|------|----------------|-----------|
| **显存占用** | ~2.5 GB | ~6.0+ GB |
| **启动时间** | ~30 秒 | ~5-10 分钟 |
| **稳定性** | ✅ 稳定 | ❌ 崩溃 |
| **推理速度** | ~20 tokens/s | ~50 tokens/s |
| **并发能力** | 低 | 高 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐ |

## 🔧 故障排查

### 问题 1：显存不足

**错误信息**：
```
RuntimeError: CUDA out of memory
```

**解决方案**：
1. 关闭其他占用显存的程序（游戏、浏览器硬件加速等）
2. 重启电脑释放显存
3. 检查是否有其他 AI 程序在运行

### 问题 2：模型路径不存在

**错误信息**：
```
[ModelContainer] [ERROR] 本地模型不存在：models/Qwen/Qwen3___5-2B
```

**解决方案**：
```bash
# 检查模型目录
ls models/Qwen/

# 如果不存在，需要下载模型
# 从 ModelScope 下载：https://modelscope.cn/models/Qwen/Qwen3.5-2B
```

### 问题 3：INT4 量化失败

**错误信息**：
```
ImportError: bitsandbytes is not installed
```

**解决方案**：
```bash
pip install bitsandbytes
```

### 问题 4：vLLM 仍在运行

**错误信息**：
```
Connection refused: http://localhost:8000/v1/models
```

**解决方案**：
1. 停止 vLLM 服务（如果在 WSL2 中运行）
2. 设置 `USE_VLLM_FOR_L2=false`

## 🎯 总结

### 为什么选择 Transformer 原生加载？

1. ✅ **显存占用低**：2B + 0.8B 仅需 ~2.5 GB
2. ✅ **启动快速**：30 秒 vs 5-10 分钟
3. ✅ **稳定可靠**：无崩溃风险
4. ✅ **配置简单**：无需复杂参数
5. ✅ **兼容性好**：支持任意 HuggingFace 模型

### 什么时候使用 vLLM？

- 当你有 12GB+ GPU 时
- 当你需要高并发推理时
- 当你找到 AWQ/GPTQ 量化版本时
- 当你需要生产级性能时

### 下一步建议

1. ✅ 使用 Transformer 原生加载稳定运行
2. ⚠️ 如果未来升级 GPU，可以考虑 vLLM
3. ⚠️ 如果需要更高性能，可以查找 AWQ 量化版本
4. ⚠️ 如果需要更大模型，可以使用在线 API

## 🔗 相关文档

- [TRANSFORMER_VS_VLLM.md](TRANSFORMER_VS_VLLM.md) - 详细对比分析
- [VLLM_CONFIG_SUMMARY.md](VLLM_CONFIG_SUMMARY.md) - vLLM 配置总结
- [VLLM_QWEN2B_CONFIG.md](VLLM_QWEN2B_CONFIG.md) - 2B 模型配置指南
