# WSL2 + vLLM 完整实施指南

## 📋 概述

本指南详细说明如何在 WSL2 (Windows Subsystem for Linux) 中部署 vLLM Server，实现完整的 **Function Calling** 功能。

## 🎯 为什么使用 WSL2？

- ✅ **完整支持**：WSL2 提供完整的 Linux 环境，vLLM 所有功能都可用
- ✅ **性能优异**：直接访问 Windows 的 GPU，性能接近原生 Linux
- ✅ **兼容性好**：避免 Windows 上的各种兼容性问题
- ✅ **开发友好**：同时享受 Windows 和 Linux 的优势

## 🚀 快速开始

### 前置条件检查

1. **Windows 版本**：Windows 10 21H2+ 或 Windows 11
2. **NVIDIA GPU**：支持 CUDA 的 NVIDIA 显卡
3. **NVIDIA 驱动**：Windows 主机已安装最新驱动
4. **WSL2**：已安装并配置

**检查 WSL2 状态：**
```bash
wsl --status
```

**预期输出：**
```
默认分发：Ubuntu-22.04
默认版本：2
```

### 步骤 1: 安装 vLLM (在 WSL2 中)

**方式 A: 使用自动化脚本（推荐）**

在 Windows PowerShell 中运行：
```powershell
cd d:\AI\project\zulong_beta4
wsl bash d:\AI\project\zulong_beta4\scripts\wsl2_install_vllm.sh
```

**方式 B: 手动安装**

1. 进入 WSL2 终端：
```bash
wsl
```

2. 创建虚拟环境：
```bash
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
```

3. 安装 PyTorch：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. 安装 vLLM：
```bash
pip install vllm==0.6.3
```

### 步骤 2: 启动 vLLM Server

**方式 A: 使用 Windows 启动脚本**

双击运行：
```
d:\AI\project\zulong_beta4\scripts\start_vllm_wsl2.bat
```

**方式 B: 手动启动**

1. 在 WSL2 中：
```bash
wsl
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

**启动成功标志：**
```
INFO:     Uvicorn running on http://0.0.0.0:8000/v1
INFO:     Application startup complete.
```

### 步骤 3: 测试连接

**在 Windows 上测试：**
```powershell
curl http://localhost:8000/v1/models
```

**预期输出：**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3.5-0.8B",
      "object": "model",
      ...
    }
  ]
}
```

### 步骤 4: 配置 Zulong 使用 vLLM

**设置环境变量：**
```powershell
$env:USE_VLLM_FOR_L2="true"
$env:VLLM_BASE_URL="http://localhost:8000/v1"
python zulong/bootstrap.py
```

**或修改代码：**

编辑 `zulong/models/container.py`：
```python
USE_VLLM_FOR_L2 = True
VLLM_BASE_URL = "http://localhost:8000/v1"
```

### 步骤 5: 测试工具调用

通过 OpenClaw Web 界面（http://localhost:8080）发送消息：

```
搜索关于 AI MAX395 的信息
```

**预期日志：**
```
🌐 [网络搜索] 使用 vLLM OpenAI API 调用工具...
🌐 [vLLM 搜索] 调用 OpenAI API...
🌐 [vLLM 搜索] 检测到工具调用：1 个
🌐 [vLLM 搜索] 执行搜索：query=AI MAX395, count=3
🌐 [vLLM 搜索] 搜索完成，找到 3 个结果
```

## 🔧 网络配置

### WSL2 网络原理

WSL2 使用虚拟化网络，但会自动将 `localhost` 端口转发到 Windows：

```
WSL2 (Ubuntu)          Windows Host
┌─────────────┐       ┌─────────────┐
│ vLLM :8000  │ ────► │ localhost   │
│             │       │   :8000     │
└─────────────┘       └─────────────┘
```

### 常见问题解决

#### 问题 1: Windows 无法访问 WSL2 端口

**症状：**
```
curl: (7) Failed to connect to localhost port 8000
```

**解决方案：**

1. 检查 WSL2 版本：
```bash
wsl --list --verbose
```

确保 VERSION 为 `2`

2. 重启 WSL2：
```bash
wsl --shutdown
wsl
```

3. 使用 WSL2 IP 直接访问：
```powershell
wsl hostname -I
# 输出：172.x.x.x
curl http://172.x.x.x:8000/v1/models
```

#### 问题 2: GPU 不可用

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**

1. 在 WSL2 中检查 GPU：
```bash
nvidia-smi
```

2. 更新 Windows 主机 NVIDIA 驱动

3. 降低显存使用：
```bash
vllm serve ... --gpu-memory-utilization 0.6
```

#### 问题 3: 端口被占用

**症状：**
```
Address already in use
```

**解决方案：**

1. 查找占用端口的进程：
```powershell
netstat -ano | findstr :8000
```

2. 终止进程或更换端口：
```bash
vllm serve ... --port 8001
```

## 📊 性能优化

### 1. 显存优化

```bash
vllm serve ... \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192
```

### 2. 推理加速

```bash
vllm serve ... \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096
```

### 3. 并发优化

```bash
vllm serve ... \
  --max-num-seqs 16 \
  --max-running-requests 10
```

## 🐛 调试技巧

### 检查 vLLM 状态

```bash
# 在 WSL2 中
curl http://localhost:8000/v1/models

# 在 Windows 上
curl http://localhost:8000/v1/models
```

### 查看 GPU 状态

```bash
# 在 WSL2 中
nvidia-smi
```

### 测试工具调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# 测试工具调用
tools = [
    {
        "type": "function",
        "function": {
            "name": "openclaw_search",
            "description": "网络搜索工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "搜索今天的新闻"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message)
```

### 查看日志

**vLLM 日志：**
```bash
# 启动时自动输出
```

**Zulong 日志：**
```python
# 在 inference_engine.py 中
logger.setLevel(logging.DEBUG)
```

## 📝 配置文件

### 环境变量配置

在 PowerShell 中：
```powershell
$env:USE_VLLM_FOR_L2="true"
$env:VLLM_BASE_URL="http://localhost:8000/v1"
```

### 永久配置

编辑 `zulong/models/container.py`：
```python
USE_VLLM_FOR_L2 = True
VLLM_BASE_URL = "http://localhost:8000/v1"
```

## 🎓 最佳实践

### 1. 开发环境
- 使用较小的 `max-model-len` (8192) 节省显存
- 保留本地模型作为后备
- 定期重启 WSL2 清理内存

### 2. 生产环境
- 使用完整的 `max-model-len` (262144)
- 启用所有优化工具
- 监控 GPU 温度和显存

### 3. 日常维护
- 每周更新 vLLM：`pip install --upgrade vllm`
- 定期清理 WSL2 磁盘空间
- 监控 NVIDIA 驱动更新

## 📚 参考资料

- [WSL2 官方文档](https://docs.microsoft.com/en-us/windows/wsl/)
- [vLLM 官方文档](https://docs.vllm.ai/)
- [NVIDIA CUDA on WSL](https://developer.nvidia.com/cuda/wsl)
- [Qwen3.5 ModelScope](https://modelscope.cn/models/Qwen/Qwen3.5-0.8B)

## 🆘 故障排除

### 快速诊断脚本

创建 `diagnose_wsl2_vllm.ps1`：
```powershell
Write-Host "=== WSL2 vLLM 诊断脚本 ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. 检查 WSL 状态..." -ForegroundColor Yellow
wsl --status

Write-Host ""
Write-Host "2. 检查 WSL2 IP..." -ForegroundColor Yellow
wsl hostname -I

Write-Host ""
Write-Host "3. 检查端口 8000..." -ForegroundColor Yellow
netstat -ano | findstr :8000

Write-Host ""
Write-Host "4. 测试 vLLM 连接..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/v1/models" -TimeoutSec 5
    Write-Host "✅ vLLM 连接成功！" -ForegroundColor Green
    $response.Content
} catch {
    Write-Host "❌ vLLM 连接失败" -ForegroundColor Red
    $_.Exception.Message
}

Write-Host ""
Write-Host "5. 检查 GPU 状态..." -ForegroundColor Yellow
wsl nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

Write-Host ""
Write-Host "诊断完成！" -ForegroundColor Cyan
```

运行：
```powershell
.\diagnose_wsl2_vllm.ps1
```

---

**最后更新**: 2026-04-09
**版本**: v1.0
