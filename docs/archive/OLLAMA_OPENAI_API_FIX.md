# Ollama OpenAI API 模型缺失问题修复

## 🔍 问题诊断

### 症状
- **Ollama 原生 API** (`ollama list`) 显示所有模型：
  - ✅ `qwen3.5:4b`
  - ✅ `deepseek-v3.1:671b-cloud`
  - ✅ 其他模型...

- **OpenAI 兼容 API** (`http://localhost:11434/v1/models`) 只显示 3 个模型：
  - ❌ `qwen2.5:0.5b`
  - ❌ `qwen2.5:3b`
  - ❌ `llama3:latest`

### 根本原因

**Ollama 服务使用了不同的模型存储路径！**

1. **原生 API** 使用 `OLLAMA_MODELS` 环境变量指定的路径：`D:\AI\models`
2. **OpenAI 兼容 API** 使用默认路径：`C:\Users\HiWin11\.ollama\models`

这导致两个 API 看到的是**不同的模型仓库**！

## 🛠️ 解决方案

### 方案 1：使用系统环境变量（推荐）

运行修复脚本：

```batch
cd d:\AI\project\zulong_beta4
.\restart_ollama_service.bat
```

这个脚本会：
1. 设置系统级环境变量 `OLLAMA_MODELS=D:\AI\models`
2. 停止现有 Ollama 进程
3. 使用新环境变量重启 Ollama 服务
4. 验证模型列表

### 方案 2：手动修复

#### 步骤 1：设置系统环境变量

**方法 A：使用命令行（需要管理员权限）**

```batch
setx OLLAMA_MODELS "D:\AI\models" /M
```

**方法 B：使用系统属性 GUI**

1. 右键"此电脑" → "属性"
2. "高级系统设置"
3. "环境变量"
4. 在"系统变量"部分点击"新建"
5. 变量名：`OLLAMA_MODELS`
6. 变量值：`D:\AI\models`
7. 确定保存

#### 步骤 2：重启 Ollama 服务

```powershell
# 停止 Ollama
Stop-Process -Name "ollama" -Force

# 等待 2 秒
Start-Sleep -Seconds 2

# 启动 Ollama 服务（会继承系统环境变量）
Start-Process "ollama" -ArgumentList "serve"
```

#### 步骤 3：验证

```powershell
# 检查原生 API
ollama list

# 检查 OpenAI API
Invoke-RestMethod -Uri 'http://localhost:11434/v1/models' -Method Get | 
    Select-Object -ExpandProperty data | 
    Select-Object id
```

### 方案 3：复制模型到默认路径（备选）

如果上述方法不起作用，可以将模型复制到默认路径：

```powershell
# 停止 Ollama
Stop-Process -Name "ollama" -Force

# 复制模型文件
Copy-Item "D:\AI\models\*" "$env:USERPROFILE\.ollama\models\" -Recurse -Force

# 重启 Ollama
Start-Process "ollama" -ArgumentList "serve"
```

## ✅ 验证步骤

### 1. 检查 OpenAI API

```powershell
$models = Invoke-RestMethod -Uri 'http://localhost:11434/v1/models' -Method Get
$models.data | Select-Object id
```

应该看到所有模型，包括：
- `qwen3.5:4b`
- `deepseek-v3.1:671b-cloud`
- 等等...

### 2. 测试模型调用

```powershell
$body = @{
    model = "qwen3.5:4b"
    messages = @(@{role = "user"; content = "你好" })
} | ConvertTo-Json

$result = Invoke-RestMethod -Uri "http://localhost:11434/v1/chat/completions" `
    -Method Post -Body $body -ContentType "application/json"

Write-Host "响应：$($result.choices.message.content)"
```

### 3. 重启祖龙系统

在 IDE 终端中设置环境变量并重启：

```powershell
# 设置环境变量
$env:OLLAMA_MODELS = "D:\AI\models"
$env:LLM_BACKEND = "ollama"
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL_ID = "qwen3.5:4b"
$env:LLM_MODEL_ID_BACKUP = "qwen3.5:4b"

# 启动祖龙系统
cd d:\AI\project\zulong_beta4
$env:PYTHONPATH = "d:\AI\project\zulong_beta4"
python zulong\bootstrap.py
```

## 📝 配置说明

### 推荐的环境变量设置

**系统环境变量（永久生效）**：
```
OLLAMA_MODELS=D:\AI\models
```

**IDE 终端临时设置（每次启动时）**：
```powershell
$env:LLM_BACKEND = "ollama"
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL_ID = "qwen3.5:4b"
$env:LLM_MODEL_ID_BACKUP = "qwen3.5:4b"
```

### 为什么需要设置系统环境变量？

Ollama 服务是一个独立进程，它：
1. **不继承** PowerShell 临时环境变量
2. **只读取** 系统环境变量或启动脚本中的设置
3. 需要**重启服务**才能应用新的环境变量

## 🎯 快速修复流程

**最简单的方法**：

```batch
# 1. 运行修复脚本（管理员权限）
cd d:\AI\project\zulong_beta4
.\restart_ollama_service.bat

# 2. 在 IDE 终端设置环境变量
$env:OLLAMA_MODELS = "D:\AI\models"
$env:LLM_BACKEND = "ollama"
$env:LLM_MODEL_ID = "qwen3.5:4b"

# 3. 重启祖龙系统
python zulong\bootstrap.py
```

## 🔧 故障排查

### 问题 1：设置系统环境变量后仍然看不到模型

**原因**：Ollama 服务没有重启或没有继承环境变量

**解决**：
```powershell
# 确认 Ollama 进程已停止
Get-Process ollama -ErrorAction SilentlyContinue

# 如果有进程，强制停止
Stop-Process -Name "ollama" -Force

# 等待 5 秒后重启
Start-Sleep -Seconds 5
Start-Process "ollama" -ArgumentList "serve"

# 验证
ollama list
```

### 问题 2：权限错误

**症状**：无法写入系统环境变量或访问模型目录

**解决**：
1. 以**管理员身份**运行 PowerShell 或命令提示符
2. 确保对 `D:\AI\models` 有完全控制权限

### 问题 3：模型文件损坏

**症状**：模型列表显示但调用失败

**解决**：
```powershell
# 重新下载模型
ollama pull qwen3.5:4b
ollama pull deepseek-v3.1:671b-cloud
```

## 📊 验证清单

- [ ] 系统环境变量 `OLLAMA_MODELS` 已设置
- [ ] Ollama 服务已重启
- [ ] `ollama list` 显示所有模型
- [ ] OpenAI API `/v1/models` 显示所有模型
- [ ] 能够成功调用模型（测试 `/v1/chat/completions`）
- [ ] 祖龙系统能够正常调用 Ollama 模型

## 🎉 成功标志

当所有步骤完成后，应该看到：

1. **OpenAI API 模型列表**包含 `qwen3.5:4b` 和 `deepseek-v3.1:671b-cloud`
2. **祖龙系统**不再返回"抱歉，我当前响应较慢"
3. **正常对话**能够流畅进行

## 📞 总结

**核心问题**：Ollama 的 OpenAI 兼容 API 和原生 API 使用不同的模型存储路径

**解决方法**：
1. 设置系统环境变量 `OLLAMA_MODELS=D:\AI\models`
2. 重启 Ollama 服务
3. 验证两个 API 都能看到所有模型
4. 重启祖龙系统并设置相同的环境变量

**关键**：确保 Ollama 服务、IDE 终端、祖龙系统都使用**相同的环境变量**！
