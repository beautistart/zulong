# 祖龙系统 LLM 配置修复指南

## 🔍 问题诊断

### 当前状态
- **Ollama 服务**: 运行在 Windows 终端
- **祖龙系统**: 运行在 IDE 终端
- **模型存储**: `D:\AI\models`
- **API 地址**: `http://localhost:11434`

### 问题根源
**终端不同不影响调用！** 真正的问题是：

1. **环境变量未正确设置**
   - IDE 终端启动时没有设置 `LLM_BACKEND`, `LLM_BASE_URL`, `LLM_MODEL_ID`
   - 导致使用默认配置（vLLM @ localhost:8000）

2. **模型配置不匹配**
   - 配置要求：`deepseek-v3.1:671b-cloud` (云端) 和 `qwen3.5:4b` (本地)
   - 实际 Ollama API 返回：`qwen2.5:0.5b`, `qwen2.5:3b`, `llama3:latest`

3. **权限问题**
   - 下载模型时出现 "Access is denied"
   - Ollama 模型目录权限可能不正确

## 🛠️ 解决方案

### 方案 1：使用启动脚本（推荐）

使用 `start_ollama_fixed.bat`，它会自动设置环境变量：

```batch
start_ollama_fixed.bat
```

这个脚本会：
- 设置正确的环境变量
- 使用已有的模型 `qwen2.5:3b` 和 `qwen2.5:0.5b`
- 自动启动祖龙系统和连接器

### 方案 2：手动设置环境变量（PowerShell）

在 IDE 终端中先设置环境变量：

```powershell
# 设置环境变量
$env:LLM_BACKEND = "ollama"
$env:LLM_BASE_URL = "http://localhost:11434/v1"
$env:LLM_MODEL_ID = "qwen2.5:3b"
$env:LLM_MODEL_ID_BACKUP = "qwen2.5:0.5b"
$env:LLM_BASE_URL_BACKUP = "http://localhost:11434/v1"
$env:LLM_API_KEY = "EMPTY"

# 显示配置
Write-Host "LLM 配置:" -ForegroundColor Green
Write-Host "  后端：$env:LLM_BACKEND"
Write-Host "  API:  $env:LLM_BASE_URL"
Write-Host "  CORE: $env:LLM_MODEL_ID"
Write-Host "  BACKUP: $env:LLM_MODEL_ID_BACKUP"

# 启动祖龙系统
cd d:\AI\project\zulong_beta4
$env:PYTHONPATH = "d:\AI\project\zulong_beta4"
python zulong\bootstrap.py
```

### 方案 3：修复 Ollama 模型权限

如果需要使用 `qwen3.5:4b`，先修复权限：

```powershell
# 以管理员身份运行 PowerShell
# 获取模型目录
$modelsPath = "D:\AI\models"

# 获取当前用户
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

# 添加完全控制权限
$acl = Get-Acl $modelsPath
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule($currentUser, "FullControl", "ContainerInherit,ObjectInherit", "None", "Allow")
$acl.AddAccessRule($rule)
Set-Acl $modelsPath $acl

Write-Host "权限已修复！" -ForegroundColor Green

# 重新下载模型
ollama pull qwen3.5:4b
```

## ✅ 验证步骤

### 1. 检查 Ollama 服务

```powershell
# 测试 Ollama 原生 API
curl http://localhost:11434/api/tags

# 测试 OpenAI 兼容 API
curl http://localhost:11434/v1/models
```

### 2. 检查环境变量

```powershell
# 在 IDE 终端中运行
$env:LLM_BACKEND
$env:LLM_BASE_URL
$env:LLM_MODEL_ID
```

### 3. 测试模型调用

```powershell
# 使用 PowerShell 测试 Ollama OpenAI API
$body = @{
    model = "qwen2.5:3b"
    messages = @(@{role = "user"; content = "你好" })
} | ConvertTo-Json

$result = Invoke-RestMethod -Uri "http://localhost:11434/v1/chat/completions" -Method Post -Body $body -ContentType "application/json"
$result.choices.message.content
```

## 📝 配置说明

### 推荐配置（使用已有模型）

```
LLM_BACKEND=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_ID=qwen2.5:3b          # 主模型，1.8GB，性能较好
LLM_MODEL_ID_BACKUP=qwen2.5:0.5b  # 备用模型，0.37GB，快速响应
```

### 原配置（需要修复）

```
LLM_BACKEND=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_ID=deepseek-v3.1:671b-cloud  # 云端模型，可能需要特殊配置
LLM_MODEL_ID_BACKUP=qwen3.5:4b         # 本地模型，需要修复权限后下载
```

## 🎯 快速修复

**最简单的方法**：运行修复后的启动脚本

```batch
cd d:\AI\project\zulong_beta4
.\start_ollama_fixed.bat
```

这会自动：
1. 检查并下载所需模型
2. 设置正确的环境变量
3. 启动祖龙系统
4. 启动 OpenClaw 连接器

## 📊 性能对比

| 模型 | 大小 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|----------|
| `qwen2.5:0.5b` | 0.37GB | ⚡⚡ 极快 | ⭐⭐ 基础 | 快速响应、简单对话 |
| `qwen2.5:3b` | 1.8GB | ⚡⚡ 快 | ⭐⭐⭐ 良好 | 日常使用（推荐） |
| `qwen3.5:4b` | 3.4GB | ⚡ 中等 | ⭐⭐⭐⭐ 优秀 | 复杂任务 |
| `deepseek-v3.1:671b-cloud` | 云端 | 🐌 慢 | ⭐⭐⭐⭐⭐ 最佳 | 高质量需求 |

## 🔧 故障排查

### 问题 1：模型找不到

**症状**: "model 'xxx' not found"

**解决**:
```powershell
# 检查可用模型
ollama list

# 下载缺失模型
ollama pull qwen2.5:3b
```

### 问题 2：连接被拒绝

**症状**: "Connection refused" 或 "Cannot connect to Ollama"

**解决**:
```powershell
# 检查 Ollama 服务是否运行
Get-Process ollama -ErrorAction SilentlyContinue

# 如果没有运行，启动它
ollama serve
```

### 问题 3：权限错误

**症状**: "Access is denied" 下载模型时

**解决**: 以管理员身份运行 PowerShell，然后执行权限修复（见方案 3）

## 📞 总结

**终端不同不是问题！** 关键是：
1. ✅ 确保环境变量正确设置
2. ✅ 确保模型已下载且可用
3. ✅ 确保 Ollama 服务正常运行
4. ✅ 使用正确的模型名称

**推荐**: 使用 `start_ollama_fixed.bat` 一键启动！
