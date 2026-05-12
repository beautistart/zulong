# L2 服务日志监控指南

**创建时间**: 2026-04-10  
**目的**: 监控 L2 CORE 和 L2 BACKUP 的实际请求情况  

---

## 🔍 监控方法

### 方法 1: 查看 vLLM 服务日志

#### L2 CORE (端口 8000)
**终端**: 31  
**命令**: 
```bash
# 查看最近的推理请求
# 在终端 31 中滚动查看日志
```

**关键词**:
- `INFO` - 推理请求
- `chat.completions.create` - 调用记录
- `model: Qwen3___5-0.8B-AWQ` - 模型标识

#### L2 BACKUP (端口 8001)
**终端**: 32  
**命令**: 
```bash
# 查看最近的推理请求
# 在终端 32 中滚动查看日志
```

**关键词**:
- `INFO` - 推理请求
- `chat.completions.create` - 调用记录
- `model: Qwen3___5-0.8B-AWQ-backup` - 模型标识

---

### 方法 2: 实时监控连接

**PowerShell 脚本** (保存为 `monitor_l2_ports.ps1`):

```powershell
# 持续监控 L2 端口连接
while ($true) {
    Clear-Host
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] === L2 端口监控 ===" -ForegroundColor Cyan
    
    Write-Host "`n=== L2 CORE (8000) ===" -ForegroundColor Green
    $conn_8000 = netstat -ano | findstr :8000 | findstr ESTABLISHED
    if ($conn_8000) {
        Write-Host $conn_8000
        Write-Host "活跃连接数：$($conn_8000.Count)" -ForegroundColor Green
    } else {
        Write-Host "无活跃连接" -ForegroundColor Gray
    }
    
    Write-Host "`n=== L2 BACKUP (8001) ===" -ForegroundColor Yellow
    $conn_8001 = netstat -ano | findstr :8001 | findstr ESTABLISHED
    if ($conn_8001) {
        Write-Host $conn_8001
        Write-Host "活跃连接数：$($conn_8001.Count)" -ForegroundColor Yellow
    } else {
        Write-Host "无活跃连接" -ForegroundColor Gray
    }
    
    Start-Sleep -Seconds 2
}
```

**使用方法**:
```powershell
.\monitor_l2_ports.ps1
```

---

### 方法 3: 统计请求数量

**PowerShell 脚本** (保存为 `count_l2_requests.ps1`):

```powershell
# 统计过去 1 分钟的连接数
Write-Host "统计过去 1 分钟的 L2 请求..." -ForegroundColor Cyan

# 使用 Performance Counter 或事件日志
# 这里简化为检查当前连接

$requests_core = (netstat -ano | findstr :8000 | findstr ESTABLISHED).Count
$requests_backup = (netstat -ano | findstr :8001 | findstr ESTABLISHED).Count

Write-Host "`nL2 CORE (8000) 请求数：$requests_core" -ForegroundColor Green
Write-Host "L2 BACKUP (8001) 请求数：$requests_backup" -ForegroundColor Yellow

if ($requests_backup -gt $requests_core) {
    Write-Host "`n⚠️  警告：L2 BACKUP 请求数多于 L2 CORE" -ForegroundColor Red
    Write-Host "💡 系统可能在使用 L2 BACKUP 进行推理" -ForegroundColor Yellow
} else {
    Write-Host "`n✅ 正常：L2 CORE 是主要服务" -ForegroundColor Green
}
```

---

## 📊 日志分析要点

### L2 CORE 日志特征

**正常日志**:
```
INFO:     127.0.0.1:xxxxx - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:     Model: Qwen3___5-0.8B-AWQ
INFO:     Generated xxx tokens in x.xx sec
```

**异常日志**:
```
ERROR:    Connection timeout
ERROR:    Model loading failed
WARNING:  High latency detected
```

---

### L2 BACKUP 日志特征

**正常日志**:
```
INFO:     127.0.0.1:xxxxx - "POST /v1/chat/completions HTTP/1.1" 200 OK
INFO:     Model: Qwen3___5-0.8B-AWQ-backup
INFO:     Generated xxx tokens in x.xx sec
```

**如果看到大量日志**:
- 可能系统在使用 L2 BACKUP 作为主服务
- 可能有其他模块在调用 L2 BACKUP

---

## 🎯 验证步骤

### 步骤 1: 进行对话测试

在调试控制台中输入：
```
你好，请介绍一下你自己
```

然后立即查看：
1. **终端 31** (L2 CORE) 是否有推理日志
2. **终端 32** (L2 BACKUP) 是否有推理日志
3. **终端 33** (祖龙系统) 的响应日志

---

### 步骤 2: 对比日志时间戳

**关键**: 对比三个终端的日志时间戳

**预期**:
- 用户发送消息：`10:00:00`
- L2 CORE 接收请求：`10:00:00.100`
- L2 CORE 返回响应：`10:00:01.500`
- 祖龙系统输出：`10:00:01.600`

**如果实际是**:
- 用户发送消息：`10:00:00`
- L2 BACKUP 接收请求：`10:00:00.100` ← 异常
- L2 BACKUP 返回响应：`10:00:01.500`
- 祖龙系统输出：`10:00:01.600`

说明系统在使用 L2 BACKUP

---

### 步骤 3: 检查推理引擎日志

**搜索关键词**:
```
[vLLM] OpenAI 客户端已初始化
```

**预期看到**:
```
✅ [vLLM] OpenAI 客户端已初始化：http://localhost:8000/v1
```

**如果看到**:
```
✅ [vLLM] OpenAI 客户端已初始化：http://localhost:8001/v1
```
说明系统初始化时使用了 L2 BACKUP

---

## 🔧 故障排查

### 问题 1: L2 BACKUP 请求异常多

**可能原因**:
1. L2 CORE 启动失败，自动故障转移到 BACKUP
2. 配置文件中硬编码了 8001 端口
3. 其他模块直接调用 L2 BACKUP

**排查步骤**:
```powershell
# 1. 检查 L2 CORE 状态
netstat -ano | findstr :8000

# 2. 搜索硬编码端口
grep -r "localhost:8001" zulong/

# 3. 查看启动日志
# 在终端 33 中搜索 "L2" 关键词
```

---

### 问题 2: L2 CORE 无请求

**可能原因**:
1. vLLM 服务未正确启动
2. 防火墙阻止 8000 端口
3. 推理引擎配置错误

**排查步骤**:
```powershell
# 1. 重启 L2 CORE
# 在终端 31 中按 Ctrl+C，然后重新启动

# 2. 检查防火墙
netsh advfirewall firewall show rule name=all | findstr 8000

# 3. 检查推理引擎配置
# 查看 inference_engine.py 第 134-138 行
```

---

## 📝 报告模板

### 日志监控报告

**监控时间**: 2026-04-10 HH:MM - HH:MM  
**监控时长**: X 分钟

| 指标 | L2 CORE (8000) | L2 BACKUP (8001) |
|------|----------------|------------------|
| 请求总数 | X | X |
| 平均响应时间 | X ms | X ms |
| 错误数 | X | X |
| 活跃连接峰值 | X | X |

**结论**:
- [ ] L2 CORE 是主要服务
- [ ] L2 BACKUP 是主要服务
- [ ] 两者负载均衡

**建议**:
- ...

---

## 🎓 技术说明

### 为什么会有两个 L2 服务？

**设计目的**:
1. **冗余备份**: L2 CORE 出问题时 L2 BACKUP 可以接管
2. **热切换**: 不需要重启系统即可切换模型
3. **负载均衡**: 理论上可以分担请求压力

**实际使用**:
- 正常情况下只使用 L2 CORE
- L2 BACKUP 作为备用，平时不处理请求
- 只有在 L2 CORE 故障时才启用

---

**文档创建**: AI Assistant  
**创建日期**: 2026-04-10  
**用途**: 监控和诊断 L2 服务使用情况
