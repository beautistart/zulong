# L2 CORE vs L2 BACKUP 调用分析

**分析时间**: 2026-04-10  
**问题**: L2 BACKUP 请求很多，但 L2 CORE 请求很少  

---

## 🔍 配置分析

### 1. 默认配置

**文件**: `zulong/models/container.py` 第 62 行

```python
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
```

**默认值**: `http://localhost:8000/v1` → **L2 CORE**

---

### 2. 实际调用情况

根据日志观察：
- ❌ **L2 CORE (端口 8000)**: 请求很少
- ✅ **L2 BACKUP (端口 8001)**: 请求很多

**推测原因**: 系统可能在使用 L2 BACKUP 进行推理

---

## 🐛 可能的问题

### 问题 1: 环境变量覆盖

**检查点**: `zulong/bootstrap.py`

```python
os.environ['USE_VLLM_FOR_L2'] = 'true'
os.environ['USE_VLLM_FOR_L2_BACKUP'] = 'true'
```

**问题**: 没有设置 `VLLM_BASE_URL` 环境变量，使用默认值 8000

---

### 问题 2: 模型容器配置

**文件**: `zulong/models/container.py` 第 127-136 行

```python
# L2_CORE 配置
if USE_VLLM_FOR_L2:
    print(f"[ModelContainer] [vLLM] L2_CORE 将使用 vLLM OpenAI API，跳过本地加载")
    print(f"[ModelContainer] [INFO] 使用 0.8B AWQ 量化模型（端口 8000）")
    # 占位符：端口 8000
    self.resident_models[model_id] = {
        'path': 'vllm', 
        'type': 'remote', 
        'endpoint': 'http://localhost:8000/v1',  # ← L2 CORE
        'model_name': 'Qwen3___5-0.8B-AWQ'
    }
```

**但是**: 推理引擎可能在使用 L2 BACKUP 的 endpoint

---

### 问题 3: 推理引擎配置

**文件**: `zulong/l2/inference_engine.py` 第 134-138 行

```python
self.vllm_client = OpenAI(
    base_url=VLLM_BASE_URL,  # ← 使用默认的 8000 端口
    api_key="EMPTY"
)
logger.info(f"✅ [vLLM] OpenAI 客户端已初始化：{VLLM_BASE_URL}")
```

**预期**: 应该连接到 `http://localhost:8000/v1` (L2 CORE)

---

## 🔎 验证方法

### 方法 1: 检查日志中的初始化信息

在终端 33 中搜索：
```bash
# 查找 vLLM 客户端初始化日志
"[vLLM] OpenAI 客户端已初始化"
```

**预期看到**:
```
✅ [vLLM] OpenAI 客户端已初始化：http://localhost:8000/v1
```

**如果看到**:
```
✅ [vLLM] OpenAI 客户端已初始化：http://localhost:8001/v1
```
说明系统在使用 L2 BACKUP

---

### 方法 2: 检查实际请求

在 PowerShell 中运行：
```powershell
# 监控 8000 和 8001 端口的连接
netstat -ano | findstr :8000
netstat -ano | findstr :8001
```

**预期**:
- 8000 端口应该有大量 ESTABLISHED 连接（如果是 L2 CORE 在工作）
- 8001 端口应该很少连接

**实际可能**:
- 8000 端口连接很少
- 8001 端口连接很多 ← 说明在使用 L2 BACKUP

---

### 方法 3: 查看 vLLM 服务日志

**终端 31** (L2 CORE - 端口 8000):
```
vllm serve ... --port 8000
```

**终端 32** (L2 BACKUP - 端口 8001):
```
vllm serve ... --port 8001
```

查看哪个终端显示更多的推理请求日志。

---

## 📊 当前系统状态

### vLLM 服务运行状态

| 服务 | 端口 | 终端 | 状态 | 模型 |
|------|------|------|------|------|
| **L2 CORE** | 8000 | 终端 31 | ✅ 运行中 | Qwen3___5-0.8B-AWQ |
| **L2 BACKUP** | 8001 | 终端 32 | ✅ 运行中 | Qwen3___5-0.8B-AWQ-backup |

### 配置值

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | 默认指向 L2 CORE |
| `USE_VLLM_FOR_L2` | `true` | 启用 L2 CORE vLLM |
| `USE_VLLM_FOR_L2_BACKUP` | `true` | 启用 L2 BACKUP vLLM |

---

## 🐛 问题根源分析

### 可能原因 1: 代码中硬编码了 8001 端口

**检查**: 是否有代码直接使用了 `http://localhost:8001/v1`

**搜索**:
```bash
grep -r "localhost:8001" zulong/
```

---

### 可能原因 2: 环境变量被覆盖

**检查**: 是否有其他地方设置了 `VLLM_BASE_URL`

**搜索**:
```bash
grep -r "VLLM_BASE_URL" zulong/ --include="*.py"
```

---

### 可能原因 3: 推理引擎故障转移

**推测流程**:
1. 系统启动时尝试连接 L2 CORE (8000)
2. 连接失败或超时
3. 自动故障转移到 L2 BACKUP (8001)
4. 但日志没有明确显示故障转移

**检查代码**: `inference_engine.py` 中是否有故障转移逻辑

---

## ✅ 解决方案

### 方案 1: 强制使用 L2 CORE

**修改**: `zulong/bootstrap.py`

```python
# 添加明确配置
os.environ['VLLM_BASE_URL'] = 'http://localhost:8000/v1'  # 强制使用 L2 CORE
os.environ['USE_VLLM_FOR_L2'] = 'true'
os.environ['USE_VLLM_FOR_L2_BACKUP'] = 'false'  # 禁用 L2 BACKUP
```

---

### 方案 2: 添加故障转移日志

**修改**: `zulong/l2/inference_engine.py`

在 vLLM 客户端初始化时添加详细日志：

```python
try:
    self.vllm_client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key="EMPTY"
    )
    logger.info(f"✅ [vLLM] 客户端初始化成功：{VLLM_BASE_URL}")
    logger.info(f"🔍 [vLLM] 当前使用：{'L2 CORE' if '8000' in VLLM_BASE_URL else 'L2 BACKUP'}")
except Exception as e:
    logger.error(f"❌ [vLLM] 初始化失败：{e}")
    # 故障转移逻辑
    logger.info("🔄 [vLLM] 尝试故障转移到 L2 BACKUP...")
    self.vllm_client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="EMPTY"
    )
```

---

### 方案 3: 添加运行时状态显示

**修改**: 在调试控制台中添加命令查看当前使用的模型：

```python
# 在调试控制台中执行
from zulong.models.container import VLLM_BASE_URL
print(f"当前 vLLM 地址：{VLLM_BASE_URL}")
print(f"使用的模型：{'L2 CORE' if '8000' in VLLM_BASE_URL else 'L2 BACKUP'}")
```

---

## 📝 验证步骤

### 步骤 1: 确认当前使用的模型

在调试控制台中执行：
```python
from zulong.models.container import VLLM_BASE_URL
from zulong.l2.inference_engine import InferenceEngine

print(f"VLLM_BASE_URL: {VLLM_BASE_URL}")
print(f"使用的服务：{'L2 CORE (8000)' if '8000' in VLLM_BASE_URL else 'L2 BACKUP (8001)'}")

# 检查推理引擎状态
ie = InferenceEngine.get_instance()
if ie.vllm_client:
    print(f"vLLM 客户端：已初始化")
    print(f"客户端地址：{ie.vllm_client.base_url}")
else:
    print(f"vLLM 客户端：未初始化")
```

---

### 步骤 2: 监控端口连接

在 PowerShell 中运行：
```powershell
# 持续监控端口连接
while ($true) {
    Clear-Host
    Write-Host "=== L2 CORE (8000) ===" -ForegroundColor Green
    netstat -ano | findstr :8000 | findstr ESTABLISHED
    
    Write-Host "`n=== L2 BACKUP (8001) ===" -ForegroundColor Yellow
    netstat -ano | findstr :8001 | findstr ESTABLISHED
    
    Start-Sleep -Seconds 2
}
```

---

### 步骤 3: 对比日志

**终端 31** (L2 CORE):
- 查看是否有推理请求日志
- 记录请求数量

**终端 32** (L2 BACKUP):
- 查看是否有推理请求日志
- 记录请求数量

**终端 33** (祖龙系统):
- 搜索 `[vLLM]` 关键词
- 查看初始化地址

---

## 🎯 预期结果

### 正常情况（使用 L2 CORE）

- ✅ `VLLM_BASE_URL = http://localhost:8000/v1`
- ✅ 终端 31 (8000) 有大量请求日志
- ✅ 终端 32 (8001) 很少或没有请求
- ✅ 调试控制台显示使用 L2 CORE

### 异常情况（使用 L2 BACKUP）

- ❌ `VLLM_BASE_URL = http://localhost:8001/v1` (或被覆盖)
- ❌ 终端 31 (8000) 很少请求
- ❌ 终端 32 (8001) 有大量请求
- ❌ 调试控制台显示使用 L2 BACKUP

---

## 📞 后续操作

1. **运行验证脚本**确认当前使用的模型
2. **检查端口连接**确认请求流向
3. **对比日志**确认哪个服务在处理请求
4. **根据结果调整配置**确保使用正确的模型

---

**分析人员**: AI Assistant  
**分析日期**: 2026-04-10  
**状态**: 🔍 待验证
