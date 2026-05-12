# 祖龙系统启动指南

## 📦 核心启动文件说明

### ✅ 推荐使用的启动文件

| 文件 | 用途 | 说明 |
|------|------|------|
| **`scripts/start_zulong.py`** | **主启动脚本** | 设置环境变量并启动祖龙系统 |
| **`scripts/start_vllm_wsl2_2b_awq.bat`** | L2_CORE vLLM 服务 | 启动 Qwen3.5-2B-AWQ（端口 8000） |
| **`scripts/start_vllm_wsl2_2b_awq_backup.bat`** | L2_BACKUP vLLM 服务 | 启动 Qwen3.5-2B-AWQ-backup（端口 8001） |

### ⚠️ 不要直接运行的文件

| 文件 | 原因 |
|------|------|
| `zulong/bootstrap.py` | 这是核心引导模块，被 `start_zulong.py` 导入使用，直接运行会导致环境变量不生效 |

## 🚀 标准启动流程

### 方案 A：单实例模式（推荐，节省显存）

只启动 L2_CORE，L2_BACKUP 不加载：

```bash
# 1. 启动 L2_CORE 的 vLLM 服务
scripts\start_vllm_wsl2_2b_awq.bat

# 2. 启动祖龙系统（新终端）
python scripts\start_zulong.py
```

**适用场景**:
- ✅ 日常使用
- ✅ 显存有限（<4GB）
- ✅ 开发测试

### 方案 B：双实例模式（冗余备份）

同时启动 L2_CORE 和 L2_BACKUP：

```bash
# 1. 启动 L2_CORE 的 vLLM 服务（端口 8000）
scripts\start_vllm_wsl2_2b_awq.bat

# 2. 启动 L2_BACKUP 的 vLLM 服务（端口 8001）
scripts\start_vllm_wsl2_2b_awq_backup.bat

# 3. 修改 scripts/start_zulong.py 第 9 行：
#    os.environ["USE_VLLM_FOR_L2_BACKUP"] = "true"

# 4. 启动祖龙系统（新终端）
python scripts\start_zulong.py
```

**适用场景**:
- ✅ 生产环境
- ✅ 需要高可用性
- ✅ 显存充足（>=4GB）

## 📝 配置说明

### 环境变量（在 `scripts/start_zulong.py` 中设置）

```python
os.environ["USE_VLLM_FOR_L2"] = "true"           # 是否使用 vLLM 加载 L2_CORE
os.environ["USE_VLLM_FOR_L2_BACKUP"] = "false"   # 是否使用 vLLM 加载 L2_BACKUP
os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"      # L2_CORE 的 vLLM 地址
os.environ["VLLM_BACKUP_BASE_URL"] = "http://localhost:8001/v1"  # L2_BACKUP 的 vLLM 地址
```

### vLLM 服务配置

| 参数 | L2_CORE | L2_BACKUP |
|------|---------|-----------|
| **模型** | Qwen3___5-2B-AWQ | Qwen3___5-2B-AWQ-backup |
| **端口** | 8000 | 8001 |
| **量化格式** | AWQ 4bit | AWQ 4bit |
| **显存利用率** | 0.8 | 0.4 |
| **最大上下文** | 4096 | 4096 |

## 🔍 验证服务状态

### 检查 vLLM 服务

```bash
# 检查 L2_CORE
curl http://localhost:8000/v1/models

# 检查 L2_BACKUP
curl http://localhost:8001/v1/models
```

### 检查祖龙系统

启动成功后会看到：
```
✅ [vLLM] OpenAI 客户端已初始化：http://localhost:8000/v1
✅ InferenceEngine 初始化完成
✅ DebugConsole initialized
```

## 🛠️ 故障排除

### 问题 1: 启动时提示 "模块不存在"
**解决**: 确保使用 `python scripts/start_zulong.py` 而不是直接运行 `bootstrap.py`

### 问题 2: vLLM 服务无法连接
**解决**: 
1. 检查 WSL2 是否运行：`wsl --status`
2. 检查端口是否被占用：`netstat -ano | findstr :8000`
3. 重启 vLLM 服务

### 问题 3: 显存不足
**解决**:
- 降低 `--gpu-memory-utilization` 参数（如 0.6）
- 只启动单实例模式
- 关闭其他 GPU 应用

## 📂 保留的脚本文件清单

### ✅ 保留的文件
- `scripts/start_zulong.py` - 主启动脚本
- `scripts/start_vllm_wsl2_2b_awq.bat` - L2_CORE vLLM 启动
- `scripts/start_vllm_wsl2_2b_awq_backup.bat` - L2_BACKUP vLLM 启动
- `scripts/download_qwen3_5_2b_awq.bat` - 模型下载脚本
- `scripts/wsl2_install_vllm.sh` - WSL2 环境安装
- `scripts/start_l2_backup_vllm.sh` - L2_BACKUP 启动（备用）
- `scripts/wsl2_network_config.bat` - WSL2 网络配置

### ❌ 已删除的文件
- `start_vllm_server.bat` - 废弃
- `start_vllm_wsl2.bat` - 废弃
- `start_vllm_wsl2_2b.bat` - 废弃
- `start_vllm_wsl2_2b_gptq.bat` - 废弃（使用 AWQ 替代）
- `start_vllm_wsl2_2b_simple.bat` - 废弃
- `start_zulong_with_dual_vllm.bat` - 废弃（使用 Python 版本）

## 💡 最佳实践

1. **开发环境**: 使用单实例模式，节省显存
2. **生产环境**: 使用双实例模式，提高可靠性
3. **不要直接运行** `bootstrap.py`，它是核心模块
4. **始终使用** `scripts/start_zulong.py` 启动系统
5. **先启动 vLLM**，再启动祖龙系统
