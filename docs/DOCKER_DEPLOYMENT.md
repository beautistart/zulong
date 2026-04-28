# 祖龙 (ZULONG) Docker 部署指南

**版本**: v1.0  
**创建日期**: 2026-03-30  
**适用环境**: 生产环境

---

## 📋 目录

1. [前置要求](#前置要求)
2. [快速开始](#快速开始)
3. [配置说明](#配置说明)
4. [部署步骤](#部署步骤)
5. [监控访问](#监控访问)
6. [故障排查](#故障排查)
7. [最佳实践](#最佳实践)

---

## 🎯 前置要求

### 硬件要求

- **GPU**: NVIDIA RTX 3060 6GB 或更高
- **内存**: 16GB RAM 或更高
- **存储**: 50GB 可用空间
- **网络**: 稳定的互联网连接（首次部署）

### 软件要求

- **操作系统**: Ubuntu 20.04+ / Windows 10/11
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA 驱动**: 515+
- **NVIDIA Container Toolkit**: 1.10+

### 安装 NVIDIA Container Toolkit

**Ubuntu**:
```bash
# 添加仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows**:
```powershell
# 使用 WSL2 + Docker Desktop
# 1. 安装 WSL2
wsl --install

# 2. 安装 Docker Desktop
# 下载：https://www.docker.com/products/docker-desktop

# 3. 启用 WSL2 集成
# Docker Desktop -> Settings -> Resources -> WSL Integration
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd zulong_beta4
```

### 2. 准备模型文件

```bash
# 创建模型目录
mkdir -p models

# 下载模型（或使用已有模型）
# 参考：docs/MODEL_DOWNLOAD.md
```

### 3. 一键启动

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看日志
docker-compose logs -f

# 检查状态
docker-compose ps
```

### 4. 访问服务

- **API 服务**: http://localhost:8000
- **健康检查**: http://localhost:8000/health
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/zulong123)

---

## ⚙️ 配置说明

### 环境变量

可以在 `docker-compose.yml` 中配置环境变量：

```yaml
environment:
  - ZULONG_ENV=production        # 环境名称
  - ZULONG_LOG_LEVEL=INFO        # 日志级别
  - ZULONG_MODEL_PATH=/models    # 模型路径
  - ZULONG_DATA_PATH=/data       # 数据路径
```

### 卷挂载

```yaml
volumes:
  - ./models:/models:ro          # 模型文件（只读）
  - ./data:/data                 # 数据持久化
  - ./logs:/logs                 # 日志文件
  - ./checkpoints:/checkpoints   # 检查点
```

### 资源限制

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
    
    limits:
      cpus: '4.0'
      memory: 16G
```

---

## 📦 部署步骤

### 步骤 1: 环境检查

```bash
# 检查 Docker
docker --version
docker-compose --version

# 检查 NVIDIA 支持
docker run --rm --gpus all nvidia/cuda:11.7-base nvidia-smi
```

### 步骤 2: 构建镜像

```bash
# 构建镜像
docker-compose build

# 查看镜像大小
docker images zulong-system
```

### 步骤 3: 启动服务

```bash
# 启动所有服务
docker-compose up -d

# 启动单个服务
docker-compose up -d zulong
docker-compose up -d prometheus
docker-compose up -d grafana
```

### 步骤 4: 验证部署

```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs zulong

# 测试 API
curl http://localhost:8000/health
```

### 步骤 5: 运行测试

```bash
# 进入容器
docker-compose exec zulong bash

# 运行测试
python -m pytest tests/ -v
```

---

## 📊 监控访问

### Prometheus 指标

访问：http://localhost:9090

**关键指标**:
- `zulong_inference_latency_seconds` - 推理延迟
- `zulong_cache_hit_rate` - 缓存命中率
- `zulong_gpu_memory_usage_bytes` - GPU 显存使用
- `zulong_task_success_total` - 任务成功数

### Grafana 仪表盘

访问：http://localhost:3000

**默认仪表盘**:
1. **系统资源监控** - CPU、内存、GPU
2. **性能指标监控** - 延迟、吞吐量
3. **业务指标监控** - 任务数、成功率
4. **错误追踪** - 错误类型、频率
5. **缓存性能** - 命中率、驱逐率

### 配置告警

在 Grafana 中配置告警规则：

1. 访问 Dashboard
2. 选择面板
3. 点击 Alert -> Create Alert
4. 设置阈值和通知渠道

---

## 🔧 故障排查

### 问题 1: 容器无法启动

**症状**: `Error response from daemon: could not select device driver`

**解决方案**:
```bash
# 安装 NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 问题 2: GPU 不可用

**症状**: `CUDA out of memory` 或 `No CUDA device`

**解决方案**:
```bash
# 检查 GPU 访问
docker run --rm --gpus all nvidia/cuda:11.7-base nvidia-smi

# 调整资源限制
# 编辑 docker-compose.yml，增加内存限制
```

### 问题 3: 镜像过大

**症状**: 镜像大小超过 10GB

**解决方案**:
```bash
# 使用多阶段构建
# 参考 Dockerfile 中的 builder 阶段

# 清理未使用的镜像
docker image prune -a
```

### 问题 4: 日志过多

**症状**: 磁盘空间不足

**解决方案**:
```bash
# 配置日志轮转
# 在 docker-compose.yml 中：
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"

# 清理日志
docker-compose down -v
```

---

## 🎯 最佳实践

### 1. 镜像优化

**多阶段构建**:
```dockerfile
FROM python:3.10-slim as builder
# 构建阶段...

FROM nvidia/cuda:11.7-runtime
# 运行阶段，只复制必要文件
COPY --from=builder /app /app
```

**层缓存优化**:
```dockerfile
# 先复制依赖文件
COPY requirements.txt .
RUN pip install -r requirements.txt

# 再复制代码
COPY . .
```

### 2. 资源管理

**限制资源使用**:
```yaml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 16G
    reservations:
      cpus: '2.0'
      memory: 8G
      devices:
        - driver: nvidia
          count: 1
```

### 3. 健康检查

**配置健康检查**:
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### 4. 日志管理

**结构化日志**:
```python
import logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

**日志轮转**:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 5. 数据持久化

**使用卷挂载**:
```yaml
volumes:
  - ./data:/data
  - ./models:/models:ro
  - ./logs:/logs
```

**避免数据丢失**:
```bash
# 定期备份
docker run --rm \
  -v zulong_data:/data \
  -v $(pwd)/backup:/backup \
  alpine tar czf /backup/data.tar.gz /data
```

---

## 📈 性能基准

### 镜像大小

```
理想大小:
├─ 基础镜像：~1GB
├─ Python 依赖：~2GB
├─ 应用代码：~100MB
└─ 总计：<5GB
```

### 启动时间

```
启动阶段:
├─ 容器启动：<5s
├─ 模型加载：<20s
└─ 总计：<30s
```

### 资源使用

```
运行时资源:
├─ GPU 显存：4-6GB
├─ CPU: 2-4 核
├─ 内存：8-16GB
└─ 磁盘：10-20GB
```

---

## 🔄 更新与回滚

### 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建
docker-compose build

# 重启服务
docker-compose up -d --force-recreate
```

### 回滚

```bash
# 使用旧版本镜像
docker-compose down
docker-compose up -d zulong-system:previous
```

---

## 📚 相关文档

- [Phase 8 规划](PHASE8_PLAN.md)
- [监控配置指南](MONITORING_SETUP.md)
- [生产环境指南](PRODUCTION_SETUP.md)
- [性能基准报告](PERFORMANCE_BENCHMARK.md)

---

**文档版本**: v1.0  
**最后更新**: 2026-03-30  
**维护者**: 祖龙 (ZULONG) 系统架构组
