# File: Dockerfile
# 祖龙 (ZULONG) 系统生产环境 Docker 镜像

# 多阶段构建 - 第一阶段：构建环境
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（分层缓存优化）
RUN pip install --no-cache-dir --user -r requirements.txt

# 多阶段构建 - 第二阶段：运行环境
FROM nvidia/cuda:11.7-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # 模型配置
    ZULONG_MODEL_PATH=/models \
    ZULONG_DATA_PATH=/data \
    ZULONG_LOG_PATH=/logs \
    # 默认配置
    ZULONG_ENV=production \
    ZULONG_LOG_LEVEL=INFO \
    # GPU 配置
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# 创建非 root 用户（安全最佳实践）
RUN groupadd -r zulong && useradd -r -g zulong zulong

# 设置工作目录
WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /root/.local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin

# 复制项目代码
COPY zulong/ ./zulong/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

# 复制配置文件
COPY config/production.yml ./config.yml

# 创建必要的目录
RUN mkdir -p /models /data /logs /checkpoints && \
    chown -R zulong:zulong /app /models /data /logs /checkpoints

# 切换到非 root 用户
USER zulong

# 暴露端口（API 服务）
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# 入口点脚本
COPY <<EOF /entrypoint.sh
#!/bin/bash
set -e

echo "=== 祖龙 (ZULONG) 系统启动 ==="
echo "环境：\${ZULONG_ENV}"
echo "日志级别：\${ZULONG_LOG_LEVEL}"

# 创建必要目录
mkdir -p \${ZULONG_LOG_PATH} \${ZULONG_DATA_PATH}

# 检查模型文件
if [ ! -d "\${ZULONG_MODEL_PATH}/internvl" ]; then
    echo "⚠️  模型文件不存在，请挂载模型卷"
    echo "示例：docker run -v /path/to/models:/models ..."
fi

# 启动服务
echo "🚀 启动服务..."
exec python -m zulong.bootstrap
EOF

RUN chmod +x /entrypoint.sh

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]

# 默认命令
CMD []
