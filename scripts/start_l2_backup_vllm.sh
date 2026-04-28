#!/bin/bash

# L2-BACKUP vLLM 启动脚本
# 模型：Qwen3.5-0.8B-int4-AutoRound
# 端口：8001
# 量化：INT4-AutoRound (W4A16)
# 加速：gptq_marlin

echo "================================================================================"
echo "启动 L2-BACKUP vLLM 服务"
echo "================================================================================"
echo ""

# 激活虚拟环境
source ~/vllm-env/bin/activate

# 设置环境变量
export VLLM_USE_MODELSCOPE=true
echo "[INFO] 已设置 VLLM_USE_MODELSCOPE=true"
echo ""

# 模型配置
MODEL_PATH="Intel/Qwen3.5-0.8B-int4-AutoRound"
PORT=8001
GPU_MEMORY=0.8
MAX_LEN=4096
QUANT="gptq_marlin"

echo "模型信息:"
echo "  - 模型 ID: ${MODEL_PATH}"
echo "  - 量化格式：INT4-AutoRound (W4A16)"
echo "  - 加速后端：${QUANT}"
echo "  - 服务端口：${PORT}"
echo "  - 显存占用：${GPU_MEMORY}"
echo "  - 最大长度：${MAX_LEN}"
echo ""

echo "================================================================================"
echo "启动 vLLM 服务..."
echo "================================================================================"
echo ""

# 启动 vLLM 服务
vllm serve ${MODEL_PATH} \
    --port ${PORT} \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization ${GPU_MEMORY} \
    --max-model-len ${MAX_LEN} \
    --trust-remote-code \
    --dtype auto \
    --quantization ${QUANT}

echo ""
echo "================================================================================"
echo "vLLM 服务已停止"
echo "================================================================================"
