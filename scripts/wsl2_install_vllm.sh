#!/bin/bash
# WSL2 vLLM 安装脚本
# 用于在 WSL2 (Ubuntu) 环境中安装 vLLM 和相关依赖

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  WSL2 vLLM 安装脚本"
echo "=========================================="
echo ""

# 检查是否为 WSL 环境
if ! grep -q "microsoft" /proc/version 2>/dev/null; then
    echo "❌ 此脚本必须在 WSL 环境中运行"
    exit 1
fi

echo "✅ 检测到 WSL 环境"
echo ""

# 1. 检查 Python 版本
echo "📦 步骤 1: 检查 Python 环境..."
python3 --version || {
    echo "❌ Python3 未安装，请先安装：sudo apt install python3 python3-pip"
    exit 1
}

PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d'.' -f1,2)
echo "   Python 版本：$PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" ]]; then
    echo "⚠️  警告：Python 版本可能不兼容，建议使用 Python 3.10-3.12"
fi
echo ""

# 2. 检查 NVIDIA GPU
echo "📦 步骤 2: 检查 NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi 未找到，请确保已安装 NVIDIA 驱动"
    echo "   在 Windows 主机上更新 NVIDIA 驱动，WSL2 会自动使用"
    exit 1
fi

nvidia-smi --query-gpu=name --format=csv,noheader | head -1
echo "✅ NVIDIA GPU 已就绪"
echo ""

# 3. 检查 CUDA
echo "📦 步骤 3: 检查 CUDA..."
if [ -f /usr/local/cuda/version.txt ]; then
    cat /usr/local/cuda/version.txt
else
    echo "⚠️  CUDA 版本信息不可读，继续安装..."
fi
echo ""

# 4. 创建虚拟环境
echo "📦 步骤 4: 创建 Python 虚拟环境..."
VENV_DIR="$HOME/vllm-env"

if [ -d "$VENV_DIR" ]; then
    echo "   虚拟环境已存在，将重新创建"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
echo "✅ 虚拟环境创建成功：$VENV_DIR"
echo ""

# 5. 激活虚拟环境
echo "📦 步骤 5: 激活虚拟环境..."
source "$VENV_DIR/bin/activate"
echo "✅ 虚拟环境已激活"
echo ""

# 6. 升级 pip
echo "📦 步骤 6: 升级 pip 和构建工具..."
pip install --upgrade pip setuptools wheel
echo ""

# 7. 安装 PyTorch (CUDA 版本)
echo "📦 步骤 7: 安装 PyTorch (CUDA 版本)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# 8. 安装 vLLM
echo "📦 步骤 8: 安装 vLLM..."
pip install vllm==0.6.3
echo ""

# 9. 安装 ModelScope (可选，用于国内加速)
echo "📦 步骤 9: 安装 ModelScope..."
pip install modelscope
echo ""

# 10. 验证安装
echo "📦 步骤 10: 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
echo ""

# 11. 创建启动脚本
echo "📦 步骤 11: 创建启动脚本..."
cat > "$HOME/start_vllm.sh" << 'EOF'
#!/bin/bash
# vLLM 启动脚本

source "$HOME/vllm-env/bin/activate"

echo "=========================================="
echo "  启动 vLLM Server"
echo "=========================================="
echo ""

# 设置环境变量
export VLLM_USE_MODELSCOPE=true

# 启动 vLLM
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8
EOF

chmod +x "$HOME/start_vllm.sh"
echo "✅ 启动脚本创建成功：$HOME/start_vllm.sh"
echo ""

# 12. 创建配置说明
echo "📦 步骤 12: 创建配置说明..."
cat > "$HOME/README.txt" << 'EOF'
vLLM 安装成功！

使用方法:
1. 启动 vLLM Server:
   ~/start_vllm.sh

2. 测试连接 (在另一个终端):
   curl http://localhost:8000/v1/models

3. Windows 主机访问:
   http://localhost:8000/v1

注意:
- WSL2 网络是隔离的，需要确保端口可访问
- 如果遇到网络问题，可能需要配置 WSL2 网络镜像
EOF

echo "✅ 配置说明创建成功：$HOME/README.txt"
echo ""

echo "=========================================="
echo "  🎉 vLLM 安装完成！"
echo "=========================================="
echo ""
echo "下一步操作:"
echo "1. 运行 ~/start_vllm.sh 启动 vLLM Server"
echo "2. 在 Windows 上测试连接 http://localhost:8000/v1"
echo "3. 启动 Zulong 系统，启用 vLLM 支持"
echo ""
echo "详细文档：d:\\AI\\project\\zulong_beta4\\docs\\VLLM_MIGRATION.md"
echo ""
