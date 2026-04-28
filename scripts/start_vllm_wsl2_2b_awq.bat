@echo off
chcp 65001 >nul
REM ========================================
REM WSL2 vLLM Server 启动脚本 (L2_CORE 专用)
REM 模型：Qwen3.5-0.8B-AWQ
REM 端口：8000
REM ========================================

echo ================================================================================
echo              WSL2 vLLM Server 启动脚本 (L2_CORE 专用)
echo ================================================================================
echo.
echo 模型配置:
echo   - L2_CORE: Qwen3.5-0.8B-AWQ
echo   - 服务端口：8000
echo   - gpu-memory-utilization: 0.5
echo   - max-model-len: 4096
echo   - 支持工具调用 (qwen3_xml)
echo.
echo 按 Ctrl+C 停止服务
echo ================================================================================
echo.

REM 检查 WSL 是否可用
wsl --status >nul 2>&1
if errorlevel 1 (
    echo [ERROR] WSL 未安装或不可用
    pause
    exit /b 1
)

echo [OK] WSL 已就绪
echo.
echo [START] 启动 vLLM Server (L2_CORE, 端口 8000)...
echo.

wsl bash -c "source ~/vllm-env/bin/activate && export VLLM_USE_MODELSCOPE=true && vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.5 --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser qwen3_xml"

pause
