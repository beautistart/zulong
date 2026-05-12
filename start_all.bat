@echo off
chcp 65001 >nul
echo ================================================================================
echo   祖龙 (ZULONG) 系统 - 完整启动（vLLM + 主系统 + 连接器）
echo ================================================================================
echo.
echo 启动顺序：
echo   [1] WSL vLLM  -  L2_CORE 推理服务 (端口 8000, gpu-util 0.5)
echo   [2] WSL vLLM  -  L2_BACKUP 推理服务 (端口 8001, gpu-util 0.4)
echo   [3] ZULONG    -  祖龙主系统
echo   [4] Bridge    -  OpenClaw 连接器 (等待主系统就绪后自动连接)
echo ================================================================================
echo.

REM 设置环境变量
set USE_VLLM_FOR_L2=true

echo [1/4] 正在启动 WSL vLLM L2_CORE 推理服务...
start "vLLM L2_CORE (Port 8000)" cmd /k "wsl bash -c ""source ~/vllm-env/bin/activate && export VLLM_USE_MODELSCOPE=true && vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.5 --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser qwen3_xml"""
echo   L2_CORE 启动中...
echo.

echo [2/4] 正在启动 WSL vLLM L2_BACKUP 推理服务...
start "vLLM L2_BACKUP (Port 8001)" cmd /k "wsl bash -c ""source ~/vllm-env/bin/activate && export VLLM_USE_MODELSCOPE=true && vllm serve /mnt/d/AI/project/zulong_beta4/models/Qwen/Qwen3___5-0.8B-AWQ-backup --port 8001 --host 0.0.0.0 --gpu-memory-utilization 0.4 --max-model-len 4096"""
echo   L2_BACKUP 启动中...
echo.

timeout /t 5 /nobreak >nul

echo [3/4] 正在启动祖龙主系统...
start "ZULONG System" cmd /k "cd /d d:\AI\project\zulong_beta4 && python -m zulong.bootstrap"
echo.

echo [4/4] 正在启动 OpenClaw Bridge 连接器（将自动等待主系统就绪）...
start "OpenClaw Bridge" cmd /k "cd /d d:\AI\project\zulong_beta4 && python -m openclaw_bridge.bootstrap"
echo.

echo ================================================================================
echo   系统启动中...
echo ================================================================================
echo.
echo   🤖 vLLM L2_CORE：http://localhost:8000/v1
echo   🤖 vLLM L2_BACKUP：http://localhost:8001/v1
echo   🔌 WebSocket：ws://localhost:5555
echo   🌐 Web 界面：http://localhost:8080
echo   💬 API 服务：http://localhost:3000
echo.
echo   注意：vLLM 需要约 60 秒加载模型，祖龙和连接器会自动等待。
echo   全部就绪后浏览器将自动打开 Web 界面。
echo ================================================================================
echo.

timeout /t 65 /nobreak >nul
start http://localhost:8080

pause
