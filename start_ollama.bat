@echo off
chcp 65001 >nul
echo ================================================================================
echo   祖龙 (ZULONG) 系统 - Ollama 模式启动
echo ================================================================================
echo.
echo 启动顺序：
echo   [1] Ollama     -  本地 LLM 推理服务 (端口 11434)
echo   [2] ZULONG     -  祖龙主系统
echo   [3] Bridge     -  OpenClaw 连接器
echo ================================================================================
echo.

REM ============================
REM  LLM 后端配置
REM ============================
REM 后端类型: ollama
set LLM_BACKEND=ollama

REM Ollama 默认 API 地址（OpenAI 兼容）
set LLM_BASE_URL=http://localhost:11434/v1

REM ============================
REM  L2 CORE 主模型（云端 DeepSeek）
REM ============================
REM 云端模型，质量高但可能较慢
set LLM_MODEL_ID=deepseek-v3.1:671b-cloud

REM ============================
REM  L2 BACKUP 备用模型（本地 Qwen3.5）
REM ============================
REM 本地模型，速度快用于降级
set LLM_MODEL_ID_BACKUP=qwen3.5:4b
set LLM_BASE_URL_BACKUP=http://localhost:11434/v1

REM API Key（Ollama 本地不需要）
set LLM_API_KEY=EMPTY

REM 🔥 关键：不设置 USE_VLLM_FOR_L2，让 bootstrap.py 使用 Ollama 配置
REM set USE_VLLM_FOR_L2=true

echo [配置]
echo   后端:        %LLM_BACKEND%
echo   API:         %LLM_BASE_URL%
echo   CORE 模型:   %LLM_MODEL_ID%
echo   BACKUP 模型: %LLM_MODEL_ID_BACKUP%
echo.

REM ============================
REM  [1/3] 启动 Ollama 服务
REM ============================
echo [1/3] 检查 Ollama 服务...

REM 检查 Ollama 是否已安装
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo   [ERROR] Ollama 未安装！
    echo   请访问 https://ollama.com 下载安装
    echo   安装后运行: ollama pull %LLM_MODEL_ID%
    pause
    exit /b 1
)

REM 拉取模型（如果没有）
echo   正在确认 CORE 模型 %LLM_MODEL_ID% 已下载...
ollama pull %LLM_MODEL_ID%
echo   正在确认 BACKUP 模型 %LLM_MODEL_ID_BACKUP% 已下载...
ollama pull %LLM_MODEL_ID_BACKUP%
echo.

REM 启动 Ollama 服务（如果未运行）
echo   启动 Ollama 服务...
start "Ollama Server" cmd /k "ollama serve"
echo   等待 Ollama 就绪...
timeout /t 5 /nobreak >nul
echo.

REM ============================
REM  [2/3] 启动祖龙主系统
REM ============================
echo [2/3] 正在启动祖龙主系统...
start "ZULONG System" cmd /k "cd /d d:\AI\project\zulong_beta4 && python -m zulong.bootstrap"
echo.

REM ============================
REM  [3/3] 启动 OpenClaw 连接器
REM ============================
echo [3/3] 正在启动 OpenClaw Bridge 连接器...
start "OpenClaw Bridge" cmd /k "cd /d d:\AI\project\zulong_beta4 && python -m openclaw_bridge.bootstrap"
echo.

echo ================================================================================
echo   系统启动中 (Ollama 模式)
echo ================================================================================
echo.
echo   LLM CORE:   Ollama %LLM_MODEL_ID% (cloud)
echo   LLM BACKUP: Ollama %LLM_MODEL_ID_BACKUP% (local)
echo   Endpoint:   http://localhost:11434
echo   WebSocket:  ws://localhost:5555
echo   Web UI:     http://localhost:8080
echo   API:        http://localhost:3000
echo.
echo   降级策略: CORE 超时/不可用 -^> BACKUP 本地模型 -^> 静态降级回复
echo   如需更换模型，修改本文件中的 LLM_MODEL_ID / LLM_MODEL_ID_BACKUP
echo ================================================================================
echo.

timeout /t 15 /nobreak >nul
start http://localhost:8080

pause
