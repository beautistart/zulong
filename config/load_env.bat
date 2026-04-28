@echo off
REM ================================================================================
REM 祖龙 (ZULONG) 系统环境变量加载脚本 (Windows 批处理版本)
REM 用法：call load_env.bat
REM ================================================================================

echo ================================================================
echo   祖龙系统 - 环境变量加载器
echo ================================================================

REM 设置默认环境变量
set ZULONG_ENV=production
set ZULONG_LLM_BACKEND=ollama
set ZULONG_OLLAMA_BASE_URL=http://localhost:11434/v1
set ZULONG_OLLAMA_MODEL_ID=deepseek-v3.1:671b-cloud
set ZULONG_OLLAMA_BACKUP_MODEL_ID=qwen3.5:4b
set ZULONG_OLLAMA_API_KEY=EMPTY

REM L2 推理配置
set ZULONG_L2_CORE_MODEL=%ZULONG_OLLAMA_MODEL_ID%
set ZULONG_L2_BACKUP_MODEL=%ZULONG_OLLAMA_BACKUP_MODEL_ID%
set ZULONG_L2_MAX_TOKENS=1024
set ZULONG_L2_TEMPERATURE=0.3
set ZULONG_L2_TOP_P=0.85

REM 视觉系统配置
set ZULONG_CAMERA_ENABLED=false
set ZULONG_YOLO_MODEL_PATH=yolov10n.pt

REM 音频系统配置
set ZULONG_MICROPHONE_ENABLED=true
set ZULONG_SPEAKER_ENABLED=true
set ZULONG_TTS_BACKEND=cosyvoice

REM 记忆系统配置
set ZULONG_RAG_ENABLED=true
set ZULONG_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

REM 工具系统配置
set ZULONG_OPENCLAW_ENABLED=true
set ZULONG_OPENCLAW_API_URL=http://localhost:3000
set ZULONG_OPENCLAW_WEBSOCKET_URL=ws://localhost:5555
set ZULONG_WEB_SEARCH_ENABLED=true

REM Web 服务配置
set ZULONG_API_HOST=localhost
set ZULONG_API_PORT=3000
set ZULONG_WEBSOCKET_HOST=localhost
set ZULONG_WEBSOCKET_PORT=5555

REM 安全配置
set ZULONG_API_KEY=zulong-default-key-change-in-production

REM 监控配置
set ZULONG_PERFORMANCE_MONITORING_ENABLED=true
set ZULONG_DEBUG_CONSOLE_ENABLED=true

REM 日志配置
set ZULONG_LOG_LEVEL=INFO
set ZULONG_DEBUG_MODE=false

echo [OK] 环境变量已设置
echo   - LLM 后端：%ZULONG_LLM_BACKEND%
echo   - Ollama 地址：%ZULONG_OLLAMA_BASE_URL%
echo   - 核心模型：%ZULONG_L2_CORE_MODEL%
echo   - 备用模型：%ZULONG_L2_BACKUP_MODEL%
echo   - 环境：%ZULONG_ENV%
echo ================================================================

REM 如果存在自定义 .env 文件，加载它
if exist "config\.env" (
    echo [INFO] 发现自定义 .env 文件，正在加载...
    for /f "delims=" %%a in (config\.env) do (
        setlocal enabledelayedexpansion
        set "line=%%a"
        if not "!line:~0,1!"=="#" (
            if not "!line!"=="" (
                for /f "tokens=1,* delims==" %%b in ("!line!") do (
                    endlocal
                    set "%%b=%%c"
                    echo   [加载] %%b=%%c
                    setlocal enabledelayedexpansion
                )
            )
        )
        endlocal
    )
    echo [OK] 自定义环境变量已加载
) else (
    echo [INFO] 未找到 config\.env 文件，使用默认配置
)

echo ================================================================
echo 环境变量加载完成！
echo ================================================================
