@echo off
REM ================================================================================
REM 祖龙 (ZULONG) 系统环境变量加载脚本 (Windows 批处理版本)
REM 用法：call load_env.bat
REM ================================================================================

echo ================================================================
echo   祖龙系统 - 环境变量加载器
echo ================================================================

REM ====== 项目路径配置（所有模块从此派生）======
REM ZULONG_HOME: 项目根目录（自动检测脚本所在目录的父目录）
for %%I in ("%~dp0..") do set "ZULONG_HOME=%%~fI"
set ZULONG_MODEL_BASE_DIR=%ZULONG_HOME%\models
set ZULONG_DATA_DIR=%ZULONG_HOME%\data

REM 设置默认环境变量
set ZULONG_ENV=production

REM 注意：LLM 后端 / 模型 / 密钥不再在此硬编码。
REM LLM 配置的唯一权威来源是 config/zulong_config.yaml 的 models.registry，
REM 由 Web 端「模型配置」页面（/api/models/registry/*）维护。
REM 此脚本只设置非 LLM 的平台默认值。

REM 视觉系统配置
set ZULONG_CAMERA_ENABLED=false
set ZULONG_YOLO_MODEL_PATH=models\yolov10n.pt

REM 音频系统配置
set ZULONG_MICROPHONE_ENABLED=true
set ZULONG_SPEAKER_ENABLED=true
set ZULONG_TTS_BACKEND=cosyvoice

REM 记忆系统配置
set ZULONG_RAG_ENABLED=true
set ZULONG_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

REM 工具系统配置
set ZULONG_WEB_SEARCH_ENABLED=true

REM Web 服务配置
set ZULONG_API_HOST=127.0.0.1
set ZULONG_API_PORT=8090
set ZULONG_WEBSOCKET_HOST=127.0.0.1
set ZULONG_WEBSOCKET_PORT=8090

REM 安全配置
set ZULONG_API_KEY=zulong-default-key-change-in-production

REM 监控配置
set ZULONG_PERFORMANCE_MONITORING_ENABLED=true
set ZULONG_DEBUG_CONSOLE_ENABLED=true

REM 日志配置
set ZULONG_LOG_LEVEL=INFO
set ZULONG_DEBUG_MODE=false

echo [OK] 环境变量已设置
echo   - 环境：%ZULONG_ENV%
echo   - LLM 配置：见 Web 端「模型配置」(models.registry)
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
                    echo   [加载] %%b
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

REM LLM_*_BACKUP 兼容变量已不再需要（registry 为唯一来源）
if not defined USE_VLLM_FOR_L2_BACKUP set "USE_VLLM_FOR_L2_BACKUP=false"

echo ================================================================
echo 环境变量加载完成！
echo ================================================================
