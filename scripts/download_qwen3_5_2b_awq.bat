@echo off
REM ========================================
REM 下载 Qwen3.5-2B-AWQ-4bit 模型
REM ========================================

echo ================================================================================
echo                下载 Qwen3.5-2B-AWQ-4bit 模型
echo ================================================================================
echo.
echo 模型信息:
echo   - 名称：Qwen3.5-2B-AWQ-4bit
echo   - 来源：ModelScope (cyankiwi)
echo   - 量化：AWQ 4bit
echo   - 显存占用：~1.5 GB
echo   - 兼容性：vLLM, SGLang, Transformers
echo.
echo 下载位置：models/Qwen/Qwen3___5-2B-AWQ
echo.
echo 按 Ctrl+C 取消下载
echo ================================================================================
echo.

REM 检查 modelscope-cli 是否安装
where modelscope >nul 2>nul
if errorlevel 1 (
    echo [ERROR] modelscope-cli 未安装
    echo.
    echo 请先安装：pip install modelscope
    echo.
    pause
    exit /b 1
)

echo [OK] modelscope-cli 已就绪
echo.

REM 创建模型目录
if not exist "models\Qwen" mkdir "models\Qwen"

REM 下载模型
echo [START] 开始下载...
echo.

modelscope download cyankiwi/Qwen3.5-2B-AWQ-4bit --local_dir models\Qwen\Qwen3___5-2B-AWQ

echo.
echo [OK] 下载完成！
echo.
echo 模型路径：models\Qwen\Qwen3___5-2B-AWQ
echo.
echo 下一步：运行 start_vllm_wsl2_2b_awq.bat 启动 vLLM 服务
echo.

pause
