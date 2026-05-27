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

if not defined ZULONG_HOME set "ZULONG_HOME=%~dp0.."
REM 将 Windows 路径转为 WSL 路径 (D:\path\to\project -> /mnt/d/path/to/project)
for %%I in ("%ZULONG_HOME%") do set "ZULONG_DRIVE=%%~dI"
set "ZULONG_DRIVE_LOWER=%ZULONG_DRIVE:~0,1%"
call :tolower ZULONG_DRIVE_LOWER
set "ZULONG_WSL=/mnt/%ZULONG_DRIVE_LOWER%%%ZULONG_HOME:~2%"
set "ZULONG_WSL=%ZULONG_WSL:\=/%"

wsl bash -c "source ~/vllm-env/bin/activate && export VLLM_USE_MODELSCOPE=true && vllm serve %ZULONG_WSL%/models/Qwen/Qwen3___5-0.8B-AWQ --port 8000 --host 0.0.0.0 --gpu-memory-utilization 0.5 --max-model-len 4096 --enable-auto-tool-choice --tool-call-parser qwen3_xml"
goto :eof

:tolower
for %%a in ("A=a" "B=b" "C=c" "D=d" "E=e" "F=f" "G=g" "H=h" "I=i" "J=j" "K=k" "L=l" "M=m" "N=n" "O=o" "P=p" "Q=q" "R=r" "S=s" "T=t" "U=u" "V=v" "W=w" "X=x" "Y=y" "Z=z") do call set "%1=%%%1:%%~a%%"
goto :eof

pause
