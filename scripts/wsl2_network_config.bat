@echo off
REM ========================================
REM WSL2 网络配置脚本
REM 配置 WSL2 网络镜像，使 Windows 可以访问 WSL2 端口
REM ========================================

echo ================================================================================
echo                         WSL2 网络配置脚本
echo ================================================================================
echo.
echo 此脚本将配置 WSL2 网络镜像，使 Windows 可以访问 WSL2 中的服务
echo.
echo ================================================================================
echo.

REM 获取 WSL2 IP 地址
echo [INFO] 获取 WSL2 IP 地址...
wsl hostname -I >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 无法获取 WSL2 IP 地址
    echo 请确保 WSL2 正在运行
    pause
    exit /b 1
)

for /f "delims=" %%i in ('wsl hostname -I') do set WSL2_IP=%%i
echo [OK] WSL2 IP: %WSL2_IP%
echo.

REM 配置端口转发（可选，通常不需要）
echo [INFO] 配置说明:
echo.
echo 1. WSL2 会自动将 localhost 端口转发到 Windows
echo 2. 启动 vLLM Server 后，Windows 可以通过 http://localhost:8000 访问
echo.
echo 如果遇到网络问题，可以尝试以下方法:
echo.
echo 方法 1: 使用 WSL2 IP 直接访问
echo   http://%WSL2_IP%:8000/v1
echo.
echo 方法 2: 配置防火墙规则
echo   在 Windows Defender 防火墙中添加入站规则，允许端口 8000
echo.
echo 方法 3: 重启 WSL2 网络
echo   wsl --shutdown
echo   然后重新启动 vLLM Server
echo.
echo ================================================================================
echo.

pause
