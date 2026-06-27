"""
祖龙统一启动器入口

用法:
    python start.py

启动后自动打开浏览器，在 Web 页面选择启动模式 (Full / IDE)。
按 Ctrl+C 优雅关闭所有服务。
"""

import sys
import os
import signal
import threading
import webbrowser
from pathlib import Path

# 🔥 设置项目根目录环境变量（所有模块从此派生路径）
_PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("ZULONG_HOME", str(_PROJECT_ROOT))


def _ensure_project_venv() -> None:
    """确保统一使用项目内 zulong_env 运行后端。

    用户可能直接运行系统 Python: `python start.py`。为避免出现系统 Python
    与项目虚拟环境依赖不一致（例如 litellm 只装在其中一个环境）的情况，
    启动入口在导入业务模块前自动重进 zulong_env 的解释器。
    """
    if os.environ.get("ZULONG_SKIP_VENV_REEXEC", "").lower() in {"1", "true", "yes"}:
        return

    if sys.platform == "win32":
        venv_python = _PROJECT_ROOT / "zulong_env" / "Scripts" / "python.exe"
    else:
        venv_python = _PROJECT_ROOT / "zulong_env" / "bin" / "python"

    if not venv_python.exists():
        print(f"[Zulong] 未找到项目虚拟环境解释器: {venv_python}", flush=True)
        print("[Zulong] 将继续使用当前 Python，但建议先创建 zulong_env。", flush=True)
        return

    current = Path(sys.executable).resolve()
    target = venv_python.resolve()
    if os.name == "nt":
        same_python = str(current).casefold() == str(target).casefold()
    else:
        same_python = current == target
    if same_python:
        return

    print(f"[Zulong] 切换到项目虚拟环境: {target}", flush=True)
    os.execv(str(target), [str(target), str(_PROJECT_ROOT / "start.py"), *sys.argv[1:]])


if __name__ == "__main__":
    _ensure_project_venv()
    if "--show-python" in sys.argv:
        print(sys.executable)
        raise SystemExit(0)

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(_PROJECT_ROOT))

import uvicorn
from zulong.launcher.app import LauncherApp

# 全局关闭标志
_shutdown_flag = threading.Event()


def main():
    launcher = LauncherApp()
    host, port = launcher.host, launcher.port

    # 延迟打开浏览器（等 uvicorn 就绪）
    if launcher.auto_open_browser:
        def _open():
            import time
            time.sleep(1.5)
            url = f"http://{host}:{port}"
            print(f"[Zulong] 正在打开浏览器: {url}")
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    # 🔥 信号处理：确保 Ctrl+C 能优雅退出
    def _handle_shutdown(signum, frame):
        signame = signal.Signals(signum).name
        print(f"\n[Zulong] 收到 {signame} 信号，正在关闭...")
        _shutdown_flag.set()
        # 使用 os._exit 确保立即退出（Windows 上 uvicorn 可能无法正常响应信号）
        os._exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    # Windows Ctrl+Break 事件
    if sys.platform == 'win32':
        try:
            signal.signal(signal.SIGBREAK, _handle_shutdown)
        except (AttributeError, OSError):
            pass

    print(f"[Zulong] 统一启动器已启动: http://{host}:{port}")
    print(f"[Zulong] 按 Ctrl+C 停止服务")

    try:
        uvicorn.run(
            launcher.app, host=host, port=port, log_level="info",
            ws_ping_interval=None, ws_ping_timeout=None,
            timeout_graceful_shutdown=3,  # 🔥 3 秒后强制退出，不再卡住
        )
    except KeyboardInterrupt:
        print("\n[Zulong] 收到 KeyboardInterrupt，正在关闭...")
    finally:
        if not _shutdown_flag.is_set():
            print("[Zulong] system stopped")


if __name__ == "__main__":
    main()
