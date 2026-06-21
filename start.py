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

# 🔥 设置项目根目录环境变量（所有模块从此派生路径）
os.environ.setdefault("ZULONG_HOME", os.path.dirname(os.path.abspath(__file__)))

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
