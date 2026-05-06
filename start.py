"""
祖龙统一启动器入口

用法:
    python start.py

启动后自动打开浏览器，在 Web 页面选择启动模式 (Full / IDE)。
"""

import sys
import os
import threading
import webbrowser

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from zulong.launcher.app import LauncherApp


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

    print(f"[Zulong] 统一启动器已启动: http://{host}:{port}")
    uvicorn.run(
        launcher.app, host=host, port=port, log_level="info",
        ws_ping_interval=None, ws_ping_timeout=None,
    )


if __name__ == "__main__":
    main()
