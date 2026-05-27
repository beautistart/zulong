#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时监控WebSocket断开原因"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# 设置UTF-8编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

APPDATA = os.environ.get('APPDATA', '')
CODEARTS_LOGS = Path(APPDATA) / 'codearts-agent' / 'logs'
ZULONG_HOME = Path(os.environ.get('ZULONG_HOME', Path(__file__).resolve().parent))
ZULONG_BACKEND = ZULONG_HOME / 'logs'

print("=" * 80)
print("WebSocket断开监控 - 等待任务执行...")
print("=" * 80)
print("\n请在CodeArts Agent中执行一个任务（如输入'你好'）\n")

# 监控后端IDE日志
backend_logs = sorted(ZULONG_BACKEND.glob('zulong_ide_*.log'), reverse=True)
if backend_logs:
    backend_log = backend_logs[0]
    print(f"监控后端日志: {backend_log}")
    
    # 使用tail -f实时监控
    proc = subprocess.Popen(
        ['tail', '-f', str(backend_log)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        errors='ignore',
        bufsize=1
    )
    
    keywords = [
        'WebSocket', 'connected', 'disconnect', 'close',
        'session_start', 'task_complete', 'task_error',
        'FC任务', '会话清理', '断开', 'IDE_SESSION'
    ]
    
    try:
        for line in iter(proc.stdout.readline, ''):
            # 高亮关键日志
            if any(kw.lower() in line.lower() for kw in keywords):
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"[{timestamp}] {line.rstrip()}")
                
                # 特别标记WebSocket关闭
                if 'WebSocket 断开' in line or 'WebSocket closed' in line:
                    print("\n" + "!" * 80)
                    print("!!! WebSocket断开检测 !!!")
                    print("!" * 80 + "\n")
                    
    except KeyboardInterrupt:
        proc.terminate()
        print("\n监控已停止")
    except Exception as e:
        proc.terminate()
        print(f"\n错误: {e}")
