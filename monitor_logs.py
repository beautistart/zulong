#!/usr/bin/env python3
"""实时监控祖龙IDE前后端日志"""

import os
import time
import subprocess
from pathlib import Path
from datetime import datetime

# 日志路径
APPDATA = os.environ.get('APPDATA', '')
CODEARTS_LOGS = Path(APPDATA) / 'codearts-agent' / 'logs'
ZULONG_HOME = Path(os.environ.get('ZULONG_HOME', Path(__file__).resolve().parent))
ZULONG_BACKEND_LOGS = ZULONG_HOME / 'logs'

# 查找最新的前端日志目录
def find_latest_frontend_log():
    log_dirs = sorted(CODEARTS_LOGS.glob('2026*'), reverse=True)
    if log_dirs:
        latest = log_dirs[0]
        # 查找output_logging目录
        for output_dir in latest.rglob('output_logging_*'):
            zulong_log = output_dir / '1-Zulong.log'
            if zulong_log.exists():
                return zulong_log
    return None

# 查找最新的后端日志
def find_latest_backend_log():
    logs = sorted(ZULONG_BACKEND_LOGS.glob('zulong_ide_*.log'), reverse=True)
    return logs[0] if logs else None

# 使用tail -f监控日志
def monitor_logs():
    frontend_log = find_latest_frontend_log()
    backend_log = find_latest_backend_log()
    
    print(f"前端日志: {frontend_log}")
    print(f"后端日志: {backend_log}")
    print("=" * 80)
    
    # 启动tail -f监控两个日志文件
    processes = []
    
    try:
        # 监控前端日志
        if frontend_log and frontend_log.exists():
            p1 = subprocess.Popen(
                ['tail', '-f', str(frontend_log)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append(('前端', p1))
        
        # 监控后端日志
        if backend_log and backend_log.exists():
            p2 = subprocess.Popen(
                ['tail', '-f', str(backend_log)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            processes.append(('后端', p2))
        
        # 持续读取输出
        import select
        while True:
            for name, proc in processes:
                if proc.stdout:
                    # 非阻塞读取
                    import sys
                    import msvcrt  # Windows
                    
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
        for _, proc in processes:
            proc.terminate()

if __name__ == '__main__':
    # 先打印现有日志内容
    frontend_log = find_latest_frontend_log()
    backend_log = find_latest_backend_log()
    
    print(f"{'='*80}")
    print(f"前端日志: {frontend_log}")
    print(f"{'='*80}")
    if frontend_log and frontend_log.exists():
        content = frontend_log.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        for line in lines[-50:]:  # 最后50行
            print(f"[前端] {line}")
    
    print(f"\n{'='*80}")
    print(f"后端日志: {backend_log}")
    print(f"{'='*80}")
    if backend_log and backend_log.exists():
        content = backend_log.read_text(encoding='utf-8')
        lines = content.strip().split('\n')
        for line in lines[-50:]:  # 最后50行
            print(f"[后端] {line}")
