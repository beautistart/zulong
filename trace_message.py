#!/usr/bin/env python3
"""追踪消息发送路径"""

import os
import re
from pathlib import Path

ZULONG_HOME = Path(os.environ.get('ZULONG_HOME', Path(__file__).resolve().parent))
runner_file = ZULONG_HOME / "zulong/ide/ide_fc_runner.py"
content = runner_file.read_text(encoding='utf-8')

print("=" * 80)
print("【后端】ide_fc_runner.py 中的 display_text 发送点分析")
print("=" * 80)

pattern1 = r'send_callback\("display_text"'
pattern2 = r'_send_message_safe\([^,]+,\s*"display_text"'

matches1 = list(re.finditer(pattern1, content))
matches2 = list(re.finditer(pattern2, content))

print(f"\n方法1: send_callback('display_text', ...) - {len(matches1)} 处")
for i, match in enumerate(matches1, 1):
    line_num = content[:match.start()].count('\n') + 1
    print(f"  位置 {i}: 行 {line_num}")

print(f"\n方法2: _send_message_safe(cb, 'display_text', ...) - {len(matches2)} 处")
for i, match in enumerate(matches2, 1):
    line_num = content[:match.start()].count('\n') + 1
    print(f"  位置 {i}: 行 {line_num}")

print("\n" + "=" * 80)
print("【分析结论】")
print("=" * 80)
print(f"总发送点: {len(matches1) + len(matches2)} 处")
print("\n⚠️ 可能的重复来源:")
if len(matches2) > 3:
    print(f"  - _send_message_safe 调用了 {len(matches2)} 次（流式推送）")
print("  - 检查是否存在两次调用模型的情况")
print("  - 检查重试逻辑是否重复发送")
