# -*- coding: utf-8 -*-
import re, sys, os

TSD_PATH = os.path.join(r'd:\AI\project\zulong_beta4\docs', 'TSD_v2.4.md')
NEW_CH8_PATH = os.path.join(r'd:\AI\project\zulong_beta4\docs', '_new_ch8.md')

with open(TSD_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

with open(NEW_CH8_PATH, 'r', encoding='utf-8') as f:
    new_ch8 = f.read()

pattern = r'# 第 8 章：记忆系统架构.*?(?=# 第 9 章：数据存储架构)'
match = re.search(pattern, content, flags=re.DOTALL)
if not match:
    sys.stdout.write("ERROR: no match\n")
    sys.exit(1)

sys.stdout.write("Match found: %d chars\n" % len(match.group(0)))

new_content = content[:match.start()] + new_ch8 + content[match.end():]
sys.stdout.write("New size: %d (was %d)\n" % (len(new_content), len(content)))

with open(TSD_PATH, 'w', encoding='utf-8') as f:
    f.write(new_content)

sys.stdout.write("DONE\n")
sys.stdout.flush()
