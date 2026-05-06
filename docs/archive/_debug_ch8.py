import re

TSD_PATH = r'd:\AI\project\zulong_beta4\docs\TSD_v2.4.md'

with open(TSD_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Test match
pattern = r'# 第 8 章：记忆系统架构.*?(?=# 第 9 章：数据存储架构)'
match = re.search(pattern, content, flags=re.DOTALL)
with open(r'd:\AI\project\zulong_beta4\docs\_debug.txt', 'w', encoding='utf-8') as f:
    f.write(f"match found: {match is not None}\n")
    if match:
        f.write(f"match length: {len(match.group(0))}\n")
        f.write(f"match start: {match.start()}\n")
        f.write(f"match end: {match.end()}\n")
        f.write(f"first 200 chars: {repr(match.group(0)[:200])}\n")
        f.write(f"last 200 chars: {repr(match.group(0)[-200:])}\n")
    f.write(f"file length: {len(content)}\n")

# Simple replacement test
new_ch8 = "# REPLACED_CHAPTER_8\n\nThis is a test replacement.\n\n---\n\n"
new_content = re.sub(pattern, new_ch8, content, count=1, flags=re.DOTALL)
with open(r'd:\AI\project\zulong_beta4\docs\_debug.txt', 'a', encoding='utf-8') as f:
    f.write(f"replacement happened: {content != new_content}\n")
    f.write(f"new length: {len(new_content)}\n")
