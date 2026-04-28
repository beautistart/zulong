# -*- coding: utf-8 -*-
# 修复 2: 优化向量缓存参数

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           修复 2: 优化向量缓存参数")
print("=" * 80)
print()

# 读取原文件
file_path = os.path.join(os.path.dirname(__file__), '..', 'zulong', 'memory', 'vector_cache.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 1: 优化缓存参数
old_code = '''        self.embedding_manager = embedding_manager
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.time_decay_lambda = time_decay_lambda'''

new_code = '''        self.embedding_manager = embedding_manager
        # 🔥 优化：增加缓存容量，延长 TTL，减缓衰减
        self.max_cache_size = max_cache_size if max_cache_size else 100  # 从 50 增加到 100
        self.ttl_seconds = ttl_seconds if ttl_seconds else 7200  # 从 3600 增加到 7200 (2 小时)
        self.time_decay_lambda = time_decay_lambda if time_decay_lambda else 0.0005  # 从 0.001 减少到 0.0005 (半衰期~20 分钟)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ 修复 1: 缓存参数已优化")
    print("   - max_cache_size: 50 → 100")
    print("   - ttl_seconds: 3600 → 7200")
    print("   - time_decay_lambda: 0.001 → 0.0005")
else:
    print("⚠️  未找到目标代码，可能已修复或版本不匹配")

# 保存修复后的文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print()
print("=" * 80)
print("✅ 修复完成！请重启系统以应用更改")
print("=" * 80)
