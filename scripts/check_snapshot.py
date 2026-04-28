import gzip
import json

with gzip.open('data/shared_memory_pool/snapshot_20260410_105945.json.gz', 'rt', encoding='utf-8') as f:
    data = json.load(f)

mem_zone = data.get('memory_zone', [])
print(f'Memory Zone 条目数：{len(mem_zone)}')
print()
print('前 3 条数据:')

for i, entry in enumerate(mem_zone[:3]):
    print(f'Entry {i}:')
    print(f'  Type: {entry.get("type", "N/A")}')
    print(f'  Metadata: {entry.get("metadata", {})}')
    print()
