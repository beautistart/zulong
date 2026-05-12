# 数据存储模块快速开始指南

**版本**: v1.0  
**日期**: 2026-03-29  
**适用**: Phase 1 完成后的部署与测试

---

## 📦 1. 安装依赖

```bash
# 进入项目目录
cd d:\AI\project\zulong_beta4

# 安装 Python 依赖
pip install pymongo minio sentence-transformers jieba numpy
```

---

## 🚀 2. 启动必需服务

### 2.1 MongoDB（热存储）

**方式 1: Docker（推荐）**
```bash
docker run -d -p 27017:27017 \
  --name zulong-mongo \
  -v mongo_data:/data/db \
  mongo:latest
```

**方式 2: 本地安装**
- 下载：https://www.mongodb.com/try/download/community
- 安装后默认运行在 `localhost:27017`

**验证连接**:
```bash
mongosh --eval "db.adminCommand('ping')"
```

---

### 2.2 MinIO（冷存储）

**方式 1: Docker（推荐）**
```bash
# 创建数据目录
mkdir -p d:/data/minio

# 启动 MinIO
docker run -d -p 9000:9000 -p 9001:9001 \
  --name zulong-minio \
  -v d:/data/minio:/data \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

**访问控制台**: http://localhost:9001  
**用户名**: `minioadmin`  
**密码**: `minioadmin`

**方式 2: 使用 S3**
- 修改环境变量：
  ```bash
  set MINIO_ENDPOINT=s3.amazonaws.com
  set MINIO_ACCESS_KEY=your_access_key
  set MINIO_SECRET_KEY=your_secret_key
  ```

**验证连接**:
```bash
curl http://localhost:9000/minio/health/live
```

---

### 2.3 Qdrant（向量数据库，可选）

```bash
docker run -d -p 6333:6333 \
  --name zulong-qdrant \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

**访问控制台**: http://localhost:6333/dashboard

---

## ⚙️ 3. 配置环境变量

创建 `.env` 文件（或设置系统环境变量）：

```bash
# MongoDB 配置
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=zulong_hot

# MinIO 配置
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=zulong-cold-storage

# 经验库配置
EXPERIENCE_DB_PATH=data/experience_db
ENABLE_PERSISTENCE=true
ENABLE_SMART_TAGGING=true

# 日志配置
LOG_LEVEL=INFO
LOG_PATH=data/logs
```

---

## 🧪 4. 运行测试

### 4.1 单元测试

```bash
# 测试存储模块（Mock 模式，不需要真实服务）
python tests\test_storage_module.py

# 测试经验库
python tests\test_enhanced_experience_store.py

# 测试智能打标
python tests\test_smart_tagging.py
```

### 4.2 集成测试（需要 MongoDB 和 MinIO）

```bash
# 创建测试脚本
cat > tests\test_integration.py << 'EOF'
"""集成测试：完整数据流"""
import asyncio
from datetime import datetime
from zulong.storage.hot_storage import get_hot_storage
from zulong.storage.cold_storage import get_cold_storage
from zulong.storage.logger import get_log_collector
from zulong.storage.migration import get_data_migration

async def test_full_flow():
    # 1. 初始化
    hot = get_hot_storage(enable_ttl=False)
    cold = get_cold_storage()
    collector = get_log_collector(hot_storage=hot, cold_storage=cold)
    migration = get_data_migration(hot_storage=hot, cold_storage=cold)
    
    # 2. 启动收集器
    await collector.start()
    
    # 3. 收集日志
    for i in range(50):
        log_data = {
            'trace_id': f'integration_test_{i}',
            'timestamp': datetime.utcnow(),
            'status': 'SUCCESS',
            'user_input': {'text': f'测试问题 {i}'},
            'assistant_output': {'text': f'回复 {i}'}
        }
        collector.collect(log_data)
    
    # 4. 等待刷新
    await asyncio.sleep(3)
    
    # 5. 检查统计
    stats = collector.get_stats()
    print(f"收集器统计：{stats}")
    
    # 6. 预览迁移
    preview = await migration.run_migration(dry_run=True)
    print(f"迁移预览：{preview}")
    
    # 7. 停止收集器
    await collector.stop()
    
    print("集成测试完成！")

if __name__ == "__main__":
    asyncio.run(test_full_flow())
EOF

# 运行集成测试
python tests\test_integration.py
```

---

## 📖 5. 使用示例

### 5.1 直接使用存储模块

```python
from zulong.storage.hot_storage import get_hot_storage
from zulong.storage.cold_storage import get_cold_storage
from datetime import datetime

# 获取热存储实例
hot = get_hot_storage()

# 存储日志
log_data = {
    'trace_id': 'log_20260329_001',
    'timestamp': datetime.utcnow(),
    'status': 'SUCCESS',
    'user_input': {'text': '网络慢怎么办？'},
    'assistant_output': {'text': '建议检查路由器...'},
    'metadata': {
        'l2_status': 'IDLE',
        'power_state': 'ACTIVE',
        'duration_ms': 350,
        'tokens_used': 100
    }
}

log_id = hot.store_log(log_data)
print(f"日志已存储：{log_id}")

# 查询日志
logs = hot.query_logs(
    start_time=datetime.utcnow().replace(hour=0, minute=0),
    limit=10
)

for log in logs:
    print(f"{log['trace_id']}: {log['user_input']['text']}")

# 统计信息
stats = hot.get_statistics(time_range="24h")
print(f"24 小时统计：{stats}")
```

---

### 5.2 使用日志收集器（推荐用于 L1-B）

```python
import asyncio
from zulong.storage.logger import get_log_collector

async def main():
    # 获取收集器实例
    collector = get_log_collector(
        batch_size=100,
        flush_interval_seconds=5.0
    )
    
    # 启动
    await collector.start()
    
    # 在 L1-B 核心循环中调用
    for i in range(1000):
        log_data = {
            'trace_id': f'log_{i}',
            'timestamp': datetime.utcnow(),
            'status': 'SUCCESS',
            'user_input': {'text': f'问题 {i}'},
            'assistant_output': {'text': f'回复 {i}'}
        }
        
        # 同步调用（非阻塞）
        collector.collect(log_data)
    
    # 等待后台刷新
    await asyncio.sleep(10)
    
    # 查看统计
    stats = collector.get_stats()
    print(f"统计：{stats}")
    
    # 停止
    await collector.stop()

asyncio.run(main())
```

---

### 5.3 数据迁移

```python
import asyncio
from zulong.storage.migration import get_data_migration

async def migrate_old_logs():
    # 获取迁移实例
    migration = get_data_migration(
        migration_age_days=14,  # 迁移 14 天前的数据
        batch_size=1000
    )
    
    # 预览
    preview = await migration.run_migration(dry_run=True)
    print(f"预览：{preview}")
    
    # 实际迁移
    result = await migration.run_migration()
    print(f"迁移结果：{result}")
    
    # 如果需要回滚
    # rollback = await migration.rollback_migration(
    #     migration_batch=result['batch_id'],
    #     restore_to_hot=True
    # )

asyncio.run(migrate_old_logs())
```

---

### 5.4 集成到 L1-B 核心循环

```python
# zulong/l1b/core.py

from zulong.storage.logger import get_log_collector

class L1BCore:
    def __init__(self):
        # 初始化日志收集器
        self.log_collector = get_log_collector()
        
    async def start(self):
        # 启动收集器
        await self.log_collector.start()
        
        # ... 其他初始化
        
    async def process_event(self, event):
        # 处理事件
        
        # 记录日志
        log_data = {
            'trace_id': event.trace_id,
            'timestamp': datetime.utcnow(),
            'status': 'SUCCESS' if success else 'FAILED',
            'user_input': event.payload,
            'assistant_output': response,
            'metadata': {
                'l2_status': self.l2_status,
                'power_state': self.power_state,
                'duration_ms': duration,
                'tokens_used': tokens
            }
        }
        
        # 异步收集（非阻塞）
        self.log_collector.collect(log_data)
        
    async def stop(self):
        # 停止收集器
        await self.log_collector.stop()
        
        # ... 其他清理
```

---

## 🔍 6. 监控与调试

### 6.1 查看 MongoDB 数据

```bash
# 连接到 MongoDB
mongosh

# 使用数据库
use zulong_hot

# 查看日志集合
db.logs.find().limit(10)

# 统计日志数量
db.logs.countDocuments()

# 查看最近 1 小时的日志
db.logs.find({
    timestamp: {
        $gte: new Date(Date.now() - 3600000)
    }
})

# 聚合分析
db.logs.aggregate([
    {
        $match: {
            timestamp: {
                $gte: new Date(Date.now() - 86400000)
            }
        }
    },
    {
        $group: {
            _id: "$status",
            count: { $sum: 1 },
            avgDuration: { $avg: "$metadata.duration_ms" }
        }
    }
])
```

---

### 6.2 查看 MinIO 数据

**方式 1: Web 控制台**
1. 访问 http://localhost:9001
2. 登录：`minioadmin` / `minioadmin`
3. 浏览 `zulong-cold-storage` 桶

**方式 2: 命令行工具**
```bash
# 安装 mc 客户端
# Windows: choco install minio-client

# 添加别名
mc alias set myminio http://localhost:9000 minioadmin minioadmin

# 列出桶
mc ls myminio

# 列出对象
mc ls --recursive myminio/zulong-cold-storage

# 下载对象
mc cp myminio/zulong-cold-storage/logs/2026/03/29/archive.json.gz ./downloaded.json.gz
```

---

### 6.3 查看收集器状态

```python
from zulong.storage.logger import get_log_collector

collector = get_log_collector()
stats = collector.get_stats()

print(f"""
日志收集器状态:
- 总接收：{stats['total_received']}
- 总刷新：{stats['total_flushed']}
- 总失败：{stats['total_failed']}
- 队列大小：{stats['queue_size']}
- 队列使用率：{stats['queue_usage_percent']:.2f}%
- 最后刷新：{stats['last_flush_time']}
- 最后刷新大小：{stats['last_flush_size']}
""")
```

---

## 🐛 7. 常见问题

### Q1: MongoDB 连接失败

**错误**: `ServerSelectionTimeoutError: localhost:27017: [WinError 10061]`

**解决**:
```bash
# 检查 MongoDB 是否运行
docker ps | grep mongo

# 如果没有运行，启动
docker start zulong-mongo

# 或者重新启动
docker restart zulong-mongo
```

---

### Q2: MinIO 连接失败

**错误**: `Connection refused` 或 `Access denied`

**解决**:
```bash
# 检查 MinIO 是否运行
docker ps | grep minio

# 查看日志
docker logs zulong-minio

# 重启 MinIO
docker restart zulong-minio
```

---

### Q3: 收集器队列堆积

**现象**: `queue_usage_percent > 80%`

**解决**:
1. 增加批量大小
   ```python
   collector = get_log_collector(batch_size=200)  # 默认 100
   ```

2. 缩短刷新间隔
   ```python
   collector = get_log_collector(flush_interval_seconds=2.0)  # 默认 5s
   ```

3. 检查 MongoDB 性能
   ```bash
   mongosh --eval "db.logs.getIndexes()"
   ```

---

### Q4: 迁移失败

**错误**: `Migration failed: No logs to migrate`

**解决**:
- 确认热存储中有足够老的日志（> 14 天）
- 或者使用强制模式：
  ```python
  result = await migration.run_migration(force=True)
  ```

---

### Q5: 持久化文件未清理

**现象**: `data/logs/pending` 目录下有大量文件

**解决**:
```python
# 手动清理
from pathlib import Path
import shutil

pending_dir = Path("data/logs/pending")
if pending_dir.exists():
    shutil.rmtree(pending_dir)
    print("已清理持久化目录")
```

---

## 📊 8. 性能优化建议

### 8.1 MongoDB 优化

**创建索引**:
```javascript
// 时间索引（加速时间范围查询）
db.logs.createIndex({ timestamp: -1 })

// 复合索引（加速状态 + 时间查询）
db.logs.createIndex({ status: 1, timestamp: -1 })

// TTL 索引（自动清理 14 天前数据）
db.logs.createIndex({ timestamp: 1 }, { expireAfterSeconds: 1209600 })
```

---

### 8.2 批量大小调优

根据负载调整批量参数：

| 场景 | batch_size | flush_interval |
|------|-----------|----------------|
| 低负载 (< 100 条/分钟) | 50 | 10s |
| 中负载 (100-500 条/分钟) | 100 | 5s |
| 高负载 (> 500 条/分钟) | 200 | 2s |

---

### 8.3 压缩策略

```python
# 高压缩比（节省空间，增加 CPU）
cold.create_archive(logs, archive_name, compress=True)

# 低延迟（不压缩，快速存取）
cold.create_archive(logs, archive_name, compress=False)
```

---

## 📝 9. 下一步

完成 Phase 1 后，继续：

1. **Phase 2**: 复盘机制
   - 三重触发机制
   - 成功经验提炼
   - 失败案例分析
   - 防重复机制

2. **Phase 3**: 时间标签与优化
   - 三维时间标签
   - 冷热数据管理
   - 降智回滚机制

---

## 📞 10. 获取帮助

**文档**:
- [`TSD_v2.3.md`](file://d:\AI\project\zulong_beta4\docs\TSD_v2.3.md) - 技术规格说明书
- [`PHASE1_COMPLETION_SUMMARY.md`](file://d:\AI\project\zulong_beta4\docs\PHASE1_COMPLETION_SUMMARY.md) - Phase 1 总结

**测试**:
- `tests/test_storage_module.py` - 存储模块测试
- `tests/test_enhanced_experience_store.py` - 经验库测试

**代码**:
- `zulong/storage/hot_storage.py` - MongoDB 热存储
- `zulong/storage/cold_storage.py` - MinIO 冷存储
- `zulong/storage/logger.py` - 日志收集器
- `zulong/storage/migration.py` - 数据迁移

---

**祝您使用愉快！🎉**

如有问题，请查看文档或联系架构团队。
