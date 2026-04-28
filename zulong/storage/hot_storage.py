# 数据存储模块：MongoDB 热存储
"""
功能:
- MongoDB 连接管理
- TTL 索引配置 (14 天自动清理)
- 日志存储与查询
- 聚合分析

对应 TSD v2.3 第 9.1 节
"""

from pymongo import MongoClient, ASCENDING, DESCENDING
import gzip
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class HotStorage:
    """热存储：MongoDB（最近 14 天数据）"""
    
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017",
                 db_name: str = "zulong_hot",
                 enable_ttl: bool = True,
                 ttl_days: int = 14):
        """初始化热存储
        
        Args:
            mongo_uri: MongoDB 连接 URI
            db_name: 数据库名称
            enable_ttl: 是否启用 TTL 自动清理
            ttl_days: TTL 天数（默认 14 天）
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.enable_ttl = enable_ttl
        self.ttl_days = ttl_days
        
        self.client: Optional[MongoClient] = None
        self.db = None
        
        # 集合引用
        self.logs = None  # 日志集合
        self.sessions = None  # 会话集合
        
        self._connect()
        
        if enable_ttl:
            self._create_ttl_indexes()
        
        logger.info(f"[HotStorage] 初始化完成：{mongo_uri}/{db_name}")
    
    def _connect(self):
        """连接到 MongoDB"""
        try:
            self.client = MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=45000,
                connectTimeoutMS=20000,
                retryWrites=True,
                w="majority"
            )
            
            # 测试连接
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            
            # 初始化集合
            self.logs = self.db['logs']
            self.sessions = self.db['sessions']
            
            # 创建索引
            self._create_indexes()
            
            logger.info(f"[HotStorage] MongoDB 连接成功")
            
        except ConnectionFailure as e:
            logger.error(f"[HotStorage] 连接失败：{e}")
            raise
        except OperationFailure as e:
            logger.error(f"[HotStorage] 操作失败：{e}")
            raise
    
    def _create_indexes(self):
        """创建基础索引"""
        # logs 集合索引
        self.logs.create_index([("timestamp", DESCENDING)])
        self.logs.create_index([("trace_id", ASCENDING)])
        self.logs.create_index([("status", ASCENDING)])
        self.logs.create_index([("user_input.text", ASCENDING)])
        self.logs.create_index([("execution_steps.agent", ASCENDING)])
        
        # 复合索引（常用查询）
        self.logs.create_index([
            ("timestamp", DESCENDING),
            ("status", ASCENDING)
        ])
        
        self.logs.create_index([
            ("user_input.text", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        
        # sessions 集合索引
        self.sessions.create_index([("session_id", ASCENDING)], unique=True)
        self.sessions.create_index([("last_activity", DESCENDING)])
        
        logger.info(f"[HotStorage] 索引创建完成")
    
    def _create_ttl_indexes(self):
        """创建 TTL 索引（自动清理）"""
        try:
            # logs 集合：14 天后自动删除
            self.logs.create_index(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=self.ttl_days * 24 * 3600,
                name="logs_ttl_index"
            )
            
            # sessions 集合：30 天未活动自动删除
            self.sessions.create_index(
                [("last_activity", ASCENDING)],
                expireAfterSeconds=30 * 24 * 3600,
                name="sessions_ttl_index"
            )
            
            logger.info(f"[HotStorage] TTL 索引创建完成（{self.ttl_days}天）")
            
        except OperationFailure as e:
            logger.warning(f"[HotStorage] TTL 索引创建失败：{e}")
            logger.warning(f"[HotStorage] 将使用手动清理策略")
    
    def store_log(self, log_data: Dict[str, Any]) -> str:
        """存储日志
        
        Args:
            log_data: 日志数据（符合 TSD 2.2 格式）
            
        Returns:
            str: 日志 ID
        """
        try:
            # 添加元数据
            log_data['_id'] = log_data.get('trace_id', f"log_{datetime.utcnow().isoformat()}")
            log_data['timestamp'] = log_data.get('timestamp', datetime.utcnow())
            log_data['stored_at'] = datetime.utcnow()
            
            # 插入数据库
            result = self.logs.insert_one(log_data)
            
            logger.debug(f"[HotStorage] 日志已存储：{log_data['_id']}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"[HotStorage] 存储日志失败：{e}")
            raise
    
    def store_logs_batch(self, logs: List[Dict[str, Any]]) -> int:
        """批量存储日志
        
        Args:
            logs: 日志列表
            
        Returns:
            int: 成功存储的数量
        """
        try:
            if not logs:
                return 0
            
            # 添加元数据
            for log in logs:
                log['timestamp'] = log.get('timestamp', datetime.utcnow())
                log['stored_at'] = datetime.utcnow()
            
            # 批量插入
            result = self.logs.insert_many(logs, ordered=False)
            
            logger.info(f"[HotStorage] 批量存储 {len(result.inserted_ids)} 条日志")
            
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"[HotStorage] 批量存储失败：{e}")
            return 0
    
    def get_log(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """获取单条日志
        
        Args:
            trace_id: 追踪 ID
            
        Returns:
            Optional[Dict]: 日志数据，不存在返回 None
        """
        try:
            log = self.logs.find_one({"trace_id": trace_id})
            
            if log:
                # 转换 ObjectId
                log['_id'] = str(log['_id'])
                return log
            
            return None
            
        except Exception as e:
            logger.error(f"[HotStorage] 查询日志失败：{e}")
            return None
    
    def query_logs(self,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   status: Optional[str] = None,
                   agent: Optional[str] = None,
                   keyword: Optional[str] = None,
                   limit: int = 100,
                   skip: int = 0) -> List[Dict[str, Any]]:
        """查询日志（支持多条件）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            status: 状态（SUCCESS/FAILED）
            agent: Agent 名称
            keyword: 关键词（搜索 user_input.text）
            limit: 返回数量限制
            skip: 跳过数量
            
        Returns:
            List[Dict]: 日志列表
        """
        try:
            # 构建查询条件
            query = {}
            
            if start_time:
                query['timestamp'] = {'$gte': start_time}
            
            if end_time:
                if 'timestamp' in query:
                    query['timestamp']['$lte'] = end_time
                else:
                    query['timestamp'] = {'$lte': end_time}
            
            if status:
                query['status'] = status
            
            if agent:
                query['execution_steps.agent'] = agent
            
            if keyword:
                query['user_input.text'] = {'$regex': keyword, '$options': 'i'}
            
            # 执行查询
            cursor = self.logs.find(query).sort('timestamp', DESCENDING).skip(skip).limit(limit)
            
            logs = []
            for log in cursor:
                log['_id'] = str(log['_id'])
                logs.append(log)
            
            logger.debug(f"[HotStorage] 查询到 {len(logs)} 条日志")
            
            return logs
            
        except Exception as e:
            logger.error(f"[HotStorage] 查询失败：{e}")
            return []
    
    def get_statistics(self, 
                       time_range: str = "24h") -> Dict[str, Any]:
        """获取统计信息
        
        Args:
            time_range: 时间范围（"1h"/"24h"/"7d"/"14d"）
            
        Returns:
            Dict: 统计信息
        """
        try:
            # 计算开始时间
            now = datetime.utcnow()
            
            if time_range == "1h":
                start_time = now - timedelta(hours=1)
            elif time_range == "24h":
                start_time = now - timedelta(days=1)
            elif time_range == "7d":
                start_time = now - timedelta(days=7)
            elif time_range == "14d":
                start_time = now - timedelta(days=14)
            else:
                start_time = now - timedelta(days=1)
            
            # 聚合查询
            pipeline = [
                {'$match': {'timestamp': {'$gte': start_time}}},
                {'$group': {
                    '_id': '$status',
                    'count': {'$sum': 1},
                    'total_time': {'$sum': '$cost_stats.total_time_ms'},
                    'total_tokens': {'$sum': '$cost_stats.token_usage'}
                }}
            ]
            
            result = list(self.logs.aggregate(pipeline))
            
            stats = {
                'time_range': time_range,
                'total_logs': 0,
                'success_count': 0,
                'failed_count': 0,
                'avg_time_ms': 0,
                'total_tokens': 0
            }
            
            for item in result:
                status = item['_id']
                count = item['count']
                
                stats['total_logs'] += count
                
                if status == 'SUCCESS':
                    stats['success_count'] = count
                elif status == 'FAILED':
                    stats['failed_count'] = count
                
                stats['total_tokens'] += item.get('total_tokens', 0)
                
                if 'total_time' in item:
                    stats['avg_time_ms'] += item['total_time']
            
            # 计算平均时间
            if stats['total_logs'] > 0:
                stats['avg_time_ms'] = stats['avg_time_ms'] / stats['total_logs']
            
            logger.debug(f"[HotStorage] 统计信息：{stats}")
            
            return stats
            
        except Exception as e:
            logger.error(f"[HotStorage] 统计失败：{e}")
            return {}
    
    def cleanup_old_logs(self, 
                         older_than_days: int = 14,
                         dry_run: bool = False) -> int:
        """清理旧日志（手动清理策略）
        
        Args:
            older_than_days: 清理多少天之前的日志
            dry_run: 是否只是预览（不真正删除）
            
        Returns:
            int: 清理的日志数量
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=older_than_days)
            
            if dry_run:
                # 预览
                count = self.logs.count_documents({'timestamp': {'$lt': cutoff_time}})
                logger.info(f"[HotStorage] 预览：将清理 {count} 条日志")
                return count
            else:
                # 实际删除
                result = self.logs.delete_many({'timestamp': {'$lt': cutoff_time}})
                logger.info(f"[HotStorage] 已清理 {result.deleted_count} 条日志")
                return result.deleted_count
                
        except Exception as e:
            logger.error(f"[HotStorage] 清理失败：{e}")
            return 0
    
    def export_logs(self, 
                    start_time: datetime,
                    end_time: datetime,
                    output_path: str,
                    compress: bool = True) -> str:
        """导出日志到文件（用于迁移到冷存储）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            output_path: 输出文件路径
            compress: 是否压缩
            
        Returns:
            str: 输出文件路径
        """
        import gzip
        
        try:
            # 查询日志
            logs = self.query_logs(
                start_time=start_time,
                end_time=end_time,
                limit=10000  # 最多 1 万条
            )
            
            if not logs:
                logger.warning(f"[HotStorage] 没有可导出的日志")
                return ""
            
            # 转换为 JSON
            json_str = json.dumps(logs, ensure_ascii=False, indent=2)
            
            # 写入文件
            if compress:
                output_path += '.gz'
                with gzip.open(output_path, 'wt', encoding='utf-8') as f:
                    f.write(json_str)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            logger.info(f"[HotStorage] 已导出 {len(logs)} 条日志到 {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"[HotStorage] 导出失败：{e}")
            return ""
    
    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info(f"[HotStorage] 连接已关闭")


# 全局单例
_hot_storage_instance = None


def get_hot_storage(mongo_uri: str = None,
                    db_name: str = "zulong_hot",
                    enable_ttl: bool = True) -> HotStorage:
    """获取热存储单例
    
    Args:
        mongo_uri: MongoDB URI（可选，默认从环境变量读取）
        db_name: 数据库名称
        enable_ttl: 是否启用 TTL
        
    Returns:
        HotStorage: 单例实例
    """
    global _hot_storage_instance
    
    if _hot_storage_instance is None:
        # 从环境变量读取配置
        if mongo_uri is None:
            import os
            mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        
        _hot_storage_instance = HotStorage(
            mongo_uri=mongo_uri,
            db_name=db_name,
            enable_ttl=enable_ttl
        )
    
    return _hot_storage_instance
