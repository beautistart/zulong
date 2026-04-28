# File: zulong/storage/migration.py
# 祖龙 (ZULONG) 冷热数据迁移脚本

"""
冷热数据迁移 - 自动化数据分层

功能:
1. 定时任务调度
2. 自动迁移（14 天前数据）
3. 验证与清理
4. 迁移日志

对应 TSD v2.3 第 9.1 节：分层存储策略
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import json

logger = logging.getLogger(__name__)


class DataMigrationService:
    """数据迁移服务"""
    
    def __init__(
        self,
        hot_storage=None,
        cold_storage=None,
        retention_days: int = 14,
        batch_size: int = 100
    ):
        """
        初始化迁移服务
        
        Args:
            hot_storage: 热存储（MongoDB）
            cold_storage: 冷存储（MinIO/S3）
            retention_days: 热数据保留天数
            batch_size: 批量处理大小
        """
        self.hot_storage = hot_storage
        self.cold_storage = cold_storage
        self.retention_days = retention_days
        self.batch_size = batch_size
        
        # 迁移统计
        self.stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'total_bytes_migrated': 0,
            'last_migration_time': None
        }
        
        logger.info(f"数据迁移服务已初始化 (retention_days={retention_days})")
    
    async def migrate_old_data(
        self,
        cutoff_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        迁移旧数据到冷存储
        
        Args:
            cutoff_date: 截止日期（默认 14 天前）
        
        Returns:
            迁移统计信息
        """
        if not cutoff_date:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        logger.info(f"开始迁移 {cutoff_date} 之前的数据")
        
        migration_result = {
            'start_time': datetime.utcnow().isoformat(),
            'cutoff_date': cutoff_date.isoformat(),
            'migrated_count': 0,
            'failed_count': 0,
            'bytes_migrated': 0
        }
        
        try:
            # 查询需要迁移的数据
            if self.hot_storage:
                old_logs = await self._query_old_logs(cutoff_date)
                
                if old_logs:
                    # 分批迁移
                    for i in range(0, len(old_logs), self.batch_size):
                        batch = old_logs[i:i + self.batch_size]
                        result = await self._migrate_batch(batch, cutoff_date)
                        
                        migration_result['migrated_count'] += result['success_count']
                        migration_result['failed_count'] += result['fail_count']
                        migration_result['bytes_migrated'] += result['bytes']
                        
                        logger.info(f"已迁移批次 {i//self.batch_size + 1}: {result['success_count']} 成功，{result['fail_count']} 失败")
            
            # 更新统计
            self.stats['total_migrations'] += 1
            self.stats['successful_migrations'] += migration_result['migrated_count']
            self.stats['failed_migrations'] += migration_result['failed_count']
            self.stats['total_bytes_migrated'] += migration_result['bytes_migrated']
            self.stats['last_migration_time'] = datetime.utcnow().isoformat()
            
            migration_result['end_time'] = datetime.utcnow().isoformat()
            
            logger.info(f"迁移完成：{migration_result['migrated_count']} 成功，{migration_result['failed_count']} 失败")
            
            return migration_result
            
        except Exception as e:
            logger.error(f"迁移失败：{e}")
            migration_result['error'] = str(e)
            return migration_result
    
    async def _query_old_logs(self, cutoff_date: datetime) -> List[Dict[str, Any]]:
        """
        查询旧日志
        
        Args:
            cutoff_date: 截止日期
        
        Returns:
            日志列表
        """
        if not self.hot_storage:
            return []
        
        try:
            # 从热存储查询
            query = {
                'timestamp': {'$lt': cutoff_date.isoformat()}
            }
            
            logs = await self.hot_storage.find(query, limit=1000)
            
            logger.info(f"查询到 {len(logs)} 条旧日志")
            
            return logs
            
        except Exception as e:
            logger.error(f"查询旧日志失败：{e}")
            return []
    
    async def _migrate_batch(
        self,
        logs: List[Dict[str, Any]],
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """
        批量迁移
        
        Args:
            logs: 日志列表
            cutoff_date: 截止日期
        
        Returns:
            迁移结果
        """
        result = {
            'success_count': 0,
            'fail_count': 0,
            'bytes': 0
        }
        
        try:
            # 按日期分组
            logs_by_date = {}
            for log in logs:
                date_str = log['timestamp'][:10]  # YYYY-MM-DD
                if date_str not in logs_by_date:
                    logs_by_date[date_str] = []
                logs_by_date[date_str].append(log)
            
            # 按日期归档
            for date_str, date_logs in logs_by_date.items():
                if self.cold_storage:
                    # 上传到冷存储
                    archive_path = self.cold_storage.archive_logs(date_logs, date_str)
                    
                    result['success_count'] += len(date_logs)
                    result['bytes'] += len(json.dumps(date_logs))
                    
                    logger.debug(f"已归档 {date_str} 到 {archive_path}")
                
                # 从热存储删除
                if self.hot_storage:
                    await self._delete_old_logs(date_str)
            
        except Exception as e:
            logger.error(f"批量迁移失败：{e}")
            result['fail_count'] = len(logs)
        
        return result
    
    async def _delete_old_logs(self, date_str: str):
        """
        删除旧日志
        
        Args:
            date_str: 日期字符串
        """
        if not self.hot_storage:
            return
        
        try:
            query = {'timestamp': {'$regex': f'^{date_str}'}}
            await self.hot_storage.delete_many(query)
            
            logger.info(f"已删除 {date_str} 的日志")
            
        except Exception as e:
            logger.error(f"删除旧日志失败：{e}")
    
    async def restore_data(
        self,
        date_str: str
    ) -> Dict[str, Any]:
        """
        恢复归档数据
        
        Args:
            date_str: 日期字符串
        
        Returns:
            恢复结果
        """
        logger.info(f"开始恢复 {date_str} 的数据")
        
        result = {
            'success': False,
            'restored_count': 0,
            'error': None
        }
        
        try:
            if not self.cold_storage:
                raise ValueError("冷存储未初始化")
            
            # 从冷存储下载
            logs = self.cold_storage.restore_logs(date_str)
            
            if logs:
                # 恢复到热存储
                if self.hot_storage:
                    await self.hot_storage.batch_insert(logs)
                
                result['success'] = True
                result['restored_count'] = len(logs)
                
                logger.info(f"已恢复 {len(logs)} 条日志")
            
        except Exception as e:
            logger.error(f"恢复数据失败：{e}")
            result['error'] = str(e)
        
        return result
    
    async def cleanup_old_archives(
        self,
        max_age_days: int = 365
    ) -> Dict[str, Any]:
        """
        清理过期的归档
        
        Args:
            max_age_days: 最大保留天数
        
        Returns:
            清理结果
        """
        logger.info(f"开始清理 {max_age_days} 天前的归档")
        
        result = {
            'deleted_count': 0,
            'freed_bytes': 0
        }
        
        try:
            if not self.cold_storage:
                return result
            
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            # 列出所有归档
            archives = self.cold_storage.list_archives(prefix="logs/", recursive=True)
            
            # 删除过期归档
            for archive in archives:
                # 从路径提取日期
                # 路径格式：logs/YYYY-MM-DD.json.gz
                try:
                    date_str = archive['name'].split('/')[-1].replace('.json.gz', '')
                    archive_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if archive_date < cutoff_date:
                        # 删除
                        self.cold_storage.delete_archive(archive['name'])
                        
                        result['deleted_count'] += 1
                        result['freed_bytes'] += archive['size']
                        
                        logger.info(f"已删除过期归档：{archive['name']}")
                
                except Exception as e:
                    logger.warning(f"处理归档失败：{archive['name']} - {e}")
            
            logger.info(f"清理完成：删除 {result['deleted_count']} 个归档，释放 {result['freed_bytes']} 字节")
            
        except Exception as e:
            logger.error(f"清理归档失败：{e}")
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        return {
            **self.stats,
            'retention_days': self.retention_days,
            'batch_size': self.batch_size
        }


# 定时任务调度器
class MigrationScheduler:
    """迁移任务调度器"""
    
    def __init__(
        self,
        migration_service: DataMigrationService,
        schedule_time: str = "02:00"  # 凌晨 2 点执行
    ):
        """
        初始化调度器
        
        Args:
            migration_service: 迁移服务
            schedule_time: 执行时间（HH:MM）
        """
        self.migration_service = migration_service
        self.schedule_time = schedule_time
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        logger.info(f"迁移调度器已初始化 (schedule_time={schedule_time})")
    
    async def start(self):
        """启动调度器"""
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info("迁移调度器已启动")
    
    async def stop(self):
        """停止调度器"""
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("迁移调度器已停止")
    
    async def _schedule_loop(self):
        """调度循环"""
        while self._running:
            try:
                # 计算下次执行时间
                next_run = self._calculate_next_run()
                wait_seconds = (next_run - datetime.utcnow()).total_seconds()
                
                logger.info(f"下次迁移时间：{next_run.isoformat()} ({wait_seconds:.0f}秒后)")
                
                # 等待
                await asyncio.sleep(wait_seconds)
                
                # 执行迁移
                if self._running:
                    logger.info("开始定时迁移任务")
                    await self.migration_service.migrate_old_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"调度循环错误：{e}")
                # 等待 1 分钟后重试
                await asyncio.sleep(60)
    
    def _calculate_next_run(self) -> datetime:
        """
        计算下次执行时间
        
        Returns:
            下次执行时间
        """
        now = datetime.utcnow()
        hour, minute = map(int, self.schedule_time.split(':'))
        
        # 今天的执行时间
        today_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        # 如果今天的时间已过，则明天执行
        if now >= today_run:
            return today_run + timedelta(days=1)
        else:
            return today_run


# 便捷函数
async def run_migration(
    hot_storage=None,
    cold_storage=None,
    retention_days: int = 14
):
    """
    运行迁移任务
    
    Args:
        hot_storage: 热存储
        cold_storage: 冷存储
        retention_days: 保留天数
    
    Returns:
        迁移结果
    """
    migration_service = DataMigrationService(
        hot_storage=hot_storage,
        cold_storage=cold_storage,
        retention_days=retention_days
    )
    
    return await migration_service.migrate_old_data()
