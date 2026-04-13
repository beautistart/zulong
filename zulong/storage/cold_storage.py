# File: zulong/storage/cold_storage.py
# 祖龙 (ZULONG) 冷存储模块（MinIO/S3）

"""
冷存储模块 - 基于对象存储的长期归档

功能:
1. MinIO/S3 对象存储连接
2. 数据压缩（.json.gz）
3. 归档管理
4. 下载恢复

对应 TSD v2.3 第 9.1 节：分层存储策略
"""

import io
import json
import gzip
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from minio import Minio
    from minio.error import S3Error
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logging.warning("MinIO 未安装，请运行：pip install minio")

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logging.warning("Boto3 未安装，请运行：pip install boto3")

logger = logging.getLogger(__name__)


class ColdStorage:
    """冷存储管理器（支持 MinIO 和 S3）"""
    
    def __init__(
        self,
        storage_type: str = "minio",  # "minio" 或 "s3"
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket_name: str = "zulong-cold-storage",
        region: str = "us-east-1",
        secure: bool = False
    ):
        """
        初始化冷存储
        
        Args:
            storage_type: 存储类型 ("minio" 或 "s3")
            endpoint: 服务端点（MinIO 必需，S3 可选）
            access_key: 访问密钥
            secret_key: 密钥
            bucket_name: 存储桶名称
            region: 区域（S3 使用）
            secure: 是否使用 HTTPS
        """
        self.storage_type = storage_type
        self.bucket_name = bucket_name
        self.region = region
        self.secure = secure
        
        # 初始化客户端
        if storage_type == "minio":
            if not MINIO_AVAILABLE:
                raise ImportError("MinIO 库未安装")
            self.client = self._init_minio(endpoint, access_key, secret_key)
        elif storage_type == "s3":
            if not BOTO3_AVAILABLE:
                raise ImportError("Boto3 库未安装")
            self.client = self._init_s3(endpoint, access_key, secret_key, region)
        else:
            raise ValueError(f"不支持的存储类型：{storage_type}")
        
        # 确保 bucket 存在
        self._ensure_bucket()
        
        logger.info(f"冷存储已初始化：{storage_type}://{bucket_name}")
    
    def _init_minio(self, endpoint: str, access_key: str, secret_key: str) -> Minio:
        """初始化 MinIO 客户端"""
        if not endpoint:
            raise ValueError("MinIO endpoint 必需")
        
        return Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=self.secure
        )
    
    def _init_s3(self, endpoint: Optional[str], access_key: Optional[str], 
                 secret_key: Optional[str], region: str):
        """初始化 S3 客户端"""
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        if endpoint:
            return session.client('s3', endpoint_url=endpoint)
        else:
            return session.client('s3')
    
    def _ensure_bucket(self):
        """确保存储桶存在"""
        try:
            if self.storage_type == "minio":
                if not self.client.bucket_exists(self.bucket_name):
                    self.client.make_bucket(self.bucket_name)
            else:  # S3
                # S3 通常自动创建 bucket
                pass
        except Exception as e:
            logger.warning(f"创建存储桶失败：{e}")
    
    def compress_data(self, data: Dict[str, Any]) -> bytes:
        """
        压缩数据为 .json.gz 格式
        
        Args:
            data: 要压缩的字典数据
        
        Returns:
            压缩后的二进制数据
        """
        json_str = json.dumps(data, ensure_ascii=False, default=str)
        json_bytes = json_str.encode('utf-8')
        
        buffer = io.BytesIO()
        with gzip.open(buffer, 'wb') as f:
            f.write(json_bytes)
        
        return buffer.getvalue()
    
    def decompress_data(self, compressed_data: bytes) -> Dict[str, Any]:
        """
        解压 .json.gz 数据
        
        Args:
            compressed_data: 压缩的二进制数据
        
        Returns:
            解压后的字典数据
        """
        buffer = io.BytesIO(compressed_data)
        with gzip.open(buffer, 'rt', encoding='utf-8') as f:
            json_str = f.read()
        
        return json.loads(json_str)
    
    def upload_archive(
        self,
        data: Dict[str, Any],
        object_name: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        上传归档数据
        
        Args:
            data: 要归档的数据
            object_name: 对象名称（包含路径）
            metadata: 元数据
        
        Returns:
            对象路径
        """
        try:
            # 压缩数据
            compressed = self.compress_data(data)
            
            # 添加元数据
            if metadata is None:
                metadata = {}
            metadata['upload_time'] = datetime.utcnow().isoformat()
            metadata['original_size'] = str(len(json.dumps(data)))
            metadata['compressed_size'] = str(len(compressed))
            
            # 上传
            if self.storage_type == "minio":
                self.client.put_object(
                    self.bucket_name,
                    object_name,
                    io.BytesIO(compressed),
                    len(compressed),
                    content_type='application/gzip',
                    metadata=metadata
                )
            else:  # S3
                self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_name,
                    Body=compressed,
                    ContentType='application/gzip',
                    Metadata=metadata
                )
            
            logger.info(f"归档已上传：{object_name}")
            return object_name
            
        except Exception as e:
            logger.error(f"上传归档失败：{e}")
            raise
    
    def download_archive(self, object_name: str) -> Dict[str, Any]:
        """
        下载归档数据
        
        Args:
            object_name: 对象名称
        
        Returns:
            解压后的数据
        """
        try:
            # 下载
            if self.storage_type == "minio":
                response = self.client.get_object(self.bucket_name, object_name)
                compressed = response.read()
                response.close()
                response.release_conn()
            else:  # S3
                response = self.client.get_object(
                    Bucket=self.bucket_name,
                    Key=object_name
                )
                compressed = response['Body'].read()
            
            # 解压
            return self.decompress_data(compressed)
            
        except S3Error as e:
            logger.error(f"下载归档失败：{e}")
            raise
    
    def list_archives(
        self,
        prefix: str = "",
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        列出归档列表
        
        Args:
            prefix: 路径前缀
            recursive: 是否递归列出
        
        Returns:
            归档信息列表
        """
        archives = []
        
        try:
            if self.storage_type == "minio":
                objects = self.client.list_objects(
                    self.bucket_name,
                    prefix=prefix,
                    recursive=recursive
                )
                for obj in objects:
                    archives.append({
                        'name': obj.object_name,
                        'size': obj.size,
                        'last_modified': obj.last_modified,
                        'etag': obj.etag
                    })
            else:  # S3
                response = self.client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                if 'Contents' in response:
                    for obj in response['Contents']:
                        archives.append({
                            'name': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag']
                        })
        except Exception as e:
            logger.error(f"列出归档失败：{e}")
        
        return archives
    
    def delete_archive(self, object_name: str) -> bool:
        """
        删除归档
        
        Args:
            object_name: 对象名称
        
        Returns:
            是否成功删除
        """
        try:
            if self.storage_type == "minio":
                self.client.remove_object(self.bucket_name, object_name)
            else:  # S3
                self.client.delete_object(
                    Bucket=self.bucket_name,
                    Key=object_name
                )
            
            logger.info(f"归档已删除：{object_name}")
            return True
            
        except Exception as e:
            logger.error(f"删除归档失败：{e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息
        
        Returns:
            统计信息
        """
        stats = {
            'storage_type': self.storage_type,
            'bucket': self.bucket_name,
            'total_objects': 0,
            'total_size': 0,
            'compression_ratio': 0.0
        }
        
        try:
            archives = self.list_archives(recursive=True)
            stats['total_objects'] = len(archives)
            stats['total_size'] = sum(arch['size'] for arch in archives)
            
            # 估算压缩比（需要元数据）
            # 简化处理：假设压缩比为 5:1
            stats['compression_ratio'] = 5.0
            
        except Exception as e:
            logger.error(f"获取统计信息失败：{e}")
        
        return stats
    
    def archive_logs(
        self,
        logs: List[Dict[str, Any]],
        date_str: str
    ) -> str:
        """
        归档日志数据
        
        Args:
            logs: 日志列表
            date_str: 日期字符串（格式：YYYY-MM-DD）
        
        Returns:
            归档路径
        """
        # 创建归档数据
        archive_data = {
            'date': date_str,
            'count': len(logs),
            'logs': logs,
            'archive_time': datetime.utcnow().isoformat()
        }
        
        # 生成对象名称
        object_name = f"logs/{date_str}.json.gz"
        
        # 上传
        return self.upload_archive(archive_data, object_name)
    
    def restore_logs(self, date_str: str) -> List[Dict[str, Any]]:
        """
        恢复归档的日志
        
        Args:
            date_str: 日期字符串
        
        Returns:
            日志列表
        """
        object_name = f"logs/{date_str}.json.gz"
        archive_data = self.download_archive(object_name)
        
        return archive_data.get('logs', [])


# 单例模式
_cold_storage_instance: Optional[ColdStorage] = None


def get_cold_storage(
    storage_type: str = "minio",
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    bucket_name: str = "zulong-cold-storage",
    **kwargs
) -> ColdStorage:
    """
    获取冷存储单例
    
    Args:
        storage_type: 存储类型
        endpoint: 服务端点
        access_key: 访问密钥
        secret_key: 密钥
        bucket_name: 存储桶名称
        **kwargs: 其他参数
    
    Returns:
        ColdStorage 实例
    """
    global _cold_storage_instance
    
    if _cold_storage_instance is None:
        _cold_storage_instance = ColdStorage(
            storage_type=storage_type,
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            **kwargs
        )
    
    return _cold_storage_instance
