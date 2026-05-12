"""
AT-06: 事件档案序列化
定义.dossier文件格式，将多层级日志打包归档
关键指标: 支持压缩存储，包含完整的元数据（时间、任务ID、结果）
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
import json
import gzip
import os
from datetime import datetime

from .clock_synchronizer import get_unified_timestamp, ClockSynchronizer
from .ring_buffer import RingBufferSlot, MultiLayerRingBuffer

logger = logging.getLogger(__name__)


class DossierStatus(Enum):
    CREATED = "created"
    FROZEN = "frozen"
    ARCHIVED = "archived"
    LOADED = "loaded"


@dataclass
class DossierMetadata:
    """档案元数据"""
    dossier_id: str
    event_id: str
    task_id: str
    created_at: float
    frozen_at: Optional[float] = None
    status: DossierStatus = DossierStatus.CREATED
    layer_count: Dict[str, int] = field(default_factory=dict)
    total_slots: int = 0
    compressed: bool = False
    file_size_bytes: int = 0
    checksum: str = ""
    result: str = "unknown"
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dossier_id": self.dossier_id,
            "event_id": self.event_id,
            "task_id": self.task_id,
            "created_at": self.created_at,
            "frozen_at": self.frozen_at,
            "status": self.status.value,
            "layer_count": self.layer_count,
            "total_slots": self.total_slots,
            "compressed": self.compressed,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "result": self.result,
            "error_message": self.error_message
        }


@dataclass
class Dossier:
    """事件档案"""
    metadata: DossierMetadata
    slots: List[RingBufferSlot] = field(default_factory=list)
    l2_snapshot: Optional[Dict[str, Any]] = None
    l3_snapshot: Optional[Dict[str, Any]] = None
    analysis_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "slots": [slot.to_dict() for slot in self.slots],
            "l2_snapshot": self.l2_snapshot,
            "l3_snapshot": self.l3_snapshot,
            "analysis_result": self.analysis_result
        }


class DossierSerializer:
    """
    事件档案序列化器
    
    负责将环形缓冲区数据打包为.dossier文件
    支持压缩存储、元数据管理、档案加载
    """
    
    DOSSIER_VERSION = "1.0"
    FILE_EXTENSION = ".dossier"
    
    def __init__(self, storage_path: str = "dossiers"):
        """
        初始化序列化器
        
        Args:
            storage_path: 档案存储路径
        """
        self.storage_path = storage_path
        self._clock = ClockSynchronizer()
        self._dossier_counter = 0
        
        os.makedirs(storage_path, exist_ok=True)
        
        self._stats = {
            "dossiers_created": 0,
            "dossiers_saved": 0,
            "dossiers_loaded": 0,
            "total_bytes_written": 0
        }
        
        logger.info(f"[DossierSerializer] 初始化完成，存储路径: {storage_path}")
    
    def generate_dossier_id(self) -> str:
        """生成档案ID"""
        self._dossier_counter += 1
        return f"DOSSIER_{get_unified_timestamp():.0f}_{self._dossier_counter}"
    
    def create_dossier(
        self,
        event_id: str,
        task_id: str,
        slots: List[RingBufferSlot],
        result: str = "unknown",
        error_message: Optional[str] = None
    ) -> Dossier:
        """
        创建档案
        
        Args:
            event_id: 关联的事件ID
            task_id: 任务ID
            slots: 槽位数据列表
            result: 结果状态
            error_message: 错误信息
        
        Returns:
            Dossier: 创建的档案对象
        """
        dossier_id = self.generate_dossier_id()
        
        layer_count = {}
        for slot in slots:
            if slot.layer not in layer_count:
                layer_count[slot.layer] = 0
            layer_count[slot.layer] += 1
        
        metadata = DossierMetadata(
            dossier_id=dossier_id,
            event_id=event_id,
            task_id=task_id,
            created_at=get_unified_timestamp(),
            status=DossierStatus.CREATED,
            layer_count=layer_count,
            total_slots=len(slots),
            result=result,
            error_message=error_message
        )
        
        dossier = Dossier(
            metadata=metadata,
            slots=slots
        )
        
        self._stats["dossiers_created"] += 1
        logger.info(f"[DossierSerializer] 档案创建: {dossier_id}, 槽位数: {len(slots)}")
        
        return dossier
    
    def save_dossier(
        self,
        dossier: Dossier,
        compress: bool = True,
        include_snapshots: bool = True
    ) -> str:
        """
        保存档案到文件
        
        Args:
            dossier: 档案对象
            compress: 是否压缩
            include_snapshots: 是否包含快照
        
        Returns:
            str: 文件路径
        """
        dossier.metadata.frozen_at = get_unified_timestamp()
        dossier.metadata.status = DossierStatus.FROZEN
        dossier.metadata.compressed = compress
        
        file_name = f"{dossier.metadata.dossier_id}{self.FILE_EXTENSION}"
        if compress:
            file_name += ".gz"
        
        file_path = os.path.join(self.storage_path, file_name)
        
        data = dossier.to_dict()
        data["version"] = self.DOSSIER_VERSION
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        try:
            if compress:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(json_str)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
            
            file_size = os.path.getsize(file_path)
            dossier.metadata.file_size_bytes = file_size
            dossier.metadata.checksum = self._calculate_checksum(json_str)
            
            self._stats["dossiers_saved"] += 1
            self._stats["total_bytes_written"] += file_size
            
            logger.info(f"[DossierSerializer] 档案保存: {file_path}, 大小: {file_size} bytes")
            return file_path
            
        except Exception as e:
            logger.error(f"[DossierSerializer] 档案保存失败: {e}")
            raise
    
    def load_dossier(self, file_path: str) -> Dossier:
        """
        从文件加载档案
        
        Args:
            file_path: 文件路径
        
        Returns:
            Dossier: 档案对象
        """
        try:
            if file_path.endswith('.gz'):
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            metadata = DossierMetadata(
                dossier_id=data["metadata"]["dossier_id"],
                event_id=data["metadata"]["event_id"],
                task_id=data["metadata"]["task_id"],
                created_at=data["metadata"]["created_at"],
                frozen_at=data["metadata"].get("frozen_at"),
                status=DossierStatus(data["metadata"]["status"]),
                layer_count=data["metadata"].get("layer_count", {}),
                total_slots=data["metadata"].get("total_slots", 0),
                compressed=data["metadata"].get("compressed", False),
                file_size_bytes=data["metadata"].get("file_size_bytes", 0),
                checksum=data["metadata"].get("checksum", ""),
                result=data["metadata"].get("result", "unknown"),
                error_message=data["metadata"].get("error_message")
            )
            
            slots = [RingBufferSlot.from_dict(s) for s in data.get("slots", [])]
            
            dossier = Dossier(
                metadata=metadata,
                slots=slots,
                l2_snapshot=data.get("l2_snapshot"),
                l3_snapshot=data.get("l3_snapshot"),
                analysis_result=data.get("analysis_result")
            )
            
            dossier.metadata.status = DossierStatus.LOADED
            
            self._stats["dossiers_loaded"] += 1
            logger.info(f"[DossierSerializer] 档案加载: {file_path}")
            
            return dossier
            
        except Exception as e:
            logger.error(f"[DossierSerializer] 档案加载失败: {e}")
            raise
    
    def freeze_from_buffer(
        self,
        ring_buffer: MultiLayerRingBuffer,
        event_id: str,
        task_id: str,
        result: str = "unknown",
        error_message: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Dossier:
        """
        从环形缓冲区创建并冻结档案
        
        Args:
            ring_buffer: 环形缓冲区
            event_id: 事件ID
            task_id: 任务ID
            result: 结果状态
            error_message: 错误信息
            start_time: 起始时间
            end_time: 结束时间
        
        Returns:
            Dossier: 档案对象
        """
        slots = ring_buffer.snapshot(start_time=start_time, end_time=end_time)
        
        dossier = self.create_dossier(
            event_id=event_id,
            task_id=task_id,
            slots=slots,
            result=result,
            error_message=error_message
        )
        
        return dossier
    
    def list_dossiers(self) -> List[Dict[str, Any]]:
        """
        列出所有档案
        
        Returns:
            List[Dict]: 档案列表
        """
        dossiers = []
        
        for file_name in os.listdir(self.storage_path):
            if file_name.endswith(self.FILE_EXTENSION) or file_name.endswith(self.FILE_EXTENSION + ".gz"):
                file_path = os.path.join(self.storage_path, file_name)
                try:
                    stat = os.stat(file_path)
                    dossiers.append({
                        "file_name": file_name,
                        "file_path": file_path,
                        "size_bytes": stat.st_size,
                        "modified_time": stat.st_mtime
                    })
                except Exception as e:
                    logger.warning(f"[DossierSerializer] 无法读取档案信息: {file_name}, {e}")
        
        return dossiers
    
    def delete_dossier(self, file_path: str) -> bool:
        """删除档案"""
        try:
            os.remove(file_path)
            logger.info(f"[DossierSerializer] 档案删除: {file_path}")
            return True
        except Exception as e:
            logger.error(f"[DossierSerializer] 档案删除失败: {e}")
            return False
    
    def _calculate_checksum(self, data: str) -> str:
        """计算校验和"""
        import hashlib
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "storage_path": self.storage_path,
            "stats": self._stats.copy()
        }


_global_serializer: Optional[DossierSerializer] = None


def get_serializer(storage_path: str = "dossiers") -> DossierSerializer:
    """获取全局序列化器"""
    global _global_serializer
    if _global_serializer is None:
        _global_serializer = DossierSerializer(storage_path=storage_path)
    return _global_serializer
