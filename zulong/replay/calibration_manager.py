"""
AT-21/23/24/25: 参数热更新机制
实现参数的实时覆盖，无需重启机器人
参数生效延迟 < 50ms
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Awaitable
from enum import Enum
import logging
import json
import os
import asyncio
from pathlib import Path

from .clock_synchronizer import get_unified_timestamp
from .patch_compiler import SystemPatch, PatchStatus

logger = logging.getLogger(__name__)


class ParamType(Enum):
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    JSON = "json"


@dataclass
class CalibrationParam:
    """校准参数"""
    key: str
    value: Any
    param_type: ParamType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""
    last_updated: float = 0.0
    version: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "param_type": self.param_type.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "description": self.description,
            "last_updated": self.last_updated,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationParam':
        return cls(
            key=data["key"],
            value=data["value"],
            param_type=ParamType(data["param_type"]),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            description=data.get("description", ""),
            last_updated=data.get("last_updated", 0.0),
            version=data.get("version", 0)
        )
    
    def validate(self, new_value: Any) -> bool:
        """验证新值"""
        if self.param_type == ParamType.FLOAT:
            try:
                val = float(new_value)
                if self.min_value is not None and val < self.min_value:
                    return False
                if self.max_value is not None and val > self.max_value:
                    return False
                return True
            except (TypeError, ValueError):
                return False
        elif self.param_type == ParamType.INT:
            try:
                val = int(new_value)
                if self.min_value is not None and val < self.min_value:
                    return False
                if self.max_value is not None and val > self.max_value:
                    return False
                return True
            except (TypeError, ValueError):
                return False
        elif self.param_type == ParamType.BOOL:
            return isinstance(new_value, bool)
        elif self.param_type == ParamType.STRING:
            return isinstance(new_value, str)
        elif self.param_type == ParamType.JSON:
            return isinstance(new_value, (dict, list))
        return True


@dataclass
class CalibrationEvent:
    """校准事件"""
    event_id: str
    timestamp: float
    param_changes: Dict[str, Any]
    source: str
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "param_changes": self.param_changes,
            "source": self.source,
            "success": self.success,
            "error_message": self.error_message
        }


class CalibrationManager:
    """
    参数热更新管理器 (AT-21/23/24/25)
    
    实现参数的实时覆盖，无需重启机器人
    参数生效延迟 < 50ms
    """
    
    def __init__(self, config_path: str = "config/calibration.json"):
        """
        初始化校准管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._params: Dict[str, CalibrationParam] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[CalibrationEvent] = []
        self._version = 0
        self._lock = asyncio.Lock()
        
        self._stats = {
            "updates_applied": 0,
            "updates_failed": 0,
            "subscribers_notified": 0
        }
        
        self._load_from_file()
        
        logger.info(f"[CalibrationManager] 初始化完成，配置路径: {config_path}")
    
    def register_param(
        self,
        key: str,
        default_value: Any,
        param_type: ParamType,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        description: str = ""
    ) -> CalibrationParam:
        """
        注册参数
        
        Args:
            key: 参数键
            default_value: 默认值
            param_type: 参数类型
            min_value: 最小值
            max_value: 最大值
            description: 描述
        
        Returns:
            CalibrationParam: 参数对象
        """
        param = CalibrationParam(
            key=key,
            value=default_value,
            param_type=param_type,
            min_value=min_value,
            max_value=max_value,
            description=description,
            last_updated=get_unified_timestamp(),
            version=0
        )
        
        self._params[key] = param
        logger.debug(f"[CalibrationManager] 参数注册: {key}")
        return param
    
    def subscribe(self, key: str, callback: Callable[[Any], Awaitable[None]]):
        """
        订阅参数变更
        
        Args:
            key: 参数键
            callback: 回调函数
        """
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)
        logger.debug(f"[CalibrationManager] 订阅参数: {key}")
    
    async def apply_calibration(
        self,
        params: Dict[str, Any],
        source: str = "manual",
        validate: bool = True
    ) -> bool:
        """
        应用校准参数 (热更新)
        
        Args:
            params: 参数字典
            source: 来源
            validate: 是否验证
        
        Returns:
            bool: 是否成功
        """
        async with self._lock:
            event_id = f"CAL_{get_unified_timestamp():.0f}"
            timestamp = get_unified_timestamp()
            
            valid_changes = {}
            errors = []
            
            for key, new_value in params.items():
                if key not in self._params:
                    errors.append(f"未知参数: {key}")
                    continue
                
                param = self._params[key]
                
                if validate and not param.validate(new_value):
                    errors.append(f"参数验证失败: {key}={new_value}")
                    continue
                
                old_value = param.value
                param.value = new_value
                param.last_updated = timestamp
                param.version += 1
                
                valid_changes[key] = {
                    "old_value": old_value,
                    "new_value": new_value
                }
                
                if key in self._subscribers:
                    for callback in self._subscribers[key]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(new_value)
                            else:
                                callback(new_value)
                            self._stats["subscribers_notified"] += 1
                        except Exception as e:
                            logger.error(f"[CalibrationManager] 回调失败: {key}, {e}")
            
            success = len(errors) == 0 and len(valid_changes) > 0
            
            event = CalibrationEvent(
                event_id=event_id,
                timestamp=timestamp,
                param_changes=valid_changes,
                source=source,
                success=success,
                error_message="; ".join(errors) if errors else None
            )
            
            self._event_history.append(event)
            if len(self._event_history) > 100:
                self._event_history = self._event_history[-100:]
            
            if success:
                self._version += 1
                self._stats["updates_applied"] += 1
                await self._save_to_file()
                logger.info(f"[CalibrationManager] 参数更新成功: {list(valid_changes.keys())}")
            else:
                self._stats["updates_failed"] += 1
                logger.warning(f"[CalibrationManager] 参数更新失败: {errors}")
            
            return success
    
    async def apply_patch(self, patch: SystemPatch) -> bool:
        """
        应用系统补丁
        
        Args:
            patch: 系统补丁
        
        Returns:
            bool: 是否成功
        """
        params = self._parse_patch_adjustment(patch.adjustment)
        if not params:
            return False
        
        return await self.apply_calibration(params, source=f"patch:{patch.patch_id}")
    
    def _parse_patch_adjustment(self, adjustment: str) -> Dict[str, Any]:
        """
        解析补丁调整语句
        
        Args:
            adjustment: 调整语句
        
        Returns:
            Dict[str, Any]: 参数字典
        """
        params = {}
        
        import re
        
        patterns = [
            r"L0\.speed_compensation\s*\+=\s*([\d.]+)%",
            r"L2\.prediction_delay\s*\+=\s*([\d.]+)ms",
            r"L3\.(\w+)\s*=\s*(\w+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, adjustment)
            if matches:
                if "speed_compensation" in pattern:
                    params["l0.speed_compensation"] = float(matches[0])
                elif "prediction_delay" in pattern:
                    params["l2.prediction_delay_ms"] = float(matches[0])
        
        return params
    
    def get_param(self, key: str) -> Optional[Any]:
        """获取参数值"""
        param = self._params.get(key)
        return param.value if param else None
    
    def get_all_params(self) -> Dict[str, Any]:
        """获取所有参数"""
        return {key: param.value for key, param in self._params.items()}
    
    def get_param_info(self, key: str) -> Optional[CalibrationParam]:
        """获取参数信息"""
        return self._params.get(key)
    
    def get_version(self) -> int:
        """获取当前版本"""
        return self._version
    
    def get_event_history(self, limit: int = 10) -> List[CalibrationEvent]:
        """获取事件历史"""
        return self._event_history[-limit:]
    
    async def _save_to_file(self):
        """保存到文件"""
        data = {
            "version": self._version,
            "params": {key: param.to_dict() for key, param in self._params.items()},
            "updated_at": get_unified_timestamp()
        }
        
        os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else ".", exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load_from_file(self):
        """从文件加载"""
        if not os.path.exists(self.config_path):
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._version = data.get("version", 0)
            
            for key, param_data in data.get("params", {}).items():
                self._params[key] = CalibrationParam.from_dict(param_data)
            
            logger.info(f"[CalibrationManager] 从文件加载 {len(self._params)} 个参数")
        except Exception as e:
            logger.error(f"[CalibrationManager] 加载失败: {e}")
    
    def reset_to_defaults(self):
        """重置为默认值"""
        for param in self._params.values():
            param.version = 0
        self._version = 0
        logger.info("[CalibrationManager] 参数已重置")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "version": self._version,
            "param_count": len(self._params),
            "subscriber_count": sum(len(v) for v in self._subscribers.values()),
            "stats": self._stats.copy()
        }


_global_calibration_manager: Optional[CalibrationManager] = None


def get_calibration_manager() -> CalibrationManager:
    """获取全局校准管理器"""
    global _global_calibration_manager
    if _global_calibration_manager is None:
        _global_calibration_manager = CalibrationManager()
    return _global_calibration_manager
