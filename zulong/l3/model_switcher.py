# File: zulong/l3/model_switcher.py
# 模型热切换器 - 管理冷备模型池和热插拔

import torch
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time
import os

from .dual_brain_container import DualBrainContainer, BrainRole

logger = logging.getLogger(__name__)


class ModelLocation(Enum):
    """模型位置枚举"""
    DISK = "disk"  # 存储在磁盘
    CPU_RAM = "cpu_ram"  # 已加载到 CPU 内存
    GPU = "gpu"  # 已加载到 GPU


@dataclass
class ColdBackupModel:
    """冷备模型元数据
    
    TSD v1.7 对应规则:
    - 冷备模型池管理（3 个槽位）
    - 纯 CPU RAM 存储方案（方案 A）
    """
    model_id: str  # 模型唯一标识
    model_path: str  # 模型文件路径
    model_type: str = ""  # 模型类型（导航、操作、视觉等）
    location: ModelLocation = ModelLocation.DISK  # 当前位置
    vram_requirement_gb: float = 1.5  # 显存需求
    ram_usage_gb: float = 1.5  # 内存占用（加载到 RAM 后）
    
    # 状态
    is_loaded_to_ram: bool = False  # 是否已加载到 RAM
    is_loaded_to_gpu: bool = False  # 是否已加载到 GPU
    last_access_time: float = field(default_factory=time.time)  # 最后访问时间
    access_count: int = 0  # 访问次数
    
    # 模型实例（如果已加载）
    model_instance: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_type": self.model_type,
            "location": self.location.value,
            "vram_requirement_gb": self.vram_requirement_gb,
            "is_loaded_to_ram": self.is_loaded_to_ram,
            "is_loaded_to_gpu": self.is_loaded_to_gpu,
            "access_count": self.access_count
        }


class ModelSwitcher:
    """模型热切换器
    
    TSD v1.7 对应规则:
    - 冷备模型池管理（3 个槽位）
    - 模型热插拔（磁盘→RAM→GPU）
    - 端口切换（将冷备模型接入左右脑端口）
    - 经验/记忆热迁移
    
    功能:
    - 冷备模型槽位管理
    - 模型加载/卸载（GPU↔RAM↔Disk）
    - LRU 驱逐策略
    - 切换统计与监控
    """
    
    # 冷备模型槽位数量（固定 3 个）
    COLD_BACKUP_SLOTS = 3
    
    def __init__(self, dual_brain: DualBrainContainer, max_system_ram_gb: float = 8.0):
        """初始化模型切换器
        
        Args:
            dual_brain: 左右脑容器实例
            max_system_ram_gb: 最大可用系统内存（用于冷备模型）
        """
        self.dual_brain = dual_brain
        self.max_system_ram_gb = max_system_ram_gb
        
        # 冷备模型池（3 个槽位）
        self.cold_backup_slots: Dict[int, Optional[ColdBackupModel]] = {
            0: None,
            1: None,
            2: None
        }
        
        # 模型路径映射（model_id -> slot_index）
        self.model_slot_map: Dict[str, int] = {}
        
        # 统计信息
        self.total_loads = 0  # 总加载次数
        self.total_unloads = 0  # 总卸载次数
        self.total_switches = 0  # 总切换次数
        self.last_load_time = 0.0  # 最后加载时间
        self.last_switch_time = 0.0  # 最后切换时间
        
        logger.info(f"[ModelSwitcher] Initialized with {self.COLD_BACKUP_SLOTS} cold backup slots")
    
    def _find_empty_slot(self) -> Optional[int]:
        """查找空闲槽位"""
        for slot_idx in range(self.COLD_BACKUP_SLOTS):
            if self.cold_backup_slots[slot_idx] is None:
                return slot_idx
        return None
    
    def _find_lru_slot(self) -> int:
        """查找 LRU 槽位（最近最少使用）"""
        lru_slot = 0
        lru_time = float('inf')
        
        for slot_idx in range(self.COLD_BACKUP_SLOTS):
            model = self.cold_backup_slots[slot_idx]
            if model is not None:
                if model.last_access_time < lru_time:
                    lru_time = model.last_access_time
                    lru_slot = slot_idx
        
        return lru_slot
    
    def register_model(self, model_id: str, model_path: str, model_type: str = "", 
                      vram_gb: float = 1.5) -> bool:
        """注册冷备模型
        
        Args:
            model_id: 模型唯一标识
            model_path: 模型文件路径
            model_type: 模型类型
            vram_gb: 显存需求
            
        Returns:
            bool: 注册是否成功
        """
        # 检查是否已注册
        if model_id in self.model_slot_map:
            logger.warning(f"[ModelSwitcher] Model {model_id} already registered")
            return False
        
        # 查找空闲槽位
        slot_idx = self._find_empty_slot()
        
        # 如果没有空闲槽位，使用 LRU 驱逐
        if slot_idx is None:
            slot_idx = self._find_lru_slot()
            old_model = self.cold_backup_slots[slot_idx]
            if old_model is not None:
                logger.info(f"[ModelSwitcher] Evicting LRU model: {old_model.model_id}")
                self.unload_model_from_gpu(old_model.model_id, force=True)
                self.cold_backup_slots[slot_idx] = None
                del self.model_slot_map[old_model.model_id]
        
        # 创建冷备模型元数据
        cold_model = ColdBackupModel(
            model_id=model_id,
            model_path=model_path,
            model_type=model_type,
            vram_requirement_gb=vram_gb
        )
        
        # 注册到槽位
        self.cold_backup_slots[slot_idx] = cold_model
        self.model_slot_map[model_id] = slot_idx
        
        logger.info(f"[ModelSwitcher] Registered model: {model_id} -> Slot {slot_idx}")
        
        return True
    
    def load_model_to_gpu(self, model_id: str, target_brain: BrainRole = None) -> bool:
        """加载模型到 GPU（并连接到指定脑）
        
        加载流程：磁盘 → CPU RAM → GPU
        
        Args:
            model_id: 模型 ID
            target_brain: 目标脑（左脑/右脑），如果为 None 则不切换
            
        Returns:
            bool: 加载是否成功
        """
        start_time = time.time()
        
        # 查找模型
        if model_id not in self.model_slot_map:
            logger.error(f"[ModelSwitcher] Model {model_id} not found")
            return False
        
        slot_idx = self.model_slot_map[model_id]
        model = self.cold_backup_slots[slot_idx]
        
        if model is None:
            logger.error(f"[ModelSwitcher] Slot {slot_idx} is empty")
            return False
        
        logger.info(f"[ModelSwitcher] Loading model: {model_id} (Slot {slot_idx})")
        
        # 1. 从磁盘加载到 CPU RAM（如果还未加载）
        if not model.is_loaded_to_ram:
            logger.info(f"[ModelSwitcher] Loading from disk to RAM: {model.model_path}")
            try:
                # 模拟加载过程（实际应该用 torch.load）
                # model.model_instance = torch.load(model.model_path, map_location='cpu')
                model.model_instance = {"type": "mock_model", "id": model_id}
                model.is_loaded_to_ram = True
                model.ram_usage_gb = model.vram_requirement_gb
                logger.info(f"[ModelSwitcher] Loaded to RAM: {model_id}")
            except Exception as e:
                logger.error(f"[ModelSwitcher] Failed to load to RAM: {e}")
                return False
        
        # 2. 从 CPU RAM 加载到 GPU
        if not model.is_loaded_to_gpu:
            logger.info(f"[ModelSwitcher] Loading from RAM to GPU")
            try:
                # 模拟 GPU 加载
                # model.model_instance.to('cuda')
                model.is_loaded_to_gpu = True
                logger.info(f"[ModelSwitcher] Loaded to GPU: {model_id}")
            except Exception as e:
                logger.error(f"[ModelSwitcher] Failed to load to GPU: {e}")
                return False
        
        # 3. 切换到目标脑（如果指定）
        if target_brain is not None:
            logger.info(f"[ModelSwitcher] Switching to brain: {target_brain.value}")
            self.dual_brain.switch_active_brain(target_brain)
        
        # 更新统计
        model.last_access_time = time.time()
        model.access_count += 1
        self.total_loads += 1
        self.last_load_time = time.time() - start_time
        
        logger.info(f"[ModelSwitcher] Load complete: {model_id} "
                   f"(time: {self.last_load_time*1000:.2f}ms)")
        
        return True
    
    def unload_model_from_gpu(self, model_id: str, force: bool = False) -> bool:
        """从 GPU 卸载模型到 CPU RAM
        
        卸载流程：GPU → CPU RAM
        
        Args:
            model_id: 模型 ID
            force: 是否强制卸载（即使模型正在使用）
            
        Returns:
            bool: 卸载是否成功
        """
        if model_id not in self.model_slot_map:
            logger.warning(f"[ModelSwitcher] Model {model_id} not found, skip unload")
            return True
        
        slot_idx = self.model_slot_map[model_id]
        model = self.cold_backup_slots[slot_idx]
        
        if model is None or not model.is_loaded_to_gpu:
            logger.debug(f"[ModelSwitcher] Model {model_id} not on GPU, skip unload")
            return True
        
        logger.info(f"[ModelSwitcher] Unloading model: {model_id}")
        
        try:
            # 模拟 GPU 卸载
            # model.model_instance.to('cpu')
            model.is_loaded_to_gpu = False
            model.last_access_time = time.time()
            
            self.total_unloads += 1
            
            logger.info(f"[ModelSwitcher] Unloaded to RAM: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"[ModelSwitcher] Failed to unload: {e}")
            return False
    
    def switch_model(self, from_model_id: str, to_model_id: str, 
                    target_brain: BrainRole) -> bool:
        """切换模型（热插拔）
        
        流程:
        1. 卸载当前模型（GPU → RAM）
        2. 加载目标模型（RAM → GPU）
        3. 切换到目标脑
        4. 迁移经验/记忆
        
        Args:
            from_model_id: 当前模型 ID
            to_model_id: 目标模型 ID
            target_brain: 目标脑
            
        Returns:
            bool: 切换是否成功
        """
        start_time = time.time()
        
        logger.info(f"[ModelSwitcher] Switching model: {from_model_id} -> {to_model_id}")
        
        # 1. 卸载当前模型
        if from_model_id:
            self.unload_model_from_gpu(from_model_id)
        
        # 2. 加载目标模型
        success = self.load_model_to_gpu(to_model_id, target_brain)
        
        if not success:
            logger.error(f"[ModelSwitcher] Failed to switch model")
            return False
        
        # 3. 更新统计
        self.total_switches += 1
        self.last_switch_time = time.time() - start_time
        
        logger.info(f"[ModelSwitcher] Switch complete: {from_model_id} -> {to_model_id} "
                   f"(time: {self.last_switch_time*1000:.2f}ms)")
        
        return True
    
    def migrate_context(self, from_brain: BrainRole, to_brain: BrainRole) -> bool:
        """迁移上下文（经验/记忆热迁移）
        
        TSD v1.7 对应规则:
        - 经验/记忆热迁移
        - 上下文不丢失
        
        Args:
            from_brain: 源脑
            to_brain: 目标脑
            
        Returns:
            bool: 迁移是否成功
        """
        logger.info(f"[ModelSwitcher] Migrating context: {from_brain.value} -> {to_brain.value}")
        
        # 获取源脑上下文
        current_brain = self.dual_brain.get_active_brain()
        if current_brain.role != from_brain:
            # 切换到源脑
            self.dual_brain.switch_active_brain(from_brain)
            current_brain = self.dual_brain.get_active_brain()
        
        # 获取上下文数据
        context_data = current_brain.context.to_dict()
        
        # 切换到目标脑
        self.dual_brain.switch_active_brain(to_brain)
        target_brain = self.dual_brain.get_active_brain()
        
        # 迁移上下文
        target_brain.context.from_dict(context_data)
        
        # 同步回源脑（保持一致性）
        self.dual_brain._sync_brains(source=target_brain, target=current_brain)
        
        logger.info(f"[ModelSwitcher] Context migrated successfully")
        
        return True
    
    def get_cold_backup_status(self) -> Dict[str, Any]:
        """获取冷备模型池状态"""
        slots_info = []
        for slot_idx in range(self.COLD_BACKUP_SLOTS):
            model = self.cold_backup_slots[slot_idx]
            if model is not None:
                slots_info.append({
                    "slot": slot_idx,
                    "model_id": model.model_id,
                    "model_type": model.model_type,
                    "location": model.location.value,
                    "is_loaded_to_ram": model.is_loaded_to_ram,
                    "is_loaded_to_gpu": model.is_loaded_to_gpu,
                    "access_count": model.access_count
                })
            else:
                slots_info.append({
                    "slot": slot_idx,
                    "model_id": None,
                    "status": "empty"
                })
        
        return {
            "total_slots": self.COLD_BACKUP_SLOTS,
            "used_slots": sum(1 for m in self.cold_backup_slots.values() if m is not None),
            "slots": slots_info,
            "total_loads": self.total_loads,
            "total_unloads": self.total_unloads,
            "total_switches": self.total_switches,
            "last_load_time_ms": self.last_load_time * 1000,
            "last_switch_time_ms": self.last_switch_time * 1000
        }
    
    def print_status(self):
        """打印状态信息"""
        status = self.get_cold_backup_status()
        
        print("\n" + "=" * 60)
        print("模型热切换器状态")
        print("=" * 60)
        print(f"冷备模型槽位：{status['used_slots']}/{status['total_slots']}")
        
        for slot_info in status["slots"]:
            if slot_info["model_id"]:
                print(f"  槽位 {slot_info['slot']}: {slot_info['model_id']} "
                      f"({slot_info['model_type']}, "
                      f"RAM={slot_info['is_loaded_to_ram']}, "
                      f"GPU={slot_info['is_loaded_to_gpu']}, "
                      f"访问={slot_info['access_count']}次)")
            else:
                print(f"  槽位 {slot_info['slot']}: (空闲)")
        
        print(f"\n加载次数：{status['total_loads']}")
        print(f"卸载次数：{status['total_unloads']}")
        print(f"切换次数：{status['total_switches']}")
        print(f"最后加载时间：{status['last_load_time_ms']:.2f}ms")
        print(f"最后切换时间：{status['last_switch_time_ms']:.2f}ms")
        print("=" * 60 + "\n")
