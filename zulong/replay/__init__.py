"""
ZULONG 事件复盘机制模块

包含六阶段原子任务体系的完整实现:
- AT-01: 硬件时钟同步协议
- AT-02: L0执行层日志扩展
- AT-03: L1-A反射层日志扩展
- AT-04: L2/L3上下文快照接口
- AT-05: 环形缓冲区实现
- AT-06: 事件档案序列化
- AT-11/12/13: 归因决策树引擎
- AT-14/17: 复盘编译器与System_Patch
- AT-16: 经验向量库
- AT-21/23/24/25: 参数热更新机制
"""

from .clock_synchronizer import (
    ClockSynchronizer,
    ClockSyncStatus,
    get_unified_timestamp,
    get_unified_timestamp_ms
)
from .ring_buffer import (
    RingBufferSlot,
    MultiLayerRingBuffer,
    get_ring_buffer
)
from .l0_logger import (
    L0Logger,
    L0EventType,
    JointState,
    MotorCommand,
    get_l0_logger
)
from .l1a_logger import (
    L1ALogger,
    L1AEventType,
    VADData,
    VisionDetectionData,
    ReflexAction,
    get_l1a_logger
)
from .context_snapshot import (
    ContextSnapshotManager,
    L2Snapshot,
    L3Snapshot,
    get_snapshot_manager
)
from .dossier_serializer import (
    DossierSerializer,
    Dossier,
    DossierMetadata,
    DossierStatus,
    get_serializer
)
from .attributor import (
    AttributionEngine,
    TrajectoryAligner,
    TimingAnalyzer,
    FaultLayer,
    AdjustmentType,
    Trajectory,
    TrajectoryPoint,
    AlignmentResult,
    TimingAnalysis,
    AttributionResult,
    get_attribution_engine
)
from .patch_compiler import (
    PatchCompiler,
    SystemPatch,
    PatchStatus,
    get_patch_compiler
)
from .experience_store import (
    ExperienceStore,
    ExperienceEntry,
    StoreBackend,
    get_experience_store
)
from .calibration_manager import (
    CalibrationManager,
    CalibrationParam,
    CalibrationEvent,
    ParamType,
    get_calibration_manager
)
from .integration import (
    ReplayIntegration,
    ReplayContext,
    get_replay_integration
)

__all__ = [
    "ClockSynchronizer",
    "ClockSyncStatus",
    "get_unified_timestamp",
    "get_unified_timestamp_ms",
    "RingBufferSlot",
    "MultiLayerRingBuffer",
    "get_ring_buffer",
    "L0Logger",
    "L0EventType",
    "JointState",
    "MotorCommand",
    "get_l0_logger",
    "L1ALogger",
    "L1AEventType",
    "VADData",
    "VisionDetectionData",
    "ReflexAction",
    "get_l1a_logger",
    "ContextSnapshotManager",
    "L2Snapshot",
    "L3Snapshot",
    "get_snapshot_manager",
    "DossierSerializer",
    "Dossier",
    "DossierMetadata",
    "DossierStatus",
    "get_serializer",
    "AttributionEngine",
    "TrajectoryAligner",
    "TimingAnalyzer",
    "FaultLayer",
    "AdjustmentType",
    "Trajectory",
    "TrajectoryPoint",
    "AlignmentResult",
    "TimingAnalysis",
    "AttributionResult",
    "get_attribution_engine",
    "PatchCompiler",
    "SystemPatch",
    "PatchStatus",
    "get_patch_compiler",
    "ExperienceStore",
    "ExperienceEntry",
    "StoreBackend",
    "get_experience_store",
    "CalibrationManager",
    "CalibrationParam",
    "CalibrationEvent",
    "ParamType",
    "get_calibration_manager",
    "ReplayIntegration",
    "ReplayContext",
    "get_replay_integration",
]
