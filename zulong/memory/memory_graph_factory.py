"""
MemoryGraph 工厂方法 - 运行态统一使用分片 Hybrid 存储。

旧的 NetworkX + 单 JSON 存储已退出运行链路。MemoryGraph 类中的枚举、
GraphNode 数据结构和部分兼容方法仍被工具层复用，但不再作为持久化后端。
"""

import logging
from typing import Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


def create_memory_graph(persist_path: str = "./data/memory_graph", config: Optional[Dict] = None):
    """
    创建分片 MemoryGraph 实例。
    
    Args:
        persist_path: 旧参数，保留给调用方兼容；分片后端实际使用
            memory.hybrid_storage.data_dir。
        config: 可选的配置字典（用于测试覆盖）
    
    Returns:
        ShardedMemoryGraph 实例
    """
    if isinstance(persist_path, dict) and config is None:
        config = persist_path
        persist_path = config.get('data_dir', './data/memory_graph_hybrid')

    try:
        from zulong.config.config_manager import get_config
    except ImportError:
        get_config = lambda key, default=None: default

    logger.info("[MemoryGraphFactory] 使用分片 Hybrid 存储后端 (igraph + LMDB)")

    try:
        from .storage_hybrid import ShardedMemoryGraph, ShardStrategy
    except ImportError as e:
        logger.error(f"[MemoryGraphFactory] 无法导入 hybrid 存储: {e}")
        raise RuntimeError("MemoryGraph 分片存储不可用，已禁止回退到单 JSON 后端") from e

    def _cfg(name: str, default):
        if config and config.get(name) is not None:
            return config.get(name)
        return get_config(f'memory.hybrid_storage.{name}', default)

    data_dir = _cfg('data_dir', './data/memory_graph_hybrid')
    map_size_gb = _cfg('map_size_gb', 10)
    shard_strategy = _cfg('shard_strategy', 'month')
    max_active_shards = _cfg('max_active_shards', 3)
    enable_vector_index = _cfg('enable_vector_index', False)
    max_nodes_per_shard = _cfg('max_nodes_per_shard', 150000)
    max_shard_property_mb_warning = _cfg('max_shard_property_mb_warning', 150)
    max_shard_property_mb_split = _cfg('max_shard_property_mb_split', 200)
    max_shard_topology_mb_warning = _cfg('max_shard_topology_mb_warning', 64)
    max_shard_topology_delta_mb_compact = _cfg('max_shard_topology_delta_mb_compact', 32)
    max_active_skeleton_nodes = _cfg('max_active_skeleton_nodes', 50000)
    local_index_map_size_mb = _cfg('local_index_map_size_mb', 64)
    shard_size_check_interval_nodes = _cfg('shard_size_check_interval_nodes', 100)
    global_index_map_size_mb = _cfg('global_index_map_size_mb', 256)

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"[MemoryGraphFactory] 启用分片策略: {shard_strategy}")
    strategy_map = {
        'month': ShardStrategy.MONTHLY,
        'week': ShardStrategy.WEEKLY,
        'day': ShardStrategy.DAILY,
    }
    return ShardedMemoryGraph(
        base_dir=data_dir,
        shard_strategy=strategy_map.get(shard_strategy, ShardStrategy.MONTHLY),
        max_active_shards=max_active_shards,
        map_size_gb=map_size_gb,
        enable_vector_index=enable_vector_index,
        max_nodes_per_shard=max_nodes_per_shard,
        max_shard_property_mb_warning=max_shard_property_mb_warning,
        max_shard_property_mb_split=max_shard_property_mb_split,
        max_shard_topology_mb_warning=max_shard_topology_mb_warning,
        max_shard_topology_delta_mb_compact=max_shard_topology_delta_mb_compact,
        max_active_skeleton_nodes=max_active_skeleton_nodes,
        local_index_map_size_mb=local_index_map_size_mb,
        shard_size_check_interval_nodes=shard_size_check_interval_nodes,
        global_index_map_size_mb=global_index_map_size_mb
    )


def get_memory_graph_type(graph_instance) -> str:
    """获取图谱实例的类型"""
    class_name = graph_instance.__class__.__name__
    if 'Sharded' in class_name:
        return "sharded"
    if 'Hybrid' in class_name:
        return "hybrid"
    return "networkx"


def get_memory_graph_stats(graph_instance) -> Dict[str, Any]:
    """获取存储统计信息"""
    result = {
        "type": get_memory_graph_type(graph_instance),
        "node_count": len(graph_instance) if hasattr(graph_instance, '__len__') else 0,
    }
    
    if hasattr(graph_instance, 'get_stats'):
        result.update(graph_instance.get_stats())
    
    return result
