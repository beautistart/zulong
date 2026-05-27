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

    data_dir = config.get('data_dir') if config else get_config('memory.hybrid_storage.data_dir', './data/memory_graph_hybrid')
    map_size_gb = config.get('map_size_gb') if config else get_config('memory.hybrid_storage.map_size_gb', 10)
    shard_strategy = config.get('shard_strategy') if config else get_config('memory.hybrid_storage.shard_strategy', 'month')
    max_active_shards = config.get('max_active_shards') if config else get_config('memory.hybrid_storage.max_active_shards', 3)
    enable_vector_index = config.get('enable_vector_index') if config else get_config('memory.hybrid_storage.enable_vector_index', False)
    max_nodes_per_shard = config.get('max_nodes_per_shard') if config else get_config('memory.hybrid_storage.max_nodes_per_shard', 150000)

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
        max_nodes_per_shard=max_nodes_per_shard
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
