# File: zulong/tools/core_tool_manager.py
# CoreToolManager - 热/冷工具管理器
#
# 管理哪些工具是"热工具"（始终在 prompt 中），
# 哪些是"冷工具"（存入 ToolRAG，需要时通过 search_tools 检索）。
#
# 设计原则：
# - 热工具数量受限（默认 ≤ 8 个），确保不占满上下文窗口
# - 冷工具数量无限，通过语义检索按需加载
# - search_tools 本身永远是热工具（它是发现冷工具的入口）
# - 内置工具默认为热工具，技能包工具默认为冷工具（可配置）

import logging
from typing import Dict, Any, List, Optional, Set

logger = logging.getLogger(__name__)

# 默认热工具白名单（始终在 prompt 中）
DEFAULT_HOT_TOOLS: Set[str] = {
    "openclaw_search",     # 核心搜索
    "read_memory_detail",  # 记忆检索
    "search_tools",        # 元工具（发现冷工具的入口）
}

# 热工具数量上限
MAX_HOT_TOOLS = 8


class CoreToolManager:
    """热/冷工具管理器
    
    职责：
    1. 维护热工具集合（始终注入 prompt）
    2. 管理冷工具（存入 ToolRAG）
    3. 在推理迭代中，将 search_tools 发现的冷工具临时提升为热工具
    4. 提供统一接口获取当前应该注入 prompt 的工具列表
    """
    
    def __init__(self, tool_rag=None, max_hot_tools: int = MAX_HOT_TOOLS):
        """
        Args:
            tool_rag: ToolRAG 实例（管理冷工具索引）
            max_hot_tools: 热工具数量上限
        """
        self._tool_rag = tool_rag
        self._max_hot_tools = max_hot_tools
        
        # 持久化的热工具集合
        self._hot_tools: Set[str] = set(DEFAULT_HOT_TOOLS)
        
        # 临时提升的工具（单次推理有效，推理结束后清空）
        self._temp_promoted: Set[str] = set()
        
        # 所有已知工具名 -> schema 的缓存
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"[CoreToolManager] Initialized: hot={self._hot_tools}, max={max_hot_tools}")
    
    def set_tool_rag(self, tool_rag):
        """延迟注入 ToolRAG"""
        self._tool_rag = tool_rag
    
    # ==================== 热/冷分类 ====================
    
    def register_tool(self, tool_name: str, schema: Dict[str, Any],
                      source: str = "builtin", force_hot: bool = False):
        """注册工具并自动分类为热/冷
        
        分类策略:
        - 在 DEFAULT_HOT_TOOLS 白名单中 → 热
        - source == "builtin" 且热工具未满 → 热
        - 否则 → 冷（写入 ToolRAG）
        
        Args:
            tool_name: 工具名
            schema: OpenAI Function Calling schema
            source: 来源 ("builtin" / skill_pack_id)
            force_hot: 强制设为热工具
        """
        # 缓存 schema
        self._schema_cache[tool_name] = schema
        
        # 判断是否应该是热工具
        is_hot = (
            force_hot
            or tool_name in DEFAULT_HOT_TOOLS
            or (source == "builtin" and len(self._hot_tools) < self._max_hot_tools)
        )
        
        if is_hot:
            self._hot_tools.add(tool_name)
            logger.debug(f"[CoreToolManager] {tool_name} -> HOT")
        else:
            # 冷工具：写入 ToolRAG
            self._register_cold_tool(tool_name, schema, source)
            logger.debug(f"[CoreToolManager] {tool_name} -> COLD (ToolRAG)")
    
    def unregister_tool(self, tool_name: str):
        """注销工具"""
        self._hot_tools.discard(tool_name)
        self._temp_promoted.discard(tool_name)
        self._schema_cache.pop(tool_name, None)
        
        # 从 ToolRAG 中移除
        if self._tool_rag and self._tool_rag.has_tool(tool_name):
            self._tool_rag.remove_tool(tool_name)
        
        logger.debug(f"[CoreToolManager] Unregistered: {tool_name}")
    
    # ==================== 推理时的工具列表 ====================
    
    def get_active_schemas(self) -> List[Dict[str, Any]]:
        """获取当前应该注入 prompt 的所有工具 schema
        
        包括：
        1. 所有热工具
        2. 本轮临时提升的冷工具
        
        Returns:
            List[Dict]: OpenAI Function Calling schema 列表
        """
        active_names = self._hot_tools | self._temp_promoted
        schemas = []
        
        for name in active_names:
            schema = self._schema_cache.get(name)
            if schema:
                schemas.append(schema)
        
        return schemas
    
    def promote_tools(self, tool_names: List[str]):
        """临时提升冷工具为热工具（当前推理周期内有效）
        
        由 inference_engine 在处理 search_tools 返回结果后调用。
        
        Args:
            tool_names: 要提升的工具名列表
        """
        for name in tool_names:
            if name in self._schema_cache:
                self._temp_promoted.add(name)
                logger.info(f"[CoreToolManager] Temp promoted: {name}")
            else:
                # schema 可能在 ToolRAG 中，需要加载
                if self._tool_rag:
                    schema = self._tool_rag.get_tool_schema(name)
                    if schema:
                        self._schema_cache[name] = schema
                        self._temp_promoted.add(name)
                        logger.info(f"[CoreToolManager] Temp promoted (from ToolRAG): {name}")
    
    def clear_temp_promoted(self):
        """清空临时提升的工具（推理周期结束时调用）"""
        if self._temp_promoted:
            logger.debug(f"[CoreToolManager] Clearing temp promoted: {self._temp_promoted}")
        self._temp_promoted.clear()
    
    # ==================== 查询 ====================
    
    def is_hot(self, tool_name: str) -> bool:
        """检查工具是否为热工具"""
        return tool_name in self._hot_tools
    
    def get_hot_tools(self) -> Set[str]:
        """获取热工具集合"""
        return set(self._hot_tools)
    
    def get_cold_tools(self) -> List[str]:
        """获取冷工具列表"""
        if self._tool_rag:
            return self._tool_rag.list_all_tools()
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "hot_tools": list(self._hot_tools),
            "hot_count": len(self._hot_tools),
            "cold_count": len(self.get_cold_tools()),
            "temp_promoted": list(self._temp_promoted),
            "total_cached_schemas": len(self._schema_cache),
            "max_hot_tools": self._max_hot_tools,
        }
    
    # ==================== 内部方法 ====================
    
    def _register_cold_tool(self, tool_name: str, schema: Dict[str, Any], source: str):
        """将工具注册为冷工具（写入 ToolRAG）"""
        if not self._tool_rag:
            logger.warning(f"[CoreToolManager] ToolRAG not available, cannot register cold tool: {tool_name}")
            return
        
        # 从 schema 中提取信息
        func_info = schema.get("function", {})
        description = func_info.get("description", "")
        params = func_info.get("parameters", {}).get("properties", {})
        param_names = list(params.keys())
        
        self._tool_rag.add_tool(
            tool_name=tool_name,
            description=description,
            param_names=param_names,
            source=source,
            function_schema=schema,
        )
