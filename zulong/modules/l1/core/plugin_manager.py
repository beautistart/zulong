# File: zulong/modules/l1/core/plugin_manager.py
"""
L1 层插件管理器

TSD v1.7 对应:
- 2.2.2 L1 层拆分
- 3.2 智能路由逻辑
- 4.1 L1-A 受控反射引擎

功能:
- 动态加载/卸载插件
- 按优先级排序执行
- 异常隔离 (单个插件崩溃不影响其他插件)
- 健康检查监控
- 事件收集与路由
"""

import importlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import yaml

from .interface import IL1Module, L1PluginBase, ZulongEvent, EventPriority, EventType

logger = logging.getLogger(__name__)


class PluginManager:
    """
    L1 层插件管理器
    
    架构原则:
    - 模块化软插接：插件可动态加载/卸载
    - 优先级调度：CRITICAL > HIGH > NORMAL > LOW
    - 异常隔离：try-except 包裹每个插件
    - 松耦合：通过 shared_memory 交换数据
    
    使用示例:
    ```python
    manager = PluginManager()
    manager.load_plugins("config/l1_plugins.yaml")
    
    while True:
        events = manager.run_cycle(shared_memory)
        for event in events:
            event_bus.publish(event)
    ```
    """
    
    def __init__(self):
        """初始化插件管理器"""
        self._plugins: Dict[str, IL1Module] = {}  # module_id -> instance
        self._plugin_configs: Dict[str, Dict] = {}  # module_id -> config
        self._shared_memory: Dict[str, Any] = {}  # 共享内存
        self._running = False
        self._cycle_count = 0
        self._last_health_check = 0.0
        self._health_check_interval = 10.0  # 每 10 秒健康检查
        
        # 🎯 性能统计
        self._cycle_times: Dict[str, float] = {}  # module_id -> last_cycle_time
        self._error_counts: Dict[str, int] = {}  # module_id -> error_count
    
    # ========== 插件生命周期管理 ==========
    
    def load_plugins(self, config_path: str) -> int:
        """
        从配置文件批量加载插件
        
        Args:
            config_path: YAML 配置文件路径
        
        Returns:
            int: 成功加载的插件数量
        
        配置文件格式 (l1_plugins.yaml):
        ```yaml
        plugins:
          - module_id: "L1A/Motor"
            class_path: "zulong.plugins.motor.L1A_MotorPlugin"
            enabled: true
            priority: "HIGH"
            config:
              max_speed: 0.8
              obstacle_threshold: 0.3
          
          - module_id: "L1C/Vision"
            class_path: "zulong.plugins.vision.L1C_VisionPlugin"
            enabled: true
            priority: "NORMAL"
        ```
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.error(f"❌ 配置文件不存在：{config_path}")
                return 0
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            plugins_config = config.get('plugins', [])
            loaded_count = 0
            
            logger.info(f"🔌 开始加载 L1 插件 (共{len(plugins_config)}个)...")
            
            for plugin_cfg in plugins_config:
                module_id = plugin_cfg.get('module_id')
                enabled = plugin_cfg.get('enabled', True)
                
                if not enabled:
                    logger.info(f"⏭️  跳过插件：{module_id} (disabled)")
                    continue
                
                try:
                    self._load_single_plugin(plugin_cfg)
                    loaded_count += 1
                    logger.info(f"✅ 插件加载成功：{module_id}")
                except Exception as e:
                    logger.error(f"❌ 插件加载失败：{module_id} - {e}", exc_info=True)
            
            logger.info(f"✅ L1 插件加载完成：{loaded_count}/{len(plugins_config)}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"❌ 加载插件配置失败：{e}", exc_info=True)
            return 0
    
    def _load_single_plugin(self, plugin_cfg: Dict):
        """
        加载单个插件
        
        Args:
            plugin_cfg: 插件配置字典
        """
        module_id = plugin_cfg['module_id']
        class_path = plugin_cfg['class_path']
        priority_str = plugin_cfg.get('priority', 'NORMAL')
        config = plugin_cfg.get('config', {})
        
        # 1. 动态导入类
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        plugin_class: Type[IL1Module] = getattr(module, class_name)
        
        # 2. 实例化插件
        plugin_instance = plugin_class(config=config)
        
        # 3. 验证接口
        if not isinstance(plugin_instance, IL1Module):
            raise TypeError(f"插件 {module_id} 未实现 IL1Module 接口")
        
        # 4. 初始化插件
        if not plugin_instance.initialize(self._shared_memory):
            raise RuntimeError(f"插件 {module_id} 初始化失败")
        
        # 5. 注册插件
        self._plugins[module_id] = plugin_instance
        self._plugin_configs[module_id] = plugin_cfg
        self._error_counts[module_id] = 0
        
        logger.info(f"🔌 插件已注册：{module_id} (优先级：{priority_str})")
    
    def unload_plugin(self, module_id: str) -> bool:
        """
        卸载单个插件
        
        Args:
            module_id: 模块 ID
        
        Returns:
            bool: 是否成功卸载
        """
        if module_id not in self._plugins:
            logger.warning(f"⚠️  插件不存在：{module_id}")
            return False
        
        try:
            plugin = self._plugins[module_id]
            plugin.shutdown()
            
            del self._plugins[module_id]
            del self._plugin_configs[module_id]
            
            logger.info(f"✅ 插件已卸载：{module_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 卸载插件失败：{module_id} - {e}", exc_info=True)
            return False
    
    def unload_all(self):
        """卸载所有插件"""
        logger.info(f"🔄 正在卸载所有插件 (共{len(self._plugins)}个)...")
        
        for module_id in list(self._plugins.keys()):
            self.unload_plugin(module_id)
        
        logger.info("✅ 所有插件已卸载")
    
    # ========== 运行周期管理 ==========
    
    def run_cycle(self, shared_memory: Optional[Dict] = None) -> List[ZulongEvent]:
        """
        执行单周期处理 (核心方法)
        
        流程:
        1. 按优先级排序插件
        2. 依次调用每个插件的 process_cycle
        3. 收集所有事件
        4. 异常隔离 (单个插件崩溃不影响其他插件)
        5. 定期健康检查
        
        Args:
            shared_memory: 共享内存 (可选，默认使用内部共享内存)
        
        Returns:
            List[ZulongEvent]: 本周期产生的所有事件
        
        TSD v1.7 对应:
        - 3.2 智能路由逻辑 (优先级调度)
        - 4.1 L1-A 受控反射引擎 (快速响应)
        """
        if shared_memory:
            self._shared_memory = shared_memory
        
        all_events: List[ZulongEvent] = []
        start_time = time.time()
        
        # 1. 按优先级排序插件 (CRITICAL > HIGH > NORMAL > LOW)
        sorted_plugins = sorted(
            self._plugins.values(),
            key=lambda p: p.priority.value,
            reverse=True
        )
        
        # 2. 依次执行每个插件
        for plugin in sorted_plugins:
            module_id = plugin.module_id
            cycle_start = time.time()
            
            try:
                # 🔥 关键：异常隔离
                events = plugin.process_cycle(self._shared_memory)
                
                # 验证事件格式
                for event in events:
                    if self._validate_event(event):
                        all_events.append(event)
                    else:
                        logger.warning(f"⚠️  插件 {module_id} 产生无效事件")
                
                # 记录性能
                cycle_time = time.time() - cycle_start
                self._cycle_times[module_id] = cycle_time
                
                # 性能警告
                if cycle_time > 0.01:  # >10ms
                    logger.warning(f"⚠️  插件 {module_id} 执行过慢：{cycle_time*1000:.2f}ms")
                
            except Exception as e:
                # 🔥 关键：异常隔离 (单个插件崩溃不影响其他插件)
                self._error_counts[module_id] += 1
                logger.error(
                    f"❌ 插件 {module_id} 执行失败：{e} "
                    f"(错误次数：{self._error_counts[module_id]})",
                    exc_info=True
                )
                
                # 产生错误事件
                error_event = ZulongEvent(
                    type=EventType.SYSTEM_STATE_CHANGE,
                    priority=EventPriority.HIGH,
                    source=module_id,
                    payload={
                        "error": str(e),
                        "error_count": self._error_counts[module_id]
                    }
                )
                all_events.append(error_event)
        
        # 3. 定期健康检查
        current_time = time.time()
        if current_time - self._last_health_check > self._health_check_interval:
            self._run_health_check()
            self._last_health_check = current_time
        
        # 4. 性能统计
        self._cycle_count += 1
        total_cycle_time = time.time() - start_time
        
        if self._cycle_count % 100 == 0:
            logger.debug(
                f"📊 周期 #{self._cycle_count} 完成 | "
                f"插件数：{len(self._plugins)} | "
                f"事件数：{len(all_events)} | "
                f"耗时：{total_cycle_time*1000:.2f}ms"
            )
        
        return all_events
    
    def _validate_event(self, event: ZulongEvent) -> bool:
        """验证事件格式"""
        from .interface import validate_event
        return validate_event(event)
    
    def _run_health_check(self):
        """执行健康检查"""
        logger.info("🏥 执行插件健康检查...")
        
        for module_id, plugin in self._plugins.items():
            try:
                health = plugin.health_check()
                status = health.get('status', 'UNKNOWN')
                
                if status == 'ERROR':
                    logger.error(f"❌ 插件 {module_id} 健康检查失败：{health}")
                elif status == 'WARNING':
                    logger.warning(f"⚠️  插件 {module_id} 健康警告：{health}")
                else:
                    logger.debug(f"✅ 插件 {module_id} 健康状态：OK")
                
            except Exception as e:
                logger.error(f"❌ 插件 {module_id} 健康检查异常：{e}", exc_info=True)
    
    # ========== 事件路由与回调 ==========
    
    def dispatch_event(self, event: ZulongEvent):
        """
        分发事件到相关插件
        
        当其他层 (L0, L1-B, L2) 发送事件给 L1 时调用
        
        Args:
            event: 事件对象
        
        TSD v1.7 对应:
        - 3.2 智能路由逻辑
        """
        for plugin in self._plugins.values():
            try:
                # 插件可选择性处理事件
                plugin.on_event(event, self._shared_memory)
            except Exception as e:
                logger.error(
                    f"❌ 插件 {plugin.module_id} 事件处理失败：{e}",
                    exc_info=True
                )
    
    # ========== 共享内存管理 ==========
    
    def get_shared_memory(self) -> Dict:
        """获取共享内存"""
        return self._shared_memory
    
    def set_shared_value(self, key: str, value: Any):
        """
        设置共享内存值
        
        用法:
        manager.set_shared_value("motor.speed", 0.5)
        manager.set_shared_value("obstacle.distance", 1.2)
        """
        self._shared_memory[key] = value
    
    def get_shared_value(self, key: str, default: Any = None) -> Any:
        """获取共享内存值"""
        return self._shared_memory.get(key, default)
    
    # ========== 监控与管理接口 ==========
    
    def get_plugin_list(self) -> List[Dict]:
        """
        获取插件列表
        
        Returns:
            List[Dict]: 插件信息列表
        """
        result = []
        for module_id, plugin in self._plugins.items():
            result.append({
                "module_id": module_id,
                "class_name": plugin.__class__.__name__,
                "priority": plugin.priority.name,
                "status": plugin.health_check().get('status', 'UNKNOWN'),
                "cycle_time_ms": self._cycle_times.get(module_id, 0) * 1000,
                "error_count": self._error_counts.get(module_id, 0)
            })
        return result
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "total_plugins": len(self._plugins),
            "cycle_count": self._cycle_count,
            "total_errors": sum(self._error_counts.values()),
            "avg_cycle_time_ms": (
                sum(self._cycle_times.values()) / len(self._cycle_times) * 1000
                if self._cycle_times else 0
            )
        }
    
    def start(self):
        """启动插件管理器"""
        self._running = True
        logger.info("🚀 插件管理器已启动")
    
    def stop(self):
        """停止插件管理器"""
        self._running = False
        self.unload_all()
        logger.info("🛑 插件管理器已停止")


# ========== 全局单例 (可选) ==========

# 如果需要全局单例，可以这样使用:
# plugin_manager = PluginManager()
