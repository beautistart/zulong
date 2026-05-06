"""
ModuleManager — 模块注册、拓扑排序、生命周期编排

职责：
- register(module)     注册模块
- launch(mode, on_progress)  按模式启动
- start_module(name)   运行时启动可选模块
- stop_module(name)    运行时停止可选模块
- get_status()         全局状态快照
- shutdown()           逆序停止所有模块
"""

import asyncio
import logging
from collections import deque
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from zulong.launcher.module_base import Module, ModuleState

logger = logging.getLogger(__name__)

# on_progress 回调签名:
#   async def callback(step: int, total: int, module_name: str, display: str, message: str)
ProgressCallback = Callable[..., Coroutine[Any, Any, None]]


class ModuleManager:
    """模块生命周期管理器"""

    def __init__(self):
        self._modules: Dict[str, Module] = {}
        # 共享上下文：模块间传递单例引用（如 InferenceEngine 实例）
        self.context: Dict[str, Any] = {}
        self._launched = False
        self._mode: Optional[str] = None

    # ── 注册 ──────────────────────────────────────────

    def register(self, module: Module) -> None:
        if module.name in self._modules:
            logger.warning(f"[ModuleManager] 模块 {module.name} 已注册，跳过")
            return
        module.set_context(self.context)
        self._modules[module.name] = module
        logger.debug(f"[ModuleManager] 已注册: {module.name} ({module.display_name})")

    # ── 拓扑排序 ──────────────────────────────────────

    def _topo_sort(self, names: Set[str]) -> List[str]:
        """Kahn 算法对指定模块名做拓扑排序，返回启动顺序"""
        # 构建子图
        in_degree: Dict[str, int] = {n: 0 for n in names}
        adj: Dict[str, List[str]] = {n: [] for n in names}
        for n in names:
            mod = self._modules[n]
            for dep in mod.dependencies:
                if dep in names:
                    adj[dep].append(n)
                    in_degree[n] += 1

        queue = deque(n for n in names if in_degree[n] == 0)
        result: List[str] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(names):
            missing = names - set(result)
            raise RuntimeError(f"模块存在循环依赖: {missing}")
        return result

    # ── 按模式启动 ────────────────────────────────────

    async def launch(
        self,
        mode: str,
        on_progress: Optional[ProgressCallback] = None,
    ) -> None:
        """按模式启动模块组

        Args:
            mode: "full" 或 "ide"
            on_progress: 进度回调 (step, total, module_name, display, message)
        """
        self._mode = mode

        # 筛选需要启动的模块：core 一定启动；full 仅 Full 模式
        target_tags = {"core"}
        if mode == "full":
            target_tags.add("full")

        names: Set[str] = set()
        for mod in self._modules.values():
            if mod.mode_tags & target_tags:
                names.add(mod.name)

        # 补全依赖（依赖链中的模块也必须启动）
        changed = True
        while changed:
            changed = False
            for n in list(names):
                for dep in self._modules[n].dependencies:
                    if dep not in names and dep in self._modules:
                        names.add(dep)
                        changed = True

        order = self._topo_sort(names)
        total = len(order)
        logger.info(f"[ModuleManager] 启动模式={mode}, 模块数={total}, 顺序={order}")

        for step, name in enumerate(order, 1):
            mod = self._modules[name]
            mod.progress_message = f"正在启动 {mod.display_name}..."
            mod.state = ModuleState.STARTING

            if on_progress:
                await on_progress(step, total, name, mod.display_name, mod.progress_message)

            try:
                await mod.start()
                mod.state = ModuleState.RUNNING
                mod.progress_message = f"{mod.display_name} 已就绪"
                logger.info(f"[ModuleManager] [{step}/{total}] {mod.display_name} -> RUNNING")
            except Exception as e:
                mod.state = ModuleState.ERROR
                mod.error_message = str(e)
                mod.progress_message = f"{mod.display_name} 启动失败: {e}"
                logger.error(f"[ModuleManager] {mod.display_name} 启动失败: {e}", exc_info=True)
                # 非核心模块失败不阻断启动流程
                if "core" in mod.mode_tags:
                    raise

        self._launched = True
        logger.info(f"[ModuleManager] 所有模块启动完成 (mode={mode})")

    # ── 动态开关 ──────────────────────────────────────

    async def start_module(self, name: str) -> None:
        """运行时启动一个可选模块"""
        if name not in self._modules:
            raise ValueError(f"未注册的模块: {name}")
        mod = self._modules[name]
        if mod.state == ModuleState.RUNNING:
            return

        # 检查依赖
        for dep in mod.dependencies:
            dep_mod = self._modules.get(dep)
            if not dep_mod or dep_mod.state != ModuleState.RUNNING:
                raise RuntimeError(f"依赖模块 {dep} 未运行，无法启动 {name}")

        mod.state = ModuleState.STARTING
        try:
            await mod.start()
            mod.state = ModuleState.RUNNING
            logger.info(f"[ModuleManager] 动态启动: {mod.display_name} -> RUNNING")
        except Exception as e:
            mod.state = ModuleState.ERROR
            mod.error_message = str(e)
            raise

    async def stop_module(self, name: str) -> None:
        """运行时停止一个可选模块"""
        if name not in self._modules:
            raise ValueError(f"未注册的模块: {name}")
        mod = self._modules[name]
        if mod.state not in (ModuleState.RUNNING, ModuleState.ERROR):
            return

        # 检查是否有其他 RUNNING 模块依赖它
        for other in self._modules.values():
            if other.state == ModuleState.RUNNING and name in other.dependencies:
                raise RuntimeError(f"模块 {other.name} 依赖 {name}，无法停止")

        mod.state = ModuleState.STOPPING
        try:
            await mod.stop()
            mod.state = ModuleState.STOPPED
            logger.info(f"[ModuleManager] 动态停止: {mod.display_name} -> STOPPED")
        except Exception as e:
            mod.state = ModuleState.ERROR
            mod.error_message = str(e)
            logger.error(f"[ModuleManager] {mod.display_name} 停止失败: {e}")

    # ── 查询 ──────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        modules = {}
        for mod in self._modules.values():
            modules[mod.name] = mod.to_status()
        return {
            "launched": self._launched,
            "mode": self._mode,
            "modules": modules,
        }

    # ── 全局关闭 ──────────────────────────────────────

    async def shutdown(self) -> None:
        """逆拓扑序停止所有 RUNNING 模块"""
        running = {n for n, m in self._modules.items() if m.state == ModuleState.RUNNING}
        if not running:
            return
        order = self._topo_sort(running)
        order.reverse()
        logger.info(f"[ModuleManager] 开始关闭, 顺序={order}")
        for name in order:
            mod = self._modules[name]
            mod.state = ModuleState.STOPPING
            try:
                await mod.stop()
                mod.state = ModuleState.STOPPED
            except Exception as e:
                mod.state = ModuleState.ERROR
                logger.error(f"[ModuleManager] {mod.display_name} 停止失败: {e}")
        logger.info("[ModuleManager] 所有模块已关闭")
