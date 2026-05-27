"""统一 FC 循环执行器 [已迁移，保留向后兼容]

本文件的逻辑已迁移至:
- zulong.l2.fc_nodes — 节点工厂函数 (4个工厂函数 + 辅助函数)
- zulong.l2.fc_runner — 核心FC循环引擎 (FCRunner + run_fc_loop)

新代码请使用:
    from zulong.l2.fc_runner import FCRunner, run_fc_loop
"""

from zulong.l2.fc_runner import FCRunner as _FCRunner, run_fc_loop

# 向后兼容别名
UnifiedFCRunner = _FCRunner

__all__ = ["UnifiedFCRunner", "run_fc_loop"]
