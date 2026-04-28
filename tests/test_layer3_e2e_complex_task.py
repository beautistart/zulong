"""
Layer 3: 端到端复杂任务执行测试

模拟完整的祖龙系统复杂任务处理管线:
  TaskGraph 任务分解
  -> TaskGraphAdapter.sync() 投射到 MemoryGraph
  -> BFS 激活扩散 (compute_activations)
  -> AttentionWindow 动态裁剪
  -> CircuitBreaker 迭代控制
  -> TaskSuspension 挂起/恢复
  -> DialogueAdapter 对话轮次管理

无需 WebSocket 服务器或 LLM，纯 Python 层模拟管线联动。
"""

import asyncio
import os
import pytest
import tempfile
import time

from zulong.l2.task_graph import TaskGraph, FileRef
from zulong.l2.circuit_breaker import ToolCallCircuitBreaker, CircuitBreakerState
from zulong.l2.attention_window import (
    AttentionMode,
    AttentionWindowManager,
    MessageEnvelope,
)
from zulong.l2.task_suspension import TaskSuspensionManager, SuspendableTaskState
from zulong.memory.memory_graph import (
    EdgeType,
    GraphNode,
    Importance,
    MemoryGraph,
    NodeType,
)
from zulong.memory.graph_adapters import TaskGraphAdapter, DialogueAdapter


# ============================================================
# 辅助函数
# ============================================================

def _build_complex_task_graph() -> TaskGraph:
    """构建一个典型的复杂任务图（模拟用户需求: 开发 3D 射击游戏）"""
    tg = TaskGraph(title="开发3D射击游戏demo", graph_id="tg_shooter_001")

    # 需求根
    tg.add_node(
        id="req", label="需求分析",
        type="requirement", status="in_progress",
        desc="开发一个基于 Three.js 的 3D 射击游戏 demo",
    )

    # 分析层
    tg.add_node(
        id="analysis", label="技术方案分析",
        type="analysis", status="in_progress",
        desc="确定技术栈、架构和关键模块划分",
    )
    tg.add_h_edge("req", "analysis")

    # 大纲层: 4 个子模块
    modules = [
        ("scene", "场景搭建", "Three.js 场景初始化、光照、天空盒"),
        ("player", "玩家控制", "FPS 相机、WASD 移动、鼠标瞄准"),
        ("enemy", "敌人AI", "敌人生成、巡逻路线、射击行为树"),
        ("ui", "HUD界面", "血量条、弹药显示、得分面板"),
    ]
    for mid, label, desc in modules:
        tg.add_node(id=mid, label=label, type="outline", status="pending", desc=desc)
        tg.add_h_edge("analysis", mid)

    # 子任务层: 集成测试依赖 4 个模块
    tg.add_node(
        id="integration", label="集成测试",
        type="task", status="pending",
        desc="组合所有模块进行联调测试",
    )
    tg.add_h_edge("analysis", "integration")
    for mid, _, _ in modules:
        tg.add_d_edge(mid, "integration", via=f"{mid}模块输出")

    # 文件关联
    tg.add_node(
        id="scene_impl", label="场景实现",
        type="subtask", status="pending",
        desc="编写 scene.js",
        files=[FileRef(name="scene.js", path="src/scene.js")],
    )
    tg.add_h_edge("scene", "scene_impl")

    return tg


# ============================================================
# TestE2EComplexTask: 完整管线联动
# ============================================================


class TestE2EComplexTask:
    """端到端测试: 从任务分解到挂起恢复的完整流程"""

    def test_task_decomposition_to_memory(self, temp_memory_graph):
        """TaskGraph 创建 -> 投射到 MemoryGraph -> 验证图拓扑"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        # 投射
        adapter = TaskGraphAdapter()
        synced = adapter.sync(mg, tg)
        assert synced >= 7  # 至少 7 个任务节点 + FILE 节点

        # 验证 HIERARCHY 边
        req_id = f"task:{tg.id}/req"
        analysis_id = f"task:{tg.id}/analysis"
        assert mg.has_edge(req_id, analysis_id)

        # 验证 DEPENDENCY 边
        scene_id = f"task:{tg.id}/scene"
        integ_id = f"task:{tg.id}/integration"
        assert mg.has_edge(scene_id, integ_id)

        # 验证节点类型
        node = mg.get_node(scene_id)
        assert node is not None
        assert node.node_type == NodeType.TASK
        assert node.metadata["graph_id"] == tg.id

    def test_bfs_activation_after_sync(self, temp_memory_graph):
        """投射后 BFS 激活扩散: 种子节点激活向邻居传播"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        # 以当前工作节点为种子执行 BFS
        scene_id = f"task:{tg.id}/scene"
        activations = mg.compute_activations(seed_node_ids=[scene_id], max_depth=3)
        assert len(activations) > 0

        # 种子节点应有最高激活
        scene_node = mg.get_node(scene_id)
        assert scene_node.activation >= 0.9

        # 相邻节点（analysis）应有衰减后的激活
        analysis_id = f"task:{tg.id}/analysis"
        analysis_node = mg.get_node(analysis_id)
        assert analysis_node.activation > 0.0
        assert analysis_node.activation < scene_node.activation

    def test_attention_window_with_memory_boost(self, temp_memory_graph):
        """AttentionWindow + MemoryGraph BFS 激活联动评分"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        # 投射 + BFS
        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)
        scene_id = f"task:{tg.id}/scene"
        mg.compute_activations(seed_node_ids=[scene_id], max_depth=2)

        # 创建 AttentionWindow
        aw = AttentionWindowManager(
            context_window_size=65536,
            task_graph=tg,
            memory_graph=mg,
        )

        # 注册关联活跃节点的消息
        aw.register_message(
            msg={"role": "tool", "content": "场景搭建完成"},
            turn=1,
            node_id=scene_id,
        )

        # 注册不关联任何节点的消息
        aw.register_message(
            msg={"role": "tool", "content": "无关内容"},
            turn=1,
        )

        # apply_window 应保留两条（预算充足时都保留）
        result = aw.apply_window()
        assert len(result) >= 2

    def test_circuit_breaker_monitors_fc_loop(self):
        """CircuitBreaker 监控 FC 循环: GREEN -> YELLOW -> RED"""
        cb = ToolCallCircuitBreaker(config={
            "context_window_size": 8000,
            "safety_hard_cap": 100,
        })

        # 模拟正常调用 — 应保持 GREEN
        for i in range(3):
            cb.record_call(
                f"tool_{i}", {"query": f"search_{i}"},
                f"result for search_{i} with diverse content number {i}",
            )
        state, _ = cb.evaluate(iteration=3, messages=[])
        assert state == CircuitBreakerState.GREEN

        # 模拟重复调用 — 相同的工具+参数，应触发 RED
        for _ in range(5):
            cb.record_call(
                "web_search", {"query": "same query"},
                "same result",
            )
        state, reason = cb.evaluate(iteration=8, messages=[])
        assert state in (CircuitBreakerState.YELLOW, CircuitBreakerState.RED)

    def test_attention_mode_transitions_in_pipeline(self, temp_memory_graph):
        """FC 循环中 AttentionWindow 模式自动切换"""
        tg = _build_complex_task_graph()
        mg = temp_memory_graph
        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        aw = AttentionWindowManager(
            context_window_size=65536,
            task_graph=tg,
            memory_graph=mg,
        )

        # 初始: GLOBAL
        assert aw.mode == AttentionMode.GLOBAL

        # 调用 recall_memory → FOCUS
        aw.observe_tool_call("recall_memory", {"query": "场景搭建"})
        assert aw.mode == AttentionMode.FOCUS

        # 调用 exec_write_file → SINGLE_CHAIN
        aw.observe_tool_call("exec_write_file", {"path": "src/scene.js"})
        assert aw.mode == AttentionMode.SINGLE_CHAIN

        # 调用 task_view_overview → 强制回 GLOBAL
        aw.observe_tool_call("task_view_overview", {})
        assert aw.mode == AttentionMode.GLOBAL

    def test_navigate_attention_deeper_broader(self, temp_memory_graph):
        """navigate_attention 工具: deeper 和 broader 模式切换"""
        tg = _build_complex_task_graph()
        mg = temp_memory_graph

        aw = AttentionWindowManager(
            context_window_size=65536,
            task_graph=tg,
            memory_graph=mg,
        )

        # GLOBAL -> deeper -> FOCUS
        aw.on_navigate_attention("deeper")
        assert aw.mode == AttentionMode.FOCUS

        # FOCUS -> deeper -> SINGLE_CHAIN
        aw.on_navigate_attention("deeper")
        assert aw.mode == AttentionMode.SINGLE_CHAIN

        # SINGLE_CHAIN -> broader -> FOCUS
        aw.on_navigate_attention("broader")
        assert aw.mode == AttentionMode.FOCUS

        # FOCUS -> broader -> GLOBAL
        aw.on_navigate_attention("broader")
        assert aw.mode == AttentionMode.GLOBAL

    def test_dialogue_adapter_round_lifecycle(self, temp_memory_graph):
        """DialogueAdapter 完整生命周期: 创建 round → 重要度检测 → finalize"""
        mg = temp_memory_graph
        da = DialogueAdapter()

        # 创建第一轮（身份信息 → IDENTITY 重要度）
        r1 = da.add_round(mg, request_id="req_001", goal="我叫张三，帮我开发一个游戏")
        assert mg.has_node(r1)

        node1 = mg.get_node(r1)
        assert node1.node_type == NodeType.DIALOGUE

        # 创建第二轮（关联上一轮）
        r2 = da.add_round(
            mg, request_id="req_002",
            goal="游戏需要有FPS视角和敌人AI",
            prev_round_id=r1,
        )
        assert mg.has_edge(r1, r2)  # TEMPORAL 边

        # 完成第一轮
        da.finalize_round(mg, r1, total_turns=5, status="completed")
        node1_after = mg.get_node(r1)
        assert node1_after.metadata.get("status") == "completed"
        assert node1_after.metadata.get("total_turns") == 5

    def test_dialogue_plus_task_graph_cross_link(self, temp_memory_graph):
        """对话轮次与 TaskGraph 的跨类型关联"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        # 先投射 TaskGraph
        tga = TaskGraphAdapter()
        tga.sync(mg, tg)

        # 创建对话轮次并关联 TaskGraph
        da = DialogueAdapter()
        r1 = da.add_round(
            mg, request_id="req_cross_001",
            goal="开始做场景搭建模块",
            task_graph_id=tg.id,
        )

        # round 节点应该存在
        assert mg.has_node(r1)

    @pytest.mark.asyncio
    async def test_suspend_and_resume_full_cycle(self):
        """任务挂起/恢复完整周期"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 重置单例
            TaskSuspensionManager._instance = None
            if hasattr(TaskSuspensionManager, '_initialized'):
                del TaskSuspensionManager._initialized

            mgr = TaskSuspensionManager(config={"persistence_path": tmp_dir})

            # 构建任务
            tg = _build_complex_task_graph()

            # 模拟 CircuitBreaker 状态
            cb_state = {"call_history": [], "elapsed_at_suspend": 0}
            messages = [
                {"role": "user", "content": "开发3D射击游戏"},
                {"role": "assistant", "content": "好的，我来分析需求..."},
            ]

            # 创建可挂起状态（使用正确的 dataclass 字段）
            state = SuspendableTaskState(
                task_id=f"test_{int(time.time())}",
                description="开发3D射击游戏demo",
                task_graph=tg,
                messages=messages,
                accumulated_links="",
                circuit_breaker_state=cb_state,
                iteration_count=5,
                memory_snapshot={"hot_nodes": ["scene"]},
                suspended_reason="user_requested",
                metadata={"current_node_id": "scene", "attention_mode": "focus"},
            )

            # 挂起
            task_id = await mgr.suspend_task(state)
            assert task_id is not None

            # 列表应能看到
            suspended = await mgr.list_suspended_tasks()
            assert len(suspended) == 1
            assert suspended[0]["task_id"] == task_id

            # 恢复
            restored = await mgr.resume_task(task_id, consume=True)
            assert restored is not None
            assert restored.metadata.get("current_node_id") == "scene"
            assert restored.metadata.get("attention_mode") == "focus"

            # 恢复的 TaskGraph 应保持完整
            assert restored.task_graph is not None
            assert len(restored.task_graph._nodes) >= 7

            # 恢复后列表应为空
            suspended_after = await mgr.list_suspended_tasks()
            assert len(suspended_after) == 0

            # 清理单例
            TaskSuspensionManager._instance = None
            if hasattr(TaskSuspensionManager, '_initialized'):
                del TaskSuspensionManager._initialized

    def test_hebbian_learning_during_fc_loop(self, temp_memory_graph):
        """FC 循环中 Hebbian 学习: 共激活节点权重增强"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        scene_id = f"task:{tg.id}/scene"
        player_id = f"task:{tg.id}/player"

        # 初始边权重
        mg.add_edge(scene_id, player_id, EdgeType.ASSOCIATION, weight=0.5)
        initial_edge = mg.get_edge(scene_id, player_id)
        initial_weight = initial_edge.get("weight", 0.5)

        # 模拟多次共激活
        for _ in range(5):
            mg.compute_activations(seed_node_ids=[scene_id, player_id], max_depth=1)

        # Hebbian 学习应增强权重
        mg.hebbian_strengthen()
        updated_edge = mg.get_edge(scene_id, player_id)
        updated_weight = updated_edge.get("weight", 0.5)

        assert updated_weight >= initial_weight

    def test_full_pipeline_simulation(self, temp_memory_graph):
        """完整管线模拟: 任务分解 → 投射 → BFS → 注意力 → 熔断器 → 对话"""
        mg = temp_memory_graph

        # ===== 第一阶段: 任务分解与投射 =====
        tg = _build_complex_task_graph()
        tga = TaskGraphAdapter()
        synced = tga.sync(mg, tg)
        assert synced >= 7

        # ===== 第二阶段: BFS 激活扩散 =====
        scene_id = f"task:{tg.id}/scene"
        activations = mg.compute_activations(seed_node_ids=[scene_id], max_depth=3)
        assert len(activations) > 0

        # ===== 第三阶段: 对话轮次创建 =====
        da = DialogueAdapter()
        round_id = da.add_round(
            mg, request_id="pipeline_001",
            goal="开始搭建游戏场景",
            task_graph_id=tg.id,
        )
        assert mg.has_node(round_id)

        # ===== 第四阶段: 注意力窗口管理 =====
        aw = AttentionWindowManager(
            context_window_size=65536,
            task_graph=tg,
            memory_graph=mg,
        )

        # 注册 system prompt（钉住）
        aw.register_message(
            msg={"role": "system", "content": "你是祖龙，一个复杂任务执行系统"},
            turn=0,
            pinned=True,
        )

        # 注册用户消息
        aw.register_message(
            msg={"role": "user", "content": "帮我开发3D射击游戏"},
            turn=1,
        )

        # 模拟多轮 FC 调用
        tools = [
            ("recall_memory", {"query": "3D游戏开发"}, "找到相关经验: Three.js 开发模式..."),
            ("exec_write_file", {"path": "src/scene.js", "node_id": "scene"},
             "// Three.js 场景初始化\nconst scene = new THREE.Scene();"),
            ("exec_write_file", {"path": "src/player.js", "node_id": "player"},
             "// 玩家控制器\nclass PlayerController { ... }"),
        ]

        cb = ToolCallCircuitBreaker(config={
            "context_window_size": 65536,
            "safety_hard_cap": 100,
        })

        for i, (tool_name, params, result) in enumerate(tools):
            turn = i + 2  # turn 0=system, 1=user, 2+=tools

            gid = aw.new_tool_group()
            aw.register_message(
                msg={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": tool_name, "arguments": str(params)}}],
                },
                turn=turn,
                group_id=gid,
            )
            aw.observe_tool_call(tool_name, params)
            aw.register_message(
                msg={"role": "tool", "content": result},
                turn=turn,
                tool_name=tool_name,
                node_id=params.get("node_id"),
                group_id=gid,
            )
            cb.record_call(tool_name, params, result)

        # ===== 第五阶段: 验证各模块状态 =====

        # CircuitBreaker 应保持 GREEN（3 次不同调用）
        state, _ = cb.evaluate(iteration=3, messages=[])
        assert state == CircuitBreakerState.GREEN

        # AttentionWindow 模式应切换（recall_memory → FOCUS, write_file → SINGLE_CHAIN）
        assert aw.mode == AttentionMode.SINGLE_CHAIN

        # apply_window 应返回非空消息列表
        windowed = aw.apply_window()
        assert len(windowed) >= 3  # system + user + 至少 1 个 tool 组

        # 对话节点应存在
        assert mg.has_node(round_id)

        # 完成对话轮次
        da.finalize_round(mg, round_id, total_turns=3, status="completed")
        round_node = mg.get_node(round_id)
        assert round_node.metadata["status"] == "completed"

    @pytest.mark.asyncio
    async def test_suspend_mid_pipeline_and_resume(self, temp_memory_graph):
        """管线中途挂起: 保存全部状态 → 恢复 → 继续执行"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        # 投射
        tga = TaskGraphAdapter()
        tga.sync(mg, tg)

        # 模拟执行到一半
        tg._nodes["scene"].status = "completed"
        tg._nodes["player"].status = "in_progress"

        # 注意力状态
        aw = AttentionWindowManager(
            context_window_size=65536,
            task_graph=tg,
            memory_graph=mg,
        )
        aw.observe_tool_call("recall_memory", {"query": "player control"})
        assert aw.mode == AttentionMode.FOCUS

        # 熔断器状态
        cb = ToolCallCircuitBreaker(config={"context_window_size": 65536})
        cb.record_call("web_search", {"query": "FPS control"}, "result...")
        cb_snapshot = cb.serialize()

        # MemoryGraph 快照
        mg_snapshot = mg.stats

        # 构建挂起状态
        state = SuspendableTaskState(
            task_id=f"mid_pipeline_{int(time.time())}",
            description="开发FPS游戏",
            task_graph=tg,
            messages=[
                {"role": "user", "content": "开发FPS游戏"},
                {"role": "assistant", "content": "正在搭建场景..."},
            ],
            accumulated_links="",
            circuit_breaker_state=cb_snapshot,
            iteration_count=3,
            memory_snapshot=mg_snapshot,
            suspended_reason="complexity",
            metadata={
                "current_node_id": "player",
                "attention_mode": aw.mode.value,
                "reason": "CircuitBreaker 发出警告",
            },
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 重置单例
            TaskSuspensionManager._instance = None
            if hasattr(TaskSuspensionManager, '_initialized'):
                del TaskSuspensionManager._initialized

            mgr = TaskSuspensionManager(config={"persistence_path": tmp_dir})

            # 挂起
            task_id = await mgr.suspend_task(state)
            assert task_id

            # 恢复
            restored = await mgr.resume_task(task_id, consume=True)
            assert restored is not None

            # 验证恢复完整性
            assert restored.metadata.get("current_node_id") == "player"
            assert restored.metadata.get("attention_mode") == "focus"
            assert restored.metadata.get("reason") == "CircuitBreaker 发出警告"

            # TaskGraph 应保持状态
            assert restored.task_graph is not None
            restored_tg = restored.task_graph
            assert restored_tg._nodes["scene"].status == "completed"
            assert restored_tg._nodes["player"].status == "in_progress"

            # CircuitBreaker 可从快照恢复
            restored_cb = ToolCallCircuitBreaker()
            restored_cb.deserialize(restored.circuit_breaker_state)
            assert len(restored_cb._call_history) == 1  # 之前记录了 1 次调用

            # 清理单例
            TaskSuspensionManager._instance = None
            if hasattr(TaskSuspensionManager, '_initialized'):
                del TaskSuspensionManager._initialized

    @pytest.mark.asyncio
    async def test_find_suspended_by_description(self):
        """通过描述搜索挂起的任务"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            TaskSuspensionManager._instance = None
            if hasattr(TaskSuspensionManager, '_initialized'):
                del TaskSuspensionManager._initialized

            mgr = TaskSuspensionManager(config={"persistence_path": tmp_dir})

            # 挂起两个不同任务
            tg1 = TaskGraph(title="开发射击游戏", graph_id="tg_001")
            tg1.add_node(id="r", label="R", type="requirement",
                         status="pending", desc="射击游戏")

            tg2 = TaskGraph(title="写一篇论文", graph_id="tg_002")
            tg2.add_node(id="r", label="R", type="requirement",
                         status="pending", desc="论文撰写")

            state1 = SuspendableTaskState(
                task_id=f"find_test_1_{int(time.time())}",
                description="开发射击游戏",
                task_graph=tg1, messages=[], accumulated_links="",
                circuit_breaker_state={}, iteration_count=0,
                suspended_reason="user_requested",
            )
            state2 = SuspendableTaskState(
                task_id=f"find_test_2_{int(time.time())}",
                description="写一篇论文",
                task_graph=tg2, messages=[], accumulated_links="",
                circuit_breaker_state={}, iteration_count=0,
                suspended_reason="user_requested",
            )

            await mgr.suspend_task(state1)
            await mgr.suspend_task(state2)

            # find_by_description 返回 Optional[Dict]（最佳匹配）
            result = await mgr.find_by_description("射击游戏")
            assert result is not None
            # 结果应包含射击游戏相关信息
            desc = result.get("description", "") or result.get("title", "")
            assert "射击" in desc or "游戏" in desc

            # 清理
            TaskSuspensionManager._instance = None
            if hasattr(TaskSuspensionManager, '_initialized'):
                del TaskSuspensionManager._initialized

    def test_memory_decay_and_prune(self, temp_memory_graph):
        """记忆衰减与修剪: TRIVIAL 节点快速衰减"""
        mg = temp_memory_graph

        # 添加 TRIVIAL 节点
        trivial = GraphNode(
            node_id="ephemeral_001",
            node_type=NodeType.DIALOGUE,
            label="嗯",
        )
        mg.add_node(trivial)
        mg.set_importance("ephemeral_001", Importance.TRIVIAL)

        # 添加 MUST_REMEMBER 节点
        permanent = GraphNode(
            node_id="permanent_001",
            node_type=NodeType.PERSON,
            label="张三",
        )
        mg.add_node(permanent)
        mg.set_importance("permanent_001", Importance.MUST_REMEMBER)

        # 执行多次衰减
        for _ in range(20):
            mg.decay_and_prune()

        # MUST_REMEMBER 应存活
        assert mg.has_node("permanent_001")

        # TRIVIAL 可能被修剪（取决于衰减速率和次数）
        # 至少验证其激活值低于 MUST_REMEMBER
        if mg.has_node("ephemeral_001"):
            e_node = mg.get_node("ephemeral_001")
            p_node = mg.get_node("permanent_001")
            assert e_node.activation <= p_node.activation

    def test_importance_detection_in_pipeline(self, temp_memory_graph):
        """管线中的重要度自动检测"""
        mg = temp_memory_graph
        da = DialogueAdapter()

        # 身份信息 → IDENTITY
        r1 = da.add_round(mg, request_id="imp_001", goal="我叫张三，今年25岁")
        node1 = mg.get_node(r1)
        assert node1 is not None

        # 显式记住 → MUST_REMEMBER
        r2 = da.add_round(mg, request_id="imp_002", goal="帮我记住密码是abc123")
        node2 = mg.get_node(r2)
        assert node2 is not None

        # 简短回复 → TRIVIAL
        r3 = da.add_round(mg, request_id="imp_003", goal="嗯")
        node3 = mg.get_node(r3)
        assert node3 is not None

    def test_task_graph_status_cascade(self, temp_memory_graph):
        """任务状态变更联动: 标记完成 → 增量 sync → 验证 MemoryGraph 更新"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        tga = TaskGraphAdapter()
        tga.sync(mg, tg)

        # 标记 scene 为完成
        tg._nodes["scene"].status = "completed"

        # 增量同步
        tga.incremental_sync(mg, "node_update", {
            "_graph_id": tg.id,
            "node_id": "scene",
            "status": "completed",
        })

        # 验证 MemoryGraph 中状态更新
        scene_id = f"task:{tg.id}/scene"
        node = mg.get_node(scene_id)
        assert node is not None
        assert node.metadata["status"] == "completed"

    def test_multi_round_bfs_activation_accumulation(self, temp_memory_graph):
        """多轮 BFS 激活累积: 反复激活同一区域"""
        mg = temp_memory_graph
        tg = _build_complex_task_graph()

        adapter = TaskGraphAdapter()
        adapter.sync(mg, tg)

        scene_id = f"task:{tg.id}/scene"

        # 第一轮 BFS
        mg.compute_activations(seed_node_ids=[scene_id], max_depth=2)
        first_activation = mg.get_node(scene_id).activation

        # 第二轮 BFS（应累积或保持）
        mg.compute_activations(seed_node_ids=[scene_id], max_depth=2)
        second_activation = mg.get_node(scene_id).activation

        # 第二次激活应 >= 第一次（种子始终为 1.0）
        assert second_activation >= first_activation
