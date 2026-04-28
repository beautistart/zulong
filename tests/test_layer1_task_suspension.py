"""
Layer 1: TaskSuspension 独立模块测试

覆盖: SuspendableTaskState 序列化、TaskSuspensionManager 挂起/恢复/匹配
"""

import asyncio
import os
import time

import pytest

from zulong.l2.task_suspension import SuspendableTaskState, TaskSuspensionManager
from zulong.l2.task_graph import TaskGraph


# ============================================================
# TestSuspendableTaskState
# ============================================================


class TestSuspendableTaskState:

    def _make_state(self, **kwargs) -> SuspendableTaskState:
        defaults = dict(
            task_id="task_test_001",
            description="测试任务描述",
            messages=[
                {"role": "user", "content": "帮我分析代码"},
                {"role": "assistant", "content": "好的，我来分析"},
            ],
            accumulated_links="",
            circuit_breaker_state={"call_history": [], "planning_mode": False},
            iteration_count=5,
        )
        defaults.update(kwargs)
        return SuspendableTaskState(**defaults)

    def test_create_state(self):
        state = self._make_state()
        assert state.task_id == "task_test_001"
        assert state.description == "测试任务描述"
        assert len(state.messages) == 2
        assert state.iteration_count == 5
        assert state.suspended_reason == "complexity"

    def test_to_dict_excludes_task_graph(self):
        tg = TaskGraph(title="T", graph_id="tg_001")
        state = self._make_state(task_graph=tg)

        d = state.to_dict()
        assert "task_graph" not in d
        assert d["task_id"] == "task_test_001"
        assert d["description"] == "测试任务描述"

    def test_from_dict_roundtrip(self):
        state = self._make_state(
            memory_snapshot={"focus_context": {"focused_task_node_id": "o1"}, "active_node_ids": ["o1"]},
        )
        d = state.to_dict()
        restored = SuspendableTaskState.from_dict(d)

        assert restored.task_id == state.task_id
        assert restored.description == state.description
        assert len(restored.messages) == len(state.messages)
        assert restored.iteration_count == state.iteration_count
        assert restored.memory_snapshot is not None
        assert restored.memory_snapshot["focus_context"]["focused_task_node_id"] == "o1"

    def test_from_dict_deserializes_task_graph(self):
        tg = TaskGraph(title="Restore Test", graph_id="rt_001")
        tg.add_node(id="req", label="R", type="requirement", status="pending", desc="")
        tg.add_node(id="o1", label="O1", type="outline", status="pending", desc="")
        tg.add_h_edge("req", "o1")

        state = self._make_state(
            task_graph=tg,
            task_graph_serialized=tg.serialize(),
        )
        d = state.to_dict()

        restored = SuspendableTaskState.from_dict(d)
        assert restored.task_graph is not None
        assert restored.task_graph.id == "rt_001"
        assert restored.task_graph.get_node("o1") is not None


# ============================================================
# TestTaskSuspensionManager
# ============================================================


class TestTaskSuspensionManager:

    def _make_manager(self, temp_dir) -> TaskSuspensionManager:
        return TaskSuspensionManager(config={
            "enabled": True,
            "persistence_path": temp_dir,
            "max_suspended_tasks": 5,
            "max_age_hours": 72,
        })

    def _make_state(self, task_id="task_001", desc="测试任务"):
        return SuspendableTaskState(
            task_id=task_id,
            description=desc,
            messages=[{"role": "user", "content": "test"}],
            accumulated_links="",
            circuit_breaker_state={},
            iteration_count=10,
        )

    @pytest.mark.asyncio
    async def test_suspend_and_list(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        state = self._make_state()

        task_id = await mgr.suspend_task(state)
        assert task_id == "task_001"

        tasks = await mgr.list_suspended_tasks()
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task_001"

    @pytest.mark.asyncio
    async def test_resume_consume(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        state = self._make_state()
        await mgr.suspend_task(state)

        # 消费式恢复
        restored = await mgr.resume_task("task_001", consume=True)
        assert restored is not None
        assert restored.task_id == "task_001"

        # 文件应已删除
        tasks = await mgr.list_suspended_tasks()
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_resume_no_consume(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        state = self._make_state()
        await mgr.suspend_task(state)

        # 只读式恢复
        restored = await mgr.resume_task("task_001", consume=False)
        assert restored is not None

        # 文件应仍在
        tasks = await mgr.list_suspended_tasks()
        assert len(tasks) == 1

    @pytest.mark.asyncio
    async def test_cancel_task(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        state = self._make_state()
        await mgr.suspend_task(state)

        result = await mgr.cancel_task("task_001")
        assert result is True

        tasks = await mgr.list_suspended_tasks()
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        result = await mgr.cancel_task("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_max_suspended_tasks(self, temp_dir):
        mgr = self._make_manager(temp_dir)

        # 挂起 6 个任务（上限 5）
        for i in range(6):
            s = self._make_state(task_id=f"task_{i:03d}", desc=f"任务{i}")
            s.suspended_at = time.time() + i  # 递增时间
            await mgr.suspend_task(s)

        tasks = await mgr.list_suspended_tasks()
        assert len(tasks) <= 5

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, temp_dir):
        import json as _json
        mgr = self._make_manager(temp_dir)

        # 先正常挂起
        state = self._make_state(task_id="old_task")
        await mgr.suspend_task(state)

        # suspend_task 内部会覆盖 suspended_at 为 time.time()，
        # 所以需要直接修改磁盘文件来模拟过期
        fp = os.path.join(temp_dir, "old_task.json")
        with open(fp, "r", encoding="utf-8") as f:
            data = _json.load(f)
        data["suspended_at"] = time.time() - 80 * 3600  # 80h 前
        with open(fp, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False)

        cleaned = await mgr.cleanup_expired()
        assert cleaned >= 1

        tasks = await mgr.list_suspended_tasks()
        assert len(tasks) == 0

    @pytest.mark.asyncio
    async def test_find_by_description_exact(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        state = self._make_state(task_id="find_001", desc="帮我分析nginx配置")
        await mgr.suspend_task(state)

        found = await mgr.find_by_description("nginx配置")
        assert found is not None
        assert found["task_id"] == "find_001"

    @pytest.mark.asyncio
    async def test_find_by_description_bigram(self, temp_dir):
        mgr = self._make_manager(temp_dir)
        state = self._make_state(task_id="find_002", desc="帮我做一个3D射击游戏")
        await mgr.suspend_task(state)

        found = await mgr.find_by_description("射击游戏")
        assert found is not None
        assert found["task_id"] == "find_002"
