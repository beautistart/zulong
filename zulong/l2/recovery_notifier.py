"""启动恢复通知器

启动时扫描崩溃检查点和挂起任务，通知用户有可恢复的任务。
"""

import json
import logging
import os
import shutil
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = "./data/checkpoints"
SUSPENDED_DIR = "./data/suspended_tasks"


class RecoveryNotifier:
    """启动时扫描可恢复任务并通知用户"""

    _notified = False  # 防止重复通知

    @classmethod
    def check_and_notify(cls):
        """扫描检查点和挂起任务，提升检查点为挂起任务，发送通知"""
        if cls._notified:
            return
        cls._notified = True

        try:
            # 1. 提升检查点：将 checkpoint 文件复制到 suspended_tasks
            cls._promote_checkpoints()
            # 2. 列出所有可恢复任务
            tasks = cls._list_recoverable_tasks()
            if not tasks:
                return
            # 3. 发布通知事件
            cls._notify_user(tasks)
        except Exception as e:
            logger.warning(f"[RecoveryNotifier] 扫描可恢复任务失败: {e}")

    @classmethod
    def _promote_checkpoints(cls):
        """将检查点文件提升为挂起任务（避免修改现有恢复流程）"""
        if not os.path.exists(CHECKPOINT_DIR):
            return

        os.makedirs(SUSPENDED_DIR, exist_ok=True)

        for filename in os.listdir(CHECKPOINT_DIR):
            if not filename.endswith(".json"):
                continue

            ckpt_path = os.path.join(CHECKPOINT_DIR, filename)
            try:
                with open(ckpt_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                task_id = data.get("task_id", "")
                if not task_id:
                    continue

                # 检查 suspended_tasks 中是否已有同 task_id 的文件
                target_path = os.path.join(SUSPENDED_DIR, f"{task_id}.json")
                if os.path.exists(target_path):
                    # 已有挂起任务，删除检查点
                    os.remove(ckpt_path)
                    continue

                # 修改 suspended_reason 为 crash_recovery
                data["suspended_reason"] = "crash_recovery"
                if "metadata" not in data:
                    data["metadata"] = {}
                data["metadata"]["promoted_from_checkpoint"] = True
                data["metadata"]["promoted_at"] = time.time()

                # 写入 suspended_tasks
                temp_path = target_path + ".tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, target_path)

                # 删除原检查点
                os.remove(ckpt_path)
                logger.info(
                    f"[RecoveryNotifier] 检查点提升为挂起任务: {task_id}"
                )
            except Exception as e:
                logger.warning(
                    f"[RecoveryNotifier] 提升检查点失败 {filename}: {e}"
                )

    @classmethod
    def _list_recoverable_tasks(cls) -> List[Dict]:
        """列出所有可恢复的挂起任务"""
        tasks = []
        if not os.path.exists(SUSPENDED_DIR):
            return tasks

        for filename in os.listdir(SUSPENDED_DIR):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(SUSPENDED_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tasks.append({
                    "task_id": data.get("task_id", "?"),
                    "description": data.get("description", "未知任务"),
                    "suspended_at": data.get("suspended_at", 0),
                    "suspended_reason": data.get("suspended_reason", "unknown"),
                    "iteration_count": data.get("iteration_count", 0),
                })
            except Exception:
                continue

        # 按挂起时间降序排列
        tasks.sort(key=lambda t: t.get("suspended_at", 0), reverse=True)
        return tasks

    @classmethod
    def _notify_user(cls, tasks: List[Dict]):
        """通过 EventBus 发送 L2_OUTPUT 通知"""
        try:
            from zulong.core.event_bus import event_bus
            from zulong.core.types import EventType, ZulongEvent, EventPriority

            lines = []
            for i, t in enumerate(tasks[:5], 1):  # 最多显示 5 个
                desc = t["description"][:60]
                reason = t.get("suspended_reason", "unknown")
                elapsed = time.time() - t.get("suspended_at", 0)

                if elapsed < 3600:
                    time_ago = f"{int(elapsed / 60)} 分钟前"
                elif elapsed < 86400:
                    time_ago = f"{elapsed / 3600:.1f} 小时前"
                else:
                    time_ago = f"{elapsed / 86400:.1f} 天前"

                reason_map = {
                    "crash_recovery": "崩溃恢复",
                    "checkpoint": "检查点",
                    "time_limit": "超时",
                    "turn_exhausted": "轮数耗尽",
                    "user_no_response": "等待回答超时",
                    "user_new_task": "新任务中断",
                    "external_interrupt": "外部中断",
                }
                reason_text = reason_map.get(reason, reason)
                lines.append(f"  {i}. 「{desc}」— {reason_text}，暂停于 {time_ago}")

            text = (
                f"系统检测到 {len(tasks)} 个未完成的任务：\n"
                + "\n".join(lines)
                + "\n\n有未完成的任务可供恢复，你可以告诉我需要继续哪个任务，或直接开始新的任务。"
            )

            event = ZulongEvent(
                type=EventType.L2_OUTPUT,
                source="RecoveryNotifier",
                payload={
                    "text": text,
                    "recovery_notification": True,
                    "recoverable_count": len(tasks),
                    "timestamp": time.time(),
                },
                priority=EventPriority.NORMAL,
            )
            event_bus.publish(event)
            logger.info(
                f"[RecoveryNotifier] 已通知用户 {len(tasks)} 个可恢复任务"
            )
        except Exception as e:
            logger.warning(f"[RecoveryNotifier] 通知发送失败: {e}")
