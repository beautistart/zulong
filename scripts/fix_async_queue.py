# -*- coding: utf-8 -*-
# 修复 1: 异步复盘队列

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("=" * 80)
print("           修复 1: 异步复盘队列")
print("=" * 80)
print()

# 读取原文件
file_path = os.path.join(os.path.dirname(__file__), '..', 'zulong', 'memory', 'episodic_memory.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 修复 1: 将同步队列改为异步队列
old_code = '''        # 🔥 第 1 周优化：动态容量管理（替代写死的 max_episodes=50）
        # 配置将在 initialize_async 中根据模型上下文窗口动态计算
        self.max_episodes = 50  # 初始值，会被动态更新
        self.max_tokens_reserved = 0  # 动态计算的 token 预算
        self.estimated_average_turn_tokens = 150  # 估算每轮对话的 token 数
        
        self.summary_max_length = 100  # 摘要最大长度
        self.ttl_seconds = 7200  # 2 小时 TTL
        
        # 🔥 第 1 周优化：异步复盘队列（使用同步队列，避免事件循环问题）
        import queue
        self._pending_summarization_queue: queue.Queue = queue.Queue()
        self._summarization_task = None'''

new_code = '''        # 🔥 第 1 周优化：动态容量管理（替代写死的 max_episodes=50）
        # 配置将在 initialize_async 中根据模型上下文窗口动态计算
        self.max_episodes = 50  # 初始值，会被动态更新
        self.max_tokens_reserved = 0  # 动态计算的 token 预算
        self.estimated_average_turn_tokens = 150  # 估算每轮对话的 token 数
        
        self.summary_max_length = 100  # 摘要最大长度
        self.ttl_seconds = 7200  # 2 小时 TTL
        
        # 🔥 第 1 周优化：异步复盘队列（修复：使用异步队列）
        self._pending_summarization_queue: asyncio.Queue = asyncio.Queue()
        self._summarization_tasks: list = []
        self._num_workers = 2  # 🔥 增加工作者数量，提升处理速度'''

if old_code in content:
    content = content.replace(old_code, new_code)
    print("✅ 修复 1: 异步队列已应用")
else:
    print("⚠️  未找到目标代码，可能已修复或版本不匹配")

# 修复 2: 启动多个工作者
old_code2 = '''    def _start_summarization_worker(self):
        """启动异步复盘任务"""
        if self._summarization_task is None:
            self._summarization_task = asyncio.create_task(self._summarization_worker())
            logger.info("[EpisodicMemory] 复盘工作者已启动")'''

new_code2 = '''    async def _start_summarization_worker(self):
        """启动多个复盘工作者"""
        for i in range(self._num_workers):
            task = asyncio.create_task(self._summarization_worker(i))
            self._summarization_tasks.append(task)
            logger.info(f"[EpisodicMemory] 复盘工作者 #{i} 已启动")'''

if old_code2 in content:
    content = content.replace(old_code2, new_code2)
    print("✅ 修复 2: 多工作者已应用")
else:
    print("⚠️  未找到目标代码 2")

# 修复 3: 工作者协程支持多个实例
old_code3 = '''    async def _summarization_worker(self):
        """复盘工作者协程"""
        logger.info("[EpisodicMemory] 复盘工作者启动")
        
        while True:
            try:
                # 从队列中获取任务
                episode_data = await self._pending_summarization_queue.get()
                
                if episode_data is None:
                    # 退出信号
                    break
                
                # 生成摘要
                summary = await self._generate_summary(episode_data)
                
                # 更新索引
                episode_id = episode_data.get('episode_id')
                if episode_id and summary:
                    self._episode_index[episode_id]['summary'] = summary
                    logger.info(f"[EpisodicMemory] 复盘完成：Episode {episode_id}")
                
                self._stats["summarizations_completed"] += 1
                
            except Exception as e:
                logger.error(f"[EpisodicMemory] 复盘工作者错误：{e}", exc_info=True)'''

new_code3 = '''    async def _summarization_worker(self, worker_id: int = 0):
        """复盘工作者协程（支持多个工作者）"""
        logger.info(f"[EpisodicMemory] 复盘工作者 #{worker_id} 启动")
        
        while True:
            try:
                # 从队列中获取任务
                episode_data = await self._pending_summarization_queue.get()
                
                if episode_data is None:
                    # 退出信号
                    logger.info(f"[EpisodicMemory] 复盘工作者 #{worker_id} 收到退出信号")
                    break
                
                # 生成摘要
                summary = await self._generate_summary(episode_data)
                
                # 更新索引
                episode_id = episode_data.get('episode_id')
                if episode_id and summary:
                    self._episode_index[episode_id]['summary'] = summary
                    logger.info(f"[EpisodicMemory] 复盘完成：Episode {episode_id} (Worker #{worker_id})")
                
                self._stats["summarizations_completed"] += 1
                
                # 标记任务完成
                self._pending_summarization_queue.task_done()
                
            except Exception as e:
                logger.error(f"[EpisodicMemory] 复盘工作者 #{worker_id} 错误：{e}", exc_info=True)'''

if old_code3 in content:
    content = content.replace(old_code3, new_code3)
    print("✅ 修复 3: 多工作者协程已应用")
else:
    print("⚠️  未找到目标代码 3")

# 保存修复后的文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print()
print("=" * 80)
print("✅ 修复完成！请重启系统以应用更改")
print("=" * 80)
