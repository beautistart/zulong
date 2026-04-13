# File: zulong/l2/vlm_agent.py
"""
L2 VLM 代理 (支持快照与重组)

TSD v1.8 对应:
- 2.2.3 L2: 中枢 - 任务规划层 (Cortex)
- 3.3 状态机流转 (State Machine Transitions)
- 4.3 L2: 动态加载与任务冻结

三层注意力机制原子任务对应:
- 🔄 第五步：L2 适配 (支持快照与重组)

核心功能:
1. **上下文快照**: 保存 L2 当前状态 (KV Cache, 对话历史)
2. **任务恢复**: 从快照恢复 L2 状态
3. **强制响应**: 中断当前生成，立即响应新 Prompt
4. **自我总结**: 调用 LLM 总结当前任务

输入:
- Prompt: 用户输入或重组后的 Prompt
- ContextSnapshot: 任务快照 (恢复时使用)

输出:
- 响应文本
- ContextSnapshot: 任务快照

TSD v1.8 对应:
- 第 4.3 节：L2 动态加载与任务冻结
"""

import logging
from typing import Optional, List, Dict, Any
import time
import threading

from zulong.core.attention_atoms import ContextSnapshot, AttentionEvent, MacroCommand
from zulong.models.container import ModelContainer
from zulong.models.config import ModelID

logger = logging.getLogger(__name__)


class VLMAgent:
    """
    L2 VLM 代理 (支持快照与重组)
    
    核心状态:
    - current_task_id: 当前任务 ID
    - history: 对话历史
    - kv_cache: KV Cache (用于加速生成)
    - status: 状态 (IDLE/BUSY/SUSPENDED)
    
    核心方法:
    - create_snapshot(): 生成快照
    - load_snapshot(): 加载快照
    - force_respond(): 强制响应
    - process_event(): 处理事件
    """
    
    def __init__(self):
        """初始化 VLM 代理"""
        self.current_task_id = ""
        self.history: List[Dict[str, str]] = []
        self.kv_cache = None
        self.generation_state = {}
        
        self.status = "IDLE"  # IDLE, BUSY, SUSPENDED
        self._lock = threading.Lock()
        
        # 加载 L2 模型
        try:
            self.model_container = ModelContainer()
            self.l2_model = self.model_container.get_model(ModelID.L2_CORE)
            logger.info("✅ [VLMAgent] L2 模型加载成功")
        except Exception as e:
            logger.warning(f"⚠️ [VLMAgent] L2 模型加载失败：{e}")
            self.l2_model = None
        
        # 统计信息
        self.stats = {
            "snapshots_created": 0,
            "snapshots_loaded": 0,
            "prompts_processed": 0,
            "generations_aborted": 0
        }
        
        logger.info("🧠 [VLMAgent] 初始化完成")
    
    def create_snapshot(self) -> ContextSnapshot:
        """
        生成快照
        
        流程:
        1. 调用 LLM 自我总结当前任务 (Summary)
        2. 保存 KV Cache
        3. 保存对话历史
        4. 保存生成状态
        
        Returns:
            ContextSnapshot: 任务快照
        
        TSD v1.8 对应:
        - 第 4.3 节：任务冻结
        """
        logger.info("📸 [VLMAgent] 正在生成快照...")
        self.stats["snapshots_created"] += 1
        
        # 1. 调用 LLM 自我总结当前任务
        summary = self._ask_llm_to_summarize("请用一句话总结当前正在进行的任务。")
        
        # 2. 创建快照对象
        snapshot = ContextSnapshot(
            task_id=self.current_task_id,
            summary=summary,
            full_history=self.history[-10:],  # 保留最近 10 轮
            kv_cache_ptr=self.kv_cache,
            generation_state=self.generation_state.copy(),
            pause_reason="紧急事件中断"
        )
        
        logger.info(f"✅ [VLMAgent] 快照生成完成：{snapshot.task_id} -> {snapshot.summary}")
        return snapshot
    
    def load_snapshot(self, snapshot: ContextSnapshot) -> None:
        """
        恢复快照
        
        流程:
        1. 恢复对话历史
        2. 恢复 KV Cache
        3. 恢复任务 ID
        4. 更新状态
        
        Args:
            snapshot: 任务快照
        
        TSD v1.8 对应:
        - 第 4.3 节：任务恢复
        """
        logger.info(f"📸 [VLMAgent] 正在加载快照：{snapshot.task_id}")
        self.stats["snapshots_loaded"] += 1
        
        with self._lock:
            # 恢复历史
            self.history = snapshot.full_history.copy()
            
            # 恢复 KV Cache
            self.kv_cache = snapshot.kv_cache_ptr
            
            # 恢复任务 ID
            self.current_task_id = snapshot.task_id
            
            # 恢复生成状态
            self.generation_state = snapshot.generation_state.copy()
            
            # 更新状态
            self.status = "IDLE"  # 恢复后设为空闲，等待新 Prompt
        
        logger.info(f"✅ [VLMAgent] 快照加载完成：{snapshot.task_id}")
    
    def force_respond(self, prompt: str, priority: str = "NORMAL") -> None:
        """
        中断当前生成，立即响应新 Prompt
        
        流程:
        1. 中止当前生成 (如果有)
        2. 将 Prompt 添加到历史
        3. 开始新生成
        
        Args:
            prompt: 用户 Prompt
            priority: 优先级 ("IMMEDIATE", "NORMAL", "LOW")
        
        TSD v1.8 对应:
        - 第 3.2 节：智能路由逻辑 (注入 Prompt)
        """
        logger.info(f"💬 [VLMAgent] 强制响应 (优先级：{priority}): {prompt[:50]}...")
        
        with self._lock:
            # 1. 中止当前生成
            if self.status == "BUSY":
                self._abort_generation()
                self.stats["generations_aborted"] += 1
            
            # 2. 添加 Prompt 到历史
            self.history.append({"role": "user", "content": prompt})
            self.stats["prompts_processed"] += 1
            
            # 3. 开始新生成
            self._start_generation(priority=priority)
    
    def process_event(self, prompt: str) -> None:
        """
        处理事件 (直通模式)
        
        Args:
            prompt: 用户 Prompt
        
        TSD v1.8 对应:
        - 第 3.2 节：智能路由逻辑 (场景 C: 正常空闲)
        """
        logger.info(f"📥 [VLMAgent] 处理事件：{prompt[:50]}...")
        
        with self._lock:
            # 添加 Prompt 到历史
            self.history.append({"role": "user", "content": prompt})
            self.stats["prompts_processed"] += 1
            
            # 开始生成
            self._start_generation(priority="NORMAL")
    
    def _ask_llm_to_summarize(self, instruction: str) -> str:
        """
        调用 LLM 总结当前任务
        
        Args:
            instruction: 总结指令
        
        Returns:
            str: 总结文本
        """
        if not self.l2_model:
            return "任务总结 (模型未加载)"
        
        try:
            # 构建总结 Prompt
            summary_prompt = (
                f"{instruction}\n\n"
                f"当前对话历史:\n"
            )
            
            for msg in self.history[-5:]:  # 最近 5 轮
                role = "用户" if msg["role"] == "user" else "助手"
                summary_prompt += f"{role}: {msg['content'][:100]}...\n"
            
            summary_prompt += "\n请用一句话 (不超过 20 字) 总结当前任务:"
            
            # 调用模型生成
            # 简化实现：返回模拟总结
            # 实际应调用 self.l2_model.generate()
            summary = f"正在执行用户指令 (模拟总结)"
            
            logger.info(f"📝 [VLMAgent] 总结完成：{summary}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ [VLMAgent] 总结失败：{e}")
            return "任务总结 (异常)"
    
    def _abort_generation(self) -> None:
        """
        中止当前生成
        
        实际实现应:
        1. 设置中断标志
        2. 等待生成线程结束
        3. 清理资源
        """
        logger.info("🛑 [VLMAgent] 中止当前生成")
        # 简化实现：仅打印日志
        # 实际应实现真实的中断逻辑
    
    def _start_generation(self, priority: str = "NORMAL") -> None:
        """
        开始生成
        
        Args:
            priority: 优先级
        """
        logger.info(f"🚀 [VLMAgent] 开始生成 (优先级：{priority})")
        self.status = "BUSY"
        
        # 简化实现：仅打印日志
        # 实际应启动生成线程
        
        # 模拟生成完成
        self._simulate_generation()
    
    def _simulate_generation(self) -> None:
        """
        模拟生成 (简化版)
        
        实际应调用 L2 模型生成响应
        """
        if not self.history:
            return
        
        # 获取最后一个用户 Prompt
        last_prompt = self.history[-1]["content"]
        
        # 模拟响应
        response = f"收到指令：{last_prompt[:50]}... (模拟响应)"
        
        # 添加到历史
        self.history.append({"role": "assistant", "content": response})
        
        # 更新状态
        self.status = "IDLE"
        
        logger.info(f"✅ [VLMAgent] 生成完成：{response[:50]}...")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取代理状态
        
        Returns:
            Dict: 状态信息
        """
        return {
            "status": self.status,
            "current_task_id": self.current_task_id,
            "history_length": len(self.history),
            "has_kv_cache": self.kv_cache is not None,
            "stats": self.stats
        }
    
    def reset(self) -> None:
        """
        重置代理状态
        """
        with self._lock:
            self.current_task_id = ""
            self.history = []
            self.kv_cache = None
            self.generation_state = {}
            self.status = "IDLE"
        
        logger.info("🔄 [VLMAgent] 状态已重置")


class VLMAgentMock:
    """
    VLM 代理 Mock 对象 (用于测试)
    
    模拟 L2 的行为，支持:
    - create_snapshot(): 返回模拟快照
    - load_snapshot(): 加载快照
    - force_respond(): 打印 Prompt
    - process_event(): 处理事件
    """
    
    def __init__(self):
        self.current_task_id = ""
        self.history: List[Dict[str, str]] = []
        self.kv_cache = None
        self.status = "IDLE"
        self.snapshots_created = 0
        self.snapshots_loaded = 0
        self.prompts_received = []
        self.generations_aborted = 0
    
    def create_snapshot(self) -> ContextSnapshot:
        """模拟生成快照"""
        self.snapshots_created += 1
        
        snapshot = ContextSnapshot(
            task_id=f"task_{self.snapshots_created}",
            summary=f"模拟任务 {self.snapshots_created}: 正在执行某项工作",
            full_history=self.history[-10:],
            kv_cache_ptr=self.kv_cache,
            pause_reason="紧急事件中断"
        )
        
        logger.info(f"📸 [MockVLM] 创建快照：{snapshot.task_id}")
        return snapshot
    
    def load_snapshot(self, snapshot: ContextSnapshot) -> None:
        """模拟加载快照"""
        self.snapshots_loaded += 1
        self.history = snapshot.full_history.copy()
        self.kv_cache = snapshot.kv_cache_ptr
        self.current_task_id = snapshot.task_id
        self.status = "IDLE"
        
        logger.info(f"📸 [MockVLM] 加载快照：{snapshot.task_id}")
    
    def force_respond(self, prompt: str, priority: str = "NORMAL") -> None:
        """模拟强制响应"""
        if self.status == "BUSY":
            self.generations_aborted += 1
        
        self.prompts_received.append((prompt, priority))
        self.history.append({"role": "user", "content": prompt})
        self.status = "BUSY"
        
        logger.info(f"💬 [MockVLM] 收到 Prompt (优先级：{priority}): {prompt[:50]}...")
        
        # 模拟生成完成
        response = f"模拟响应：{prompt[:30]}..."
        self.history.append({"role": "assistant", "content": response})
        self.status = "IDLE"
    
    def process_event(self, prompt: str) -> None:
        """模拟处理事件"""
        self.prompts_received.append((prompt, "NORMAL"))
        self.history.append({"role": "user", "content": prompt})
        self.status = "BUSY"
        
        logger.info(f"📥 [MockVLM] 处理事件：{prompt[:50]}...")
        
        # 模拟生成完成
        response = f"模拟响应：{prompt[:30]}..."
        self.history.append({"role": "assistant", "content": response})
        self.status = "IDLE"
    
    def get_status(self) -> Dict:
        """获取状态"""
        return {
            "status": self.status,
            "current_task_id": self.current_task_id,
            "history_length": len(self.history),
            "snapshots_created": self.snapshots_created,
            "snapshots_loaded": self.snapshots_loaded,
            "prompts_received": len(self.prompts_received),
            "generations_aborted": self.generations_aborted
        }


if __name__ == "__main__":
    # ========== 测试代码 ==========
    print("🧪 测试 L2 VLM 代理 (支持快照与重组)...")
    
    # 创建 VLM 代理
    agent = VLMAgentMock()
    
    print(f"\n📊 初始状态：{agent.get_status()}")
    
    # ========== 测试场景 1: 处理普通事件 ==========
    print("\n" + "="*60)
    print("场景 1: 处理普通事件")
    print("="*60)
    
    agent.process_event("帮我导航到厨房")
    print(f"处理后状态：{agent.get_status()}")
    
    # ========== 测试场景 2: 创建快照 ==========
    print("\n" + "="*60)
    print("场景 2: 创建快照")
    print("="*60)
    
    snapshot = agent.create_snapshot()
    print(f"快照信息:")
    print(f"   - 任务 ID: {snapshot.task_id}")
    print(f"   - 摘要：{snapshot.summary}")
    print(f"   - 历史长度：{len(snapshot.full_history)}")
    print(f"处理后状态：{agent.get_status()}")
    
    # ========== 测试场景 3: 强制响应 (中断) ==========
    print("\n" + "="*60)
    print("场景 3: 强制响应 (中断)")
    print("="*60)
    
    agent.force_respond("⚠️ 紧急事件：检测到摔倒", priority="IMMEDIATE")
    print(f"处理后状态：{agent.get_status()}")
    
    # ========== 测试场景 4: 加载快照 (恢复) ==========
    print("\n" + "="*60)
    print("场景 4: 加载快照 (恢复)")
    print("="*60)
    
    agent.load_snapshot(snapshot)
    print(f"恢复后状态：{agent.get_status()}")
    
    # ========== 总结 ==========
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"✅ 最终状态：{agent.get_status()}")
    print(f"✅ 对话历史：{len(agent.history)} 条")
    print("\n✅ 所有测试完成!")
