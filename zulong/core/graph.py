# File: zulong/core/graph.py
# 祖龙系统主图，连接 L1 和 L2 层

from langgraph import StateGraph, END
from zulong.state import ZulongState
from zulong.l1.scheduler import SchedulerNode
from zulong.l2.intent_recognition_node import IntentRecognitionNode


class ZulongGraph:
    """祖龙系统主图"""
    
    def __init__(self):
        """初始化主图"""
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """构建主图"""
        # 创建状态图
        workflow = StateGraph(ZulongState)
        
        # 创建节点实例
        scheduler = SchedulerNode()
        intent_recognizer = IntentRecognitionNode()
        
        # 定义节点函数
        def scheduler_node(state):
            """调度器节点"""
            return scheduler.schedule(state)
        
        def intent_recognition_node(state):
            """意图识别节点"""
            text = state.get('user_input', '')
            if not text:
                return state
            
            # 识别意图
            result = intent_recognizer.recognize_intent(text)
            
            # 更新状态
            state['l2_intent'] = result.dict()
            
            # 根据意图设置下一步
            if result.intent == 'STOP':
                state['next_step'] = 'reflex'
            elif result.intent in ['QUERY_STATUS', 'UNKNOWN']:
                state['next_step'] = 'knowledge'
            else:
                state['next_step'] = 'core'
            
            return state
        
        # 添加节点
        workflow.add_node('scheduler', scheduler_node)
        workflow.add_node('intent_recognition', intent_recognition_node)
        
        # 定义边
        def should_route_to_intent(state):
            """判断是否需要路由到意图识别"""
            # 检查调度器状态
            scheduler_state = state.get('scheduler_state', {})
            is_idle = scheduler_state.get('status') == 'IDLE'
            
            # 检查是否有用户输入
            has_input = bool(state.get('user_input', ''))
            
            return is_idle and has_input
        
        # 调度器 -> 意图识别 (如果空闲且有输入)
        workflow.add_conditional_edges(
            'scheduler',
            should_route_to_intent,
            {
                True: 'intent_recognition',
                False: END
            }
        )
        
        # 意图识别 -> END (后续由其他模块处理)
        workflow.add_edge('intent_recognition', END)
        
        # 设置入口点
        workflow.set_entry_point('scheduler')
        
        return workflow.compile()
    
    def run(self, initial_state):
        """运行图"""
        return self.graph.invoke(initial_state)
