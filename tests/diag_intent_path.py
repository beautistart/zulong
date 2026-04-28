"""
祖龙系统 — 意图分类与 FC 路径精准诊断
=======================================

模拟 InferenceEngine._process_with_memory() 的完整决策链路，
逐步验证每个判断条件，定位复杂任务被当作 CHAT 处理的根因。

运行方式: python tests/diag_intent_path.py
"""

import sys
import os
import json
import time

# 添加项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 测试用的复杂任务输入
COMPLEX_INPUT = (
    "请帮我分析一下 Python 3.12 和 3.13 版本的主要新特性差异，"
    "列出每个版本的 3 个最重要的新功能，并给出升级建议。"
)

def banner(step, title):
    print(f"\n{'='*70}")
    print(f"  Step {step}: {title}")
    print(f"{'='*70}")


def step1_check_model_type():
    """Step 1: 验证 l2_model 的类型（dict vs object）"""
    banner(1, "验证 l2_model 类型")
    
    try:
        from zulong.models.container import ModelContainer
        from zulong.models.config import ModelID
        
        mc = ModelContainer()
        l2_model = mc.get_model(ModelID.L2_CORE)
        
        print(f"  l2_model type:  {type(l2_model)}")
        print(f"  l2_model value: {l2_model}")
        print(f"  isinstance(dict): {isinstance(l2_model, dict)}")
        
        if isinstance(l2_model, dict):
            print(f"  model path:     {l2_model.get('path', 'N/A')}")
            print(f"  model type:     {l2_model.get('type', 'N/A')}")
            print(f"  endpoint:       {l2_model.get('endpoint', 'N/A')}")
            print(f"  model_name:     {l2_model.get('model_name', 'N/A')}")
            print(f"\n  >> PASS: l2_model 是 dict，FC 路径条件 1 满足")
            return True
        else:
            print(f"\n  >> FAIL: l2_model 不是 dict ({type(l2_model).__name__})")
            print(f"  >> 这意味着系统会走本地模型路径，不支持 FC/工具调用！")
            return False
    except Exception as e:
        print(f"  >> ERROR: {e}")
        return False


def step2_check_vllm_client():
    """Step 2: 验证 vllm_client 是否可用"""
    banner(2, "验证 vllm_client (OpenAI 兼容客户端)")
    
    try:
        from openai import OpenAI
        print(f"  OpenAI SDK: 已安装")
    except ImportError:
        print(f"  >> FAIL: OpenAI SDK 未安装，远程推理完全禁用")
        return False, None
    
    try:
        from zulong.models.container import (
            LLM_BACKEND, LLM_BASE_URL, LLM_MODEL_ID, LLM_API_KEY
        )
        print(f"  LLM_BACKEND:  {LLM_BACKEND}")
        print(f"  LLM_BASE_URL: {LLM_BASE_URL}")
        print(f"  LLM_MODEL_ID: {LLM_MODEL_ID}")
        print(f"  LLM_API_KEY:  {'***' if LLM_API_KEY and LLM_API_KEY != 'EMPTY' else LLM_API_KEY}")
        
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        print(f"  client type:   {type(client)}")
        print(f"  client is None: {client is None}")
        
        # 快速验证连接
        try:
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"  可用模型:      {model_ids[:5]}")
            print(f"\n  >> PASS: vllm_client 可用，FC 路径条件 2 满足")
            return True, client
        except Exception as e:
            print(f"  连接测试失败: {e}")
            print(f"\n  >> FAIL: 虽然客户端创建成功，但无法连接到 LLM 后端")
            return False, client
    except Exception as e:
        print(f"  >> ERROR: {e}")
        return False, None


def step3_gate_check(model_ok, client_ok):
    """Step 3: 验证 FC 路径门控条件"""
    banner(3, "FC 路径门控条件 (line 1241)")
    
    print(f"  条件: isinstance(self.l2_model, dict) AND self.vllm_client is not None")
    print(f"  l2_model is dict:       {model_ok}")
    print(f"  vllm_client is not None: {client_ok}")
    print(f"  门控结果:               {'FC 路径 (远程API)' if model_ok and client_ok else '本地模型路径 (无FC)'}")
    
    if model_ok and client_ok:
        print(f"\n  >> PASS: 系统应该走 FC 路径")
        return True
    else:
        print(f"\n  >> FAIL: 系统走本地模型路径，这就是复杂任务被当作 CHAT 的根因！")
        if not model_ok:
            print(f"  >> 根因: l2_model 不是 dict")
        if not client_ok:
            print(f"  >> 根因: vllm_client 是 None（LLM 后端不可用）")
        return False


def step4_intent_classification(client):
    """Step 4: 验证意图分类（Round 1）"""
    banner(4, "Round 1 意图分类")
    
    if not client:
        print(f"  >> SKIP: 无可用客户端")
        return None
    
    try:
        from zulong.l2.intent_prompt_builder import build_round1_system_prompt, get_round1_tools
        from zulong.models.container import LLM_MODEL_ID
        
        round1_messages = [
            {"role": "system", "content": build_round1_system_prompt()},
            {"role": "user", "content": COMPLEX_INPUT},
        ]
        round1_tools = get_round1_tools()
        
        print(f"  System prompt: {len(round1_messages[0]['content'])} chars")
        print(f"  User input:    {COMPLEX_INPUT[:60]}...")
        print(f"  Tools:         {[t['function']['name'] for t in round1_tools]}")
        print(f"  tool_choice:   required (start_session)")
        print(f"  Model:         {LLM_MODEL_ID}")
        
        print(f"\n  正在调用 LLM API (Round 1)...")
        t0 = time.time()
        
        api_response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=round1_messages,
            tools=round1_tools,
            tool_choice={"type": "function", "function": {"name": "start_session"}},
            max_tokens=256,
            temperature=0.1,
            stream=False,
        )
        
        t1 = time.time()
        print(f"  API 响应耗时:  {t1-t0:.1f}s")
        
        msg = api_response.choices[0].message
        print(f"  content:       {msg.content[:200] if msg.content else '(none)'}")
        print(f"  tool_calls:    {len(msg.tool_calls) if msg.tool_calls else 0}")
        
        if not msg.tool_calls:
            print(f"\n  >> FAIL: Round 1 未返回 tool_call，系统将默认 CHAT！")
            return "chat"
        
        tc = msg.tool_calls[0]
        args = json.loads(tc.function.arguments)
        intent = args.get("intent", "chat")
        reason = args.get("reason", "")
        task_desc = args.get("task_description", "")
        
        print(f"\n  分类结果:")
        print(f"    intent:          {intent}")
        print(f"    reason:          {reason}")
        print(f"    task_description: {task_desc}")
        
        if intent == "complex":
            print(f"\n  >> PASS: 意图正确分类为 COMPLEX")
        elif intent == "chat":
            print(f"\n  >> FAIL: 意图被错误分类为 CHAT！")
            print(f"  >> 这就是复杂任务被当作闲聊处理的根因")
        else:
            print(f"\n  >> INFO: 意图分类为 {intent}")
        
        return intent
    except Exception as e:
        print(f"  >> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def step5_scaffold_operation(intent):
    """Step 5: 验证 StartSessionTool 骨架操作"""
    banner(5, "StartSessionTool 骨架操作")
    
    if intent != "complex":
        print(f"  >> SKIP: 意图不是 COMPLEX ({intent})，无需创建任务图")
        return {}
    
    try:
        from zulong.tools.session_tool import StartSessionTool
        from zulong.tools.base import ToolRequest
        
        session_tool = StartSessionTool()
        tool_request = ToolRequest(
            tool_name="start_session",
            action="execute",
            parameters={
                "intent": "complex",
                "reason": "测试复杂任务",
                "task_description": "分析 Python 3.12 vs 3.13",
                "user_input": COMPLEX_INPUT,
            },
        )
        
        print(f"  执行 StartSessionTool.execute()...")
        result = session_tool.execute(tool_request)
        
        print(f"  success: {result.success}")
        print(f"  data:    {json.dumps(result.data, ensure_ascii=False, default=str)[:500] if result.data else '(none)'}")
        
        if result.success and result.data:
            graph_id = result.data.get("graph_id", "")
            title = result.data.get("title", "")
            print(f"  graph_id: {graph_id}")
            print(f"  title:    {title}")
            
            if graph_id:
                print(f"\n  >> PASS: 任务图骨架创建成功")
                
                # 验证全局 active_task_graph 是否设置
                from zulong.tools.task_tools import get_active_task_graph
                tg = get_active_task_graph()
                print(f"  active_task_graph: {tg}")
                if tg:
                    print(f"  task_graph.id:     {tg.id}")
                    print(f"  task_graph.nodes:  {len(tg._nodes)}")
                else:
                    print(f"\n  >> WARN: StartSessionTool 成功但 active_task_graph 为 None")
            else:
                print(f"\n  >> FAIL: StartSessionTool 返回成功但无 graph_id")
        else:
            print(f"\n  >> FAIL: StartSessionTool 执行失败: {result.error}")
        
        return result.data if result.success and result.data else {}
    except Exception as e:
        print(f"  >> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {}


def step6_round2_prompt(intent, scaffold_data):
    """Step 6: 验证 Round 2 提示词构建"""
    banner(6, "Round 2 场景化提示词")
    
    try:
        from zulong.l2.intent_prompt_builder import IntentType, build_round2_system_prompt
        
        intent_map = {"chat": IntentType.CHAT, "complex": IntentType.COMPLEX, "resume": IntentType.RESUME}
        intent_type = intent_map.get(intent, IntentType.CHAT)
        
        messages = build_round2_system_prompt(
            intent_type, COMPLEX_INPUT, None, None, scaffold_data
        )
        
        print(f"  意图类型:   {intent_type.value}")
        print(f"  消息数量:   {len(messages)}")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            print(f"  messages[{i}]: role={role}, length={len(content)}")
            if role == "system":
                # 检查关键内容是否存在
                has_task_rules = "任务管理规则" in content or "任务规划" in content
                has_graph_id = scaffold_data.get("graph_id", "") in content if scaffold_data.get("graph_id") else False
                has_task_add_node = "task_add_node" in content
                print(f"    包含任务管理规则: {has_task_rules}")
                print(f"    包含 graph_id:    {has_graph_id}")
                print(f"    包含 task_add_node: {has_task_add_node}")
                print(f"    提示词前500字符:\n    {content[:500]}")
        
        print(f"\n  >> PASS: Round 2 提示词构建完成")
        return messages
    except Exception as e:
        print(f"  >> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def step7_tool_collection(intent):
    """Step 7: 验证工具收集"""
    banner(7, "Round 2 工具收集")
    
    try:
        from zulong.l2.intent_prompt_builder import IntentType, get_round2_tool_names
        
        intent_map = {"chat": IntentType.CHAT, "complex": IntentType.COMPLEX, "resume": IntentType.RESUME}
        intent_type = intent_map.get(intent, IntentType.CHAT)
        
        allowed_names = get_round2_tool_names(intent_type)
        
        if allowed_names is None:
            print(f"  COMPLEX 场景: 不过滤，使用全部工具")
            
            # 列出实际可用的工具
            from zulong.tools.tool_engine import ToolEngine
            te = ToolEngine()
            all_tools = []
            for name, tool in te.registry.tools.items():
                if tool.enabled:
                    try:
                        schema = tool.get_function_schema()
                        all_tools.append(name)
                    except:
                        pass
            
            print(f"  可用工具数:  {len(all_tools)}")
            print(f"  工具列表:    {all_tools}")
            
            # 关键检查：task_add_node 是否在工具列表中
            has_add_node = "task_add_node" in all_tools
            has_create_plan = "task_create_plan" in all_tools
            has_view_overview = "task_view_overview" in all_tools
            has_mark_status = "task_mark_status" in all_tools
            
            print(f"\n  任务工具可用性:")
            print(f"    task_add_node:      {has_add_node}")
            print(f"    task_create_plan:   {has_create_plan}")
            print(f"    task_view_overview: {has_view_overview}")
            print(f"    task_mark_status:   {has_mark_status}")
            
            if has_add_node:
                print(f"\n  >> PASS: task_add_node 工具可用")
            else:
                print(f"\n  >> FAIL: task_add_node 工具不可用！")
            return all_tools
        else:
            print(f"  {intent} 场景: 允许 {len(allowed_names)} 个工具")
            print(f"  工具列表: {allowed_names}")
            return list(allowed_names)
    except Exception as e:
        print(f"  >> ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []


def step8_fc_single_call(client, messages, tool_names):
    """Step 8: 测试单次 FC 调用，看模型是否调用工具"""
    banner(8, "FC 单次调用测试（模型是否会调用工具？）")
    
    if not client or not messages:
        print(f"  >> SKIP: 无可用客户端或消息")
        return
    
    try:
        from zulong.models.container import LLM_MODEL_ID
        from zulong.tools.tool_engine import ToolEngine
        
        # 收集工具 schema
        te = ToolEngine()
        tool_definitions = []
        for name, tool in te.registry.tools.items():
            if tool.enabled:
                try:
                    schema = tool.get_function_schema()
                    tool_definitions.append(schema)
                except:
                    pass
        
        print(f"  模型:      {LLM_MODEL_ID}")
        print(f"  消息数:    {len(messages)}")
        print(f"  工具数:    {len(tool_definitions)}")
        print(f"  tool_choice: auto")
        
        print(f"\n  正在调用 LLM API (Round 2, FC Turn 1)...")
        t0 = time.time()
        
        response = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=messages,
            tools=tool_definitions if tool_definitions else None,
            tool_choice="auto" if tool_definitions else None,
            max_tokens=2048,
            temperature=0.3,
            stream=False,
        )
        
        t1 = time.time()
        print(f"  API 响应耗时: {t1-t0:.1f}s")
        
        msg = response.choices[0].message
        content = msg.content or ""
        has_tool_calls = bool(msg.tool_calls)
        
        print(f"  content length: {len(content)}")
        if content:
            print(f"  content前300字: {content[:300]}")
        
        print(f"  tool_calls:     {len(msg.tool_calls) if msg.tool_calls else 0}")
        
        if has_tool_calls:
            for i, tc in enumerate(msg.tool_calls):
                args = tc.function.arguments
                print(f"    [{i}] {tc.function.name}({args[:200]})")
            print(f"\n  >> PASS: 模型主动调用了工具（FC 机制正常工作）")
        else:
            print(f"\n  >> WARN: 模型直接回复，未调用任何工具")
            print(f"  >> 这意味着即使意图分类为 COMPLEX，模型也没有创建任务计划")
            print(f"  >> 可能原因:")
            print(f"  >>   1. 4B 模型能力不足，无法正确理解任务管理指令")
            print(f"  >>   2. 提示词中的工具调用指令不够明确")
            print(f"  >>   3. 工具定义过多导致模型困惑")
            
            # 用简化的工具集再测一次
            print(f"\n  --- 简化测试：仅提供 3 个关键任务工具 ---")
            key_tools = [t for t in tool_definitions if t["function"]["name"] in ("task_add_node", "task_view_overview", "task_mark_status")]
            if key_tools:
                print(f"  简化工具数: {len(key_tools)}")
                t2 = time.time()
                response2 = client.chat.completions.create(
                    model=LLM_MODEL_ID,
                    messages=messages,
                    tools=key_tools,
                    tool_choice="auto",
                    max_tokens=2048,
                    temperature=0.3,
                    stream=False,
                )
                t3 = time.time()
                msg2 = response2.choices[0].message
                print(f"  API 响应耗时: {t3-t2:.1f}s")
                print(f"  tool_calls: {len(msg2.tool_calls) if msg2.tool_calls else 0}")
                if msg2.tool_calls:
                    for i, tc in enumerate(msg2.tool_calls):
                        print(f"    [{i}] {tc.function.name}({tc.function.arguments[:200]})")
                    print(f"\n  >> INFO: 简化工具集时模型能调用工具 → 问题是工具数量过多导致模型困惑")
                else:
                    content2 = msg2.content or ""
                    print(f"  content前300字: {content2[:300]}")
                    print(f"\n  >> INFO: 即使简化工具集模型仍不调用工具 → 模型能力问题或提示词问题")
            
    except Exception as e:
        print(f"  >> ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 70)
    print("  祖龙系统 — 意图分类与 FC 路径精准诊断")
    print(f"  测试输入: {COMPLEX_INPUT[:60]}...")
    print("=" * 70)
    
    # Step 1: l2_model 类型
    model_ok = step1_check_model_type()
    
    # Step 2: vllm_client 可用性
    client_ok, client = step2_check_vllm_client()
    
    # Step 3: 门控条件
    fc_path = step3_gate_check(model_ok, client_ok)
    
    if not fc_path:
        print("\n" + "=" * 70)
        print("  诊断结论: FC 路径被阻断！")
        print("  系统走本地模型路径 → 无 FC/工具调用 → 无 TaskGraph")
        print("  修复建议: 确保 l2_model 返回 dict 且 vllm_client 连接正常")
        print("=" * 70)
        return
    
    # Step 4: 意图分类
    intent = step4_intent_classification(client)
    
    if not intent:
        print("\n  >> 无法继续后续步骤")
        return
    
    # Step 5: 骨架操作
    scaffold_data = step5_scaffold_operation(intent)
    
    # Step 6: Round 2 提示词
    messages = step6_round2_prompt(intent, scaffold_data)
    
    # Step 7: 工具收集
    tool_names = step7_tool_collection(intent)
    
    # Step 8: FC 单次调用
    step8_fc_single_call(client, messages, tool_names)
    
    # 总结
    print("\n" + "=" * 70)
    print("  诊断完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
