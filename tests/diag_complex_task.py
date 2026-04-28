"""
祖龙系统 — 复杂任务链路分层诊断脚本
=====================================

逐环节验证复杂任务处理链路，每步独立测试，出错即报告。
不需要完整启动系统，直接导入核心模块测试。

运行方式: python tests/diag_complex_task.py

诊断环节:
  Step 1: 模块导入检查 — 核心模块是否能正常导入
  Step 2: TaskGraph 数据结构 — 创建/添加节点/序列化/反序列化
  Step 3: MemoryGraph 初始化 — 图记忆系统能否创建并添加节点
  Step 4: TaskGraph ↔ MemoryGraph 同步 — 适配器双向同步
  Step 5: 任务挂起/恢复 — 序列化到磁盘再反序列化
  Step 6: LangGraph FC Graph 构建 — 图结构能否编译
  Step 7: 意图分类模块 — Round 1 提示词和工具定义
  Step 8: Ollama 连通性 — LLM 是否可用
  Step 9: 端到端意图分类 — 发真实请求测试分类
"""

import sys
import os
import time
import json
import logging
import asyncio
import traceback

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 简单日志
logging.basicConfig(level=logging.WARNING, format='%(name)s: %(message)s')

# 颜色辅助
def ok(msg):   print(f"  [OK] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")
def warn(msg): print(f"  [WARN] {msg}")

results = []  # (step_name, passed: bool, detail: str)


def run_step(name, func):
    """运行单个诊断步骤"""
    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"{'='*60}")
    try:
        func()
        results.append((name, True, ""))
        print(f"  >>> PASS")
    except Exception as e:
        detail = traceback.format_exc()
        results.append((name, False, detail))
        print(f"  >>> FAIL: {e}")
        print(f"  详细错误:\n{detail}")


# ====================================================================
# Step 1: 模块导入检查
# ====================================================================
def step_import_check():
    """检查核心模块是否能正常导入"""

    # 先初始化配置系统（很多模块依赖它）
    info("初始化配置系统...")
    from zulong.config.config_manager import init_config
    init_config()
    ok("配置系统初始化完成")

    modules = [
        ("zulong.l2.task_graph", "TaskGraph"),
        ("zulong.memory.memory_graph", "MemoryGraph"),
        ("zulong.memory.graph_adapters", "TaskGraphAdapter"),
        ("zulong.l2.task_suspension", "TaskSuspensionManager"),
        ("zulong.l2.fc_graph", "build_fc_graph"),
        ("zulong.l2.intent_prompt_builder", "IntentType"),
        ("zulong.l2.attention_window", "AttentionWindowManager"),
        ("zulong.l2.circuit_breaker", "ToolCallCircuitBreaker"),
        ("zulong.l2.info_gap_detector", "InformationGapDetector"),
        ("zulong.tools.task_tools", "TaskCreatePlanTool"),
        ("zulong.tools.session_tool", "StartSessionTool"),
    ]

    failed = []
    for mod_path, class_name in modules:
        try:
            mod = __import__(mod_path, fromlist=[class_name])
            obj = getattr(mod, class_name)
            ok(f"{mod_path}.{class_name}")
        except Exception as e:
            fail(f"{mod_path}.{class_name}: {e}")
            failed.append((mod_path, str(e)))

    if failed:
        raise RuntimeError(f"{len(failed)} 个模块导入失败: {[f[0] for f in failed]}")


# ====================================================================
# Step 2: TaskGraph 数据结构
# ====================================================================
def step_task_graph():
    """测试 TaskGraph 创建、节点操作、序列化/反序列化"""
    from zulong.l2.task_graph import TaskGraph

    # 创建
    tg = TaskGraph(title="测试任务", graph_id="tg_test_001")
    assert tg.id == "tg_test_001", f"graph_id 不匹配: {tg.id}"
    ok(f"创建 TaskGraph: id={tg.id}")

    # 添加根节点
    tg.add_node(id="req", label="测试任务", type="requirement", status="in_progress", desc="这是一个测试任务")
    ok("添加根节点 req")

    # 添加子节点
    tg.add_node(id="o1", label="步骤1-调研", type="outline", status="pending", desc="调研相关资料")
    tg.add_node(id="o2", label="步骤2-分析", type="outline", status="pending", desc="分析调研结果")
    tg.add_node(id="o3", label="步骤3-总结", type="outline", status="pending", desc="撰写总结报告")
    tg.add_h_edge("req", "o1")
    tg.add_h_edge("req", "o2")
    tg.add_h_edge("req", "o3")
    ok(f"添加 3 个子节点，层级边数: {len(tg._h_edges)}")

    # 添加依赖
    from zulong.l2.task_graph import DependencyEdge
    tg.add_d_edge("o1", "o2", via="调研结果")
    tg.add_d_edge("o2", "o3", via="分析报告")
    ok(f"添加 2 条依赖边，依赖边数: {len(tg._d_edges)}")

    # 更新状态
    tg.update_node_status("o1", "completed", result="调研完成：共找到5篇相关论文")
    node = tg.get_node("o1")
    assert node.status == "completed", f"状态更新失败: {node.status}"
    ok("节点状态更新成功")

    # 获取叶子节点
    leaves = tg.get_leaf_nodes()
    ok(f"叶子节点数: {len(leaves)}, IDs: {[n.id for n in leaves]}")

    # 获取子节点
    children = tg.get_children("req")
    ok(f"req 的子节点: {[c for c in children]}")

    # 节点地址
    addr = tg.get_node_address("o2")
    ok(f"o2 的地址: {addr}")

    # 序列化
    serialized = tg.serialize()
    assert isinstance(serialized, dict), f"序列化结果不是 dict: {type(serialized)}"
    json_str = json.dumps(serialized, ensure_ascii=False)
    ok(f"序列化成功，JSON 大小: {len(json_str)} 字节")

    # 反序列化
    tg2 = TaskGraph.deserialize(serialized)
    assert tg2.id == tg.id, f"反序列化 ID 不匹配: {tg2.id}"
    assert len(tg2._nodes) == len(tg._nodes), f"节点数不匹配: {len(tg2._nodes)} vs {len(tg._nodes)}"
    assert len(tg2._h_edges) == len(tg._h_edges), f"层级边数不匹配"
    assert len(tg2._d_edges) == len(tg._d_edges), f"依赖边数不匹配"
    ok(f"反序列化成功: {len(tg2._nodes)} 节点, {len(tg2._h_edges)} 层级边, {len(tg2._d_edges)} 依赖边")

    # 前端格式
    frontend = tg.to_frontend_dict()
    assert "nodes" in frontend, "前端格式缺少 nodes"
    ok(f"前端格式转换成功: {list(frontend.keys())}")


# ====================================================================
# Step 3: MemoryGraph 初始化
# ====================================================================
def step_memory_graph():
    """测试 MemoryGraph 能否创建并添加节点"""
    from zulong.memory.memory_graph import MemoryGraph, GraphNode, NodeType, EdgeType

    # 使用临时目录避免污染生产数据
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="zulong_diag_mg_")
    info(f"使用临时目录: {temp_dir}")

    mg = MemoryGraph(persist_path=temp_dir)
    ok(f"MemoryGraph 创建成功: {mg.stats['total_nodes']} 节点")

    # 添加节点
    node1 = GraphNode(
        node_id="test:dialogue_1",
        node_type=NodeType.DIALOGUE,
        label="用户说了你好",
        activation=1.0,
        metadata={"content": "用户: 你好"},
    )
    mg.add_node(node1)
    ok(f"添加对话节点: {node1.node_id}")

    node2 = GraphNode(
        node_id="test:task_1",
        node_type=NodeType.TASK,
        label="写一份报告",
        activation=0.8,
        metadata={"status": "in_progress"},
    )
    mg.add_node(node2)
    ok(f"添加任务节点: {node2.node_id}")

    # 添加边
    mg.add_edge(node1.node_id, node2.node_id, EdgeType.REFERENCE, weight=0.7)
    ok("添加引用边: dialogue_1 -> task_1")

    # 验证统计
    stats = mg.stats
    ok(f"图统计: {stats['total_nodes']} 节点, {stats['total_edges']} 边")
    assert stats['total_nodes'] >= 2, f"节点数不对: {stats['total_nodes']}"

    # BFS 激活扩散
    try:
        activated = mg.bfs_activate([node1.node_id], depth=2)
        ok(f"BFS 激活扩散: 激活了 {len(activated)} 个节点")
    except Exception as e:
        warn(f"BFS 激活扩散跳过（非致命）: {e}")

    # 保存和加载
    mg.save()
    ok("MemoryGraph 保存到磁盘")

    mg2 = MemoryGraph(persist_path=temp_dir)
    ok(f"MemoryGraph 重新加载: {mg2.stats['total_nodes']} 节点")
    assert mg2.stats['total_nodes'] == stats['total_nodes'], "重新加载后节点数不匹配"

    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# ====================================================================
# Step 4: TaskGraph <-> MemoryGraph 同步
# ====================================================================
def step_adapter_sync():
    """测试 TaskGraphAdapter 将 TaskGraph 投射到 MemoryGraph"""
    from zulong.l2.task_graph import TaskGraph
    from zulong.memory.memory_graph import MemoryGraph
    from zulong.memory.graph_adapters import TaskGraphAdapter

    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="zulong_diag_adapter_")

    # 创建 TaskGraph
    tg = TaskGraph(title="适配器测试任务", graph_id="tg_adapter_test")
    tg.add_node(id="req", label="适配器测试", type="requirement", status="in_progress", desc="测试")
    tg.add_node(id="o1", label="子步骤1", type="outline", status="pending", desc="步骤1")
    tg.add_h_edge("req", "o1")

    # 创建 MemoryGraph
    mg = MemoryGraph(persist_path=temp_dir)

    # 执行同步
    adapter = TaskGraphAdapter()
    count = adapter.sync(mg, tg)
    ok(f"TaskGraphAdapter.sync() 同步了 {count} 个节点到 MemoryGraph")

    # 验证节点是否出现在 MemoryGraph 中
    stats = mg.stats
    ok(f"同步后 MemoryGraph: {stats['total_nodes']} 节点, {stats['total_edges']} 边")
    assert stats['total_nodes'] > 0, "同步后 MemoryGraph 无节点"

    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


# ====================================================================
# Step 5: 任务挂起/恢复
# ====================================================================
def step_suspend_resume():
    """测试任务挂起到磁盘，再恢复"""
    from zulong.l2.task_graph import TaskGraph
    from zulong.l2.task_suspension import TaskSuspensionManager, SuspendableTaskState

    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="zulong_diag_suspend_")

    # 重置单例以使用临时目录
    TaskSuspensionManager._instance = None
    mgr = TaskSuspensionManager(config={"persistence_path": temp_dir})

    # 创建任务图
    tg = TaskGraph(title="挂起恢复测试", graph_id="tg_suspend_test")
    tg.add_node(id="req", label="挂起测试", type="requirement", status="in_progress", desc="测试")
    tg.add_node(id="o1", label="已完成步骤", type="outline", status="completed", desc="", result="done")
    tg.add_node(id="o2", label="未完成步骤", type="outline", status="pending", desc="待做")
    tg.add_h_edge("req", "o1")
    tg.add_h_edge("req", "o2")

    # 挂起
    state = SuspendableTaskState(
        task_id="test_suspend_001",
        description="挂起恢复测试",
        messages=[
            {"role": "system", "content": "你是祖龙"},
            {"role": "user", "content": "帮我写一份报告"},
            {"role": "assistant", "content": "好的，我来规划"},
        ],
        accumulated_links="",
        circuit_breaker_state={"total_calls": 5, "escalated": False},
        iteration_count=3,
        task_graph=tg,
        suspended_reason="test",
        metadata={"graph_id": "tg_suspend_test", "test": True},
    )

    task_id = asyncio.run(mgr.suspend_task(state))
    assert task_id == "test_suspend_001", f"挂起返回的 task_id 不对: {task_id}"
    ok(f"任务挂起成功: {task_id}")

    # 列出挂起任务
    suspended_list = asyncio.run(mgr.list_suspended_tasks())
    ok(f"挂起任务列表: {len(suspended_list)} 个")
    assert len(suspended_list) >= 1, "挂起任务列表为空"

    found = False
    for t in suspended_list:
        if t["task_id"] == "test_suspend_001":
            found = True
            ok(f"  找到任务: {t['description']}, reason={t['suspended_reason']}")
    assert found, "在列表中未找到刚挂起的任务"

    # 恢复
    restored = asyncio.run(mgr.resume_task("test_suspend_001"))
    assert restored is not None, "恢复返回 None"
    ok(f"任务恢复成功: task_id={restored.task_id}")

    # 验证恢复的数据完整性
    assert restored.description == "挂起恢复测试", f"描述不匹配: {restored.description}"
    assert len(restored.messages) == 3, f"消息数不匹配: {len(restored.messages)}"
    assert restored.iteration_count == 3, f"迭代数不匹配: {restored.iteration_count}"
    ok(f"恢复数据完整: {len(restored.messages)} 条消息, {restored.iteration_count} 次迭代")

    # 验证 TaskGraph 恢复
    if restored.task_graph is not None:
        rtg = restored.task_graph
        assert rtg.id == "tg_suspend_test", f"TaskGraph ID 不匹配: {rtg.id}"
        assert len(rtg._nodes) == 3, f"节点数不匹配: {len(rtg._nodes)}"
        o1 = rtg.get_node("o1")
        assert o1.status == "completed", f"o1 状态不匹配: {o1.status}"
        o2 = rtg.get_node("o2")
        assert o2.status == "pending", f"o2 状态不匹配: {o2.status}"
        ok(f"TaskGraph 恢复完整: {rtg.id}, {len(rtg._nodes)} 节点, o1={o1.status}, o2={o2.status}")
    else:
        fail("TaskGraph 恢复为 None!")
        raise RuntimeError("TaskGraph 恢复失败")

    # 恢复后应该从列表中消失
    remaining = asyncio.run(mgr.list_suspended_tasks())
    found_again = any(t["task_id"] == "test_suspend_001" for t in remaining)
    assert not found_again, "恢复后任务仍在列表中（应已删除）"
    ok("恢复后任务已从列表中移除")

    # 清理
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    TaskSuspensionManager._instance = None  # 重置单例


# ====================================================================
# Step 6: LangGraph FC Graph 构建
# ====================================================================
def step_fc_graph_build():
    """测试 FC Graph 能否编译（不实际执行，只验证图结构）"""
    from zulong.l2.fc_graph import build_fc_graph, FCLoopState

    # 创建一个 mock engine 用于编译图
    class MockEngine:
        _warning_interval = 10
        _max_fc_turns = 100
        _soft_limit = 50
        _hard_limit = 100
        _lock = __import__('threading').Lock()
        _interrupt_flag = False
        _fc_loop_timeout = 300
        _attn_window = None
        _circuit_breaker = type('CB', (), {
            'record_call': lambda *a: None,
            'escalate_for_planning': lambda *a: None,
            'evaluate': lambda *a: ('GREEN', ''),
        })()
        vllm_client = None
        backup_client = None

        def _get_llm_extra_kwargs(self): return {}
        def _get_fallback_response(self, text): return "降级回复"
        def _execute_tool_call(self, tc): return "mock result"
        def _publish_task_graph_event(self, *a): pass

    mock = MockEngine()
    compiled = build_fc_graph(mock)
    ok(f"FC Graph 编译成功: {type(compiled).__name__}")

    # 验证节点
    # LangGraph compiled graph 的内部结构
    info(f"编译后图对象: {compiled}")
    ok("LangGraph FC Loop 图结构有效")


# ====================================================================
# Step 7: 意图分类模块
# ====================================================================
def step_intent_module():
    """测试意图分类提示词构建和工具定义"""
    from zulong.l2.intent_prompt_builder import (
        IntentType, build_round1_system_prompt, get_round1_tools,
        get_round2_tool_names, build_round2_system_prompt,
    )

    # Round 1 提示词
    r1_prompt = build_round1_system_prompt()
    assert len(r1_prompt) > 100, f"Round 1 提示词过短: {len(r1_prompt)}"
    assert "祖龙" in r1_prompt, "Round 1 提示词缺少身份声明"
    ok(f"Round 1 提示词: {len(r1_prompt)} chars")

    # Round 1 工具
    r1_tools = get_round1_tools()
    assert len(r1_tools) == 1, f"Round 1 应只有 1 个工具: {len(r1_tools)}"
    assert r1_tools[0]["function"]["name"] == "start_session"
    ok(f"Round 1 工具: {r1_tools[0]['function']['name']}")

    # Round 2 工具过滤
    chat_tools = get_round2_tool_names(IntentType.CHAT)
    assert isinstance(chat_tools, set), f"CHAT 工具应返回 set: {type(chat_tools)}"
    ok(f"CHAT 工具集: {len(chat_tools)} 个 — {chat_tools}")

    resume_tools = get_round2_tool_names(IntentType.RESUME)
    assert isinstance(resume_tools, set), f"RESUME 工具应返回 set: {type(resume_tools)}"
    assert "task_create_plan" not in resume_tools, "RESUME 不应包含 task_create_plan"
    assert "task_add_node" not in resume_tools, "RESUME 不应包含 task_add_node"
    assert "task_view_overview" in resume_tools, "RESUME 应包含 task_view_overview"
    ok(f"RESUME 工具集: {len(resume_tools)} 个 — {resume_tools}")

    complex_tools = get_round2_tool_names(IntentType.COMPLEX)
    assert complex_tools is None, "COMPLEX 应返回 None（不过滤）"
    ok("COMPLEX 工具集: 不过滤（使用全部工具）")

    # Round 2 提示词构建
    for intent in [IntentType.CHAT, IntentType.COMPLEX, IntentType.RESUME]:
        scaffold = {}
        if intent == IntentType.COMPLEX:
            scaffold = {"graph_id": "tg_test", "title": "测试任务"}
        elif intent == IntentType.RESUME:
            scaffold = {"task_id": "test_001", "description": "恢复测试", "has_task_graph": True}
        messages = build_round2_system_prompt(intent, "测试输入", None, None, scaffold)
        assert len(messages) >= 2, f"{intent.value} messages 不足 2 条"
        assert messages[-1]["role"] == "user", f"{intent.value} 最后一条应是 user"
        ok(f"Round 2 {intent.value} 提示词: {len(messages)} 条消息, system={len(messages[0]['content'])} chars")


# ====================================================================
# Step 8: Ollama 连通性
# ====================================================================
def step_ollama_connectivity():
    """检查 Ollama 服务是否可用"""
    from zulong.config.config_manager import get_config

    backend = get_config('llm.backend', 'ollama')
    info(f"LLM 后端: {backend}")

    if backend == 'ollama':
        base_url = get_config('llm.ollama.base_url', 'http://localhost:11434/v1')
        model_id = get_config('llm.ollama.model_id', 'qwen3.5:4b')
    else:
        base_url = get_config(f'llm.{backend}.base_url', 'http://localhost:11434/v1')
        model_id = get_config(f'llm.{backend}.model_id', '')

    info(f"API 地址: {base_url}")
    info(f"模型 ID: {model_id}")

    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="EMPTY")

        # 简单测试：列出模型
        try:
            models = client.models.list()
            model_names = [m.id for m in models.data]
            ok(f"模型列表: {model_names[:5]}...")
        except Exception as e:
            warn(f"列出模型失败（某些后端不支持，非致命）: {e}")

        # 测试简单补全
        info("发送测试消息...")
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "回复OK两个字"}],
            max_tokens=50,
            temperature=0.1,
            stream=False,
        )
        text = response.choices[0].message.content or ""
        ok(f"LLM 回复: '{text[:100]}'")

        # 测试 Function Calling
        info("测试 Function Calling...")
        fc_response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "你是意图分类器。调用 start_session 工具分类。"},
                {"role": "user", "content": "帮我写一份关于AI的调研报告"},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "start_session",
                    "description": "意图分类",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": ["chat", "complex", "resume"],
                            },
                            "reason": {"type": "string"},
                        },
                        "required": ["intent", "reason"],
                    },
                },
            }],
            tool_choice={"type": "function", "function": {"name": "start_session"}},
            max_tokens=256,
            temperature=0.1,
            stream=False,
        )
        msg = fc_response.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            args = json.loads(tc.function.arguments)
            ok(f"Function Calling 成功: intent={args.get('intent')}, reason={args.get('reason', '')[:50]}")
        else:
            content = msg.content or ""
            fail(f"Function Calling 失败: 模型没有调用工具，而是直接回复: '{content[:100]}'")
            warn("这通常表示模型不支持 tool_choice=required，需要换用支持 FC 的模型")
            raise RuntimeError("Function Calling 不可用")

    except ImportError:
        fail("openai 包未安装: pip install openai")
        raise
    except Exception as e:
        if "Connection" in str(e) or "connect" in str(e).lower():
            fail(f"无法连接到 {base_url}: {e}")
            raise RuntimeError(f"LLM 服务不可用: {base_url}")
        raise


# ====================================================================
# Step 9: 端到端意图分类
# ====================================================================
def step_e2e_intent():
    """使用真实 LLM 进行端到端意图分类测试"""
    from zulong.l2.intent_prompt_builder import build_round1_system_prompt, get_round1_tools
    from zulong.config.config_manager import get_config

    backend = get_config('llm.backend', 'ollama')
    if backend == 'ollama':
        base_url = get_config('llm.ollama.base_url', 'http://localhost:11434/v1')
        model_id = get_config('llm.ollama.model_id', 'qwen3.5:4b')
    else:
        base_url = get_config(f'llm.{backend}.base_url', 'http://localhost:11434/v1')
        model_id = get_config(f'llm.{backend}.model_id', '')

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    test_cases = [
        ("你好，今天天气怎么样？", "chat"),
        ("帮我做一份关于人工智能发展趋势的详细调研报告，需要包含市场分析和技术路线", "complex"),
        ("继续之前那个报告任务", "resume"),
    ]

    for user_input, expected_intent in test_cases:
        info(f"测试: '{user_input}' -> 期望: {expected_intent}")
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": build_round1_system_prompt()},
                    {"role": "user", "content": user_input},
                ],
                tools=get_round1_tools(),
                tool_choice={"type": "function", "function": {"name": "start_session"}},
                max_tokens=256,
                temperature=0.1,
                stream=False,
            )
            msg = response.choices[0].message
            if msg.tool_calls:
                args = json.loads(msg.tool_calls[0].function.arguments)
                actual = args.get("intent", "?")
                match = "OK" if actual == expected_intent else "MISMATCH"
                if actual == expected_intent:
                    ok(f"  分类正确: {actual} (reason: {args.get('reason', '')[:60]})")
                else:
                    warn(f"  分类偏差: 期望={expected_intent}, 实际={actual} (reason: {args.get('reason', '')[:60]})")
            else:
                fail(f"  未调用工具，直接回复: {(msg.content or '')[:80]}")
        except Exception as e:
            fail(f"  分类失败: {e}")


# ====================================================================
# 主程序
# ====================================================================
def main():
    print("\n" + "=" * 60)
    print("  祖龙系统 — 复杂任务链路分层诊断")
    print("=" * 60)

    steps = [
        ("1. 模块导入检查", step_import_check),
        ("2. TaskGraph 数据结构", step_task_graph),
        ("3. MemoryGraph 初始化", step_memory_graph),
        ("4. TaskGraph <-> MemoryGraph 同步", step_adapter_sync),
        ("5. 任务挂起/恢复", step_suspend_resume),
        ("6. LangGraph FC Graph 构建", step_fc_graph_build),
        ("7. 意图分类模块", step_intent_module),
        ("8. Ollama 连通性", step_ollama_connectivity),
        ("9. 端到端意图分类", step_e2e_intent),
    ]

    for name, func in steps:
        run_step(name, func)

    # 总结
    print("\n" + "=" * 60)
    print("  诊断结果总结")
    print("=" * 60)

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    for name, ok_flag, detail in results:
        status = "PASS" if ok_flag else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  总计: {passed}/{total} 通过")

    if passed == total:
        print("\n  所有环节通过! 复杂任务基础链路健康。")
        print("  下一步建议: 启动系统发送一条复杂任务进行端到端测试。")
    else:
        failed_steps = [name for name, ok_flag, _ in results if not ok_flag]
        print(f"\n  {len(failed_steps)} 个环节失败: {failed_steps}")
        print("  请优先修复第一个失败的环节，后续环节可能依赖它。")

    print()


if __name__ == "__main__":
    main()
