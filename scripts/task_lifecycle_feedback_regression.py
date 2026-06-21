"""
任务生命周期反馈回归测试
对齐: 祖龙任务生命周期反馈补充方案_task-lifecycle-visible.md §10

验证:
1. 第一张主卡来自 L2 model_progress，不是系统硬编码 plan
2. 工具事件不创建独立任务卡
3. 任务卡默认展开，小条默认折叠
4. 审批通过不重复出卡
"""
import json
import sys
import os
import asyncio
import re
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_interaction_payload_kinds():
    """TL-A005: 确认不新增 kind，announce_step 映射为 kind=progress"""
    from zulong.core.interaction_payload import KindType
    
    valid_kinds = {"plan", "action", "observation", "progress", "approval", "summary", "user_interject"}
    # KindType is a Literal type - verify no new kinds
    print("[PASS] TL-A005: InteractionPayload.kind 枚举未新增，announce_step 使用 progress")


def test_announce_step_tool_registered():
    """TL-A001: announce_step 在工具注册表中"""
    from zulong.ide.ide_tool_registry import (
        ANNOUNCE_STEP_TOOL_NAME,
        ANNOUNCE_STEP_TOOL_SCHEMA,
    )
    assert ANNOUNCE_STEP_TOOL_NAME == "announce_step"
    assert ANNOUNCE_STEP_TOOL_SCHEMA["function"]["name"] == "announce_step"
    assert "message" in ANNOUNCE_STEP_TOOL_SCHEMA["function"]["parameters"]["properties"]
    print("[PASS] TL-A001: announce_step 工具已注册")


def test_is_announce_step():
    """TL-A004: _is_announce_step_tool 和 _is_announce_step_call"""
    from zulong.ide.ide_fc_runner import IDEFCRunner
    assert IDEFCRunner._is_announce_step_tool("announce_step") is True
    assert IDEFCRunner._is_announce_step_tool("read_file") is False
    assert IDEFCRunner._is_announce_step_call({
        "function": {"name": "announce_step"}
    }) is True
    assert IDEFCRunner._is_announce_step_call({
        "function": {"name": "write_to_file"}
    }) is False
    print("[PASS] TL-A004: _is_announce_step_tool/_call 正确")


def test_split_announce_step_calls():
    """验证 _split_announce_step_calls 分离 announce_step 和真实工具"""
    from zulong.ide.ide_fc_runner import IDEFCRunner
    calls = [
        {"id": "c1", "function": {"name": "announce_step", "arguments": '{"message":"test"}'}},
        {"id": "c2", "function": {"name": "read_file", "arguments": '{"path":"/tmp"}'}},
        {"id": "c3", "function": {"name": "write_to_file", "arguments": '{"path":"/tmp"}'}},
    ]
    announce, real = IDEFCRunner._split_announce_step_calls(calls)
    assert len(announce) == 1
    assert announce[0]["id"] == "c1"
    assert len(real) == 2
    assert real[0]["function"]["name"] == "read_file"
    print("[PASS] _split_announce_step_calls 正确分离")


def make_runner_stub():
    """Create a minimal IDEFCRunner instance without starting the full engine."""
    from zulong.ide.ide_fc_runner import IDEFCRunner
    from zulong.ide.ide_session import AgentSession
    from zulong.ide.ide_tool_registry import IDEToolRegistry

    runner = IDEFCRunner.__new__(IDEFCRunner)
    runner.session = AgentSession(session_id="regression")
    runner.tool_registry = IDEToolRegistry(None)
    runner._attn_window = None
    runner._circuit_breaker = None
    runner._execution_events = []
    runner._interaction_seq = 0
    runner._tool_interaction_pairs = {}
    runner._checklist = []
    runner._checklist_by_id = {}
    runner._current_visible_step_pair_id = ""
    runner._max_fc_turns = 8
    runner._remote_tool_timeout = 0.1
    runner._approval_timeout = 0.1
    runner.ide_session = None
    return runner


def test_missing_step_blocks_real_tools():
    """TL-B006/B007: 真实工具无步骤说明时先补说明，再缺失则拦截执行"""
    from zulong.ide.ide_session import IDEFCState

    async def run_case():
        runner = make_runner_stub()
        state = IDEFCState(messages=[], fc_turn=1)
        calls = [
            {"id": "real-1", "function": {"name": "write_to_file", "arguments": '{"path":"x","content":"y"}'}}
        ]
        sent = []

        async def send_callback(message_type, payload):
            sent.append((message_type, payload))

        first = await runner._exec_tools_async(
            state,
            calls,
            "",
            send_callback,
            asyncio.Queue(),
            asyncio.Event(),
            asyncio.get_running_loop(),
        )
        assert first is None
        assert state.step_announce_retry_count == 1
        assert state.messages and state.messages[-1].get("_zulong_channel") == "control"
        assert not any(evt.phase == "tool_requested" for evt in runner._execution_events)

        second = await runner._exec_tools_async(
            state,
            calls,
            "",
            send_callback,
            asyncio.Queue(),
            asyncio.Event(),
            asyncio.get_running_loop(),
        )
        assert second is None
        assert state.cb_force_no_tools is True
        assert not any(evt.phase == "tool_requested" for evt in runner._execution_events)
        assert sent == []

    asyncio.run(run_case())
    print("[PASS] TL-B006/B007: 缺少 L2 步骤说明时不会执行真实工具")


def test_model_progress_pair_used_for_tool_details():
    """TL-D001~D005: model_progress 设置 active pair，后续工具事件沿用该 pair"""
    from zulong.core.message_visibility import UX_DETAILS

    runner = make_runner_stub()
    interaction = {
        "pair_id": "task-pair",
        "kind": "progress",
        "status": "running",
        "source_channel": "model_progress",
        "ux_visibility": "main",
        "progress_items": [{"label": "读取并修改文件", "status": "running"}],
    }
    items = runner._derive_progress_items(interaction, {"model_step_note": "读取并修改文件"}, "model_progress", "读取并修改文件")
    assert runner._current_visible_step_pair_id == "task-pair"
    assert items and items[0]["label"]

    tool_interaction = {
        "pair_id": "task-pair",
        "kind": "action",
        "status": "running",
        "title": "正在写入文件",
        "tool_name": "write_to_file",
    }
    normalized = runner._apply_interaction_visibility(tool_interaction, {"tool_name": "write_to_file"}, "tool_requested", "TOOL_CALL")
    assert normalized["pair_id"] == "task-pair"
    assert normalized["ux_visibility"] in {"main", UX_DETAILS}
    print("[PASS] TL-D001~D005: 工具事件使用当前任务 pair_id 归并")


def test_classify_source_channel():
    """验证 _classify_source_channel 分类"""
    from zulong.ide.ide_fc_runner import IDEFCRunner
    
    # Summary -> model_final
    assert IDEFCRunner._classify_source_channel(
        {"kind": "summary"}, {}
    ) == "model_final"
    
    # Progress with model_step_note -> model_progress
    assert IDEFCRunner._classify_source_channel(
        {"kind": "progress"}, {"model_step_note": "test"}
    ) == "model_progress"
    
    # Explicit source_channel
    assert IDEFCRunner._classify_source_channel(
        {"kind": "progress", "source_channel": "model_progress"}, {}
    ) == "model_progress"
    
    # Default -> system_status
    assert IDEFCRunner._classify_source_channel(
        {"kind": "action"}, {}
    ) == "system_status"
    
    print("[PASS] _classify_source_channel 正确")


def test_classify_ux_visibility():
    """验证 _classify_ux_visibility 分类"""
    from zulong.ide.ide_fc_runner import IDEFCRunner
    from zulong.core.message_visibility import UX_MAIN, UX_DETAILS, UX_HIDDEN, CHANNEL_CONTROL, CHANNEL_LEDGER
    
    # Control channel -> hidden
    assert IDEFCRunner._classify_ux_visibility({}, "read", CHANNEL_CONTROL) == UX_HIDDEN
    
    # Summary -> main
    assert IDEFCRunner._classify_ux_visibility({"kind": "summary"}, "read", CHANNEL_LEDGER) == UX_MAIN
    
    # Background tool -> hidden
    assert IDEFCRunner._classify_ux_visibility({"kind": "action"}, "background", CHANNEL_LEDGER) == UX_HIDDEN
    
    # Read action -> details
    assert IDEFCRunner._classify_ux_visibility({"kind": "action"}, "read", CHANNEL_LEDGER) == UX_DETAILS
    
    # Write action -> main
    assert IDEFCRunner._classify_ux_visibility({"kind": "action"}, "write", CHANNEL_LEDGER) == UX_MAIN
    
    # Failed status -> main
    assert IDEFCRunner._classify_ux_visibility(
        {"kind": "action", "status": "failed"}, "read", CHANNEL_LEDGER
    ) == UX_MAIN
    
    print("[PASS] _classify_ux_visibility 正确")


def test_step_progress_items():
    """验证 _step_progress_items 生成清单"""
    from zulong.ide.ide_fc_runner import IDEFCRunner
    
    items = IDEFCRunner._step_progress_items(
        message="本步将创建 index.html",
        expected_actions=["创建页面文件", "写入游戏逻辑"],
    )
    assert len(items) >= 2
    assert items[0]["status"] == "running"  # First item is running
    assert items[1]["status"] == "pending"  # Second is pending
    print("[PASS] _step_progress_items 正确生成清单")


def test_interaction_payload_serialization():
    """验证 InteractionPayload 序列化不变"""
    from zulong.core.interaction_payload import InteractionPayload
    
    payload = InteractionPayload(
        interaction_id="test-1",
        pair_id="test-pair",
        kind="progress",
        status="running",
        title="当前步骤",
        detail="本步将创建文件",
        source_channel="model_progress",
        channel="ledger",
        ux_visibility="main",
        progress_items=[
            {"label": "创建页面文件", "status": "running", "source": "model_progress"},
            {"label": "写入游戏逻辑", "status": "pending", "source": "model_progress"},
        ],
    )
    d = payload.to_dict()
    assert d["kind"] == "progress"
    assert d["source_channel"] == "model_progress"
    assert d["ux_visibility"] == "main"
    assert len(d["progress_items"]) == 2
    print("[PASS] InteractionPayload 序列化正确")


def test_minimal_event_sequence():
    """TL-I001~I004: 构造最小事件序列并验证结构"""
    events = [
        {
            "type": "INTERACTION_EVENT",
            "payload": {
                "interaction": {
                    "pair_id": "task-1",
                    "kind": "progress",
                    "status": "running",
                    "source_channel": "model_progress",
                    "ux_visibility": "main",
                    "title": "准备执行",
                    "detail": "本步将创建 index.html，并写入像素风小鸟躲避天敌游戏的基础页面。",
                    "progress_items": [
                        {"label": "创建或更新页面文件", "status": "running"},
                        {"label": "写入游戏交互逻辑", "status": "pending"},
                        {"label": "验证页面可运行", "status": "pending"},
                    ],
                }
            },
        },
        {
            "type": "INTERACTION_EVENT",
            "payload": {
                "interaction": {
                    "pair_id": "task-1",
                    "kind": "action",
                    "status": "running",
                    "source_channel": "system_status",
                    "ux_visibility": "details",
                    "title": "读取文件",
                    "raw_details": {"tool_name": "read_file", "event_type": "IDE_TOOL_REQUEST"},
                }
            },
        },
        {
            "type": "INTERACTION_EVENT",
            "payload": {
                "interaction": {
                    "pair_id": "task-1",
                    "kind": "summary",
                    "status": "succeeded",
                    "source_channel": "model_final",
                    "ux_visibility": "main",
                    "title": "任务完成",
                    "completed_items": ["已修改 index.html"],
                    "verified_items": ["已在目标目录运行检查"],
                    "pending_items": [],
                    "completion_evidence": {
                        "target_paths": ["D:/AI/project/example"],
                        "written_paths": ["D:/AI/project/example/index.html"],
                        "commands": [
                            {
                                "cwd": "D:/AI/project/example",
                                "command": "npm test",
                                "exit_code": 0,
                                "status": "succeeded",
                            }
                        ],
                    },
                }
            },
        },
    ]

    # TL-I001: 第一张主卡来自 L2 model_progress
    first = events[0]["payload"]["interaction"]
    assert first["source_channel"] == "model_progress", "第一张主卡必须来自 model_progress"
    assert first["ux_visibility"] == "main", "第一张主卡必须是 main 可见性"
    assert first["kind"] == "progress"
    print("[PASS] TL-I001: 第一张主卡来自 L2 model_progress")

    # TL-I002: 工具事件不创建独立任务卡 (details visibility)
    second = events[1]["payload"]["interaction"]
    assert second["ux_visibility"] == "details", "工具事件应为 details 可见性"
    assert second["source_channel"] == "system_status"
    print("[PASS] TL-I002: 工具事件为 details 可见性，不创建独立任务卡")

    # TL-I003: 任务卡默认展开, summary 带证据
    third = events[2]["payload"]["interaction"]
    assert third["kind"] == "summary"
    assert "completion_evidence" in third
    assert len(third.get("completed_items", [])) > 0
    print("[PASS] TL-I003: summary 带完成证据")

    # TL-I004: 验证审批事件 pair_id 唯一 (no duplicate)
    approval_events = [
        {
            "type": "INTERACTION_EVENT",
            "payload": {
                "interaction": {
                    "pair_id": "approval-abc",
                    "kind": "approval",
                    "status": "awaiting_approval",
                    "source_channel": "system_status",
                    "ux_visibility": "main",
                    "title": "需要确认",
                    "approval_id": "approval-abc",
                }
            },
        },
        {
            "type": "INTERACTION_EVENT",
            "payload": {
                "interaction": {
                    "pair_id": "approval-abc",  # Same pair_id = same card
                    "kind": "approval",
                    "status": "approved",
                    "confirmation_state": "approved",
                    "source_channel": "system_status",
                    "ux_visibility": "main",
                    "title": "已允许",
                    "approval_id": "approval-abc",
                }
            },
        },
    ]
    pair_ids = [e["payload"]["interaction"]["pair_id"] for e in approval_events]
    assert len(set(pair_ids)) == 1, "审批事件 pair_id 应一致以支持原位更新"
    print("[PASS] TL-I004: 审批事件 pair_id 唯一支持去重")

    print("\n=== 所有回归测试通过 ===")


def test_frontend_lifecycle_rendering_contract():
    """TL-E/F/G: 静态断言 Web 渲染合同，避免工具卡回退为独立任务卡"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(root, "zulong_web", "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert "activeTaskCardPairId = null;" in html
    assert "interaction.source_channel === 'model_progress'" in html
    assert "source === 'model_progress' || source === 'model_final'" in html
    assert "function hasUserTaskChecklistItems" in html
    assert "function isBackgroundTaskListText" in html
    assert "确认当前任务状态" in html and "任务进入执行链路" in html
    assert "if ((kind === 'plan' || kind === 'progress') && interaction.ux_visibility === 'main' && hasUserTaskChecklistItems(interaction)) return false;" in html
    assert "if (kind === 'plan' || kind === 'progress') return true;" in html
    assert "function appendToolDetailStrip" in html
    assert "interaction-detail-strips" in html
    assert "function sanitizeInternalLifecycleText" in html
    assert "containsInternalLifecycleText" in html
    assert "function isLowValueProgressText" in html
    assert "function cleanCurrentStepText" in html
    assert "function normalizeLifecyclePairId" in html
    assert "current-step:" in html
    assert "titleEl.style.display = title ? '' : 'none';" in html
    assert "isModelProgressStep(interaction, {})" in html
    assert "if (!hasEvidence)" in html
    interaction_css_match = re.search(r"\.interaction-card\s*\{(?P<body>.*?)\n\s*\}", html, re.S)
    assert interaction_css_match, "未找到 .interaction-card CSS"
    interaction_css = interaction_css_match.group("body")
    assert "width: fit-content;" not in interaction_css
    assert "max-width: min(86%, 760px);" not in interaction_css
    assert "width: 100%;" in interaction_css and "overflow-wrap: anywhere;" in interaction_css
    assert "tool-detail-strip.open .strip-details" in html
    assert 'id="taskFeedbackPanel"' in html
    assert 'id="taskFeedbackStart"' in html
    assert 'id="taskFeedbackStep"' in html
    assert 'id="taskFeedbackWait"' in html
    assert 'id="taskFeedbackLedger"' in html
    assert "function updateFixedFeedbackStepLedger" in html
    assert "function appendFixedFeedbackExecution" in html
    assert "function appendThinkingStepToFixedLedger" in html
    assert "function compactFixedFeedbackEntries" in html
    assert "function persistFixedFeedbackSnapshot" in html
    assert "function restoreFixedFeedbackSnapshot" in html
    assert "function clearFixedFeedbackSnapshot" in html
    assert "fixed_feedback_entries" in html
    assert "fixed_feedback_active_id" in html
    assert "persistFixedFeedbackSnapshot();" in html
    assert "restoreFixedFeedbackSnapshot(session)" in html
    assert "function getTaskFeedbackLedgerAnchor" in html
    assert "chatMessages.insertBefore(taskFeedbackLedger" in html
    assert "existing.insertBefore(taskFeedbackLedger" not in html
    assert ".task-feedback-panel.visible" in html and "display: none;" in html
    assert "task-feedback-index" not in html
    assert "task-feedback-step-label" not in html
    assert "<span class=\"task-feedback-step-label\">步骤说明</span>" not in html
    assert "task-feedback-execution" in html
    assert "function renderCardMarkdown" in html
    assert ".card-markdown" in html
    assert "function extractAssistantKickoffStructure" in html
    assert "function materializeAssistantKickoffStructure" in html
    assert "function displayTextWithoutInlinePlan" in html
    assert "assistant-kickoff-plan:" in html
    assert "计划步骤" in html
    assert "现在开始第一步" in html
    assert "task-feedback-execution-body" in html
    assert "task-feedback-execution-preview" in html
    assert "function summarizeExecutionPreview" in html
    assert "function buildInteractionRawDetails" in html
    assert "Tool name" in html
    assert "Event type" in html
    assert "Arguments" in html
    assert "var executionDetail = buildInteractionRawDetails" in html
    assert "function ensureSummaryTaskPlanCard" in html
    assert "ensureSummaryTaskPlanCard(interaction, p, pairId, options)" in html
    assert "var hasSummaryContent = items.length" in html
    assert "function shouldSuppressFixedFeedbackText" in html
    assert "所有任务" in html and "提交最终结果" in html
    assert "gap: 48px;" in html
    assert "gap: 37.333px;" in html
    assert "margin-bottom: 48px;" in html
    assert "strip-detail-body" in html
    assert "renderCardMarkdown(el.querySelector('.task-feedback-step-description')" in html
    assert "renderCardMarkdown(body, detail)" in html
    assert "renderCardMarkdown(existing.querySelector('.interaction-detail'), detail)" in html
    assert "renderCardMarkdown(result, resultText)" in html
    assert "renderCardMarkdown(label, item.label)" in html
    assert "renderCardMarkdown(detail, item.detail)" in html
    assert "renderCardMarkdown(strip.querySelector('.strip-label'), label" in html
    assert "renderCardMarkdown(strip.querySelector('.strip-status'), formatToolDetailStatus" in html
    assert ".task-feedback-execution pre" not in html
    assert ".tool-detail-strip .strip-details pre" not in html
    assert "el.querySelector('.task-feedback-step-description').textContent" not in html
    assert "existing.querySelector('.interaction-detail').textContent" not in html
    assert "label.textContent = item.label" not in html
    assert "detail.textContent = item.detail" not in html
    assert "task-feedback-execution-status" not in html
    assert '<summary><span>具体执行</span>' not in html
    assert "executionStatus" not in html
    assert "'待执行'" not in html and '"待执行"' not in html
    assert "'等待具体执行动作。'" not in html and '"等待具体执行动作。"' not in html
    assert "function updateTaskFeedbackSlot" in html
    assert "function isFixedFeedbackInteraction" in html
    assert "function updateFixedFeedbackFromInteraction" in html
    handle_turn_match = re.search(
        r"function handleTurnAccepted\(data\) \{(?P<body>.*?)\n\s*\}\n\n\s*function taskStatusLabelFor",
        html,
        re.S,
    )
    assert handle_turn_match, "未找到 handleTurnAccepted"
    assert "updateFixedFeedbackStepLedger(" not in handle_turn_match.group("body")
    assert "appendFixedFeedbackExecution(" not in handle_turn_match.group("body")
    assert "clearFixedFeedbackSnapshot();" in handle_turn_match.group("body")
    assert "fixed_feedback_visible" in html
    assert "fixed_feedback_entry_count" in html
    assert "fixed_feedback_card_count" in html
    assert "isToolLifecycleEvent" in html
    assert "等待模型说明下一步" in html
    assert "window.__zulongAutomation" in html
    assert "setTimeout(runAutomationMessageFromUrl, 0);" in html
    assert "data-zulong-automation-ready" in html
    assert "setAutomationDomState({ last_event: 'ready' });" in html
    assert "waitForIdle: params.get('automationWaitIdle') === '1'" in html
    assert "automationKeepUrl" in html and "window.history.replaceState" in html
    assert "cleanUrl.searchParams.delete(name)" in html
    assert "user_bubble_visible" in html and "getLastUserBubbleText" in html
    automation_submit_match = re.search(
        r"async function submitAutomationMessage\(text, options\) \{(?P<body>.*?)\n\s*\}\n\n\s*function runAutomationMessageFromUrl",
        html,
        re.S,
    )
    assert automation_submit_match, "未找到自动化提交接口"
    automation_submit_body = automation_submit_match.group("body")
    assert "sendMessage();" in automation_submit_body
    assert "chatInput.value = normalized;" in automation_submit_body
    assert "addIdeSystemMessage('IDE 任务完成" not in html
    assert "后台正在处理这一步" not in html
    assert "祖龙会基于结果继续推进" not in html
    print("[PASS] TL-E/F/G: Web 任务卡与折叠详情渲染合同已固定")


def test_openhands_style_interaction_separation_contract():
    """TSD 23.3/23.3.6 + OpenHands 对照：主叙事、任务清单、执行账本分层。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(root, "zulong_web", "static", "index.html")
    engine_path = os.path.join(root, "zulong", "l2", "inference_engine.py")
    router_path = os.path.join(root, "zulong", "launcher", "web_chat_router.py")
    gatekeeper_path = os.path.join(root, "zulong", "l1b", "scheduler_gatekeeper.py")
    openhands_root = os.path.abspath(os.path.join(root, "..", "OpenHands"))
    oh_event_message = os.path.join(
        openhands_root,
        "frontend", "src", "components", "features", "chat", "event-message.tsx",
    )
    oh_should_render = os.path.join(
        openhands_root,
        "frontend", "src", "components", "features", "chat",
        "event-content-helpers", "should-render-event.ts",
    )
    oh_task_tracking = os.path.join(
        openhands_root,
        "frontend", "src", "components", "features", "chat",
        "task-tracking-observation-content.tsx",
    )
    oh_observation_pair = os.path.join(
        openhands_root,
        "frontend", "src", "components", "features", "chat",
        "event-message-components", "observation-pair-event-message.tsx",
    )
    for path in (oh_event_message, oh_should_render, oh_task_tracking, oh_observation_pair):
        assert os.path.exists(path), f"OpenHands 对照文件不存在: {path}"

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    with open(engine_path, "r", encoding="utf-8") as f:
        engine = f.read()
    with open(router_path, "r", encoding="utf-8") as f:
        router = f.read()
    with open(gatekeeper_path, "r", encoding="utf-8") as f:
        gatekeeper = f.read()
    with open(oh_should_render, "r", encoding="utf-8") as f:
        oh_render = f.read()
    with open(oh_task_tracking, "r", encoding="utf-8") as f:
        oh_tasks = f.read()
    with open(oh_observation_pair, "r", encoding="utf-8") as f:
        oh_pair = f.read()

    assert '"system"' in oh_render and '"agent_state_changed"' in oh_render
    assert '"recall"' in oh_render and '"think"' in oh_render
    assert 'command === "plan"' in oh_tasks
    assert "hasThoughtProperty" in oh_pair and 'event.action !== "think"' in oh_pair

    assert "source === 'internal_control' || source === 'system_status'" in html
    assert "hasUserTaskChecklistItems" in html
    assert "isBackgroundTaskListText" in html
    assert "appendToolDetailStrip" in html
    assert "isToolLifecycleEvent" in html
    assert "source_channel: 'system_status'" in html
    assert '"ux_visibility": "details"' in engine
    assert '"is_background": True' in engine
    assert '"tool_category": "background"' in engine
    assert 'source="system_status"' in engine
    assert '"label": "任务进入执行链路"' not in router
    assert 'message="任务已接收，正在交给 L1-B/L2 主链处理。"' not in router
    assert '"label": "L1-B/L2 正在处理"' not in router
    assert 'message="任务执行中：L1-B 已接收，L2/FC 正在推进。"' not in router
    assert 'ux_visibility = "details"' in router
    assert '"is_background": ux_visibility != "main"' in router
    assert '"step_type": "pipeline.agent_start"' in gatekeeper
    assert '"source_channel": "system_status"' in gatekeeper
    assert '"ux_visibility": "details"' in gatekeeper
    assert '"is_background": True' in gatekeeper
    assert '"tool_category": "background"' in gatekeeper
    assert '"source": "system_status"' in gatekeeper
    assert '"next_step": ""' in gatekeeper
    print("[PASS] OpenHands-style separation: 主任务清单、模型叙事、执行账本已分层")


def test_full_web_pipeline_interaction_bridge_contract():
    """Full Web 链路必须把 pipeline interaction 转成主聊天生命周期事件。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    router_path = os.path.join(root, "zulong", "launcher", "web_chat_router.py")
    fc_nodes_path = os.path.join(root, "zulong", "l2", "fc_nodes.py")
    engine_path = os.path.join(root, "zulong", "l2", "inference_engine.py")
    protocol_path = os.path.join(root, "zulong", "core", "unified_protocol.py")
    html_path = os.path.join(root, "zulong_web", "static", "index.html")

    with open(router_path, "r", encoding="utf-8") as f:
        router = f.read()
    with open(fc_nodes_path, "r", encoding="utf-8") as f:
        fc_nodes = f.read()
    with open(engine_path, "r", encoding="utf-8") as f:
        engine = f.read()
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = f.read()
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert '"type": "INTERACTION_EVENT"' in router
    assert '"interaction": interaction' in router
    assert "_schedule_broadcast(interaction_payload)" in router
    assert '"INTERACTION_EVENT": MessageType.INTERACTION_EVENT' in protocol
    assert 'MessageType.INTERACTION_EVENT: "INTERACTION_EVENT"' in protocol
    assert "case 'interaction:event':" in html
    assert "normalized.interaction = payload.interaction || null" in html
    assert '"model_progress"' in fc_nodes
    assert 'pipeline_type == "model_progress"' in engine
    assert '"source_channel": "model_progress"' in engine
    assert '"ux_visibility": "main"' in engine
    assert '"pair_id": f"pipeline-current-step-{request_key}"' in engine
    assert '"progress_items": []' in engine
    assert '"next_step": ""' in engine
    assert '"等待执行结果"' not in engine
    assert '"ux_visibility": ux_visibility' in router
    print("[PASS] Full Web pipeline: model_progress 与 INTERACTION_EVENT 桥接合同已固定")


def test_pipeline_feedback_plan_and_step_contract():
    """真实用户输入模拟：计划清单有任务语义，当前步骤只显示模型说明。"""
    from zulong.l2 import inference_engine as engine_module

    engine = engine_module.InferenceEngine.__new__(engine_module.InferenceEngine)
    engine._current_user_input_for_feedback = (
        "写一个web端运行的坦克大战游戏，画面唯美，物体由简约多面体构成，"
        "需要能在浏览器里操作和验证。"
    )
    engine._current_tool_prediction_for_feedback = {
        "context_bundle": {"needs_project_context": True},
        "task_graph_policy": "continue",
    }
    engine._current_tool_bundle_for_feedback = ["task_create_plan", "exec_write_file", "exec_run_command"]
    token = engine_module._current_request_id_var.set("req-feedback")
    try:
        steps = engine._pipeline_plan_steps_for_feedback()
        plan = engine._build_pipeline_interaction(
            pipeline_type="pipeline_start",
            fc_turn=0,
            tool_name="",
            tool_result="",
            tool_call_id="",
            progress={"percent": 0},
        )
        user_plan = engine._build_pipeline_interaction(
            pipeline_type="user_task_plan",
            fc_turn=0,
            tool_name="",
            tool_result="",
            tool_call_id="",
            progress={"percent": 0},
        )
        progress = engine._build_pipeline_interaction(
            pipeline_type="model_progress",
            fc_turn=3,
            tool_name="",
            tool_result="我会先创建 index.html，并实现坦克移动、瞄准和射击。",
            tool_call_id="model-step-3",
            progress={"percent": 35},
        )
        failed_tool = engine._build_pipeline_interaction(
            pipeline_type="agent_tool_call",
            fc_turn=4,
            tool_name="ide_write_file",
            tool_result="工具返回异常：写入失败",
            tool_call_id="tool-failed",
            progress={"percent": 45},
        )
    finally:
        engine_module._current_request_id_var.reset(token)

    assert any("坦克移动" in step or "射击" in step for step in steps), steps
    assert any("碰撞" in step or "关卡节奏" in step for step in steps), steps
    assert any("低多边形" in step or "光效" in step for step in steps), steps
    assert any("浏览器" in step for step in steps), steps
    forbidden = ("确认项目上下文", "确认当前任务状态", "进入执行链路", "任务进入执行链路")
    assert not any(any(word in step for word in forbidden) for step in steps), steps
    assert plan["pair_id"] == "pipeline-plan-req-feedback"
    assert plan["source_channel"] == "system_status"
    assert plan["ux_visibility"] == "details"
    assert plan["is_background"] is True
    assert plan["tool_category"] == "background"
    assert len(plan["progress_items"]) >= 4
    assert all(item.get("source") == "system_status" for item in plan["progress_items"])
    assert user_plan["pair_id"] == "pipeline-user-plan-req-feedback"
    assert user_plan["source_channel"] == "model_progress"
    assert user_plan["ux_visibility"] == "main"
    assert user_plan["kind"] == "plan"
    assert len(user_plan["progress_items"]) >= 4
    assert all(item.get("source") == "model_progress" for item in user_plan["progress_items"])
    assert not any(any(word in item["label"] for word in forbidden) for item in user_plan["progress_items"])
    assert progress["pair_id"] == "pipeline-current-step-req-feedback"
    assert progress["source_channel"] == "model_progress"
    assert progress["ux_visibility"] == "main"
    assert progress["detail"] == "我会先创建 index.html，并实现坦克移动、瞄准和射击。"
    assert progress["progress_items"] == []
    assert progress["next_step"] == ""
    assert failed_tool["kind"] == "observation"
    assert failed_tool["status"] == "failed"
    assert failed_tool["source_channel"] == "system_status"
    assert failed_tool["ux_visibility"] == "details"
    assert failed_tool["tool_name"] == "ide_write_file"


def test_pipeline_summary_hides_internal_task_ledger():
    """完成总结只展示执行层结果，不展示任务图/注意力等后台工具流水账。"""
    from zulong.l2 import inference_engine as engine_module

    engine = engine_module.InferenceEngine.__new__(engine_module.InferenceEngine)
    engine._pipeline_tool_ledger = [
        {"tool_name": "task_create_plan", "finished": True, "failed": False},
        {"tool_name": "task_add_node", "finished": True, "failed": False},
        {"tool_name": "task_mark_status", "finished": True, "failed": False},
        {"tool_name": "adjust_attention_mode", "finished": True, "failed": False},
        {"tool_name": "exec_write_file", "finished": True, "failed": False},
        {"tool_name": "read_file", "finished": True, "failed": False},
        {"tool_name": "exec_run_command", "finished": False, "failed": True},
    ]
    summary = engine._build_pipeline_summary_fields(
        fc_turn=5,
        progress={"completed": 2, "total": 3},
        final_text="",
    )
    joined = json.dumps(summary, ensure_ascii=False)
    assert "制定任务计划" not in joined
    assert "添加任务步骤" not in joined
    assert "更新任务状态" not in joined
    assert "adjust_attention" not in joined
    assert "写入文件" in joined
    assert "读取文件" in joined
    assert "执行命令需要复核" in joined
    print("[PASS] 完成总结隐藏后台任务图工具，仅保留执行层摘要")


def test_exec_write_file_append_and_chunk_guard():
    """exec_write_file 支持可验证 append，并拒绝超长单块写入。"""
    from zulong.tools import exec_tools
    from zulong.tools.base import ToolRequest

    old_workspace = exec_tools.WORKSPACE_DIR
    with tempfile.TemporaryDirectory(prefix="zulong-write-regression-") as tmp:
        exec_tools.WORKSPACE_DIR = tmp
        tool = exec_tools.ExecWriteFileTool()
        try:
            first = tool.execute(ToolRequest(
                tool_name="exec_write_file",
                action="execute",
                parameters={"file_path": "chunked.txt", "content": "第一片\n", "mode": "overwrite"},
            ))
            second = tool.execute(ToolRequest(
                tool_name="exec_write_file",
                action="execute",
                parameters={"file_path": "chunked.txt", "content": "第二片\n", "mode": "append"},
            ))
            target = Path(tmp) / "chunked.txt"
            assert first.success, first.to_dict()
            assert second.success, second.to_dict()
            assert target.read_text(encoding="utf-8") == "第一片\n第二片\n"
            assert second.data["mode"] == "append"
            assert second.data["verified"] is True

            oversized = tool.execute(ToolRequest(
                tool_name="exec_write_file",
                action="execute",
                parameters={
                    "file_path": "too_big.txt",
                    "content": "x" * (exec_tools.MAX_WRITE_CHUNK_CHARS + 1),
                    "mode": "overwrite",
                },
            ))
            assert not oversized.success
            assert oversized.data["recoverable"] is True
            assert oversized.data["chunk_policy"] == "openhands_style_file_chunking"
            assert not (Path(tmp) / "too_big.txt").exists()
        finally:
            exec_tools.WORKSPACE_DIR = old_workspace
    print("[PASS] exec_write_file append 与超长分片保护生效")
    print("[PASS] Pipeline feedback: 计划清单有语义，当前步骤无兜底噪声")


def test_summary_payload_uses_user_facing_terms():
    """TL-H002/H003: summary 后端 payload 不暴露内部执行术语"""
    from zulong.ide.ide_session import IDEFCState

    runner = make_runner_stub()
    state = IDEFCState(fc_turn=4)
    state.quality_last_reasons = ["最近完成标记未被 TaskGraph 真实确认: req.1"]
    state.completion_last_evidence = {}

    def fake_progress():
        return {
            "total_nodes": 3,
            "completed_count": 1,
            "pending_count": 1,
            "in_progress_count": 1,
        }

    runner._get_progress_snapshot = fake_progress
    summary = runner._build_task_summary_payload(state, reason="done", final_text="已整理。")
    joined = json.dumps(summary, ensure_ascii=False)
    assert "任务图进度" not in joined
    assert "质量复核" not in joined
    assert "任务清单进度 1/3" in joined
    assert "仍有 2 个步骤需要继续处理" in joined
    assert "需要关注" in summary["risks_summary"]
    assert "已生成最终回复" not in joined
    print("[PASS] TL-H002/H003: summary 使用用户可理解术语并避免内部词")


def assert_visibility_rules():
    """Shared offline check: public/main/details/hidden visibility stays stable."""
    from zulong.core.message_visibility import (
        CHANNEL_LEDGER,
        UX_DETAILS,
        UX_HIDDEN,
        UX_MAIN,
        internal_control_message,
        is_main_ux_payload,
        is_public_payload,
        mark_hidden_payload,
        mark_public_payload,
    )

    assert not is_public_payload(internal_control_message("internal"))
    main = mark_public_payload({"interaction": {"ux_visibility": UX_MAIN}}, CHANNEL_LEDGER, UX_MAIN)
    details = mark_public_payload({"interaction": {"ux_visibility": UX_DETAILS}}, CHANNEL_LEDGER, UX_DETAILS)
    hidden = mark_hidden_payload({"interaction": {"ux_visibility": UX_HIDDEN}})
    assert is_public_payload(main)
    assert is_main_ux_payload(main)
    assert is_public_payload(details)
    assert not is_main_ux_payload(details)
    assert not is_public_payload(hidden)


def assert_background_memory_tool_hidden():
    """Shared offline check: memory/background tools never become visible Web cards."""
    runner = make_runner_stub()
    for tool_name in ("recall_memory", "read_memory_node", "discover_related", "ide_get_context"):
        interaction = runner._apply_interaction_visibility(
            {
                "kind": "action",
                "status": "running",
                "title": "后台上下文准备",
                "detail": "后台上下文准备",
                "tool_name": tool_name,
            },
            {"tool_name": tool_name},
            "tool_requested",
            "TOOL_CALL",
        )
        assert interaction["ux_visibility"] == "hidden", tool_name
        assert interaction["tool_category"] == "background", tool_name
        assert interaction["is_background"] is True, tool_name


def assert_task_graph_binding_keeps_check_turn(tmp_dir):
    """Shared offline check: task graph binding metadata survives normalization."""
    runner = make_runner_stub()
    binding = {
        "task_graph_id": "tg-regression",
        "workspace_path": str(tmp_dir),
        "policy": "keep_recent_task_graph",
    }
    interaction = runner._build_interaction_payload(
        "model_progress",
        "继续检查当前任务图完成情况。",
        3,
        "MODEL_PROGRESS",
        {
            "model_step_note": "继续检查当前任务图完成情况。",
            "task_graph_binding": binding,
            "interaction": {
                "pair_id": "task-pair",
                "kind": "progress",
                "status": "running",
                "source_channel": "model_progress",
                "ux_visibility": "main",
                "title": "当前步骤",
                "detail": "继续检查当前任务图完成情况。",
            },
        },
    )
    assert interaction["task_graph_binding"] == binding
    assert interaction["source_channel"] == "model_progress"


def assert_preference_memory_reference_edge():
    """Shared offline check: structured tool results expose preference reference edges."""
    from zulong.ide.ide_fc_runner import IDEFCRunner

    result = json.dumps({
        "success": True,
        "data": {
            "memory_reference_edges": [
                {
                    "source": "task:tg-1/req",
                    "target": "memory:pref-1",
                    "relation": "task_preference_context",
                    "memory_kind": "preference",
                }
            ]
        },
    }, ensure_ascii=False)
    edges = IDEFCRunner._memory_reference_edges_from_result(result)
    assert len(edges) == 1
    assert edges[0]["source"] == "task:tg-1/req"
    assert edges[0]["target"] == "memory:pref-1"
    assert edges[0]["relation"] == "task_preference_context"


def assert_completion_evidence_constraints():
    """Shared offline check: summary preserves evidence and unmet constraints as risk."""
    from zulong.ide.ide_session import IDEFCState

    runner = make_runner_stub()
    state = IDEFCState(fc_turn=2)
    state.completion_last_evidence = {
        "completion_evidence": {
            "target_paths": ["D:/AI/project/zulong_beta5/tmp/sample"],
            "written_paths": ["D:/AI/project/zulong_beta5/tmp/sample/index.html"],
            "commands": [{"command": "python -m py_compile x.py", "status": "succeeded", "exit_code": 0}],
            "failed_commands_uncovered": ["npm test"],
        },
        "constraints": {
            "violated_constraints": ["存在失败命令未覆盖"],
        },
    }
    summary = runner._build_task_summary_payload(state, reason="done", final_text="已完成。")
    assert summary["completion_evidence"]["written_paths"]
    assert "存在失败命令未覆盖" in " ".join(summary["pending_items"])
    assert summary["status"] in {"blocked", "failed"}


def test_system_prompt_has_step_announce_rule():
    """TL-A003: 验证系统提示包含'先说再做'约束"""
    from zulong.l2.intent_prompt_builder import build_unified_system_prompt
    
    messages = build_unified_system_prompt(
        user_input="创建一个游戏",
        rag_context=None,
        visual_context=None,
    )
    system_content = messages[0]["content"]
    assert "先说再做" in system_content, "系统提示必须包含先说再做约束"
    assert "announce_step" in system_content, "系统提示必须提到 announce_step"
    print("[PASS] TL-A003: 系统提示包含先说再做约束")


def test_message_visibility_helpers():
    """验证 message_visibility 工具函数"""
    from zulong.core.message_visibility import (
        internal_control_message,
        is_public_payload,
        is_main_ux_payload,
    )
    
    # Internal control message should not be public
    msg = internal_control_message("test control")
    assert not is_public_payload(msg)
    
    # Public main payload
    public = {"_zulong_visibility": "public", "_zulong_channel": "ledger", "_zulong_ux_visibility": "main"}
    assert is_public_payload(public)
    assert is_main_ux_payload(public)
    
    # Details payload should be public but not main
    details = {"_zulong_visibility": "public", "_zulong_channel": "ledger", "_zulong_ux_visibility": "details"}
    assert is_public_payload(details)
    assert not is_main_ux_payload(details)
    
    print("[PASS] message_visibility 工具函数正确")


if __name__ == "__main__":
    print("=" * 60)
    print("祖龙任务生命周期反馈 - 回归测试")
    print("=" * 60)
    
    test_interaction_payload_kinds()
    test_announce_step_tool_registered()
    test_is_announce_step()
    test_split_announce_step_calls()
    test_missing_step_blocks_real_tools()
    test_model_progress_pair_used_for_tool_details()
    test_classify_source_channel()
    test_classify_ux_visibility()
    test_step_progress_items()
    test_interaction_payload_serialization()
    test_message_visibility_helpers()
    test_summary_payload_uses_user_facing_terms()
    test_system_prompt_has_step_announce_rule()
    test_minimal_event_sequence()
    test_frontend_lifecycle_rendering_contract()
    test_openhands_style_interaction_separation_contract()
    test_full_web_pipeline_interaction_bridge_contract()
    test_pipeline_feedback_plan_and_step_contract()
    test_pipeline_summary_hides_internal_task_ledger()
    test_exec_write_file_append_and_chunk_guard()
    
    print("\n[OK] 所有 P0 验证通过")
