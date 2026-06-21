const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "tmp");
fs.mkdirSync(outDir, { recursive: true });
const { chromium } = require(path.join(root, "zulong-ide", "node_modules", "playwright"));

const url = process.env.ZULONG_WEB_URL || "http://127.0.0.1:8090/";
const screenshotPath = path.join(outDir, "web_visual_surface_audit.png");
const auditPath = path.join(outDir, "web_visual_surface_audit.json");

function browserExecutable() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  ].filter(Boolean);
  return candidates.find((item) => fs.existsSync(item));
}

async function main() {
  const executablePath = browserExecutable();
  const browser = await chromium.launch({
    headless: true,
    executablePath,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const consoleMessages = [];
  page.on("console", (msg) => consoleMessages.push({ type: msg.type(), text: msg.text() }));
  page.on("pageerror", (err) => consoleMessages.push({ type: "pageerror", text: String(err && err.message || err) }));

  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForSelector("#chatMessages", { timeout: 15000 });

  const result = await page.evaluate(async () => {
    const sid = "visual-audit-session";
    const rid = "visual-audit-request";
    if (typeof currentSessionId !== "undefined") {
      if (!currentSessionId) {
        if (typeof createNewSession === "function") createNewSession();
        currentSessionId = currentSessionId || sid;
      }
    }
    const activeSid = typeof currentSessionId !== "undefined" && currentSessionId ? currentSessionId : sid;
    if (typeof currentRequestId !== "undefined") currentRequestId = rid;

    const graph = {
      id: "tg_visual_audit",
      title: "可视化验收任务图",
      owner_conversation_id: activeSid,
      owner_session_node_id: activeSid,
      workspace_path: "D:\\AI\\project\\zulong_beta5",
      metadata: {
        owner_conversation_id: activeSid,
        owner_session_node_id: activeSid,
        workspace_path: "D:\\AI\\project\\zulong_beta5",
      },
      nodes: [
        {
          id: "req",
          label: "用户交互验收",
          type: "requirement",
          status: "completed",
          desc: "确认任务卡只展示用户规划步骤",
          address: "tg:tg_visual_audit/req",
        },
        {
          id: "plan_1",
          label: "检查主任务卡",
          type: "task",
          status: "in_progress",
          desc: "后台动作折叠到执行细节，主卡保留规划步骤",
          address: "tg:tg_visual_audit/req/plan_1",
          result: "正在实机验收",
        },
        {
          id: "plan_2",
          label: "质量自查",
          type: "task",
          status: "pending",
          desc: "对照 TSD 与 OpenHands 展示规则复核",
          address: "tg:tg_visual_audit/req/plan_2",
        },
      ],
      hEdges: [["req", "plan_1"], ["req", "plan_2"]],
      dEdges: [{ s: "plan_1", t: "plan_2", via: "完成后进入质量审查", satisfied: false }],
      activeNodeId: "plan_1",
    };

    if (typeof handleMessage === "function") {
      handleMessage({
        type: "TASK_GRAPH_UPDATE",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          task_graph_id: graph.id,
          graph,
          progress: { percent: 55, completed_count: 1, total_nodes: 3 },
          workspace_path: "D:\\AI\\project\\zulong_beta5",
        },
      });
      handleMessage({
        type: "ATTENTION_UPDATE",
        request_id: rid,
        session_id: activeSid,
        payload: {
          mode: "focus",
          turn: 3,
          focus_node_id: "plan_1",
          budget_usage: 55,
          context_pressure: 0.55,
          pressure_tier: "yellow",
          task_graph_id: graph.id,
        },
      });
    } else if (typeof addTaskGraph === "function") {
      addTaskGraph(graph);
      if (typeof updateAttentionStatusPanel === "function") {
        updateAttentionStatusPanel({ modeLabel: "局部注意", focusNode: "plan_1", budget: 55, tier: "yellow", turn: 3 });
      }
    }

    if (typeof updateThoughtView === "function") {
      updateThoughtView({
        nodes: [
          { id: "plan_1", label: "当前步骤", type: "task", activation: 0.95 },
          { id: "ctx_1", label: "上下文", type: "message", activation: 0.5 },
          { id: "review_1", label: "质量审查", type: "task", activation: 0.45 },
        ],
        edges: [
          { source: "ctx_1", target: "plan_1", type: "reference", weight: 0.7 },
          { source: "plan_1", target: "review_1", type: "dependency", weight: 0.6 },
        ],
        center_ids: ["plan_1"],
      });
    }

    const wrapper = document.getElementById("appWrapper");
    if (wrapper && !wrapper.classList.contains("split-mode") && typeof toggleTaskGraph === "function") {
      toggleTaskGraph();
    }
    if (typeof switchGraphTab === "function") switchGraphTab("task");
    await new Promise((resolve) => setTimeout(resolve, 900));

    if (typeof renderGraphById === "function" && typeof getSessionGraphs === "function") {
      const graphs = getSessionGraphs();
      if (graphs && graphs[0]) renderGraphById(graphs[0].id);
    }
    await new Promise((resolve) => setTimeout(resolve, 900));

    const cyCount = typeof _cyInstance !== "undefined" && _cyInstance ? _cyInstance.nodes().length : 0;
    const detail = document.getElementById("gp-detail")?.textContent || "";
    if (typeof _cyInstance !== "undefined" && _cyInstance && _cyInstance.$id("plan_1").length) {
      _cyInstance.$id("plan_1").select();
      if (typeof updateNodeDetailPanel === "function") {
        updateNodeDetailPanel("plan_1", _cyInstance.$id("plan_1").data());
      }
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    if (typeof switchGraphPanelTab === "function") switchGraphPanelTab("gp-llm");
    if (typeof onGraphSelect === "function") {
      const graphs = typeof getSessionGraphs === "function" ? getSessionGraphs() : [];
      const g = graphs.find((item) => item.backendId === graph.id || item.title === graph.title) || graphs[0];
      if (g) {
        const map = {};
        (g.nodes || []).forEach((node) => { map[node.id] = node; });
        onGraphSelect("plan_1", map, g.dEdges || []);
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 500));

    const feedbackCardCountBefore = document.querySelectorAll("#chatMessages .interaction-card, #chatMessages .tool-detail-strip").length;
    if (typeof currentRequestShowsTaskStatus !== "undefined") currentRequestShowsTaskStatus = true;
    if (typeof handleMessage === "function") {
      handleMessage({
        type: "TURN_ACCEPTED",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          route: "coding_task",
          reason: "visual audit",
        },
      });
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-step",
            kind: "progress",
            status: "running",
            title: "当前步骤",
            detail: "正在固定展示**当前步骤**，不写入滚动区。\n\n- 已解析计划\n- 准备执行",
            source_channel: "model_progress",
            ux_visibility: "main",
            channel: "ledger",
          },
        },
      });
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-action-1",
            kind: "action",
            status: "running",
            title: "读取文件",
            detail: "读取**目标文件**并检查当前内容。",
            tool_name: "read_file",
            source_channel: "system_status",
            ux_visibility: "details",
            channel: "ledger",
            raw_details: {
              event_type: "IDE_TOOL_REQUEST",
              tool_name: "read_file",
              call_id: "call-read-1",
              tool_args: {
                path: "tmp/demo/index.html",
                mode: "read",
              },
              result_preview: "Read 128 bytes from tmp/demo/index.html",
            },
          },
        },
      });
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-step-2",
            kind: "progress",
            status: "running",
            title: "当前步骤",
            detail: "下一步固定新增第二条步骤说明。\n\n| 项 | 状态 |\n| --- | --- |\n| Markdown 表格 | 已渲染 |",
            source_channel: "model_progress",
            ux_visibility: "main",
            channel: "ledger",
          },
        },
      });
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-wait",
            kind: "progress",
            status: "running",
            title: "记录验证结果",
            detail: "执行详情：已确认第二条步骤说明完成渲染。",
            source_channel: "system_status",
            ux_visibility: "details",
            channel: "status",
          },
        },
      });
      const ledgerCountBeforeFinalNoise = document.querySelectorAll("#taskFeedbackLedger .task-feedback-entry").length;
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-final-noise",
            kind: "progress",
            status: "running",
            title: "所有任务完成！让我关联代码文件并提交最终结果。",
            detail: "所有任务完成！让我关联代码文件并提交最终结果。",
            source_channel: "model_progress",
            ux_visibility: "main",
            channel: "ledger",
          },
        },
      });
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-final-tool",
            kind: "action",
            status: "running",
            title: "整理最终回复",
            detail: "等待整理最终回复完成。",
            tool_name: "submit_final_answer",
            source_channel: "system_status",
            ux_visibility: "details",
            channel: "ledger",
          },
        },
      });
      window.__zulongFinalNoiseLedgerCountBefore = ledgerCountBeforeFinalNoise;
      handleMessage({
        type: "IDE_APPROVAL_STATUS",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          approval_id: "visual-approval-modal",
          request_id: rid,
          session_id: activeSid,
          workspace_path: "D:\\AI\\project\\zulong_beta5",
          tool_name: "exec_run_command",
          action_summary: "需要确认是否执行一次高风险命令",
          risk_level: "HIGH",
          risk_reason: "visual audit approval modal",
        },
      });
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fixed-feedback-placeholder-noise",
            kind: "progress",
            status: "running",
            title: "任务执行中",
            detail: "任务执行中，祖龙正在推进下一步。",
            source_channel: "system_status",
            ux_visibility: "details",
            channel: "status",
          },
        },
      });
      handleMessage({
        type: "STOP_ACK",
        request_id: rid,
        session_id: activeSid,
        message: "停止指令已确认",
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
    const fixedFeedbackPanel = document.getElementById("taskFeedbackPanel");
    const fixedFeedbackLedger = document.getElementById("taskFeedbackLedger");
    const fixedFeedbackText = (id) => document.getElementById(id)?.textContent?.trim() || "";
    const ledgerEntries = Array.from(document.querySelectorAll("#taskFeedbackLedger .task-feedback-entry")).map((entry) => ({
      step: entry.querySelector(".task-feedback-step-description")?.textContent?.trim() || "",
      stepHtml: entry.querySelector(".task-feedback-step-description")?.innerHTML || "",
      preview: entry.querySelector(".task-feedback-execution-preview")?.textContent?.trim() || "",
      summaryText: entry.querySelector(".task-feedback-execution summary")?.textContent?.trim() || "",
      execution: entry.querySelector(".task-feedback-execution-body")?.textContent?.trim() || "",
      executionHtml: entry.querySelector(".task-feedback-execution-body")?.innerHTML || "",
      collapsed: !entry.querySelector(".task-feedback-execution")?.hasAttribute("open"),
      executionHidden: getComputedStyle(entry.querySelector(".task-feedback-execution")).display === "none",
    }));
    const fallbackOnlyRequest = "fallback-only-request";
    const savedFixedEntries = typeof fixedFeedbackEntries !== "undefined" ? JSON.parse(JSON.stringify(fixedFeedbackEntries || [])) : null;
    const savedFixedActiveId = typeof activeFixedFeedbackEntryId !== "undefined" ? activeFixedFeedbackEntryId : "";
    const savedFixedStepCounter = typeof fixedFeedbackStepCounter !== "undefined" ? fixedFeedbackStepCounter : 0;
    if (typeof resetFixedFeedbackLedger === "function") resetFixedFeedbackLedger();
    if (typeof currentRequestId !== "undefined") currentRequestId = fallbackOnlyRequest;
    if (typeof handleMessage === "function") {
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: fallbackOnlyRequest,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: fallbackOnlyRequest,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "fallback-only-tool",
            kind: "action",
            status: "running",
            title: "写入签到表页面",
            detail: "执行详情：正在写入签到表页面文件。",
            tool_name: "exec_write_file",
            source_channel: "system_status",
            ux_visibility: "details",
            channel: "ledger",
            raw_details: {
              event_type: "IDE_TOOL_REQUEST",
              tool_name: "exec_write_file",
              call_id: "call-write-fallback",
              tool_args: {
                path: "D:\\AI\\zulong_stress_lab\\checkin\\index.html",
                mode: "overwrite",
              },
              result_preview: "Wrote 15858 bytes",
            },
          },
        },
      });
    }
    window.__zulongFallbackOnlyLedgerEntries = Array.from(document.querySelectorAll("#taskFeedbackLedger .task-feedback-entry")).map((entry) => ({
      step: entry.querySelector(".task-feedback-step-description")?.textContent?.trim() || "",
      execution: entry.querySelector(".task-feedback-execution-body")?.textContent?.trim() || "",
    }));
    if (savedFixedEntries) {
      fixedFeedbackEntries = savedFixedEntries;
      activeFixedFeedbackEntryId = savedFixedActiveId;
      fixedFeedbackStepCounter = savedFixedStepCounter;
      if (typeof updateTaskFeedbackSlot === "function") {
        updateTaskFeedbackSlot(document.getElementById("taskFeedbackWait"), "暂无等待事项。", {
          placeholder: "暂无等待事项。",
          waiting: true,
        });
      }
      if (typeof syncFixedFeedbackLedger === "function") syncFixedFeedbackLedger();
      if (typeof currentRequestId !== "undefined") currentRequestId = rid;
    }
    const feedbackCardCountAfter = document.querySelectorAll("#chatMessages .interaction-card, #chatMessages .tool-detail-strip").length;
    if (typeof handleMessage === "function") {
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: rid,
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: rid,
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "markdown-summary-card",
            kind: "summary",
            status: "succeeded",
            title: "**这轮处理完成**",
            detail: "已整理 **Markdown** 卡片。\n\n- 步骤说明已渲染\n- 具体执行已渲染",
            completed_items: ["**任务卡** 已渲染 Markdown"],
            source_channel: "model_final",
            ux_visibility: "main",
            channel: "main",
          },
        },
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
    const kickoffText = "好的！我来为你创建这个任务审计看板 Demo。先说明一下整体计划：目标：在 D:\\AI\\project\\zulong_beta5\\tmp\\zulong-interaction-audit-317db11c2b 目录下创建一个单页任务审计看板，包含添加任务、切换完成状态、过滤（全部/待办/完成）、localStorage 持久化、底部统计等功能。计划步骤：1. 创建项目目录结构 2. 编写完整的 index.html（含 HTML + CSS + JS）3. 验证文件完整性并做质量自查 现在开始第一步：创建目录。";
    if (typeof addMessage === "function") {
      addMessage(kickoffText, "assistant", false, "kickoff-structured");
    }
    if (typeof handleMessage === "function") {
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: "kickoff-progress",
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: "kickoff-progress",
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "kickoff-progress-card",
            kind: "progress",
            status: "running",
            title: kickoffText,
            detail: kickoffText,
            source_channel: "model_progress",
            ux_visibility: "main",
            channel: "ledger",
          },
        },
      });
    }
    if (typeof handleMessage === "function") {
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: "kickoff-progress",
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: "kickoff-progress",
          session_id: activeSid,
          conversation_id: activeSid,
          interaction: {
            pair_id: "merge-summary-card",
            kind: "summary",
            status: "succeeded",
            title: "这轮处理完成",
            detail: "本轮执行过程已整理。",
            completed_items: ["写入文件已完成"],
            verified_items: ["浏览器验证已完成"],
            pending_items: [],
            risks_summary: "",
            source_channel: "model_final",
            ux_visibility: "main",
            channel: "final",
          },
        },
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
    if (typeof resetActiveTaskPlanState === "function") resetActiveTaskPlanState();
    if (typeof handleMessage === "function") {
      handleMessage({
        type: "INTERACTION_EVENT",
        request_id: "summary-only-request",
        session_id: activeSid,
        conversation_id: activeSid,
        payload: {
          request_id: "summary-only-request",
          session_id: activeSid,
          conversation_id: activeSid,
          message: "summary only fallback detail",
          interaction: {
            pair_id: "summary-only-card",
            kind: "summary",
            status: "succeeded",
            title: "summary only fallback",
            detail: "**execution finished**\n\n- should restore as a task checklist card",
            source_channel: "model_final",
            ux_visibility: "main",
            channel: "final",
          },
        },
      });
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
    const summaryCard = document.querySelector('[data-pair-id="markdown-summary-card"]');
    const mergedSummaryCard = document.querySelector('[data-pair-id="merge-summary-card"]');
    const summaryOnlyCard = document.querySelector('[data-pair-id="summary-only-card"]');
    const completionProcessLedgerCount = document.querySelectorAll('[id^="task-feedback-process-ledger-"] .task-feedback-entry').length;
    const completionProcessLedgerText = Array.from(document.querySelectorAll('[id^="task-feedback-process-ledger-"]')).map((node) =>
      node.textContent || ""
    ).join("\n");
    const kickoffBubble = document.querySelector('.message.assistant[data-request-id="kickoff-structured"] .message-content');
    const kickoffPlanCard = document.querySelector('[data-pair-id="assistant-kickoff-plan:kickoff-structured"]');
    const kickoffProgressPlanCard = document.querySelector('[data-pair-id="assistant-kickoff-plan:kickoff-progress"]');
    const kickoffFirstStep = Array.from(document.querySelectorAll("#taskFeedbackLedger .task-feedback-entry")).find((entry) =>
      (entry.textContent || "").includes("现在开始第一步") || (entry.textContent || "").includes("创建目录")
    );
    const fixedFeedbackHtml = fixedFeedbackLedger?.innerHTML || "";
    const summaryHtml = summaryCard?.innerHTML || "";
    const fixedFeedbackRawText = fixedFeedbackLedger?.textContent || "";
    const chatVisibleText = document.querySelector("#chatMessages")?.textContent || "";
    const summaryVisibleText = [
      summaryCard?.querySelector(".interaction-title")?.textContent || "",
      summaryCard?.querySelector(".interaction-detail")?.textContent || "",
      summaryCard?.querySelector(".interaction-checklist")?.textContent || "",
      summaryCard?.querySelector(".interaction-meta")?.textContent || "",
      summaryCard?.querySelector(".interaction-result")?.textContent || "",
    ].join("\n");
    const chatMessagesEl = document.getElementById("chatMessages");
    const firstUserMessage = document.querySelector("#chatMessages .message.user");
    const firstAssistantContent = document.querySelector("#chatMessages .message.assistant .message-content");
    const approvalBackdrop = document.getElementById("approvalModalBackdrop");
    const approvalModal = document.getElementById("approvalModal");
    const approvalBox = approvalModal?.querySelector(".approval-message");
    document.getElementById("approvalModeBtn")?.click();
    await new Promise((resolve) => setTimeout(resolve, 100));
    const approvalModeControl = document.getElementById("approvalModeControl");
    const approvalModeBtn = document.getElementById("approvalModeBtn");
    const approvalModeMenu = document.getElementById("approvalModeMenu");
    const approvalModeRect = approvalModeBtn?.getBoundingClientRect();
    const chatInputRect = document.getElementById("chatInput")?.getBoundingClientRect();
    const taskCardDiagnostics = Array.from(document.querySelectorAll("#chatMessages .interaction-card.task-card")).map((card) => {
      const rect = card.getBoundingClientRect();
      const style = getComputedStyle(card);
      const checklistText = card.querySelector(".interaction-checklist")?.textContent?.trim() || "";
      const resultText = card.querySelector(".interaction-result")?.textContent?.trim() || "";
      return {
        pairId: card.getAttribute("data-pair-id") || "",
        title: card.querySelector(".interaction-title")?.textContent?.trim() || "",
        height: Math.round(rect.height),
        scrollHeight: card.scrollHeight,
        overflow: style.overflow,
        flex: style.flex,
        hasVisibleBody: !!(checklistText || resultText),
        checklistText,
        resultText,
      };
    });

    return {
      url: location.href,
      graphPanelOpen: !!document.getElementById("appWrapper")?.classList.contains("split-mode"),
      cyNodeCount: cyCount || (typeof _cyInstance !== "undefined" && _cyInstance ? _cyInstance.nodes().length : 0),
      graphCards: document.querySelectorAll("#graphSvgArea .tg-single-node-card").length,
      graphDetailText: document.getElementById("gp-detail")?.textContent || "",
      graphLlmText: document.getElementById("gp-llm")?.textContent || "",
      activeGraphProgress: document.getElementById("gp-ptext")?.textContent || "",
      attentionMode: document.getElementById("attentionModeLabel")?.textContent || "",
      attentionFocus: document.getElementById("attentionFocusLabel")?.textContent || "",
      attentionPressureText: document.getElementById("attentionPressureLabel")?.textContent || "",
      attentionPressure: document.getElementById("attentionStatusPanel")?.getAttribute("data-context-pressure") || "",
      thoughtNodeCountText: document.getElementById("tvNodeCount")?.textContent || "",
      thoughtDomNodes: document.querySelectorAll("#thoughtViewSvg .tv-node").length,
      fixedFeedbackVisible: !!fixedFeedbackLedger && fixedFeedbackLedger.parentElement?.id === "chatMessages",
      fixedFeedbackTopPanelVisible: !!fixedFeedbackPanel?.classList.contains("visible"),
      fixedFeedbackInsideTaskCard: !!fixedFeedbackLedger?.closest(".interaction-card.task-card"),
      fixedFeedbackInTaskStream: fixedFeedbackLedger?.parentElement?.id === "chatMessages",
      fixedFeedbackIndexCount: document.querySelectorAll("#taskFeedbackLedger .task-feedback-index").length,
      fixedFeedbackHardLabelCount: document.querySelectorAll("#taskFeedbackLedger .task-feedback-step-label").length,
      fixedFeedbackStart: fixedFeedbackText("taskFeedbackStart"),
      fixedFeedbackStep: fixedFeedbackText("taskFeedbackStep"),
      fixedFeedbackWait: fixedFeedbackText("taskFeedbackWait"),
      fallbackOnlyLedgerEntries: window.__zulongFallbackOnlyLedgerEntries || [],
      fixedFeedbackLedgerEntries: ledgerEntries,
      fixedFeedbackFinalNoiseCountBefore: window.__zulongFinalNoiseLedgerCountBefore || 0,
      chatContentMaxWidth: getComputedStyle(document.documentElement).getPropertyValue("--z-chat-content-max-width").trim(),
      chatDirectMaxWidth: firstUserMessage ? Math.round(firstUserMessage.getBoundingClientRect().width) : 0,
      assistantContentMaxWidth: firstAssistantContent ? Math.round(firstAssistantContent.getBoundingClientRect().width) : 0,
      chatMessagesAlignItems: chatMessagesEl ? getComputedStyle(chatMessagesEl).alignItems : "",
      approvalModalVisible: !!approvalBackdrop && !approvalBackdrop.classList.contains("is-hidden"),
      approvalModalText: approvalModal?.textContent?.trim() || "",
      approvalInChatStreamCount: document.querySelectorAll("#chatMessages > .approval-message, #chatMessages > .message.approval-message").length,
      approvalModalButtonCount: approvalBox?.querySelectorAll(".approval-btn").length || 0,
      approvalHeaderButtonCount: document.querySelectorAll(".chat-header #approvalModeBtn").length,
      approvalInputButtonCount: document.querySelectorAll(".chat-input-container #approvalModeBtn").length,
      approvalModeButtonText: approvalModeBtn?.textContent?.trim() || "",
      approvalModeButtonColor: approvalModeBtn ? getComputedStyle(approvalModeBtn).color : "",
      approvalModeMenuOpen: !!approvalModeControl?.classList.contains("open"),
      approvalModeMenuVisible: approvalModeMenu ? getComputedStyle(approvalModeMenu).display !== "none" : false,
      approvalModeOptionCount: document.querySelectorAll("#approvalModeMenu .approval-mode-option").length,
      approvalModeActiveCount: document.querySelectorAll("#approvalModeMenu .approval-mode-option.active").length,
      approvalModeBeforeInput: !!(approvalModeRect && chatInputRect && approvalModeRect.right <= chatInputRect.left + 2),
      chatCardGap: getComputedStyle(document.getElementById("chatMessages")).gap,
      fixedFeedbackLedgerGap: fixedFeedbackLedger ? getComputedStyle(fixedFeedbackLedger).gap : "",
      fixedFeedbackScrollCardDelta: feedbackCardCountAfter - feedbackCardCountBefore,
      fixedFeedbackMarkdownStrongCount: fixedFeedbackLedger?.querySelectorAll("strong").length || 0,
      fixedFeedbackMarkdownTableCount: fixedFeedbackLedger?.querySelectorAll("table").length || 0,
      fixedFeedbackMarkdownListCount: fixedFeedbackLedger?.querySelectorAll("ul li, ol li").length || 0,
      fixedFeedbackRawMarkdownLeak: /\*\*|\| --- \|/.test(fixedFeedbackRawText),
      visiblePlaceholderLeak: /任务执行中，祖龙正在推进下一步|祖龙正在推进下一步|等待模型补充可见步骤说明/.test(chatVisibleText),
      stopAckVisible: /停止指令已确认|任务正在终止/.test(chatVisibleText),
      fixedFeedbackHtml,
      summaryMarkdownStrongCount: summaryCard?.querySelectorAll("strong").length || 0,
      summaryMarkdownListCount: summaryCard?.querySelectorAll("ul li, ol li").length || 0,
      summaryRawMarkdownLeak: /\*\*/.test(summaryVisibleText),
      summaryHtml,
      mergedSummaryStandalone: !!mergedSummaryCard,
      mergedPlanText: kickoffProgressPlanCard?.textContent?.trim() || "",
      mergedPlanChecklistText: kickoffProgressPlanCard?.querySelector(".interaction-checklist")?.textContent?.trim() || "",
      mergedPlanResultText: kickoffProgressPlanCard?.querySelector(".interaction-result")?.textContent?.trim() || "",
      summaryOnlyClassName: summaryOnlyCard?.className || "",
      summaryOnlyTitle: summaryOnlyCard?.querySelector(".interaction-title")?.textContent?.trim() || "",
      summaryOnlyChecklistText: summaryOnlyCard?.querySelector(".interaction-checklist")?.textContent?.trim() || "",
      summaryOnlyResultText: summaryOnlyCard?.querySelector(".interaction-result")?.textContent?.trim() || "",
      summaryOnlyMarkdownStrongCount: summaryOnlyCard?.querySelectorAll("strong").length || 0,
      summaryOnlyMarkdownListCount: summaryOnlyCard?.querySelectorAll("ul li, ol li").length || 0,
      summaryOnlyRawMarkdownLeak: /\*\*/.test(summaryOnlyCard?.textContent || ""),
      completionProcessLedgerCount,
      completionProcessLedgerText,
      taskCardDiagnostics,
      kickoffBubbleText: kickoffBubble?.textContent?.trim() || "",
      kickoffPlanCardText: kickoffPlanCard?.textContent?.trim() || "",
      kickoffPlanItemCount: kickoffPlanCard?.querySelectorAll(".interaction-check-item").length || 0,
      kickoffProgressPlanCardText: kickoffProgressPlanCard?.textContent?.trim() || "",
      kickoffProgressPlanItemCount: kickoffProgressPlanCard?.querySelectorAll(".interaction-check-item").length || 0,
      kickoffFirstStepText: kickoffFirstStep?.querySelector(".task-feedback-step-description")?.textContent?.trim() || "",
      kickoffProgressCardText: document.querySelector('[data-pair-id="kickoff-progress-card"]')?.textContent?.trim() || "",
      kickoffOrder: {
        bubbleBeforePlan: !!(kickoffBubble && kickoffPlanCard && (kickoffBubble.closest(".message").compareDocumentPosition(kickoffPlanCard) & Node.DOCUMENT_POSITION_FOLLOWING)),
        planBeforeFirstStep: !!(kickoffPlanCard && kickoffFirstStep && (kickoffPlanCard.compareDocumentPosition(kickoffFirstStep) & Node.DOCUMENT_POSITION_FOLLOWING)),
      },
      computedColors: {
        bodyBackground: getComputedStyle(document.body).backgroundColor,
        graphPanelBackground: getComputedStyle(document.getElementById("graphPanel")).backgroundColor,
        attentionBackground: getComputedStyle(document.getElementById("attentionStatusPanel")).backgroundColor,
        attentionBorder: getComputedStyle(document.getElementById("attentionStatusPanel")).borderColor,
      },
    };
  });

  const replayResult = await page.evaluate(async () => {
    const kickoffText = "好的！我来为你创建这个任务审计看板 Demo。先说明一下整体计划：目标：在 D:\\AI\\project\\zulong_beta5\\tmp\\zulong-interaction-audit-317db11c2b 目录下创建一个单页任务审计看板，包含添加任务、切换完成状态、过滤（全部/待办/完成）、localStorage 持久化、底部统计等功能。计划步骤：1. 创建项目目录结构 2. 编写完整的 index.html（含 HTML + CSS + JS）3. 验证文件完整性并做质量自查 现在开始第一步：创建目录。";
    if (typeof renderLocalSessionMessages === "function") {
      renderLocalSessionMessages([{
        id: "history-kickoff-message",
        text: kickoffText,
        sender: "assistant",
        request_id: "history-kickoff",
      }]);
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
    const bubble = document.querySelector('.message.assistant[data-request-id="history-kickoff"] .message-content');
    const planCard = document.querySelector('[data-pair-id="assistant-kickoff-plan:history-kickoff"]');
    const firstStepLedger = document.getElementById("assistant-first-step-ledger-history-kickoff");
    const firstStep = Array.from((firstStepLedger || document).querySelectorAll(".task-feedback-entry")).find((entry) =>
      (entry.textContent || "").includes("现在开始第一步") || (entry.textContent || "").includes("创建目录")
    );
    const liveLedgerText = document.getElementById("taskFeedbackLedger")?.textContent || "";
    return {
      bubbleText: bubble?.textContent?.trim() || "",
      planCardText: planCard?.textContent?.trim() || "",
      planItemCount: planCard?.querySelectorAll(".interaction-check-item").length || 0,
      firstStepText: firstStep?.querySelector(".task-feedback-step-description")?.textContent?.trim() || "",
      entryCount: document.querySelectorAll("#taskFeedbackLedger .task-feedback-entry").length,
      historyLedgerCount: document.querySelectorAll(".task-feedback-history-ledger").length,
      liveLedgerText,
      order: {
        bubbleBeforePlan: !!(bubble && planCard && (bubble.closest(".message").compareDocumentPosition(planCard) & Node.DOCUMENT_POSITION_FOLLOWING)),
        planBeforeFirstStep: !!(planCard && firstStep && (planCard.compareDocumentPosition(firstStep) & Node.DOCUMENT_POSITION_FOLLOWING)),
      },
    };
  });
  result.kickoffReplay = replayResult;

  const sessionHistoryResult = await page.evaluate(async () => {
    const sid = "history-event-restore-session";
    const now = Date.now() / 1000;
    if (typeof currentSessionId !== "undefined") currentSessionId = sid;
    if (typeof renderSessionMessages === "function") {
      renderSessionMessages(sid, [], [
        {
          node_id: "hist-u-old",
          event_id: "hist-u-old",
          event_type: "user_message",
          role: "user",
          text: "旧任务，不应该恢复过程",
          content: "旧任务，不应该恢复过程",
          created_at: now,
          payload: {},
        },
        {
          node_id: "hist-old-progress",
          event_id: "hist-old-progress",
          event_type: "pipeline.model_progress",
          role: "tool",
          text: "当前步骤 旧任务步骤",
          content: "当前步骤 旧任务步骤",
          created_at: now + 1,
          payload: {
            interaction: {
              pair_id: "hist-old-progress",
              kind: "progress",
              status: "running",
              title: "当前步骤",
              detail: "旧任务步骤",
              source_channel: "model_progress",
              ux_visibility: "main",
              channel: "ledger",
            },
          },
        },
        {
          node_id: "hist-u-latest",
          event_id: "hist-u-latest",
          event_type: "user_message",
          role: "user",
          text: "写一个签到表，web页面",
          content: "写一个签到表，web页面",
          created_at: now + 2,
          payload: {},
        },
        {
          node_id: "hist-step-1",
          event_id: "hist-step-1",
          event_type: "pipeline.model_progress",
          role: "tool",
          text: "当前步骤 好的，我来帮你做一个签到表 Web 页面。",
          content: "当前步骤 好的，我来帮你做一个签到表 Web 页面。",
          created_at: now + 3,
          payload: {
            interaction: {
              pair_id: "hist-step-1",
              kind: "progress",
              status: "running",
              title: "当前步骤",
              detail: "好的，我来帮你做一个签到表 Web 页面。",
              source_channel: "model_progress",
              ux_visibility: "main",
              channel: "ledger",
            },
          },
        },
        {
          node_id: "hist-step-2",
          event_id: "hist-step-2",
          event_type: "pipeline.model_progress",
          role: "tool",
          text: "当前步骤 创建 checkin 目录并写入完整 HTML。",
          content: "当前步骤 创建 checkin 目录并写入完整 HTML。",
          created_at: now + 4,
          payload: {
            interaction: {
              pair_id: "hist-step-2",
              kind: "progress",
              status: "running",
              title: "当前步骤",
              detail: "创建 `checkin` 目录并写入完整 HTML。",
              source_channel: "model_progress",
              ux_visibility: "main",
              channel: "ledger",
            },
          },
        },
        {
          node_id: "hist-tool",
          event_id: "hist-tool",
          event_type: "pipeline.agent_tool_call",
          role: "tool",
          text: "使用 exec_write_file 等待 exec_write_file 返回。",
          content: "使用 exec_write_file 等待 exec_write_file 返回。",
          created_at: now + 5,
          payload: {
            interaction: {
              pair_id: "hist-tool",
              kind: "action",
              status: "running",
              title: "写入签到表页面",
              detail: "写入 `checkin/index.html`。",
              tool_name: "exec_write_file",
              source_channel: "system_status",
              ux_visibility: "details",
              channel: "ledger",
              raw_details: {
                event_type: "IDE_TOOL_REQUEST",
                tool_name: "exec_write_file",
                call_id: "hist-write-file",
                tool_args: {
                  path: "D:\\AI\\zulong_stress_lab\\checkin\\index.html",
                  mode: "overwrite",
                },
              },
            },
          },
        },
        {
          node_id: "hist-summary",
          event_id: "hist-summary",
          event_type: "pipeline.agent_done",
          role: "tool",
          text: "这轮处理完成 共推进 5 轮。",
          content: "这轮处理完成 共推进 5 轮。",
          created_at: now + 6,
          payload: {
            interaction: {
              pair_id: "hist-summary",
              kind: "summary",
              status: "succeeded",
              title: "这轮处理完成",
              detail: "共推进 5 轮。",
              source_channel: "model_final",
              ux_visibility: "main",
              channel: "final",
            },
          },
        },
        {
          node_id: "hist-final",
          event_id: "hist-final",
          event_type: "assistant_message",
          role: "assistant",
          text: "签到表页面已经做好了！",
          content: "签到表页面已经做好了！",
          created_at: now + 7,
          payload: {},
        },
      ]);
    }
    await new Promise((resolve) => setTimeout(resolve, 150));
    const historyLedger = document.querySelector(".task-feedback-history-ledger");
    const entries = Array.from(document.querySelectorAll(".task-feedback-history-ledger .task-feedback-entry")).map((entry) => ({
      step: entry.querySelector(".task-feedback-step-description")?.textContent?.trim() || "",
      execution: entry.querySelector(".task-feedback-execution-body")?.textContent?.trim() || "",
      collapsed: !entry.querySelector(".task-feedback-execution")?.hasAttribute("open"),
    }));
    const finalMessage = Array.from(document.querySelectorAll("#chatMessages .message.assistant")).find((node) =>
      (node.textContent || "").includes("签到表页面已经做好了")
    );
    return {
      chatText: document.getElementById("chatMessages")?.textContent || "",
      ledgerExists: !!historyLedger,
      entries,
      entryCount: entries.length,
      finalExists: !!finalMessage,
      ledgerBeforeFinal: !!(historyLedger && finalMessage && (historyLedger.compareDocumentPosition(finalMessage) & Node.DOCUMENT_POSITION_FOLLOWING)),
      liveLedgerText: document.getElementById("taskFeedbackLedger")?.textContent || "",
    };
  });
  result.sessionHistoryRestore = sessionHistoryResult;

  await page.screenshot({ path: screenshotPath, fullPage: true });
  const failures = [];
  if (!result.graphPanelOpen) failures.push("graph panel is not open");
  if ((result.cyNodeCount + result.graphCards) < 1) failures.push("task graph did not render nodes");
  if (!result.graphDetailText.includes("节点地址")) failures.push("node detail panel missing address");
  if (!result.graphLlmText.includes("LLM 聚焦上下文")) failures.push("LLM focus panel missing");
  if (!result.attentionMode.includes("局部注意")) failures.push("attention mode missing");
  if (result.attentionPressure !== "55") failures.push("context pressure missing");
  if (!result.thoughtNodeCountText.includes("3") && result.thoughtDomNodes < 3) failures.push("thought view nodes missing");
  if (!result.fixedFeedbackInTaskStream) failures.push("fixed feedback ledger is not in the task stream");
  if (result.fixedFeedbackTopPanelVisible) failures.push("fixed feedback ledger leaked into the top standalone panel");
  if (result.fixedFeedbackInsideTaskCard) failures.push("fixed feedback ledger is still nested inside a task card");
  const configuredContentMax = parseFloat(result.chatContentMaxWidth || "0") || 1080;
  if (result.chatMessagesAlignItems !== "center") failures.push("chat messages are not centered within the max content column");
  if (result.chatDirectMaxWidth > configuredContentMax + 2) failures.push("opening/user message exceeds max content width");
  if (result.assistantContentMaxWidth > configuredContentMax + 2) failures.push("assistant content exceeds max content width");
  if (!result.approvalModalVisible) failures.push("approval request did not open in a dialog");
  if (result.approvalInChatStreamCount !== 0) failures.push("approval request still appears in the chat stream");
  if (result.approvalModalButtonCount < 2) failures.push("approval dialog lost approve/reject buttons");
  if (result.approvalHeaderButtonCount !== 0) failures.push("approval mode button still appears in the header");
  if (result.approvalInputButtonCount !== 1) failures.push("approval mode button is not in the input bar");
  if (!/完全访问/.test(result.approvalModeButtonText || "")) failures.push("approval mode button does not show the current permission label");
  if (!result.approvalModeBeforeInput) failures.push("approval mode button is not positioned to the left of the input field");
  if (!result.approvalModeMenuOpen || !result.approvalModeMenuVisible) failures.push("approval mode menu did not open from the input bar button");
  if (result.approvalModeOptionCount < 4) failures.push("approval mode menu options are incomplete");
  if (result.approvalModeActiveCount !== 1) failures.push("approval mode menu should show exactly one active option");
  if (result.fixedFeedbackIndexCount !== 0) failures.push("fixed ledger still shows step indexes");
  if (result.fixedFeedbackHardLabelCount !== 0) failures.push("fixed ledger still shows hard-coded step labels");
  if (result.fixedFeedbackLedgerEntries.length < 2) failures.push("fixed ledger did not append visible model steps");
  if (Math.abs(parseFloat(result.chatCardGap || "0") - 48) > 0.5) failures.push("chat card gap is not 48px");
  if (Math.abs(parseFloat(result.fixedFeedbackLedgerGap || "0") - 37.333) > 0.75) failures.push("fixed ledger gap is not 37.333px");
  if (result.fixedFeedbackFinalNoiseCountBefore && result.fixedFeedbackLedgerEntries.length !== result.fixedFeedbackFinalNoiseCountBefore) failures.push("final-answer noise created a fixed ledger step");
  if (result.fixedFeedbackLedgerEntries.some((entry) => /所有任务完成|最终回复|提交最终结果/.test(entry.step + "\n" + entry.execution))) {
    failures.push("final-answer noise leaked into fixed ledger");
  }
  if (result.fixedFeedbackLedgerEntries.some((entry) => /任务已开始|任务已接收|确认目标/.test(entry.step))) {
    failures.push("background accepted status leaked into fixed ledger");
  }
  if (result.fixedFeedbackLedgerEntries.some((entry) => /已收到任务|开始处理/.test(entry.step + "\n" + entry.execution))) {
    failures.push("hard-coded start feedback leaked into fixed ledger");
  }
  const fallbackOnlyEntry = (result.fallbackOnlyLedgerEntries || [])[0];
  if (!fallbackOnlyEntry?.step.includes("写入签到表页面")) failures.push("tool-only fallback process card missing");
  if (/已收到任务|开始处理|祖龙正在推进/.test(fallbackOnlyEntry?.step || "")) failures.push("tool-only fallback used hard-coded lifecycle text");
  if (!fallbackOnlyEntry?.execution.includes("Tool name: exec_write_file")) failures.push("tool-only fallback lost tool name detail");
  if (!/checkin\\+index\.html/.test(fallbackOnlyEntry?.execution || "")) failures.push("tool-only fallback lost write path detail");
  if (!result.fixedFeedbackLedgerEntries[0]?.step.includes("固定展示当前步骤")) failures.push("first visible ledger entry is not the real model step");
  const firstWorkEntry = result.fixedFeedbackLedgerEntries.find((entry) => entry.step.includes("固定展示当前步骤"));
  const secondWorkEntry = result.fixedFeedbackLedgerEntries.find((entry) => entry.step.includes("第二条步骤说明"));
  if (!firstWorkEntry) failures.push("first fixed ledger step missing");
  if (!firstWorkEntry?.execution.includes("Tool name: read_file")) failures.push("first fixed ledger lost tool name detail");
  if (!firstWorkEntry?.execution.includes("Event type: IDE_TOOL_REQUEST")) failures.push("first fixed ledger lost event type detail");
  if (!firstWorkEntry?.execution.includes("Arguments:")) failures.push("first fixed ledger lost tool arguments detail");
  if (!firstWorkEntry?.execution.includes("tmp/demo/index.html")) failures.push("first fixed ledger lost tool path detail");
  if (!firstWorkEntry?.execution.includes("读取目标文件")) failures.push("first fixed ledger execution missing");
  if (!firstWorkEntry?.preview.includes("读取目标文件")) failures.push("first fixed ledger execution preview missing");
  if (!secondWorkEntry) failures.push("second fixed ledger step missing");
  if (!secondWorkEntry?.execution.includes("第二条步骤说明完成渲染")) failures.push("second fixed ledger execution missing");
  if (!result.fixedFeedbackLedgerEntries.every((entry) => entry.collapsed)) failures.push("fixed ledger execution is not collapsed by default");
  if (result.fixedFeedbackLedgerEntries.some((entry) => /具体执行|待执行/.test(entry.summaryText))) {
    failures.push("fixed ledger still shows hard-coded execution summary/status");
  }
  if (result.fixedFeedbackLedgerEntries.some((entry) => entry.execution && entry.executionHidden)) {
    failures.push("fixed ledger hides execution block even when execution content exists");
  }
  if (result.fixedFeedbackScrollCardDelta !== 0) failures.push("fixed feedback leaked into scroll stream");
  if (result.fixedFeedbackMarkdownStrongCount < 1) failures.push("fixed ledger markdown strong text did not render");
  if (result.fixedFeedbackMarkdownTableCount < 1) failures.push("fixed ledger markdown table did not render");
  if (result.fixedFeedbackMarkdownListCount < 1) failures.push("fixed ledger markdown list did not render");
  if (result.fixedFeedbackRawMarkdownLeak) failures.push("fixed ledger still leaks raw markdown syntax");
  if (result.visiblePlaceholderLeak) failures.push("frontend lifecycle placeholder leaked into visible chat");
  if (result.stopAckVisible) failures.push("STOP_ACK control message leaked into visible chat");
  if (result.completionProcessLedgerCount < 3) failures.push("completion did not preserve per-step process ledger");
  if (!/Tool name: read_file/.test(result.completionProcessLedgerText || "")) failures.push("completion process ledger lost tool call details");
  if (result.mergedSummaryStandalone) failures.push("completion summary created a standalone card instead of merging into task checklist");
  if (!/task-card/.test(result.summaryOnlyClassName || "")) failures.push("summary-only fallback did not render as a task card");
  if (!/execution finished/.test((result.summaryOnlyChecklistText || "") + "\n" + (result.summaryOnlyResultText || ""))) failures.push("summary-only fallback task card lost its summary detail");
  if (result.summaryOnlyMarkdownStrongCount < 1) failures.push("summary-only task card markdown strong text did not render");
  if (result.summaryOnlyMarkdownListCount < 1) failures.push("summary-only task card markdown list did not render");
  if (result.summaryOnlyRawMarkdownLeak) failures.push("summary-only task card still leaks raw markdown syntax");
  const clippedTaskCards = (result.taskCardDiagnostics || []).filter((card) =>
    card.hasVisibleBody &&
    (card.height < 80 || card.height + 8 < card.scrollHeight || card.overflow === "hidden")
  );
  if (clippedTaskCards.length) failures.push(`task checklist card is clipped: ${clippedTaskCards.map((card) => card.pairId || card.title).join(", ")}`);
  if (!/任务清单/.test(result.mergedPlanText || "") || !/执行完成[:：].*写入文件/.test(result.mergedPlanResultText || "")) {
    failures.push("completion execution summary did not merge into the active task checklist");
  }
  if (!result.mergedPlanChecklistText.includes("创建项目目录结构")) failures.push("active task checklist lost the original plan steps after summary merge");
  if (/写入文件已完成|浏览器验证已完成/.test(result.mergedPlanChecklistText || "")) {
    failures.push("execution summary leaked into task checklist items");
  }
  if (/计划步骤|现在开始第一步/.test(result.kickoffBubbleText)) failures.push("assistant kickoff bubble still contains inline plan/first step");
  if (!result.kickoffPlanCardText.includes("创建项目目录结构")) failures.push("kickoff plan did not create a separate plan card");
  if (result.kickoffPlanItemCount < 3) failures.push("kickoff plan card did not render plan steps");
  if (!result.kickoffProgressPlanCardText.includes("创建项目目录结构")) failures.push("model_progress kickoff plan did not create a separate plan card");
  if (result.kickoffProgressPlanItemCount < 3) failures.push("model_progress kickoff plan card did not render plan steps");
  if (/计划步骤|现在开始第一步/.test(result.kickoffProgressCardText)) failures.push("model_progress kickoff leaked inline plan card");
  if (!result.kickoffFirstStepText.includes("创建目录")) failures.push("kickoff first step did not become a separate ledger card");
  if (!result.kickoffOrder.bubbleBeforePlan) failures.push("kickoff plan card is not after the opening bubble");
  if (!result.kickoffOrder.planBeforeFirstStep) failures.push("kickoff first step is not after the plan card");
  if (/计划步骤|现在开始第一步/.test(result.kickoffReplay?.bubbleText || "")) failures.push("history kickoff bubble still contains inline plan/first step");
  if (!result.kickoffReplay?.planCardText?.includes("创建项目目录结构")) failures.push("history kickoff plan card missing");
  if ((result.kickoffReplay?.planItemCount || 0) < 3) failures.push("history kickoff plan steps missing");
  if (!result.kickoffReplay?.firstStepText?.includes("创建目录")) failures.push("history kickoff first step card missing");
  if ((result.kickoffReplay?.historyLedgerCount || 0) < 1) failures.push("history kickoff first step did not use a history ledger");
  if (/创建目录/.test(result.kickoffReplay?.liveLedgerText || "")) failures.push("history kickoff polluted the live fixed ledger");
  if (!result.kickoffReplay?.order?.bubbleBeforePlan) failures.push("history kickoff plan card is not after the opening bubble");
  if (!result.kickoffReplay?.order?.planBeforeFirstStep) failures.push("history kickoff first step is not after the plan card");
  if (!result.sessionHistoryRestore?.ledgerExists) failures.push("SESSION_MESSAGES history events did not restore a process ledger");
  if ((result.sessionHistoryRestore?.entryCount || 0) < 2) failures.push("SESSION_MESSAGES history events restored too few process steps");
  if (!/签到表 Web 页面|checkin/.test((result.sessionHistoryRestore?.entries || []).map((entry) => entry.step).join("\n"))) {
    failures.push("SESSION_MESSAGES history events lost latest task process steps");
  }
  if (/旧任务步骤/.test(result.sessionHistoryRestore?.chatText || "")) failures.push("SESSION_MESSAGES history restore leaked an older turn process step");
  if (!/Tool name: exec_write_file/.test((result.sessionHistoryRestore?.entries || []).map((entry) => entry.execution).join("\n"))) {
    failures.push("SESSION_MESSAGES history events lost tool call details");
  }
  if (!/checkin\\+index\.html/.test((result.sessionHistoryRestore?.entries || []).map((entry) => entry.execution).join("\n"))) {
    failures.push("SESSION_MESSAGES history events lost tool write path");
  }
  if (!result.sessionHistoryRestore?.entries?.every((entry) => entry.collapsed)) failures.push("SESSION_MESSAGES restored execution details are not collapsed");
  if (!result.sessionHistoryRestore?.finalExists) failures.push("SESSION_MESSAGES history restore lost final assistant message");
  if (!result.sessionHistoryRestore?.ledgerBeforeFinal) failures.push("SESSION_MESSAGES restored process ledger is not before the final answer");
  if (/签到表 Web 页面|checkin/.test(result.sessionHistoryRestore?.liveLedgerText || "")) {
    failures.push("SESSION_MESSAGES history restore polluted the live fixed ledger");
  }

  const audit = {
    ok: failures.length === 0,
    failures,
    result,
    console_messages: consoleMessages.slice(-80),
    screenshot: screenshotPath,
  };
  fs.writeFileSync(auditPath, JSON.stringify(audit, null, 2), "utf8");
  await browser.close();
  if (failures.length) {
    console.error(JSON.stringify(audit, null, 2));
    process.exit(1);
  }
  console.log(JSON.stringify(audit, null, 2));
}

main().catch((err) => {
  fs.writeFileSync(auditPath, JSON.stringify({ ok: false, error: String(err && err.stack || err) }, null, 2), "utf8");
  console.error(err);
  process.exit(1);
});
