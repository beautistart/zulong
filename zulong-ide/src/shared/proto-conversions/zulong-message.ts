import { ZulongAsk as AppZulongAsk, ZulongMessage as AppZulongMessage, ZulongSay as AppZulongSay } from "@shared/ExtensionMessage"
import { ClineAsk, ClineMessageType, ClineSay, ClineMessage as ProtoClineMessage } from "@shared/proto/zulong/ui"

// Helper function to convert ZulongAsk string to enum
function convertZulongAskToProtoEnum(ask: AppZulongAsk | undefined): ClineAsk | undefined {
	if (!ask) {
		return undefined
	}

	const mapping: Record<AppZulongAsk, ClineAsk> = {
		followup: ClineAsk.FOLLOWUP,
		plan_mode_respond: ClineAsk.PLAN_MODE_RESPOND,
		act_mode_respond: ClineAsk.ACT_MODE_RESPOND,
		command: ClineAsk.COMMAND,
		command_output: ClineAsk.COMMAND_OUTPUT,
		completion_result: ClineAsk.COMPLETION_RESULT,
		tool: ClineAsk.TOOL,
		api_req_failed: ClineAsk.API_REQ_FAILED,
		resume_task: ClineAsk.RESUME_TASK,
		resume_completed_task: ClineAsk.RESUME_COMPLETED_TASK,
		mistake_limit_reached: ClineAsk.MISTAKE_LIMIT_REACHED,
		browser_action_launch: ClineAsk.BROWSER_ACTION_LAUNCH,
		use_mcp_server: ClineAsk.USE_MCP_SERVER,
		new_task: ClineAsk.NEW_TASK,
		condense: ClineAsk.CONDENSE,
		summarize_task: ClineAsk.SUMMARIZE_TASK,
		report_bug: ClineAsk.REPORT_BUG,
		use_subagents: ClineAsk.USE_SUBAGENTS,
	}

	return mapping[ask]
}

// Helper function to convert ZulongAsk enum to string
function convertProtoEnumToZulongAsk(ask: ClineAsk): AppZulongAsk | undefined {
	if (ask === ClineAsk.UNRECOGNIZED) {
		return undefined
	}

	const mapping: Record<Exclude<ClineAsk, ClineAsk.UNRECOGNIZED>, AppZulongAsk> = {
		[ClineAsk.FOLLOWUP]: "followup",
		[ClineAsk.PLAN_MODE_RESPOND]: "plan_mode_respond",
		[ClineAsk.ACT_MODE_RESPOND]: "act_mode_respond",
		[ClineAsk.COMMAND]: "command",
		[ClineAsk.COMMAND_OUTPUT]: "command_output",
		[ClineAsk.COMPLETION_RESULT]: "completion_result",
		[ClineAsk.TOOL]: "tool",
		[ClineAsk.API_REQ_FAILED]: "api_req_failed",
		[ClineAsk.RESUME_TASK]: "resume_task",
		[ClineAsk.RESUME_COMPLETED_TASK]: "resume_completed_task",
		[ClineAsk.MISTAKE_LIMIT_REACHED]: "mistake_limit_reached",
		[ClineAsk.BROWSER_ACTION_LAUNCH]: "browser_action_launch",
		[ClineAsk.USE_MCP_SERVER]: "use_mcp_server",
		[ClineAsk.NEW_TASK]: "new_task",
		[ClineAsk.CONDENSE]: "condense",
		[ClineAsk.SUMMARIZE_TASK]: "summarize_task",
		[ClineAsk.REPORT_BUG]: "report_bug",
		[ClineAsk.USE_SUBAGENTS]: "use_subagents",
		[ClineAsk.QUESTION]: "followup",
		[ClineAsk.AUTO_APPROVAL_LIMIT_REACHED]: "resume_task",
		[ClineAsk.PLAN_MODE_RESUME]: "resume_task",
		[ClineAsk.CLINE_IGNORE_RULES]: "followup",
	}

	return mapping[ask]
}

// Helper function to convert ZulongSay string to enum
function convertZulongSayToProtoEnum(say: AppZulongSay | undefined): ClineSay | undefined {
	if (!say) {
		return undefined
	}

	const mapping: Record<AppZulongSay, ClineSay> = {
		task: ClineSay.TASK,
		error: ClineSay.ERROR,
		api_req_started: ClineSay.API_REQ_STARTED,
		api_req_finished: ClineSay.API_REQ_FINISHED,
		text: ClineSay.TEXT,
		reasoning: ClineSay.REASONING,
		completion_result: ClineSay.COMPLETION_RESULT_SAY,
		user_feedback: ClineSay.USER_FEEDBACK,
		user_feedback_diff: ClineSay.USER_FEEDBACK_DIFF,
		api_req_retried: ClineSay.API_REQ_RETRIED,
		command: ClineSay.COMMAND_SAY,
		command_output: ClineSay.COMMAND_OUTPUT_SAY,
		tool: ClineSay.TOOL_SAY,
		shell_integration_warning: ClineSay.SHELL_INTEGRATION_WARNING,
		shell_integration_warning_with_suggestion: ClineSay.SHELL_INTEGRATION_WARNING,
		browser_action_launch: ClineSay.BROWSER_ACTION_LAUNCH_SAY,
		browser_action: ClineSay.BROWSER_ACTION,
		browser_action_result: ClineSay.BROWSER_ACTION_RESULT,
		mcp_server_request_started: ClineSay.MCP_SERVER_REQUEST_STARTED,
		mcp_server_response: ClineSay.MCP_SERVER_RESPONSE,
		mcp_notification: ClineSay.MCP_NOTIFICATION,
		use_mcp_server: ClineSay.USE_MCP_SERVER_SAY,
		diff_error: ClineSay.DIFF_ERROR,
		deleted_api_reqs: ClineSay.DELETED_API_REQS,
		zulongignore_error: ClineSay.CLINEIGNORE_ERROR,
		command_permission_denied: ClineSay.COMMAND_PERMISSION_DENIED,
		checkpoint_created: ClineSay.CHECKPOINT_CREATED,
		load_mcp_documentation: ClineSay.LOAD_MCP_DOCUMENTATION,
		info: ClineSay.INFO,
		task_progress: ClineSay.TASK_PROGRESS,
		error_retry: ClineSay.ERROR_RETRY,
		hook_status: ClineSay.HOOK_STATUS,
		hook_output_stream: ClineSay.HOOK_OUTPUT_STREAM,
		conditional_rules_applied: ClineSay.CONDITIONAL_RULES_APPLIED,
		subagent: ClineSay.SUBAGENT_STATUS,
		use_subagents: ClineSay.USE_SUBAGENTS_SAY,
		subagent_usage: ClineSay.SUBAGENT_USAGE,
		generate_explanation: ClineSay.GENERATE_EXPLANATION,
	}

	return mapping[say]
}

// Helper function to convert ZulongSay enum to string
function convertProtoEnumToZulongSay(say: ClineSay): AppZulongSay | undefined {
	if (say === ClineSay.UNRECOGNIZED) {
		return undefined
	}

	const mapping: Record<Exclude<ClineSay, ClineSay.UNRECOGNIZED>, AppZulongSay> = {
		[ClineSay.TASK]: "task",
		[ClineSay.ERROR]: "error",
		[ClineSay.API_REQ_STARTED]: "api_req_started",
		[ClineSay.API_REQ_FINISHED]: "api_req_finished",
		[ClineSay.TEXT]: "text",
		[ClineSay.REASONING]: "reasoning",
		[ClineSay.COMPLETION_RESULT_SAY]: "completion_result",
		[ClineSay.USER_FEEDBACK]: "user_feedback",
		[ClineSay.USER_FEEDBACK_DIFF]: "user_feedback_diff",
		[ClineSay.API_REQ_RETRIED]: "api_req_retried",
		[ClineSay.COMMAND_SAY]: "command",
		[ClineSay.COMMAND_OUTPUT_SAY]: "command_output",
		[ClineSay.TOOL_SAY]: "tool",
		[ClineSay.SHELL_INTEGRATION_WARNING]: "shell_integration_warning",
		[ClineSay.BROWSER_ACTION_LAUNCH_SAY]: "browser_action_launch",
		[ClineSay.BROWSER_ACTION]: "browser_action",
		[ClineSay.BROWSER_ACTION_RESULT]: "browser_action_result",
		[ClineSay.MCP_SERVER_REQUEST_STARTED]: "mcp_server_request_started",
		[ClineSay.MCP_SERVER_RESPONSE]: "mcp_server_response",
		[ClineSay.MCP_NOTIFICATION]: "mcp_notification",
		[ClineSay.USE_MCP_SERVER_SAY]: "use_mcp_server",
		[ClineSay.DIFF_ERROR]: "diff_error",
		[ClineSay.DELETED_API_REQS]: "deleted_api_reqs",
		[ClineSay.CLINEIGNORE_ERROR]: "zulongignore_error",
		[ClineSay.COMMAND_PERMISSION_DENIED]: "command_permission_denied",
		[ClineSay.CHECKPOINT_CREATED]: "checkpoint_created",
		[ClineSay.LOAD_MCP_DOCUMENTATION]: "load_mcp_documentation",
		[ClineSay.INFO]: "info",
		[ClineSay.TASK_PROGRESS]: "task_progress",
		[ClineSay.ERROR_RETRY]: "error_retry",
		[ClineSay.GENERATE_EXPLANATION]: "generate_explanation",
		[ClineSay.HOOK_STATUS]: "hook_status",
		[ClineSay.HOOK_OUTPUT_STREAM]: "hook_output_stream",
		[ClineSay.CONDITIONAL_RULES_APPLIED]: "conditional_rules_applied",
		[ClineSay.SUBAGENT_STATUS]: "subagent",
		[ClineSay.USE_SUBAGENTS_SAY]: "use_subagents",
		[ClineSay.SUBAGENT_USAGE]: "subagent_usage",
		[ClineSay.ANTHROPIC_RATE_LIMIT]: "error",
		[ClineSay.PLAN_MODE_ACT]: "task",
		[ClineSay.SAY_ACCOUNT_LIMIT_REACHED]: "error",
		[ClineSay.SAY_AUTO_APPROVAL_LIMIT_REACHED]: "error",
		[ClineSay.CUSTOM_PROMPT_COMPLETION]: "completion_result",
		[ClineSay.PROMPT_COMPLETION]: "completion_result",
		[ClineSay.CHECKPOINT_REVERTED]: "info",
		[ClineSay.CREDIT_LIMIT_REACHED]: "error",
		[ClineSay.HOSTED_LOGIN]: "info",
		[ClineSay.MAX_CONTEXT_COMPOSITION_WINDOW]: "info",
		[ClineSay.TASK_CANCEL]: "task",
		[ClineSay.TASK_RESUME]: "task",
		[ClineSay.ZULONG_RULES_CREATED]: "info",
		[ClineSay.SAFE_MODE_LIMIT_REACHED]: "error",
		[ClineSay.PREEMPTIVE_RATE_LIMIT]: "error",
		[ClineSay.MCP_SERVER_REQUESTED]: "info",
		[ClineSay.PLAN_MODE_SWITCH]: "task",
		[ClineSay.ACT_MODE_SWITCH]: "task",
		[ClineSay.NEW_TASK_CREATED]: "info",
		[ClineSay.TASK_COMPLETION_RESULT]: "completion_result",
		[ClineSay.ZULONG_RULES_RECOGNITION]: "info",
	}

	return mapping[say]
}

/**
 * Convert application ZulongMessage to proto ClineMessage
 */
export function convertZulongMessageToProto(message: AppZulongMessage): ProtoClineMessage {
	const askEnum = message.ask ? convertZulongAskToProtoEnum(message.ask) : undefined
	const sayEnum = message.say ? convertZulongSayToProtoEnum(message.say) : undefined

	let finalAskEnum: ClineAsk = ClineAsk.FOLLOWUP
	let finalSayEnum: ClineSay = ClineSay.TEXT

	if (message.type === "ask") {
		finalAskEnum = askEnum ?? ClineAsk.FOLLOWUP
	} else if (message.type === "say") {
		finalSayEnum = sayEnum ?? ClineSay.TEXT
	}

	const protoMessage: ProtoClineMessage = {
		ts: message.ts,
		type: message.type === "ask" ? ClineMessageType.ASK : ClineMessageType.SAY,
		ask: finalAskEnum,
		say: finalSayEnum,
		text: message.text ?? "",
		reasoning: message.reasoning ?? "",
		images: message.images ?? [],
		files: message.files ?? [],
		partial: message.partial ?? false,
		lastCheckpointHash: message.lastCheckpointHash ?? "",
		isCheckpointCheckedOut: message.isCheckpointCheckedOut ?? false,
		isOperationOutsideWorkspace: message.isOperationOutsideWorkspace ?? false,
		conversationHistoryIndex: message.conversationHistoryIndex ?? 0,
		conversationHistoryDeletedRange: message.conversationHistoryDeletedRange
			? {
					startIndex: message.conversationHistoryDeletedRange[0],
					endIndex: message.conversationHistoryDeletedRange[1],
				}
			: undefined,
		sayTool: undefined,
		sayBrowserAction: undefined,
		browserActionResult: undefined,
		askUseMcpServer: undefined,
		planModeResponse: undefined,
		askQuestion: undefined,
		askNewTask: undefined,
		apiReqInfo: undefined,
		modelInfo: message.modelInfo ?? undefined,
	}

	return protoMessage
}

/**
 * Convert proto ClineMessage to application ZulongMessage
 */
export function convertProtoToZulongMessage(protoMessage: ProtoClineMessage): AppZulongMessage {
	const message: AppZulongMessage = {
		ts: protoMessage.ts,
		type: protoMessage.type === ClineMessageType.ASK ? "ask" : "say",
	}

	if (protoMessage.type === ClineMessageType.ASK) {
		const ask = convertProtoEnumToZulongAsk(protoMessage.ask)
		if (ask !== undefined) {
			message.ask = ask
		}
	}

	if (protoMessage.type === ClineMessageType.SAY) {
		const say = convertProtoEnumToZulongSay(protoMessage.say)
		if (say !== undefined) {
			message.say = say
		}
	}

	if (protoMessage.text !== "") {
		message.text = protoMessage.text
	}
	if (protoMessage.reasoning !== "") {
		message.reasoning = protoMessage.reasoning
	}
	if (protoMessage.images.length > 0) {
		message.images = protoMessage.images
	}
	if (protoMessage.files.length > 0) {
		message.files = protoMessage.files
	}
	if (protoMessage.partial) {
		message.partial = protoMessage.partial
	}
	if (protoMessage.lastCheckpointHash !== "") {
		message.lastCheckpointHash = protoMessage.lastCheckpointHash
	}
	if (protoMessage.isCheckpointCheckedOut) {
		message.isCheckpointCheckedOut = protoMessage.isCheckpointCheckedOut
	}
	if (protoMessage.isOperationOutsideWorkspace) {
		message.isOperationOutsideWorkspace = protoMessage.isOperationOutsideWorkspace
	}
	if (protoMessage.conversationHistoryIndex !== 0) {
		message.conversationHistoryIndex = protoMessage.conversationHistoryIndex
	}

	if (protoMessage.conversationHistoryDeletedRange) {
		message.conversationHistoryDeletedRange = [
			protoMessage.conversationHistoryDeletedRange.startIndex,
			protoMessage.conversationHistoryDeletedRange.endIndex,
		]
	}

	return message
}
