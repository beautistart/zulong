/**
 * Zulong WebSocket Transport Layer
 *
 * Manages a persistent WebSocket connection to the Zulong IDE Server (Python backend).
 * Handles message framing, reconnection, and event dispatching.
 *
 * Protocol:
 *   Plugin → Backend: task:start / task:resume / tool:result / task:cancel
 *   Backend → Plugin: tool:request / text:stream / reasoning / task:complete / task:error / task:progress / task:ack
 * Legacy IDE message names are still accepted for backward compatibility.
 */

import { EventEmitter } from "events"
import { Logger } from "@/shared/services/Logger"
import type { InteractionPayload } from "@/shared/ExtensionMessage"
import type { ApprovalMode } from "@/shared/ApprovalWhitelist"

export const DEFAULT_ZULONG_IDE_WS_URL = "ws://127.0.0.1:8090/ide"

export function normalizeZulongWebSocketUrl(serverUrl?: string): string {
	const raw = (serverUrl || DEFAULT_ZULONG_IDE_WS_URL).trim() || DEFAULT_ZULONG_IDE_WS_URL
	const wsUrl = raw.replace(/^http/i, "ws")
	const parsed = new URL(wsUrl.replace(/^ws/i, "http"))
	if (parsed.pathname === "/" || parsed.pathname === "") {
		parsed.pathname = "/ide"
	}
	return parsed.toString().replace(/^http/i, "ws")
}

// ── Message types ────────────────────────────────────

export interface ZulongMessage {
	msg_id: string
	type: string
	session_id: string
	ts: number
	payload: Record<string, any>
}

export interface ZulongToolRequest {
	tool_calls: Array<{
		id: string
		function: { name: string; arguments: string }
	}>
	call_ids: string[]
	tool_names: string[]
}

const LEGACY_TO_UNIFIED: Record<string, string> = {
	session_start: "task:start",
	session_resume: "task:resume",
	user_cancel: "task:cancel",
	tool_result: "tool:result",
	session_ack: "task:ack",
	task_complete: "task:complete",
	task_error: "task:error",
	task_progress: "task:progress",
	status_update: "task:progress",
	display_text: "text:stream",
	display_reasoning: "reasoning",
	tool_request: "tool:request",
	audio_start: "audio:start",
	audio_chunk: "audio:chunk",
	audio_end: "audio:end",
	audio_transcript: "audio:transcript",
	audio_start_ack: "audio:transcript",
	tool_prediction: "tool:prediction",
	task_plan: "task:plan",
	task_summary: "task:summary",
	approval_required: "approval:required",
	approval_result: "approval:result",
	attention_update: "attention:update",
	graph_memory_diff: "graph:memory:diff",
	interaction_event: "interaction:event",
}

const INTERACTION_EVENT_TYPES = new Set([
	"tool_prediction",
	"task_plan",
	"task_summary",
	"approval_required",
	"approval_result",
	"attention_update",
	"graph_memory_diff",
	"interaction_event",
])

const UNIFIED_TO_LEGACY: Record<string, string> = Object.fromEntries(
	Object.entries(LEGACY_TO_UNIFIED).map(([legacy, unified]) => [unified, legacy]),
)

// ── Transport class ──────────────────────────────────

export class ZulongWebSocket extends EventEmitter {
	private ws: WebSocket | null = null
	private serverUrl: string
	private sessionId: string = ""
	private reconnectAttempts = 0
	private readonly maxReconnectDelay = 30_000
	private reconnectDelay = 1000
	private reconnectTimer: NodeJS.Timeout | null = null
	private connectingPromise: Promise<string> | null = null
	private disposed = false
	private pendingMessages: Array<Record<string, any>> = []

	// Heartbeat mechanism
	private heartbeatInterval: NodeJS.Timeout | null = null
	private pongTimeout: NodeJS.Timeout | null = null
	private missedPongs = 0
	private readonly HEARTBEAT_INTERVAL = 30000 // 30 seconds
	private readonly PONG_TIMEOUT = 5000 // 5 seconds
	private readonly MAX_MISSED_PONGS = 3

	constructor(serverUrl: string) {
		super()
		this.serverUrl = normalizeZulongWebSocketUrl(serverUrl)
	}

	get isConnected(): boolean {
		return this.ws !== null && this.ws.readyState === WebSocket.OPEN
	}

	get currentSessionId(): string {
		return this.sessionId
	}

	/**
	 * Connect to the Zulong IDE Server.
	 * Resolves when session_ack is received; rejects on failure.
	 */
	async connect(): Promise<string> {
		if (this.disposed) {
			throw new Error("Transport has been disposed")
		}
		if (this.isConnected && this.sessionId) {
			return this.sessionId
		}
		if (this.connectingPromise) {
			return this.connectingPromise
		}

		Logger.info(`[ZulongWS] Connecting to ${this.serverUrl}`)

		this.connectingPromise = new Promise<string>((resolve, reject) => {
			let settled = false
			const settleResolve = (sessionId: string) => {
				if (settled) {
					return
				}
				settled = true
				this.connectingPromise = null
				resolve(sessionId)
			}
			const settleReject = (error: Error) => {
				if (settled) {
					return
				}
				settled = true
				this.connectingPromise = null
				reject(error)
			}

			try {
				this.ws = new WebSocket(this.serverUrl)
			} catch (err) {
				Logger.error(`[ZulongWS] Failed to create WebSocket: ${err}`)
				const error = new Error(`Failed to create WebSocket: ${err}`)
				settleReject(error)
				this.scheduleReconnect("create_failed")
				return
			}

			const timeout = setTimeout(() => {
				Logger.error("[ZulongWS] Connection timeout (10s)")
				settleReject(new Error("WebSocket connection timeout (10s)"))
				this.scheduleReconnect("timeout")
				this.ws?.close()
			}, 10_000)

			this.ws.onopen = () => {
				Logger.info("[ZulongWS] WebSocket connected")
				this.missedPongs = 0
				this.startHeartbeat()
				// 统一协议握手: 声明客户端类型
				this.sendRaw({
					msg_id: this.generateMsgId(),
					type: "handshake",
					session_id: "",
					ts: Date.now() / 1000,
					payload: { client_type: "ide_plugin", api_version: "2.0" },
				})
				this.emit("connected")
			}

			this.ws.onmessage = (event: MessageEvent) => {
				try {
					const msg: ZulongMessage = JSON.parse(
						typeof event.data === "string" ? event.data : event.data.toString(),
					)
					const normalizedMsg = this.normalizeIncoming(msg)
					Logger.info(`[ZulongWS] \u2190 RECV ${msg.type} normalized=${normalizedMsg.type} session=${msg.session_id?.slice(0, 12)} msg_id=${msg.msg_id}`)
					if (normalizedMsg.type === "tool_request") {
						const p = normalizedMsg.payload as ZulongToolRequest
						Logger.info(`[ZulongWS] \u2190 tool_request: tools=[${p.tool_names?.join(", ")}], call_ids=[${p.call_ids?.join(", ")}]`)
					}
					this.handleMessage(normalizedMsg)

					// Resolve on session_ack
					if (normalizedMsg.type === "session_ack") {
						clearTimeout(timeout)
						this.sessionId = normalizedMsg.payload?.session_id || normalizedMsg.session_id
						Logger.info(`[ZulongWS] Session established: ${this.sessionId?.slice(0, 12)}`)
						if (normalizedMsg.payload?.context_window_size && typeof normalizedMsg.payload.context_window_size === "number") {
							this.emit("model_info", {
								contextWindow: normalizedMsg.payload.context_window_size,
							})
							Logger.info(`[ZulongWS] Backend context_window_size: ${normalizedMsg.payload.context_window_size}`)
						}
						// Flush pending messages
						for (const pending of this.pendingMessages) {
							this.sendRaw(pending)
						}
						this.pendingMessages = []
						this.reconnectAttempts = 0
						settleResolve(this.sessionId)
					}
				} catch (e) {
					Logger.error(`[ZulongWS] Message parse error: ${e}`)
					this.emit("error", new Error(`Message parse error: ${e}`))
				}
			}

			this.ws.onerror = (event: Event) => {
				clearTimeout(timeout)
				const errMsg = `WebSocket error connecting to ${this.serverUrl}`
				Logger.error(`[ZulongWS] ${errMsg}`)
				this.emit("error", new Error(errMsg))
				settleReject(new Error(errMsg))
				this.scheduleReconnect("error")
			}

			this.ws.onclose = (event: CloseEvent) => {
				clearTimeout(timeout)
				this.stopHeartbeat()
				Logger.warn(`[ZulongWS] WebSocket closed: code=${event.code} reason=${event.reason}`)
				this.emit("disconnected", event.code, event.reason)
				settleReject(new Error(`WebSocket closed: code=${event.code} reason=${event.reason || ""}`))
				if (!this.disposed) {
					this.scheduleReconnect("closed")
				}
			}
		})
		return this.connectingPromise
	}

	/**
	 * Send session_start to begin a new task.
	 */
	sendSessionStart(task: string, cwd: string, zulongSystemPrompt?: string, projectId?: string, approvalMode?: ApprovalMode): void {
		const payload: Record<string, string> = {
			task,
			cwd,
			ide_system_prompt: zulongSystemPrompt || "",
		}
		if (projectId) {
			payload.project_id = projectId
		}
		if (approvalMode) {
			payload.approval_mode = approvalMode
		}
		this.send("task:start", payload)
	}

	/**
	 * Send session_resume to continue a previous task.
	 */
	sendSessionResume(task: string, cwd: string, zulongSystemPrompt?: string, graphId?: string): void {
		const payload: Record<string, string> = {
			task,
			cwd,
			ide_system_prompt: zulongSystemPrompt || "",
		}
		if (graphId) {
			payload.graph_id = graphId
		}
		this.send("task:resume", payload)
	}

	/**
	 * Send tool execution result back to the backend.
	 */
	sendToolResult(callId: string, toolName: string, result: string, isError: boolean = false): void {
		this.send("tool:result", {
			call_id: callId,
			tool_name: toolName,
			result,
			is_error: isError,
		})
	}

	/**
	 * Send user cancel signal.
	 */
	sendCancel(): void {
		this.send("task:cancel", {})
	}

	sendIdeContext(payload: Record<string, any>): void {
		this.send("ide:context", payload)
	}

	sendIdeFileChanged(payload: Record<string, any>): void {
		this.send("ide:file_changed", payload)
	}

	sendIdeTerminalStatus(payload: Record<string, any>): void {
		this.send("ide:terminal_status", payload)
	}

	sendIdeApprovalStatus(payload: Record<string, any>): void {
		this.send("ide:approval_status", payload)
	}

	sendIdeApprovalResult(payload: Record<string, any>): void {
		this.send("ide:approval_result", payload)
	}

	sendIdeDiffStatus(payload: Record<string, any>): void {
		this.send("ide:diff_status", payload)
	}

	sendIdeCheckpointStatus(payload: Record<string, any>): void {
		this.send("ide:checkpoint_status", payload)
	}

	/**
	 * Send audio data for real-time transcription.
	 * @param audioBase64 Base64 encoded audio data
	 * @param format Audio format (webm, mp4, wav)
	 */
	sendAudioChunk(audioBase64: string, format: string = "webm"): void {
		this.send("audio:chunk", {
			audio: audioBase64,
			format,
			sample_rate: 16000,
		})
	}

	/**
	 * Signal audio stream start.
	 */
	sendAudioStart(): void {
		this.send("audio:start", {})
	}

	/**
	 * Signal audio stream end.
	 */
	sendAudioEnd(): void {
		this.send("audio:end", {})
	}

	/**
	 * Disconnect and clean up.
	 */
	dispose(): void {
		Logger.info("[ZulongWS] Transport disposed")
		this.disposed = true
		this.pendingMessages = []
		this.stopHeartbeat()
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer)
			this.reconnectTimer = null
		}
		this.connectingPromise = null
		if (this.ws) {
			this.ws.onclose = null
			this.ws.onerror = null
			this.ws.onmessage = null
			this.ws.close()
			this.ws = null
		}
		this.removeAllListeners()
	}

	// ── Internal ─────────────────────────────────────

	private send(type: string, payload: Record<string, any>): void {
		const msg = {
			msg_id: this.generateMsgId(),
			type,
			session_id: this.sessionId,
			ts: Date.now() / 1000,
			payload,
		}
		Logger.info(`[ZulongWS] \u2192 SEND ${type} session=${this.sessionId?.slice(0, 12)}`)
		if (type === "tool:result") {
			Logger.info(`[ZulongWS] \u2192 tool_result: call_id=${payload.call_id}, tool=${payload.tool_name}, is_error=${payload.is_error}`)
		}
		if (this.isConnected) {
			this.sendRaw(msg)
		} else {
			Logger.warn(`[ZulongWS] Not connected, queuing message: ${type}`)
			this.pendingMessages.push(msg)
		}
	}

	private sendRaw(msg: Record<string, any>): void {
		try {
			this.ws?.send(JSON.stringify(msg))
		} catch (e) {
			Logger.error(`[ZulongWS] Send failed: ${e}`)
			this.emit("error", new Error(`Send failed: ${e}`))
		}
	}

	private handleMessage(msg: ZulongMessage): void {
		// Handle pong first (heartbeat)
		if (msg.type === "pong") {
			this.handlePong()
			return
		}

		// Emit typed events that ZulongHandler listens to
		switch (msg.type) {
			case "handshake_ack":
				// 握手确认: 记录服务器版本信息
				Logger.info(`[ZulongWS] handshake_ack: server_version=${msg.payload?.server_version}`)
				break
			case "tool_request":
				this.emit("tool_request", msg.payload as ZulongToolRequest)
				break
			case "display_text":
				// 🔥 修复：传递完整payload，包含task_result和task_status
				Logger.info(`[ZulongWS] display_text payload: ${JSON.stringify(msg.payload).substring(0, 200)}`)
				this.emit("display_text", msg.payload.text || "", msg.payload.turn, msg.payload)
				break
			case "display_reasoning":
				this.emit("display_reasoning", msg.payload.reasoning || "")
				break
			case "task_progress":
				// 🎯 P3改进：任务进度汇报
				Logger.info(`[ZulongWS] task_progress: phase=${msg.payload.phase}, message=${msg.payload.message}`)
				this.emit("task_progress", msg.payload)
				break
			case "task_complete":
				this.emit("task_complete", msg.payload.result || "", msg.payload)
				break
			case "task_error":
				this.emit("task_error", msg.payload.error || "Unknown error")
				break
			case "status_update":
				this.emit("status_update", msg.payload)
				break
			case "session_ack":
				// Handled in connect()
				break
			case "audio_transcript":
				this.emit("audio_transcript", msg.payload.text || "", msg.payload.is_final || false)
				break
			case "ide_open_workspace":
				this.emit("ide_open_workspace", msg.payload)
				break
			case "ide_open_file":
				this.emit("ide_open_file", msg.payload)
				break
			case "ide_open_terminal":
				this.emit("ide_open_terminal", msg.payload)
				break
			case "ide_show_diff":
				this.emit("ide_show_diff", msg.payload)
				break
			case "ide_approval_result":
				this.emit("ide_approval_result", msg.payload)
				break
			case "ide_runtime_settings":
				this.emit("ide_runtime_settings", msg.payload)
				break
			case "ide_get_context":
				this.emit("ide_get_context", msg.payload)
				break
			case "system_ready":
				this.emit("system_ready", msg.payload)
				break
			default:
				if (INTERACTION_EVENT_TYPES.has(msg.type)) {
					this.emit("interaction", this.toInteractionPayload(msg), msg)
					break
				}
				this.emit("unknown_message", msg)
		}
	}

	private toInteractionPayload(msg: ZulongMessage): InteractionPayload {
		const payload = msg.payload || {}
		const existing = payload.interaction
		const base =
			existing && typeof existing === "object"
				? { ...(existing as Partial<InteractionPayload>) }
				: this.buildFallbackInteraction(msg)
		const approvalId = (base as any).approval_id || payload.approval_id || payload.approvalId
		const interactionId =
			base.interaction_id ||
			approvalId ||
			payload.interaction_id ||
			payload.call_id ||
			payload.msg_id ||
			msg.msg_id ||
			`${msg.type}:${Date.now()}`
		const pairId = base.pair_id || payload.pair_id || approvalId || payload.call_id || interactionId
		const riskLevel = this.normalizeRiskLevel((base as any).risk_level || payload.risk_level || payload.risk)
		const approvalMode = this.normalizeApprovalMode((base as any).approval_mode || payload.approval_mode)

		const interaction = {
			...base,
			interaction_id: String(interactionId),
			pair_id: String(pairId),
			approval_id: approvalId ? String(approvalId) : (base as any).approval_id,
			kind: this.normalizeInteractionKind(base.kind),
			status: this.normalizeInteractionStatus(base.status),
			title: base.title || this.defaultInteractionTitle(msg),
			detail: base.detail || payload.reason || payload.message || "",
			timestamp: base.timestamp ?? payload.timestamp ?? msg.ts,
			turn: base.turn ?? payload.turn,
		} as InteractionPayload
		if (riskLevel) {
			interaction.risk_level = riskLevel
		}
		if (approvalMode) {
			interaction.approval_mode = approvalMode
		}
		return interaction
	}

	private buildFallbackInteraction(msg: ZulongMessage): Partial<InteractionPayload> {
		const payload = msg.payload || {}
		switch (msg.type) {
			case "tool_prediction":
				return {
					kind: "plan",
					status: "running",
					title: "已预判可用工具",
					detail: payload.reason || "已根据当前任务准备工具包。",
					tool_args: {
						suggested_tools: payload.suggested_tools || payload.prediction?.suggested_tools || [],
						confidence: payload.confidence ?? payload.prediction?.confidence,
					},
				}
			case "task_plan":
				return {
					kind: "plan",
					status: "running",
					title: payload.task ? `任务启动: ${String(payload.task).slice(0, 80)}` : "任务已启动",
					detail: payload.intent ? `意图: ${payload.intent}` : "正在准备上下文和工具。",
					tool_args: {
						suggested_tools: payload.tool_prediction?.suggested_tools || [],
					},
				}
			case "approval_required":
				return {
					kind: "approval",
					status: "awaiting_approval",
					title: payload.tool_name ? `审批请求: ${payload.tool_name}` : "需要审批",
					detail: payload.reason || "该操作需要用户确认后继续。",
					tool_name: payload.tool_name,
					tool_args: this.parseToolArgs(payload.tool_args),
					risk_level: payload.risk_level,
					risk_reason: payload.reason,
					approval_mode: payload.approval_mode,
				}
			case "task_summary":
				return {
					kind: "summary",
					status: "succeeded",
					title: "任务完成",
					detail: payload.detail || "",
					completed_items: payload.completed_items,
					verified_items: payload.verified_items,
					pending_items: payload.pending_items,
					risks_summary: payload.risks_summary,
					next_step: payload.next_step,
					memory_changes: payload.memory_changes,
				}
			case "graph_memory_diff":
				return {
					kind: "observation",
					status: "succeeded",
					title: "图记忆已更新",
					detail: "本轮任务产生了记忆图谱变化。",
					memory_changes: payload.memory_changes,
				}
			case "attention_update":
				return {
					kind: "progress",
					status: "running",
					title: "注意力状态更新",
					detail: payload.mode || payload.state || "",
				}
			case "approval_result":
				return {
					kind: "approval",
					status: payload.action === "reject" ? "rejected" : "approved",
					title: "审批结果已记录",
					detail: payload.action || "",
				}
			default:
				return {
					kind: "observation",
					status: "running",
					title: this.defaultInteractionTitle(msg),
					detail: payload.message || "",
				}
		}
	}

	private defaultInteractionTitle(msg: ZulongMessage): string {
		switch (msg.type) {
			case "tool_prediction":
				return "工具预判"
			case "task_plan":
				return "任务启动"
			case "task_summary":
				return "任务总结"
			case "approval_required":
				return "审批请求"
			case "graph_memory_diff":
				return "图记忆变化"
			case "attention_update":
				return "注意力更新"
			case "approval_result":
				return "审批结果"
			case "interaction_event":
				return "任务事件"
			default:
				return "任务事件"
		}
	}

	private normalizeInteractionKind(kind: unknown): InteractionPayload["kind"] {
		switch (kind) {
			case "plan":
			case "action":
			case "observation":
			case "progress":
			case "approval":
			case "summary":
			case "user_interject":
				return kind
			case "state":
				return "progress"
			case "user_adjustment":
				return "user_interject"
			default:
				return "observation"
		}
	}

	private normalizeInteractionStatus(status: unknown): InteractionPayload["status"] {
		switch (status) {
			case "pending":
			case "running":
			case "awaiting_approval":
			case "approved":
			case "rejected":
			case "succeeded":
			case "failed":
			case "blocked":
			case "cancelled":
				return status
			case "completed":
			case "complete":
				return "succeeded"
			default:
				return "running"
		}
	}

	private normalizeRiskLevel(risk: unknown): InteractionPayload["risk_level"] | undefined {
		switch (String(risk || "").trim().toUpperCase()) {
			case "LOW":
				return "LOW"
			case "MEDIUM":
				return "MEDIUM"
			case "HIGH":
				return "HIGH"
			case "CRITICAL":
			case "DANGER":
			case "POPUP":
				return "CRITICAL"
			default:
				return undefined
		}
	}

	private normalizeApprovalMode(mode: unknown): InteractionPayload["approval_mode"] | undefined {
		switch (String(mode || "").trim().toLowerCase()) {
			case "full":
			case "full_auto":
				return "full_auto"
			case "read_only":
			case "whitelist":
				return "whitelist"
			case "manual":
			case "off":
				return "manual"
			case "popup":
				return "popup"
			default:
				return undefined
		}
	}

	private parseToolArgs(raw: unknown): Record<string, any> | undefined {
		if (!raw) {
			return undefined
		}
		if (typeof raw === "object") {
			return raw as Record<string, any>
		}
		if (typeof raw !== "string") {
			return { value: raw }
		}
		try {
			const parsed = JSON.parse(raw)
			return parsed && typeof parsed === "object" ? parsed : { value: raw }
		} catch {
			return { value: raw }
		}
	}

	private normalizeIncoming(msg: ZulongMessage): ZulongMessage {
		const legacyType = UNIFIED_TO_LEGACY[msg.type] || msg.type
		const payload = { ...(msg.payload || {}) }
		if (msg.type === "text:final" && !payload.result && payload.text) {
			payload.result = payload.text
		}
		if (msg.type.startsWith("ide:")) {
			return {
				...msg,
				type: msg.type.replace("ide:", "ide_"),
				payload,
			}
		}
		return {
			...msg,
			type: legacyType,
			payload,
		}
	}

	private scheduleReconnect(reason: string): void {
		if (this.disposed || this.reconnectTimer || this.isConnected) {
			return
		}
		this.reconnectAttempts++
		const delay = Math.min(
			this.maxReconnectDelay,
			this.reconnectDelay * Math.pow(2, Math.min(this.reconnectAttempts - 1, 5)),
		)
		Logger.warn(`[ZulongWS] Reconnecting after ${reason}... attempt ${this.reconnectAttempts} (delay ${delay}ms)`)
		this.reconnectTimer = setTimeout(() => {
			this.reconnectTimer = null
			if (!this.disposed) {
				this.emit("reconnecting", this.reconnectAttempts)
				this.connect().catch(() => {
					// Reconnect failure is handled by scheduleReconnect from error/close/timeout.
				})
			}
		}, delay)
	}

	// ── Heartbeat ─────────────────────────────────────

	private startHeartbeat(): void {
		this.stopHeartbeat()
		this.heartbeatInterval = setInterval(() => {
			if (this.isConnected) {
				this.sendRaw({
					msg_id: this.generateMsgId(),
					type: "ping",
					session_id: this.sessionId,
					ts: Date.now() / 1000,
					payload: {},
				})
				this.startPongTimeout()
			}
		}, this.HEARTBEAT_INTERVAL)
	}

	private stopHeartbeat(): void {
		if (this.heartbeatInterval) {
			clearInterval(this.heartbeatInterval)
			this.heartbeatInterval = null
		}
		if (this.pongTimeout) {
			clearTimeout(this.pongTimeout)
			this.pongTimeout = null
		}
	}

	private startPongTimeout(): void {
		if (this.pongTimeout) {
			clearTimeout(this.pongTimeout)
		}
		this.pongTimeout = setTimeout(() => {
			this.missedPongs++
			Logger.warn(`[ZulongWS] Pong timeout, missed ${this.missedPongs}/${this.MAX_MISSED_PONGS}`)
			if (this.missedPongs >= this.MAX_MISSED_PONGS) {
				Logger.error("[ZulongWS] Too many missed pongs, closing connection")
				this.ws?.close()
			}
		}, this.PONG_TIMEOUT)
	}

	private handlePong(): void {
		if (this.pongTimeout) {
			clearTimeout(this.pongTimeout)
			this.pongTimeout = null
		}
		this.missedPongs = 0
		Logger.debug("[ZulongWS] Received pong")
	}

	private generateMsgId(): string {
		return Math.random().toString(36).substring(2, 14)
	}
}
