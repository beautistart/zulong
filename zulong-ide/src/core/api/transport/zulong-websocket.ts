/**
 * Zulong WebSocket Transport Layer
 *
 * Manages a persistent WebSocket connection to the Zulong IDE Server (Python backend).
 * Handles message framing, reconnection, and event dispatching.
 *
 * Protocol (matching zulong_ide_server.py):
 *   Plugin → Backend: session_start / session_resume / tool_result / user_cancel
 *   Backend → Plugin: tool_request / display_text / display_reasoning / task_complete / task_error / status_update / session_ack
 */

import { EventEmitter } from "events"
import { Logger } from "@/shared/services/Logger"

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

// ── Transport class ──────────────────────────────────

export class ZulongWebSocket extends EventEmitter {
	private ws: WebSocket | null = null
	private serverUrl: string
	private sessionId: string = ""
	private reconnectAttempts = 0
	private maxReconnectAttempts = 5
	private reconnectDelay = 1000
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
		// Normalize URL: ensure ws:// or wss://
		this.serverUrl = serverUrl.replace(/^http/, "ws")
		if (!this.serverUrl.endsWith("/ide")) {
			this.serverUrl = this.serverUrl.replace(/\/$/, "") + "/ide"
		}
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

		Logger.info(`[ZulongWS] Connecting to ${this.serverUrl}`)

		return new Promise<string>((resolve, reject) => {
			try {
				this.ws = new WebSocket(this.serverUrl)
			} catch (err) {
				Logger.error(`[ZulongWS] Failed to create WebSocket: ${err}`)
				reject(new Error(`Failed to create WebSocket: ${err}`))
				return
			}

			const timeout = setTimeout(() => {
				Logger.error("[ZulongWS] Connection timeout (10s)")
				reject(new Error("WebSocket connection timeout (10s)"))
				this.ws?.close()
			}, 10_000)

			this.ws.onopen = () => {
				Logger.info("[ZulongWS] WebSocket connected")
				this.reconnectAttempts = 0
				this.missedPongs = 0
				this.startHeartbeat()
				this.emit("connected")
			}

			this.ws.onmessage = (event: MessageEvent) => {
				try {
					const msg: ZulongMessage = JSON.parse(
						typeof event.data === "string" ? event.data : event.data.toString(),
					)
					Logger.info(`[ZulongWS] \u2190 RECV ${msg.type} session=${msg.session_id?.slice(0, 12)} msg_id=${msg.msg_id}`)
					if (msg.type === "tool_request") {
						const p = msg.payload as ZulongToolRequest
						Logger.info(`[ZulongWS] \u2190 tool_request: tools=[${p.tool_names?.join(", ")}], call_ids=[${p.call_ids?.join(", ")}]`)
					}
					this.handleMessage(msg)

					// Resolve on session_ack
					if (msg.type === "session_ack") {
						clearTimeout(timeout)
						this.sessionId = msg.payload?.session_id || msg.session_id
						Logger.info(`[ZulongWS] Session established: ${this.sessionId?.slice(0, 12)}`)
						if (msg.payload?.context_window_size && typeof msg.payload.context_window_size === "number") {
							this.emit("model_info", {
								contextWindow: msg.payload.context_window_size,
							})
							Logger.info(`[ZulongWS] Backend context_window_size: ${msg.payload.context_window_size}`)
						}
						// Flush pending messages
						for (const pending of this.pendingMessages) {
							this.sendRaw(pending)
						}
						this.pendingMessages = []
						resolve(this.sessionId)
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
				reject(new Error(errMsg))
			}

			this.ws.onclose = (event: CloseEvent) => {
				clearTimeout(timeout)
				this.stopHeartbeat()
				Logger.warn(`[ZulongWS] WebSocket closed: code=${event.code} reason=${event.reason}`)
				this.emit("disconnected", event.code, event.reason)
				if (!this.disposed) {
					this.attemptReconnect()
				}
			}
		})
	}

	/**
	 * Send session_start to begin a new task.
	 */
	sendSessionStart(task: string, cwd: string, zulongSystemPrompt?: string, projectId?: string): void {
		const payload: Record<string, string> = {
			task,
			cwd,
			ide_system_prompt: zulongSystemPrompt || "",
		}
		if (projectId) {
			payload.project_id = projectId
		}
		this.send("session_start", payload)
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
		this.send("session_resume", payload)
	}

	/**
	 * Send tool execution result back to the backend.
	 */
	sendToolResult(callId: string, toolName: string, result: string, isError: boolean = false): void {
		this.send("tool_result", {
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
		this.send("user_cancel", {})
	}

	/**
	 * Send audio data for real-time transcription.
	 * @param audioBase64 Base64 encoded audio data
	 * @param format Audio format (webm, mp4, wav)
	 */
	sendAudioChunk(audioBase64: string, format: string = "webm"): void {
		this.send("audio_chunk", {
			audio: audioBase64,
			format,
			sample_rate: 16000,
		})
	}

	/**
	 * Signal audio stream start.
	 */
	sendAudioStart(): void {
		this.send("audio_start", {})
	}

	/**
	 * Signal audio stream end.
	 */
	sendAudioEnd(): void {
		this.send("audio_end", {})
	}

	/**
	 * Disconnect and clean up.
	 */
	dispose(): void {
		Logger.info("[ZulongWS] Transport disposed")
		this.disposed = true
		this.pendingMessages = []
		this.stopHeartbeat()
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
		if (type === "tool_result") {
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
			case "tool_request":
				this.emit("tool_request", msg.payload as ZulongToolRequest)
				break
			case "display_text":
				this.emit("display_text", msg.payload.text || "", msg.payload.turn)
				break
			case "display_reasoning":
				this.emit("display_reasoning", msg.payload.reasoning || "")
				break
			case "task_complete":
				this.emit("task_complete", msg.payload.result || "")
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
			default:
				this.emit("unknown_message", msg)
		}
	}

	private attemptReconnect(): void {
		if (this.reconnectAttempts >= this.maxReconnectAttempts) {
			Logger.error(`[ZulongWS] Max reconnection attempts (${this.maxReconnectAttempts}) reached`)
			this.emit("error", new Error("Max reconnection attempts reached"))
			return
		}
		this.reconnectAttempts++
		const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
		Logger.warn(`[ZulongWS] Reconnecting... attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} (delay ${delay}ms)`)
		setTimeout(() => {
			if (!this.disposed) {
				this.emit("reconnecting", this.reconnectAttempts)
				this.connect().catch(() => {
					// Reconnect failure handled by onclose
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
