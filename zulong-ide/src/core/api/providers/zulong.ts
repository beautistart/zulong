/**
 * Zulong Provider Handler
 *
 * Implements ApiHandler by delegating to the Zulong Python backend via WebSocket.
 * The backend runs the full FC loop (model calls + internal tools); this handler
 * only executes remote (Zulong-side) tools and streams results back to Task.
 *
 * Data flow:
 *   createMessage() → WS session_start → backend FC loop
 *   ← tool_request  → yield ApiStreamToolCallsChunk (Task executes tool)
 *   → tool_result    → backend continues FC loop
 *   ← display_text   → yield ApiStreamTextChunk
 *   ← task_complete   → generator returns
 */

import { ModelInfo } from "@shared/api"
import type { ApprovalMode } from "@shared/ApprovalWhitelist"
import { ZulongStorageMessage } from "@/shared/messages/content"
import { ZulongTool } from "@/shared/tools"
import { ApiHandler, ApiHandlerModel, CommonApiHandlerOptions } from "../index"
import { ApiStream, ApiStreamChunk } from "../transform/stream"
import { DEFAULT_ZULONG_IDE_WS_URL, ZulongWebSocket, ZulongToolRequest } from "../transport/zulong-websocket"
import { Logger } from "@/shared/services/Logger"
import fs from "fs"
import path from "path"

export interface ZulongHandlerOptions extends CommonApiHandlerOptions {
	zulongServerUrl?: string
	zulongApprovalMode?: ApprovalMode
}

const ZULONG_MODEL_INFO: ModelInfo = {
	name: "zulong-agent",
	maxTokens: 16384,
	contextWindow: 131072,
	supportsImages: false,
	supportsPromptCache: false,
	supportsReasoning: false,
}

export class ZulongHandler implements ApiHandler {
	private options: ZulongHandlerOptions
	private transport: ZulongWebSocket | null = null
	private abortController: AbortController | null = null
	private dynamicModelInfo: ModelInfo = { ...ZULONG_MODEL_INFO }
	private connectionPromise: Promise<string> | null = null

	constructor(options: ZulongHandlerOptions) {
		this.options = options
		// 🔥 长连接模式：在构造函数中初始化WebSocket
		const serverUrl = options.zulongServerUrl || DEFAULT_ZULONG_IDE_WS_URL
		this.transport = new ZulongWebSocket(serverUrl)
		Logger.info(`[ZulongHandler] WebSocket initialized (long connection mode), serverUrl=${serverUrl}`)
	}

	/**
	 * 确保WebSocket已连接，支持自动重连
	 */
	private async ensureConnected(): Promise<string> {
		if (!this.transport) {
			const serverUrl = this.options.zulongServerUrl || DEFAULT_ZULONG_IDE_WS_URL
			this.transport = new ZulongWebSocket(serverUrl)
		}

		if (this.transport.isConnected) {
			return this.transport.currentSessionId
		}

		// 复用正在进行的连接尝试
		if (this.connectionPromise) {
			return this.connectionPromise
		}

		Logger.info(`[ZulongHandler] Connecting to backend...`)
		this.connectionPromise = this.transport.connect()
		
		try {
			const sessionId = await this.connectionPromise
			Logger.info(`[ZulongHandler] Connected to backend, sessionId=${sessionId?.slice(0, 12)}`)
			return sessionId
		} catch (err) {
			Logger.error(`[ZulongHandler] Connection failed: ${err}`)
			throw err
		} finally {
			this.connectionPromise = null
		}
	}

	getModel(): ApiHandlerModel {
		return {
			id: "zulong-agent",
			info: this.dynamicModelInfo,
		}
	}

	updateModelInfo(updates: Partial<ModelInfo>): void {
		this.dynamicModelInfo = { ...this.dynamicModelInfo, ...updates }
		Logger.info(`[ZulongHandler] Model info updated: contextWindow=${this.dynamicModelInfo.contextWindow}, maxTokens=${this.dynamicModelInfo.maxTokens}`)
	}

	sendIdeApprovalResult(payload: Record<string, any>): void {
		void (async () => {
			await this.ensureConnected()
			if (!this.transport) {
				throw new Error("Transport not initialized")
			}
			this.transport.sendIdeApprovalResult(payload)
		})().catch((error) => {
			Logger.error(`[ZulongHandler] Failed to send approval result: ${error}`)
		})
	}

	async *createMessage(
		systemPrompt: string,
		messages: ZulongStorageMessage[],
		tools?: ZulongTool[],
		_useResponseApi?: boolean,
	): ApiStream {
		this.abortController = new AbortController()

		Logger.info(`[ZulongHandler] createMessage() starting (reusing connection)`)

		// Extract task text from the last user message
		let taskText = ""
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i]
			if (msg.role === "user") {
				if (typeof msg.content === "string") {
					taskText = msg.content
				} else if (Array.isArray(msg.content)) {
					// Prefer the block containing <task> tags (pure user input)
					const textBlocks = msg.content.filter((b: any) => b.type === "text") as Array<{ type: "text"; text: string }>
					const taskBlock = textBlocks.find((b) => /<task>/.test(b.text))
					if (taskBlock) {
						taskText = taskBlock.text
					} else {
						taskText = textBlocks.map((b) => b.text).join("\n")
					}
				}
				break
			}
		}

		if (!taskText) {
			Logger.warn("[ZulongHandler] No user message found in messages array")
			yield { type: "text" as const, text: "[Zulong] No user message found" }
			return
		}

		// Strip <task> wrapper and focus chain noise before sending to backend
		const taskTagMatch = taskText.match(/<task>\s*([\s\S]*?)\s*<\/task>/)
		if (taskTagMatch) {
			taskText = taskTagMatch[1].trim()
		}

		Logger.info(`[ZulongHandler] Task text extracted (${taskText.length} chars)`)

		// Get working directory from environment info in system prompt
		const cwdMatch = systemPrompt.match(/Current Working Directory[:\s]+([^\n]+)/i)
		const cwd = cwdMatch?.[1]?.trim() || "."

		// 🔥 长连接模式：复用现有连接，不创建新的WebSocket
		try {
			await this.ensureConnected()
			Logger.info("[ZulongHandler] Connection ready for task")

			if (!this.transport) {
				throw new Error("Transport not initialized")
			}

			this.transport.on("audio_transcript", (text: string, isFinal: boolean) => {
				Logger.info(`[ZulongHandler] ← audio_transcript: "${text}" (is_final=${isFinal})`)
			})
		} catch (err) {
			Logger.error(`[ZulongHandler] Connection failed: ${err}`)
			const serverUrl = this.options.zulongServerUrl || DEFAULT_ZULONG_IDE_WS_URL
			yield {
				type: "text" as const,
				text: `[Zulong] WebSocket connection failed: ${err}\nPlease ensure the Zulong IDE Server is running at ${serverUrl}`,
			}
			return
		}

		// Use async generator bridge pattern:
		// WS events push chunks into a queue, the generator yields from the queue.
		const chunkQueue: Array<ApiStreamChunk | { type: "done" } | { type: "error"; error: string }> = []
		let resolveWaiting: (() => void) | null = null

		const pushChunk = (chunk: ApiStreamChunk | { type: "done" } | { type: "error"; error: string }) => {
			chunkQueue.push(chunk)
			if (resolveWaiting) {
				resolveWaiting()
				resolveWaiting = null
			}
		}

		// Register WS event listeners
		this.transport.on("display_text", (text: string, turn?: number, payload?: any) => {
			Logger.info(`[ZulongHandler] ← display_text: text=${text.length} chars, turn=${turn}, payload keys=${payload ? Object.keys(payload).join(",") : "none"}`)
			// 🔥 修复：忽略task_result字段，避免重复显示
			// task_result是完成标记中的完整文本，不应再次显示
			if (text && !payload?.task_result) {
				pushChunk({ type: "text" as const, text })
			}
			// task_status=completed 只表示文本流结束；v2.7 的 task_summary / graph_memory_diff
			// 会在其后抵达，必须等待正式 task_complete 再结束 generator。
			if (payload && payload.complete && payload.task_status === "completed") {
				Logger.info(`[ZulongHandler] ← display_text标记文本完成，等待task_complete以接收最终interaction事件`)
			}
		})

		this.transport.on("display_reasoning", (reasoning: string) => {
			if (reasoning) {
				Logger.info(`[ZulongHandler] \u2190 display_reasoning (${reasoning.length} chars)`)
				pushChunk({ type: "reasoning" as const, reasoning })
			}
		})

		this.transport.on("tool_request", (req: ZulongToolRequest) => {
			// Convert each tool call to an ApiStreamToolCallsChunk
			const toolCalls = req.tool_calls || []
			Logger.info(`[ZulongHandler] \u2190 tool_request: ${toolCalls.length} calls: ${toolCalls.map((t) => t.function.name).join(", ")}`)
			for (let i = 0; i < toolCalls.length; i++) {
				const tc = toolCalls[i]
				pushChunk({
					type: "tool_calls" as const,
					tool_call: {
						call_id: tc.id,
						function: {
							id: tc.id,
							name: tc.function.name,
							arguments: tc.function.arguments,
						},
					},
					// Signal that all tool calls for this request are ready for execution
					isComplete: i === toolCalls.length - 1,
				})
			}
		})

		this.transport.on("task_complete", (_result: string, payload?: any) => {
			const phase = payload?.phase || payload?.status || "succeeded"
			Logger.info(`[ZulongHandler] \u2190 task_complete phase=${phase}`)
			pushChunk({ type: "status_update", phase })
			pushChunk({ type: "done" })
		})

		this.transport.on("task_error", (error: string) => {
			Logger.error(`[ZulongHandler] \u2190 task_error: ${error}`)
			pushChunk({ type: "error", error })
		})

		// 🎯 P3改进：任务进度汇报（作为心跳保持流存活，不显示给用户）
		this.transport.on("task_progress", (progress: { phase: string; message: string; current_turn?: number; max_turns?: number }) => {
			Logger.info(`[ZulongHandler] \u2190 task_progress: phase=${progress.phase}, message=${progress.message}`)
			// 不作为 text 显示给用户，仅作为 heartbeat chunk 重置流超时
			pushChunk({ type: "status_update", turn: progress.current_turn, phase: progress.phase })
		})

		// P2-15: 监听FC循环状态更新（进度展示）
		this.transport.on("status_update", (payload: { turn?: number; phase?: string }) => {
			Logger.info(`[ZulongHandler] \u2190 status_update: turn=${payload.turn} phase=${payload.phase}`)
			// Push as chunk so the stream loop can check abort flag
			pushChunk({ type: "status_update", turn: payload.turn, phase: payload.phase })
		})

		this.transport.on("interaction", (interaction: any, rawMsg?: { type?: string; payload?: Record<string, any> }) => {
			Logger.info(
				`[ZulongHandler] \u2190 interaction: kind=${interaction?.kind} status=${interaction?.status} source=${rawMsg?.type || "unknown"}`,
			)
			pushChunk({
				type: "interaction",
				interaction,
				sourceEvent: rawMsg?.type,
				turn: interaction?.turn ?? rawMsg?.payload?.turn,
			})
		})

		this.transport.on("error", (err: Error) => {
			Logger.error(`[ZulongHandler] \u2190 transport error: ${err.message}`)
			pushChunk({ type: "error", error: err.message })
		})

		this.transport.on("model_info", (info: { contextWindow?: number }) => {
			if (info.contextWindow && info.contextWindow > 0) {
				this.updateModelInfo({ contextWindow: info.contextWindow })
			}
		})

		this.transport.on("disconnected", (code: number, reason: string) => {
			Logger.warn(`[ZulongHandler] \u2190 disconnected: code=${code} reason=${reason}`)
			// 🔥 P0修复：WS断开时推送text+done（而非error+done），确保generator走正常完成路径
			// 正常done路径会触发finalizeApiReqMsg()设置partial=false，从而重置前端"思考中"状态
			// 之前用error chunk会导致generator提前break，done永远不会被处理，partial残留为true
			pushChunk({ type: "text" as const, text: `\n[Zulong] WebSocket连接已断开 (code=${code}): ${reason}` })
			pushChunk({ type: "done" })
		})

		this.transport.on("system_ready", (payload: { status?: string; failed_modules?: string[] }) => {
			Logger.info(`[ZulongHandler] \u2190 system_ready: status=${payload.status}, failed_modules=${payload.failed_modules?.join(",") || "none"}`)
		})

		// 恢复任务属于 L1-B/L2 的语义判断，provider 不再用关键词硬判。
		// 这里统一进入 session_start；带明确 graph_id 的协议级恢复仍由后端
		// session_resume 通道处理。
		let projectId: string | undefined
		try {
			const projectJsonPath = path.join(cwd, ".zulong", "project.json")
			if (fs.existsSync(projectJsonPath)) {
				const projData = JSON.parse(fs.readFileSync(projectJsonPath, "utf-8"))
				projectId = projData.project_id
			}
		} catch {
			// 忽略读取失败
		}
		Logger.info(`[ZulongHandler] \u2192 session_start, cwd=${cwd}, project_id=${projectId || "none"}`)
		this.transport.sendSessionStart(taskText, cwd, systemPrompt, projectId, this.options.zulongApprovalMode)

		// Yield chunks from the queue
		const STREAM_TIMEOUT_MS = 330 * 1000 // 后端CORE超时300s + 30s缓冲，防止IDE永久卡在"思考中"
		let lastChunkTime = Date.now()
		try {
			while (true) {
				if (this.abortController?.signal.aborted) {
					Logger.warn("[ZulongHandler] Abort detected, sending cancel")
					this.transport.sendCancel()
					break
				}

				if (chunkQueue.length === 0) {
					// Wait for next chunk with timeout protection
					const timeoutMs = STREAM_TIMEOUT_MS - (Date.now() - lastChunkTime)
					if (timeoutMs <= 0) {
						Logger.error("[ZulongHandler] 模型响应超时，请稍后重试")
						pushChunk({ type: "done" })
						continue
					}
					await new Promise<void>((resolve) => {
						let resolved = false
						const done = () => { if (!resolved) { resolved = true; resolve() } }
						resolveWaiting = done
						if (this.abortController) {
							this.abortController.signal.addEventListener("abort", done, { once: true })
						}
						setTimeout(done, timeoutMs)
					})
					continue
				}

				lastChunkTime = Date.now()
				const chunk = chunkQueue.shift()!

				if ("type" in chunk && chunk.type === "done") {
					Logger.info("[ZulongHandler] Task done, yielding final usage")
					// Yield final usage estimate
					yield {
						type: "usage" as const,
						inputTokens: 0,
						outputTokens: 0,
					}
					break
				}

				if ("type" in chunk && chunk.type === "error") {
					Logger.error(`[ZulongHandler] Yielding error: ${(chunk as any).error}`)
					yield {
						type: "text" as const,
						text: `\n[Zulong Error] ${(chunk as any).error}`,
					}
					break
				}

				Logger.debug(`[ZulongHandler] Yielding chunk: type=${chunk.type}`)
				yield chunk as ApiStreamChunk
			}
		} finally {
			// 🔥 长连接模式：不断开WebSocket，保持连接复用
			Logger.info("[ZulongHandler] Task completed, WebSocket kept alive for next task")
		}
	}

	/**
	 * Send tool execution result back to backend.
	 * Called by Task layer after executing a remote tool.
	 */
	sendToolResult(callId: string, toolName: string, result: string, isError: boolean = false): void {
		Logger.info(`[ZulongHandler] \u2192 sendToolResult: call_id=${callId}, tool=${toolName}, is_error=${isError}, result_len=${result.length}`)
		if (this.transport?.isConnected) {
			this.transport.sendToolResult(callId, toolName, result, isError)
		} else {
			Logger.error(`[ZulongHandler] Cannot send tool result - transport not connected! call_id=${callId}, tool=${toolName}`)
		}
	}

	/**
	 * 显式关闭WebSocket连接（IDE关闭时调用）
	 */
	dispose(): void {
		Logger.info("[ZulongHandler] Explicit dispose called, closing WebSocket")
		this.transport?.dispose()
		this.transport = null
	}

	abort(): void {
		Logger.warn("[ZulongHandler] Abort requested")
		this.abortController?.abort()
		if (this.transport?.isConnected) {
			this.transport.sendCancel()
		}
	}
}
