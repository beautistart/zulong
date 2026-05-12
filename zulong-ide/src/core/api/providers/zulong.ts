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
import { ZulongStorageMessage } from "@/shared/messages/content"
import { ZulongTool } from "@/shared/tools"
import { ApiHandler, ApiHandlerModel, CommonApiHandlerOptions } from "../index"
import { ApiStream, ApiStreamChunk } from "../transform/stream"
import { ZulongWebSocket, ZulongToolRequest } from "../transport/zulong-websocket"
import { Logger } from "@/shared/services/Logger"
import fs from "fs"
import path from "path"

export interface ZulongHandlerOptions extends CommonApiHandlerOptions {
	zulongServerUrl?: string
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

	constructor(options: ZulongHandlerOptions) {
		this.options = options
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

	async *createMessage(
		systemPrompt: string,
		messages: ZulongStorageMessage[],
		tools?: ZulongTool[],
		_useResponseApi?: boolean,
	): ApiStream {
		const serverUrl = this.options.zulongServerUrl || "ws://127.0.0.1:8090"
		this.abortController = new AbortController()

		Logger.info(`[ZulongHandler] createMessage() starting, serverUrl=${serverUrl}`)

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

		// Connect to backend
		this.transport = new ZulongWebSocket(serverUrl)

		try {
			await this.transport.connect()
			Logger.info("[ZulongHandler] Connected to backend")

			this.transport.on("audio_transcript", (text: string, isFinal: boolean) => {
				Logger.info(`[ZulongHandler] ← audio_transcript: "${text}" (is_final=${isFinal})`)
			})
		} catch (err) {
			Logger.error(`[ZulongHandler] Connection failed: ${err}`)
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
		this.transport.on("display_text", (text: string) => {
			if (text) {
				Logger.info(`[ZulongHandler] \u2190 display_text (${text.length} chars)`)
				pushChunk({ type: "text" as const, text })
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

		this.transport.on("task_complete", (_result: string) => {
			Logger.info("[ZulongHandler] \u2190 task_complete")
			pushChunk({ type: "done" })
		})

		this.transport.on("task_error", (error: string) => {
			Logger.error(`[ZulongHandler] \u2190 task_error: ${error}`)
			pushChunk({ type: "error", error })
		})

		// P2-15: 监听FC循环状态更新（进度展示）
		this.transport.on("status_update", (payload: { turn?: number; phase?: string }) => {
			Logger.info(`[ZulongHandler] \u2190 status_update: turn=${payload.turn} phase=${payload.phase}`)
			// Push as chunk so the stream loop can check abort flag
			pushChunk({ type: "status_update", turn: payload.turn, phase: payload.phase })
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
			// WS 意外断开时，通知 generator 退出，避免永久 hang
			pushChunk({ type: "done" })
		})

		// Detect resume: if there are prior assistant messages, this is a resumed session
		const hasHistory = messages.some((m) => m.role === "assistant")
		if (hasHistory) {
			const graphId = this.extractGraphId(messages)
			Logger.info(`[ZulongHandler] \u2192 session_resume (detected prior history), cwd=${cwd}, graph_id=${graphId || "none"}`)
			this.transport.sendSessionResume(taskText, cwd, systemPrompt, graphId)
		} else {
			// 检测当前 cwd 是否为祖龙项目，提取 project_id
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
			this.transport.sendSessionStart(taskText, cwd, systemPrompt, projectId)
		}

		// Yield chunks from the queue
		const STREAM_TIMEOUT_MS = 5 * 60 * 1000 // 5分钟无数据超时，防止IDE永久卡在"思考中"
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
						Logger.error("[ZulongHandler] Stream timeout - no data received, forcing done")
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
			Logger.info("[ZulongHandler] Session ended, transport disposed")
			this.transport.dispose()
			this.transport = null
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
	 * 从历史消息中提取最近的 graph_id (格式: tg_NNNNNNNNNN)
	 */
	private extractGraphId(messages: ZulongStorageMessage[]): string | undefined {
		const pattern = /\btg_\d{10,13}\b/
		for (let i = messages.length - 1; i >= 0; i--) {
			const msg = messages[i]
			if (msg.role !== "assistant") continue
			const content = msg.content
			if (typeof content === "string") {
				const match = content.match(pattern)
				if (match) return match[0]
			} else if (Array.isArray(content)) {
				for (const block of content) {
					if (typeof block === "object" && block !== null) {
						const text = (block as any).text || JSON.stringify(block)
						const match = text.match(pattern)
						if (match) return match[0]
					}
				}
			}
		}
		return undefined
	}

	abort(): void {
		Logger.warn("[ZulongHandler] Abort requested")
		this.abortController?.abort()
		if (this.transport?.isConnected) {
			this.transport.sendCancel()
		}
	}
}
