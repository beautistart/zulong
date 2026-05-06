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

export interface ZulongHandlerOptions extends CommonApiHandlerOptions {
	zulongServerUrl?: string
}

// Default model info for Zulong backend-managed model
const ZULONG_MODEL_INFO: ModelInfo = {
	name: "zulong-agent",
	maxTokens: 8192,
	contextWindow: 32768,
	supportsImages: false,
	supportsPromptCache: false,
	supportsReasoning: false,
}

export class ZulongHandler implements ApiHandler {
	private options: ZulongHandlerOptions
	private transport: ZulongWebSocket | null = null
	private abortController: AbortController | null = null

	constructor(options: ZulongHandlerOptions) {
		this.options = options
	}

	getModel(): ApiHandlerModel {
		return {
			id: "zulong-agent",
			info: ZULONG_MODEL_INFO,
		}
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

		this.transport.on("error", (err: Error) => {
			Logger.error(`[ZulongHandler] \u2190 transport error: ${err.message}`)
			pushChunk({ type: "error", error: err.message })
		})

		// Detect resume: if there are prior assistant messages, this is a resumed session
		const hasHistory = messages.some((m) => m.role === "assistant")
		if (hasHistory) {
			Logger.info(`[ZulongHandler] \u2192 session_resume (detected prior history), cwd=${cwd}`)
			this.transport.sendSessionResume(taskText, cwd, systemPrompt)
		} else {
			Logger.info(`[ZulongHandler] \u2192 session_start, cwd=${cwd}`)
			this.transport.sendSessionStart(taskText, cwd, systemPrompt)
		}

		// Yield chunks from the queue
		try {
			while (true) {
				if (this.abortController?.signal.aborted) {
					Logger.warn("[ZulongHandler] Abort detected, sending cancel")
					this.transport.sendCancel()
					break
				}

				if (chunkQueue.length === 0) {
					// Wait for next chunk
					await new Promise<void>((resolve) => {
						resolveWaiting = resolve
						// Also resolve on abort
						if (this.abortController) {
							this.abortController.signal.addEventListener(
								"abort",
								() => resolve(),
								{ once: true },
							)
						}
					})
					continue
				}

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

	abort(): void {
		Logger.warn("[ZulongHandler] Abort requested")
		this.abortController?.abort()
		if (this.transport?.isConnected) {
			this.transport.sendCancel()
		}
	}
}
