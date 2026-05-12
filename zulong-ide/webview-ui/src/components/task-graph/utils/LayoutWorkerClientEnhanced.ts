/**
 * 布局计算 Worker 客户端（增强版）
 * 
 * 支持：
 * - 分批布局计算
 * - 超时降级处理
 * - 布局缓存
 * - 渐进式结果推送
 */

import type { Position } from "../types"

export type LayoutAlgorithm = "hierarchy" | "force" | "force_incremental"
export type LayoutMessageType = 
	| "layout_request"
	| "layout_result"
	| "layout_partial"
	| "layout_refined"
	| "layout_error"
	| "layout_stop"

export interface LayoutRequestMessage {
	type: "layout_request"
	nodes: Array<{ id: string; parentId: string | null }>
	edges: Array<{ source: string; target: string; type: string }>
	config: {
		algorithm: LayoutAlgorithm
		width: number
		height: number
		batchSize?: number
		timeout?: number
	}
	requestId: string
}

export interface LayoutResultMessage {
	type: "layout_result" | "layout_partial" | "layout_refined"
	positions: Record<string, Position>
	durationMs?: number
	isPartial?: boolean
	requestId?: string
}

export interface LayoutErrorMessage {
	type: "layout_error"
	error: string
	requestId?: string
}

export interface LayoutStopMessage {
	type: "layout_stop"
}

type LayoutMessage = LayoutRequestMessage | LayoutResultMessage | LayoutErrorMessage | LayoutStopMessage

export interface LayoutOptions {
	algorithm?: LayoutAlgorithm
	width?: number
	height?: number
	batchSize?: number
	timeout?: number
	onPartialResult?: (positions: Record<string, Position>) => void
	onProgress?: (progress: number) => void
}

const DEFAULT_OPTIONS: Required<Omit<LayoutOptions, 'onPartialResult' | 'onProgress'>> = {
	algorithm: "hierarchy",
	width: 1200,
	height: 800,
	batchSize: 200,
	timeout: 5000,
}

export class LayoutWorkerClient {
	private worker: Worker | null = null
	private pendingRequests: Map<string, {
		resolve: (positions: Record<string, Position>) => void
		reject: (error: Error) => void
		onPartialResult?: (positions: Record<string, Position>) => void
		timeout: NodeJS.Timeout | null
	}> = new Map()
	private requestIdCounter: number = 0

	constructor() {
		this.initWorker()
	}

	private initWorker(): void {
		try {
			this.worker = new Worker(
				new URL("../workers/layout-worker.ts", import.meta.url),
				{ type: "module" }
			)

			this.worker.onmessage = this.handleWorkerMessage.bind(this)
			this.worker.onerror = this.handleWorkerError.bind(this)

			console.log("[LayoutWorkerClient] Worker initialized")
		} catch (error) {
			console.error("[LayoutWorkerClient] Failed to initialize worker:", error)
		}
	}

	private handleWorkerMessage(event: MessageEvent): void {
		const message = event.data as LayoutResultMessage | LayoutErrorMessage

		if (message.type === "layout_result" || message.type === "layout_refined") {
			const requestId = message.requestId || ""
			const pending = this.pendingRequests.get(requestId)

			if (pending) {
				if (pending.timeout) {
					clearTimeout(pending.timeout)
				}
				this.pendingRequests.delete(requestId)
				pending.resolve(message.positions)
			}
		} else if (message.type === "layout_partial") {
			const requestId = message.requestId || ""
			const pending = this.pendingRequests.get(requestId)

			if (pending && pending.onPartialResult) {
				pending.onPartialResult(message.positions)
			}

			if (!requestId) {
				for (const [, req] of this.pendingRequests) {
					if (req.onPartialResult) {
						req.onPartialResult(message.positions)
					}
				}
			}
		} else if (message.type === "layout_error") {
			const requestId = message.requestId || ""
			const pending = this.pendingRequests.get(requestId)

			if (pending) {
				if (pending.timeout) {
					clearTimeout(pending.timeout)
				}
				this.pendingRequests.delete(requestId)
				pending.reject(new Error(message.error))
			}
		}
	}

	private handleWorkerError(error: ErrorEvent): void {
		console.error("[LayoutWorkerClient] Worker error:", error)

		for (const [requestId, pending] of this.pendingRequests) {
			if (pending.timeout) {
				clearTimeout(pending.timeout)
			}
			this.pendingRequests.delete(requestId)
			pending.reject(new Error(`Worker error: ${error.message}`))
		}
	}

	/**
	 * 计算布局
	 */
	async computeLayout(
		nodes: Array<{ id: string; parentId: string | null }>,
		edges: Array<{ source: string; target: string; type: string }>,
		options: LayoutOptions = {},
	): Promise<Record<string, Position>> {
		if (!this.worker) {
			console.warn("[LayoutWorkerClient] Worker not available, using fallback")
			return this.fallbackLayout(nodes, options)
		}

		const config = { ...DEFAULT_OPTIONS, ...options }
		const requestId = this.generateRequestId()

		return new Promise((resolve, reject) => {
			const timeoutId = setTimeout(() => {
				this.pendingRequests.delete(requestId)
				console.warn(`[LayoutWorkerClient] Layout timeout for request ${requestId}, using fallback`)
				resolve(this.fallbackLayout(nodes, options))
			}, config.timeout)

			this.pendingRequests.set(requestId, {
				resolve,
				reject,
				onPartialResult: options.onPartialResult,
				timeout: timeoutId,
			})

			const message: LayoutRequestMessage = {
				type: "layout_request",
				nodes,
				edges,
				config: {
					algorithm: config.algorithm,
					width: config.width,
					height: config.height,
					batchSize: config.batchSize,
					timeout: config.timeout,
				},
				requestId,
			}

			this.worker!.postMessage(message)
		})
	}

	/**
	 * 分批计算布局
	 */
	async computeLayoutBatched(
		nodes: Array<{ id: string; parentId: string | null }>,
		edges: Array<{ source: string; target: string; type: string }>,
		options: LayoutOptions = {},
	): Promise<Record<string, Position>> {
		const config = { ...DEFAULT_OPTIONS, ...options }
		const batchSize = config.batchSize

		if (nodes.length <= batchSize) {
			return this.computeLayout(nodes, edges, options)
		}

		console.log(`[LayoutWorkerClient] Computing layout in batches: ${nodes.length} nodes, batch size ${batchSize}`)

		const positions: Record<string, Position> = {}
		const batches = this.createBatches(nodes, batchSize)

		for (let i = 0; i < batches.length; i++) {
			const batch = batches[i]
			options.onProgress?.((i + 1) / batches.length)

			const batchPositions = await this.computeLayout(
				batch,
				edges.filter(e => batch.some(n => n.id === e.source || n.id === e.target)),
				options,
			)

			Object.assign(positions, batchPositions)

			await this.nextFrame()
		}

		return positions
	}

	/**
	 * 创建批次
	 */
	private createBatches(
		nodes: Array<{ id: string; parentId: string | null }>,
		batchSize: number,
	): Array<Array<{ id: string; parentId: string | null }>> {
		const batches: Array<Array<{ id: string; parentId: string | null }>> = []

		for (let i = 0; i < nodes.length; i += batchSize) {
			batches.push(nodes.slice(i, i + batchSize))
		}

		return batches
	}

	/**
	 * 降级布局（随机分布）
	 */
	private fallbackLayout(
		nodes: Array<{ id: string; parentId: string | null }>,
		options: LayoutOptions = {},
	): Record<string, Position> {
		const config = { ...DEFAULT_OPTIONS, ...options }
		const positions: Record<string, Position> = {}

		const cols = Math.ceil(Math.sqrt(nodes.length))
		const spacing = 100

		nodes.forEach((node, index) => {
			const row = Math.floor(index / cols)
			const col = index % cols

			positions[node.id] = {
				x: col * spacing + config.width / 2 - (cols * spacing) / 2,
				y: row * spacing + config.height / 2 - (Math.ceil(nodes.length / cols) * spacing) / 2,
			}
		})

		console.log(`[LayoutWorkerClient] Using fallback layout for ${nodes.length} nodes`)
		return positions
	}

	/**
	 * 停止布局计算
	 */
	stop(): void {
		if (this.worker) {
			const message: LayoutStopMessage = { type: "layout_stop" }
			this.worker.postMessage(message)
		}

		for (const [requestId, pending] of this.pendingRequests) {
			if (pending.timeout) {
				clearTimeout(pending.timeout)
			}
			this.pendingRequests.delete(requestId)
		}

		console.log("[LayoutWorkerClient] Layout stopped")
	}

	/**
	 * 销毁客户端
	 */
	dispose(): void {
		this.stop()

		if (this.worker) {
			this.worker.terminate()
			this.worker = null
		}

		console.log("[LayoutWorkerClient] Disposed")
	}

	/**
	 * 生成请求ID
	 */
	private generateRequestId(): string {
		return `layout_req_${++this.requestIdCounter}_${Date.now()}`
	}

	/**
	 * 等待下一帧
	 */
	private nextFrame(): Promise<void> {
		return new Promise(resolve => requestAnimationFrame(resolve))
	}

	/**
	 * 检查 Worker 是否可用
	 */
	isAvailable(): boolean {
		return this.worker !== null
	}
}

export const layoutWorkerClient = new LayoutWorkerClient()
