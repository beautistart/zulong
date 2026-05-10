import type { Position, LayoutWorkerMessage, LayoutRequestMessage } from "../types"

export class LayoutWorkerClient {
	private worker: Worker | null = null
	private resolveMap = new Map<
		string,
		{
			resolve: (positions: Record<string, Position>) => void
			reject: (error: Error) => void
		}
	>()
	private onPartialCallback: ((positions: Record<string, Position>) => void) | null = null
	private onRefinedCallback: ((positions: Record<string, Position>) => void) | null = null
	private requestId = 0
	private timeoutTimer: ReturnType<typeof setTimeout> | null = null

	private ensureWorker(): Worker {
		if (this.worker) return this.worker
		this.worker = new Worker(
			new URL("../workers/layout-worker.ts", import.meta.url),
			{ type: "module" }
		)
		this.worker.onmessage = (e: MessageEvent<LayoutWorkerMessage>) => {
			this.handleMessage(e.data)
		}
		this.worker.onerror = (e: ErrorEvent) => {
			console.error("LayoutWorker error:", e.message)
			for (const [, entry] of this.resolveMap) {
				entry.reject(new Error(`Worker error: ${e.message}`))
			}
			this.resolveMap.clear()
		}
		return this.worker
	}

	private handleMessage(msg: LayoutWorkerMessage): void {
		switch (msg.type) {
			case "layout_result": {
				const key = `layout_${this.requestId}`
				const entry = this.resolveMap.get(key)
				if (entry) {
					entry.resolve(msg.positions)
					this.resolveMap.delete(key)
				}
				this.clearTimeout()
				break
			}
			case "layout_partial": {
				if (this.onPartialCallback) {
					this.onPartialCallback(msg.positions)
				}
				break
			}
			case "layout_refined": {
				if (this.onRefinedCallback) {
					this.onRefinedCallback(msg.positions)
				}
				const key = `layout_${this.requestId}`
				const entry = this.resolveMap.get(key)
				if (entry) {
					entry.resolve(msg.positions)
					this.resolveMap.delete(key)
				}
				break
			}
			case "layout_error": {
				const key = `layout_${this.requestId}`
				const entry = this.resolveMap.get(key)
				if (entry) {
					entry.reject(new Error(msg.error))
					this.resolveMap.delete(key)
				}
				this.clearTimeout()
				break
			}
		}
	}

	async requestLayout(
		nodes: Array<{ id: string; parentId: string | null }>,
		edges: Array<{ source: string; target: string; type: string }>,
		config: LayoutRequestMessage["config"],
		timeoutMs = 10000
	): Promise<Record<string, Position>> {
		const worker = this.ensureWorker()
		this.requestId++
		const key = `layout_${this.requestId}`

		const promise = new Promise<Record<string, Position>>((resolve, reject) => {
			this.resolveMap.set(key, { resolve, reject })
		})

		this.timeoutTimer = setTimeout(() => {
			const entry = this.resolveMap.get(key)
			if (entry) {
				entry.reject(new Error("Layout timeout"))
				this.resolveMap.delete(key)
			}
		}, timeoutMs)

		const msg: LayoutRequestMessage = { type: "layout_request", nodes, edges, config }
		worker.postMessage(msg)

		return promise
	}

	onPartial(callback: (positions: Record<string, Position>) => void): void {
		this.onPartialCallback = callback
	}

	onRefined(callback: (positions: Record<string, Position>) => void): void {
		this.onRefinedCallback = callback
	}

	stop(): void {
		if (this.worker) {
			this.worker.postMessage({ type: "layout_stop" })
		}
		this.clearTimeout()
	}

	terminate(): void {
		if (this.worker) {
			this.worker.terminate()
			this.worker = null
		}
		for (const [, entry] of this.resolveMap) {
			entry.reject(new Error("Worker terminated"))
		}
		this.resolveMap.clear()
		this.clearTimeout()
	}

	private clearTimeout(): void {
		if (this.timeoutTimer) {
			clearTimeout(this.timeoutTimer)
			this.timeoutTimer = null
		}
	}
}
