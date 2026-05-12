/**
 * 渐进式加载控制器
 * 
 * 管理首屏和增量加载
 * - 首屏加载：请求第1页数据，立即触发布局计算和渲染
 * - 增量加载：首屏渲染完成后，后台异步请求剩余页
 * - 加载状态管理：idle | loading_first_page | rendering_first_page | loading_incremental | completed
 */

import { useGraphStore } from "../store/useGraphStore"

export type LoadingState = "idle" | "loading_first_page" | "rendering_first_page" | "loading_incremental" | "completed"

export interface PageInfo {
	page: number
	pageSize: number
	totalNodes: number
	totalPages: number
	hasNext: boolean
	hasPrev: boolean
	cursor?: string
}

export interface ProgressiveLoaderConfig {
	graphId: string
	pageSize?: number
	onProgress?: (loaded: number, total: number) => void
	onComplete?: () => void
	onError?: (error: Error) => void
}

const DEFAULT_PAGE_SIZE = 500

export class ProgressiveLoader {
	private static instance: ProgressiveLoader | null = null

	private loadingState: LoadingState = "idle"
	private currentPage: number = 1
	private totalPages: number = 0
	private totalNodes: number = 0
	private loadedNodes: number = 0
	private abortController: AbortController | null = null
	private config: ProgressiveLoaderConfig | null = null

	private constructor() {}

	static getInstance(): ProgressiveLoader {
		if (!ProgressiveLoader.instance) {
			ProgressiveLoader.instance = new ProgressiveLoader()
		}
		return ProgressiveLoader.instance
	}

	/**
	 * 开始渐进式加载
	 */
	async startLoading(config: ProgressiveLoaderConfig): Promise<void> {
		if (this.loadingState !== "idle") {
			console.warn(`[ProgressiveLoader] Already loading, current state: ${this.loadingState}`)
			return
		}

		this.config = config
		this.currentPage = 1
		this.loadedNodes = 0
		this.abortController = new AbortController()

		try {
			this.loadingState = "loading_first_page"
			console.log(`[ProgressiveLoader] Starting first page load for graph: ${config.graphId}`)

			const pageSize = config.pageSize || DEFAULT_PAGE_SIZE
			const firstPageData = await this.fetchPage(config.graphId, 1, pageSize)

			this.totalNodes = firstPageData.pageInfo.totalNodes
			this.totalPages = firstPageData.pageInfo.totalPages
			this.loadedNodes = firstPageData.nodes.length

			this.applyData(firstPageData)

			config.onProgress?.(this.loadedNodes, this.totalNodes)

			this.loadingState = "rendering_first_page"
			console.log(`[ProgressiveLoader] First page loaded: ${this.loadedNodes}/${this.totalNodes} nodes`)

			await this.waitForRenderComplete()

			if (firstPageData.pageInfo.hasNext) {
				this.loadingState = "loading_incremental"
				await this.loadIncremental()
			}

			this.loadingState = "completed"
			console.log(`[ProgressiveLoader] Loading completed: ${this.loadedNodes}/${this.totalNodes} nodes`)
			config.onComplete?.()

		} catch (error) {
			if (error instanceof Error && error.name === "AbortError") {
				console.log("[ProgressiveLoader] Loading aborted")
			} else {
				console.error("[ProgressiveLoader] Loading failed:", error)
				config.onError?.(error instanceof Error ? error : new Error(String(error)))
			}
			this.loadingState = "idle"
		}
	}

	/**
	 * 加载增量数据
	 */
	private async loadIncremental(): Promise<void> {
		if (!this.config) return

		const pageSize = this.config.pageSize || DEFAULT_PAGE_SIZE

		while (this.currentPage < this.totalPages && !this.isAborted()) {
			this.currentPage++

			console.log(`[ProgressiveLoader] Loading page ${this.currentPage}/${this.totalPages}`)

			const pageData = await this.fetchPage(this.config.graphId, this.currentPage, pageSize)

			this.appendData(pageData)
			this.loadedNodes += pageData.nodes.length

			this.config.onProgress?.(this.loadedNodes, this.totalNodes)

			await this.sleep(100)
		}
	}

	/**
	 * 获取单页数据
	 */
	private async fetchPage(
		graphId: string,
		page: number,
		pageSize: number,
	): Promise<{ nodes: any[]; edges: any[]; pageInfo: PageInfo }> {
		const response = await fetch(
			`/api/graph/${graphId}/nodes?page=${page}&page_size=${pageSize}`,
			{
				signal: this.abortController?.signal,
			},
		)

		if (!response.ok) {
			throw new Error(`Failed to fetch page ${page}: ${response.statusText}`)
		}

		const data = await response.json()

		return {
			nodes: data.nodes || [],
			edges: data.edges || [],
			pageInfo: {
				page: data.page_info.page,
				pageSize: data.page_info.page_size,
				totalNodes: data.page_info.total_nodes,
				totalPages: data.page_info.total_pages,
				hasNext: data.page_info.has_next,
				hasPrev: data.page_info.has_prev,
				cursor: data.page_info.cursor,
			},
		}
	}

	/**
	 * 应用首屏数据
	 */
	private applyData(data: { nodes: any[]; edges: any[] }): void {
		const store = useGraphStore.getState()

		const nodesMap = new Map()
		for (const node of data.nodes) {
			nodesMap.set(node.id, {
				id: node.id,
				label: node.label || "",
				type: node.type || "task",
				status: node.status || "pending",
				desc: node.desc || "",
				result: node.result || "",
				files: node.files || [],
				taskDomain: node.task_domain || node.taskDomain || "general",
				parentId: node.parent_id || node.parentId || null,
				position: { x: 0, y: 0 },
				metadata: node.metadata || {},
			})
		}

		const edgesMap = new Map()
		for (const edge of data.edges) {
			edgesMap.set(edge.id, {
				id: edge.id,
				source: edge.source,
				target: edge.target,
				type: edge.type || "dependency",
				via: edge.via || "",
				cross: edge.cross || false,
			})
		}

		useGraphStore.setState({
			nodes: nodesMap,
			edges: edgesMap,
		})
	}

	/**
	 * 追加增量数据
	 */
	private appendData(data: { nodes: any[]; edges: any[] }): void {
		const store = useGraphStore.getState()
		const currentNodes = new Map(store.nodes)
		const currentEdges = new Map(store.edges)

		for (const node of data.nodes) {
			if (!currentNodes.has(node.id)) {
				currentNodes.set(node.id, {
					id: node.id,
					label: node.label || "",
					type: node.type || "task",
					status: node.status || "pending",
					desc: node.desc || "",
					result: node.result || "",
					files: node.files || [],
					taskDomain: node.task_domain || node.taskDomain || "general",
					parentId: node.parent_id || node.parentId || null,
					position: { x: 0, y: 0 },
					metadata: node.metadata || {},
				})
			}
		}

		for (const edge of data.edges) {
			if (!currentEdges.has(edge.id)) {
				currentEdges.set(edge.id, {
					id: edge.id,
					source: edge.source,
					target: edge.target,
					type: edge.type || "dependency",
					via: edge.via || "",
					cross: edge.cross || false,
				})
			}
		}

		useGraphStore.setState({
			nodes: currentNodes,
			edges: currentEdges,
		})
	}

	/**
	 * 等待渲染完成
	 */
	private async waitForRenderComplete(): Promise<void> {
		return new Promise((resolve) => {
			requestAnimationFrame(() => {
				requestAnimationFrame(() => {
					resolve()
				})
			})
		})
	}

	/**
	 * 取消加载
	 */
	cancel(): void {
		if (this.abortController) {
			this.abortController.abort()
			this.abortController = null
		}
		this.loadingState = "idle"
		console.log("[ProgressiveLoader] Loading cancelled")
	}

	/**
	 * 检查是否已中止
	 */
	private isAborted(): boolean {
		return this.abortController?.signal.aborted || false
	}

	/**
	 * 获取加载状态
	 */
	getLoadingState(): LoadingState {
		return this.loadingState
	}

	/**
	 * 获取加载进度
	 */
	getProgress(): { loaded: number; total: number; percentage: number } {
		return {
			loaded: this.loadedNodes,
			total: this.totalNodes,
			percentage: this.totalNodes > 0 ? Math.round((this.loadedNodes / this.totalNodes) * 100) : 0,
		}
	}

	/**
	 * 延迟函数
	 */
	private sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms))
	}

	/**
	 * 重置
	 */
	reset(): void {
		this.cancel()
		this.currentPage = 1
		this.totalPages = 0
		this.totalNodes = 0
		this.loadedNodes = 0
		this.config = null
	}
}

export const progressiveLoader = ProgressiveLoader.getInstance()
