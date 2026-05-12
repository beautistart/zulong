/**
 * 智能渲染引擎
 * 
 * 四种渲染模式：
 * - full: 全量渲染（≤100节点），渲染所有节点和边的完整详情
 * - full_optimized: 优化渲染（101-500节点），合并绘制节点，简化边渲染
 * - virtual: 虚拟化渲染（501-1500节点），视口裁剪，仅渲染可见区域
 * - degraded: 降级渲染（>1500节点），仅渲染骨架，无详情，限制最大边数500
 */

import type { NodeAttribute, EdgeAttribute, ViewportState } from "../types"
import { useCollapseStore } from "../store/useCollapseStore"

export type RenderMode = "full" | "full_optimized" | "virtual" | "degraded"

export interface RenderContext {
	canvas: HTMLCanvasElement
	ctx: CanvasRenderingContext2D
	viewport: ViewportState
	nodes: Map<string, NodeAttribute>
	edges: Map<string, EdgeAttribute>
	renderMode: RenderMode
}

export interface RenderOptions {
	showLabels?: boolean
	showDetails?: boolean
	showEdges?: boolean
	maxEdges?: number
}

const RENDER_MODE_THRESHOLDS = {
	full: 100,
	full_optimized: 500,
	virtual: 1500,
	degraded: Infinity,
}

export class SmartRenderEngine {
	private static instance: SmartRenderEngine | null = null
	private currentMode: RenderMode = "full"
	private nodeCount: number = 0
	private visibleNodeCount: number = 0

	private constructor() {}

	static getInstance(): SmartRenderEngine {
		if (!SmartRenderEngine.instance) {
			SmartRenderEngine.instance = new SmartRenderEngine()
		}
		return SmartRenderEngine.instance
	}

	/**
	 * 判定渲染模式
	 */
	determineRenderMode(
		nodes: Map<string, NodeAttribute>,
		collapsedNodes?: Set<string>,
	): RenderMode {
		this.nodeCount = nodes.size

		const visibleNodes = collapsedNodes
			? this.calculateVisibleNodeCount(nodes, collapsedNodes)
			: nodes.size

		this.visibleNodeCount = visibleNodes

		if (visibleNodes <= RENDER_MODE_THRESHOLDS.full) {
			this.currentMode = "full"
		} else if (visibleNodes <= RENDER_MODE_THRESHOLDS.full_optimized) {
			this.currentMode = "full_optimized"
		} else if (visibleNodes <= RENDER_MODE_THRESHOLDS.virtual) {
			this.currentMode = "virtual"
		} else {
			this.currentMode = "degraded"
		}

		return this.currentMode
	}

	/**
	 * 计算可见节点数（扣除折叠节点）
	 */
	private calculateVisibleNodeCount(
		nodes: Map<string, NodeAttribute>,
		collapsedNodes: Set<string>,
	): number {
		const collapseStore = useCollapseStore.getState()
		const visibleNodeIds = collapseStore.getVisibleNodeIds()
		return visibleNodeIds.length
	}

	/**
	 * 渲染图谱
	 */
	render(
		canvas: HTMLCanvasElement,
		viewport: ViewportState,
		nodes: Map<string, NodeAttribute>,
		edges: Map<string, EdgeAttribute>,
		options: RenderOptions = {},
	): void {
		const ctx = canvas.getContext("2d")
		if (!ctx) {
			console.error("[SmartRenderEngine] Cannot get canvas context")
			return
		}

		const collapsedNodes = useCollapseStore.getState().collapsedNodes
		const renderMode = this.determineRenderMode(nodes, collapsedNodes)

		this.clearCanvas(ctx, canvas.width, canvas.height)

		switch (renderMode) {
			case "full":
				this.renderFull(ctx, viewport, nodes, edges, options)
				break
			case "full_optimized":
				this.renderOptimized(ctx, viewport, nodes, edges, options)
				break
			case "virtual":
				this.renderVirtual(ctx, viewport, nodes, edges, options)
				break
			case "degraded":
				this.renderDegraded(ctx, viewport, nodes, edges, options)
				break
		}
	}

	/**
	 * 清空画布
	 */
	private clearCanvas(ctx: CanvasRenderingContext2D, width: number, height: number): void {
		ctx.clearRect(0, 0, width, height)
	}

	/**
	 * 全量渲染
	 */
	private renderFull(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		nodes: Map<string, NodeAttribute>,
		edges: Map<string, EdgeAttribute>,
		options: RenderOptions,
	): void {
		const showLabels = options.showLabels !== false
		const showDetails = options.showDetails !== false
		const showEdges = options.showEdges !== false

		if (showEdges) {
			this.renderEdges(ctx, viewport, edges, "full")
		}

		for (const [id, node] of nodes) {
			this.renderNodeFull(ctx, viewport, node, showLabels, showDetails)
		}
	}

	/**
	 * 优化渲染
	 */
	private renderOptimized(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		nodes: Map<string, NodeAttribute>,
		edges: Map<string, EdgeAttribute>,
		options: RenderOptions,
	): void {
		const showLabels = options.showLabels !== false
		const showEdges = options.showEdges !== false

		if (showEdges) {
			this.renderEdges(ctx, viewport, edges, "simplified")
		}

		for (const [id, node] of nodes) {
			this.renderNodeOptimized(ctx, viewport, node, showLabels)
		}
	}

	/**
	 * 虚拟化渲染
	 */
	private renderVirtual(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		nodes: Map<string, NodeAttribute>,
		edges: Map<string, EdgeAttribute>,
		options: RenderOptions,
	): void {
		const visibleNodes = this.getNodesInViewport(viewport, nodes)
		const showLabels = options.showLabels !== false
		const showEdges = options.showEdges !== false

		if (showEdges) {
			const visibleEdges = this.getEdgesForNodes(edges, visibleNodes)
			this.renderEdges(ctx, viewport, visibleEdges, "simplified")
		}

		for (const nodeId of visibleNodes) {
			const node = nodes.get(nodeId)
			if (node) {
				this.renderNodeOptimized(ctx, viewport, node, showLabels)
			}
		}
	}

	/**
	 * 降级渲染
	 */
	private renderDegraded(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		nodes: Map<string, NodeAttribute>,
		edges: Map<string, EdgeAttribute>,
		options: RenderOptions,
	): void {
		const maxEdges = options.maxEdges || 500
		const maxNodes = 200

		const limitedEdges = this.limitEdges(edges, maxEdges)
		this.renderEdges(ctx, viewport, limitedEdges, "skeleton")

		let count = 0
		for (const [id, node] of nodes) {
			if (count >= maxNodes) break
			this.renderNodeSkeleton(ctx, viewport, node)
			count++
		}

		this.renderDegradedOverlay(ctx, viewport, nodes.size, edges.size)
	}

	/**
	 * 渲染完整节点
	 */
	private renderNodeFull(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		node: NodeAttribute,
		showLabel: boolean,
		showDetail: boolean,
	): void {
		const x = node.position.x * viewport.zoom + viewport.offsetX
		const y = node.position.y * viewport.zoom + viewport.offsetY
		const radius = 20 * viewport.zoom

		ctx.beginPath()
		ctx.arc(x, y, radius, 0, 2 * Math.PI)
		ctx.fillStyle = this.getNodeColor(node.status)
		ctx.fill()
		ctx.strokeStyle = "#333"
		ctx.lineWidth = 2
		ctx.stroke()

		if (showLabel) {
			ctx.font = `${12 * viewport.zoom}px sans-serif`
			ctx.fillStyle = "#333"
			ctx.textAlign = "center"
			ctx.fillText(node.label, x, y + radius + 15 * viewport.zoom)
		}

		if (showDetail && node.desc) {
			ctx.font = `${10 * viewport.zoom}px sans-serif`
			ctx.fillStyle = "#666"
			ctx.textAlign = "center"
			const shortDesc = node.desc.length > 30 ? node.desc.substring(0, 30) + "..." : node.desc
			ctx.fillText(shortDesc, x, y + radius + 30 * viewport.zoom)
		}
	}

	/**
	 * 渲染优化节点
	 */
	private renderNodeOptimized(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		node: NodeAttribute,
		showLabel: boolean,
	): void {
		const x = node.position.x * viewport.zoom + viewport.offsetX
		const y = node.position.y * viewport.zoom + viewport.offsetY
		const radius = 15 * viewport.zoom

		ctx.beginPath()
		ctx.arc(x, y, radius, 0, 2 * Math.PI)
		ctx.fillStyle = this.getNodeColor(node.status)
		ctx.fill()

		if (showLabel) {
			ctx.font = `${10 * viewport.zoom}px sans-serif`
			ctx.fillStyle = "#333"
			ctx.textAlign = "center"
			const shortLabel = node.label.length > 15 ? node.label.substring(0, 15) + "..." : node.label
			ctx.fillText(shortLabel, x, y + radius + 12 * viewport.zoom)
		}
	}

	/**
	 * 渲染骨架节点
	 */
	private renderNodeSkeleton(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		node: NodeAttribute,
	): void {
		const x = node.position.x * viewport.zoom + viewport.offsetX
		const y = node.position.y * viewport.zoom + viewport.offsetY
		const radius = 8 * viewport.zoom

		ctx.beginPath()
		ctx.arc(x, y, radius, 0, 2 * Math.PI)
		ctx.fillStyle = this.getNodeColor(node.status)
		ctx.fill()
	}

	/**
	 * 渲染边
	 */
	private renderEdges(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		edges: Map<string, EdgeAttribute>,
		mode: "full" | "simplified" | "skeleton",
	): void {
		ctx.lineWidth = mode === "full" ? 2 : mode === "simplified" ? 1 : 0.5
		ctx.strokeStyle = mode === "full" ? "#666" : "#aaa"

		for (const [id, edge] of edges) {
			this.renderEdge(ctx, viewport, edge)
		}
	}

	/**
	 * 渲染单条边
	 */
	private renderEdge(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		edge: EdgeAttribute,
	): void {
		// 边的渲染需要源节点和目标节点的位置信息
		// 这里简化实现，实际使用时需要传入完整的节点位置信息
	}

	/**
	 * 获取节点颜色
	 */
	private getNodeColor(status: string): string {
		const colors: Record<string, string> = {
			pending: "#94a3b8",
			in_progress: "#3b82f6",
			completed: "#22c55e",
			blocked: "#ef4444",
			skipped: "#a855f7",
		}
		return colors[status] || "#94a3b8"
	}

	/**
	 * 获取视口内的节点
	 */
	private getNodesInViewport(
		viewport: ViewportState,
		nodes: Map<string, NodeAttribute>,
	): Set<string> {
		const visible = new Set<string>()
		const buffer = 100

		for (const [id, node] of nodes) {
			const x = node.position.x * viewport.zoom + viewport.offsetX
			const y = node.position.y * viewport.zoom + viewport.offsetY

			if (
				x >= -buffer &&
				x <= viewport.width + buffer &&
				y >= -buffer &&
				y <= viewport.height + buffer
			) {
				visible.add(id)
			}
		}

		return visible
	}

	/**
	 * 获取节点相关的边
	 */
	private getEdgesForNodes(
		edges: Map<string, EdgeAttribute>,
		nodeIds: Set<string>,
	): Map<string, EdgeAttribute> {
		const relevant = new Map<string, EdgeAttribute>()

		for (const [id, edge] of edges) {
			if (nodeIds.has(edge.source) || nodeIds.has(edge.target)) {
				relevant.set(id, edge)
			}
		}

		return relevant
	}

	/**
	 * 限制边数量
	 */
	private limitEdges(
		edges: Map<string, EdgeAttribute>,
		maxEdges: number,
	): Map<string, EdgeAttribute> {
		if (edges.size <= maxEdges) {
			return edges
		}

		const limited = new Map<string, EdgeAttribute>()
		let count = 0

		for (const [id, edge] of edges) {
			if (count >= maxEdges) break
			limited.set(id, edge)
			count++
		}

		return limited
	}

	/**
	 * 渲染降级模式覆盖层
	 */
	private renderDegradedOverlay(
		ctx: CanvasRenderingContext2D,
		viewport: ViewportState,
		totalNodes: number,
		totalEdges: number,
	): void {
		ctx.fillStyle = "rgba(0, 0, 0, 0.7)"
		ctx.fillRect(10, 10, 250, 80)

		ctx.font = "14px sans-serif"
		ctx.fillStyle = "#fff"
		ctx.textAlign = "left"
		ctx.fillText(`降级模式: ${totalNodes} 节点`, 20, 35)
		ctx.fillText(`仅显示部分内容以保证性能`, 20, 55)
		ctx.fillText(`点击节点可查看详情`, 20, 75)
	}

	/**
	 * 获取当前渲染模式
	 */
	getCurrentMode(): RenderMode {
		return this.currentMode
	}

	/**
	 * 获取渲染统计
	 */
	getStats(): {
		mode: RenderMode
		nodeCount: number
		visibleNodeCount: number
	} {
		return {
			mode: this.currentMode,
			nodeCount: this.nodeCount,
			visibleNodeCount: this.visibleNodeCount,
		}
	}
}

export const smartRenderEngine = SmartRenderEngine.getInstance()
