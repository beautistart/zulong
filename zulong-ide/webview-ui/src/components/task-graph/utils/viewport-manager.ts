import type { NodeAttribute, ViewportState, RenderConfig, SelectionRect } from "../types"
import { DEFAULT_RENDER_CONFIG } from "../types"
import { SpatialIndex } from "./spatial-index"

export class ViewportManager {
	private spatialIndex: SpatialIndex
	private config: RenderConfig
	private throttleTimer: ReturnType<typeof setTimeout> | null = null
	private lastViewport: ViewportState | null = null
	private static readonly THROTTLE_MS = 16

	constructor(config?: Partial<RenderConfig>) {
		this.config = { ...DEFAULT_RENDER_CONFIG, ...config }
		this.spatialIndex = new SpatialIndex()
	}

	init(config?: Partial<RenderConfig>): void {
		if (config) {
			this.config = { ...DEFAULT_RENDER_CONFIG, ...config }
		}
		this.spatialIndex = new SpatialIndex()
	}

	buildIndex(nodes: Map<string, NodeAttribute>): void {
		this.spatialIndex.buildIndex(nodes)
	}

	getVisibleNodeIds(viewport: ViewportState): string[] {
		return this.spatialIndex.queryVisible(viewport, this.config.viewportBufferRatio)
	}

	onViewportChange(viewport: ViewportState, callback: (visibleIds: string[]) => void): void {
		if (this.throttleTimer) {
			clearTimeout(this.throttleTimer)
		}
		this.throttleTimer = setTimeout(() => {
			const visibleIds = this.getVisibleNodeIds(viewport)
			callback(visibleIds)
			this.lastViewport = { ...viewport }
		}, ViewportManager.THROTTLE_MS)
	}

	fitAll(nodes: Map<string, NodeAttribute>, canvasWidth: number, canvasHeight: number): ViewportState {
		if (nodes.size === 0) {
			return { offsetX: 0, offsetY: 0, zoom: 1, width: canvasWidth, height: canvasHeight }
		}

		let minX = Number.POSITIVE_INFINITY
		let minY = Number.POSITIVE_INFINITY
		let maxX = Number.NEGATIVE_INFINITY
		let maxY = Number.NEGATIVE_INFINITY

		for (const node of nodes.values()) {
			if (node.position.x < minX) minX = node.position.x
			if (node.position.y < minY) minY = node.position.y
			if (node.position.x > maxX) maxX = node.position.x
			if (node.position.y > maxY) maxY = node.position.y
		}

		const padding = 80
		const contentWidth = maxX - minX + padding * 2
		const contentHeight = maxY - minY + padding * 2

		const zoomX = canvasWidth / contentWidth
		const zoomY = canvasHeight / contentHeight
		const zoom = Math.min(zoomX, zoomY, 2.0)

		const centerX = (minX + maxX) / 2
		const centerY = (minY + maxY) / 2

		return {
			offsetX: canvasWidth / 2 - centerX * zoom,
			offsetY: canvasHeight / 2 - centerY * zoom,
			zoom,
			width: canvasWidth,
			height: canvasHeight,
		}
	}

	fitSelection(rect: SelectionRect, canvasWidth: number, canvasHeight: number): ViewportState {
		const padding = 40
		const contentWidth = rect.width + padding * 2
		const contentHeight = rect.height + padding * 2

		const zoomX = canvasWidth / contentWidth
		const zoomY = canvasHeight / contentHeight
		const zoom = Math.min(zoomX, zoomY, 5.0)

		const centerX = rect.x + rect.width / 2
		const centerY = rect.y + rect.height / 2

		return {
			offsetX: canvasWidth / 2 - centerX * zoom,
			offsetY: canvasHeight / 2 - centerY * zoom,
			zoom,
			width: canvasWidth,
			height: canvasHeight,
		}
	}

	getConfig(): RenderConfig {
		return { ...this.config }
	}

	clear(): void {
		this.spatialIndex.clear()
		this.lastViewport = null
		if (this.throttleTimer) {
			clearTimeout(this.throttleTimer)
			this.throttleTimer = null
		}
	}
}
