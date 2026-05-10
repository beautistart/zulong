import type { ViewportState, SearchResult, SelectionRect } from "../types"

type EventMap = Record<string, unknown>

type EventHandler<T> = (event: T) => void

export class InteractionController {
	private handlers = new Map<string, Set<EventHandler<unknown>>>()
	private viewport: ViewportState
	private isPanning = false
	private isSelecting = false
	private panStart = { x: 0, y: 0 }
	private selectionStart = { x: 0, y: 0 }
	private selectionRect: SelectionRect | null = null
	private searchResults: SearchResult | null = null

	private static readonly MIN_ZOOM = 0.1
	private static readonly MAX_ZOOM = 5.0
	private static readonly ZOOM_STEP = 0.1
	private static readonly ZOOM_STEP_FINE = 0.05

	constructor(viewport: ViewportState) {
		this.viewport = { ...viewport }
	}

	on<T extends EventMap>(event: string, handler: EventHandler<T[keyof T]>): void {
		if (!this.handlers.has(event)) {
			this.handlers.set(event, new Set())
		}
		this.handlers.get(event)?.add(handler as EventHandler<unknown>)
	}

	off<T extends EventMap>(event: string, handler: EventHandler<T[keyof T]>): void {
		this.handlers.get(event)?.delete(handler as EventHandler<unknown>)
	}

	emit(event: string, data: unknown): void {
		this.handlers.get(event)?.forEach((handler) => {
			try {
				handler(data)
			} catch (err) {
				console.error(`Event handler error [${event}]:`, err)
			}
		})
	}

	handleWheel(e: WheelEvent, canvasRect: DOMRect): ViewportState {
		e.preventDefault()
		const mouseX = e.clientX - canvasRect.left
		const mouseY = e.clientY - canvasRect.top

		const step = e.ctrlKey ? InteractionController.ZOOM_STEP_FINE : InteractionController.ZOOM_STEP
		const delta = e.deltaY > 0 ? -step : step
		const newZoom = Math.max(
			InteractionController.MIN_ZOOM,
			Math.min(InteractionController.MAX_ZOOM, this.viewport.zoom * (1 + delta))
		)

		const worldX = (mouseX - this.viewport.offsetX) / this.viewport.zoom
		const worldY = (mouseY - this.viewport.offsetY) / this.viewport.zoom

		this.viewport = {
			...this.viewport,
			zoom: newZoom,
			offsetX: mouseX - worldX * newZoom,
			offsetY: mouseY - worldY * newZoom,
		}

		this.emit("viewport_changed", this.viewport)
		this.emit("zoom_changed", { zoom: newZoom, source: "wheel" as const })
		return this.viewport
	}

	handleMouseDown(e: MouseEvent, canvasRect: DOMRect): void {
		const mouseX = e.clientX - canvasRect.left
		const mouseY = e.clientY - canvasRect.top

		if (e.shiftKey) {
			this.isSelecting = true
			this.selectionStart = { x: mouseX, y: mouseY }
			this.selectionRect = { x: mouseX, y: mouseY, width: 0, height: 0 }
			return
		}

		this.isPanning = true
		this.panStart = { x: mouseX - this.viewport.offsetX, y: mouseY - this.viewport.offsetY }
	}

	handleMouseMove(e: MouseEvent, canvasRect: DOMRect): ViewportState {
		const mouseX = e.clientX - canvasRect.left
		const mouseY = e.clientY - canvasRect.top

		if (this.isSelecting) {
			this.selectionRect = {
				x: Math.min(this.selectionStart.x, mouseX),
				y: Math.min(this.selectionStart.y, mouseY),
				width: Math.abs(mouseX - this.selectionStart.x),
				height: Math.abs(mouseY - this.selectionStart.y),
			}
			this.emit("selection_changed", this.selectionRect)
			return this.viewport
		}

		if (this.isPanning) {
			this.viewport = {
				...this.viewport,
				offsetX: mouseX - this.panStart.x,
				offsetY: mouseY - this.panStart.y,
			}
			this.emit("viewport_changed", this.viewport)
			return this.viewport
		}

		const worldX = (mouseX - this.viewport.offsetX) / this.viewport.zoom
		const worldY = (mouseY - this.viewport.offsetY) / this.viewport.zoom
		this.emit("hover_changed", { worldX, worldY, screenX: mouseX, screenY: mouseY })
		return this.viewport
	}

	handleMouseUp(): void {
		if (this.isSelecting && this.selectionRect) {
			this.emit("selection_complete", this.selectionRect)
		}
		this.isPanning = false
		this.isSelecting = false
		this.selectionRect = null
	}

	handleNodeClick(nodeId: string): void {
		this.emit("node_selected", { nodeId, source: "canvas" })
	}

	handleSearch(keyword: string, nodes: Array<{ id: string; label: string; desc: string }>): SearchResult {
		if (!keyword.trim()) {
			this.searchResults = { nodeIds: [], currentIndex: 0, total: 0 }
			this.emit("search_result", this.searchResults)
			return this.searchResults
		}

		const lower = keyword.toLowerCase()
		const matched = nodes
			.filter((n) => n.label.toLowerCase().includes(lower) || n.desc.toLowerCase().includes(lower))
			.map((n) => n.id)

		this.searchResults = { nodeIds: matched, currentIndex: 0, total: matched.length }
		this.emit("search_result", this.searchResults)
		return this.searchResults
	}

	handleSearchNext(): string | null {
		if (!this.searchResults || this.searchResults.total === 0) return null
		this.searchResults.currentIndex = (this.searchResults.currentIndex + 1) % this.searchResults.total
		this.emit("search_result", this.searchResults)
		return this.searchResults.nodeIds[this.searchResults.currentIndex]
	}

	getViewport(): ViewportState {
		return { ...this.viewport }
	}

	setViewport(viewport: ViewportState): void {
		this.viewport = { ...viewport }
		this.emit("viewport_changed", this.viewport)
	}

	getSelectionRect(): SelectionRect | null {
		return this.selectionRect
	}

	destroy(): void {
		this.handlers.clear()
	}
}
