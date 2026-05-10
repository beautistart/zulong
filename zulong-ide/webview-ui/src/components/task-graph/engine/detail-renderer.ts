import type { NodeAttribute } from "../types"

interface DetailItem {
	nodeId: string
	element: HTMLElement
}

export class DetailRenderer {
	private container: HTMLElement | null = null
	private items: Map<string, DetailItem> = new Map()
	private pendingQueue: string[] = []
	private isProcessing = false

	setContainer(container: HTMLElement): void {
		this.container = container
	}

	scheduleDetails(nodes: NodeAttribute[], visibleIds: Set<string>): void {
		this.pendingQueue = []
		for (const id of visibleIds) {
			if (!this.items.has(id)) {
				this.pendingQueue.push(id)
			}
		}

		const toRemove: string[] = []
		for (const [id] of this.items) {
			if (!visibleIds.has(id)) {
				toRemove.push(id)
			}
		}

		for (const id of toRemove) {
			this.removeItem(id)
		}

		if (!this.isProcessing && this.pendingQueue.length > 0) {
			this.processNext(nodes)
		}
	}

	private processNext(nodes: NodeAttribute[]): void {
		if (this.pendingQueue.length === 0 || !this.container) {
			this.isProcessing = false
			return
		}

		this.isProcessing = true
		const nodeId = this.pendingQueue.shift() as string
		const node = nodes.find((n) => n.id === nodeId)
		if (!node) {
			this.processNext(nodes)
			return
		}

		const scheduleNext = () => {
			if (typeof requestIdleCallback !== "undefined") {
				requestIdleCallback(() => this.processNext(nodes))
			} else {
				setTimeout(() => this.processNext(nodes), 16)
			}
		}

		const el = this.createDetailElement(node)
		if (el) {
			this.items.set(nodeId, { nodeId, element: el })
			this.container.appendChild(el)
		}
		scheduleNext()
	}

	private createDetailElement(node: NodeAttribute): HTMLElement | null {
		const el = document.createElement("div")
		el.className = "graph-node-detail"
		el.dataset.nodeId = node.id
		el.style.position = "absolute"
		el.style.left = `${node.position.x}px`
		el.style.top = `${node.position.y}px`
		el.style.pointerEvents = "auto"
		return el
	}

	private removeItem(id: string): void {
		const item = this.items.get(id)
		if (item?.element.parentNode) {
			item.element.parentNode.removeChild(item.element)
		}
		this.items.delete(id)
	}

	clear(): void {
		for (const [, item] of this.items) {
			if (item.element.parentNode) {
				item.element.parentNode.removeChild(item.element)
			}
		}
		this.items.clear()
		this.pendingQueue = []
		this.isProcessing = false
	}

	getItemCount(): number {
		return this.items.size
	}
}
