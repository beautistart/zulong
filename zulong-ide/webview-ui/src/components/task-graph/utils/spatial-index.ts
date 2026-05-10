import RBush from "rbush"
import type { NodeAttribute, ViewportState } from "../types"

interface NodeBBox {
	minX: number
	minY: number
	maxX: number
	maxY: number
	nodeId: string
}

const NODE_SIZE = 60

export class SpatialIndex {
	private rtree: RBush<NodeBBox>
	private nodeSize: number

	constructor(nodeSize = NODE_SIZE) {
		this.rtree = new RBush()
		this.nodeSize = nodeSize
	}

	buildIndex(nodes: Map<string, NodeAttribute>): void {
		this.rtree.clear()
		const items: NodeBBox[] = []
		const half = this.nodeSize / 2
		for (const [id, node] of nodes) {
			items.push({
				minX: node.position.x - half,
				minY: node.position.y - half,
				maxX: node.position.x + half,
				maxY: node.position.y + half,
				nodeId: id,
			})
		}
		this.rtree.load(items)
	}

	queryVisible(viewport: ViewportState, bufferRatio = 0.2): string[] {
		const bufferW = viewport.width * bufferRatio
		const bufferH = viewport.height * bufferRatio
		const invZoom = 1 / viewport.zoom

		const worldLeft = (0 - viewport.offsetX - bufferW) * invZoom
		const worldRight = (viewport.width - viewport.offsetX + bufferW) * invZoom
		const worldTop = (0 - viewport.offsetY - bufferH) * invZoom
		const worldBottom = (viewport.height - viewport.offsetY + bufferH) * invZoom

		const results = this.rtree.search({
			minX: worldLeft,
			minY: worldTop,
			maxX: worldRight,
			maxY: worldBottom,
		})

		return results.map((r) => r.nodeId)
	}

	clear(): void {
		this.rtree.clear()
	}
}
