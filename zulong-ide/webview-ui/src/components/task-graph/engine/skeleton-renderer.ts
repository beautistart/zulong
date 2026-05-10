import type { NodeAttribute, EdgeAttribute, ViewportState, Position } from "../types"
import { NODE_STATUS_COLORS, EDGE_TYPE_COLORS } from "../types"

const NODE_RADIUS = 20
const LABEL_FONT_SIZE = 12
const MIN_ZOOM_FOR_LABEL = 0.2

export class SkeletonRenderer {
	private _ctx: CanvasRenderingContext2D
	private _viewport: ViewportState

	constructor(ctx: CanvasRenderingContext2D, viewport: ViewportState) {
		this._ctx = ctx
		this._viewport = viewport
	}

	renderEdges(edges: EdgeAttribute[], nodePositions: Map<string, Position>): void {
		const ctx = this._ctx
		const viewport = this._viewport
		ctx.lineWidth = 1

		// 🔥 性能优化：根据缩放级别简化边渲染
		const zoom = viewport.zoom
		const useSimpleLine = zoom < 0.3  // 小缩放时使用简单直线

		for (const edge of edges) {
			const sourcePos = nodePositions.get(edge.source)
			const targetPos = nodePositions.get(edge.target)
			if (!sourcePos || !targetPos) continue

			const sx = sourcePos.x * viewport.zoom + viewport.offsetX
			const sy = sourcePos.y * viewport.zoom + viewport.offsetY
			const tx = targetPos.x * viewport.zoom + viewport.offsetX
			const ty = targetPos.y * viewport.zoom + viewport.offsetY

			// 🔥 视口裁剪：跳过屏幕外的边
			const margin = 100
			if (sx < -margin && tx < -margin) continue
			if (sx > viewport.width + margin && tx > viewport.width + margin) continue
			if (sy < -margin && ty < -margin) continue
			if (sy > viewport.height + margin && ty > viewport.height + margin) continue

			ctx.strokeStyle = EDGE_TYPE_COLORS[edge.type] || "#888888"
			ctx.beginPath()
			ctx.moveTo(sx, sy)
			ctx.lineTo(tx, ty)
			ctx.stroke()
		}
	}

	renderNodes(nodes: NodeAttribute[], selectedId: string | null): void {
		const ctx = this._ctx
		const viewport = this._viewport
		const zoom = viewport.zoom

		// 🔥 优化6：批量绘制 - 按状态分组，减少context切换
		const nodesByStatus = new Map<string, NodeAttribute[]>()
		for (const node of nodes) {
			if (!nodesByStatus.has(node.status)) {
				nodesByStatus.set(node.status, [])
			}
			nodesByStatus.get(node.status)!.push(node)
		}

		// 批量绘制每种状态的节点
		for (const [, statusNodes] of nodesByStatus) {
			const firstNode = statusNodes[0]
			ctx.fillStyle = NODE_STATUS_COLORS[firstNode.status]
			ctx.beginPath()

			for (const node of statusNodes) {
				const sx = node.position.x * zoom + viewport.offsetX
				const sy = node.position.y * zoom + viewport.offsetY

				if (zoom <= 0.1) {
					// 超远视角：使用矩形（更快）
					ctx.moveTo(sx + 1.5, sy - 1.5)
					ctx.rect(sx - 1.5, sy - 1.5, 3, 3)
				} else {
					// 正常视角：使用圆形
					const radius = Math.max(NODE_RADIUS * zoom, 4)
					ctx.moveTo(sx + radius, sy)
					ctx.arc(sx, sy, radius, 0, Math.PI * 2)
				}
			}

			ctx.fill() // 一次性绘制所有相同颜色的节点
		}

		// 单独绘制选中节点（高亮）
		if (selectedId) {
			const selectedNode = nodes.find((n) => n.id === selectedId)
			if (selectedNode && zoom > 0.1) {
				const sx = selectedNode.position.x * zoom + viewport.offsetX
				const sy = selectedNode.position.y * zoom + viewport.offsetY
				const radius = Math.max(NODE_RADIUS * zoom, 4)

				ctx.strokeStyle = "#ffffff"
				ctx.lineWidth = 2
				ctx.beginPath()
				ctx.arc(sx, sy, radius, 0, Math.PI * 2)
				ctx.stroke()
			}
		}

		// 绘制标签
		if (zoom >= MIN_ZOOM_FOR_LABEL) {
			const fontSize = Math.max(LABEL_FONT_SIZE * zoom, 8)
			ctx.font = `${fontSize}px sans-serif`
			ctx.fillStyle = "#ffffff"
			ctx.textAlign = "center"
			ctx.textBaseline = "top"

			for (const node of nodes) {
				const sx = node.position.x * zoom + viewport.offsetX
				const sy = node.position.y * zoom + viewport.offsetY
				const radius = Math.max(NODE_RADIUS * zoom, 4)
				const label = node.label.length > 12 ? `${node.label.slice(0, 12)}…` : node.label
				ctx.fillText(label, sx, sy + radius + 4)
			}
		}
	}

	renderViewportFrame(
		mainViewport: ViewportState,
		totalBounds: { minX: number; minY: number; maxX: number; maxY: number },
		canvasWidth: number,
		canvasHeight: number
	): void {
		const ctx = this._ctx
		const { minX, minY, maxX, maxY } = totalBounds
		const totalW = maxX - minX || 1
		const totalH = maxY - minY || 1
		const scale = Math.min(canvasWidth / totalW, canvasHeight / totalH) * 0.9

		const frameX = ((0 - mainViewport.offsetX) / mainViewport.zoom - minX) * scale + canvasWidth * 0.05
		const frameY = ((0 - mainViewport.offsetY) / mainViewport.zoom - minY) * scale + canvasHeight * 0.05
		const frameW = (mainViewport.width / mainViewport.zoom) * scale
		const frameH = (mainViewport.height / mainViewport.zoom) * scale

		ctx.strokeStyle = "#4488ff"
		ctx.lineWidth = 1.5
		ctx.setLineDash([4, 2])
		ctx.strokeRect(frameX, frameY, frameW, frameH)
		ctx.setLineDash([])
	}
}
