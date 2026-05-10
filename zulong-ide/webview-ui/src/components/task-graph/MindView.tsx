import { useCallback, useEffect, useRef } from "react"
import { useGraphStore } from "../store/useGraphStore"
import { NODE_STATUS_COLORS } from "../types"

interface MindViewProps {
	width?: number
	height?: number
}

export default function MindView({ width = 200, height = 150 }: MindViewProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null)
	const nodes = useGraphStore((s) => s.nodes)
	const edges = useGraphStore((s) => s.edges)
	const viewport = useGraphStore((s) => s.viewport)
	const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
	const selectNode = useGraphStore((s) => s.selectNode)

	const render = useCallback(() => {
		const canvas = canvasRef.current
		if (!canvas) return
		const ctx = canvas.getContext("2d")
		if (!ctx) return

		ctx.clearRect(0, 0, canvas.width, canvas.height)
		ctx.fillStyle = "#252525"
		ctx.fillRect(0, 0, canvas.width, canvas.height)

		if (nodes.size === 0) {
			ctx.fillStyle = "#888"
			ctx.font = "11px sans-serif"
			ctx.textAlign = "center"
			ctx.fillText("无数据", canvas.width / 2, canvas.height / 2)
			return
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

		const padding = 10
		const totalW = maxX - minX || 1
		const totalH = maxY - minY || 1
		const scaleX = (canvas.width - padding * 2) / totalW
		const scaleY = (canvas.height - padding * 2) / totalH
		const scale = Math.min(scaleX, scaleY)
		const offsetX = padding + (canvas.width - padding * 2 - totalW * scale) / 2 - minX * scale
		const offsetY = padding + (canvas.height - padding * 2 - totalH * scale) / 2 - minY * scale

		for (const edge of edges.values()) {
			const sourceNode = nodes.get(edge.source)
			const targetNode = nodes.get(edge.target)
			if (!sourceNode || !targetNode) continue

			const sx = sourceNode.position.x * scale + offsetX
			const sy = sourceNode.position.y * scale + offsetY
			const tx = targetNode.position.x * scale + offsetX
			const ty = targetNode.position.y * scale + offsetY

			ctx.strokeStyle = "#555"
			ctx.lineWidth = 0.5
			ctx.beginPath()
			ctx.moveTo(sx, sy)
			ctx.lineTo(tx, ty)
			ctx.stroke()
		}

		for (const node of nodes.values()) {
			const nx = node.position.x * scale + offsetX
			const ny = node.position.y * scale + offsetY

			ctx.fillStyle = NODE_STATUS_COLORS[node.status]
			if (node.id === selectedNodeId) {
				ctx.fillRect(nx - 2.5, ny - 2.5, 5, 5)
			} else {
				ctx.fillRect(nx - 1.5, ny - 1.5, 3, 3)
			}
		}

		const vpLeft = (0 - viewport.offsetX) / viewport.zoom
		const vpTop = (0 - viewport.offsetY) / viewport.zoom
		const vpRight = viewport.width / viewport.zoom + vpLeft
		const vpBottom = viewport.height / viewport.zoom + vpTop

		const frameX = vpLeft * scale + offsetX
		const frameY = vpTop * scale + offsetY
		const frameW = (vpRight - vpLeft) * scale
		const frameH = (vpBottom - vpTop) * scale

		ctx.strokeStyle = "#4488ff"
		ctx.lineWidth = 1
		ctx.setLineDash([3, 2])
		ctx.strokeRect(frameX, frameY, frameW, frameH)
		ctx.setLineDash([])
	}, [nodes, edges, viewport, selectedNodeId])

	useEffect(() => {
		const raf = requestAnimationFrame(render)
		return () => cancelAnimationFrame(raf)
	}, [render])

	const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
		const canvas = canvasRef.current
		if (!canvas) return

		const rect = canvas.getBoundingClientRect()
		const mouseX = e.clientX - rect.left
		const mouseY = e.clientY - rect.top

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

		const padding = 10
		const totalW = maxX - minX || 1
		const totalH = maxY - minY || 1
		const scaleX = (canvas.width - padding * 2) / totalW
		const scaleY = (canvas.height - padding * 2) / totalH
		const scale = Math.min(scaleX, scaleY)
		const offsetX = padding + (canvas.width - padding * 2 - totalW * scale) / 2 - minX * scale
		const offsetY = padding + (canvas.height - padding * 2 - totalH * scale) / 2 - minY * scale

		let closestId: string | null = null
		let closestDist = 8

		for (const node of nodes.values()) {
			const nx = node.position.x * scale + offsetX
			const ny = node.position.y * scale + offsetY
			const dist = Math.sqrt((mouseX - nx) ** 2 + (mouseY - ny) ** 2)
			if (dist < closestDist) {
				closestDist = dist
				closestId = node.id
			}
		}

		selectNode(closestId)
	}

	return (
		<div style={{ border: "1px solid #444", borderRadius: 4, overflow: "hidden" }}>
			<div
				style={{
					background: "#333",
					padding: "2px 6px",
					fontSize: 11,
					color: "#aaa",
					display: "flex",
					justifyContent: "space-between",
				}}
			>
				<span>思维视图</span>
				<span>({nodes.size})</span>
			</div>
			<canvas
				ref={canvasRef}
				width={width}
				height={height}
				onClick={handleClick}
				style={{ display: "block", cursor: "pointer" }}
			/>
		</div>
	)
}
