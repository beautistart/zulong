import { useCallback, useEffect, useRef, useState } from "react"
import { useGraphStore } from "../store/useGraphStore"
import { ViewportManager } from "../utils/viewport-manager"
import { LayoutWorkerClient } from "../utils/layout-worker-client"
import { SkeletonRenderer } from "../engine/skeleton-renderer"
import { InteractionController } from "../interaction/interaction-controller"
import { selectRenderMode, isPointLineMode } from "../engine/render-engine"
import type { NodeAttribute, ViewportState, Position } from "../types"

interface GraphCanvasProps {
	width?: number
	height?: number
}

export default function GraphCanvas({ width = 800, height = 600 }: GraphCanvasProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null)
	const overlayRef = useRef<HTMLDivElement>(null)
	const viewportManagerRef = useRef(new ViewportManager())
	const layoutClientRef = useRef(new LayoutWorkerClient())
	const interactionRef = useRef<InteractionController | null>(null)
	const animFrameRef = useRef<number>(0)
	const [retryCount, setRetryCount] = useState(0)

	// 🔥 优化3：批量更新防抖 - 避免节点逐个添加时频繁触发布局
	const layoutTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
	const pendingLayoutRef = useRef(false)

	const nodes = useGraphStore((s) => s.nodes)
	const edges = useGraphStore((s) => s.edges)
	const renderStatus = useGraphStore((s) => s.renderStatus)
	const renderMode = useGraphStore((s) => s.renderMode)
	const viewport = useGraphStore((s) => s.viewport)
	const _selectedNodeId = useGraphStore((s) => s.selectedNodeId)
	const errorMessage = useGraphStore((s) => s.errorMessage)
	const setRenderStatus = useGraphStore((s) => s.setRenderStatus)
	const setRenderMode = useGraphStore((s) => s.setRenderMode)
	const setViewport = useGraphStore((s) => s.setViewport)
	const setLayoutPositions = useGraphStore((s) => s.setLayoutPositions)
	const selectNode = useGraphStore((s) => s.selectNode)
	const setErrorMessage = useGraphStore((s) => s.setErrorMessage)
	const _setMetrics = useGraphStore((s) => s.setMetrics)

	const nodeArray = Array.from(nodes.values())
	const edgeArray = Array.from(edges.values())
	const nodeCount = nodes.size

	const performLayout = useCallback(async () => {
		if (nodeCount === 0) return

		const client = layoutClientRef.current
		const algorithm = nodeCount <= 200 ? "hierarchy" : "force"

		//  优化2：监听渐进式布局结果
		client.onPartial((positions) => {
			const posMap = new Map(Object.entries(positions))
			setLayoutPositions(posMap)
			// 提前进入渲染状态，让用户看到初步布局
			if (renderStatus === "layouting") {
				setRenderStatus("rendering")
			}
		})

		client.onRefined((positions) => {
			const posMap = new Map(Object.entries(positions))
			setLayoutPositions(posMap)
		})

		try {
			setRenderStatus("layouting")
			const layoutNodes = nodeArray.map((n) => ({ id: n.id, parentId: n.parentId }))
			const layoutEdges = edgeArray.map((e) => ({
				source: e.source,
				target: e.target,
				type: e.type,
			}))

			// 🔥 优化2：根据节点数动态调整超时时间
			const timeoutMs = nodeCount <= 500 ? 10000 : nodeCount <= 2000 ? 20000 : 30000

			const positions = await client.requestLayout(layoutNodes, layoutEdges, {
				algorithm,
				width: width * 2,
				height: height * 2,
			}, timeoutMs)

			const posMap = new Map<string, Position>()
			for (const [id, pos] of Object.entries(positions)) {
				posMap.set(id, pos)
			}
			setLayoutPositions(posMap)

			viewportManagerRef.current.buildIndex(nodes)

			const fitViewport = viewportManagerRef.current.fitAll(nodes, width, height)
			setViewport(fitViewport)

			setRenderStatus("rendering")
		} catch (err) {
			console.error("Layout failed:", err)
			setErrorMessage(`布局计算失败: ${err instanceof Error ? err.message : String(err)}`)
			setRenderStatus("error")
		}
	}, [nodeCount, nodeArray, edgeArray, nodes, width, height, setRenderStatus, setLayoutPositions, setViewport, setErrorMessage, renderStatus])

	// 🔥 优化3：调度布局（带防抖）
	const scheduleLayout = useCallback(() => {
		if (pendingLayoutRef.current) return

		// 清除之前的定时器
		if (layoutTimerRef.current) {
			clearTimeout(layoutTimerRef.current)
		}

		pendingLayoutRef.current = true

		// 防抖500ms，批量添加节点后只布局一次
		layoutTimerRef.current = setTimeout(() => {
			pendingLayoutRef.current = false
			if (nodeCount > 0 && renderStatus === "idle") {
				performLayout()
			}
		}, 500)
	}, [nodeCount, renderStatus, performLayout])

	useEffect(() => {
		const newMode = selectRenderMode(nodeCount)
		if (newMode !== renderMode) {
			setRenderMode(newMode)
		}
	}, [nodeCount, renderMode, setRenderMode])

	// 🔥 优化3：节点数量变化时调度布局（而非直接执行）
	useEffect(() => {
		scheduleLayout()
		return () => {
			if (layoutTimerRef.current) {
				clearTimeout(layoutTimerRef.current)
			}
		}
	}, [nodeCount, scheduleLayout])

	const renderCanvas = useCallback(() => {
		const canvas = canvasRef.current
		if (!canvas) return

		const ctx = canvas.getContext("2d")
		if (!ctx) return

		ctx.clearRect(0, 0, canvas.width, canvas.height)
		ctx.fillStyle = "#1e1e1e"
		ctx.fillRect(0, 0, canvas.width, canvas.height)

		if (nodeCount === 0) {
			ctx.fillStyle = "#888888"
			ctx.font = "14px sans-serif"
			ctx.textAlign = "center"
			ctx.fillText("暂无图谱数据", canvas.width / 2, canvas.height / 2)
			return
		}

		const currentViewport = useGraphStore.getState().viewport
		const renderer = new SkeletonRenderer(ctx, currentViewport)
		const currentSelectedId = useGraphStore.getState().selectedNodeId

		const positions = new Map<string, Position>()
		for (const node of nodes.values()) {
			positions.set(node.id, node.position)
		}

		// 🔥 优化1：边渲染视口裁剪 - 只渲染可见节点相关的边
		let visibleEdges = edgeArray
		if (renderMode === "virtual" || renderMode === "degraded") {
			const visibleIds = viewportManagerRef.current.getVisibleNodeIds(currentViewport)
			const visibleSet = new Set(visibleIds)
			visibleEdges = edgeArray.filter(
				(e) => visibleSet.has(e.source) && visibleSet.has(e.target)
			)
			
			// 🔥 降级模式额外优化：限制最大边数
			if (renderMode === "degraded" && visibleEdges.length > 500) {
				visibleEdges = visibleEdges.slice(0, 500)
			}
		}

		renderer.renderEdges(visibleEdges, positions)

		let visibleNodes: NodeAttribute[]
		if (renderMode === "virtual" || renderMode === "degraded") {
			const visibleIds = viewportManagerRef.current.getVisibleNodeIds(currentViewport)
			const visibleSet = new Set(visibleIds)
			visibleNodes = nodeArray.filter((n) => visibleSet.has(n.id))
			
			// 🔥 降级模式额外优化：限制最大节点数
			if (renderMode === "degraded" && visibleNodes.length > 200) {
				visibleNodes = visibleNodes.slice(0, 200)
			}
		} else {
			visibleNodes = nodeArray
		}

		renderer.renderNodes(visibleNodes, currentSelectedId)
		setRenderStatus("completed")
	}, [nodeCount, nodes, edgeArray, nodeArray, renderMode, setRenderStatus])

	useEffect(() => {
		if (renderStatus === "rendering" || renderStatus === "completed" || renderStatus === "layouting") {
			animFrameRef.current = requestAnimationFrame(renderCanvas)
		}
		return () => {
			if (animFrameRef.current) {
				cancelAnimationFrame(animFrameRef.current)
			}
		}
	}, [renderStatus, renderCanvas])

	useEffect(() => {
		const canvas = canvasRef.current
		if (!canvas) return

		const initialViewport: ViewportState = { offsetX: 0, offsetY: 0, zoom: 1, width, height }
		const interaction = new InteractionController(initialViewport)
		interactionRef.current = interaction

		interaction.on("viewport_changed", (vp: unknown) => {
			setViewport(vp as ViewportState)
		})

		interaction.on("node_selected", (data: unknown) => {
			const { nodeId } = data as { nodeId: string }
			selectNode(nodeId)
		})

		const handleWheel = (e: WheelEvent) => {
			const rect = canvas.getBoundingClientRect()
			interaction.handleWheel(e, rect)
		}

		const handleMouseDown = (e: MouseEvent) => {
			const rect = canvas.getBoundingClientRect()
			interaction.handleMouseDown(e, rect)
		}

		const handleMouseMove = (e: MouseEvent) => {
			const rect = canvas.getBoundingClientRect()
			interaction.handleMouseMove(e, rect)
		}

		const handleMouseUp = () => {
			interaction.handleMouseUp()
		}

		canvas.addEventListener("wheel", handleWheel, { passive: false })
		canvas.addEventListener("mousedown", handleMouseDown)
		window.addEventListener("mousemove", handleMouseMove)
		window.addEventListener("mouseup", handleMouseUp)

		return () => {
			canvas.removeEventListener("wheel", handleWheel)
			canvas.removeEventListener("mousedown", handleMouseDown)
			window.removeEventListener("mousemove", handleMouseMove)
			window.removeEventListener("mouseup", handleMouseUp)
			interaction.destroy()
		}
	}, [width, height, setViewport, selectNode])

	const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
		if (!interactionRef.current || !canvasRef.current) return
		const rect = canvasRef.current.getBoundingClientRect()
		const mouseX = e.clientX - rect.left
		const mouseY = e.clientY - rect.top
		const worldX = (mouseX - viewport.offsetX) / viewport.zoom
		const worldY = (mouseY - viewport.offsetY) / viewport.zoom

		const NODE_RADIUS = 20
		for (const node of nodes.values()) {
			const dx = node.position.x - worldX
			const dy = node.position.y - worldY
			if (dx * dx + dy * dy < NODE_RADIUS * NODE_RADIUS) {
				interactionRef.current.handleNodeClick(node.id)
				return
			}
		}
		selectNode(null)
	}

	const handleFitAll = () => {
		const fitViewport = viewportManagerRef.current.fitAll(nodes, width, height)
		setViewport(fitViewport)
	}

	const handleRetry = () => {
		if (retryCount >= 2) return
		setRetryCount((c) => c + 1)
		setRenderStatus("idle")
		setErrorMessage(null)
		performLayout()
	}

	const pointLine = isPointLineMode(viewport.zoom)

	return (
		<div style={{ position: "relative", width, height }}>
			<canvas
				ref={canvasRef}
				width={width}
				height={height}
				onClick={handleCanvasClick}
				style={{ display: "block", cursor: pointLine ? "crosshair" : "grab" }}
			/>
			<div ref={overlayRef} style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }} />
			{renderStatus === "layouting" && (
				<div
					style={{
						position: "absolute",
						top: "50%",
						left: "50%",
						transform: "translate(-50%, -50%)",
						color: "#e2ab00",
						fontSize: 14,
					}}
				>
					正在计算布局...
				</div>
			)}
			{renderStatus === "error" && (
				<div
					style={{
						position: "absolute",
						top: "50%",
						left: "50%",
						transform: "translate(-50%, -50%)",
						textAlign: "center",
						color: "#e51400",
					}}
				>
					<p>{errorMessage || "图谱渲染异常"}</p>
					{retryCount < 2 && (
						<button
							type="button"
							onClick={handleRetry}
							style={{
								marginTop: 8,
								padding: "4px 12px",
								background: "#2d2d2d",
								color: "#fff",
								border: "1px solid #888",
								cursor: "pointer",
							}}
						>
							重试
						</button>
					)}
				</div>
			)}
			<div style={{ position: "absolute", bottom: 8, left: 8, display: "flex", gap: 8 }}>
				<button
					type="button"
					onClick={handleFitAll}
					style={{
						padding: "2px 8px",
						background: "#2d2d2d",
						color: "#ccc",
						border: "1px solid #555",
						cursor: "pointer",
						fontSize: 12,
					}}
				>
					适应画布
				</button>
				<span style={{ color: "#888", fontSize: 12 }}>
					{Math.round(viewport.zoom * 100)}% | {nodeCount} 节点
				</span>
			</div>
		</div>
	)
}
