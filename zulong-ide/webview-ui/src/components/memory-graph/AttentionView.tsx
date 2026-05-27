import { memo, useMemo, useCallback, useRef, useEffect } from "react"
import {
	GlobeIcon,
	LinkIcon,
	FocusIcon,
	ZoomInIcon,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { AttentionState, AttentionMode, MemoryNode, MemoryEdge } from "./types"

interface AttentionViewProps {
	state: AttentionState
	nodes: MemoryNode[]
	edges: MemoryEdge[]
	onModeChange?: (mode: AttentionMode) => void
	onNodeClick?: (nodeId: string) => void
	className?: string
}

const MODE_CONFIG: Record<
	AttentionMode,
	{ icon: typeof GlobeIcon; label: string; desc: string }
> = {
	GLOBAL: {
		icon: GlobeIcon,
		label: "全局注意",
		desc: "关注完整上下文，所有节点淡蓝 + 焦点区域暖色",
	},
	SINGLE_CHAIN: {
		icon: LinkIcon,
		label: "单链注意",
		desc: "聚焦单条推理链，链上节点高亮，其余灰显",
	},
	FOCUS: {
		icon: ZoomInIcon,
		label: "局部注意",
		desc: "聚焦局部关键信息，放大显示邻域",
	},
}

const CANVAS_SIZE = { width: 400, height: 300 }

/**
 * 注意力三态可视化 (TSD 23.6.2)
 *
 * 使用 Canvas 2D 渲染记忆图谱的三种注意力状态:
 * - GLOBAL: 全景图 + 所有节点
 * - SINGLE_CHAIN: 单链高亮
 * - FOCUS: 局部放大
 */
const AttentionView = memo(function AttentionView({
	state,
	nodes,
	edges,
	onModeChange,
	onNodeClick,
	className,
}: AttentionViewProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null)

	const { activeSet, chainSet, focalNode, focalNeighbors } = useMemo(() => {
		const activeSet = new Set(state.active_nodes)
		const chainSet = new Set(state.chain_nodes)
		const focusNodeId = state.mode === "FOCUS" ? state.focal_node : state.focus_node
		const focalNeighbors = new Set(
			state.mode === "FOCUS" ? state.neighborhood.nodes : [],
		)
		return { activeSet, chainSet, focalNode: focusNodeId, focalNeighbors }
	}, [state])

	// 计算每个节点的颜色和大小
	const nodeRenderData = useMemo(() => {
		return nodes.map((node) => {
			const isFocus = node.id === focalNode
			const isChain = chainSet.has(node.id)
			const isActive = activeSet.has(node.id)
			const isNeighbor = focalNeighbors.has(node.id)
			const activation = state.activation_values[node.id] ?? node.activation

			let color: string
			let radius: number
			let opacity: number

			switch (state.mode) {
				case "GLOBAL":
					if (isFocus) {
						color = "#f59e0b" // gold
						radius = 16
						opacity = 1
					} else if (isActive) {
						color = "#f97316" // warm orange
						radius = 12
						opacity = 0.9
					} else {
						color = "#60a5fa" // light blue
						radius = 8
						opacity = 0.6
					}
					break
				case "SINGLE_CHAIN":
					if (isFocus) {
						color = "#f59e0b" // gold (current)
						radius = 16
						opacity = 1
					} else if (isChain) {
						color = "#f97316" // orange (chain)
						radius = 12
						opacity = 0.9
					} else {
						color = "#9ca3af" // grey
						radius = 6
						opacity = 0.3
					}
					break
				case "FOCUS":
					if (isFocus) {
						color = "#f59e0b" // gold
						radius = 20
						opacity = 1
					} else if (isNeighbor) {
						color = "#f97316" // orange
						radius = 10
						opacity = 0.8
					} else {
						color = "#94a3b8" // dim grey
						radius = 5
						opacity = 0.15
					}
					break
				default:
					color = "#60a5fa"
					radius = 8
					opacity = 0.6
			}

			return { ...node, color, radius, opacity, activation }
		})
	}, [nodes, state.mode, focalNode, chainSet, activeSet, focalNeighbors, state.activation_values])

	// Canvas 绘制
	useEffect(() => {
		const canvas = canvasRef.current
		if (!canvas) return
		const ctx = canvas.getContext("2d")
		if (!ctx) return

		const dpr = window.devicePixelRatio || 1
		canvas.width = CANVAS_SIZE.width * dpr
		canvas.height = CANVAS_SIZE.height * dpr
		canvas.style.width = `${CANVAS_SIZE.width}px`
		canvas.style.height = `${CANVAS_SIZE.height}px`
		ctx.scale(dpr, dpr)

		// 清空
		ctx.clearRect(0, 0, CANVAS_SIZE.width, CANVAS_SIZE.height)

		// 背景
		ctx.fillStyle = "rgba(15, 23, 42, 0.5)"
		ctx.fillRect(0, 0, CANVAS_SIZE.width, CANVAS_SIZE.height)

		// 网格
		ctx.strokeStyle = "rgba(255,255,255,0.03)"
		ctx.lineWidth = 0.5
		for (let x = 0; x < CANVAS_SIZE.width; x += 30) {
			ctx.beginPath()
			ctx.moveTo(x, 0)
			ctx.lineTo(x, CANVAS_SIZE.height)
			ctx.stroke()
		}
		for (let y = 0; y < CANVAS_SIZE.height; y += 30) {
			ctx.beginPath()
			ctx.moveTo(0, y)
			ctx.lineTo(CANVAS_SIZE.width, y)
			ctx.stroke()
		}

		// 绘制边
		const nodeMap = new Map(nodeRenderData.map((n) => [n.id, n]))
		edges.forEach((edge) => {
			const source = nodeMap.get(edge.source)
			const target = nodeMap.get(edge.target)
			if (!source || !target) return

			const isEdgeInChain =
				state.mode === "SINGLE_CHAIN" &&
				chainSet.has(edge.source) &&
				chainSet.has(edge.target)

			const isEdgeInFocus =
				state.mode === "FOCUS" &&
				(focalNode === edge.source ||
					focalNode === edge.target ||
					(focalNeighbors.has(edge.source) && focalNeighbors.has(edge.target)))

			let alpha: number
			let lineWidth: number
			if (isEdgeInChain || isEdgeInFocus) {
				alpha = 0.6
				lineWidth = 2
			} else if (state.mode === "SINGLE_CHAIN" || state.mode === "FOCUS") {
				alpha = 0.08
				lineWidth = 0.5
			} else {
				alpha = 0.15
				lineWidth = 1
			}

			ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`
			ctx.lineWidth = lineWidth
			ctx.beginPath()
			ctx.moveTo(source.x, source.y)
			ctx.lineTo(target.x, target.y)
			ctx.stroke()
		})

		// 绘制节点
		nodeRenderData.forEach((node) => {
			ctx.globalAlpha = node.opacity

			// 发光效果
			if (node.activation > 0.5) {
				const glow = ctx.createRadialGradient(
					node.x, node.y, node.radius * 0.5,
					node.x, node.y, node.radius * 2,
				)
				glow.addColorStop(0, `${node.color}40`)
				glow.addColorStop(1, "transparent")
				ctx.fillStyle = glow
				ctx.beginPath()
				ctx.arc(node.x, node.y, node.radius * 2, 0, Math.PI * 2)
				ctx.fill()
			}

			// 节点圆形
			ctx.fillStyle = node.color
			ctx.beginPath()
			ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
			ctx.fill()

			// 边框
			if (node.activation > 0.7) {
				ctx.strokeStyle = "rgba(255,255,255,0.5)"
				ctx.lineWidth = 1.5
				ctx.stroke()
			}

			// 标签
			if (node.radius >= 10) {
				ctx.fillStyle = "rgba(255,255,255,0.9)"
				ctx.font = `${Math.max(9, node.radius * 0.7)}px sans-serif`
				ctx.textAlign = "center"
				ctx.textBaseline = "middle"
				const label =
					node.label.length > 6
						? node.label.slice(0, 5) + ".."
						: node.label
				ctx.fillText(label, node.x, node.y + node.radius + 10)
			}

			ctx.globalAlpha = 1
		})
	}, [nodeRenderData, edges, state.mode, focalNode, chainSet, focalNeighbors])

	const handleCanvasClick = useCallback(
		(e: React.MouseEvent<HTMLCanvasElement>) => {
			if (!onNodeClick) return
			const rect = e.currentTarget.getBoundingClientRect()
			const x = e.clientX - rect.left
			const y = e.clientY - rect.top

			// 查找最近的节点
			let closest: string | null = null
			let minDist = 20
			for (const node of nodeRenderData) {
				const dx = node.x - x
				const dy = node.y - y
				const dist = Math.sqrt(dx * dx + dy * dy)
				if (dist < minDist && dist < node.radius + 10) {
					closest = node.id
					minDist = dist
				}
			}
			if (closest) {
				onNodeClick(closest)
			}
		},
		[nodeRenderData, onNodeClick],
	)

	const ConfigIcon = MODE_CONFIG[state.mode].icon

	return (
		<div
			className={cn(
				"rounded-lg border border-slate-700/50 bg-slate-900/50 overflow-hidden",
				className,
			)}
		>
			{/* 模式切换栏 */}
			<div className="flex items-center gap-1 p-2 border-b border-slate-700/30">
				<ConfigIcon className="h-4 w-4 text-slate-400" />
				<span className="text-xs font-medium text-slate-300 mr-2">
					注意力模式:
				</span>
				{(["GLOBAL", "SINGLE_CHAIN", "FOCUS"] as AttentionMode[]).map(
					(mode) => {
						const cfg = MODE_CONFIG[mode]
						const Icon = cfg.icon
						const isActive = state.mode === mode
						return (
							<button
								key={mode}
								type="button"
								onClick={() => onModeChange?.(mode)}
								className={cn(
									"inline-flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors",
									isActive
										? "bg-blue-500/20 text-blue-400"
										: "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
								)}
								title={cfg.desc}
							>
								<Icon className="h-3 w-3" />
								{cfg.label}
							</button>
						)
					},
				)}
			</div>

			{/* Canvas 区域 */}
			<canvas
				ref={canvasRef}
				onClick={handleCanvasClick}
				className="w-full cursor-pointer"
				style={{
					aspectRatio: "4/3",
				}}
			/>

			{/* 图例 */}
			<div className="flex items-center gap-3 px-2 py-1.5 border-t border-slate-700/30 text-xs text-slate-500">
				<span className="flex items-center gap-1">
					<span
						className="inline-block w-2.5 h-2.5 rounded-full"
						style={{ backgroundColor: "#f59e0b" }}
					/>
					当前焦点
				</span>
				<span className="flex items-center gap-1">
					<span
						className="inline-block w-2.5 h-2.5 rounded-full"
						style={{ backgroundColor: "#f97316" }}
					/>
					激活中
				</span>
				<span className="flex items-center gap-1">
					<span
						className="inline-block w-2.5 h-2.5 rounded-full"
						style={{ backgroundColor: "#60a5fa" }}
					/>
					未激活
				</span>
				<span className="ml-auto tabular-nums">
					{nodeRenderData.length} 节点 / {edges.length} 边
				</span>
			</div>
		</div>
	)
})

export default AttentionView
