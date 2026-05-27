import { memo, useRef, useEffect, useCallback, useMemo } from "react"
import { cn } from "@/lib/utils"
import type { MemoryNode, MemoryEdge, BFSAnimationState, BFSStep } from "./types"

interface BFSAnimationProps {
	state: BFSAnimationState
	nodes: MemoryNode[]
	edges: MemoryEdge[]
	onStepChange?: (stepIndex: number) => void
	onComplete?: () => void
	className?: string
}

const CANVAS_SIZE = { width: 400, height: 300 }

/**
 * BFS 扩散实况动画 (TSD 23.6.3)
 *
 * 使用 Canvas 2D 渲染记忆节点从种子节点开始的 BFS 扩散波纹动画。
 *
 * 动画流程:
 * 1. 种子节点激活 (t=0)
 * 2. 1-hop 扩散 (t=300ms) - 半径 50px, 透明度 0.8
 * 3. 2-hop 扩散 (t=600ms) - 半径 120px, 透明度 0.5
 * 4. 3-hop 扩散 (t=900ms) - 半径 200px, 透明度 0.2
 * 5. 稳定态 (t=1200ms) - 最终激活节点高亮, 边加粗
 */
const BFSAnimation = memo(function BFSAnimation({
	state,
	nodes,
	edges,
	onStepChange,
	onComplete,
	className,
}: BFSAnimationProps) {
	const canvasRef = useRef<HTMLCanvasElement>(null)
	const animFrameRef = useRef<number>(0)
	const startTimeRef = useRef<number>(0)
	const currentStepRef = useRef(0)

	const nodeMap = useMemo(() => {
		const map = new Map<string, MemoryNode>()
		nodes.forEach((n) => map.set(n.id, n))
		return map
	}, [nodes])

	const renderFrame = useCallback(
		(timestamp: number) => {
			const canvas = canvasRef.current
			if (!canvas) return
			const ctx = canvas.getContext("2d")
			if (!ctx) return

			if (!startTimeRef.current) {
				startTimeRef.current = timestamp
			}

			const elapsed = timestamp - startTimeRef.current
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

			// 计算当前步骤
			let stepIndex = 0
			for (let i = 0; i < state.steps.length; i++) {
				if (elapsed >= state.steps[i].timestamp_ms) {
					stepIndex = i + 1
				}
			}
			stepIndex = Math.min(stepIndex, state.steps.length)

			if (stepIndex !== currentStepRef.current) {
				currentStepRef.current = stepIndex
				onStepChange?.(stepIndex)
			}

			// 渲染边
			edges.forEach((edge) => {
				const source = nodeMap.get(edge.source)
				const target = nodeMap.get(edge.target)
				if (!source || !target) return

				const isActivated =
					stepIndex > 0 &&
					state.steps
						.slice(0, stepIndex)
						.some(
							(s) =>
								s.nodes.includes(edge.source) &&
								s.nodes.includes(edge.target),
						)

				if (isActivated) {
					ctx.strokeStyle = "rgba(251, 191, 36, 0.6)"
					ctx.lineWidth = 2
				} else {
					ctx.strokeStyle = "rgba(148, 163, 184, 0.08)"
					ctx.lineWidth = 0.5
				}
				ctx.beginPath()
				ctx.moveTo(source.x, source.y)
				ctx.lineTo(target.x, target.y)
				ctx.stroke()
			})

			// 获取所有已激活的节点
			const activatedNodes = new Set<string>()
			if (stepIndex > 0) {
				for (let i = 0; i < stepIndex; i++) {
					state.steps[i].nodes.forEach((n) => activatedNodes.add(n))
				}
			}

			// 渲染当前扩散波纹
			if (stepIndex > 0 && stepIndex <= state.steps.length) {
				const currentStepData = state.steps[stepIndex - 1]
				const hopRadius = [50, 120, 200, 280][
					Math.min(stepIndex - 1, 3)
				]
				const alpha = [0.8, 0.5, 0.2, 0.08][Math.min(stepIndex - 1, 3)]

				// 从种子节点画波纹
				const seedNode = nodeMap.get(state.seed_node)
				if (seedNode) {
					ctx.strokeStyle = `rgba(251, 191, 36, ${alpha})`
					ctx.lineWidth = 2
					ctx.setLineDash([4, 4])
					ctx.beginPath()
					ctx.arc(
						seedNode.x,
						seedNode.y,
						hopRadius,
						0,
						Math.PI * 2,
					)
					ctx.stroke()
					ctx.setLineDash([])

					// 波纹动画
					const animationProgress = Math.min(
						1,
						(elapsed - state.steps[stepIndex - 1].timestamp_ms) / 300,
					)
					if (animationProgress < 1) {
						const expandedRadius =
							hopRadius * (0.5 + animationProgress * 0.5)
						const expandedAlpha =
							alpha * (1 - animationProgress) * 0.5
						ctx.strokeStyle = `rgba(251, 191, 36, ${expandedAlpha})`
						ctx.lineWidth = 1
						ctx.beginPath()
						ctx.arc(
							seedNode.x,
							seedNode.y,
							expandedRadius,
							0,
							Math.PI * 2,
						)
						ctx.stroke()
					}
				}
			}

			// 渲染节点
			nodes.forEach((node) => {
				const isActivated = activatedNodes.has(node.id)
				const isSeed = node.id === state.seed_node
				const actValue =
					node.activation * (isActivated ? 1 : 0.2)

				// 发光
				if (actValue > 0.3) {
					const glow = ctx.createRadialGradient(
						node.x, node.y, 0,
						node.x, node.y, 16,
					)
					const color = isSeed ? "#f59e0b" : "#f97316"
					glow.addColorStop(0, `${color}50`)
					glow.addColorStop(1, "transparent")
					ctx.fillStyle = glow
					ctx.beginPath()
					ctx.arc(node.x, node.y, 16, 0, Math.PI * 2)
					ctx.fill()
				}

				// 节点圆
				const radius = isSeed ? 8 : isActivated ? 6 : 4
				ctx.globalAlpha = isActivated ? 1 : 0.3
				ctx.fillStyle = isSeed
					? "#f59e0b"
					: isActivated
						? "#f97316"
						: "#64748b"
				ctx.beginPath()
				ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
				ctx.fill()
				ctx.globalAlpha = 1

				// 标签
				if (isSeed || (isActivated && actValue > 0.6)) {
					ctx.fillStyle = "rgba(255,255,255,0.8)"
					ctx.font = "9px sans-serif"
					ctx.textAlign = "center"
					ctx.fillText(
						node.label.slice(0, 6),
						node.x,
						node.y + radius + 11,
					)
				}
			})

			// 检查是否完成
			if (stepIndex >= state.steps.length && elapsed > state.steps[state.steps.length - 1].timestamp_ms + 500) {
				onComplete?.()
				return
			}

			animFrameRef.current = requestAnimationFrame(renderFrame)
		},
		[nodes, edges, nodeMap, state, onStepChange, onComplete],
	)

	const handlePlay = useCallback(() => {
		startTimeRef.current = 0
		currentStepRef.current = 0
		animFrameRef.current = requestAnimationFrame(renderFrame)
	}, [renderFrame])

	const handleStop = useCallback(() => {
		if (animFrameRef.current) {
			cancelAnimationFrame(animFrameRef.current)
			animFrameRef.current = 0
		}
	}, [])

	useEffect(() => {
		if (state.is_playing) {
			handlePlay()
		}
		return handleStop
	}, [state.is_playing, handlePlay, handleStop])

	return (
		<div
			className={cn(
				"rounded-lg border border-purple-700/30 bg-slate-900/50 overflow-hidden",
				className,
			)}
		>
			{/* 标题 */}
			<div className="flex items-center justify-between p-2 border-b border-slate-700/30">
				<div className="flex items-center gap-2 text-xs">
					<div
						className={cn(
							"h-2 w-2 rounded-full",
							state.is_playing ? "bg-purple-500 animate-pulse" : "bg-slate-600",
						)}
					/>
					<span className="font-medium text-slate-300">
						BFS 扩散动画
					</span>
					<span className="text-slate-500">
						| 种子: {state.seed_node}
					</span>
				</div>
				<div className="flex items-center gap-1">
					{state.is_playing ? (
						<button
							type="button"
							onClick={handleStop}
							className="rounded px-2 py-0.5 text-xs bg-slate-700 text-slate-300 hover:bg-slate-600"
						>
							停止
						</button>
					) : (
						<button
							type="button"
							onClick={handlePlay}
							className="rounded px-2 py-0.5 text-xs bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
						>
							播放
						</button>
					)}
				</div>
			</div>

			{/* Canvas */}
			<canvas
				ref={canvasRef}
				className="w-full"
				style={{ aspectRatio: "4/3" }}
			/>

			{/* 步骤指示器 */}
			<div className="flex items-center gap-1 px-2 py-1.5 border-t border-slate-700/30 overflow-x-auto">
				{state.steps.map((step, i) => (
					<div
						key={i}
						className={cn(
							"flex items-center gap-1 rounded px-1.5 py-0.5 text-xs shrink-0",
							i < currentStepRef.current
								? "bg-purple-500/20 text-purple-400"
								: "text-slate-600",
						)}
					>
						<span>{step.hop}-hop</span>
						<span className="text-slate-500">
							({step.nodes.length}节点)
						</span>
					</div>
				))}
				<span className="ml-auto text-xs text-slate-600 tabular-nums">
					{currentStepRef.current}/{state.steps.length}
				</span>
			</div>
		</div>
	)
})

export default BFSAnimation
