import type { Position } from "../types"
import { hierarchyLayout } from "../utils/hierarchy-layout"

interface LayoutNode {
	id: string
	parentId: string | null
}

interface LayoutEdge {
	source: string
	target: string
	type: string
}

interface LayoutConfig {
	algorithm: "hierarchy" | "force" | "force_incremental"
	width: number
	height: number
}

interface SimNode {
	id: string
	x: number
	y: number
	index: number
	vx: number
	vy: number
	fx?: number
	fy?: number
}

// 🔥 修复TypeScript类型错误：使用any类型
let forceSimulation: any = null

self.onmessage = (e: MessageEvent) => {
	const msg = e.data

	if (msg.type === "layout_request") {
		const { nodes, edges, config } = msg as {
			type: string
			nodes: LayoutNode[]
			edges: LayoutEdge[]
			config: LayoutConfig
		}

		try {
			const startTime = performance.now()
			const timeoutMs = 10000

			if (config.algorithm === "hierarchy") {
				const positions = hierarchyLayout(nodes, config)
				const durationMs = performance.now() - startTime
				self.postMessage({ type: "layout_result", positions, durationMs })
				return
			}

			if (config.algorithm === "force" || config.algorithm === "force_incremental") {
				void computeForceLayout(nodes, edges, config, startTime, timeoutMs)
			}
		} catch (err) {
			self.postMessage({
				type: "layout_error",
				error: err instanceof Error ? err.message : String(err),
			})
		}
	}

	if (msg.type === "layout_stop") {
		if (forceSimulation) {
			forceSimulation.stop()
			forceSimulation = null
		}
	}
}

async function computeForceLayout(
	nodes: LayoutNode[],
	edges: LayoutEdge[],
	config: LayoutConfig,
	startTime: number,
	timeoutMs: number
): Promise<void> {
	const d3Force = await import("d3-force")

	const simNodes = nodes.map((n, i) => ({
		id: n.id,
		x: config.width / 2 + (Math.random() - 0.5) * 200,
		y: config.height / 2 + (Math.random() - 0.5) * 200,
		index: i,
		vx: 0,
		vy: 0,
		fx: undefined as number | undefined,
		fy: undefined as number | undefined,
	}))

	const nodeMap = new Map(nodes.map((n, i) => [n.id, i]))

	const simLinks = edges
		.map((e) => {
			const sourceIdx = nodeMap.get(e.source)
			const targetIdx = nodeMap.get(e.target)
			if (sourceIdx !== undefined && targetIdx !== undefined) {
				return { source: sourceIdx, target: targetIdx }
			}
			return null
		})
		.filter(Boolean) as Array<{ source: number; target: number }>

	// 🔥 修复TypeScript类型错误：使用any绕过d3-force的严格类型检查
	forceSimulation = d3Force
		.forceSimulation(simNodes as any)
		.force(
			"link",
			d3Force
				.forceLink(simLinks as any)
				.id((d: any, i: number) => (simNodes as any)[i].id)
				.distance(100)
		)
		.force("charge", d3Force.forceManyBody().strength(-300))
		.force("center", d3Force.forceCenter(config.width / 2, config.height / 2))
		.force("collide", d3Force.forceCollide(40))
		.alphaDecay(0.02)
		.stop()

	const maxIterations = config.algorithm === "force_incremental" ? 100 : 200  // 🔥 减少迭代次数
	const checkInterval = 10
	let partialSent = false

	// 🔥 性能优化：大图谱减少迭代，提前发送partial
	const earlyPartialThreshold = nodes.length > 1000 ? 50 : 100  // 1000+节点50次迭代就发partial

	for (let i = 0; i < maxIterations; i++) {
		forceSimulation!.tick()

		if (i % checkInterval === 0) {
			const alpha = forceSimulation!.alpha()
			if (alpha < 0.001) break

			const elapsed = performance.now() - startTime
			// 🔥 更早发送partial结果（特别是大图谱）
			if ((elapsed > 2000 || i >= earlyPartialThreshold) && !partialSent) {
				const positions = extractPositions(simNodes, nodes)
				self.postMessage({ type: "layout_partial", positions, isPartial: true })
				partialSent = true
			}
		}
	}

	const positions = extractPositions(simNodes, nodes)
	const durationMs = performance.now() - startTime

	if (partialSent) {
		self.postMessage({ type: "layout_refined", positions })
	} else {
		self.postMessage({ type: "layout_result", positions, durationMs })
	}

	forceSimulation = null
}

function extractPositions(
	simNodes: Array<{ id: string; x: number; y: number }>,
	originalNodes: LayoutNode[]
): Record<string, Position> {
	const positions: Record<string, Position> = {}
	for (let i = 0; i < originalNodes.length; i++) {
		const sim = simNodes[i]
		if (sim) {
			positions[originalNodes[i].id] = { x: sim.x, y: sim.y }
		}
	}
	return positions
}
