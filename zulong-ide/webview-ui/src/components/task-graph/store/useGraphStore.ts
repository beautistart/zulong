import { create } from "zustand"
import type {
	NodeAttribute,
	EdgeAttribute,
	ViewportState,
	RenderStatus,
	RenderMode,
	RenderMetrics,
	ProgressInfo,
	GraphFullSyncPayload,
	GraphUpdatePayload,
	Position,
} from "../types"

interface GraphState {
	nodes: Map<string, NodeAttribute>
	edges: Map<string, EdgeAttribute>
	graphId: string | null
	graphTitle: string
	renderStatus: RenderStatus
	renderMode: RenderMode
	errorMessage: string | null
	viewport: ViewportState
	selectedNodeId: string | null
	hoveredNodeId: string | null
	metrics: RenderMetrics | null
	layoutPositions: Map<string, Position>
}

interface GraphActions {
	applyFullSync: (payload: GraphFullSyncPayload) => void
	applyGraphUpdate: (payload: GraphUpdatePayload) => void
	setViewport: (viewport: Partial<ViewportState>) => void
	selectNode: (nodeId: string | null) => void
	hoverNode: (nodeId: string | null) => void
	setRenderStatus: (status: RenderStatus) => void
	setRenderMode: (mode: RenderMode) => void
	setErrorMessage: (message: string | null) => void
	setMetrics: (metrics: RenderMetrics) => void
	setLayoutPositions: (positions: Map<string, Position>) => void
	updateNodePosition: (nodeId: string, position: Position) => void
	reset: () => void
	getNodeCount: () => number
	getCompletedCount: () => number
	getProgress: () => ProgressInfo
	getVisibleNodeIds: (viewport: ViewportState, bufferRatio: number) => string[]
	getNodeChildren: (parentId: string) => string[]
	getNodeDescendants: (nodeId: string) => string[]
}

const INITIAL_VIEWPORT: ViewportState = {
	offsetX: 0,
	offsetY: 0,
	zoom: 1,
	width: 800,
	height: 600,
}

const INITIAL_STATE: GraphState = {
	nodes: new Map(),
	edges: new Map(),
	graphId: null,
	graphTitle: "",
	renderStatus: "idle",
	renderMode: "full",
	errorMessage: null,
	viewport: INITIAL_VIEWPORT,
	selectedNodeId: null,
	hoveredNodeId: null,
	metrics: null,
	layoutPositions: new Map(),
}

export const useGraphStore = create<GraphState & GraphActions>()((set, get) => ({
	...INITIAL_STATE,

	applyFullSync: (payload) => {
		const nodes = new Map<string, NodeAttribute>()
		const edges = new Map<string, EdgeAttribute>()
		let _edgeIndex = 0

		for (const rawNode of payload.nodes) {
			nodes.set(rawNode.id, {
				id: rawNode.id,
				label: rawNode.label || "",
				type: (rawNode.type as NodeAttribute["type"]) || "task",
				status: (rawNode.status as NodeAttribute["status"]) || "pending",
				desc: rawNode.desc || "",
				result: rawNode.result || "",
				files: rawNode.files || [],
				taskDomain: rawNode.taskDomain || "general",
				parentId: rawNode.parentId || null,
				position: { x: 0, y: 0 },
				metadata: rawNode.metadata || {},
			})
		}

		if (payload.hEdges) {
			for (const [parent, child] of payload.hEdges) {
				const edgeId = `h_${parent}_${child}`
				edges.set(edgeId, {
					id: edgeId,
					source: parent,
					target: child,
					type: "hierarchy",
					via: "",
					cross: false,
				})
				const childNode = nodes.get(child)
				if (childNode) {
					nodes.set(child, { ...childNode, parentId: parent })
				}
				_edgeIndex++
			}
		}

		if (payload.dEdges) {
			for (const de of payload.dEdges) {
				const edgeId = `d_${de.s}_${de.t}`
				edges.set(edgeId, {
					id: edgeId,
					source: de.s,
					target: de.t,
					type: "dependency",
					via: de.via || "",
					cross: de.cross || false,
				})
				_edgeIndex++
			}
		}

		set({
			nodes,
			edges,
			graphId: payload.id,
			graphTitle: payload.title,
			renderStatus: "idle",
			errorMessage: null,
		})
	},

	applyGraphUpdate: (payload) => {
		const { nodes, edges } = get()
		const { action, data } = payload

		switch (action) {
			case "node_created": {
				const newNodes = new Map(nodes)
				const nodeData = data as Record<string, unknown>
				const id = nodeData.id as string
				if (id) {
					newNodes.set(id, {
						id,
						label: (nodeData.label as string) || "",
						type: (nodeData.type as NodeAttribute["type"]) || "task",
						status: (nodeData.status as NodeAttribute["status"]) || "pending",
						desc: (nodeData.desc as string) || "",
						result: (nodeData.result as string) || "",
						files: (nodeData.files as NodeAttribute["files"]) || [],
						taskDomain: (nodeData.taskDomain as NodeAttribute["taskDomain"]) || "general",
						parentId: (nodeData.parentId as string) || null,
						position: { x: 0, y: 0 },
						metadata: (nodeData.metadata as Record<string, unknown>) || {},
					})
				}
				set({ nodes: newNodes })
				break
			}
			case "node_updated": {
				const newNodes = new Map(nodes)
				const nodeData = data as Record<string, unknown>
				const id = nodeData.id as string
				const existing = newNodes.get(id)
				if (existing && id) {
					newNodes.set(id, {
						...existing,
						...(nodeData.label !== undefined && { label: nodeData.label as string }),
						...(nodeData.status !== undefined && { status: nodeData.status as NodeAttribute["status"] }),
						...(nodeData.desc !== undefined && { desc: nodeData.desc as string }),
						...(nodeData.result !== undefined && { result: nodeData.result as string }),
						...(nodeData.files !== undefined && { files: nodeData.files as NodeAttribute["files"] }),
						...(nodeData.metadata !== undefined && { metadata: nodeData.metadata as Record<string, unknown> }),
					})
				}
				set({ nodes: newNodes })
				break
			}
			case "node_deleted": {
				const newNodes = new Map(nodes)
				const newEdges = new Map(edges)
				const id = (data as Record<string, unknown>).id as string
				newNodes.delete(id)
				for (const [edgeId, edge] of newEdges) {
					if (edge.source === id || edge.target === id) {
						newEdges.delete(edgeId)
					}
				}
				set({ nodes: newNodes, edges: newEdges })
				break
			}
			case "edge_created": {
				const newEdges = new Map(edges)
				const edgeData = data as Record<string, unknown>
				const source = edgeData.source as string
				const target = edgeData.target as string
				const edgeType = (edgeData.type as EdgeAttribute["type"]) || "dependency"
				const edgeId = `${edgeType.charAt(0)}_${source}_${target}`
				newEdges.set(edgeId, {
					id: edgeId,
					source,
					target,
					type: edgeType,
					via: (edgeData.via as string) || "",
					cross: (edgeData.cross as boolean) || false,
				})
				set({ edges: newEdges })
				break
			}
			case "edge_deleted": {
				const newEdges = new Map(edges)
				const edgeData = data as Record<string, unknown>
				const edgeId = edgeData.id as string
				newEdges.delete(edgeId)
				set({ edges: newEdges })
				break
			}
			case "batch_created":
			case "batch_updated": {
				const batchData = data as Record<string, unknown>
				const batchNodes = (batchData.nodes as Array<Record<string, unknown>>) || []
				const newNodes = new Map(nodes)
				for (const nd of batchNodes) {
					const id = nd.id as string
					const existing = newNodes.get(id)
					if (existing) {
						newNodes.set(id, {
							...existing,
							...(nd.label !== undefined && { label: nd.label as string }),
							...(nd.status !== undefined && { status: nd.status as NodeAttribute["status"] }),
							...(nd.desc !== undefined && { desc: nd.desc as string }),
							...(nd.result !== undefined && { result: nd.result as string }),
						})
					} else if (id) {
						newNodes.set(id, {
							id,
							label: (nd.label as string) || "",
							type: (nd.type as NodeAttribute["type"]) || "task",
							status: (nd.status as NodeAttribute["status"]) || "pending",
							desc: (nd.desc as string) || "",
							result: (nd.result as string) || "",
							files: (nd.files as NodeAttribute["files"]) || [],
							taskDomain: (nd.taskDomain as NodeAttribute["taskDomain"]) || "general",
							parentId: (nd.parentId as string) || null,
							position: { x: 0, y: 0 },
							metadata: (nd.metadata as Record<string, unknown>) || {},
						})
					}
				}
				set({ nodes: newNodes })
				break
			}
		}
	},

	setViewport: (partial) => {
		set((state) => ({ viewport: { ...state.viewport, ...partial } }))
	},

	selectNode: (nodeId) => {
		set({ selectedNodeId: nodeId })
	},

	hoverNode: (nodeId) => {
		set({ hoveredNodeId: nodeId })
	},

	setRenderStatus: (status) => {
		set({ renderStatus: status })
	},

	setRenderMode: (mode) => {
		set({ renderMode: mode })
	},

	setErrorMessage: (message) => {
		set({ errorMessage: message })
	},

	setMetrics: (metrics) => {
		set({ metrics })
	},

	setLayoutPositions: (positions) => {
		set((state) => {
			const newNodes = new Map(state.nodes)
			for (const [nodeId, position] of positions) {
				const node = newNodes.get(nodeId)
				if (node) {
					newNodes.set(nodeId, { ...node, position })
				}
			}
			return { nodes: newNodes, layoutPositions: positions }
		})
	},

	updateNodePosition: (nodeId, position) => {
		set((state) => {
			const node = state.nodes.get(nodeId)
			if (!node) return state
			const newNodes = new Map(state.nodes)
			newNodes.set(nodeId, { ...node, position })
			return { nodes: newNodes }
		})
	},

	reset: () => {
		set({
			...INITIAL_STATE,
			nodes: new Map(),
			edges: new Map(),
			layoutPositions: new Map(),
		})
	},

	getNodeCount: () => {
		return get().nodes.size
	},

	getCompletedCount: () => {
		let count = 0
		for (const node of get().nodes.values()) {
			if (node.status === "completed") count++
		}
		return count
	},

	getProgress: () => {
		const total = get().nodes.size
		const completed = get().getCompletedCount()
		const percentage = total > 0 ? Math.round((completed / total) * 100) : 0
		return {
			totalNodes: total,
			completedNodes: completed,
			percentage,
			text: `${completed}/${total} 已完成`,
		}
	},

	getVisibleNodeIds: (viewport, bufferRatio) => {
		const { nodes } = get()
		const bufferW = viewport.width * bufferRatio
		const bufferH = viewport.height * bufferRatio
		const viewLeft = viewport.offsetX - bufferW
		const viewRight = viewport.offsetX + viewport.width + bufferW
		const viewTop = viewport.offsetY - bufferH
		const viewBottom = viewport.offsetY + viewport.height + bufferH

		const result: string[] = []
		for (const [id, node] of nodes) {
			const screenX = node.position.x * viewport.zoom + viewport.offsetX
			const screenY = node.position.y * viewport.zoom + viewport.offsetY
			if (screenX >= viewLeft && screenX <= viewRight && screenY >= viewTop && screenY <= viewBottom) {
				result.push(id)
			}
		}
		return result
	},

	getNodeChildren: (parentId: string): string[] => {
		const { nodes } = get()
		const children: string[] = []
		for (const [id, node] of nodes) {
			if (node.parentId === parentId) {
				children.push(id)
			}
		}
		return children
	},

	getNodeDescendants: (nodeId: string): string[] => {
		const { nodes } = get()
		const descendants = new Set<string>()
		const queue: string[] = [nodeId]

		while (queue.length > 0) {
			const currentId = queue.shift()!
			for (const [id, node] of nodes) {
				if (node.parentId === currentId && !descendants.has(id)) {
					descendants.add(id)
					queue.push(id)
				}
			}
		}

		return Array.from(descendants)
	},
}))
