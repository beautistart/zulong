/**
 * 图谱增量数据处理器
 * 
 * 处理后端推送的增量数据，合并到现有图谱
 * - 接收增量推送消息
 * - 合并节点增删改
 * - 合并边增删改
 * - 触发重新渲染
 */

import { useGraphStore } from "../store/useGraphStore"
import type { NodeAttribute, EdgeAttribute } from "../types"

export interface GraphDelta {
	added_nodes: Array<Record<string, any>>
	updated_nodes: Array<Record<string, any>>
	deleted_nodes: string[]
	added_edges: Array<Record<string, any>>
	deleted_edges: string[]
	timestamp: number
}

export interface GraphDeltaMessage {
	type: "graph_delta"
	delta: GraphDelta
	timestamp: number
}

export class GraphDeltaProcessor {
	private static instance: GraphDeltaProcessor | null = null

	private constructor() {}

	static getInstance(): GraphDeltaProcessor {
		if (!GraphDeltaProcessor.instance) {
			GraphDeltaProcessor.instance = new GraphDeltaProcessor()
		}
		return GraphDeltaProcessor.instance
	}

	/**
	 * 处理增量消息
	 */
	processDeltaMessage(message: GraphDeltaMessage): void {
		const { delta } = message
		console.log(`[GraphDeltaProcessor] Processing delta: +${delta.added_nodes.length} nodes, -${delta.deleted_nodes.length} nodes, ~${delta.updated_nodes.length} nodes`)

		this.applyNodeAdditions(delta.added_nodes)
		this.applyNodeUpdates(delta.updated_nodes)
		this.applyNodeDeletions(delta.deleted_nodes)
		this.applyEdgeAdditions(delta.added_edges)
		this.applyEdgeDeletions(delta.deleted_edges)
	}

	/**
	 * 应用节点添加
	 */
	private applyNodeAdditions(nodes: Array<Record<string, any>>): void {
		if (nodes.length === 0) return

		const store = useGraphStore.getState()
		const currentNodes = new Map(store.nodes)

		for (const nodeData of nodes) {
			const id = nodeData.id as string
			if (!id) continue

			const node: NodeAttribute = {
				id,
				label: (nodeData.label as string) || "",
				type: (nodeData.type as NodeAttribute["type"]) || "task",
				status: (nodeData.status as NodeAttribute["status"]) || "pending",
				desc: (nodeData.desc as string) || "",
				result: (nodeData.result as string) || "",
				files: (nodeData.files as NodeAttribute["files"]) || [],
				taskDomain: (nodeData.taskDomain as NodeAttribute["taskDomain"]) || "general",
				parentId: (nodeData.parent_id as string) || (nodeData.parentId as string) || null,
				position: { x: 0, y: 0 },
				metadata: (nodeData.metadata as Record<string, unknown>) || {},
			}

			currentNodes.set(id, node)
		}

		useGraphStore.setState({ nodes: currentNodes })
		console.log(`[GraphDeltaProcessor] Added ${nodes.length} nodes`)
	}

	/**
	 * 应用节点更新
	 */
	private applyNodeUpdates(nodes: Array<Record<string, any>>): void {
		if (nodes.length === 0) return

		const store = useGraphStore.getState()
		const currentNodes = new Map(store.nodes)

		for (const nodeData of nodes) {
			const id = nodeData.id as string
			const existing = currentNodes.get(id)
			if (!existing || !id) continue

			const updated: NodeAttribute = {
				...existing,
				...(nodeData.label !== undefined && { label: nodeData.label as string }),
				...(nodeData.type !== undefined && { type: nodeData.type as NodeAttribute["type"] }),
				...(nodeData.status !== undefined && { status: nodeData.status as NodeAttribute["status"] }),
				...(nodeData.desc !== undefined && { desc: nodeData.desc as string }),
				...(nodeData.result !== undefined && { result: nodeData.result as string }),
				...(nodeData.files !== undefined && { files: nodeData.files as NodeAttribute["files"] }),
				...(nodeData.parent_id !== undefined && { parentId: nodeData.parent_id as string }),
				...(nodeData.parentId !== undefined && { parentId: nodeData.parentId as string }),
				...(nodeData.metadata !== undefined && { metadata: nodeData.metadata as Record<string, unknown> }),
			}

			currentNodes.set(id, updated)
		}

		useGraphStore.setState({ nodes: currentNodes })
		console.log(`[GraphDeltaProcessor] Updated ${nodes.length} nodes`)
	}

	/**
	 * 应用节点删除
	 */
	private applyNodeDeletions(nodeIds: string[]): void {
		if (nodeIds.length === 0) return

		const store = useGraphStore.getState()
		const currentNodes = new Map(store.nodes)
		const currentEdges = new Map(store.edges)

		for (const nodeId of nodeIds) {
			currentNodes.delete(nodeId)

			for (const [edgeId, edge] of currentEdges) {
				if (edge.source === nodeId || edge.target === nodeId) {
					currentEdges.delete(edgeId)
				}
			}
		}

		useGraphStore.setState({ nodes: currentNodes, edges: currentEdges })
		console.log(`[GraphDeltaProcessor] Deleted ${nodeIds.length} nodes`)
	}

	/**
	 * 应用边添加
	 */
	private applyEdgeAdditions(edges: Array<Record<string, any>>): void {
		if (edges.length === 0) return

		const store = useGraphStore.getState()
		const currentEdges = new Map(store.edges)

		for (const edgeData of edges) {
			const source = edgeData.source as string
			const target = edgeData.target as string
			const edgeType = (edgeData.type as EdgeAttribute["type"]) || "dependency"
			const edgeId = `${edgeType.charAt(0)}_${source}_${target}`

			const edge: EdgeAttribute = {
				id: edgeId,
				source,
				target,
				type: edgeType,
				via: (edgeData.via as string) || "",
				cross: (edgeData.cross as boolean) || false,
			}

			currentEdges.set(edgeId, edge)
		}

		useGraphStore.setState({ edges: currentEdges })
		console.log(`[GraphDeltaProcessor] Added ${edges.length} edges`)
	}

	/**
	 * 应用边删除
	 */
	private applyEdgeDeletions(edgeIds: string[]): void {
		if (edgeIds.length === 0) return

		const store = useGraphStore.getState()
		const currentEdges = new Map(store.edges)

		for (const edgeId of edgeIds) {
			currentEdges.delete(edgeId)
		}

		useGraphStore.setState({ edges: currentEdges })
		console.log(`[GraphDeltaProcessor] Deleted ${edgeIds.length} edges`)
	}

	/**
	 * 验证增量消息
	 */
	validateDeltaMessage(message: any): message is GraphDeltaMessage {
		if (!message || typeof message !== "object") {
			return false
		}

		if (message.type !== "graph_delta") {
			return false
		}

		const delta = message.delta
		if (!delta || typeof delta !== "object") {
			return false
		}

		if (!Array.isArray(delta.added_nodes)) return false
		if (!Array.isArray(delta.updated_nodes)) return false
		if (!Array.isArray(delta.deleted_nodes)) return false
		if (!Array.isArray(delta.added_edges)) return false
		if (!Array.isArray(delta.deleted_edges)) return false

		return true
	}

	/**
	 * 获取增量统计信息
	 */
	getDeltaStats(delta: GraphDelta): string {
		const parts: string[] = []

		if (delta.added_nodes.length > 0) {
			parts.push(`+${delta.added_nodes.length} nodes`)
		}
		if (delta.deleted_nodes.length > 0) {
			parts.push(`-${delta.deleted_nodes.length} nodes`)
		}
		if (delta.updated_nodes.length > 0) {
			parts.push(`~${delta.updated_nodes.length} nodes`)
		}
		if (delta.added_edges.length > 0) {
			parts.push(`+${delta.added_edges.length} edges`)
		}
		if (delta.deleted_edges.length > 0) {
			parts.push(`-${delta.deleted_edges.length} edges`)
		}

		return parts.join(", ") || "no changes"
	}
}

export const graphDeltaProcessor = GraphDeltaProcessor.getInstance()
