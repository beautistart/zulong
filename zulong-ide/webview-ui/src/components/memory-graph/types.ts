/**
 * 记忆图谱可视化类型 (TSD 23.6.2)
 */

/** 注意力三态模式 */
export type AttentionMode = "GLOBAL" | "SINGLE_CHAIN" | "FOCUS"

/** 注意力状态数据模型 */
export interface AttentionState {
	mode: AttentionMode

	// GLOBAL 模式
	all_nodes: string[]
	active_nodes: string[]
	focus_node?: string

	// SINGLE_CHAIN 模式
	chain_nodes: string[]
	chain_edges: string[]

	// FOCUS 模式
	focal_node: string
	neighborhood: {
		nodes: string[]
		edges: string[]
		radius: number
	}

	// 通用
	activation_values: Record<string, number>
	timestamp: number
}

/** BFS 扩散步骤 */
export interface BFSStep {
	hop: number
	nodes: string[]
	edges: string[]
	weights: Record<string, number>
	timestamp_ms: number
}

/** BFS 扩散动画配置 */
export interface BFSAnimationState {
	steps: BFSStep[]
	current_step: number
	is_playing: boolean
	seed_node: string
}

/** 记忆节点 (用于前端渲染) */
export interface MemoryNode {
	id: string
	label: string
	type: string
	activation: number
	x: number
	y: number
	color?: string
}

/** 记忆边 (用于前端渲染) */
export interface MemoryEdge {
	source: string
	target: string
	weight: number
	type: string
}

/** 记忆面板展示数据 */
export interface MemoryPanelData {
	conversation_id: string
	nodes_created: number
	nodes_created_details: Array<{ label: string; type: string }>
	edges_strengthened: number
	active_memories: Array<{
		id: string
		label: string
		activation: number
		temperature: "HOT" | "WARM" | "COLD"
	}>
	hierarchy: MemoryHierarchyNode[]
}

export interface MemoryHierarchyNode {
	id: string
	label: string
	type: "project" | "module" | "conversation" | "knowledge" | "experience"
	children: MemoryHierarchyNode[]
}
