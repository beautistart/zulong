/**
 * 节点折叠状态管理
 * 
 * 使用 Zustand 管理节点折叠状态
 * - collapsedNodes: 已折叠节点ID集合
 * - nodeChildrenMap: 节点直接子节点映射
 * - nodeDescendantsMap: 节点所有子孙节点映射
 * - collapseIconState: 折叠图标状态映射
 * - collapsedNodeCount: 折叠节点计数映射
 */

import { create } from "zustand"

export type CollapseIconState = "collapsed" | "expanded" | "none"

export interface NodeCollapseState {
	collapsedNodes: Set<string>
	nodeChildrenMap: Map<string, Set<string>>
	nodeDescendantsMap: Map<string, Set<string>>
	collapseIconState: Map<string, CollapseIconState>
	collapsedNodeCount: Map<string, number>
}

export interface CollapseActions {
	initializeFromNodes: (nodes: Map<string, { id: string; parentId: string | null }>) => void
	toggleNodeCollapse: (nodeId: string) => void
	collapseNode: (nodeId: string) => void
	expandNode: (nodeId: string) => void
	collapseAll: () => void
	expandAll: () => void
	isNodeCollapsed: (nodeId: string) => boolean
	isNodeVisible: (nodeId: string) => boolean
	getVisibleNodeIds: () => string[]
	getCollapsedDescendantCount: (nodeId: string) => number
	getCollapseIconState: (nodeId: string) => CollapseIconState
	reset: () => void
}

const INITIAL_STATE: NodeCollapseState = {
	collapsedNodes: new Set(),
	nodeChildrenMap: new Map(),
	nodeDescendantsMap: new Map(),
	collapseIconState: new Map(),
	collapsedNodeCount: new Map(),
}

export const useCollapseStore = create<NodeCollapseState & CollapseActions>()((set, get) => ({
	...INITIAL_STATE,

	initializeFromNodes: (nodes) => {
		const nodeChildrenMap = new Map<string, Set<string>>()
		const nodeDescendantsMap = new Map<string, Set<string>>()
		const collapseIconState = new Map<string, CollapseIconState>()
		const collapsedNodeCount = new Map<string, number>()

		for (const [id, node] of nodes) {
			if (!nodeChildrenMap.has(id)) {
				nodeChildrenMap.set(id, new Set())
			}

			if (node.parentId) {
				const children = nodeChildrenMap.get(node.parentId)
				if (children) {
					children.add(id)
				} else {
					nodeChildrenMap.set(node.parentId, new Set([id]))
				}
			}
		}

		for (const [id] of nodes) {
			const descendants = new Set<string>()
			const queue = Array.from(nodeChildrenMap.get(id) || [])
			while (queue.length > 0) {
				const childId = queue.shift()!
				if (!descendants.has(childId)) {
					descendants.add(childId)
					const childChildren = nodeChildrenMap.get(childId)
					if (childChildren) {
						queue.push(...Array.from(childChildren))
					}
				}
			}
			nodeDescendantsMap.set(id, descendants)
		}

		for (const [id, children] of nodeChildrenMap) {
			if (children.size > 0) {
				collapseIconState.set(id, "expanded")
			} else {
				collapseIconState.set(id, "none")
			}
		}

		for (const [id, descendants] of nodeDescendantsMap) {
			collapsedNodeCount.set(id, descendants.size)
		}

		set({
			nodeChildrenMap,
			nodeDescendantsMap,
			collapseIconState,
			collapsedNodeCount,
			collapsedNodes: new Set(),
		})

		console.log(`[CollapseStore] Initialized with ${nodes.size} nodes`)
	},

	toggleNodeCollapse: (nodeId) => {
		const { collapsedNodes, collapseIconState, nodeDescendantsMap } = get()
		const iconState = collapseIconState.get(nodeId)

		if (iconState === "none") {
			return
		}

		const newCollapsedNodes = new Set(collapsedNodes)
		const newCollapseIconState = new Map(collapseIconState)

		if (newCollapsedNodes.has(nodeId)) {
			newCollapsedNodes.delete(nodeId)
			newCollapseIconState.set(nodeId, "expanded")
			console.log(`[CollapseStore] Expanded node: ${nodeId}`)
		} else {
			newCollapsedNodes.add(nodeId)
			newCollapseIconState.set(nodeId, "collapsed")
			console.log(`[CollapseStore] Collapsed node: ${nodeId}`)
		}

		set({
			collapsedNodes: newCollapsedNodes,
			collapseIconState: newCollapseIconState,
		})
	},

	collapseNode: (nodeId) => {
		const { collapsedNodes, collapseIconState, nodeDescendantsMap } = get()
		const iconState = collapseIconState.get(nodeId)

		if (iconState === "none" || collapsedNodes.has(nodeId)) {
			return
		}

		const newCollapsedNodes = new Set(collapsedNodes)
		const newCollapseIconState = new Map(collapseIconState)

		newCollapsedNodes.add(nodeId)
		newCollapseIconState.set(nodeId, "collapsed")

		set({
			collapsedNodes: newCollapsedNodes,
			collapseIconState: newCollapseIconState,
		})

		console.log(`[CollapseStore] Collapsed node: ${nodeId}`)
	},

	expandNode: (nodeId) => {
		const { collapsedNodes, collapseIconState } = get()

		if (!collapsedNodes.has(nodeId)) {
			return
		}

		const newCollapsedNodes = new Set(collapsedNodes)
		const newCollapseIconState = new Map(collapseIconState)

		newCollapsedNodes.delete(nodeId)
		newCollapseIconState.set(nodeId, "expanded")

		set({
			collapsedNodes: newCollapsedNodes,
			collapseIconState: newCollapseIconState,
		})

		console.log(`[CollapseStore] Expanded node: ${nodeId}`)
	},

	collapseAll: () => {
		const { nodeChildrenMap, collapseIconState } = get()

		const newCollapsedNodes = new Set<string>()
		const newCollapseIconState = new Map(collapseIconState)

		for (const [id, children] of nodeChildrenMap) {
			if (children.size > 0) {
				newCollapsedNodes.add(id)
				newCollapseIconState.set(id, "collapsed")
			}
		}

		set({
			collapsedNodes: newCollapsedNodes,
			collapseIconState: newCollapseIconState,
		})

		console.log(`[CollapseStore] Collapsed all ${newCollapsedNodes.size} nodes`)
	},

	expandAll: () => {
		const { collapseIconState } = get()

		const newCollapseIconState = new Map(collapseIconState)
		for (const [id, state] of newCollapseIconState) {
			if (state === "collapsed") {
				newCollapseIconState.set(id, "expanded")
			}
		}

		set({
			collapsedNodes: new Set(),
			collapseIconState: newCollapseIconState,
		})

		console.log(`[CollapseStore] Expanded all nodes`)
	},

	isNodeCollapsed: (nodeId) => {
		return get().collapsedNodes.has(nodeId)
	},

	isNodeVisible: (nodeId) => {
		const { collapsedNodes, nodeDescendantsMap } = get()

		for (const [collapsedId] of collapsedNodes) {
			const descendants = nodeDescendantsMap.get(collapsedId)
			if (descendants && descendants.has(nodeId)) {
				return false
			}
		}

		return true
	},

	getVisibleNodeIds: () => {
		const { collapsedNodes, nodeDescendantsMap } = get()
		const hiddenNodes = new Set<string>()

		for (const [collapsedId] of collapsedNodes) {
			const descendants = nodeDescendantsMap.get(collapsedId)
			if (descendants) {
				for (const descendant of descendants) {
					hiddenNodes.add(descendant)
				}
			}
		}

		const visibleIds: string[] = []
		for (const [id] of nodeDescendantsMap) {
			if (!hiddenNodes.has(id)) {
				visibleIds.push(id)
			}
		}

		return visibleIds
	},

	getCollapsedDescendantCount: (nodeId) => {
		const { nodeDescendantsMap } = get()
		return nodeDescendantsMap.get(nodeId)?.size || 0
	},

	getCollapseIconState: (nodeId) => {
		return get().collapseIconState.get(nodeId) || "none"
	},

	reset: () => {
		set({
			...INITIAL_STATE,
			collapsedNodes: new Set(),
			nodeChildrenMap: new Map(),
			nodeDescendantsMap: new Map(),
			collapseIconState: new Map(),
			collapsedNodeCount: new Map(),
		})
		console.log(`[CollapseStore] Reset`)
	},
}))
