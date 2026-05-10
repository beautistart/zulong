import type { Position } from "../types"

interface HierarchyNode {
	id: string
	parentId: string | null
	children: string[]
	depth: number
}

export function hierarchyLayout(
	nodes: Array<{ id: string; parentId: string | null }>,
	config: { width: number; height: number }
): Record<string, Position> {
	if (nodes.length === 0) return {}

	const nodeMap = new Map<string, HierarchyNode>()
	for (const n of nodes) {
		nodeMap.set(n.id, { id: n.id, parentId: n.parentId, children: [], depth: 0 })
	}

	for (const n of nodes) {
		if (n.parentId && nodeMap.has(n.parentId)) {
			nodeMap.get(n.parentId)?.children.push(n.id)
		}
	}

	const roots: string[] = []
	for (const n of nodes) {
		if (!n.parentId || !nodeMap.has(n.parentId)) {
			roots.push(n.id)
		}
	}
	if (roots.length === 0 && nodes.length > 0) {
		roots.push(nodes[0].id)
	}

	const depthGroups = new Map<number, string[]>()
	const queue: Array<{ id: string; depth: number }> = roots.map((r) => ({ id: r, depth: 0 }))
	const visited = new Set<string>()

	while (queue.length > 0) {
		const item = queue.shift()
		if (!item) break
		const { id, depth } = item
		if (visited.has(id)) continue
		visited.add(id)

		if (!depthGroups.has(depth)) {
			depthGroups.set(depth, [])
		}
		depthGroups.get(depth)?.push(id)

		const node = nodeMap.get(id)
		if (node) {
			node.depth = depth
			for (const childId of node.children) {
				if (!visited.has(childId)) {
					queue.push({ id: childId, depth: depth + 1 })
				}
			}
		}
	}

	for (const n of nodes) {
		if (!visited.has(n.id)) {
			const maxDepth = Math.max(...Array.from(depthGroups.keys()), -1)
			const depth = maxDepth + 1
			if (!depthGroups.has(depth)) depthGroups.set(depth, [])
			depthGroups.get(depth)?.push(n.id)
			visited.add(n.id)
		}
	}

	const positions: Record<string, Position> = {}
	const maxDepth = Math.max(...Array.from(depthGroups.keys()), 0)
	const layerHeight = config.height / (maxDepth + 2)
	const padding = 80

	for (const [depth, nodeIds] of depthGroups) {
		const layerWidth = config.width - padding * 2
		const spacing = layerWidth / (nodeIds.length + 1)
		const y = padding + depth * layerHeight

		for (let i = 0; i < nodeIds.length; i++) {
			positions[nodeIds[i]] = {
				x: padding + (i + 1) * spacing,
				y,
			}
		}
	}

	return positions
}
