export function escapeHtml(text: string): string {
	return text
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#x27;")
}

export function sanitizePath(path: string): string {
	const parts = path.replace(/\\/g, "/").split("/")
	if (parts.length <= 2) return path
	return `…/${parts.slice(-2).join("/")}`
}

export function validateGraphData(data: unknown): { valid: boolean; errors: string[] } {
	const errors: string[] = []

	if (!data || typeof data !== "object") {
		return { valid: false, errors: ["数据不是有效对象"] }
	}

	const obj = data as Record<string, unknown>

	if (!obj.id || typeof obj.id !== "string") {
		errors.push("缺少图谱ID")
	}

	if (!Array.isArray(obj.nodes)) {
		errors.push("nodes字段不是数组")
	} else {
		const ids = new Set<string>()
		for (let i = 0; i < obj.nodes.length; i++) {
			const node = obj.nodes[i] as Record<string, unknown>
			if (!node.id || typeof node.id !== "string") {
				errors.push(`节点[${i}]缺少id`)
				continue
			}
			if (ids.has(node.id)) {
				errors.push(`节点ID重复: ${node.id}`)
			}
			ids.add(node.id)
		}

		if (Array.isArray(obj.hEdges)) {
			for (let i = 0; i < obj.hEdges.length; i++) {
				const edge = obj.hEdges[i] as Array<unknown>
				if (!Array.isArray(edge) || edge.length !== 2) {
					errors.push(`hEdges[${i}]格式错误`)
					continue
				}
				if (!ids.has(edge[0] as string)) {
					errors.push(`hEdges[${i}]源节点不存在: ${edge[0]}`)
				}
				if (!ids.has(edge[1] as string)) {
					errors.push(`hEdges[${i}]目标节点不存在: ${edge[1]}`)
				}
			}
		}

		if (Array.isArray(obj.dEdges)) {
			for (let i = 0; i < obj.dEdges.length; i++) {
				const edge = obj.dEdges[i] as Record<string, unknown>
				if (!ids.has(edge.s as string)) {
					errors.push(`dEdges[${i}]源节点不存在: ${edge.s}`)
				}
				if (!ids.has(edge.t as string)) {
					errors.push(`dEdges[${i}]目标节点不存在: ${edge.t}`)
				}
			}
		}
	}

	return { valid: errors.length === 0, errors }
}
