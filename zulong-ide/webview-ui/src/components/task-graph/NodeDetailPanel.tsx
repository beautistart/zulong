import { useGraphStore } from "../store/useGraphStore"
import { NODE_TYPE_ICONS, NODE_STATUS_COLORS } from "../types"

export default function NodeDetailPanel() {
	const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
	const nodes = useGraphStore((s) => s.nodes)

	if (!selectedNodeId) {
		return (
			<div style={{ padding: 16, color: "#888", fontSize: 13, textAlign: "center" }}>
				点击图谱中的任意节点
				<br />
				查看详情和依赖关系
			</div>
		)
	}

	const node = nodes.get(selectedNodeId)
	if (!node) {
		return (
			<div style={{ padding: 16, color: "#888", fontSize: 13 }}>
				节点不存在
			</div>
		)
	}

	const predecessorEdges: Array<{ from: string; via: string }> = []
	const successorEdges: Array<{ to: string; via: string }> = []
	const edges = useGraphStore.getState().edges

	for (const edge of edges.values()) {
		if (edge.target === node.id) {
			const fromNode = nodes.get(edge.source)
			predecessorEdges.push({ from: fromNode?.label || edge.source, via: edge.via })
		}
		if (edge.source === node.id) {
			const toNode = nodes.get(edge.target)
			successorEdges.push({ to: toNode?.label || edge.target, via: edge.via })
		}
	}

	return (
		<div style={{ padding: 12, fontSize: 13, overflowY: "auto", maxHeight: 400 }}>
			<div style={{ marginBottom: 12 }}>
				<span style={{ fontSize: 18, marginRight: 6 }}>{NODE_TYPE_ICONS[node.type]}</span>
				<span style={{ color: "#fff", fontWeight: 600 }}>{node.label || "--"}</span>
			</div>

			<InfoRow label="类型" value={node.type} />
			<InfoRow
				label="状态"
				value={
					<span style={{ color: NODE_STATUS_COLORS[node.status] }}>
						{node.status}
					</span>
				}
			/>
			<InfoRow label="描述" value={node.desc || "--"} />
			<InfoRow label="结果" value={node.result || "--"} />
			<InfoRow label="领域" value={node.taskDomain || "--"} />

			{node.files.length > 0 && (
				<div style={{ marginTop: 8 }}>
					<div style={{ color: "#aaa", marginBottom: 4 }}>关联文件</div>
					{node.files.map((f) => (
						<div key={f.path} style={{ color: "#6cb6ff", fontSize: 12, padding: "1px 0" }}>
							{f.name}
						</div>
					))}
				</div>
			)}

			{predecessorEdges.length > 0 && (
				<div style={{ marginTop: 8 }}>
					<div style={{ color: "#aaa", marginBottom: 4 }}>前驱依赖</div>
					{predecessorEdges.map((e) => (
						<div key={`${e.from}-${e.via}`} style={{ color: "#4ec9b0", fontSize: 12, padding: "1px 0" }}>
							{e.from}
							{e.via ? ` (${e.via})` : ""}
						</div>
					))}
				</div>
			)}

			{successorEdges.length > 0 && (
				<div style={{ marginTop: 8 }}>
					<div style={{ color: "#aaa", marginBottom: 4 }}>后继依赖</div>
					{successorEdges.map((e) => (
						<div key={`${e.to}-${e.via}`} style={{ color: "#6cb6ff", fontSize: 12, padding: "1px 0" }}>
							{e.to}
							{e.via ? ` (${e.via})` : ""}
						</div>
					))}
				</div>
			)}
		</div>
	)
}

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
	return (
		<div style={{ display: "flex", gap: 8, padding: "2px 0" }}>
			<span style={{ color: "#aaa", minWidth: 40 }}>{label}</span>
			<span style={{ color: "#ddd", flex: 1, wordBreak: "break-all" }}>{value}</span>
		</div>
	)
}
