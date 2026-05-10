import { useGraphStore } from "../store/useGraphStore"

export default function ProgressIndicator() {
	const nodes = useGraphStore((s) => s.nodes)
	const total = nodes.size
	let completed = 0
	for (const node of nodes.values()) {
		if (node.status === "completed") completed++
	}
	const percentage = total > 0 ? Math.round((completed / total) * 100) : 0
	const text = `${completed}/${total} 已完成`

	return (
		<div style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0" }}>
			<div
				style={{
					flex: 1,
					height: 6,
					background: "#3c3c3c",
					borderRadius: 3,
					overflow: "hidden",
				}}
			>
				<div
					style={{
						width: `${percentage}%`,
						height: "100%",
						background: percentage >= 100 ? "#388a34" : "#e2ab00",
						borderRadius: 3,
						transition: "width 0.3s ease",
					}}
				/>
			</div>
			<span style={{ color: "#aaa", fontSize: 12, whiteSpace: "nowrap" }}>{text}</span>
		</div>
	)
}
