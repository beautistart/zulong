import { useGraphStore } from "../store/useGraphStore"
import type { SearchResult } from "../types"

interface SearchBoxProps {
	onResult: (result: SearchResult) => void
}

export default function SearchBox({ onResult }: SearchBoxProps) {
	const [keyword, setKeyword] = useState("")
	const [result, setResult] = useState<SearchResult | null>(null)
	const nodes = useGraphStore((s) => s.nodes)
	const selectNode = useGraphStore((s) => s.selectNode)
	const setViewport = useGraphStore((s) => s.setViewport)

	const handleSearch = useCallback(() => {
		if (!keyword.trim()) {
			setResult(null)
			onResult({ nodeIds: [], currentIndex: 0, total: 0 })
			return
		}

		const lower = keyword.toLowerCase()
		const nodeArray = Array.from(nodes.values())
		const matched = nodeArray
			.filter((n) => n.label.toLowerCase().includes(lower) || n.desc.toLowerCase().includes(lower))
			.map((n) => n.id)

		const newResult: SearchResult = { nodeIds: matched, currentIndex: 0, total: matched.length }
		setResult(newResult)
		onResult(newResult)

		if (matched.length > 0) {
			selectNode(matched[0])
			const firstNode = nodes.get(matched[0])
			if (firstNode) {
				const vp = useGraphStore.getState().viewport
				setViewport({
					...vp,
					offsetX: vp.width / 2 - firstNode.position.x * vp.zoom,
					offsetY: vp.height / 2 - firstNode.position.y * vp.zoom,
				})
			}
		}
	}, [keyword, nodes, selectNode, setViewport, onResult])

	const handleNext = useCallback(() => {
		if (!result || result.total === 0) return
		const newIdx = (result.currentIndex + 1) % result.total
		const newResult = { ...result, currentIndex: newIdx }
		setResult(newResult)
		const nodeId = result.nodeIds[newIdx]
		selectNode(nodeId)
		const node = nodes.get(nodeId)
		if (node) {
			const vp = useGraphStore.getState().viewport
			setViewport({
				...vp,
				offsetX: vp.width / 2 - node.position.x * vp.zoom,
				offsetY: vp.height / 2 - node.position.y * vp.zoom,
			})
		}
	}, [result, nodes, selectNode, setViewport])

	return (
		<div style={{ display: "flex", gap: 4, alignItems: "center" }}>
			<input
				type="text"
				value={keyword}
				onChange={(e) => setKeyword(e.target.value)}
				onKeyDown={(e) => e.key === "Enter" && handleSearch()}
				placeholder="搜索节点..."
				style={{
					flex: 1,
					padding: "3px 8px",
					background: "#2d2d2d",
					color: "#fff",
					border: "1px solid #555",
					borderRadius: 3,
					fontSize: 12,
					outline: "none",
				}}
			/>
			<button
				onClick={handleSearch}
				style={{
					padding: "3px 8px",
					background: "#2d2d2d",
					color: "#ccc",
					border: "1px solid #555",
					cursor: "pointer",
					fontSize: 12,
				}}
			>
				搜索
			</button>
			{result && result.total > 1 && (
				<button
					onClick={handleNext}
					style={{
						padding: "3px 6px",
						background: "#2d2d2d",
						color: "#ccc",
						border: "1px solid #555",
						cursor: "pointer",
						fontSize: 11,
					}}
				>
					{result.currentIndex + 1}/{result.total}
				</button>
			)}
			{result && result.total === 0 && (
				<span style={{ color: "#e51400", fontSize: 11 }}>未找到</span>
			)}
		</div>
	)
}
