import { useEffect, useRef, useState, useCallback } from "react"
import { useGraphStore } from "./store/useGraphStore"
import GraphCanvas from "./GraphCanvas"
import MindView from "./MindView"
import NodeDetailPanel from "./NodeDetailPanel"
import ProgressIndicator from "./ProgressIndicator"
import SearchBox from "./SearchBox"
import type { SearchResult } from "./types"

interface TaskGraphViewProps {
	graphData?: unknown
	onNodeSelect?: (nodeId: string | null) => void
}

export default function TaskGraphView({ graphData, onNodeSelect }: TaskGraphViewProps) {
	const containerRef = useRef<HTMLDivElement>(null)
	const [containerSize, setContainerSize] = useState({ width: 800, height: 600 })
	const [_searchResult, setSearchResult] = useState<SearchResult | null>(null)

	const renderStatus = useGraphStore((s) => s.renderStatus)
	const renderMode = useGraphStore((s) => s.renderMode)
	const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
	const nodes = useGraphStore((s) => s.nodes)
	const applyFullSync = useGraphStore((s) => s.applyFullSync)
	const _applyGraphUpdate = useGraphStore((s) => s.applyGraphUpdate)
	const reset = useGraphStore((s) => s.reset)

	useEffect(() => {
		if (onNodeSelect) {
			onNodeSelect(selectedNodeId)
		}
	}, [selectedNodeId, onNodeSelect])

	useEffect(() => {
		if (graphData && typeof graphData === "object") {
			const data = graphData as Record<string, unknown>
			if (data.nodes && Array.isArray(data.nodes)) {
				applyFullSync(graphData as Parameters<typeof applyFullSync>[0])
			}
		}
	}, [graphData, applyFullSync])

	useEffect(() => {
		const container = containerRef.current
		if (!container) return

		const observer = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const { width, height } = entry.contentRect
				if (width > 0 && height > 0) {
					setContainerSize({ width: Math.floor(width), height: Math.floor(height) })
				}
			}
		})

		observer.observe(container)
		return () => observer.disconnect()
	}, [])

	useEffect(() => {
		return () => {
			reset()
		}
	}, [reset])

	const handleSearchResult = useCallback((result: SearchResult) => {
		setSearchResult(result)
	}, [])

	const canvasWidth = containerSize.width - 280
	const canvasHeight = containerSize.height - 60

	return (
		<div
			ref={containerRef}
			style={{
				width: "100%",
				height: "100%",
				display: "flex",
				flexDirection: "column",
				background: "#1e1e1e",
				color: "#ddd",
				fontFamily: "sans-serif",
				overflow: "hidden",
			}}
		>
			<div
				style={{
					display: "flex",
					alignItems: "center",
					gap: 8,
					padding: "6px 12px",
					borderBottom: "1px solid #333",
					flexShrink: 0,
				}}
			>
				<span style={{ fontSize: 13, fontWeight: 600, color: "#fff" }}>任务图谱</span>
				<span style={{ fontSize: 11, color: "#888" }}>
					{nodes.size} 节点 | {renderMode === "full" ? "全量渲染" : renderMode === "virtual" ? "虚拟化渲染" : renderMode === "degraded" ? "降级渲染" : "优化渲染"}
				</span>
				<div style={{ flex: 1 }} />
				<div style={{ width: 200 }}>
					<SearchBox onResult={handleSearchResult} />
				</div>
			</div>

			<div style={{ padding: "0 12px", flexShrink: 0 }}>
				<ProgressIndicator />
			</div>

			<div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>
				<div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
					{canvasWidth > 100 && canvasHeight > 100 && (
						<GraphCanvas width={canvasWidth} height={canvasHeight} />
					)}
					{renderStatus === "degraded" && (
						<div
							style={{
								position: "absolute",
								top: 8,
								left: "50%",
								transform: "translateX(-50%)",
								padding: "3px 10px",
								background: "#803d36",
								color: "#ff9580",
								fontSize: 12,
								borderRadius: 3,
							}}
						>
							降级模式：节点过多，已切换为简化渲染
						</div>
					)}
				</div>

				<div
					style={{
						width: 280,
						borderLeft: "1px solid #333",
						display: "flex",
						flexDirection: "column",
						overflow: "hidden",
						flexShrink: 0,
					}}
				>
					<div style={{ flex: 1, overflow: "auto", borderBottom: "1px solid #333" }}>
						<div
							style={{
								padding: "4px 8px",
								background: "#2a2a2a",
								fontSize: 12,
								color: "#aaa",
								borderBottom: "1px solid #333",
							}}
						>
							节点详情
						</div>
						<NodeDetailPanel />
					</div>

					<div style={{ padding: 8, flexShrink: 0 }}>
						<MindView width={264} height={120} />
					</div>
				</div>
			</div>
		</div>
	)
}
