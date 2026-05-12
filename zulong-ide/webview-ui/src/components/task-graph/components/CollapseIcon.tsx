/**
 * 节点折叠图标组件
 * 
 * 显示折叠/展开图标（▼/▶）和子孙节点计数
 */

import React from "react"
import { useCollapseStore, CollapseIconState } from "../store/useCollapseStore"

interface CollapseIconProps {
	nodeId: string
	onToggle?: () => void
	size?: number
}

export const CollapseIcon: React.FC<CollapseIconProps> = ({ nodeId, onToggle, size = 16 }) => {
	const iconState = useCollapseStore((state) => state.getCollapseIconState(nodeId))
	const collapsedCount = useCollapseStore((state) => state.getCollapsedDescendantCount(nodeId))
	const toggleCollapse = useCollapseStore((state) => state.toggleNodeCollapse)

	if (iconState === "none") {
		return null
	}

	const handleClick = (e: React.MouseEvent) => {
		e.stopPropagation()
		toggleCollapse(nodeId)
		onToggle?.()
	}

	const isCollapsed = iconState === "collapsed"

	return (
		<div
			onClick={handleClick}
			style={{
				display: "inline-flex",
				alignItems: "center",
				gap: "4px",
				cursor: "pointer",
				userSelect: "none",
				opacity: 0.8,
				transition: "opacity 0.2s",
			}}
			onMouseEnter={(e) => {
				e.currentTarget.style.opacity = "1"
			}}
			onMouseLeave={(e) => {
				e.currentTarget.style.opacity = "0.8"
			}}
		>
			<svg
				width={size}
				height={size}
				viewBox="0 0 16 16"
				fill="currentColor"
				style={{
					transform: isCollapsed ? "rotate(-90deg)" : "rotate(0deg)",
					transition: "transform 0.2s",
				}}
			>
				<path d="M4 6l4 4 4-4z" />
			</svg>
			{isCollapsed && collapsedCount > 0 && (
				<span
					style={{
						fontSize: "12px",
						color: "var(--vscode-descriptionForeground, #6e6e6e)",
						fontWeight: 500,
					}}
				>
					({collapsedCount})
				</span>
			)}
		</div>
	)
}

interface NodeLabelWithCollapseProps {
	nodeId: string
	label: string
	onToggleCollapse?: () => void
}

export const NodeLabelWithCollapse: React.FC<NodeLabelWithCollapseProps> = ({
	nodeId,
	label,
	onToggleCollapse,
}) => {
	return (
		<div
			style={{
				display: "inline-flex",
				alignItems: "center",
				gap: "6px",
			}}
		>
			<CollapseIcon nodeId={nodeId} onToggle={onToggleCollapse} />
			<span>{label}</span>
		</div>
	)
}
