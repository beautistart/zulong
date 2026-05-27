import { memo, useState, useCallback } from "react"
import {
	BrainIcon,
	ChevronDownIcon,
	ChevronRightIcon,
	SearchIcon,
	NetworkIcon,
	PlusIcon,
	TrendingUpIcon,
	FlameIcon,
	ThermometerIcon,
	FolderIcon,
	MessageSquareIcon,
	BookOpenIcon,
	ZapIcon,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { MemoryPanelData, MemoryHierarchyNode } from "./types"

interface MemoryPanelProps {
	data: MemoryPanelData
	onExpandAll?: () => void
	onSearchMemory?: (query: string) => void
	onViewKnowledgeGraph?: () => void
	className?: string
}

const tempConfig: Record<string, { icon: typeof FlameIcon; color: string; label: string }> = {
	HOT: { icon: FlameIcon, color: "text-red-500", label: "HOT" },
	WARM: { icon: ThermometerIcon, color: "text-amber-500", label: "WARM" },
	COLD: { icon: ThermometerIcon, color: "text-blue-400", label: "COLD" },
}

const typeConfig: Record<string, { icon: typeof FolderIcon; label: string }> = {
	project: { icon: FolderIcon, label: "项目" },
	module: { icon: FolderIcon, label: "模块" },
	conversation: { icon: MessageSquareIcon, label: "对话" },
	knowledge: { icon: BookOpenIcon, label: "知识" },
	experience: { icon: ZapIcon, label: "经验" },
}

/**
 * 记忆面板 (TSD 23.6.4)
 *
 * 展示:
 * - 本次对话产生的记忆
 * - 当前激活的记忆 (按激活值排序)
 * - 记忆层级结构
 */
const MemoryPanel = memo(function MemoryPanel({
	data,
	onExpandAll,
	onSearchMemory,
	onViewKnowledgeGraph,
	className,
}: MemoryPanelProps) {
	const [searchQuery, setSearchQuery] = useState("")
	const [hierarchyExpanded, setHierarchyExpanded] = useState(true)
	const [activeMemoriesExpanded, setActiveMemoriesExpanded] = useState(true)

	const handleSearch = useCallback(() => {
		onSearchMemory?.(searchQuery)
	}, [searchQuery, onSearchMemory])

	return (
		<div
			className={cn(
				"rounded-lg border border-purple-700/30 bg-slate-900/50 overflow-hidden",
				className,
			)}
		>
			{/* 标题 */}
			<div className="flex items-center gap-2 p-2 border-b border-slate-700/30">
				<BrainIcon className="h-4 w-4 text-purple-400" />
				<span className="text-xs font-medium text-slate-300">
					记忆活跃
				</span>
				{data.conversation_id && (
					<span className="text-xs text-slate-500 ml-auto">
						{data.conversation_id}
					</span>
				)}
			</div>

			<div className="p-3 space-y-3 max-h-[500px] overflow-y-auto">
				{/* 本次对话产生 */}
				<div className="space-y-1">
					<div className="text-xs font-medium text-slate-400">
						本次对话产生:
					</div>
					<div className="space-y-0.5 text-xs text-slate-500">
						{data.nodes_created > 0 && (
							<div className="flex items-center gap-1.5">
								<PlusIcon className="h-3 w-3 text-green-500" />
								<span>
									{data.nodes_created} 个新节点
								</span>
							</div>
						)}
						{data.nodes_created_details.slice(0, 3).map((detail, i) => (
							<div
								key={i}
								className="flex items-center gap-1.5 pl-4 text-slate-500"
							>
								{detail.type === "knowledge" ? (
									<BookOpenIcon className="h-3 w-3 text-blue-400" />
								) : detail.type === "experience" ? (
									<ZapIcon className="h-3 w-3 text-amber-400" />
								) : (
									<MessageSquareIcon className="h-3 w-3 text-slate-500" />
								)}
								<span>{detail.label}</span>
							</div>
						))}
						{data.edges_strengthened > 0 && (
							<div className="flex items-center gap-1.5">
								<TrendingUpIcon className="h-3 w-3 text-blue-500" />
								<span>强化了 {data.edges_strengthened} 条关联边</span>
							</div>
						)}
					</div>
				</div>

				{/* 当前激活的记忆 */}
				<div className="space-y-1 border-t border-slate-700/20 pt-2">
					<button
						type="button"
						onClick={() =>
							setActiveMemoriesExpanded(!activeMemoriesExpanded)
						}
						className="flex items-center gap-1 text-xs font-medium text-slate-400 hover:text-slate-300 w-full text-left"
					>
						{activeMemoriesExpanded ? (
							<ChevronDownIcon className="h-3 w-3" />
						) : (
							<ChevronRightIcon className="h-3 w-3" />
						)}
						当前激活的记忆:
					</button>
					{activeMemoriesExpanded && (
						<div className="space-y-1">
							{data.active_memories.length === 0 && (
								<div className="text-xs text-slate-600 py-2 text-center">
									暂无激活记忆
								</div>
							)}
							{data.active_memories.map((mem) => {
								const cfg = tempConfig[mem.temperature]
								const TempIcon = cfg?.icon || ThermometerIcon
								return (
									<div
										key={mem.id}
										className="flex items-center gap-2 rounded bg-slate-800/50 px-2 py-1"
									>
										<TempIcon
											className={cn("h-3 w-3", cfg?.color)}
										/>
										<span
											className={cn("text-xs font-medium", cfg?.color)}
										>
											{cfg?.label || mem.temperature}
										</span>
										<span className="text-xs text-slate-300 flex-1 truncate">
											{mem.label}
										</span>
										<div className="flex items-center gap-1">
											<div className="h-1 w-12 rounded-full bg-slate-700 overflow-hidden">
												<div
													className="h-full rounded-full bg-purple-500 transition-all"
													style={{
														width: `${Math.round(mem.activation * 100)}%`,
													}}
												/>
											</div>
											<span className="text-xs text-slate-500 tabular-nums w-8">
												{(mem.activation * 100).toFixed(0)}%
											</span>
										</div>
									</div>
								)
							})}
						</div>
					)}
				</div>

				{/* 记忆层级 */}
				<div className="space-y-1 border-t border-slate-700/20 pt-2">
					<button
						type="button"
						onClick={() =>
							setHierarchyExpanded(!hierarchyExpanded)
						}
						className="flex items-center gap-1 text-xs font-medium text-slate-400 hover:text-slate-300 w-full text-left"
					>
						{hierarchyExpanded ? (
							<ChevronDownIcon className="h-3 w-3" />
						) : (
							<ChevronRightIcon className="h-3 w-3" />
						)}
						记忆层级:
					</button>
					{hierarchyExpanded && (
						<div className="pl-2">
							{data.hierarchy.map((node) => (
								<HierarchyNodeItem key={node.id} node={node} />
							))}
						</div>
					)}
				</div>
			</div>

			{/* 底部操作栏 */}
			<div className="flex items-center gap-2 p-2 border-t border-slate-700/30">
				<div className="flex-1 flex items-center gap-1 bg-slate-800/50 rounded px-2 py-1">
					<SearchIcon className="h-3 w-3 text-slate-500" />
					<input
						type="text"
						value={searchQuery}
						onChange={(e) => setSearchQuery(e.target.value)}
						onKeyDown={(e) => e.key === "Enter" && handleSearch()}
						placeholder="搜索记忆..."
						className="flex-1 text-xs bg-transparent text-slate-300 placeholder:text-slate-600 outline-none"
					/>
				</div>
				{onExpandAll && (
					<button
						type="button"
						onClick={onExpandAll}
						className="rounded px-2 py-1 text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-800/50"
					>
						展开全部
					</button>
				)}
				{onViewKnowledgeGraph && (
					<button
						type="button"
						onClick={onViewKnowledgeGraph}
						className="rounded px-2 py-1 text-xs text-purple-400 hover:text-purple-300 hover:bg-purple-500/10"
					>
						<NetworkIcon className="h-3 w-3 inline mr-1" />
						知识图谱
					</button>
				)}
			</div>
		</div>
	)
})

function HierarchyNodeItem({ node }: { node: MemoryHierarchyNode }) {
	const [expanded, setExpanded] = useState(true)
	const hasChildren = node.children.length > 0
	const cfg = typeConfig[node.type]
	const Icon = cfg?.icon || FolderIcon

	return (
		<div>
			<div
				className="flex items-center gap-1 py-0.5 text-xs cursor-pointer hover:text-slate-300 text-slate-400"
				onClick={() => hasChildren && setExpanded(!expanded)}
			>
				{hasChildren ? (
					expanded ? (
						<ChevronDownIcon className="h-3 w-3 shrink-0" />
					) : (
						<ChevronRightIcon className="h-3 w-3 shrink-0" />
					)
				) : (
					<span className="w-3" />
				)}
				<Icon className="h-3 w-3 text-slate-500 shrink-0" />
				<span className="truncate">{node.label}</span>
			</div>
			{hasChildren && expanded && (
				<div className="pl-3">
					{node.children.map((child) => (
						<HierarchyNodeItem key={child.id} node={child} />
					))}
				</div>
			)}
		</div>
	)
}

export default MemoryPanel
