import { memo } from "react"
import {
	CheckCircleIcon,
	AlertTriangleIcon,
	ArrowRightIcon,
	BrainIcon,
	ShieldCheckIcon,
	PlusIcon,
	TrendingUpIcon,
	Trash2Icon,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { InteractionPayload } from "@shared/ExtensionMessage"

interface SummaryCardProps {
	interaction: InteractionPayload
	className?: string
}

/**
 * 结束总结卡片 (TSD 23.3.2 第三层)
 *
 * 在任务完成时展示:
 * - 完成项
 * - 验证项
 * - 风险/待处理
 * - 下一步建议
 * - 记忆变化
 */
const SummaryCard = memo(function SummaryCard({
	interaction,
	className,
}: SummaryCardProps) {
	const time = interaction.timestamp
		? new Date(interaction.timestamp * 1000).toLocaleTimeString("zh-CN", {
				hour: "2-digit",
				minute: "2-digit",
				second: "2-digit",
			})
		: null

	const completed = interaction.completed_items ?? []
	const verified = interaction.verified_items ?? []
	const pending = interaction.pending_items ?? []
	const mem = interaction.memory_changes

	const hasMemoryChanges =
		mem && (mem.created > 0 || mem.strengthened > 0 || mem.pruned > 0)

	return (
		<div
			className={cn(
				"rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4 space-y-3",
				"animate-in fade-in slide-in-from-bottom-2 duration-300",
				className,
			)}
		>
			{/* 标题行 */}
			<div className="flex items-center justify-between text-sm">
				<div className="flex items-center gap-2 text-emerald-600 dark:text-emerald-400 font-medium">
					<CheckCircleIcon className="h-4 w-4" />
					任务完成
				</div>
				{time && (
					<span className="text-muted-foreground text-xs tabular-nums">
						{time}
					</span>
				)}
			</div>

			{/* 标题 */}
			{interaction.title && (
				<div className="text-sm font-medium">{interaction.title}</div>
			)}

			{/* 已完成项 */}
			{completed.length > 0 && (
				<div className="space-y-1">
					<div className="text-xs font-medium text-muted-foreground">
						已完成
					</div>
					<ul className="space-y-0.5">
						{completed.map((item: string, i: number) => (
							<li
								key={i}
								className="flex items-start gap-1.5 text-xs text-muted-foreground"
							>
								<CheckCircleIcon className="h-3.5 w-3.5 mt-0.5 text-emerald-500 shrink-0" />
								<span>{item}</span>
							</li>
						))}
					</ul>
				</div>
			)}

			{/* 已验证项 */}
			{verified.length > 0 && (
				<div className="space-y-1 border-t border-emerald-500/10 pt-2">
					<div className="text-xs font-medium text-muted-foreground">
						已验证
					</div>
					<ul className="space-y-0.5">
						{verified.map((item: string, i: number) => (
							<li
								key={i}
								className="flex items-start gap-1.5 text-xs text-muted-foreground"
							>
								<ShieldCheckIcon className="h-3.5 w-3.5 mt-0.5 text-blue-500 shrink-0" />
								<span>{item}</span>
							</li>
						))}
					</ul>
				</div>
			)}

			{/* 风险/待处理项 */}
			{pending.length > 0 && (
				<div className="space-y-1 border-t border-emerald-500/10 pt-2">
					<div className="text-xs font-medium text-muted-foreground">
						待处理
					</div>
					<ul className="space-y-0.5">
						{pending.map((item: string, i: number) => (
							<li
								key={i}
								className="flex items-start gap-1.5 text-xs text-muted-foreground"
							>
								<AlertTriangleIcon className="h-3.5 w-3.5 mt-0.5 text-amber-500 shrink-0" />
								<span>{item}</span>
							</li>
						))}
					</ul>
				</div>
			)}

			{/* 风险摘要 */}
			{interaction.risks_summary && (
				<div className="flex items-start gap-2 text-xs border-t border-emerald-500/10 pt-2">
					<AlertTriangleIcon className="h-3.5 w-3.5 mt-0.5 text-amber-500 shrink-0" />
					<div>
						<span className="font-medium text-amber-600 dark:text-amber-400">
							风险:
						</span>{" "}
						<span className="text-muted-foreground">
							{interaction.risks_summary}
						</span>
					</div>
				</div>
			)}

			{/* 下一步 */}
			{interaction.next_step && (
				<div className="flex items-center gap-2 text-xs border-t border-emerald-500/10 pt-2">
					<ArrowRightIcon className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
					<span className="text-emerald-600 dark:text-emerald-400 font-medium">
						下一步:
					</span>
					<span className="text-muted-foreground">
						{interaction.next_step}
					</span>
				</div>
			)}

			{/* 记忆变化 */}
			{hasMemoryChanges && (
				<div className="flex items-center gap-2 text-xs border-t border-purple-500/10 pt-2">
					<BrainIcon className="h-3.5 w-3.5 text-purple-500 shrink-0" />
					<span className="text-muted-foreground">记忆更新:</span>
					<div className="flex items-center gap-2">
						{mem!.created > 0 && (
							<span className="inline-flex items-center gap-1 text-purple-600 dark:text-purple-400">
								<PlusIcon className="h-3 w-3" />
								新增 {mem!.created}
							</span>
						)}
						{mem!.strengthened > 0 && (
							<span className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400">
								<TrendingUpIcon className="h-3 w-3" />
								强化 {mem!.strengthened}
							</span>
						)}
						{mem!.pruned > 0 && (
							<span className="inline-flex items-center gap-1 text-gray-500">
								<Trash2Icon className="h-3 w-3" />
								淘汰 {mem!.pruned}
							</span>
						)}
					</div>
				</div>
			)}

			{/* 进度条 */}
			{interaction.progress != null && (
				<div className="space-y-1">
					<div className="h-1.5 rounded-full bg-emerald-500/10 overflow-hidden">
						<div
							className="h-full rounded-full bg-emerald-500 transition-all duration-500"
							style={{
								width: `${Math.min(100, Math.max(0, interaction.progress))}%`,
							}}
						/>
					</div>
				</div>
			)}
		</div>
	)
})

export default SummaryCard
