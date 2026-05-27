import { memo } from "react"
import {
	CalendarClockIcon,
	LightbulbIcon,
	PackageIcon,
	ListChecksIcon,
	WrenchIcon,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { InteractionPayload } from "@shared/ExtensionMessage"

interface StartupCardProps {
	interaction: InteractionPayload
	className?: string
}

/**
 * 启动说明卡片 (TSD 23.3.2 第一层)
 *
 * 在任务开始时展示:
 * - 接收时间
 * - 任务分析
 * - 预判工具包
 * - 规划步骤
 */
const StartupCard = memo(function StartupCard({
	interaction,
	className,
}: StartupCardProps) {
	const time = interaction.timestamp
		? new Date(interaction.timestamp * 1000).toLocaleTimeString("zh-CN", {
				hour: "2-digit",
				minute: "2-digit",
				second: "2-digit",
			})
		: null

	return (
		<div
			className={cn(
				"rounded-lg border border-blue-500/30 bg-blue-500/5 p-4 space-y-3",
				"animate-in fade-in slide-in-from-top-2 duration-300",
				className,
			)}
		>
			{/* 标题行: 状态 + 时间 */}
			<div className="flex items-center justify-between text-sm">
				<div className="flex items-center gap-2 text-blue-600 dark:text-blue-400 font-medium">
					<div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
					任务已接收
				</div>
				{time && (
					<span className="text-muted-foreground text-xs tabular-nums">
						{time}
					</span>
				)}
			</div>

			{/* 任务分析 */}
			{interaction.title && (
				<div className="flex items-start gap-2 text-sm">
					<LightbulbIcon className="h-4 w-4 mt-0.5 text-amber-500 shrink-0" />
					<div>
						<span className="text-muted-foreground">任务分析: </span>
						<span className="font-medium">{interaction.title}</span>
					</div>
				</div>
			)}

			{/* 预判工具包 */}
			{interaction.tool_args?.suggested_tools &&
				Array.isArray(interaction.tool_args.suggested_tools) &&
				interaction.tool_args.suggested_tools.length > 0 && (
					<div className="flex items-start gap-2 text-sm">
						<PackageIcon className="h-4 w-4 mt-0.5 text-indigo-500 shrink-0" />
						<div className="flex-1 min-w-0">
							<span className="text-muted-foreground">预判工具包: </span>
							<div className="flex flex-wrap gap-1 mt-1">
								{interaction.tool_args.suggested_tools.map(
									(tool: string) => (
										<span
											key={tool}
											className="inline-flex items-center gap-1 rounded bg-indigo-500/10 
												px-1.5 py-0.5 text-xs text-indigo-600 dark:text-indigo-400 font-mono"
										>
											<WrenchIcon className="h-3 w-3" />
											{tool}
										</span>
									),
								)}
							</div>
						</div>
					</div>
				)}

			{/* 规划步骤 */}
			{interaction.plan_steps &&
				Array.isArray(interaction.plan_steps) &&
				interaction.plan_steps.length > 0 && (
					<div className="flex items-start gap-2 text-sm">
						<ListChecksIcon className="h-4 w-4 mt-0.5 text-emerald-500 shrink-0" />
						<div className="flex-1 min-w-0">
							<span className="text-muted-foreground">规划:</span>
							<ol className="mt-1 space-y-0.5 list-decimal list-inside text-muted-foreground">
								{interaction.plan_steps.map(
									(step: string, i: number) => (
										<li key={i} className="text-xs">
											{step}
										</li>
									),
								)}
							</ol>
						</div>
					</div>
				)}

			{/* 预判理由 */}
			{interaction.detail && (
				<div className="flex items-start gap-2 text-xs text-muted-foreground border-t border-blue-500/10 pt-2">
					<CalendarClockIcon className="h-3.5 w-3.5 mt-0.5 shrink-0" />
					<span>{interaction.detail}</span>
				</div>
			)}
		</div>
	)
})

export default StartupCard
