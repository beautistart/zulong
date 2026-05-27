import { memo } from "react"
import {
	CheckCircleIcon,
	XCircleIcon,
	LoaderCircleIcon,
	AlertTriangleIcon,
	InfoIcon,
	ChevronDownIcon,
	ChevronRightIcon,
	ShieldIcon,
	ClockIcon,
	ArrowRightIcon,
	BrainIcon,
	ListChecksIcon,
	MessageSquarePlusIcon,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { InteractionPayload } from "@shared/ExtensionMessage"

export type { InteractionPayload }

interface InteractionCardProps {
	interaction: InteractionPayload
	isExpanded?: boolean
	onToggleExpand?: () => void
	onApprove?: (interactionId: string) => void
	onReject?: (interactionId: string) => void
}

const statusConfig: Record<
	string,
	{ icon: typeof ClockIcon; color: string; bg: string; animate?: boolean }
> = {
	pending: { icon: ClockIcon, color: "text-description", bg: "bg-secondary" },
	running: { icon: LoaderCircleIcon, color: "text-blue-500", bg: "bg-blue-50 dark:bg-blue-950", animate: true },
	awaiting_approval: { icon: ShieldIcon, color: "text-yellow-600", bg: "bg-yellow-50 dark:bg-yellow-950" },
	approved: { icon: CheckCircleIcon, color: "text-green-600", bg: "bg-green-50 dark:bg-green-950" },
	rejected: { icon: XCircleIcon, color: "text-red-600", bg: "bg-red-50 dark:bg-red-950" },
	succeeded: { icon: CheckCircleIcon, color: "text-green-600", bg: "bg-green-50 dark:bg-green-950" },
	failed: { icon: XCircleIcon, color: "text-red-600", bg: "bg-red-50 dark:bg-red-950" },
	blocked: { icon: AlertTriangleIcon, color: "text-orange-600", bg: "bg-orange-50 dark:bg-orange-950" },
	cancelled: { icon: XCircleIcon, color: "text-description", bg: "bg-secondary" },
}

const riskColorMap: Record<NonNullable<InteractionPayload["risk_level"]>, string> = {
	LOW: "text-green-600 bg-green-100 dark:bg-green-900",
	MEDIUM: "text-yellow-600 bg-yellow-100 dark:bg-yellow-900",
	HIGH: "text-orange-600 bg-orange-100 dark:bg-orange-900",
	CRITICAL: "text-red-600 bg-red-100 dark:bg-red-900",
}

const kindIconMap: Record<InteractionPayload["kind"], typeof InfoIcon> = {
	plan: ListChecksIcon,
	action: ArrowRightIcon,
	observation: CheckCircleIcon,
	progress: LoaderCircleIcon,
	approval: ShieldIcon,
	summary: CheckCircleIcon,
	user_interject: MessageSquarePlusIcon,
}

const kindLabelMap: Record<InteractionPayload["kind"], string> = {
	plan: "任务规划",
	action: "执行动作",
	observation: "执行结果",
	progress: "进度更新",
	approval: "等待确认",
	summary: "任务总结",
	user_interject: "中途插入",
}

export const InteractionCard = memo(
	({ interaction, isExpanded = false, onToggleExpand, onApprove, onReject }: InteractionCardProps) => {
		const { kind, status, title, detail, tool_name, risk_level, risk_reason, progress, next_step, thought } = interaction

		const statusCfg = statusConfig[status] || statusConfig.pending
		const StatusIcon = statusCfg.icon
		const KindIcon = kindIconMap[kind] || InfoIcon
		const kindLabel = kindLabelMap[kind] || kind

		return (
			<div className={cn("rounded-md border my-1 overflow-hidden", statusCfg.bg)}>
				<div className="flex items-start gap-2 p-2">
					<div className={cn("flex-shrink-0 mt-0.5", statusCfg.color)}>
						<StatusIcon className={cn("size-4", statusCfg.animate && "animate-spin")} />
					</div>

					<div className="flex-1 min-w-0">
						<div className="flex items-center gap-2 flex-wrap">
							<span className={cn("text-xs font-medium", statusCfg.color)}>{kindLabel}</span>
							{tool_name && (
								<span className="text-xs text-description font-mono bg-secondary px-1.5 py-0.5 rounded">
									{tool_name}
								</span>
							)}
							{risk_level && (
								<span className={cn("text-xs px-1.5 py-0.5 rounded font-medium", riskColorMap[risk_level])}>
									{risk_level}
								</span>
							)}
						</div>

						<div className="text-sm font-medium mt-0.5 leading-snug">{title}</div>

						{thought && (
							<div className="text-xs text-description mt-1 italic leading-relaxed border-l-2 border-description/30 pl-2">
								💡 {thought}
							</div>
						)}

						{progress !== undefined && progress > 0 && (
							<div className="mt-1.5">
								<div className="h-1.5 bg-secondary rounded-full overflow-hidden">
									<div
										className="h-full bg-blue-500 transition-all"
										style={{ width: `${Math.min(progress, 100)}%` }}
									/>
								</div>
							</div>
						)}

						{kind === "approval" && status === "awaiting_approval" && (
							<div className="flex gap-2 mt-2">
								<Button
									size="sm"
									variant="default"
									className="h-7 text-xs"
									onClick={() => onApprove?.(interaction.interaction_id)}>
									批准
								</Button>
								<Button
									size="sm"
									variant="outline"
									className="h-7 text-xs"
									onClick={() => onReject?.(interaction.interaction_id)}>
									拒绝
								</Button>
							</div>
						)}
					</div>

					{(detail || risk_reason || next_step) && (
						<Button
							size="icon"
							variant="ghost"
							className="size-6 p-0 flex-shrink-0"
							onClick={onToggleExpand}>
							{isExpanded ? (
								<ChevronDownIcon className="size-3" />
							) : (
								<ChevronRightIcon className="size-3" />
							)}
						</Button>
					)}
				</div>

				{isExpanded && (detail || risk_reason || next_step) && (
					<div className="px-2 pb-2 pt-0 space-y-1.5 border-t">
						{detail && (
							<div className="text-xs text-description leading-relaxed">
								<span className="font-medium text-foreground">详情：</span>
								{detail}
							</div>
						)}
						{risk_reason && (
							<div className="text-xs text-orange-600 leading-relaxed">
								<span className="font-medium">风险说明：</span>
								{risk_reason}
							</div>
						)}
						{next_step && (
							<div className="text-xs text-blue-600 leading-relaxed">
								<span className="font-medium">下一步：</span>
								{next_step}
							</div>
						)}
					</div>
				)}
			</div>
		)
	}
)

InteractionCard.displayName = "InteractionCard"

export const interactionCardMap: Record<InteractionPayload["kind"], typeof InteractionCard> = {
	plan: InteractionCard,
	action: InteractionCard,
	observation: InteractionCard,
	progress: InteractionCard,
	approval: InteractionCard,
	summary: InteractionCard,
	user_interject: InteractionCard,
}

export function renderInteractionCard(
	interaction: InteractionPayload,
	options?: {
		isExpanded?: boolean
		onToggleExpand?: () => void
		onApprove?: (id: string) => void
		onReject?: (id: string) => void
	}
) {
	const CardComponent = interactionCardMap[interaction.kind] || InteractionCard
	return (
		<CardComponent
			key={interaction.interaction_id}
			interaction={interaction}
			isExpanded={options?.isExpanded}
			onToggleExpand={options?.onToggleExpand}
			onApprove={options?.onApprove}
			onReject={options?.onReject}
		/>
	)
}
