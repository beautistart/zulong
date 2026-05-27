import { memo, useState, useCallback, useEffect, useRef } from "react"
import {
	ShieldIcon,
	ShieldCheckIcon,
	ShieldXIcon,
	ShieldAlertIcon,
	AlertTriangleIcon,
	XIcon,
	CheckIcon,
	PlusIcon,
	FileCodeIcon,
	FolderIcon,
	TerminalIcon,
	ExternalLinkIcon,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ApprovalMode } from "@shared/ApprovalWhitelist"
import type { InteractionPayload } from "@shared/ExtensionMessage"

interface ApprovalCardProps {
	interaction: InteractionPayload
	className?: string
	onApprove: (interactionId: string, addToWhitelist?: string) => void
	onReject: (interactionId: string) => void
	onViewDiff?: (path: string) => void
}

const riskConfig: Record<string, {
	icon: typeof ShieldCheckIcon
	label: string
	color: string
	bg: string
	border: string
}> = {
	LOW: {
		icon: ShieldCheckIcon,
		label: "低风险",
		color: "text-green-600 dark:text-green-400",
		bg: "bg-green-500/5",
		border: "border-green-500/20",
	},
	MEDIUM: {
		icon: ShieldIcon,
		label: "中风险",
		color: "text-amber-600 dark:text-amber-400",
		bg: "bg-amber-500/5",
		border: "border-amber-500/20",
	},
	HIGH: {
		icon: ShieldAlertIcon,
		label: "高风险",
		color: "text-orange-600 dark:text-orange-400",
		bg: "bg-orange-500/5",
		border: "border-orange-500/20",
	},
	CRITICAL: {
		icon: ShieldXIcon,
		label: "严重风险",
		color: "text-red-600 dark:text-red-400",
		bg: "bg-red-500/5",
		border: "border-red-500/20",
	},
}

const modeLabel: Record<ApprovalMode, string> = {
	full_auto: "完全权限",
	whitelist: "白名单自动放行",
	manual: "手动审批",
	popup: "高风险弹窗",
}

/**
 * 多模式审批卡片 (TSD 23.4.4)
 *
 * 支持四种审批模式:
 * - 完全权限: 静默执行（不显示卡片）
 * - 白名单放行: 显示"已自动批准"标记
 * - 手动审批: 显示批准/拒绝按钮
 * - 高风险弹窗: 模态弹窗 + 需明确确认
 */
const ApprovalCard = memo(function ApprovalCard({
	interaction,
	className,
	onApprove,
	onReject,
	onViewDiff,
}: ApprovalCardProps) {
	const [showWhitelistOptions, setShowWhitelistOptions] = useState(false)
	const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
	const risk = interaction.risk_level || "LOW"
	const config = riskConfig[risk] || riskConfig.LOW
	const Icon = config.icon
	const mode = interaction.approval_mode as ApprovalMode | undefined
	const isCritical = risk === "CRITICAL"
	const isPopup = mode === "popup" || isCritical
	const isFullAuto = mode === "full_auto"

	// TSD §23.4.5 步骤4: 60s 超时自动拒绝
	useEffect(() => {
		if (interaction.status !== "awaiting_approval") return
		if (isFullAuto || mode === "whitelist") return

		timeoutRef.current = setTimeout(() => {
			onReject(interaction.interaction_id)
		}, 60_000)

		return () => {
			if (timeoutRef.current) clearTimeout(timeoutRef.current)
		}
	}, [interaction.status, interaction.interaction_id, isFullAuto, mode, onReject])

	// 完全权限模式: 不渲染卡片 (TSD §23.4.2)
	if (isFullAuto) {
		return null
	}

	// 已自动批准（白名单模式）
	if (mode === "whitelist" && interaction.status === "approved") {
		return (
			<div
				className={cn(
					"rounded-lg border border-green-500/20 bg-green-500/5 p-3",
					"flex items-center gap-2 text-xs text-green-600 dark:text-green-400",
					className,
				)}
			>
				<ShieldCheckIcon className="h-3.5 w-3.5" />
				<span>
					{interaction.tool_name} — 已自动批准 (白名单)
				</span>
			</div>
		)
	}

	const handleApproveWithWhitelist = useCallback(
		(addToWhitelist?: string) => {
			onApprove(interaction.interaction_id, addToWhitelist)
		},
		[interaction.interaction_id, onApprove],
	)

	const handleReject = useCallback(() => {
		onReject(interaction.interaction_id)
	}, [interaction.interaction_id, onReject])

	return (
		<div
			className={cn(
				"rounded-lg border p-4 space-y-3",
				config.bg,
				config.border,
				isPopup && "ring-2 ring-red-500/30",
				className,
			)}
		>
			{/* 标题行: 图标 + 模式 + 风险等级 */}
			<div className="flex items-center justify-between text-sm">
				<div className={cn("flex items-center gap-2 font-medium", config.color)}>
					<Icon className="h-4 w-4" />
					{isPopup ? "⚠️ 高风险操作确认" : "🛡️ 需要确认"}
				</div>
				<div className="flex items-center gap-2">
					{mode && (
						<span className="text-xs text-muted-foreground">
							{modeLabel[mode]}
						</span>
					)}
					<span
						className={cn(
							"inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium",
							config.bg,
							config.color,
						)}
					>
						<AlertTriangleIcon className="h-3 w-3" />
						风险: {risk}
					</span>
				</div>
			</div>

			{/* 工具信息 */}
			{interaction.tool_name && (
				<div className="space-y-1">
					<div className="flex items-center gap-2 text-sm">
						{interaction.tool_name === "write_to_file" ||
						interaction.tool_name === "replace_in_file" ? (
							<FileCodeIcon className="h-4 w-4 text-muted-foreground" />
						) : interaction.tool_name === "execute_command" ? (
							<TerminalIcon className="h-4 w-4 text-muted-foreground" />
						) : interaction.tool_name === "delete_file" ? (
							<AlertTriangleIcon className="h-4 w-4 text-red-500" />
						) : (
							<FolderIcon className="h-4 w-4 text-muted-foreground" />
						)}
						<span className="font-mono text-sm font-medium">
							{interaction.tool_name}
						</span>
					</div>

					{/* 路径 (如果是文件操作) */}
					{interaction.tool_args?.path && (
						<div className="flex items-center gap-2 text-xs text-muted-foreground pl-6">
							<span className="font-mono">
								{interaction.tool_args.path as string}
							</span>
							{onViewDiff &&
								(interaction.tool_name === "write_to_file" ||
									interaction.tool_name === "replace_in_file") && (
									<button
										type="button"
										onClick={() =>
											onViewDiff(interaction.tool_args!.path as string)
										}
										className="inline-flex items-center gap-0.5 text-blue-500 hover:text-blue-600"
									>
										<ExternalLinkIcon className="h-3 w-3" />
										查看 Diff
									</button>
								)}
						</div>
					)}

					{/* 命令 (如果是命令执行) */}
					{interaction.tool_args?.command && (
						<div className="pl-6">
							<code className="text-xs font-mono bg-muted/50 rounded px-1 py-0.5 max-w-full overflow-x-auto block whitespace-pre-wrap">
								{interaction.tool_args.command as string}
							</code>
						</div>
					)}
				</div>
			)}

			{/* 风险说明 */}
			{interaction.risk_reason && (
				<div className="flex items-start gap-2 text-xs text-muted-foreground">
					<AlertTriangleIcon
						className={cn(
							"h-3.5 w-3.5 mt-0.5 shrink-0",
							isCritical ? "text-red-500" : "text-amber-500",
						)}
					/>
					<span>{interaction.risk_reason}</span>
				</div>
			)}

			{/* 严重风险警告 */}
			{isCritical && (
				<div className="space-y-1 rounded bg-red-500/10 border border-red-500/20 p-2 text-xs">
					<p className="text-red-600 dark:text-red-400 font-medium">
						🔴 此操作不可逆！
					</p>
					{interaction.risk_reason && (
						<p className="text-red-500/80">
							影响范围: {interaction.risk_reason}
						</p>
					)}
				</div>
			)}

			{/* 操作按钮 */}
			<div className="flex items-center gap-2 pt-1">
				{onViewDiff &&
					(interaction.tool_name === "write_to_file" ||
						interaction.tool_name === "replace_in_file") && (
						<Button
							variant="outline"
							size="sm"
							onClick={() =>
								onViewDiff(
									(interaction.tool_args?.path as string) || "",
								)
							}
						>
							<ExternalLinkIcon className="h-3.5 w-3.5 mr-1" />
							查看 Diff
						</Button>
					)}

				<div className="flex-1" />

				{isCritical ? (
					<>
						<Button variant="outline" size="sm" onClick={handleReject}>
							<XIcon className="h-3.5 w-3.5 mr-1" />
							取消
						</Button>
						<Button
							variant="danger"
							size="sm"
							onClick={() => handleApproveWithWhitelist()}
						>
							<CheckIcon className="h-3.5 w-3.5 mr-1" />
							我确认执行此操作
						</Button>
					</>
				) : (
					<>
						<Button variant="outline" size="sm" onClick={handleReject}>
							<XIcon className="h-3.5 w-3.5 mr-1" />
							拒绝
						</Button>
						<Button
							variant="default"
							size="sm"
							onClick={() => handleApproveWithWhitelist()}
						>
							<CheckIcon className="h-3.5 w-3.5 mr-1" />
							批准
						</Button>
					</>
				)}

				{/* 加入白名单快捷操作 */}
				{interaction.tool_args?.path && (
					<div className="relative">
						<Button
							variant="ghost"
							size="sm"
							onClick={() =>
								setShowWhitelistOptions(!showWhitelistOptions)
							}
						>
							<PlusIcon className="h-3.5 w-3.5 mr-1" />
							加入白名单
						</Button>
						{showWhitelistOptions && (
							<div className="absolute bottom-full right-0 mb-1 rounded border bg-popover p-1 shadow-md z-50 min-w-[160px]">
								<button
									type="button"
									className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-xs hover:bg-accent"
									onClick={() => {
										handleApproveWithWhitelist(
											`dir:${interaction.tool_args!.path}`,
										)
										setShowWhitelistOptions(false)
									}}
								>
									<FolderIcon className="h-3 w-3" />
									目录: {interaction.tool_args.path as string}
								</button>
								<button
									type="button"
									className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-xs hover:bg-accent"
									onClick={() => {
										handleApproveWithWhitelist(
											`tool:${interaction.tool_name}`,
										)
										setShowWhitelistOptions(false)
									}}
								>
									<ShieldCheckIcon className="h-3 w-3" />
									工具: {interaction.tool_name}
								</button>
							</div>
						)}
					</div>
				)}
			</div>
		</div>
	)
})

export default ApprovalCard
