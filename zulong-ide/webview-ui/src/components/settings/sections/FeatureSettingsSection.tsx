import { UpdateSettingsRequest } from "@shared/proto/zulong/state"
import { memo, type ReactNode, useCallback } from "react"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useExtensionState } from "@/context/ExtensionStateContext"
import Section from "../Section"
import SettingsSlider from "../SettingsSlider"
import { updateSetting } from "../utils/settingsHandlers"

// Reusable checkbox component for feature settings
interface FeatureCheckboxProps {
	checked: boolean | undefined
	onChange: (checked: boolean) => void
	label: string
	description: ReactNode
	disabled?: boolean
	isRemoteLocked?: boolean
	remoteTooltip?: string
	isVisible?: boolean
}

// Interface for feature toggle configuration
interface FeatureToggle {
	id: string
	label: string
	description: ReactNode
	settingKey: keyof UpdateSettingsRequest
	stateKey: string
	/** If set, the setting value is nested with this key (e.g., "enabled" -> { enabled: checked }) */
	nestedKey?: string
}

const agentFeatures: FeatureToggle[] = [
	{
		id: "subagents",
		label: "子智能体",
		description: "让 Zulong 并行运行专注的子智能体来为您探索代码库。",
		stateKey: "subagentsEnabled",
		settingKey: "subagentsEnabled",
	},
	{
		id: "native-tool-call",
		label: "原生工具调用",
		description: "可用时使用原生函数调用",
		stateKey: "nativeToolCallSetting",
		settingKey: "nativeToolCallEnabled",
	},
	{
		id: "parallel-tool-calling",
		label: "并行工具调用",
		description: "同时执行多个工具调用",
		stateKey: "enableParallelToolCalling",
		settingKey: "enableParallelToolCalling",
	},
	{
		id: "strict-plan-mode",
		label: "严格计划模式",
		description: "在计划模式下阻止文件编辑",
		stateKey: "strictPlanModeEnabled",
		settingKey: "strictPlanModeEnabled",
	},
	{
		id: "auto-compact",
		label: "自动压缩",
		description: "自动压缩对话历史。",
		stateKey: "useAutoCondense",
		settingKey: "useAutoCondense",
	},
	{
		id: "focus-chain",
		label: "聚焦链",
		description: "跨交互维护上下文聚焦",
		stateKey: "focusChainEnabled",
		settingKey: "focusChainSettings",
		nestedKey: "enabled",
	},
]

const editorFeatures: FeatureToggle[] = [
	{
		id: "show-feature-tips",
		label: "功能提示",
		description: "在思考阶段显示轮播提示，帮助您发现 Zulong 功能。",
		stateKey: "showFeatureTips",
		settingKey: "showFeatureTips",
	},
	{
		id: "background-edit",
		label: "后台编辑",
		description: "允许编辑而不抢占编辑器焦点",
		stateKey: "backgroundEditEnabled",
		settingKey: "backgroundEditEnabled",
	},
	{
		id: "checkpoints",
		label: "检查点",
		description: "在关键点保存进度以便回滚",
		stateKey: "enableCheckpointsSetting",
		settingKey: "enableCheckpointsSetting",
	},
	{
		id: "zulong-web-tools",
		label: "Zulong 网络工具",
		description: "访问网页浏览和搜索功能",
		stateKey: "zulongWebToolsEnabled",
		settingKey: "clineWebToolsEnabled",
	},
	{
		id: "worktrees",
		label: "工作树",
		description: "启用 git worktree 管理以并行运行 Zulong 任务。",
		stateKey: "worktreesEnabled",
		settingKey: "worktreesEnabled",
	},
]

const experimentalFeatures: FeatureToggle[] = [
	{
		id: "yolo",
		label: "极速模式",
		description:
			"无需用户确认即执行任务。自动从计划模式切换到执行模式并禁用提问工具。请谨慎使用。",
		stateKey: "yoloModeToggled",
		settingKey: "yoloModeToggled",
	},
	{
		id: "double-check-completion",
		label: "完成双重检查",
		description:
			"拒绝第一次完成尝试，要求模型对照原始任务需求重新验证后才接受。",
		stateKey: "doubleCheckCompletionEnabled",
		settingKey: "doubleCheckCompletionEnabled",
	},
	{
		id: "lazy-teammate",
		label: "摸鱼模式",
		description: "有时候 Zulong 今天就是不太想干活。仅供娱乐。",
		stateKey: "lazyTeammateModeEnabled",
		settingKey: "lazyTeammateModeEnabled",
	},
]

const advancedFeatures: FeatureToggle[] = [
	{
		id: "hooks",
		label: "钩子",
		description: "在任务执行期间启用生命周期和工具钩子。",
		stateKey: "hooksEnabled",
		settingKey: "hooksEnabled",
	},
]

const FeatureRow = memo(
	({
		checked = false,
		onChange,
		label,
		description,
		disabled,
		isRemoteLocked,
		isVisible = true,
		remoteTooltip,
	}: FeatureCheckboxProps) => {
		if (!isVisible) {
			return null
		}

		const checkbox = (
			<div className="flex items-center justify-between w-full">
				<div>{label}</div>
				<div>
					<Switch
						checked={checked}
						className="shrink-0"
						disabled={disabled || isRemoteLocked}
						id={label}
						onCheckedChange={onChange}
						size="lg"
					/>
					{isRemoteLocked && <i className="codicon codicon-lock text-description text-sm" />}
				</div>
			</div>
		)

		return (
			<div className="flex flex-col items-start justify-between gap-4 py-3 w-full">
				<div className="space-y-0.5 flex-1 w-full">
					{isRemoteLocked ? (
						<Tooltip>
							<TooltipTrigger asChild>{checkbox}</TooltipTrigger>
							<TooltipContent className="max-w-xs" side="top">
								{remoteTooltip}
							</TooltipContent>
						</Tooltip>
					) : (
						checkbox
					)}
				</div>
				<div className="text-xs text-description">{description}</div>
			</div>
		)
	},
)

interface FeatureSettingsSectionProps {
	renderSectionHeader: (tabId: string) => JSX.Element | null
}

const FeatureSettingsSection = ({ renderSectionHeader }: FeatureSettingsSectionProps) => {
	const {
		enableCheckpointsSetting,
		hooksEnabled,
		mcpDisplayMode,
		strictPlanModeEnabled,
		yoloModeToggled,
		useAutoCondense,
		subagentsEnabled,
		zulongWebToolsEnabled,
		worktreesEnabled,
		focusChainSettings,
		remoteConfigSettings,
		nativeToolCallSetting,
		enableParallelToolCalling,
		backgroundEditEnabled,
		doubleCheckCompletionEnabled,
		lazyTeammateModeEnabled,
		showFeatureTips,
	} = useExtensionState()

	const handleFocusChainIntervalChange = useCallback(
		(value: number) => {
			updateSetting("focusChainSettings", { ...focusChainSettings, remindZulongInterval: value })
		},
		[focusChainSettings],
	)

	const isYoloRemoteLocked = remoteConfigSettings?.yoloModeToggled !== undefined

	// State lookup for mapped features
	const featureState: Record<string, boolean | undefined> = {
		showFeatureTips,
		enableCheckpointsSetting,
		strictPlanModeEnabled,
		hooksEnabled,
		nativeToolCallSetting,
		focusChainEnabled: focusChainSettings?.enabled,
		useAutoCondense,
		subagentsEnabled,
		zulongWebToolsEnabled: zulongWebToolsEnabled?.user,
		worktreesEnabled: worktreesEnabled?.user,
		enableParallelToolCalling,
		backgroundEditEnabled,
		doubleCheckCompletionEnabled,
		lazyTeammateModeEnabled,
		yoloModeToggled: isYoloRemoteLocked ? remoteConfigSettings?.yoloModeToggled : yoloModeToggled,
	}

	// Visibility lookup for features with feature flags
	const featureVisibility: Record<string, boolean | undefined> = {
		zulongWebToolsEnabled: zulongWebToolsEnabled?.featureFlag,
		worktreesEnabled: worktreesEnabled?.featureFlag,
	}

	// Handler for feature toggle changes, supports nested settings like focusChainSettings
	const handleFeatureChange = useCallback(
		(feature: FeatureToggle, checked: boolean) => {
			if (feature.nestedKey) {
				// For nested settings, spread the existing value and set the nested key
				let currentValue = {}
				if (feature.settingKey === "focusChainSettings") {
					currentValue = focusChainSettings ?? {}
				}
				updateSetting(feature.settingKey, { ...currentValue, [feature.nestedKey]: checked })
			} else {
				updateSetting(feature.settingKey, checked)
			}
		},
		[focusChainSettings],
	)

	return (
		<div className="mb-2">
			{renderSectionHeader("features")}
			<Section>
				<div className="mb-5 flex flex-col gap-3">
					{/* Core features */}
					<div>
						<div className="text-xs font-medium text-foreground/80 uppercase tracking-wider mb-3">智能体</div>
						<div
							className="relative p-3 pt-0 my-3 rounded-md border border-editor-widget-border/50"
							id="agent-features">
							{agentFeatures.map((feature) => (
								<div key={feature.id}>
									<FeatureRow
										checked={featureState[feature.stateKey]}
										description={feature.description}
										isVisible={featureVisibility[feature.stateKey] ?? true}
										key={feature.id}
										label={feature.label}
										onChange={(checked) =>
											feature.nestedKey === "enabled"
												? handleFeatureChange(feature, checked)
												: updateSetting(feature.settingKey, checked)
										}
									/>
									{feature.id === "focus-chain" && featureState[feature.stateKey] && (
										<SettingsSlider
											label="提醒间隔 (1-10)"
											max={10}
											min={1}
											onChange={handleFocusChainIntervalChange}
											step={1}
											value={focusChainSettings?.remindZulongInterval || 6}
											valueWidth="w-6"
										/>
									)}
								</div>
							))}
						</div>
					</div>

					{/* Editor features */}
					<div>
						<div className="text-xs font-medium text-foreground/80 uppercase tracking-wider mb-3">编辑器</div>
						<div
							className="relative p-3 pt-0 my-3 rounded-md border border-editor-widget-border/50"
							id="optional-features">
							{editorFeatures.map((feature) => (
								<FeatureRow
									checked={featureState[feature.stateKey]}
									description={feature.description}
									isVisible={featureVisibility[feature.stateKey] ?? true}
									key={feature.id}
									label={feature.label}
									onChange={(checked) => handleFeatureChange(feature, checked)}
								/>
							))}
						</div>
					</div>

					{/* Experimental features */}
					<div>
						<div className="text-xs font-medium uppercase tracking-wider mb-3 text-warning/80">实验性</div>
						<div
							className="relative p-3 pt-0 my-3 rounded-md border border-editor-widget-border/50 w-full"
							id="experimental-features">
							{experimentalFeatures.map((feature) => (
								<FeatureRow
									checked={featureState[feature.stateKey]}
									description={feature.description}
									disabled={feature.id === "yolo" && isYoloRemoteLocked}
									isRemoteLocked={feature.id === "yolo" && isYoloRemoteLocked}
									isVisible={featureVisibility[feature.stateKey] ?? true}
									key={feature.id}
									label={feature.label}
									onChange={(checked) => handleFeatureChange(feature, checked)}
									remoteTooltip="此设置由您组织的远程配置管理"
								/>
							))}
						</div>
					</div>
				</div>

				{/* Advanced */}
				<div>
					<div className="text-xs font-medium text-foreground/80 uppercase tracking-wider mb-3">高级</div>
					<div className="relative p-3 my-3 rounded-md border border-editor-widget-border/50" id="advanced-features">
						<div className="space-y-3">
							{advancedFeatures.map((feature) => (
								<FeatureRow
									checked={featureState[feature.stateKey]}
									description={feature.description}
									isVisible={featureVisibility[feature.stateKey] ?? true}
									key={feature.id}
									label={feature.label}
									onChange={(checked) => handleFeatureChange(feature, checked)}
								/>
							))}

							{/* MCP Display Mode */}
							<div className="space-y-2">
								<Label className="text-sm font-medium text-foreground">MCP 显示模式</Label>
								<p className="text-xs text-muted-foreground">控制 MCP 响应的显示方式</p>
								<Select onValueChange={(v) => updateSetting("mcpDisplayMode", v)} value={mcpDisplayMode}>
									<SelectTrigger className="w-full">
										<SelectValue />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="plain">纯文本</SelectItem>
										<SelectItem value="rich">富文本显示</SelectItem>
										<SelectItem value="markdown">Markdown</SelectItem>
									</SelectContent>
								</Select>
							</div>
						</div>
					</div>
				</div>
			</Section>
		</div>
	)
}
export default memo(FeatureSettingsSection)
