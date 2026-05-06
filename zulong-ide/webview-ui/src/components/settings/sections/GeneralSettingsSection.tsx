import { VSCodeCheckbox, VSCodeLink } from "@vscode/webview-ui-toolkit/react"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useExtensionState } from "@/context/ExtensionStateContext"
import PreferredLanguageSetting from "../PreferredLanguageSetting"
import Section from "../Section"
import { updateSetting } from "../utils/settingsHandlers"

interface GeneralSettingsSectionProps {
	renderSectionHeader: (tabId: string) => JSX.Element | null
}

const GeneralSettingsSection = ({ renderSectionHeader }: GeneralSettingsSectionProps) => {
	const { telemetrySetting, remoteConfigSettings } = useExtensionState()

	return (
		<div>
			{renderSectionHeader("general")}
			<Section>
				<PreferredLanguageSetting />

				<div className="mb-[5px]">
					<Tooltip>
						<TooltipContent hidden={remoteConfigSettings?.telemetrySetting === undefined}>
							此设置由您组织的远程配置管理
						</TooltipContent>
						<TooltipTrigger asChild>
							<div className="flex items-center gap-2 mb-[5px]">
								<VSCodeCheckbox
									checked={telemetrySetting !== "disabled"}
									disabled={remoteConfigSettings?.telemetrySetting === "disabled"}
									onChange={(e: any) => {
										const checked = e.target.checked === true
										updateSetting("telemetrySetting", checked ? "enabled" : "disabled")
									}}>
									允许错误和使用报告
								</VSCodeCheckbox>
								{!!remoteConfigSettings?.telemetrySetting && (
									<i className="codicon codicon-lock text-description text-sm" />
								)}
							</div>
						</TooltipTrigger>
					</Tooltip>

					<p className="text-sm mt-[5px] text-description">
						通过发送使用数据和错误报告帮助改进 Zulong。不会发送任何代码、提示词或个人信息。请参阅我们的{" "}
						<VSCodeLink
							className="text-inherit"
							href="https://docs.zulong.ai/more-info/telemetry"
							style={{ fontSize: "inherit", textDecoration: "underline" }}>
							遥测概述
						</VSCodeLink>{" "}
						and{" "}
						<VSCodeLink
							className="text-inherit"
							href="https://zulong.ai/privacy"
							style={{ fontSize: "inherit", textDecoration: "underline" }}>
							隐私政策
						</VSCodeLink>{" "}
						了解更多详情。
					</p>
				</div>
			</Section>
		</div>
	)
}

export default GeneralSettingsSection
