import { Mode } from "@shared/storage/types"
import { VSCodeDropdown, VSCodeOption } from "@vscode/webview-ui-toolkit/react"
import { updateAutoApproveSettings } from "@/components/chat/auto-approve-menu/AutoApproveSettingsAPI"
import { useExtensionState } from "@/context/ExtensionStateContext"
import { DebouncedTextField } from "../common/DebouncedTextField"
import { useApiConfigurationHandlers } from "../utils/useApiConfigurationHandlers"

/**
 * Props for the ZulongProvider component
 */
interface ZulongProviderProps {
	showModelOptions: boolean
	isPopup?: boolean
	currentMode: Mode
}

/**
 * The Zulong provider configuration component.
 * Zulong uses a WebSocket connection to a local backend server —
 * the only configurable field is the server URL.
 */
export const ZulongProvider = ({ showModelOptions, isPopup, currentMode }: ZulongProviderProps) => {
	const { apiConfiguration, autoApprovalSettings } = useExtensionState()
	const { handleFieldChange } = useApiConfigurationHandlers()

	const zulongAutoApproveMode = autoApprovalSettings.zulongAutoApproveMode ?? "full"

	return (
		<div className="flex flex-col gap-2">
			<DebouncedTextField
				initialValue={apiConfiguration?.zulongServerUrl || "ws://127.0.0.1:8090"}
				onChange={(value) => handleFieldChange("zulongServerUrl", value || undefined)}
				placeholder="Default: ws://127.0.0.1:8090"
				style={{ width: "100%" }}>
				<span className="font-semibold">Server URL</span>
			</DebouncedTextField>
			<p className="text-xs mt-0 text-description">
				Zulong WebSocket server address. Make sure the Zulong backend is running before starting a task.
			</p>

			<div className="flex flex-col gap-1 mt-2">
				<span className="font-semibold">Auto-approve Mode</span>
				<VSCodeDropdown
					value={zulongAutoApproveMode}
					onChange={(e: any) => {
						const value = e.target.value as "full" | "read_only" | "off"
						updateAutoApproveSettings({
							...autoApprovalSettings,
							version: (autoApprovalSettings.version ?? 1) + 1,
							zulongAutoApproveMode: value,
						})
					}}
					style={{ width: "100%" }}>
					<VSCodeOption value="full">Full (auto-approve all tools)</VSCodeOption>
					<VSCodeOption value="read_only">Read-only (only read operations)</VSCodeOption>
					<VSCodeOption value="off">Off (use standard settings)</VSCodeOption>
				</VSCodeDropdown>
				<p className="text-xs mt-0 text-description">
					Controls whether Zulong backend tool calls are auto-approved. The backend CircuitBreaker provides
					safety guarantees, so "Full" mode is recommended.
				</p>
			</div>
		</div>
	)
}
