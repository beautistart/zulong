import { resolveWorkspacePath } from "@core/workspace"
import { isMultiRootEnabled } from "@core/workspace/multi-root-utils"
import { ZulongDefaultTool } from "@shared/tools"
import { StateManager } from "@/core/storage/StateManager"
import { HostProvider } from "@/hosts/host-provider"
import { getCwd, getDesktopDir, isLocatedInPath, isLocatedInWorkspace } from "@/utils/path"

export class AutoApprove {
	private stateManager: StateManager
	// Cache for workspace paths - populated on first access and reused for the task lifetime
	// NOTE: This assumes that the task has a fixed set of workspace roots(which is currently true).
	private workspacePathsCache: { paths: string[] } | null = null
	private isMultiRootScenarioCache: boolean | null = null

	constructor(stateManager: StateManager) {
		this.stateManager = stateManager
	}

	/**
	 * Check if the current provider is Zulong and the tool should be auto-approved
	 * based on the zulongAutoApproveMode setting.
	 */
	private isZulongAutoApproved(toolName: ZulongDefaultTool): boolean {
		const apiConfig = this.stateManager.getApiConfiguration()
		if (apiConfig.planModeApiProvider !== "zulong" && apiConfig.actModeApiProvider !== "zulong") return false

		const settings = this.stateManager.getGlobalSettingsKey("autoApprovalSettings")
		const mode = settings.zulongAutoApproveMode ?? "full"

		if (mode === "full") return true
		if (mode === "read_only") {
			return [ZulongDefaultTool.FILE_READ, ZulongDefaultTool.LIST_FILES, ZulongDefaultTool.LIST_CODE_DEF, ZulongDefaultTool.SEARCH].includes(
				toolName,
			)
		}
		return false // mode === "off"
	}

	/**
	 * Get workspace information with caching to avoid repeated API calls
	 * Cache is task-scoped since each task gets a new AutoApprove instance
	 */
	private async getWorkspaceInfo(): Promise<{
		workspacePaths: { paths: string[] }
		isMultiRootScenario: boolean
	}> {
		// Check if we already have cached values
		if (this.workspacePathsCache === null || this.isMultiRootScenarioCache === null) {
			// First time - fetch and cache for the lifetime of this task
			this.workspacePathsCache = await HostProvider.workspace.getWorkspacePaths({})
			this.isMultiRootScenarioCache = isMultiRootEnabled(this.stateManager) && this.workspacePathsCache.paths.length > 1
		}

		return {
			workspacePaths: this.workspacePathsCache,
			isMultiRootScenario: this.isMultiRootScenarioCache,
		}
	}

	// Check if the tool should be auto-approved based on the settings
	// Returns bool for most tools, and tuple for tools with nested settings
	shouldAutoApproveTool(toolName: ZulongDefaultTool): boolean | [boolean, boolean] {
		// Zulong provider fast path: bypass standard approval when Zulong backend is active
		if (this.isZulongAutoApproved(toolName)) {
			return [true, true]
		}

		if (this.stateManager.getGlobalSettingsKey("yoloModeToggled")) {
			switch (toolName) {
				case ZulongDefaultTool.FILE_READ:
				case ZulongDefaultTool.LIST_FILES:
				case ZulongDefaultTool.LIST_CODE_DEF:
				case ZulongDefaultTool.SEARCH:
				case ZulongDefaultTool.NEW_RULE:
				case ZulongDefaultTool.FILE_NEW:
				case ZulongDefaultTool.FILE_EDIT:
				case ZulongDefaultTool.APPLY_PATCH:
				case ZulongDefaultTool.BASH:
				case ZulongDefaultTool.USE_SUBAGENTS:
					return [true, true]

				case ZulongDefaultTool.BROWSER:
				case ZulongDefaultTool.WEB_FETCH:
				case ZulongDefaultTool.WEB_SEARCH:
				case ZulongDefaultTool.MCP_ACCESS:
				case ZulongDefaultTool.MCP_USE:
					return true
			}
		}

		if (this.stateManager.getGlobalSettingsKey("autoApproveAllToggled")) {
			switch (toolName) {
				case ZulongDefaultTool.FILE_READ:
				case ZulongDefaultTool.LIST_FILES:
				case ZulongDefaultTool.LIST_CODE_DEF:
				case ZulongDefaultTool.SEARCH:
				case ZulongDefaultTool.NEW_RULE:
				case ZulongDefaultTool.FILE_NEW:
				case ZulongDefaultTool.FILE_EDIT:
				case ZulongDefaultTool.APPLY_PATCH:
				case ZulongDefaultTool.BASH:
				case ZulongDefaultTool.USE_SUBAGENTS:
					return [true, true]
				case ZulongDefaultTool.BROWSER:
				case ZulongDefaultTool.WEB_FETCH:
				case ZulongDefaultTool.WEB_SEARCH:
				case ZulongDefaultTool.MCP_ACCESS:
				case ZulongDefaultTool.MCP_USE:
					return true
			}
		}

		const autoApprovalSettings = this.stateManager.getGlobalSettingsKey("autoApprovalSettings")

		switch (toolName) {
			case ZulongDefaultTool.FILE_READ:
			case ZulongDefaultTool.LIST_FILES:
			case ZulongDefaultTool.LIST_CODE_DEF:
			case ZulongDefaultTool.SEARCH:
			case ZulongDefaultTool.USE_SUBAGENTS:
				return [autoApprovalSettings.actions.readFiles, autoApprovalSettings.actions.readFilesExternally ?? false]
			case ZulongDefaultTool.NEW_RULE:
			case ZulongDefaultTool.FILE_NEW:
			case ZulongDefaultTool.FILE_EDIT:
			case ZulongDefaultTool.APPLY_PATCH:
				return [autoApprovalSettings.actions.editFiles, autoApprovalSettings.actions.editFilesExternally ?? false]
			case ZulongDefaultTool.BASH:
				return [
					autoApprovalSettings.actions.executeSafeCommands ?? false,
					autoApprovalSettings.actions.executeAllCommands ?? false,
				]
			case ZulongDefaultTool.BROWSER:
				return autoApprovalSettings.actions.useBrowser
			case ZulongDefaultTool.WEB_FETCH:
			case ZulongDefaultTool.WEB_SEARCH:
				return autoApprovalSettings.actions.useBrowser
			case ZulongDefaultTool.MCP_ACCESS:
			case ZulongDefaultTool.MCP_USE:
				return autoApprovalSettings.actions.useMcp
		}
		return false
	}

	// Check if the tool should be auto-approved based on the settings
	// and the path of the action. Returns true if the tool should be auto-approved
	// based on the user's settings and the path of the action.
	async shouldAutoApproveToolWithPath(
		blockname: ZulongDefaultTool,
		autoApproveActionpath: string | undefined,
	): Promise<boolean> {
		// Zulong provider fast path: bypass standard approval when Zulong backend is active
		if (this.isZulongAutoApproved(blockname)) {
			return true
		}

		if (this.stateManager.getGlobalSettingsKey("yoloModeToggled")) {
			return true
		}
		if (this.stateManager.getGlobalSettingsKey("autoApproveAllToggled")) {
			return true
		}

		let isLocalRead = false
		if (autoApproveActionpath) {
			// Use cached workspace info instead of fetching every time
			const { isMultiRootScenario } = await this.getWorkspaceInfo()

			if (isMultiRootScenario) {
				// Multi-root: check if file is in ANY workspace
				isLocalRead = await isLocatedInWorkspace(autoApproveActionpath)
			} else {
				// Single-root: use existing logic
				const cwd = await getCwd(getDesktopDir())
				// When called with a string cwd, resolveWorkspacePath returns a string
				const absolutePath = resolveWorkspacePath(
					cwd,
					autoApproveActionpath,
					"AutoApprove.shouldAutoApproveToolWithPath",
				) as string
				isLocalRead = isLocatedInPath(cwd, absolutePath)
			}
		} else {
			// If we do not get a path for some reason, default to a (safer) false return
			isLocalRead = false
		}

		// Get auto-approve settings for local and external edits
		const autoApproveResult = this.shouldAutoApproveTool(blockname)
		const [autoApproveLocal, autoApproveExternal] = Array.isArray(autoApproveResult)
			? autoApproveResult
			: [autoApproveResult, false]

		if ((isLocalRead && autoApproveLocal) || (!isLocalRead && autoApproveLocal && autoApproveExternal)) {
			return true
		}
		return false
	}
}
