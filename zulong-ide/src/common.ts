import { WebviewProvider } from "./core/webview"
import "./utils/path" // necessary to have access to String.prototype.toPosix

import fs from "fs/promises"
import path from "path"
import { HostProvider } from "@/hosts/host-provider"
import { Logger } from "@/shared/services/Logger"
import type { StorageContext } from "@/shared/storage/storage-context"
import { FileContextTracker } from "./core/context/context-tracking/FileContextTracker"
import { clearOnboardingModelsCache } from "./core/controller/models/getZulongOnboardingModels"
import { HookDiscoveryCache } from "./core/hooks/HookDiscoveryCache"
import { HookProcessRegistry } from "./core/hooks/HookProcessRegistry"
import { StateManager } from "./core/storage/StateManager"
import { AgentConfigLoader } from "./core/task/tools/subagent/AgentConfigLoader"
import { ExtensionRegistryInfo } from "./registry"
import { ErrorService } from "./services/error"
import { featureFlagsService } from "./services/feature-flags"
import { getDistinctId } from "./services/logging/distinctId"
import { telemetryService } from "./services/telemetry"
import { PostHogClientProvider } from "./services/telemetry/providers/posthog/PostHogClientProvider"
import { ZulongTempManager } from "./services/temp"
import { cleanupTestMode } from "./services/test/TestMode"
import { ShowMessageType } from "./shared/proto/host/window"
import { syncWorker } from "./shared/services/worker/sync"
import { getBlobStoreSettingsFromEnv } from "./shared/services/worker/worker"
import { getLatestAnnouncementId } from "./utils/announcements"
import { arePathsEqual } from "./utils/path"

/**
 * Performs intialization for Zulong that is common to all platforms.
 *
 * @param context
 * @returns The webview provider
 * @throws ZulongConfigurationError if endpoints.json exists but is invalid
 */
export async function initialize(storageContext: StorageContext): Promise<WebviewProvider> {
	// Configure the shared Logging class to use HostProvider's output channels and debug logger
	Logger.subscribe((msg: string) => HostProvider.get().logToChannel(msg)) // File system logging
	Logger.subscribe((msg: string) => HostProvider.env.debugLog({ value: msg })) // Host debug logging

	// Initialize ZulongEndpoint configuration (reads bundled and ~/.zulong/endpoints.json if present)
	// This must be done before any other code that calls ZulongEnv.config()
	// Throws ZulongConfigurationError if config file exists but is invalid
	const { ZulongEndpoint } = await import("./config")
	await ZulongEndpoint.initialize(HostProvider.get().extensionFsPath)

	try {
		await StateManager.initialize(storageContext)
	} catch (error) {
		Logger.error("[Zulong] CRITICAL: Failed to initialize StateManager:", error)
		HostProvider.window.showMessage({
			type: ShowMessageType.ERROR,
			message: "Failed to initialize storage. Please check logs for details or try restarting the client.",
		})
	}

	// =============== External services ===============
	await ErrorService.initialize()
	// Initialize PostHog client provider (skip in self-hosted mode)
	if (!ZulongEndpoint.isSelfHosted()) {
		PostHogClientProvider.getInstance()
	}

	// =============== Webview services ===============
	const webview = HostProvider.get().createWebviewProvider()

	const stateManager = StateManager.get()
	// Non-blocking announcement check and display
	showVersionUpdateAnnouncement(stateManager)
	// Check if this workspace was opened from worktree quick launch
	await checkWorktreeAutoOpen(stateManager)
	// Check if this workspace is a Zulong project pending execution (auto-start)
	checkZulongProjectAutoStart() // non-blocking, uses setTimeout internally

	// =============== Background sync and cleanup tasks ===============
	// Use remote config blobStoreConfig if available, otherwise fall back to env vars
	const blobStoreSettings = stateManager.getRemoteConfigSettings()?.blobStoreConfig ?? getBlobStoreSettingsFromEnv()
	syncWorker().init({ ...blobStoreSettings, userDistinctId: getDistinctId() })
	// Clean up old temp files in background (non-blocking) and start periodic cleanup every 24 hours
	ZulongTempManager.startPeriodicCleanup()
	// Clean up orphaned file context warnings (startup cleanup)
	FileContextTracker.cleanupOrphanedWarnings(stateManager)

	telemetryService.captureExtensionActivated()

	return webview
}

async function showVersionUpdateAnnouncement(stateManager: StateManager) {
	// Version checking for autoupdate notification
	const currentVersion = ExtensionRegistryInfo.version
	const previousVersion = stateManager.getGlobalStateKey("zulongVersion")
	// Perform post-update actions if necessary
	try {
		if (!previousVersion || currentVersion !== previousVersion) {
			Logger.log(`Zulong version changed: ${previousVersion} -> ${currentVersion}. First run or update detected.`)

			// Check if there's a new announcement to show
			const lastShownAnnouncementId = stateManager.getGlobalStateKey("lastShownAnnouncementId")
			const latestAnnouncementId = getLatestAnnouncementId()

			if (lastShownAnnouncementId !== latestAnnouncementId) {
				// Show notification when there's a new announcement (major/minor updates or fresh installs)
				const message = previousVersion
					? `Zulong has been updated to v${currentVersion}`
					: `Welcome to Zulong v${currentVersion}`
				HostProvider.window.showMessage({
					type: ShowMessageType.INFORMATION,
					message,
				})
			}
			// Always update the main version tracker for the next launch.
			await stateManager.setGlobalState("zulongVersion", currentVersion)
		}
	} catch (error) {
		const errorMessage = error instanceof Error ? error.message : String(error)
		Logger.error(`Error during post-update actions: ${errorMessage}, Stack trace: ${error.stack}`)
	}
}

/**
 * Checks if this workspace was opened from the worktree quick launch button.
 * If so, opens the Zulong sidebar and clears the state.
 */
async function checkWorktreeAutoOpen(stateManager: StateManager): Promise<void> {
	try {
		// Read directly from globalState (not StateManager cache) since this may have been
		// set by another window right before this one opened
		const worktreeAutoOpenPath = stateManager.getGlobalStateKey("worktreeAutoOpenPath")
		if (!worktreeAutoOpenPath) {
			return
		}

		// Get current workspace path
		const workspacePaths = (await HostProvider.workspace.getWorkspacePaths({})).paths
		if (workspacePaths.length === 0) {
			return
		}

		const currentPath = workspacePaths[0]

		// Check if current workspace matches the worktree path
		if (arePathsEqual(currentPath, worktreeAutoOpenPath)) {
			// Clear the state first to prevent re-triggering
			stateManager.setGlobalState("worktreeAutoOpenPath", undefined)
			// Open the Zulong sidebar
			await HostProvider.workspace.openZulongSidebarPanel({})
		}
	} catch (error) {
		Logger.error("Error checking worktree auto-open", error)
	}
}

/**
 * 检测当前工作区是否为祖龙项目（含 .zulong/project.json 且 status 为 pending_execution）。
 * 如果是，自动打开侧边栏并启动任务执行。
 */
async function checkZulongProjectAutoStart(): Promise<void> {
	try {
		const workspacePaths = (await HostProvider.workspace.getWorkspacePaths({})).paths
		if (workspacePaths.length === 0) {
			return
		}

		const currentPath = workspacePaths[0]
		const projectJsonPath = path.join(currentPath, ".zulong", "project.json")

		// 检查 .zulong/project.json 是否存在
		try {
			await fs.access(projectJsonPath)
		} catch {
			return // 文件不存在，非项目工作区
		}

		// 读取项目配置
		const content = await fs.readFile(projectJsonPath, "utf-8")
		const projectInfo = JSON.parse(content)

		// 仅在 pending_execution 状态时自动启动
		if (projectInfo.status !== "pending_execution") {
			return
		}

		const taskDescription = projectInfo.task_description || projectInfo.description || projectInfo.name || ""
		if (!taskDescription) {
			Logger.log("[ZulongProject] project.json 中缺少任务描述，跳过自动启动")
			return
		}

		Logger.log(`[ZulongProject] 检测到待执行项目: ${projectInfo.name}, 自动启动任务...`)

		// 更新状态为 executing 防止重复触发
		projectInfo.status = "executing"
		projectInfo.updated_at = new Date().toISOString()
		await fs.writeFile(projectJsonPath, JSON.stringify(projectInfo, null, 2), "utf-8")

		// 打开侧边栏
		await HostProvider.workspace.openZulongSidebarPanel({})

		// 延迟确保 webview 已经就绪后再 initTask
		setTimeout(async () => {
			try {
				const instance = WebviewProvider.getInstance()
				if (instance && instance.controller) {
					await instance.controller.initTask(taskDescription)
					Logger.log(`[ZulongProject] 任务已自动启动: ${taskDescription.substring(0, 100)}`)
				}
			} catch (err) {
				Logger.error("[ZulongProject] 自动启动任务失败", err)
			}
		}, 3000)
	} catch (error) {
		Logger.error("[ZulongProject] 检查项目自动启动出错", error)
	}
}

/**
 * Performs cleanup when Zulong is deactivated that is common to all platforms.
 */
export async function tearDown(): Promise<void> {
	AgentConfigLoader.getInstance()?.dispose()
	PostHogClientProvider.getInstance().dispose()
	telemetryService.dispose()
	ErrorService.get().dispose()
	featureFlagsService.dispose()
	// Dispose all webview instances
	await WebviewProvider.disposeAllInstances()
	syncWorker().dispose()
	clearOnboardingModelsCache()

	// Kill any running hook processes to prevent zombies
	await HookProcessRegistry.terminateAll()
	// Clean up hook discovery cache
	HookDiscoveryCache.getInstance().dispose()
	// Stop periodic temp file cleanup
	ZulongTempManager.stopPeriodicCleanup()

	// Clean up test mode
	cleanupTestMode()
}
