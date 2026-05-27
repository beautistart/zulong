import "./utils/path" // necessary to have access to String.prototype.toPosix
import type { WebviewProvider } from "./core/webview"

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

/**
 * Performs intialization for Zulong that is common to all platforms.
 *
 * @param context
 * @throws ZulongConfigurationError if endpoints.json exists but is invalid
 */
export async function initializeBackground(storageContext: StorageContext): Promise<void> {
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

	const stateManager = StateManager.get()
	// Non-blocking announcement check and display
	showVersionUpdateAnnouncement(stateManager)

	// =============== Background sync and cleanup tasks ===============
	// Use remote config blobStoreConfig if available, otherwise fall back to env vars
	const blobStoreSettings = stateManager.getRemoteConfigSettings()?.blobStoreConfig ?? getBlobStoreSettingsFromEnv()
	syncWorker().init({ ...blobStoreSettings, userDistinctId: getDistinctId() })
	// Clean up old temp files in background (non-blocking) and start periodic cleanup every 24 hours
	ZulongTempManager.startPeriodicCleanup()
	// Clean up orphaned file context warnings (startup cleanup)
	FileContextTracker.cleanupOrphanedWarnings(stateManager)

	telemetryService.captureExtensionActivated()
}

/**
 * Standalone/external hosts still own a Webview surface.  The VS Code
 * extension must use initializeBackground() so it stays a backend bridge.
 */
export async function initialize(storageContext: StorageContext): Promise<WebviewProvider> {
	const { WebviewProvider } = await import("./core/webview")
	await initializeBackground(storageContext)
	return HostProvider.get().createWebviewProvider()
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
 * Performs cleanup when Zulong is deactivated that is common to all platforms.
 */
export async function tearDown(options: { disposeWebviews?: boolean } = {}): Promise<void> {
	const { disposeWebviews = false } = options
	AgentConfigLoader.getInstance()?.dispose()
	PostHogClientProvider.getInstance().dispose()
	telemetryService.dispose()
	ErrorService.get().dispose()
	featureFlagsService.dispose()
	if (disposeWebviews) {
		const { WebviewProvider } = await import("./core/webview")
		await WebviewProvider.disposeAllInstances()
	}
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
