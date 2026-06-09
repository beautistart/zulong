import assert from "node:assert"
import path from "node:path"
import { DIFF_VIEW_URI_SCHEME } from "@hosts/vscode/VscodeDiffViewProvider"
import * as vscode from "vscode"
import { Logger } from "@/shared/services/Logger"
import type { ExtensionContext } from "vscode"
import { HostProvider } from "@/hosts/host-provider"
import { vscodeHostBridgeClient } from "@/hosts/vscode/hostbridge/client/host-grpc-client"
import { createStorageContext } from "@/shared/storage/storage-context"
import "./utils/path"
import { initializeBackground, tearDown } from "./common"
import {
	cleanupMcpMarketplaceCatalogFromGlobalState,
	cleanupOldApiKey,
	migrateCustomInstructionsToGlobalRules,
	migrateTaskHistoryToFile,
	migrateWelcomeViewCompleted,
	migrateWorkspaceToGlobalStorage,
} from "./core/storage/state-migrations"
import { workspaceResolver } from "./core/workspace"
import { abortCommitGeneration } from "./hosts/vscode/commit-message-generator"
import { registerZulongOutputChannel } from "./hosts/vscode/hostbridge/env/debugLog"
import {
	disposeVscodeCommentReviewController,
	getVscodeCommentReviewController,
} from "./hosts/vscode/review/VscodeCommentReviewController"
import { VscodeTerminalManager } from "./hosts/vscode/terminal/VscodeTerminalManager"
import { VscodeDiffViewProvider } from "./hosts/vscode/VscodeDiffViewProvider"
import { VscodeExecutionBridge } from "./hosts/vscode/VscodeExecutionBridge"
import { exportVSCodeStorageToSharedFiles } from "./hosts/vscode/vscode-to-file-migration"
import { ExtensionRegistryInfo } from "./registry"
import { fileExistsAtPath } from "./utils/fs"

let executionBridge: VscodeExecutionBridge | undefined

export async function activate(context: vscode.ExtensionContext) {
	const activationStartTime = performance.now()

	setupHostProvider(context)

	registerDiffContentProvider(context)
	registerUriHandler(context)

	executionBridge = new VscodeExecutionBridge(context)
	context.subscriptions.push(executionBridge)

	registerBackgroundCommands(context)

	void initializeBackgroundServices(context, activationStartTime)

	Logger.log(`[Zulong] IDE bridge activated in ${performance.now() - activationStartTime} ms`)
	return createBackgroundApi()
}

async function initializeBackgroundServices(context: vscode.ExtensionContext, activationStartTime: number): Promise<void> {
	try {
		await cleanupLegacyVSCodeStorage(context)

		const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
		const storageContext = createStorageContext({ workspacePath })
		await exportVSCodeStorageToSharedFiles(context, storageContext)
		await initializeBackground(storageContext)

		Logger.log(`[Zulong] background services initialized in ${performance.now() - activationStartTime} ms`)
	} catch (error) {
		const message = `[Zulong] background services failed after bridge activation: ${
			error instanceof Error ? error.message : String(error)
		}`
		try {
			HostProvider.get().logToChannel(message)
		} catch {
			// Host logging is best-effort during early activation.
		}
		Logger.error(message, error)
	}
}

function registerDiffContentProvider(context: vscode.ExtensionContext): void {
	const diffContentProvider = new (class implements vscode.TextDocumentContentProvider {
		provideTextDocumentContent(uri: vscode.Uri): string {
			return Buffer.from(uri.query, "base64").toString("utf-8")
		}
	})()
	context.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider(DIFF_VIEW_URI_SCHEME, diffContentProvider))
}

function registerUriHandler(context: vscode.ExtensionContext): void {
	const handleUri = async (uri: vscode.Uri) => {
		Logger.info(`[Zulong] URI handled by Web统一入口: ${uri.toString()}`)
	}
	context.subscriptions.push(vscode.window.registerUriHandler({ handleUri }))
}

function registerBackgroundCommands(context: vscode.ExtensionContext): void {
	const { commands } = ExtensionRegistryInfo

	context.subscriptions.push(
		vscode.commands.registerCommand(commands.OpenCurrentFileInWebTask, async () => {
			executionBridge?.sendIdeContext()
		}),
		vscode.commands.registerCommand(commands.OpenTerminal, async () => {
			await executionBridge?.openTerminal()
		}),
		vscode.commands.registerCommand(commands.AbortCommit, () => {
			abortCommitGeneration()
		}),
		vscode.commands.registerCommand(commands.Walkthrough, async () => {
			await vscode.commands.executeCommand("workbench.action.openWalkthrough", `${context.extension.id}#ZulongWalkthrough`)
		}),
		vscode.commands.registerCommand(commands.ReconstructTaskHistory, async () => {
			const { reconstructTaskHistory } = await import("./core/commands/reconstructTaskHistory")
			await reconstructTaskHistory()
		}),
	)
}

function createBackgroundApi() {
	return {
		startNewTask: async () => {
			executionBridge?.sendIdeContext()
		},
		sendMessage: async () => {
			executionBridge?.sendIdeContext()
		},
		pressPrimaryButton: async () => {},
		pressSecondaryButton: async () => {},
	}
}

function setupHostProvider(context: ExtensionContext) {
	const outputChannel = registerZulongOutputChannel(context)
	outputChannel.appendLine("[Zulong] Setting up VS Code background bridge...")

	const createWebview = () => {
		throw new Error("Zulong VS Code webview has been removed. Use the Web端统一入口.")
	}
	const createDiffView = () => new VscodeDiffViewProvider()
	const createCommentReview = () => getVscodeCommentReviewController()
	const createTerminalManager = () => new VscodeTerminalManager()

	const getCallbackUrl = async (callbackPath: string, _preferredPort?: number) => {
		const scheme = vscode.env.uriScheme || "vscode"
		const callbackUri = vscode.Uri.parse(`${scheme}://${context.extension.id}${callbackPath}`)

		if (vscode.env.uiKind === vscode.UIKind.Web) {
			const externalUri = await vscode.env.asExternalUri(callbackUri)
			return externalUri.toString(true)
		}

		return callbackUri.toString(true)
	}
	HostProvider.initialize(
		createWebview,
		createDiffView,
		createCommentReview,
		createTerminalManager,
		vscodeHostBridgeClient,
		() => {},
		getCallbackUrl,
		getBinaryLocation,
		context.extensionUri.fsPath,
		context.globalStorageUri.fsPath,
	)
}

async function getBinaryLocation(name: string): Promise<string> {
	if (!name.startsWith("rg")) {
		throw new Error(`Binary '${name}' is not supported`)
	}

	const checkPath = async (pkgFolder: string) => {
		const fullPathResult = workspaceResolver.resolveWorkspacePath(
			vscode.env.appRoot,
			path.join(pkgFolder, name),
			"Services.ripgrep.getBinPath",
		)
		const fullPath = typeof fullPathResult === "string" ? fullPathResult : fullPathResult.absolutePath
		return (await fileExistsAtPath(fullPath)) ? fullPath : undefined
	}

	const binPath =
		(await checkPath("node_modules/@vscode/ripgrep/bin/")) ||
		(await checkPath("node_modules/vscode-ripgrep/bin")) ||
		(await checkPath("node_modules.asar.unpacked/vscode-ripgrep/bin/")) ||
		(await checkPath("node_modules.asar.unpacked/@vscode/ripgrep/bin/"))
	if (!binPath) {
		throw new Error("Could not find ripgrep binary")
	}
	return binPath
}

export async function deactivate() {
	executionBridge?.dispose()
	executionBridge = undefined
	await tearDown({ disposeWebviews: false })
	disposeVscodeCommentReviewController()
}

const IS_DEV = process.env.IS_DEV === "true"
const DEV_WORKSPACE_FOLDER = process.env.DEV_WORKSPACE_FOLDER

if (IS_DEV) {
	assert(DEV_WORKSPACE_FOLDER, "DEV_WORKSPACE_FOLDER must be set in development")
	const watcher = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(DEV_WORKSPACE_FOLDER, "src/**/*"))

	watcher.onDidChange(({ scheme, path }) => {
		Logger.info(`${scheme} ${path} changed. Reloading VSCode...`)
		vscode.commands.executeCommand("workbench.action.reloadWindow")
	})
}

async function cleanupLegacyVSCodeStorage(context: ExtensionContext): Promise<void> {
	try {
		await cleanupOldApiKey(context)
		const hasMigrated = context.globalState.get("lastShownAnnouncementId")
		if (hasMigrated !== undefined) {
			return
		}

		Logger.info("[VS Code Storage Migrations] Starting")
		await migrateCustomInstructionsToGlobalRules(context)
		await migrateWelcomeViewCompleted(context)
		await migrateWorkspaceToGlobalStorage(context)
		await migrateTaskHistoryToFile(context)
		await cleanupMcpMarketplaceCatalogFromGlobalState(context)
		Logger.info("[VS Code Storage Migrations] Completed")
	} catch (error) {
		Logger.warn("[VS Code Storage Migrations] Failed" + (error instanceof Error ? `: ${error.message}` : ""))
	}
}
