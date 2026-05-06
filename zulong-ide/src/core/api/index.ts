import { ApiConfiguration, ModelInfo } from "@shared/api"
import { Mode } from "@shared/storage/types"
import { ZulongStorageMessage } from "@/shared/messages/content"
import { Logger } from "@/shared/services/Logger"
import { ZulongTool } from "@/shared/tools"
import { ZulongHandler } from "./providers/zulong"
import { ApiStream, ApiStreamUsageChunk } from "./transform/stream"

export type CommonApiHandlerOptions = {
	onRetryAttempt?: ApiConfiguration["onRetryAttempt"]
}
export interface ApiHandler {
	createMessage(systemPrompt: string, messages: ZulongStorageMessage[], tools?: ZulongTool[], useResponseApi?: boolean): ApiStream
	getModel(): ApiHandlerModel
	getApiStreamUsage?(): Promise<ApiStreamUsageChunk | undefined>
	abort?(): void
	/**
	 * Send tool execution result back to the backend (used by Zulong provider
	 * where results flow via WebSocket rather than in the next API request).
	 */
	sendToolResult?(callId: string, toolName: string, result: string, isError?: boolean): void
}

export interface ApiHandlerModel {
	id: string
	info: ModelInfo
}

export interface ApiProviderInfo {
	providerId: string
	model: ApiHandlerModel
	mode: Mode
	customPrompt?: string // "compact"
}

export interface SingleCompletionHandler {
	completePrompt(prompt: string): Promise<string>
}

function createHandlerForProvider(
	_apiProvider: string | undefined,
	options: Omit<ApiConfiguration, "apiProvider">,
	_mode: Mode,
): ApiHandler {
	// Zulong is the only supported provider
	return new ZulongHandler({
		onRetryAttempt: options.onRetryAttempt,
		zulongServerUrl: options.zulongServerUrl,
	})
}

export function buildApiHandler(configuration: ApiConfiguration, mode: Mode): ApiHandler {
	const { planModeApiProvider, actModeApiProvider, ...options } = configuration

	const apiProvider = mode === "plan" ? planModeApiProvider : actModeApiProvider

	// Validate thinking budget tokens against model's maxTokens to prevent API errors
	// wrapped in a try-catch for safety, but this should never throw
	try {
		const thinkingBudgetTokens = mode === "plan" ? options.planModeThinkingBudgetTokens : options.actModeThinkingBudgetTokens
		if (thinkingBudgetTokens && thinkingBudgetTokens > 0) {
			const handler = createHandlerForProvider(apiProvider, options, mode)

			const modelInfo = handler.getModel().info
			if (modelInfo?.maxTokens && modelInfo.maxTokens > 0 && thinkingBudgetTokens > modelInfo.maxTokens) {
				const clippedValue = modelInfo.maxTokens - 1
				if (mode === "plan") {
					options.planModeThinkingBudgetTokens = clippedValue
				} else {
					options.actModeThinkingBudgetTokens = clippedValue
				}
			} else {
				return handler // don't rebuild unless its necessary
			}
		}
	} catch (error) {
		Logger.error("buildApiHandler error:", error)
	}

	return createHandlerForProvider(apiProvider, options, mode)
}
