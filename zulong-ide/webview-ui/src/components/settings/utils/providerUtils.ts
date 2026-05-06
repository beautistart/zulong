import {
	ApiConfiguration,
	ApiProvider,
	ModelInfo,
} from "@shared/api"
import { Mode } from "@shared/storage/types"
import * as reasoningSupport from "@shared/utils/reasoning-support"

export function supportsReasoningEffortForModelId(modelId?: string, _allowShortOpenAiIds = false): boolean {
	return reasoningSupport.supportsReasoningEffortForModel(modelId)
}

/**
 * Returns the static model list for a provider.
 * Zulong uses backend-managed models, so always returns undefined.
 */
export function getModelsForProvider(
	_provider: ApiProvider,
	_apiConfiguration?: ApiConfiguration,
	_dynamicModels: { liteLlmModels?: Record<string, ModelInfo>; basetenModels?: Record<string, ModelInfo> } = {},
): Record<string, ModelInfo> | undefined {
	return undefined
}

/**
 * Interface for normalized API configuration
 */
export interface NormalizedApiConfig {
	selectedProvider: ApiProvider
	selectedModelId: string
	selectedModelInfo: ModelInfo
}

const zulongModelInfo: ModelInfo = {
	maxTokens: 8192,
	contextWindow: 32768,
	supportsImages: false,
	supportsPromptCache: false,
	inputPrice: 0,
	outputPrice: 0,
	description: "Zulong multi-layer adaptive intelligent agent",
}

/**
 * Normalizes API configuration to ensure consistent values
 */
export function normalizeApiConfiguration(
	apiConfiguration: ApiConfiguration | undefined,
	currentMode: Mode,
): NormalizedApiConfig {
	const provider =
		(currentMode === "plan" ? apiConfiguration?.planModeApiProvider : apiConfiguration?.actModeApiProvider) || "zulong"

	return {
		selectedProvider: provider,
		selectedModelId: "zulong-agent",
		selectedModelInfo: zulongModelInfo,
	}
}

/**
 * Gets mode-specific field values from API configuration.
 * Kept for compatibility with components that destructure mode-specific fields.
 */
export function getModeSpecificFields(apiConfiguration: ApiConfiguration | undefined, mode: Mode) {
	if (!apiConfiguration) {
		return {
			apiProvider: undefined,
			apiModelId: undefined,
			thinkingBudgetTokens: undefined,
			reasoningEffort: undefined,
		}
	}

	return {
		apiProvider: mode === "plan" ? apiConfiguration.planModeApiProvider : apiConfiguration.actModeApiProvider,
		apiModelId: mode === "plan" ? apiConfiguration.planModeApiModelId : apiConfiguration.actModeApiModelId,
		thinkingBudgetTokens:
			mode === "plan" ? apiConfiguration.planModeThinkingBudgetTokens : apiConfiguration.actModeThinkingBudgetTokens,
		reasoningEffort: mode === "plan" ? apiConfiguration.planModeReasoningEffort : apiConfiguration.actModeReasoningEffort,
	}
}

/**
 * Synchronizes mode configurations by copying the source mode's settings to both modes
 */
export async function syncModeConfigurations(
	apiConfiguration: ApiConfiguration | undefined,
	sourceMode: Mode,
	handleFieldsChange: (updates: Partial<ApiConfiguration>) => Promise<void>,
): Promise<void> {
	if (!apiConfiguration) {
		return
	}

	const sourceFields = getModeSpecificFields(apiConfiguration, sourceMode)
	const { apiProvider } = sourceFields

	if (!apiProvider) {
		return
	}

	const updates: Partial<ApiConfiguration> = {
		planModeApiProvider: sourceFields.apiProvider,
		actModeApiProvider: sourceFields.apiProvider,
		planModeThinkingBudgetTokens: sourceFields.thinkingBudgetTokens,
		actModeThinkingBudgetTokens: sourceFields.thinkingBudgetTokens,
		planModeReasoningEffort: sourceFields.reasoningEffort,
		actModeReasoningEffort: sourceFields.reasoningEffort,
		planModeApiModelId: sourceFields.apiModelId,
		actModeApiModelId: sourceFields.apiModelId,
	}

	await handleFieldsChange(updates)
}

export { filterOpenRouterModelIds } from "@shared/utils/model-filters"

export const getProviderInfo = (
	_provider: ApiProvider,
	apiConfiguration: any,
	_effectiveMode: "plan" | "act",
): { modelId?: string; baseUrl?: string; helpText: string } => {
	return {
		modelId: undefined,
		baseUrl: apiConfiguration?.zulongServerUrl,
		helpText: "Make sure the Zulong backend server is running",
	}
}
