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
	maxTokens: 16384,
	contextWindow: 131072,
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

const PROVIDER_CONFIG_CACHE_KEY = "zulong_provider_config_cache"

export interface ProviderConfigCache {
	provider: ApiProvider
	config: Partial<ApiConfiguration>
	timestamp: number
}

export function extractCurrentProviderConfig(
	apiConfiguration: ApiConfiguration | undefined,
	provider: ApiProvider,
): Partial<ApiConfiguration> {
	if (!apiConfiguration) {
		return {}
	}

	const config: Partial<ApiConfiguration> = {}

	switch (provider) {
		case "zulong":
			if (apiConfiguration.zulongServerUrl) {
				config.zulongServerUrl = apiConfiguration.zulongServerUrl
			}
			break
		case "anthropic":
			if (apiConfiguration.apiKey) {
				config.apiKey = apiConfiguration.apiKey
			}
			break
		case "openrouter":
			if (apiConfiguration.openRouterApiKey) {
				config.openRouterApiKey = apiConfiguration.openRouterApiKey
			}
			break
		default:
			break
	}

	return config
}

export function cacheProviderConfig(provider: ApiProvider, config: Partial<ApiConfiguration>): void {
	try {
		const cache: ProviderConfigCache = {
			provider,
			config,
			timestamp: Date.now(),
		}
		sessionStorage.setItem(PROVIDER_CONFIG_CACHE_KEY, JSON.stringify(cache))
	} catch (error) {
		console.warn("[ProviderConfigCache] Failed to cache provider config:", error)
	}
}

export function getCachedProviderConfig(provider: ApiProvider): Partial<ApiConfiguration> | null {
	try {
		const cached = sessionStorage.getItem(PROVIDER_CONFIG_CACHE_KEY)
		if (!cached) {
			return null
		}

		const cache: ProviderConfigCache = JSON.parse(cached)
		if (cache.provider !== provider) {
			return null
		}

		const CACHE_TTL = 24 * 60 * 60 * 1000
		if (Date.now() - cache.timestamp > CACHE_TTL) {
			sessionStorage.removeItem(PROVIDER_CONFIG_CACHE_KEY)
			return null
		}

		return cache.config
	} catch (error) {
		console.warn("[ProviderConfigCache] Failed to get cached provider config:", error)
		return null
	}
}

export function restoreProviderConfig(
	apiConfiguration: ApiConfiguration | undefined,
	provider: ApiProvider,
): ApiConfiguration {
	const cachedConfig = getCachedProviderConfig(provider)

	if (cachedConfig && Object.keys(cachedConfig).length > 0) {
		return {
			...apiConfiguration,
			...cachedConfig,
		} as ApiConfiguration
	}

	return apiConfiguration || ({} as ApiConfiguration)
}
