import { ApiConfiguration, ModelInfo } from "@shared/api"
import { Mode } from "@shared/storage/types"
import { getModeSpecificFields } from "@/components/settings/utils/providerUtils"

export function validateApiConfiguration(currentMode: Mode, apiConfiguration?: ApiConfiguration): string | undefined {
	if (apiConfiguration) {
		const { apiProvider } = getModeSpecificFields(apiConfiguration, currentMode)

		switch (apiProvider) {
			case "zulong":
				// Zulong only needs the server URL which has a default value
				break
		}
	}
	return undefined
}

export function validateModelId(
	currentMode: Mode,
	apiConfiguration?: ApiConfiguration,
	openRouterModels?: Record<string, ModelInfo>,
	zulongModels?: Record<string, ModelInfo>,
): string | undefined {
	// Zulong uses backend-managed models, no client-side model validation needed
	return undefined
}
