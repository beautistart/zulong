import {
	isGLMModelFamily,
	isLocalModel,
	isNextGenModelFamily,
	isNextGenModelProvider,
	isTrinityModelFamily,
} from "@utils/model-utils"
import { ModelFamily } from "@/shared/prompts"
import { Logger } from "@/shared/services/Logger"
import { ZulongDefaultTool } from "@/shared/tools"
import { SystemPromptSection } from "../../templates/placeholders"
import { createVariant } from "../variant-builder"
import { validateVariant } from "../variant-validator"
import { baseTemplate } from "./template"

export const config = createVariant(ModelFamily.GENERIC)
	.description("The fallback prompt for generic use cases and models.")
	.version(1)
	.tags("fallback", "stable")
	.labels({
		stable: 1,
		fallback: 1,
	})
	// Generic matcher - fallback for everything that doesn't match other variants
	// This will match anything that doesn't match the other specific variants
	.matcher((context) => {
		const providerInfo = context.providerInfo
		if (!providerInfo.providerId || !providerInfo.model.id) {
			return true
		}
		const modelId = providerInfo.model.id.toLowerCase()
		return (
			// Not a local model with compact prompt enabled
			!(providerInfo.customPrompt === "compact" && isLocalModel(providerInfo)) &&
			// Not a next-gen model
			!(isNextGenModelProvider(providerInfo) && isNextGenModelFamily(modelId)) &&
			// Not a GLM model
			!isGLMModelFamily(modelId) &&
			// Not a Trinity model
			!isTrinityModelFamily(modelId)
		)
	})
	.template(baseTemplate)
	.components(
		SystemPromptSection.AGENT_ROLE,
		SystemPromptSection.TOOL_USE,
		SystemPromptSection.TASK_PROGRESS,
		SystemPromptSection.MCP,
		SystemPromptSection.EDITING_FILES,
		SystemPromptSection.ACT_VS_PLAN,
		SystemPromptSection.CAPABILITIES,
		SystemPromptSection.RULES,
		SystemPromptSection.SYSTEM_INFO,
		SystemPromptSection.OBJECTIVE,
		SystemPromptSection.USER_INSTRUCTIONS,
		SystemPromptSection.SKILLS,
	)
	.tools(
		ZulongDefaultTool.BASH,
		ZulongDefaultTool.FILE_READ,
		ZulongDefaultTool.FILE_NEW,
		ZulongDefaultTool.FILE_EDIT,
		ZulongDefaultTool.SEARCH,
		ZulongDefaultTool.LIST_FILES,
		ZulongDefaultTool.LIST_CODE_DEF,
		ZulongDefaultTool.BROWSER,
		ZulongDefaultTool.MCP_USE,
		ZulongDefaultTool.MCP_ACCESS,
		ZulongDefaultTool.ASK,
		ZulongDefaultTool.ATTEMPT,
		ZulongDefaultTool.PLAN_MODE,
		ZulongDefaultTool.MCP_DOCS,
		ZulongDefaultTool.TODO,
		ZulongDefaultTool.GENERATE_EXPLANATION,
		ZulongDefaultTool.USE_SKILL,
		ZulongDefaultTool.USE_SUBAGENTS,
	)
	.placeholders({
		MODEL_FAMILY: "generic",
	})
	.config({})
	.build()

// Compile-time validation
const validationResult = validateVariant({ ...config, id: "generic" }, { strict: true })
if (!validationResult.isValid) {
	Logger.error("Generic variant configuration validation failed:", validationResult.errors)
	throw new Error(`Invalid generic variant configuration: ${validationResult.errors.join(", ")}`)
}

if (validationResult.warnings.length > 0) {
	Logger.warn("Generic variant configuration warnings:", validationResult.warnings)
}

// Export type information for better IDE support
export type GenericVariantConfig = typeof config
