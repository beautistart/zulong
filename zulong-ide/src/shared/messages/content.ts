import { Anthropic } from "@anthropic-ai/sdk"
import { ZulongMessageMetricsInfo, ZulongMessageModelInfo } from "./metrics"

export type ZulongPromptInputContent = string

export type ZulongMessageRole = "user" | "assistant"

export interface ZulongReasoningDetailParam {
	type: "reasoning.text" | string
	text: string
	signature: string
	format: "anthropic-claude-v1" | string
	index: number
}

interface ZulongSharedMessageParam {
	// The id of the response that the block belongs to
	call_id?: string
}

export const REASONING_DETAILS_PROVIDERS = ["zulong", "openrouter"]

/**
 * An extension of Anthropic.MessageParam that includes Zulong-specific fields: reasoning_details.
 * This ensures backward compatibility where the messages were stored in Anthropic format with additional
 * fields unknown to Anthropic SDK.
 */
export interface ZulongTextContentBlock extends Anthropic.TextBlockParam, ZulongSharedMessageParam {
	// reasoning_details only exists for providers listed in REASONING_DETAILS_PROVIDERS
	reasoning_details?: ZulongReasoningDetailParam[]
	// Thought Signature associates with Gemini
	signature?: string
}

export interface ZulongImageContentBlock extends Anthropic.ImageBlockParam, ZulongSharedMessageParam {}

export interface ZulongDocumentContentBlock extends Anthropic.DocumentBlockParam, ZulongSharedMessageParam {}

export interface ZulongUserToolResultContentBlock extends Anthropic.ToolResultBlockParam, ZulongSharedMessageParam {}

/**
 * Assistant only content types
 */
export interface ZulongAssistantToolUseBlock extends Anthropic.ToolUseBlockParam, ZulongSharedMessageParam {
	// reasoning_details only exists for providers listed in REASONING_DETAILS_PROVIDERS
	reasoning_details?: unknown[] | ZulongReasoningDetailParam[]
	// Thought Signature associates with Gemini
	signature?: string
}

export interface ZulongAssistantThinkingBlock extends Anthropic.ThinkingBlock, ZulongSharedMessageParam {
	// The summary items returned by OpenAI response API
	// The reasoning details that will be moved to the text block when finalized
	summary?: unknown[] | ZulongReasoningDetailParam[]
}

export interface ZulongAssistantRedactedThinkingBlock extends Anthropic.RedactedThinkingBlockParam, ZulongSharedMessageParam {}

export type ZulongToolResponseContent = ZulongPromptInputContent | Array<ZulongTextContentBlock | ZulongImageContentBlock>

export type ZulongUserContent =
	| ZulongTextContentBlock
	| ZulongImageContentBlock
	| ZulongDocumentContentBlock
	| ZulongUserToolResultContentBlock

export type ZulongAssistantContent =
	| ZulongTextContentBlock
	| ZulongImageContentBlock
	| ZulongDocumentContentBlock
	| ZulongAssistantToolUseBlock
	| ZulongAssistantThinkingBlock
	| ZulongAssistantRedactedThinkingBlock

export type ZulongContent = ZulongUserContent | ZulongAssistantContent

/**
 * An extension of Anthropic.MessageParam that includes Zulong-specific fields.
 * This ensures backward compatibility where the messages were stored in Anthropic format,
 * while allowing for additional metadata specific to Zulong to avoid unknown fields in Anthropic SDK
 * added by ignoring the type checking for those fields.
 */
export interface ZulongStorageMessage extends Anthropic.MessageParam {
	/**
	 * Response ID associated with this message
	 */
	id?: string
	role: ZulongMessageRole
	content: ZulongPromptInputContent | ZulongContent[]
	/**
	 * NOTE: model information used when generating this message.
	 * Internal use for message conversion only.
	 * MUST be removed before sending message to any LLM provider.
	 */
	modelInfo?: ZulongMessageModelInfo
	/**
	 * LLM operational and performance metrics for this message
	 * Includes token counts, costs.
	 */
	metrics?: ZulongMessageMetricsInfo
	/**
	 * Timestamp of when the message was created
	 */
	ts?: number
}

/**
 * Converts ZulongStorageMessage to Anthropic.MessageParam by removing Zulong-specific fields
 * Zulong-specific fields (like modelInfo, reasoning_details) are properly omitted.
 */
export function convertZulongStorageToAnthropicMessage(
	zulongMessage: ZulongStorageMessage,
	provider = "anthropic",
): Anthropic.MessageParam {
	const { role, content } = zulongMessage

	// Handle string content - fast path
	if (typeof content === "string") {
		return { role, content }
	}

	// Removes thinking block that has no signature (invalid thinking block that's incompatible with Anthropic API)
	const filteredContent = content.filter((b) => b.type !== "thinking" || !!b.signature)

	// Handle array content - strip Zulong-specific fields for non-reasoning_details providers
	const shouldCleanContent = !REASONING_DETAILS_PROVIDERS.includes(provider)
	const cleanedContent = shouldCleanContent
		? filteredContent.map(cleanContentBlock)
		: (filteredContent as Anthropic.MessageParam["content"])

	return { role, content: cleanedContent }
}

/**
 * Clean a content block by removing Zulong-specific fields and returning only Anthropic-compatible fields
 */
export function cleanContentBlock(block: ZulongContent): Anthropic.ContentBlock {
	// Fast path: if no Zulong-specific fields exist, return as-is
	const hasZulongFields =
		"reasoning_details" in block ||
		"call_id" in block ||
		"summary" in block ||
		(block.type !== "thinking" && "signature" in block)

	if (!hasZulongFields) {
		return block as Anthropic.ContentBlock
	}

	// Removes Zulong-specific fields & the signature field that's added for Gemini.
	const { reasoning_details, call_id, summary, ...rest } = block as any

	// Remove signature from non-thinking blocks that were added for Gemini
	if (block.type !== "thinking" && rest.signature) {
		rest.signature = undefined
	}

	return rest satisfies Anthropic.ContentBlock
}
