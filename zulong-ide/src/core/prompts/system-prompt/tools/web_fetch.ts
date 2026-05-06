import { ModelFamily } from "@/shared/prompts"
import { ZulongDefaultTool } from "@/shared/tools"
import type { ZulongToolSpec } from "../spec"
import { TASK_PROGRESS_PARAMETER } from "../types"

const GENERIC: ZulongToolSpec = {
	variant: ModelFamily.GENERIC,
	id: ZulongDefaultTool.WEB_FETCH,
	name: "web_fetch",
	description: `Fetches content from a specified URL and analyzes it using your prompt
- Takes a URL and analysis prompt as input
- Fetches the URL content and processes based on your prompt
- Use this tool when you need to retrieve and analyze web content
- IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions.
- The URL must be a fully-formed valid URL
- The prompt must be at least 2 characters
- HTTP URLs will be automatically upgraded to HTTPS
- This tool is read-only and does not modify any files`,
	contextRequirements: (context) => context.providerInfo.providerId === "zulong" && context.zulongWebToolsEnabled === true,
	parameters: [
		{
			name: "url",
			required: true,
			instruction: "The URL to fetch content from",
			usage: "https://example.com/docs",
		},
		{
			name: "prompt",
			required: true,
			instruction: "The prompt to use for analyzing the webpage content",
			usage: "Summarize the main points and key takeaways",
		},
		TASK_PROGRESS_PARAMETER,
	],
}

const NATIVE_NEXT_GEN: ZulongToolSpec = {
	variant: ModelFamily.NATIVE_NEXT_GEN,
	id: ZulongDefaultTool.WEB_FETCH,
	name: "web_fetch",
	description:
		"Fetches and analyzes content from a specified URL. IMPORTANT: If an MCP-provided web fetch tool is available, prefer using that tool instead of this one, as it may have fewer restrictions.",
	contextRequirements: (context) => context.providerInfo.providerId === "zulong" && context.zulongWebToolsEnabled === true,
	parameters: [
		{
			name: "url",
			required: true,
			instruction: "The URL to fetch content from",
		},
		{
			name: "prompt",
			required: true,
			instruction: "Prompt for analyzing the webpage content",
		},
		TASK_PROGRESS_PARAMETER,
	],
}

const NATIVE_GPT_5: ZulongToolSpec = {
	...NATIVE_NEXT_GEN,
	variant: ModelFamily.NATIVE_GPT_5,
}

export const web_fetch_variants = [GENERIC, NATIVE_GPT_5, NATIVE_NEXT_GEN]
