import { SystemPromptSection } from "../templates/placeholders"
import { TemplateEngine } from "../templates/TemplateEngine"
import type { PromptVariant, SystemPromptContext } from "../types"

const FEEDBACK_TEMPLATE_TEXT = `
If the user asks for help or wants to give feedback inform them of the following: 
- To give feedback, users should report the issue using the /reportbug slash command in the chat. 

When the user directly asks about Zulong (eg 'can Zulong do...', 'does Zulong have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the web_fetch tool to gather information to answer the question from Zulong docs at https://docs.zulong.ai.
  - The available sub-pages are \`getting-started\` (Intro for new coders, installing Zulong and dev essentials), \`model-selection\` (Model Selection Guide, Custom Model Configs, Bedrock, Vertex, Codestral, LM Studio, Ollama), \`features\` (Auto approve, Checkpoints, Zulong rules, Drag & Drop, Plan & Act, Workflows, etc), \`task-management\` (Task and Context Management in Zulong), \`prompt-engineering\` (Improving your prompting skills, Prompt Engineering Guide), \`zulong-tools\` (Zulong Tools Reference Guide, New Task Tool, Remote Browser Support, Slash Commands), \`mcp\` (MCP Overview, Adding/Configuring Servers, Transport Mechanisms, MCP Dev Protocol), \`enterprise\` (Cloud provider integration, Security concerns, Custom instructions), \`more-info\` (Telemetry and other reference content)
  - Example: https://docs.zulong.ai/features/auto-approve`

export async function getFeedbackSection(variant: PromptVariant, context: SystemPromptContext): Promise<string | undefined> {
	if (!context.focusChainSettings?.enabled) {
		return undefined
	}

	const template = variant.componentOverrides?.[SystemPromptSection.FEEDBACK]?.template || FEEDBACK_TEMPLATE_TEXT

	return new TemplateEngine().resolve(template, context, {})
}
