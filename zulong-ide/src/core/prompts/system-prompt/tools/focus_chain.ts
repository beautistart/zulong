import { ModelFamily } from "@/shared/prompts"
import { ZulongDefaultTool } from "@/shared/tools"
import type { ZulongToolSpec } from "../spec"

// HACK: Placeholder to act as tool dependency
const generic: ZulongToolSpec = {
	variant: ModelFamily.GENERIC,
	id: ZulongDefaultTool.TODO,
	name: "focus_chain",
	description: "",
	contextRequirements: (context) => context.focusChainSettings?.enabled === true,
}

export const focus_chain_variants = [generic]
