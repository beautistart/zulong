import { ModelFamily } from "@/shared/prompts"
import { ZulongDefaultTool } from "@/shared/tools"
import type { ZulongToolSpec } from "../spec"

/**
 * ## delete_file
 * Description: Request to delete a file at the specified path. Use with caution as this action is irreversible.
 * Parameters:
 * - path: (required) The path of the file to delete (relative to the current working directory {{CWD}})
 * Usage:
 * <delete_file>
 * <path>File path here</path>
 * </delete_file>
 */

const id = ZulongDefaultTool.FILE_DELETE

const GENERIC: ZulongToolSpec = {
	variant: ModelFamily.GENERIC,
	id,
	name: "delete_file",
	description:
		"Request to delete a file at the specified path. Use with caution as this action is irreversible.",
	parameters: [
		{
			name: "path",
			required: true,
			instruction: `The path of the file to delete (relative to the current working directory {{CWD}}){{MULTI_ROOT_HINT}}`,
			usage: "File path here",
		},
	],
}

const NATIVE_NEXT_GEN: ZulongToolSpec = {
	variant: ModelFamily.NATIVE_NEXT_GEN,
	id,
	name: "delete_file",
	description:
		"[IMPORTANT: Always output the absolutePath first] Request to delete a file at the specified path. Use with caution as this action is irreversible.",
	parameters: [
		{
			name: "absolutePath",
			required: true,
			instruction: "The absolute path to the file to delete.",
		},
	],
}

const NATIVE_GPT_5: ZulongToolSpec = {
	...NATIVE_NEXT_GEN,
	variant: ModelFamily.NATIVE_GPT_5,
}

export const delete_file_variants = [GENERIC, NATIVE_NEXT_GEN, NATIVE_GPT_5]
