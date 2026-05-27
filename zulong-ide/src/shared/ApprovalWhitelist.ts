/**
 * 审批白名单配置 (TSD 23.4.3)
 *
 * 用户可在设置中配置，用于自动放行特定操作。
 */
export interface ApprovalWhitelist {
	/** 目录白名单: 该目录下的写操作自动放行 */
	directories: string[]
	/** 命令白名单: 这些命令自动放行 */
	commands: string[]
	/** 工具白名单: 这些工具在任何路径都自动放行 */
	tools: string[]
	/** 模式白名单: 匹配这些正则的自动放行 */
	patterns: string[]
}

/** 审批模式 (TSD 23.4.1) */
export type ApprovalMode = "full_auto" | "whitelist" | "manual" | "popup"

/**
 * 判断指定操作是否匹配白名单
 */
export function isWhitelisted(
	tool_name: string,
	tool_args: Record<string, any> | undefined,
	whitelist: ApprovalWhitelist,
): boolean {
	// 工具名直接匹配
	if (whitelist.tools.includes(tool_name)) {
		return true
	}

	const command = tool_args?.command as string | undefined
	const path = tool_args?.path as string | undefined

	// 命令匹配
	if (command && whitelist.commands.includes(command)) {
		return true
	}
	if (command && whitelist.patterns.some((p) => new RegExp(p).test(command))) {
		return true
	}

	// 目录匹配
	if (path && whitelist.directories.some((d) => path.startsWith(d))) {
		return true
	}

	return false
}

/**
 * 默认审批白名单（保守默认值）
 */
export const DEFAULT_APPROVAL_WHITELIST: ApprovalWhitelist = {
	directories: ["src/", "tests/", "docs/", "lib/"],
	commands: [
		"npm test",
		"python -m pytest",
		"git status",
		"git diff",
		"git log",
	],
	tools: [
		"read_file",
		"search_files",
		"list_files",
		"recall_memory",
		"read_memory_node",
		"search_knowledge",
		"discover_related",
		"open_file",
		"show_diff",
	],
	patterns: [
		"^pip install",
		"^npm run",
		"^npm install",
		"^cargo build",
	],
}
