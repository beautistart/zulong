export enum NEW_USER_TYPE {
	FREE = "free",
	POWER = "power",
	BYOK = "byok",
}

type UserTypeSelection = {
	title: string
	description: string
	type: NEW_USER_TYPE
}

export const STEP_CONFIG = {
	0: {
		title: "您将如何使用 Zulong？",
		description: "选择以下选项开始使用。",
		buttons: [
			{ text: "继续", action: "next", variant: "default" },
			{ text: "登录 Zulong", action: "signin", variant: "secondary" },
		],
	},
	[NEW_USER_TYPE.FREE]: {
		title: "选择免费模型",
		buttons: [
			{ text: "创建账号", action: "signup", variant: "default" },
			{ text: "返回", action: "back", variant: "secondary" },
		],
	},
	[NEW_USER_TYPE.POWER]: {
		title: "选择您的模型",
		buttons: [
			{ text: "Create my Account", action: "signup", variant: "default" },
			{ text: "返回", action: "back", variant: "secondary" },
		],
	},
	[NEW_USER_TYPE.BYOK]: {
		title: "配置您的提供商",
		buttons: [
			{ text: "继续", action: "done", variant: "default" },
			{ text: "Back", action: "back", variant: "secondary" },
		],
	},
	2: {
		title: "即将完成！",
		description: "在浏览器中完成账号创建，然后返回这里完成设置。",
		buttons: [{ text: "Back", action: "back", variant: "secondary" }],
	},
} as const

export const USER_TYPE_SELECTIONS: UserTypeSelection[] = [
	{ title: "完全免费", description: "零成本开始使用", type: NEW_USER_TYPE.FREE },
	{ title: "前沿模型", description: "Claude、GPT Codex、Gemini 等", type: NEW_USER_TYPE.POWER },
	{ title: "使用自己的 API 密钥", description: "使用您选择的提供商与 Zulong 配合", type: NEW_USER_TYPE.BYOK },
]
