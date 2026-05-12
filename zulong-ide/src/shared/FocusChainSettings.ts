export interface FocusChainSettings {
	enabled: boolean
	remindClineInterval: number
}

export const DEFAULT_FOCUS_CHAIN_SETTINGS: FocusChainSettings = {
	enabled: true,
	remindClineInterval: 6,
}
