export interface FocusChainSettings {
	// Enable/disable the focus chain feature
	enabled: boolean
	// Interval (in messages) to remind Zulong about focus chain
	remindZulongInterval: number
}

export const DEFAULT_FOCUS_CHAIN_SETTINGS: FocusChainSettings = {
	enabled: true,
	remindZulongInterval: 6,
}
