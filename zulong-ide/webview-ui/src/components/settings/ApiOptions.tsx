import PROVIDERS from "@shared/providers/providers.json"
import { Mode } from "@shared/storage/types"
import { VSCodeTextField } from "@vscode/webview-ui-toolkit/react"
import Fuse from "fuse.js"
import { KeyboardEvent, useEffect, useMemo, useRef, useState } from "react"
import styled from "styled-components"
import {
	normalizeApiConfiguration,
	extractCurrentProviderConfig,
	cacheProviderConfig,
	restoreProviderConfig,
} from "@/components/settings/utils/providerUtils"

function useDebounceSearch(value: string, delay: number = 300): string {
	const [debouncedValue, setDebouncedValue] = useState(value)

	useEffect(() => {
		const timer = setTimeout(() => {
			setDebouncedValue(value)
		}, delay)

		return () => {
			clearTimeout(timer)
		}
	}, [value, delay])

	return debouncedValue
}
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { PLATFORM_CONFIG, PlatformType } from "@/config/platform.config"
import { useExtensionState } from "@/context/ExtensionStateContext"
import { ZulongProvider } from "./providers/ZulongProvider"
import { useApiConfigurationHandlers } from "./utils/useApiConfigurationHandlers"

interface ApiOptionsProps {
	showModelOptions: boolean
	apiErrorMessage?: string
	modelIdErrorMessage?: string
	isPopup?: boolean
	currentMode: Mode
	initialModelTab?: "recommended" | "free"
}

export const DROPDOWN_Z_INDEX = 110

export const DropdownContainer = styled.div<{ zIndex?: number }>`
	position: relative;
	z-index: ${(props) => props.zIndex || DROPDOWN_Z_INDEX};

	// Force dropdowns to open downward
	& vscode-dropdown::part(listbox) {
		position: absolute !important;
		top: 100% !important;
		bottom: auto !important;
	}
`

const ApiOptions = ({
	showModelOptions,
	apiErrorMessage,
	modelIdErrorMessage,
	isPopup,
	currentMode,
}: ApiOptionsProps) => {
	// Use full context state for immediate save payload
	const { apiConfiguration, remoteConfigSettings } = useExtensionState()

	const { selectedProvider } = normalizeApiConfiguration(apiConfiguration, currentMode)

	const { handleModeFieldChange } = useApiConfigurationHandlers()

	// Provider search state
	const [searchTerm, setSearchTerm] = useState("")
	const [isDropdownVisible, setIsDropdownVisible] = useState(false)
	const [selectedIndex, setSelectedIndex] = useState(-1)
	const dropdownRef = useRef<HTMLDivElement>(null)
	const itemRefs = useRef<(HTMLDivElement | null)[]>([])
	const dropdownListRef = useRef<HTMLDivElement>(null)

	// Debounce search term for performance optimization
	const debouncedSearchTerm = useDebounceSearch(searchTerm, 300)

	const providerOptions = useMemo(() => {
		let providers = PROVIDERS.list
		// Filter by platform
		if (PLATFORM_CONFIG.type !== PlatformType.VSCODE) {
			providers = providers.filter((option) => option.value !== "vscode-lm")
		}

		// Filter by remote config if remoteConfiguredProviders is set
		const remoteProviders: string[] = remoteConfigSettings?.remoteConfiguredProviders || []
		if (remoteProviders.length > 0) {
			providers = providers.filter((option) => remoteProviders.includes(option.value))
		}

		return providers
	}, [remoteConfigSettings])

	const currentProviderLabel = useMemo(() => {
		return providerOptions.find((option) => option.value === selectedProvider)?.label || selectedProvider
	}, [providerOptions, selectedProvider])

	// Sync search term with current provider when not searching
	useEffect(() => {
		if (!isDropdownVisible) {
			setSearchTerm(currentProviderLabel)
		}
	}, [currentProviderLabel, isDropdownVisible])

	const searchableItems = useMemo(() => {
		return providerOptions.map((option) => ({
			value: option.value,
			html: option.label,
		}))
	}, [providerOptions])

	const fuse = useMemo(() => {
		return new Fuse(searchableItems, {
			keys: ["html"],
			threshold: 0.3,
			shouldSort: true,
			isCaseSensitive: false,
			ignoreLocation: false,
			includeMatches: true,
			minMatchCharLength: 1,
		})
	}, [searchableItems])

	const providerSearchResults = useMemo(() => {
		return debouncedSearchTerm && debouncedSearchTerm !== currentProviderLabel ? fuse.search(debouncedSearchTerm)?.map((r) => r.item) : searchableItems
	}, [searchableItems, debouncedSearchTerm, fuse, currentProviderLabel])

	const handleProviderChange = async (newProvider: string) => {
		console.log("[ApiOptions] Provider change initiated:", {
			currentProvider: selectedProvider,
			newProvider,
			currentMode,
		})

		if (selectedProvider && selectedProvider !== newProvider) {
			const currentConfig = extractCurrentProviderConfig(apiConfiguration, selectedProvider as any)
			cacheProviderConfig(selectedProvider as any, currentConfig)
			console.log("[ApiOptions] Cached current provider config:", {
				provider: selectedProvider,
				config: currentConfig,
			})
		}

		await handleModeFieldChange({ plan: "planModeApiProvider", act: "actModeApiProvider" }, newProvider as any, currentMode)

		console.log("[ApiOptions] Provider change completed:", {
			newProvider,
			apiConfiguration: apiConfiguration ? "exists" : "undefined",
		})

		setIsDropdownVisible(false)
		setSelectedIndex(-1)
	}

	const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
		if (!isDropdownVisible) {
			return
		}

		switch (event.key) {
			case "ArrowDown":
				event.preventDefault()
				setSelectedIndex((prev) => (prev < providerSearchResults.length - 1 ? prev + 1 : prev))
				break
			case "ArrowUp":
				event.preventDefault()
				setSelectedIndex((prev) => (prev > 0 ? prev - 1 : prev))
				break
			case "Enter":
				event.preventDefault()
				if (selectedIndex >= 0 && selectedIndex < providerSearchResults.length) {
					void handleProviderChange(providerSearchResults[selectedIndex].value)
				}
				break
			case "Escape":
				setIsDropdownVisible(false)
				setSelectedIndex(-1)
				setSearchTerm(currentProviderLabel)
				break
		}
	}

	// Close dropdown when clicking outside
	useEffect(() => {
		const handleClickOutside = (event: MouseEvent) => {
			if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
				setIsDropdownVisible(false)
				setSearchTerm(currentProviderLabel)
			}
		}

		document.addEventListener("mousedown", handleClickOutside)
		return () => {
			document.removeEventListener("mousedown", handleClickOutside)
		}
	}, [currentProviderLabel])

	// Reset selection when search term changes
	useEffect(() => {
		setSelectedIndex(-1)
		if (dropdownListRef.current) {
			dropdownListRef.current.scrollTop = 0
		}
	}, [searchTerm])

	// Scroll selected item into view
	useEffect(() => {
		if (selectedIndex >= 0 && itemRefs.current[selectedIndex]) {
			itemRefs.current[selectedIndex]?.scrollIntoView({
				block: "nearest",
				behavior: "smooth",
			})
		}
	}, [selectedIndex])

	return (
		<div style={{ display: "flex", flexDirection: "column", gap: 5, marginBottom: isPopup ? -10 : 0 }}>
			<style>
				{`
				.provider-item-highlight {
					background-color: var(--vscode-editor-findMatchHighlightBackground);
					color: inherit;
				}
				`}
			</style>
			<DropdownContainer className="dropdown-container">
				{remoteConfigSettings?.remoteConfiguredProviders && remoteConfigSettings.remoteConfiguredProviders.length > 0 ? (
					<Tooltip>
						<TooltipTrigger>
							<div className="flex items-center gap-2 mb-1">
								<label htmlFor="api-provider">
									<span style={{ fontWeight: 500 }}>API Provider</span>
								</label>
								<i className="codicon codicon-lock text-description text-sm" />
							</div>
						</TooltipTrigger>
						<TooltipContent>Provider options are managed by your organization's remote configuration</TooltipContent>
					</Tooltip>
				) : (
					<label htmlFor="api-provider">
						<span style={{ fontWeight: 500 }}>API Provider</span>
					</label>
				)}
				<ProviderDropdownWrapper ref={dropdownRef}>
					<VSCodeTextField
						data-testid="provider-selector-input"
						id="api-provider"
						onFocus={() => {
							setIsDropdownVisible(true)
							setSearchTerm("")
						}}
						onInput={(e) => {
							setSearchTerm((e.target as HTMLInputElement)?.value || "")
							setIsDropdownVisible(true)
						}}
						onKeyDown={handleKeyDown}
						placeholder="Search and select provider..."
						role="combobox"
						style={{
							width: "100%",
							zIndex: DROPDOWN_Z_INDEX,
							position: "relative",
							minWidth: 130,
						}}
						value={searchTerm}>
						{searchTerm && searchTerm !== currentProviderLabel && (
							<div
								aria-label="Clear search"
								className="input-icon-button codicon codicon-close"
								onClick={() => {
									setSearchTerm("")
									setIsDropdownVisible(true)
								}}
								slot="end"
								style={{
									display: "flex",
									justifyContent: "center",
									alignItems: "center",
									height: "100%",
								}}
							/>
						)}
					</VSCodeTextField>
					{isDropdownVisible && (
						<ProviderDropdownList ref={dropdownListRef} role="listbox">
							{providerSearchResults.map((item, index) => (
								<ProviderDropdownItem
									data-testid={`provider-option-${item.value}`}
									isSelected={index === selectedIndex}
									key={item.value}
									onClick={() => void handleProviderChange(item.value)}
									onMouseEnter={() => setSelectedIndex(index)}
									ref={(el) => {
										itemRefs.current[index] = el
									}}
									role="option">
									<span>{item.html}</span>
								</ProviderDropdownItem>
							))}
						</ProviderDropdownList>
					)}
				</ProviderDropdownWrapper>
			</DropdownContainer>

			{apiConfiguration && selectedProvider === "zulong" ? (
				<ZulongProvider currentMode={currentMode} isPopup={isPopup} showModelOptions={showModelOptions} />
			) : (
				!apiConfiguration && (
					<p style={{ fontSize: 12, color: "var(--vscode-descriptionForeground)", marginTop: 5 }}>
						正在加载配置...
					</p>
				)
			)}

			{apiErrorMessage && (
				<p
					style={{
						margin: "-10px 0 4px 0",
						fontSize: 12,
						color: "var(--vscode-errorForeground)",
					}}>
					{apiErrorMessage}
				</p>
			)}
			{modelIdErrorMessage && (
				<p
					style={{
						margin: "-10px 0 4px 0",
						fontSize: 12,
						color: "var(--vscode-errorForeground)",
					}}>
					{modelIdErrorMessage}
				</p>
			)}
		</div>
	)
}

export default ApiOptions

const ProviderDropdownWrapper = styled.div`
	position: relative;
	width: 100%;
`

const ProviderDropdownList = styled.div`
	position: absolute;
	top: calc(100% - 3px);
	left: 0;
	width: calc(100% - 2px);
	max-height: 200px;
	overflow-y: auto;
	background-color: var(--vscode-dropdown-background);
	border: 1px solid var(--vscode-list-activeSelectionBackground);
	z-index: ${DROPDOWN_Z_INDEX - 1};
	border-bottom-left-radius: 3px;
	border-bottom-right-radius: 3px;
`

const ProviderDropdownItem = styled.div<{ isSelected: boolean }>`
	padding: 5px 10px;
	cursor: pointer;
	word-break: break-all;
	white-space: normal;

	background-color: ${({ isSelected }) => (isSelected ? "var(--vscode-list-activeSelectionBackground)" : "inherit")};

	&:hover {
		background-color: var(--vscode-list-activeSelectionBackground);
	}
`
