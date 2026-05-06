import { HeroUIProvider } from "@heroui/react"
import { type ReactNode } from "react"
import { CustomPostHogProvider } from "./CustomPostHogProvider"
import { ZulongAuthProvider } from "./context/ZulongAuthContext"
import { ExtensionStateContextProvider } from "./context/ExtensionStateContext"
import { PlatformProvider } from "./context/PlatformContext"

export function Providers({ children }: { children: ReactNode }) {
	return (
		<PlatformProvider>
			<ExtensionStateContextProvider>
				<CustomPostHogProvider>
					<ZulongAuthProvider>
						<HeroUIProvider>{children}</HeroUIProvider>
					</ZulongAuthProvider>
				</CustomPostHogProvider>
			</ExtensionStateContextProvider>
		</PlatformProvider>
	)
}
