import type { UserOrganization } from "@shared/proto/zulong/account"
import type React from "react"
import { createContext, useContext } from "react"

// Define User type (you may need to adjust this based on your actual User type)
export interface ZulongUser {
	uid: string
	email?: string
	displayName?: string
	photoUrl?: string
	appBaseUrl?: string
}

export interface ZulongAuthContextType {
	zulongUser: ZulongUser | null
	organizations: UserOrganization[] | null
	activeOrganization: UserOrganization | null
}

export const ZulongAuthContext = createContext<ZulongAuthContextType | undefined>(undefined)

export const ZulongAuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
	// Account system removed in Zulong-only architecture
	return (
		<ZulongAuthContext.Provider
			value={{
				zulongUser: null,
				organizations: null,
				activeOrganization: null,
			}}>
			{children}
		</ZulongAuthContext.Provider>
	)
}

export const useZulongAuth = () => {
	const context = useContext(ZulongAuthContext)
	if (context === undefined) {
		throw new Error("useZulongAuth must be used within a ZulongAuthProvider")
	}
	return context
}

export const useZulongSignIn = () => {
	return {
		isLoginLoading: false,
		handleSignIn: () => {},
	}
}

export const handleSignOut = async () => {}
