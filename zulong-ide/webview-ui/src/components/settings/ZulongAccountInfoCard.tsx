import { VSCodeButton } from "@vscode/webview-ui-toolkit/react"
import { useZulongAuth } from "@/context/ZulongAuthContext"
import { useExtensionState } from "@/context/ExtensionStateContext"

export const ZulongAccountInfoCard = () => {
	const { zulongUser } = useZulongAuth()
	const { navigateToAccount } = useExtensionState()

	const user = zulongUser || undefined

	const handleShowAccount = () => {
		navigateToAccount()
	}

	return (
		<div className="max-w-[600px]">
			{user ? (
				<VSCodeButton appearance="secondary" onClick={handleShowAccount}>
					View Billing & Usage
				</VSCodeButton>
			) : (
				<div>
					<VSCodeButton className="mt-0" disabled onClick={() => {}}>
						Sign Up with Zulong
					</VSCodeButton>
				</div>
			)}
		</div>
	)
}
