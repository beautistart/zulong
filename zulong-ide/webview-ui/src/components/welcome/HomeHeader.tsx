import { EmptyRequest } from "@shared/proto/zulong/common"
import { useExtensionState } from "@/context/ExtensionStateContext"
import { UiServiceClient } from "@/services/grpc-client"

interface HomeHeaderProps {
	shouldShowQuickWins?: boolean
}

const HomeHeader = ({ shouldShowQuickWins = false }: HomeHeaderProps) => {
	const { environment, lazyTeammateModeEnabled } = useExtensionState()

	const handleTakeATour = async () => {
		try {
			await UiServiceClient.openWalkthrough(EmptyRequest.create())
		} catch (error) {
			console.error("Error opening walkthrough:", error)
		}
	}

	const headingText = lazyTeammateModeEnabled ? "好吧，我来帮忙" : "有什么我可以帮您的？"

	return (
		<div className="flex flex-col items-center mb-5">
			<div className="my-7">
				<span style={{ fontSize: "2.5rem", fontWeight: 700, color: "var(--vscode-foreground)", opacity: 0.7 }}>祖龙</span>
			</div>
			<div className="text-center flex items-center justify-center px-4">
				<h1 className="m-0 font-bold">{headingText}</h1>
			</div>
			{shouldShowQuickWins && (
				<div className="mt-4">
					<button
						className="flex items-center gap-2 px-4 py-2 rounded-full border border-border-panel bg-white/2 hover:bg-list-background-hover transition-colors duration-150 ease-in-out text-code-foreground text-sm font-medium cursor-pointer"
						onClick={handleTakeATour}
						type="button">
						开始导览
						<span className="codicon codicon-play scale-90" />
					</button>
				</div>
			)}
		</div>
	)
}

export default HomeHeader
