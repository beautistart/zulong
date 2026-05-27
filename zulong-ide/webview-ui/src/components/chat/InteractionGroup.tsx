import { memo, useMemo } from "react"
import { cn } from "@/lib/utils"
import type { InteractionPayload } from "@shared/ExtensionMessage"

interface InteractionGroupProps {
	interactions: InteractionPayload[]
	onApprove?: (interactionId: string) => void
	onReject?: (interactionId: string) => void
}

interface InteractionPair {
	action?: InteractionPayload
	observation?: InteractionPayload
	pairId: string
}

export const InteractionGroup = memo(({ interactions, onApprove, onReject }: InteractionGroupProps) => {
	const pairs = useMemo(() => {
		const pairMap = new Map<string, InteractionPair>()

		for (const interaction of interactions) {
			const pairId = interaction.pair_id
			if (!pairMap.has(pairId)) {
				pairMap.set(pairId, { pairId })
			}

			const pair = pairMap.get(pairId)!
			if (interaction.kind === "action") {
				pair.action = interaction
			} else if (interaction.kind === "observation") {
				pair.observation = interaction
			}
		}

		return Array.from(pairMap.values())
	}, [interactions])

	return (
		<div className="space-y-2">
			{pairs.map((pair) => (
				<PairLine key={pair.pairId} pair={pair} onApprove={onApprove} onReject={onReject} />
			))}
		</div>
	)
})

InteractionGroup.displayName = "InteractionGroup"

const PairLine = memo(
	({ pair, onApprove, onReject }: { pair: InteractionPair } & Pick<InteractionGroupProps, "onApprove" | "onReject">) => {
		const { action, observation } = pair
		const isSuccess = observation?.status === "succeeded"
		const isFailed = observation?.status === "failed"
		const isRunning = action?.status === "running" && !observation

		return (
			<div className="flex gap-1 items-start">
				<div className="flex flex-col items-center pt-2">
					<div
						className={cn("w-0.5 h-full min-h-4 rounded-full transition-colors", {
							"bg-blue-500": isRunning,
							"bg-green-500": isSuccess,
							"bg-red-500": isFailed,
							"bg-description/30": !isRunning && !isSuccess && !isFailed,
						})}
					/>
				</div>

				<div className="flex-1 space-y-1">
					{action && (
						<div className="text-xs">
							<span className="font-medium text-foreground">{action.title}</span>
							{action.tool_name && (
								<span className="ml-1.5 text-description font-mono">[{action.tool_name}]</span>
							)}
						</div>
					)}

					{observation && (
						<div
							className={cn("text-xs pl-2 border-l-2", {
								"border-green-500 text-green-600": isSuccess,
								"border-red-500 text-red-600": isFailed,
								"border-description/30 text-description": !isSuccess && !isFailed,
							})}>
							{observation.title}
							{observation.detail && <span className="ml-1 opacity-70">- {observation.detail}</span>}
						</div>
					)}
				</div>
			</div>
		)
	}
)

PairLine.displayName = "PairLine"
