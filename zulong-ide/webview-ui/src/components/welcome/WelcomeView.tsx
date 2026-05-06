import { BooleanRequest } from "@shared/proto/zulong/common"
import { VSCodeButton, VSCodeLink } from "@vscode/webview-ui-toolkit/react"
import { memo, useEffect, useState } from "react"
import ApiOptions from "@/components/settings/ApiOptions"
import { useExtensionState } from "@/context/ExtensionStateContext"
import { StateServiceClient } from "@/services/grpc-client"
import { validateApiConfiguration } from "@/utils/validate"

const WelcomeView = memo(() => {
	const { apiConfiguration, mode } = useExtensionState()
	const [apiErrorMessage, setApiErrorMessage] = useState<string | undefined>(undefined)
	const [showApiOptions, setShowApiOptions] = useState(false)
	const [isLoading, setIsLoading] = useState(false)

	const disableLetsGoButton = apiErrorMessage != null

	const handleLogin = () => {
		// Account system removed
	}

	const handleSubmit = async () => {
		try {
			await StateServiceClient.setWelcomeViewCompleted(BooleanRequest.create({ value: true }))
		} catch (error) {
			console.error("Failed to update API configuration or complete welcome view:", error)
		}
	}

	useEffect(() => {
		setApiErrorMessage(validateApiConfiguration(mode, apiConfiguration))
	}, [apiConfiguration, mode])

	return (
		<div className="fixed inset-0 p-0 flex flex-col">
			<div className="h-full px-5 overflow-auto flex flex-col gap-2.5">
				<h2 className="text-lg font-semibold">你好，我是祖龙</h2>
				<div className="flex justify-center my-5">
					<span style={{ fontSize: "2.5rem", fontWeight: 700, opacity: 0.7 }}>祖龙</span>
				</div>
				<p>
					得益于{" "}
					<VSCodeLink className="inline" href="https://www.anthropic.com/claude/sonnet">
						Claude 4.6 Sonnet
					</VSCodeLink>
					智能编码能力的突破以及工具的支持，我可以完成各种任务——创建和编辑文件、探索复杂项目、使用
					浏览器以及执行终端命令 <i>（当然需要您的许可）</i>。我甚至可以使用 MCP
					创建新工具来扩展自身能力。
				</p>

				<p className="text-(--vscode-descriptionForeground)">
					注册账号即可免费开始使用，或使用提供 Claude
					Sonnet 等模型访问权限的 API 密钥。
				</p>

				<VSCodeButton appearance="primary" className="w-full mt-1" disabled={isLoading} onClick={handleLogin}>
					免费开始
					{isLoading && (
						<span className="ml-1 animate-spin">
							<span className="codicon codicon-refresh" />
						</span>
					)}
				</VSCodeButton>

				{!showApiOptions && (
					<VSCodeButton
						appearance="secondary"
						className="mt-2.5 w-full"
						onClick={() => setShowApiOptions(!showApiOptions)}>
						使用您自己的 API 密钥
					</VSCodeButton>
				)}

				<div className="mt-4.5">
					{showApiOptions && (
						<div>
							<ApiOptions currentMode={mode} showModelOptions={false} />
							<VSCodeButton className="mt-0.75" disabled={disableLetsGoButton} onClick={handleSubmit}>
								开始吧！
							</VSCodeButton>
						</div>
					)}
				</div>
			</div>
		</div>
	)
})

export default WelcomeView
