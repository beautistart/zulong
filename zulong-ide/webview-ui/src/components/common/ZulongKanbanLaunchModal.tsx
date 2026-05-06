import { StringRequest } from "@shared/proto/zulong/common"
import { VSCodeButton, VSCodeCheckbox } from "@vscode/webview-ui-toolkit/react"
import React, { useEffect, useMemo, useState } from "react"
import kanbanDemoVideoMp4 from "@/assets/zulong_kanban_demo.mp4"
import kanbanDemoVideoWebm from "@/assets/zulong_kanban_demo.webm"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import { PLATFORM_CONFIG, PlatformType } from "@/config/platform.config"
import { FileServiceClient, StateServiceClient } from "@/services/grpc-client"

const INSTALL_COMMAND = "npm install -g zulong-ide"
const COPIED_TIMEOUT = 1500
const resolveAssetSrc = (src: string) => (src.startsWith("/src/") ? new URL(src, import.meta.url).toString() : src)
const kanbanDemoMp4Src = resolveAssetSrc(kanbanDemoVideoMp4)
const kanbanDemoWebmSrc = resolveAssetSrc(kanbanDemoVideoWebm)

export const ZULONG_KANBAN_MODAL_DISMISS_ID = "zulong-kanban-launch-modal-v1"

interface ZulongKanbanLaunchModalProps {
	open: boolean
	onClose: (doNotShowAgain: boolean) => void
}

export const ZulongKanbanLaunchModal: React.FC<ZulongKanbanLaunchModalProps> = ({ open, onClose }) => {
	const [doNotShowAgain, setDoNotShowAgain] = useState(false)
	const [isInstalling, setIsInstalling] = useState(false)
	const [copied, setCopied] = useState(false)

	const isVsCode = useMemo(() => PLATFORM_CONFIG.type === PlatformType.VSCODE, [])

	useEffect(() => {
		if (open) {
			setCopied(false)
			setIsInstalling(false)
		}
	}, [open])

	const handleAction = async () => {
		if (isVsCode) {
			setIsInstalling(true)
			try {
				await StateServiceClient.installZulongCli({})
			} catch (error) {
				console.error("Failed to run CLI install command:", error)
			} finally {
				setIsInstalling(false)
			}
			return
		}

		try {
			await FileServiceClient.copyToClipboard(StringRequest.create({ value: INSTALL_COMMAND }))
			setCopied(true)
			setTimeout(() => setCopied(false), COPIED_TIMEOUT)
		} catch (error) {
			console.error("Failed to copy CLI install command:", error)
		}
	}

	return (
		<Dialog onOpenChange={(isOpen) => !isOpen && onClose(doNotShowAgain)} open={open}>
			<DialogContent
				aria-describedby="zulong-kanban-description"
				aria-labelledby="zulong-kanban-title"
				className="pt-4 px-5 pb-4 gap-0 max-w-2xl">
				<div className="space-y-3" id="zulong-kanban-description">
					<div className="pr-6 min-h-6 flex items-center">
						<h2
							className="m-0 text-lg font-semibold"
							id="zulong-kanban-title"
							style={{ color: "var(--vscode-editor-foreground)" }}>
							Zulong 看板介绍
						</h2>
					</div>

					<video
						autoPlay
						className="w-full rounded-md border border-[var(--vscode-editorGroup-border)]"
						loop
						muted
						playsInline>
						<source src={kanbanDemoMp4Src} type="video/mp4" />
						<source src={kanbanDemoWebmSrc} type="video/webm" />
					</video>

					<p className="text-sm" style={{ color: "var(--vscode-descriptionForeground)" }}>
						一个更适合并行运行多个智能体和审查差异的 IDE 替代方案。启用自动提交并将卡片链接在一起，
						创建依赖链以自主完成大量工作。
					</p>

					<div className="p-1">
						<code className="block rounded-sm px-2 py-1 bg-[var(--vscode-textCodeBlock-background)] text-sm">
							{INSTALL_COMMAND}
						</code>
						<div className="mt-3">
							<VSCodeButton disabled={isInstalling} onClick={handleAction}>
								{isVsCode
									? isInstalling
										? "正在运行安装命令..."
										: "在终端中运行"
									: copied
										? "已复制"
										: "复制命令"}
							</VSCodeButton>
						</div>
					</div>

					<div className="pt-2">
						<VSCodeCheckbox
							checked={doNotShowAgain}
							onChange={(e: any) => {
								setDoNotShowAgain(e.target.checked === true)
							}}>
							不再显示
						</VSCodeCheckbox>
					</div>
				</div>
			</DialogContent>
		</Dialog>
	)
}

export default ZulongKanbanLaunchModal
