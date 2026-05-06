import { LightbulbIcon } from "lucide-react"
import { memo, useCallback, useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"

interface FeatureTipItem {
	text: string
}

const FEATURE_TIPS: FeatureTipItem[] = [
	{
		text: '在设置中启用"完成双重检查"，让 Zulong 在完成任务前验证其工作。',
	},
	{
		text: "在项目根目录添加 .zulongrules 文件，为 Zulong 提供项目特定指令。",
	},
	{
		text: "切换到计划模式，在 Zulong 执行操作前先讨论和规划方案。",
	},
	{
		text: "在聊天输入框中使用 @ 添加文件、文件夹或 URL 作为任务上下文。",
	},
	{
		text: "设置 MCP 服务器，让 Zulong 访问外部工具和 API。",
	},
	{
		text: "Zulong 在更改后创建检查点——您可以随时恢复到之前的状态。",
	},
	{
		text: "使用 /compact 压缩长对话，释放上下文窗口空间。",
	},
	{
		text: "为只读工具（如文件读取）启用自动批准，加快探索速度。",
	},
	{
		text: "使用引用按钮选择 Zulong 回复中的文本，并在您的回复中引用。",
	},
	{
		text: "您可以将图片拖放到聊天中，与 Zulong 分享截图。",
	},
	{
		text: "Zulong 可以浏览网站——让它在浏览器中测试您的本地开发服务器。",
	},
	{
		text: "使用 /reportbug 快速提交包含诊断上下文的 GitHub 问题。",
	},
	{
		text: '您可以在 设置 → 功能 → "功能提示" 中禁用这些提示。',
	},
]

const SHOW_DELAY_MS = 2000
const CYCLE_INTERVAL_MS = 8000
const FADE_DURATION_MS = 300

/**
 * Shows rotating feature tips below the "Thinking..." indicator.
 * Appears after a brief delay and cycles through tips while Zulong is thinking.
 */
export const FeatureTip = memo(() => {
	const [isVisible, setIsVisible] = useState(false)
	const [hasFadedIn, setHasFadedIn] = useState(false)
	const [isFading, setIsFading] = useState(false)
	const [tipIndex, setTipIndex] = useState(() => Math.floor(Math.random() * FEATURE_TIPS.length))
	const cycleTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
	const showTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
	const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

	const currentTip = FEATURE_TIPS[tipIndex]

	const advanceTip = useCallback(() => {
		setIsFading(true)
		fadeTimerRef.current = setTimeout(() => {
			setTipIndex((prev) => (prev + 1) % FEATURE_TIPS.length)
			setIsFading(false)
		}, FADE_DURATION_MS)
	}, [])

	useEffect(() => {
		showTimerRef.current = setTimeout(() => {
			setIsVisible(true)
			// Trigger fade-in on next frame so transition applies
			requestAnimationFrame(() => setHasFadedIn(true))
			cycleTimerRef.current = setInterval(advanceTip, CYCLE_INTERVAL_MS)
		}, SHOW_DELAY_MS)

		return () => {
			if (showTimerRef.current) {
				clearTimeout(showTimerRef.current)
			}
			if (cycleTimerRef.current) {
				clearInterval(cycleTimerRef.current)
			}
			if (fadeTimerRef.current) {
				clearTimeout(fadeTimerRef.current)
			}
		}
	}, [advanceTip])

	if (!isVisible) {
		return null
	}

	return (
		<div
			className={cn(
				"flex items-start gap-1.5 mt-2 ml-1 transition-opacity duration-300",
				!hasFadedIn || isFading ? "opacity-0" : "opacity-100",
			)}>
			<LightbulbIcon className="size-3 text-description shrink-0 mt-[1px]" />
			<span className="text-xs text-description leading-relaxed">
				<span className="font-medium">提示：</span> {currentTip.text}
			</span>
		</div>
	)
})

FeatureTip.displayName = "FeatureTip"
