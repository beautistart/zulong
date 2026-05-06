import { VSCodeLink } from "@vscode/webview-ui-toolkit/react"
import Section from "../Section"

interface AboutSectionProps {
	version: string
	renderSectionHeader: (tabId: string) => JSX.Element | null
}
const AboutSection = ({ version, renderSectionHeader }: AboutSectionProps) => {
	return (
		<div>
			{renderSectionHeader("about")}
			<Section>
				<div className="flex px-4 flex-col gap-2">
					<h2 className="text-lg font-semibold">Zulong IDE v{version}</h2>
					<p>
						一个能够使用您的命令行和编辑器的 AI 助手。Zulong 可以使用工具逐步处理复杂的软件开发任务——创建和编辑文件、探索大型项目、使用浏览器以及执行终端命令（在您授权后）。
					</p>

					<h3 className="text-md font-semibold">社区与支持</h3>
					<p>
						<VSCodeLink href="https://x.com/zulong">X</VSCodeLink>
						{" • "}
						<VSCodeLink href="https://discord.gg/zulong">Discord</VSCodeLink>
						{" • "}
						<VSCodeLink href="https://www.reddit.com/r/zulong/"> r/zulong</VSCodeLink>
					</p>

					<h3 className="text-md font-semibold">开发</h3>
					<p>
						<VSCodeLink href="https://github.com/zulong/zulong">GitHub</VSCodeLink>
						{" • "}
						<VSCodeLink href="https://github.com/zulong/zulong/issues"> Issues</VSCodeLink>
						{" • "}
						<VSCodeLink href="https://github.com/zulong/zulong/discussions/categories/feature-requests?discussions_q=is%3Aopen+category%3A%22Feature+Requests%22+sort%3Atop">
							{" "}
							功能请求
						</VSCodeLink>
					</p>

					<h3 className="text-md font-semibold">资源</h3>
					<p>
						<VSCodeLink href="https://docs.zulong.ai/">文档</VSCodeLink>
						{" • "}
						<VSCodeLink href="https://zulong.ai/">https://zulong.ai</VSCodeLink>
					</p>
				</div>
			</Section>
		</div>
	)
}

export default AboutSection
