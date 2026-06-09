import { Controller } from "@core/controller/index"
import { ZulongExtensionContext } from "@/shared/zulong"

export abstract class WebviewProvider {
	private static instance: WebviewProvider | null = null
	controller: Controller

	constructor(readonly context: ZulongExtensionContext) {
		WebviewProvider.instance = this

		// Create controller with cache service
		this.controller = new Controller(context)
	}

	async dispose() {
		await this.controller.dispose()
		WebviewProvider.instance = null
	}

	public static getInstance(): WebviewProvider {
		if (!WebviewProvider.instance) {
			throw new Error("WebviewProvider instance not initialized. Make sure to create a WebviewProvider instance first.")
		}
		return WebviewProvider.instance
	}

	public static getVisibleInstance(): WebviewProvider | undefined {
		return WebviewProvider.instance?.isVisible() ? WebviewProvider.instance : undefined
	}

	public static async disposeAllInstances() {
		if (WebviewProvider.instance) {
			await WebviewProvider.instance.dispose()
		}
	}

	/**
	 * Converts a local filesystem path to a URL that can be used within the webview.
	 *
	 * @param path - The local path to convert
	 * @returns A URL that can be used within the webview
	 */
	abstract getWebviewUrl(path: string): string

	/**
	 * Gets the Content Security Policy source for the webview.
	 *
	 * @returns The CSP source string to be used in the webview's Content-Security-Policy
	 */
	abstract getCspSource(): string

	/**
	 * Checks if the webview is currently visible to the user.
	 *
	 * @returns True if the webview is visible, false otherwise
	 */
	abstract isVisible(): boolean

	/**
	 * Historical host surface kept for standalone protocol compatibility.
	 * User interaction has moved to the Zulong Web entry.
	 */
	public getHtmlContent(): string {
		return /*html*/ `
			<!DOCTYPE html>
			<html lang="zh-CN">
				<head>
					<meta charset="utf-8">
					<meta name="viewport" content="width=device-width,initial-scale=1">
					<title>Zulong IDE Bridge</title>
				</head>
				<body>
					<main>
						<h1>祖龙 Web 统一入口已启用</h1>
						<p>VS Code 插件仅保留后台执行桥，请在祖龙 Web 端继续交互。</p>
					</main>
				</body>
			</html>
		`
	}

	protected async getHMRHtmlContent(): Promise<string> {
		return this.getHtmlContent()
	}
}
