import fs from "node:fs/promises"
import { existsSync } from "node:fs"
import path from "node:path"
import * as vscode from "vscode"
import { DEFAULT_ZULONG_IDE_WS_URL, ZulongWebSocket, ZulongToolRequest } from "@core/api/transport/zulong-websocket"
import CheckpointTracker from "@integrations/checkpoints/CheckpointTracker"
import { DIFF_VIEW_URI_SCHEME } from "./VscodeDiffViewProvider"
import { getLatestTerminalOutput } from "./terminal/get-latest-output"
import { StateManager } from "@/core/storage/StateManager"
import type { ApprovalMode } from "@/shared/ApprovalWhitelist"
import { Logger } from "@/shared/services/Logger"
import { getShellForProfile } from "@/utils/shell"

type ToolResult = {
	callId: string
	toolName: string
	result: string
	isError?: boolean
}

export class VscodeExecutionBridge {
	private transport: ZulongWebSocket
	private connected = false
	private terminal: vscode.Terminal | undefined
	private terminalsByShell = new Map<string, vscode.Terminal>()
	private checkpointTracker: CheckpointTracker | undefined
	private checkpointInitPromise: Promise<CheckpointTracker | undefined> | undefined
	private pendingApprovals = new Map<string, (approved: boolean) => void>()
	private workspaceOverride: string | undefined
	private disposed = false
	private runtimeApprovalMode: ApprovalMode | undefined

	constructor(private readonly context: vscode.ExtensionContext) {
		const serverUrl = process.env.ZULONG_SERVER_URL || DEFAULT_ZULONG_IDE_WS_URL
		this.transport = new ZulongWebSocket(serverUrl)
		this.registerHandlers()
		this.context.subscriptions.push(
			vscode.workspace.onDidGrantWorkspaceTrust(() => {
				this.sendWorkspaceTrustStatus("granted")
				this.sendIdeContext()
			}),
		)
		void this.connect()
	}

	dispose(): void {
		this.disposed = true
		this.connected = false
		this.transport.dispose()
	}

	private async connect(): Promise<void> {
		try {
			await this.transport.connect()
			if (this.disposed) {
				return
			}
			this.connected = true
			Logger.info("[ZulongBridge] 后台执行桥已连接")
		} catch (error) {
			this.connected = false
			Logger.warn(`[ZulongBridge] 后台执行桥连接失败: ${error}`)
		}
	}

	private registerHandlers(): void {
		this.transport.on("tool_request", (request: ZulongToolRequest) => {
			void this.handleToolRequest(request)
		})
		this.transport.on("ide_open_workspace", (payload: Record<string, any>) => {
			void this.openWorkspace(payload)
		})
		this.transport.on("ide_open_file", (payload: Record<string, any>) => {
			void this.openFile(payload.path || payload.file_path, payload.line)
		})
		this.transport.on("ide_open_terminal", (payload: Record<string, any>) => {
			void this.openTerminal(payload.workspace_path || payload.cwd)
		})
		this.transport.on("ide_show_diff", (payload: Record<string, any>) => {
			void this.showDiff(payload)
		})
		this.transport.on("ide_get_context", () => {
			this.sendIdeContext()
		})
		this.transport.on("connected", () => {
			this.connected = true
			Logger.info("[ZulongBridge] WebSocket 已打开，发送 IDE 上下文")
			this.sendIdeContext()
		})
		this.transport.on("reconnecting", (attempt: number) => {
			this.connected = false
			Logger.info(`[ZulongBridge] 后台执行桥正在重连，第 ${attempt} 次`)
		})
		this.transport.on("disconnected", (code: number, reason: string) => {
			this.connected = false
			Logger.warn(`[ZulongBridge] 后台执行桥已断开: code=${code} reason=${reason || ""}`)
		})
		this.transport.on("error", (error: Error) => {
			this.connected = false
			Logger.warn(`[ZulongBridge] 后台执行桥事件错误: ${error.message}`)
		})
		this.transport.on("ide_approval_result", (payload: Record<string, any>) => {
			this.handleApprovalResult(payload)
		})
		this.transport.on("ide_runtime_settings", (payload: Record<string, any>) => {
			this.applyRuntimeSettings(payload)
		})
	}

	private async handleToolRequest(request: ZulongToolRequest): Promise<void> {
		const toolCalls = request.tool_calls || []
		for (const toolCall of toolCalls) {
			const callId = toolCall.id
			const toolName = toolCall.function?.name || ""
			let args: Record<string, any> = {}
			try {
				args = JSON.parse(toolCall.function?.arguments || "{}")
			} catch (error) {
				this.sendToolResult({ callId, toolName, result: `参数解析失败: ${error}`, isError: true })
				continue
			}

			try {
				this.sendBridgeInteraction(
					callId,
					"action",
					"running",
					`VS Code 开始执行: ${toolName}`,
					this.describeToolAction(toolName, args),
					toolName,
				)
				const result = await this.executeTool(toolName, args)
				this.sendToolResult({ callId, toolName, result, isError: this.isUserDeniedResult(result) })
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error)
				this.sendToolResult({ callId, toolName, result: message, isError: true })
			}
		}
	}

	private async executeTool(toolName: string, args: Record<string, any>): Promise<string> {
		this.ensureWorkspaceTrusted(toolName)
		switch (toolName) {
			case "read_file":
				return this.readFile(args)
			case "write_to_file":
				return this.writeFile(args)
			case "create_directory":
				return this.createDirectory(args)
			case "replace_in_file":
				return this.replaceInFile(args)
			case "delete_file":
				return this.deleteFile(args)
			case "list_files":
				return this.listFiles(args)
			case "search_files":
				return this.searchFiles(args)
			case "execute_command":
				return this.executeCommand(args)
			case "list_code_definition_names":
				return this.listCodeDefinitionNames(args)
			case "attempt_completion":
				return args.result || args.response || "任务完成"
			case "ask_followup_question":
				return "[Web端统一交互] 请在 Web 聊天页继续补充信息。"
			// ===== VS Code 完整控制工具 (TSD v2.7 扩充) =====
			case "vscode_run_command":
				return this.vscodeRunCommand(args)
			case "get_diagnostics":
				return this.getDiagnostics(args)
			case "ask_user_input":
				return this.askUserInput(args)
			case "ask_user_select_file":
				return this.askUserSelectFile(args)
			case "vscode_manage_extension":
				return this.vscodeManageExtension(args)
			case "open_settings":
				return this.openSettings(args)
			case "open_problems":
				return this.openProblems(args)
			// ===== 编辑器原生 API（实时编辑，进 undo stack）=====
			case "apply_edits":
				return this.applyEdits(args)
			default:
				throw new Error(`不支持的后台工具: ${toolName}`)
		}
	}

	private workspaceRoot(): string {
		return this.workspaceOverride || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || ""
	}

	private ensureWorkspaceTrusted(toolName: string): void {
		if (vscode.workspace.isTrusted) {
			return
		}
		throw new Error(`当前 VS Code 工作区尚未受信任，已暂停 ${toolName}。请在 VS Code 信任当前任务目录后，祖龙会自动继续。`)
	}

	private resolvePath(inputPath: string): string {
		if (!inputPath) {
			throw new Error("path 不能为空")
		}
		return path.isAbsolute(inputPath) ? inputPath : path.join(this.workspaceRoot(), inputPath)
	}

	private async readFile(args: Record<string, any>): Promise<string> {
		const filePath = this.resolvePath(args.path)
		this.ensureInsideWorkspace(filePath)
		const raw = await fs.readFile(filePath, "utf-8")
		const start = Number(args.start_line || 0)
		const end = Number(args.end_line || 0)
		if (start > 0 || end > 0) {
			const lines = raw.split(/\r?\n/)
			const from = Math.max(0, start > 0 ? start - 1 : 0)
			const to = end > 0 ? Math.min(lines.length, end) : lines.length
			return lines
				.slice(from, to)
				.map((line, idx) => `${from + idx + 1}: ${line}`)
				.join("\n")
		}
		return raw
	}

	private async writeFile(args: Record<string, any>): Promise<string> {
		const filePath = this.resolvePath(args.path)
		this.ensureInsideWorkspace(filePath)
		const chunk = args.content || ""
		const mode = String(args.mode || args.write_mode || "overwrite").toLowerCase()
		const existed = await this.fileExists(filePath)
		const original = existed ? await fs.readFile(filePath, "utf-8") : ""
		const content = mode === "append" ? `${original}${chunk}` : chunk
		const approved = await this.confirmFileChange({
			filePath,
			original,
			next: content,
			operation: existed ? "modify" : "create",
			summary: mode === "append" ? "追加文件内容" : existed ? "更新文件内容" : "创建新文件",
		})
		if (!approved) {
			return `用户未应用写入: ${filePath}`
		}
		await fs.mkdir(path.dirname(filePath), { recursive: true })
		await fs.writeFile(filePath, content, "utf-8")
		this.sendFileChanged(filePath, existed ? "updated" : "created")
		this.scheduleCheckpoint(
			`${mode === "append" ? "追加" : existed ? "更新" : "创建"} ${path.relative(this.workspaceRoot(), filePath)}`,
		)
		void this.openFile(filePath)
		return `已应用文件变更: ${filePath}`
	}

	private async createDirectory(args: Record<string, any>): Promise<string> {
		const dirPath = this.resolvePath(args.path || args.file_path)
		this.ensureInsideWorkspace(dirPath)
		const relPath = this.describePath(dirPath)
		const approved = await this.requestWebApproval(
			"create_directory",
			`创建文件夹 ${relPath}`,
			"medium",
			"将在宿主机文件系统中新建目录",
		)
		if (!approved) {
			return `用户未允许创建文件夹: ${dirPath}`
		}
		await fs.mkdir(dirPath, { recursive: true })
		this.sendFileChanged(dirPath, "created")
		this.scheduleCheckpoint(`创建文件夹 ${relPath}`)
		return `已创建文件夹: ${dirPath}`
	}

	private async replaceInFile(args: Record<string, any>): Promise<string> {
		const filePath = this.resolvePath(args.path)
		this.ensureInsideWorkspace(filePath)
		const original = await fs.readFile(filePath, "utf-8")
		const next = this.applySearchReplaceDiff(original, args.diff || "")
		const approved = await this.confirmFileChange({
			filePath,
			original,
			next,
			operation: "modify",
			summary: "按 SEARCH/REPLACE 差异更新文件",
		})
		if (!approved) {
			return `用户未应用替换: ${filePath}`
		}
		await fs.writeFile(filePath, next, "utf-8")
		this.sendFileChanged(filePath, "updated")
		this.scheduleCheckpoint(`替换 ${path.relative(this.workspaceRoot(), filePath)}`)
		void this.openFile(filePath)
		return `已应用文件替换: ${filePath}`
	}

	private applySearchReplaceDiff(original: string, diff: string): string {
		const blocks = diff.split("<<<<<<< SEARCH").slice(1)
		if (blocks.length === 0) {
			throw new Error("replace_in_file 需要 SEARCH/REPLACE 格式 diff")
		}
		let current = original
		for (const block of blocks) {
			const parts = block.split("=======")
			if (parts.length < 2) {
				throw new Error("diff 缺少 ======= 分隔符")
			}
			const search = parts[0].replace(/^\r?\n/, "")
			const replace = parts
				.slice(1)
				.join("=======")
				.split(">>>>>>> REPLACE")[0]
				.replace(/^\r?\n/, "")
			if (!current.includes(search)) {
				throw new Error(`未找到要替换的内容: ${search.slice(0, 120)}`)
			}
			current = current.replace(search, replace)
		}
		return current
	}

	private async deleteFile(args: Record<string, any>): Promise<string> {
		const filePath = this.resolvePath(args.path)
		this.ensureInsideWorkspace(filePath)
		const relPath = path.relative(this.workspaceRoot(), filePath)
		const approved = await this.requestWebApproval(
			"delete_file",
			`删除文件 ${relPath}`,
			"high",
			"删除动作会移除宿主机工作区文件",
		)
		if (!approved) {
			return `用户未允许删除: ${filePath}`
		}
		await fs.rm(filePath, { force: true })
		this.sendFileChanged(filePath, "deleted")
		this.scheduleCheckpoint(`删除 ${relPath}`)
		return `已删除文件: ${filePath}`
	}

	/** 编辑器原生 API：实时编辑（WorkspaceEdit，进 undo stack，用户可视） */
	private async applyEdits(args: Record<string, any>): Promise<string> {
		const filePath = this.resolvePath(args.path || args.file_path)
		this.ensureInsideWorkspace(filePath)
		const edits: any[] = args.edits || []
		if (!edits.length) {
			return "edits 参数为空，未执行任何编辑操作"
		}
		const uri = vscode.Uri.file(filePath)
		const we = new vscode.WorkspaceEdit()
		const summaryParts: string[] = []
		for (let i = 0; i < edits.length; i++) {
			const e = edits[i]
			const op = String(e.op || "").toLowerCase()
			if (op === "insert") {
				const line = Math.max(0, (Number(e.line) || 1) - 1)
				const text = String(e.text || "")
				we.insert(uri, new vscode.Position(line, 0), text)
				summaryParts.push(`L${line + 1} +${text.split("\n").length}行`)
			} else if (op === "delete") {
				const line = Math.max(0, (Number(e.line) || 1) - 1)
				const count = Math.max(1, Number(e.count) || 1)
				we.delete(
					uri,
					new vscode.Range(
						new vscode.Position(line, 0),
						new vscode.Position(line + count - 1, Number.MAX_SAFE_INTEGER),
					),
				)
				summaryParts.push(`L${line + 1} -${count}行`)
			} else if (op === "replace") {
				const sl = Math.max(0, (Number(e.start_line) || 1) - 1)
				const el = Math.max(sl, (Number(e.end_line) || sl + 1) - 1)
				const text = String(e.text || "")
				we.replace(
					uri,
					new vscode.Range(new vscode.Position(sl, 0), new vscode.Position(el, Number.MAX_SAFE_INTEGER)),
					text,
				)
				summaryParts.push(`L${sl + 1}-${el + 1} → ${text.split("\n").length}行`)
			} else if (op === "append") {
				// append = insert at end: read doc to get last line
				const doc = await vscode.workspace.openTextDocument(uri)
				const lastLine = doc.lineCount
				const text = String(e.text || "")
				we.insert(uri, new vscode.Position(lastLine, 0), "\n" + text)
				summaryParts.push(`末尾 +${text.split("\n").length}行`)
			}
		}
		const summary = summaryParts.join(", ") || `${edits.length} 个编辑操作`
		const approved = await this.requestWebApproval(
			"apply_edits",
			`编辑 ${path.relative(this.workspaceRoot(), filePath)} (${summary})`,
			"medium",
			"将通过 VS Code 编辑器原生 API 直接应用修改",
		)
		if (!approved) {
			return `用户未应用编辑: ${filePath}`
		}
		const applied = await vscode.workspace.applyEdit(we)
		if (!applied) {
			return `编辑未能应用（可能文件只读或被外部锁定）: ${filePath}`
		}
		this.sendFileChanged(filePath, "edited")
		this.scheduleCheckpoint(`编辑 ${path.relative(this.workspaceRoot(), filePath)} (${summary})`)
		void this.openFile(filePath)
		return `已应用 ${summary}: ${filePath}`
	}

	private async listFiles(args: Record<string, any>): Promise<string> {
		const root = this.resolvePath(args.path || ".")
		this.ensureInsideWorkspace(root)
		const recursive = String(args.recursive ?? "false") === "true" || args.recursive === true
		const entries: string[] = []
		await this.collectFiles(root, entries, recursive, root)
		return entries.join("\n")
	}

	private async collectFiles(dir: string, entries: string[], recursive: boolean, root: string): Promise<void> {
		const children = await fs.readdir(dir, { withFileTypes: true })
		for (const child of children) {
			if (child.name === "node_modules" || child.name === ".git") {
				continue
			}
			const full = path.join(dir, child.name)
			entries.push(path.relative(root, full) + (child.isDirectory() ? "/" : ""))
			if (recursive && child.isDirectory()) {
				await this.collectFiles(full, entries, recursive, root)
			}
		}
	}

	private async searchFiles(args: Record<string, any>): Promise<string> {
		const root = this.resolvePath(args.path || ".")
		this.ensureInsideWorkspace(root)
		const regex = new RegExp(args.regex || "", "i")
		const pattern = args.file_pattern ? new RegExp(String(args.file_pattern).replace(/\*/g, ".*")) : null
		const files: string[] = []
		await this.collectFiles(root, files, true, root)
		const matches: string[] = []
		for (const rel of files.filter((item) => !item.endsWith("/"))) {
			if (pattern && !pattern.test(rel)) {
				continue
			}
			const full = path.join(root, rel)
			try {
				const content = await fs.readFile(full, "utf-8")
				content.split(/\r?\n/).forEach((line, idx) => {
					if (regex.test(line)) {
						matches.push(`${rel}:${idx + 1}: ${line}`)
					}
				})
			} catch {
				// Binary or unreadable files are skipped.
			}
		}
		return matches.slice(0, 500).join("\n") || "未找到匹配内容"
	}

	private async executeCommand(args: Record<string, any>): Promise<string> {
		const command = args.command || ""
		if (!command) {
			throw new Error("command 不能为空")
		}
		const selected = await this.confirmCommand(command)
		if (!selected) {
			return `用户未允许执行命令: ${command}`
		}
		const shellProfile = this.normalizeCommandShell(args.shell || args.shell_type || args.terminal_shell || "auto")
		const shellPath = this.resolveCommandShellPath(shellProfile)
		const terminalKey = shellPath || "auto"
		const existing = this.terminalsByShell.get(terminalKey)
		const terminal =
			existing && existing.exitStatus === undefined
				? existing
				: vscode.window.createTerminal({
						name: shellProfile === "auto" ? "Zulong" : `Zulong ${shellProfile}`,
						cwd: this.workspaceRoot(),
						shellPath,
					})
		this.terminalsByShell.set(terminalKey, terminal)
		this.terminal = terminal
		terminal.show(false)
		terminal.sendText(command, true)
		this.transport.sendIdeTerminalStatus({
			workspace_path: this.workspaceRoot(),
			command,
			shell: shellProfile,
			status: "started",
		})
		return `命令已在 VS Code 终端执行: ${command} (shell=${shellProfile})`
	}

	private normalizeCommandShell(raw: unknown): "auto" | "cmd" | "powershell" | "git_bash" {
		const value = String(raw || "auto")
			.trim()
			.toLowerCase()
		if (!value || value === "default" || value === "system") {
			return "auto"
		}
		if (["cmd", "cmd.exe", "windows_cmd", "command_prompt"].includes(value)) {
			return "cmd"
		}
		if (["powershell", "pwsh", "powershell.exe", "ps"].includes(value)) {
			return "powershell"
		}
		if (["git_bash", "git-bash", "gitbash", "bash"].includes(value)) {
			return "git_bash"
		}
		return "auto"
	}

	private resolveCommandShellPath(shell: "auto" | "cmd" | "powershell" | "git_bash"): string | undefined {
		switch (shell) {
			case "cmd":
				return getShellForProfile("cmd")
			case "powershell":
				return existsSync(getShellForProfile("powershell-7"))
					? getShellForProfile("powershell-7")
					: getShellForProfile("powershell-legacy")
			case "git_bash":
				return getShellForProfile("git-bash")
			default:
				return undefined
		}
	}

	private async listCodeDefinitionNames(args: Record<string, any>): Promise<string> {
		return this.searchFiles({
			path: args.path || ".",
			regex: "^(export\\s+)?(class|function|interface|type|const|let|var)\\s+",
		})
	}

	private sendToolResult(result: ToolResult): void {
		this.transport.sendToolResult(result.callId, result.toolName, result.result, result.isError || false)
		this.sendBridgeInteraction(
			result.callId,
			"observation",
			result.isError ? "failed" : "succeeded",
			`${result.toolName || "工具"}执行${result.isError ? "失败" : "完成"}`,
			result.result.slice(0, 600),
			result.toolName,
		)
	}

	private isUserDeniedResult(result: string): boolean {
		const text = String(result || "").toLowerCase()
		return ["用户未应用", "用户未允许", "用户拒绝", "审批拒绝", "审批超时", "审批未通过", "未应用写入", "未允许"].some(
			(marker) => text.includes(marker.toLowerCase()),
		)
	}

	private async fileExists(filePath: string): Promise<boolean> {
		try {
			await fs.access(filePath)
			return true
		} catch {
			return false
		}
	}

	private ensureInsideWorkspace(filePath: string): void {
		if (!this.workspaceRoot()) {
			throw new Error("当前 VS Code 窗口没有打开任务工作区")
		}
		const root = path.resolve(this.workspaceRoot())
		const target = path.resolve(filePath)
		const relative = path.relative(root, target)
		if (relative.startsWith("..") || path.isAbsolute(relative)) {
			throw new Error(`拒绝访问工作区外路径: ${target}`)
		}
	}

	private describePath(filePath: string): string {
		if (!this.workspaceRoot()) {
			return path.resolve(filePath)
		}
		const root = path.resolve(this.workspaceRoot())
		const target = path.resolve(filePath)
		const relative = path.relative(root, target)
		if (!relative.startsWith("..") && !path.isAbsolute(relative)) {
			return relative || "."
		}
		return target
	}

	private describeToolAction(toolName: string, args: Record<string, any>): string {
		const target = args.path || args.file_path || args.command || args.regex || ""
		const labels: Record<string, string> = {
			read_file: "读取文件",
			write_to_file: "写入文件",
			create_directory: "创建文件夹",
			replace_in_file: "替换文件内容",
			delete_file: "删除文件",
			list_files: "列出文件",
			search_files: "搜索文件",
			execute_command: "运行终端命令",
			list_code_definition_names: "扫描代码定义",
			attempt_completion: "提交完成结果",
			ask_followup_question: "请求用户补充信息",
		}
		return `${labels[toolName] || toolName}${target ? `: ${String(target).slice(0, 180)}` : ""}`
	}

	private sendBridgeInteraction(
		pairId: string,
		kind: "plan" | "action" | "observation" | "progress" | "approval" | "summary" | "user_interject",
		status: string,
		title: string,
		detail: string,
		toolName?: string,
	): void {
		this.transport.sendIdeTerminalStatus({
			workspace_path: this.workspaceRoot(),
			status,
			message: title,
			tool_name: toolName,
			pair_id: pairId,
			interaction: {
				pair_id: pairId,
				kind,
				status,
				title,
				detail,
				tool_name: toolName || "",
			},
		})
	}

	private async confirmFileChange({
		filePath,
		original,
		next,
		operation,
		summary,
	}: {
		filePath: string
		original: string
		next: string
		operation: "create" | "modify"
		summary: string
	}): Promise<boolean> {
		this.ensureInsideWorkspace(filePath)
		const relPath = path.relative(this.workspaceRoot(), filePath)
		const actionSummary = `${summary}: ${relPath}`
		const approvalExtra = {
			path: filePath,
			operation,
			pairId: `file_change:${relPath}`,
		}
		const autoApproved = this.tryAutoApprove("write_file", actionSummary, "medium", approvalExtra)
		if (autoApproved !== undefined) {
			if (autoApproved) {
				this.sendDiffReady(filePath, operation, "approved")
			}
			return autoApproved
		}
		await this.openDiffPreview(filePath, original, next, `祖龙差异预览: ${relPath}`)
		const approved = await this.waitForWebApproval(actionSummary, "write_file", approvalExtra)
		if (!approved) {
			this.sendDiffReady(filePath, operation, "rejected")
			return false
		}
		this.sendDiffReady(filePath, operation, "approved")
		return true
	}

	private async openDiffPreview(filePath: string, original: string, next: string, title: string): Promise<void> {
		const left = Buffer.from(original, "utf-8").toString("base64")
		const right = Buffer.from(next, "utf-8").toString("base64")
		const basename = path.basename(filePath)
		const leftUri = vscode.Uri.parse(`${DIFF_VIEW_URI_SCHEME}:original-${basename}?${left}`)
		const rightUri = vscode.Uri.parse(`${DIFF_VIEW_URI_SCHEME}:zulong-${basename}?${right}`)
		await vscode.commands.executeCommand("vscode.diff", leftUri, rightUri, title, { preview: false })
	}

	private async confirmCommand(command: string): Promise<boolean> {
		const risk = this.commandRisk(command)
		return this.requestWebApproval("execute_command", `运行命令：${command}`, risk, "终端命令会在宿主机工作区执行", {
			pairId: `command:${command.slice(0, 120)}`,
			command,
		})
	}

	private commandRisk(command: string): "low" | "medium" | "high" {
		const value = command.toLowerCase()
		if (/\b(rm|del|remove-item|rd|rmdir|format|shutdown|restart-computer)\b/.test(value)) {
			return "high"
		}
		if (/[;&|`]/.test(command) || /\b(npm\s+i|npm\s+install|pip\s+install|pnpm\s+i|yarn\s+add)\b/.test(value)) {
			return "medium"
		}
		return "low"
	}

	private async getCheckpointTracker(): Promise<CheckpointTracker | undefined> {
		if (this.checkpointTracker) {
			return this.checkpointTracker
		}
		if (!this.checkpointInitPromise) {
			this.checkpointInitPromise = this.initializeCheckpointTracker()
		}
		this.checkpointTracker = await this.checkpointInitPromise
		return this.checkpointTracker
	}

	private async initializeCheckpointTracker(): Promise<CheckpointTracker | undefined> {
		try {
			const enableCheckpoints = StateManager.get().getGlobalSettingsKey("enableCheckpointsSetting") ?? true
			const tracker = await CheckpointTracker.create(`zulong-bridge-${Date.now()}`, enableCheckpoints, this.workspaceRoot())
			return tracker
		} catch (error) {
			Logger.warn(`[ZulongBridge] Checkpoint 初始化失败: ${error}`)
			return undefined
		}
	}

	private scheduleCheckpoint(summary: string): void {
		void this.createCheckpoint(summary)
	}

	private async createCheckpoint(summary: string): Promise<void> {
		const tracker = await this.withTimeout(this.getCheckpointTracker(), 8_000, `checkpoint tracker 初始化超时: ${summary}`)
		if (!tracker) {
			this.sendCheckpointStatus(summary, undefined, "skipped")
			return
		}
		try {
			const checkpointId = await this.withTimeout(tracker.commit(), 20_000, `checkpoint 创建超时: ${summary}`)
			this.sendCheckpointStatus(summary, checkpointId, "created")
		} catch (error) {
			Logger.warn(`[ZulongBridge] Checkpoint 创建失败: ${error}`)
			this.sendCheckpointStatus(summary, undefined, "failed", error instanceof Error ? error.message : String(error))
		}
	}

	private withTimeout<T>(promise: Promise<T>, timeoutMs: number, timeoutMessage: string): Promise<T> {
		let timer: NodeJS.Timeout | undefined
		const timeout = new Promise<T>((_, reject) => {
			timer = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs)
		})
		return Promise.race([promise, timeout]).finally(() => {
			if (timer) {
				clearTimeout(timer)
			}
		})
	}

	private sendApprovalRequired(
		toolName: string,
		actionSummary: string,
		riskLevel: string,
		riskReason: string,
		approvalId?: string,
		extra?: Record<string, any>,
	): void {
		this.transport.sendIdeApprovalStatus({
			workspace_path: this.workspaceRoot(),
			approval_id: approvalId,
			tool_name: toolName,
			action_summary: actionSummary,
			risk_level: riskLevel,
			risk_reason: riskReason,
			approval_mode: this.currentApprovalMode(),
			...(extra || {}),
			interaction: {
				approval_id: approvalId,
				pair_id: extra?.pairId || approvalId,
				kind: "approval",
				status: "awaiting_approval",
				title: "需要你确认",
				detail: actionSummary,
				tool_name: toolName,
				risk_level: riskLevel,
				risk_reason: riskReason,
				approval_mode: this.currentApprovalMode(),
				confirmation_state: "awaiting_confirmation",
			},
		})
	}

	private requestWebApproval(
		toolName: string,
		actionSummary: string,
		riskLevel: string,
		riskReason: string,
		extra?: Record<string, any>,
	): Promise<boolean> {
		const approvalId = `approval_${Date.now()}_${Math.random().toString(16).slice(2)}`
		const autoApproved = this.tryAutoApprove(toolName, actionSummary, riskLevel, extra, approvalId)
		if (autoApproved !== undefined) {
			return Promise.resolve(autoApproved)
		}
		this.sendApprovalRequired(toolName, actionSummary, riskLevel, riskReason, approvalId, {
			...(extra || {}),
			approval_mode: this.currentApprovalMode(),
		})
		return new Promise<boolean>((resolve) => {
			const timeout = setTimeout(() => {
				this.pendingApprovals.delete(approvalId)
				this.sendApprovalDecision(approvalId, toolName, actionSummary, false, extra)
				resolve(false)
			}, 60_000)
			this.pendingApprovals.set(approvalId, (approved) => {
				clearTimeout(timeout)
				this.sendApprovalDecision(approvalId, toolName, actionSummary, approved, extra)
				resolve(approved)
			})
		})
	}

	private waitForWebApproval(actionSummary: string, toolName: string, extra?: Record<string, any>): Promise<boolean> {
		const approvalId = `approval_${Date.now()}_${Math.random().toString(16).slice(2)}`
		const autoApproved = this.tryAutoApprove(toolName, actionSummary, "medium", extra, approvalId)
		if (autoApproved !== undefined) {
			return Promise.resolve(autoApproved)
		}
		this.transport.sendIdeApprovalStatus({
			workspace_path: this.workspaceRoot(),
			approval_id: approvalId,
			tool_name: toolName,
			action_summary: actionSummary,
			risk_level: "medium",
			risk_reason: "写入前会展示差异，只有 Web 页面确认后才保存",
			approval_mode: this.currentApprovalMode(),
			...(extra || {}),
			interaction: {
				approval_id: approvalId,
				pair_id: extra?.pairId || approvalId,
				kind: "approval",
				status: "awaiting_approval",
				title: "需要你确认文件变更",
				detail: actionSummary,
				tool_name: toolName,
				risk_level: "medium",
				risk_reason: "写入前会展示差异，只有 Web 页面确认后才保存",
				approval_mode: this.currentApprovalMode(),
				confirmation_state: "awaiting_confirmation",
			},
		})
		return new Promise<boolean>((resolve) => {
			const timeout = setTimeout(() => {
				this.pendingApprovals.delete(approvalId)
				this.sendApprovalDecision(approvalId, toolName, actionSummary, false, extra)
				resolve(false)
			}, 60_000)
			this.pendingApprovals.set(approvalId, (approved) => {
				clearTimeout(timeout)
				this.sendApprovalDecision(approvalId, toolName, actionSummary, approved, extra)
				resolve(approved)
			})
		})
	}

	private handleApprovalResult(payload: Record<string, any>): void {
		const approvalId = payload.approval_id || payload.approvalId || payload.interaction_id || payload.pair_id
		if (!approvalId) {
			return
		}
		const resolver = this.pendingApprovals.get(approvalId)
		if (!resolver) {
			return
		}
		this.pendingApprovals.delete(approvalId)
		resolver(payload.approved === true || payload.action === "approve")
	}

	private sendApprovalDecision(
		approvalId: string,
		toolName: string,
		actionSummary: string,
		approved: boolean,
		extra?: Record<string, any>,
	): void {
		const autoApproved = extra?.auto_approved === true
		this.transport.sendIdeApprovalStatus({
			workspace_path: this.workspaceRoot(),
			approval_id: approvalId,
			tool_name: toolName,
			action_summary: actionSummary,
			status: approved ? "approved" : "rejected",
			approved,
			...(extra || {}),
			interaction: {
				pair_id: extra?.pairId || approvalId,
				kind: "approval",
				status: approved ? "approved" : "rejected",
				title: autoApproved ? "自动审批已允许" : approved ? "用户已允许" : "用户已拒绝",
				detail: actionSummary,
				tool_name: toolName,
				approval_mode: extra?.approval_mode || this.currentApprovalMode(),
				auto_approved: autoApproved,
				confirmation_state: approved ? "confirmed" : "rejected",
			},
		})
	}

	private tryAutoApprove(
		toolName: string,
		actionSummary: string,
		riskLevel: string,
		extra?: Record<string, any>,
		approvalId?: string,
	): boolean | undefined {
		const mode = this.currentApprovalMode()
		if (mode !== "full_auto") {
			return undefined
		}
		const resolvedApprovalId = approvalId || `approval_${Date.now()}_${Math.random().toString(16).slice(2)}`
		Logger.info(`[ZulongBridge] 完全自动审批通过: tool=${toolName} risk=${riskLevel}`)
		this.sendApprovalDecision(resolvedApprovalId, toolName, actionSummary, true, {
			...(extra || {}),
			approval_mode: mode,
			auto_approved: true,
			risk_level: riskLevel,
		})
		return true
	}

	private currentApprovalMode(): ApprovalMode {
		if (this.runtimeApprovalMode) {
			return this.runtimeApprovalMode
		}
		try {
			const settings = StateManager.get().getGlobalSettingsKey("autoApprovalSettings")
			return this.normalizeApprovalMode(settings?.zulongAutoApproveMode)
		} catch {
			return "manual"
		}
	}

	private applyRuntimeSettings(payload: Record<string, any>): void {
		const mode = this.normalizeApprovalMode(payload?.approval_mode)
		this.runtimeApprovalMode = mode
		Logger.info(`[ZulongBridge] 运行时审批模式已同步: ${mode}`)
	}

	private normalizeApprovalMode(mode: unknown): ApprovalMode {
		switch (
			String(mode || "")
				.trim()
				.toLowerCase()
		) {
			case "full":
			case "full_auto":
				return "full_auto"
			case "read_only":
			case "whitelist":
				return "whitelist"
			case "popup":
				return "popup"
			case "manual":
			case "off":
			default:
				return "manual"
		}
	}

	private sendDiffReady(filePath: string, operation: string, status: "approved" | "rejected"): void {
		this.transport.sendIdeDiffStatus({
			workspace_path: this.workspaceRoot(),
			path: filePath,
			diff_operation: operation,
			status,
			pair_id: `file_change:${this.describePath(filePath)}`,
			interaction: {
				pair_id: `file_change:${this.describePath(filePath)}`,
				kind: "observation",
				status: status === "approved" ? "approved" : "rejected",
				title: status === "approved" ? "差异已允许" : "差异已拒绝",
				detail: `${operation}: ${this.describePath(filePath)}`,
				confirmation_state: status === "approved" ? "confirmed" : "rejected",
			},
		})
	}

	private sendCheckpointStatus(summary: string, checkpointId: string | undefined, status: string, error?: string): void {
		this.transport.sendIdeCheckpointStatus({
			workspace_path: this.workspaceRoot(),
			summary,
			checkpoint_id: checkpointId,
			status,
			error,
			pair_id: `checkpoint:${summary}`,
			interaction: {
				pair_id: `checkpoint:${summary}`,
				kind: "observation",
				status: status === "created" ? "succeeded" : status,
				title: status === "created" ? "Checkpoint 已创建" : "Checkpoint 状态更新",
				detail: error ? `${summary}: ${error}` : summary,
			},
		})
	}

	async openWorkspace(payload: string | Record<string, any> | undefined): Promise<void> {
		const workspacePath =
			typeof payload === "string" ? payload : payload?.workspace_path || payload?.cwd || payload?.workspace
		if (!workspacePath) {
			return
		}
		this.workspaceOverride = path.resolve(workspacePath)
		const activeFile = typeof payload === "string" ? undefined : payload?.active_file || payload?.file_path || payload?.path
		const line = typeof payload === "string" ? undefined : Number(payload?.line || payload?.start_line || 0)
		this.checkpointTracker = undefined
		this.checkpointInitPromise = undefined
		const currentWorkspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
		if (!currentWorkspace || path.resolve(currentWorkspace) !== this.workspaceOverride) {
			await vscode.commands.executeCommand("vscode.openFolder", vscode.Uri.file(this.workspaceOverride), false)
		}
		setTimeout(() => this.sendWorkspaceTrustStatus("opened"), 300)
		if (activeFile) {
			setTimeout(() => {
				void this.openFile(activeFile, line)
			}, 1200)
		}
		setTimeout(() => this.sendIdeContext(), 1500)
	}

	async openFile(filePath: string | undefined, line?: number): Promise<void> {
		if (!filePath) {
			return
		}
		const resolvedPath = this.resolvePath(filePath)
		if (this.workspaceRoot()) {
			this.ensureInsideWorkspace(resolvedPath)
		}
		const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(resolvedPath))
		const editor = await vscode.window.showTextDocument(doc, { preview: false })
		if (line && line > 0) {
			const position = new vscode.Position(line - 1, 0)
			editor.selection = new vscode.Selection(position, position)
			editor.revealRange(new vscode.Range(position, position))
		}
	}

	async openTerminal(cwd?: string): Promise<void> {
		if (!cwd && !this.workspaceRoot()) {
			this.transport.sendIdeTerminalStatus({
				workspace_path: "",
				status: "workspace_required",
				message: "当前 VS Code 窗口没有打开任务工作区。",
			})
			return
		}
		if (!this.terminal) {
			this.terminal = vscode.window.createTerminal({ name: "Zulong", cwd: cwd || this.workspaceRoot() })
		}
		this.terminal.show()
		this.transport.sendIdeTerminalStatus({
			workspace_path: cwd || this.workspaceRoot(),
			status: "opened",
		})
	}

	private sendFileChanged(filePath: string, operation: string): void {
		this.transport.sendIdeFileChanged({
			workspace_path: this.workspaceRoot(),
			path: filePath,
			operation,
		})
	}

	async showDiff(payload: Record<string, any>): Promise<void> {
		const filePath = this.resolvePath(payload.path || payload.file_path)
		const left = Buffer.from(payload.original || "", "utf-8").toString("base64")
		const leftUri = vscode.Uri.parse(`${DIFF_VIEW_URI_SCHEME}:${path.basename(filePath)}?${left}`)
		const rightUri = vscode.Uri.file(filePath)
		await vscode.commands.executeCommand(
			"vscode.diff",
			leftUri,
			rightUri,
			payload.title || `Zulong Diff: ${path.basename(filePath)}`,
		)
	}

	sendIdeContext(): void {
		const workspacePath = this.workspaceRoot()
		if (!workspacePath) {
			return
		}
		const activeEditor = vscode.window.activeTextEditor
		const payload = {
			workspace_path: workspacePath,
			active_file: activeEditor?.document.uri.fsPath,
			active_selection: activeEditor
				? {
						start_line: activeEditor.selection.start.line + 1,
						end_line: activeEditor.selection.end.line + 1,
					}
				: undefined,
			open_tabs: vscode.window.tabGroups.all.flatMap((group) =>
				group.tabs.map((tab) => (tab.input as any)?.uri?.fsPath).filter(Boolean),
			),
			workspace_trusted: vscode.workspace.isTrusted,
			trust_required: !vscode.workspace.isTrusted,
		}
		this.transport.sendIdeContext(payload)
	}

	private sendWorkspaceTrustStatus(status: "opened" | "granted" | "context"): void {
		this.transport.sendIdeTerminalStatus({
			workspace_path: this.workspaceRoot(),
			status: vscode.workspace.isTrusted ? "workspace_trusted" : "workspace_trust_required",
			workspace_trusted: vscode.workspace.isTrusted,
			trust_required: !vscode.workspace.isTrusted,
			message: vscode.workspace.isTrusted ? "VS Code 已信任当前任务目录。" : "VS Code 正在等待用户信任当前任务目录。",
			reason: status,
			interaction: {
				pair_id: `workspace_trust:${this.workspaceRoot()}`,
				kind: "approval",
				status: vscode.workspace.isTrusted ? "approved" : "pending",
				title: vscode.workspace.isTrusted ? "VS Code 已信任当前任务目录" : "等待 VS Code 目录信任",
				detail: vscode.workspace.isTrusted
					? "祖龙会自动继续当前任务。"
					: "请在 VS Code 的信任弹窗中确认该任务目录；确认后祖龙会自动继续。",
				tool_name: "workspace_trust",
			},
		})
	}

	async getTerminalSnapshot(): Promise<string> {
		return getLatestTerminalOutput()
	}

	// ===== VS Code 完整控制工具 (TSD v2.7 扩充) =====

	/** 命令安全分级 */
	private classifyCommand(command: string): "safe" | "high_risk" | "blocked" {
		// 阻止级 — 破坏 VS Code 稳定性
		const blocked = [
			"workbench.action.reloadWindow",
			"workbench.action.closeWindow",
			"workbench.action.quit",
			"workbench.extensions.installExtension",
			"workbench.extensions.uninstallExtension",
		]
		if (blocked.some((b) => command.startsWith(b))) {
			return "blocked"
		}

		// 高风险 — Git 操作、文件删除等
		const highRiskPrefixes = [
			"git.stage",
			"git.commit",
			"git.push",
			"git.pull",
			"git.clean",
			"git.revert",
			"deleteFile",
			"editor.action.clipboardCutAction",
		]
		if (highRiskPrefixes.some((p) => command.startsWith(p))) {
			return "high_risk"
		}

		return "safe"
	}

	private async vscodeRunCommand(args: Record<string, any>): Promise<string> {
		const command = args.command as string
		if (!command) {
			throw new Error("command 不能为空")
		}

		const classification = this.classifyCommand(command)
		if (classification === "blocked") {
			throw new Error(`命令被安全策略阻止: ${command}`)
		}
		if (classification === "high_risk") {
			const approved = await this.confirmCommandExecution(command, args.args || [])
			if (!approved) {
				return `用户拒绝了高风险命令: ${command}`
			}
		}

		try {
			const result = await vscode.commands.executeCommand(command, ...(args.args || []))
			return JSON.stringify({
				command,
				result: result !== undefined ? result : "(无返回值)",
				status: "success",
			})
		} catch (error) {
			return JSON.stringify({
				command,
				error: error instanceof Error ? error.message : String(error),
				status: "error",
			})
		}
	}

	private async confirmCommandExecution(command: string, args: string[]): Promise<boolean> {
		const autoApproved = this.tryAutoApprove("vscode_run_command", `执行 VS Code 命令：${command}`, "high", {
			command,
			args,
			pairId: `vscode_command:${command.slice(0, 120)}`,
		})
		if (autoApproved !== undefined) {
			return autoApproved
		}
		const detail = args.length > 0 ? `参数: ${args.join(", ")}` : "无参数"
		const choice = await vscode.window.showWarningMessage(
			`Zulong 请求执行高风险 VS Code 命令`,
			{ modal: true, detail: `${command}\n${detail}` },
			"批准执行",
			"拒绝",
		)
		return choice === "批准执行"
	}

	private async getDiagnostics(args: Record<string, any>): Promise<string> {
		const allDiagnostics = vscode.languages.getDiagnostics()
		// 按 severity 分组统计
		const errors: string[] = []
		const warnings: string[] = []
		const others: string[] = []

		const fileFilter = args.path ? this.resolvePath(args.path as string) : null

		for (const [uri, diags] of allDiagnostics) {
			if (diags.length === 0) continue
			if (fileFilter && uri.fsPath !== fileFilter) continue

			for (const d of diags) {
				const entry = `${uri.fsPath}:${d.range.start.line + 1}:${d.range.start.character + 1} [${d.source || "unknown"}] ${d.message}`
				switch (d.severity) {
					case vscode.DiagnosticSeverity.Error:
						errors.push(entry)
						break
					case vscode.DiagnosticSeverity.Warning:
						warnings.push(entry)
						break
					default:
						others.push(entry)
						break
				}
			}
		}

		const summary = {
			totalErrors: errors.length,
			totalWarnings: warnings.length,
			totalOthers: others.length,
			errors: errors.slice(0, 50),
			warnings: warnings.slice(0, 30),
			others: others.slice(0, 10),
		}

		return JSON.stringify(summary)
	}

	private async askUserInput(args: Record<string, any>): Promise<string> {
		const prompt = args.prompt as string
		if (!prompt) {
			throw new Error("prompt 不能为空")
		}
		const result = await vscode.window.showInputBox({
			prompt,
			placeHolder: args.placeholder as string | undefined,
			value: args.default_value as string | undefined,
		})
		return result || "(用户取消输入)"
	}

	private async askUserSelectFile(args: Record<string, any>): Promise<string> {
		const title = (args.title as string) || "选择文件"
		const isFolder = (args.type as string) === "folder"
		const result = await vscode.window.showOpenDialog({
			canSelectFiles: !isFolder,
			canSelectFolders: isFolder,
			canSelectMany: false,
			title,
		})
		if (!result || result.length === 0) {
			return "(用户取消选择)"
		}
		return result[0].fsPath
	}

	private async vscodeManageExtension(args: Record<string, any>): Promise<string> {
		const action = args.action as string
		if (!action) {
			throw new Error("action 不能为空")
		}

		switch (action) {
			case "list": {
				const extensions = vscode.extensions.all
				const list = extensions.map((ext) => ({
					id: ext.id,
					version: ext.packageJSON.version,
					isActive: ext.isActive,
				}))
				return JSON.stringify({ count: list.length, extensions: list })
			}
			case "install": {
				const extId = args.extension_id as string
				if (!extId) throw new Error("extension_id 不能为空")
				const approved = await this.confirmCommandExecution(`安装扩展: ${extId}`, [])
				if (!approved) return `用户拒绝了安装扩展: ${extId}`
				await vscode.commands.executeCommand("workbench.extensions.installExtension", extId)
				return `扩展安装请求已发送: ${extId}`
			}
			case "uninstall": {
				const extId = args.extension_id as string
				if (!extId) throw new Error("extension_id 不能为空")
				const approved = await this.confirmCommandExecution(`卸载扩展: ${extId}`, [])
				if (!approved) return `用户拒绝了卸载扩展: ${extId}`
				await vscode.commands.executeCommand("workbench.extensions.uninstallExtension", extId)
				return `扩展卸载请求已发送: ${extId}`
			}
			case "enable":
			case "disable": {
				const extId = args.extension_id as string
				if (!extId) throw new Error("extension_id 不能为空")
				const ext = vscode.extensions.getExtension(extId)
				if (!ext) throw new Error(`扩展未找到: ${extId}`)
				// 启用/禁用通过设置实现（简化版本）
				return `扩展 ${extId} ${action === "enable" ? "启用" : "禁用"}请求已记录（请手动操作）`
			}
			default:
				throw new Error(`不支持的操作: ${action}`)
		}
	}

	private async openSettings(_args: Record<string, any>): Promise<string> {
		await vscode.commands.executeCommand("workbench.action.openSettings")
		return "已打开 VS Code 设置面板"
	}

	private async openProblems(_args: Record<string, any>): Promise<string> {
		await vscode.commands.executeCommand("workbench.actions.view.problems")
		return "已打开 VS Code 问题面板"
	}
}
