import { Empty, StringRequest } from "@shared/proto/zulong/common"
import * as vscode from "vscode"

const ZULONG_OUTPUT_CHANNEL = vscode.window.createOutputChannel("Zulong")

// Appends a log message to all Zulong output channels.
export async function debugLog(request: StringRequest): Promise<Empty> {
	ZULONG_OUTPUT_CHANNEL.appendLine(request.value)
	return Empty.create({})
}

// Register the Zulong output channel within the VSCode extension context.
export function registerZulongOutputChannel(context: vscode.ExtensionContext): vscode.OutputChannel {
	context.subscriptions.push(ZULONG_OUTPUT_CHANNEL)
	return ZULONG_OUTPUT_CHANNEL
}
