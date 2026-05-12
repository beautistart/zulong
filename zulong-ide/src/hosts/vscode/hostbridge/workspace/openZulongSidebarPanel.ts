import * as vscode from "vscode"
import { ExtensionRegistryInfo } from "@/registry"
import { OpenZulongSidebarPanelRequest, OpenZulongSidebarPanelResponse } from "@/shared/proto/index.host"

export async function openZulongSidebarPanel(_: OpenZulongSidebarPanelRequest): Promise<OpenZulongSidebarPanelResponse> {
	await vscode.commands.executeCommand(`${ExtensionRegistryInfo.views.Sidebar}.focus`)
	return {}
}
