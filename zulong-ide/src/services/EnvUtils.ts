import { isMultiRootWorkspace } from "@/core/workspace/utils/workspace-detection"
import { HostProvider } from "@/hosts/host-provider"
import { ExtensionRegistryInfo } from "@/registry"
import { EmptyRequest } from "@/shared/proto/zulong/common"
import { Logger } from "@/shared/services/Logger"

// Canonical header names for extra client/host context
export const ZulongHeaders = {
	PLATFORM: "X-PLATFORM",
	PLATFORM_VERSION: "X-PLATFORM-VERSION",
	CLIENT_VERSION: "X-CLIENT-VERSION",
	CLIENT_TYPE: "X-CLIENT-TYPE",
	CORE_VERSION: "X-CORE-VERSION",
	IS_MULTIROOT: "X-IS-MULTIROOT",
} as const
export type ZulongHeaderName = (typeof ZulongHeaders)[keyof typeof ZulongHeaders]

export function buildExternalBasicHeaders(): Record<string, string> {
	return {
		"User-Agent": `Zulong/${ExtensionRegistryInfo.version}`,
	}
}

export async function buildBasicZulongHeaders(): Promise<Record<string, string>> {
	const headers: Record<string, string> = buildExternalBasicHeaders()
	try {
		const host = await HostProvider.env.getHostVersion(EmptyRequest.create({}))
		headers[ZulongHeaders.PLATFORM] = host.platform || "unknown"
		headers[ZulongHeaders.PLATFORM_VERSION] = host.version || "unknown"
		headers[ZulongHeaders.CLIENT_TYPE] = host.zulongType || "unknown"
		headers[ZulongHeaders.CLIENT_VERSION] = host.zulongVersion || "unknown"
	} catch (error) {
		Logger.log("Failed to get IDE/platform info via HostBridge EnvService.getHostVersion", error)
		headers[ZulongHeaders.PLATFORM] = "unknown"
		headers[ZulongHeaders.PLATFORM_VERSION] = "unknown"
		headers[ZulongHeaders.CLIENT_TYPE] = "unknown"
		headers[ZulongHeaders.CLIENT_VERSION] = "unknown"
	}
	headers[ZulongHeaders.CORE_VERSION] = ExtensionRegistryInfo.version

	return headers
}

export async function buildZulongExtraHeaders(): Promise<Record<string, string>> {
	const headers = await buildBasicZulongHeaders()

	try {
		const isMultiRoot = await isMultiRootWorkspace()
		headers[ZulongHeaders.IS_MULTIROOT] = isMultiRoot ? "true" : "false"
	} catch (error) {
		Logger.log("Failed to detect multi-root workspace", error)
		headers[ZulongHeaders.IS_MULTIROOT] = "false"
	}

	return headers
}
