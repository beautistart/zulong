import { Boolean } from "@shared/proto/zulong/common"
import { isZulongCliInstalled } from "@/utils/cli-detector"
import { Controller } from ".."

/**
 * Check if the Zulong CLI is installed
 * @param controller The controller instance
 * @returns Boolean indicating if CLI is installed
 */
export async function checkCliInstallation(_controller: Controller): Promise<Boolean> {
	try {
		const isInstalled = await isZulongCliInstalled()
		return Boolean.create({ value: isInstalled })
	} catch {
		return Boolean.create({ value: false })
	}
}
