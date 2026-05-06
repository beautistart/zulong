import { getWorkspaceBasename } from "@core/workspace"
import type { ToggleZulongRuleRequest } from "@shared/proto/zulong/file"
import { RuleScope, ToggleZulongRules } from "@shared/proto/zulong/file"
import { telemetryService } from "@/services/telemetry"
import { Logger } from "@/shared/services/Logger"
import type { Controller } from "../index"

/**
 * Toggles a Zulong rule (enable or disable)
 * @param controller The controller instance
 * @param request The toggle request
 * @returns The updated Zulong rule toggles
 */
export async function toggleZulongRule(controller: Controller, request: ToggleZulongRuleRequest): Promise<ToggleZulongRules> {
	const { scope, rulePath, enabled } = request

	if (!rulePath || typeof enabled !== "boolean" || scope === undefined) {
		Logger.error("toggleZulongRule: Missing or invalid parameters", {
			rulePath,
			scope,
			enabled: typeof enabled === "boolean" ? enabled : `Invalid: ${typeof enabled}`,
		})
		throw new Error("Missing or invalid parameters for toggleZulongRule")
	}

	// Handle the three different scopes
	switch (scope) {
		case RuleScope.GLOBAL: {
			const toggles = controller.stateManager.getGlobalSettingsKey("globalZulongRulesToggles")
			toggles[rulePath] = enabled
			controller.stateManager.setGlobalState("globalZulongRulesToggles", toggles)
			break
		}
		case RuleScope.LOCAL: {
			const toggles = controller.stateManager.getWorkspaceStateKey("localZulongRulesToggles")
			toggles[rulePath] = enabled
			controller.stateManager.setWorkspaceState("localZulongRulesToggles", toggles)
			break
		}
		case RuleScope.REMOTE: {
			const toggles = controller.stateManager.getGlobalStateKey("remoteRulesToggles")
			toggles[rulePath] = enabled
			controller.stateManager.setGlobalState("remoteRulesToggles", toggles)
			break
		}
		default:
			throw new Error(`Invalid scope: ${scope}`)
	}

	// Track rule toggle telemetry with current task context
	if (controller.task?.ulid) {
		// Extract just the filename for privacy (no full paths)
		const ruleFileName = getWorkspaceBasename(rulePath, "Controller.toggleZulongRule")
		const isGlobal = scope === RuleScope.GLOBAL
		telemetryService.captureZulongRuleToggled(controller.task.ulid, ruleFileName, enabled, isGlobal)
	}

	// Get the current state to return in the response
	const globalToggles = controller.stateManager.getGlobalSettingsKey("globalZulongRulesToggles")
	const localToggles = controller.stateManager.getWorkspaceStateKey("localZulongRulesToggles")
	const remoteToggles = controller.stateManager.getGlobalStateKey("remoteRulesToggles")

	return ToggleZulongRules.create({
		globalZulongRulesToggles: { toggles: globalToggles },
		localZulongRulesToggles: { toggles: localToggles },
		remoteRulesToggles: { toggles: remoteToggles },
	})
}
