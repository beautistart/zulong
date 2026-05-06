import {
	ActivatedConditionalRule,
	getRemoteRulesTotalContentWithMetadata,
	getRuleFilesTotalContentWithMetadata,
	RULE_SOURCE_PREFIX,
	RuleLoadResultWithInstructions,
	synchronizeRuleToggles,
} from "@core/context/instructions/user-instructions/rule-helpers"
import { formatResponse } from "@core/prompts/responses"
import { ensureRulesDirectoryExists, GlobalFileNames } from "@core/storage/disk"
import { StateManager } from "@core/storage/StateManager"
import { ZulongRulesToggles } from "@shared/zulong-rules"
import { fileExistsAtPath, isDirectory, readDirectory } from "@utils/fs"
import fs from "fs/promises"
import path from "path"
import { Controller } from "@/core/controller"
import { Logger } from "@/shared/services/Logger"
import { parseYamlFrontmatter } from "./frontmatter"
import { evaluateRuleConditionals, type RuleEvaluationContext } from "./rule-conditionals"

export const getGlobalZulongRules = async (
	globalZulongRulesFilePath: string,
	toggles: ZulongRulesToggles,
	opts?: { evaluationContext?: RuleEvaluationContext },
): Promise<RuleLoadResultWithInstructions> => {
	let combinedContent = ""
	const activatedConditionalRules: ActivatedConditionalRule[] = []

	// 1. Get file-based rules
	if (await fileExistsAtPath(globalZulongRulesFilePath)) {
		if (await isDirectory(globalZulongRulesFilePath)) {
			try {
				const rulesFilePaths = await readDirectory(globalZulongRulesFilePath)
				// Note: ruleNamePrefix explicitly set to "global" for clarity (matches the default)
				const rulesFilesTotal = await getRuleFilesTotalContentWithMetadata(
					rulesFilePaths,
					globalZulongRulesFilePath,
					toggles,
					{
						evaluationContext: opts?.evaluationContext,
						ruleNamePrefix: "global",
					},
				)
				if (rulesFilesTotal.content) {
					combinedContent = rulesFilesTotal.content
					activatedConditionalRules.push(...rulesFilesTotal.activatedConditionalRules)
				}
			} catch {
				Logger.error(`Failed to read .zulongrules directory at ${globalZulongRulesFilePath}`)
			}
		} else {
			Logger.error(`${globalZulongRulesFilePath} is not a directory`)
		}
	}

	// 2. Append remote config rules
	const stateManager = StateManager.get()
	const remoteConfigSettings = stateManager.getRemoteConfigSettings()
	const remoteRules = remoteConfigSettings.remoteGlobalRules || []
	const remoteToggles = stateManager.getGlobalStateKey("remoteRulesToggles") || {}
	const remoteResult = getRemoteRulesTotalContentWithMetadata(remoteRules, remoteToggles, {
		evaluationContext: opts?.evaluationContext,
	})
	if (remoteResult.content) {
		if (combinedContent) combinedContent += "\n\n"
		combinedContent += remoteResult.content
		activatedConditionalRules.push(...remoteResult.activatedConditionalRules)
	}

	// 3. Return formatted instructions
	if (!combinedContent) {
		return { instructions: undefined, activatedConditionalRules: [] }
	}

	return {
		instructions: formatResponse.zulongRulesGlobalDirectoryInstructions(globalZulongRulesFilePath, combinedContent),
		activatedConditionalRules,
	}
}

export const getLocalZulongRules = async (
	cwd: string,
	toggles: ZulongRulesToggles,
	opts?: { evaluationContext?: RuleEvaluationContext },
): Promise<RuleLoadResultWithInstructions> => {
	const zulongRulesFilePath = path.resolve(cwd, GlobalFileNames.zulongRules)

	let instructions: string | undefined
	const activatedConditionalRules: ActivatedConditionalRule[] = []

	if (await fileExistsAtPath(zulongRulesFilePath)) {
		if (await isDirectory(zulongRulesFilePath)) {
			try {
				const rulesFilePaths = await readDirectory(zulongRulesFilePath, [
					[".zulongrules", "workflows"],
					[".zulongrules", "hooks"],
					[".zulongrules", "skills"],
				])

				const rulesFilesTotal = await getRuleFilesTotalContentWithMetadata(rulesFilePaths, cwd, toggles, {
					evaluationContext: opts?.evaluationContext,
					ruleNamePrefix: "workspace",
				})
				if (rulesFilesTotal.content) {
					instructions = formatResponse.zulongRulesLocalDirectoryInstructions(cwd, rulesFilesTotal.content)
					activatedConditionalRules.push(...rulesFilesTotal.activatedConditionalRules)
				}
			} catch {
				Logger.error(`Failed to read .zulongrules directory at ${zulongRulesFilePath}`)
			}
		} else {
			try {
				if (zulongRulesFilePath in toggles && toggles[zulongRulesFilePath] !== false) {
					const raw = (await fs.readFile(zulongRulesFilePath, "utf8")).trim()
					if (raw) {
						// Keep single-file .zulongrules behavior consistent with directory/remote rules:
						// - Parse YAML frontmatter (fail-open on parse errors)
						// - Evaluate conditionals against the request's evaluation context
						const parsed = parseYamlFrontmatter(raw)
						if (parsed.hadFrontmatter && parsed.parseError) {
							// Fail-open: preserve the raw contents so the LLM can still see the author's intent.
							instructions = formatResponse.zulongRulesLocalFileInstructions(cwd, raw)
						} else {
							const { passed, matchedConditions } = evaluateRuleConditionals(
								parsed.data,
								opts?.evaluationContext ?? {},
							)
							if (passed) {
								instructions = formatResponse.zulongRulesLocalFileInstructions(cwd, parsed.body.trim())
								if (parsed.hadFrontmatter && Object.keys(matchedConditions).length > 0) {
									activatedConditionalRules.push({
										name: `${RULE_SOURCE_PREFIX.workspace}:${GlobalFileNames.zulongRules}`,
										matchedConditions,
									})
								}
							}
						}
					}
				}
			} catch {
				Logger.error(`Failed to read .zulongrules file at ${zulongRulesFilePath}`)
			}
		}
	}

	return { instructions, activatedConditionalRules }
}

export async function refreshZulongRulesToggles(
	controller: Controller,
	workingDirectory: string,
): Promise<{
	globalToggles: ZulongRulesToggles
	localToggles: ZulongRulesToggles
}> {
	// Global toggles
	const globalZulongRulesToggles = controller.stateManager.getGlobalSettingsKey("globalZulongRulesToggles")
	const globalZulongRulesFilePath = await ensureRulesDirectoryExists()
	const updatedGlobalToggles = await synchronizeRuleToggles(globalZulongRulesFilePath, globalZulongRulesToggles)
	controller.stateManager.setGlobalState("globalZulongRulesToggles", updatedGlobalToggles)

	// Local toggles
	const localZulongRulesToggles = controller.stateManager.getWorkspaceStateKey("localZulongRulesToggles")
	const localZulongRulesFilePath = path.resolve(workingDirectory, GlobalFileNames.zulongRules)
	const updatedLocalToggles = await synchronizeRuleToggles(localZulongRulesFilePath, localZulongRulesToggles, "", [
		[".zulongrules", "workflows"],
		[".zulongrules", "hooks"],
		[".zulongrules", "skills"],
	])
	controller.stateManager.setWorkspaceState("localZulongRulesToggles", updatedLocalToggles)

	return {
		globalToggles: updatedGlobalToggles,
		localToggles: updatedLocalToggles,
	}
}
