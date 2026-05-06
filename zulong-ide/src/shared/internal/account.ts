/**
 * List of email domains that are considered trusted testers for Zulong.
 */
const ZULONG_TRUSTED_TESTER_DOMAINS = ["fibilabs.tech"]

/**
 * Checks if the given email belongs to a Zulong bot user.
 * E.g. Emails ending with @zulong.ai
 */
export function isZulongBotUser(email: string): boolean {
	return email.endsWith("@zulong.ai")
}

export function isZulongInternalTester(email: string): boolean {
	return isZulongBotUser(email) || ZULONG_TRUSTED_TESTER_DOMAINS.some((d) => email.endsWith(`@${d}`))
}
