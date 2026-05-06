// ---------------------------------------------------------------------------
// Environment helpers for test setup.
//
// Usage:
//   test.use({ env: zulongEnv("default") });
//   test.use({ env: zulongEnv("claude-sonnet-4.6") });
//   test.use({ env: zulongEnv("/absolute/path/to/config") });
// ---------------------------------------------------------------------------

import path from "path"

const TEST_SUITE_ROOT = new URL("../", import.meta.url).pathname

/**
 * Build the process environment for a zulong test.
 *
 * @param configDir - Named config under `configs/`, or an absolute path.
 * @param extra     - Additional env vars to merge in (override defaults).
 */
export function zulongEnv(configDir: string, extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
	const zulongPath = path.isAbsolute(configDir) ? configDir : path.join(TEST_SUITE_ROOT, "configs", configDir)

	// Remove CI env var so Ink's `is-in-ci` check doesn't disable interactive
	// rendering. When CI=true (set by GitHub Actions / act), Ink treats the
	// environment as non-interactive and skips rendering to stdout — even
	// inside a real PTY — which causes tui-test traces to be empty.
	const { CI: _ci, ...cleanEnv } = process.env

	return {
		...cleanEnv,
		ZULONG_TELEMETRY_DISABLED: "1",
		ZULONG_DIR: zulongPath,
		NO_UPDATE_NOTIFIER: "1",
		ZULONG_NO_AUTO_UPDATE: "1",
		...extra,
	}
}

/**
 * @deprecated Use `zulongEnv` instead. Kept for backward compatibility.
 */
export function testEnv(configDir: string): NodeJS.ProcessEnv {
	return zulongEnv(configDir)
}
