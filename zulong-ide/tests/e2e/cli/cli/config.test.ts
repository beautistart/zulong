// ---------------------------------------------------------------------------
// zulong config — CLI tests
//
// Covers:
//   - `zulong config --config <dir>` — shows config for specific directory
//   - `zulong config --help`         — help page
// ---------------------------------------------------------------------------

import { test } from "@microsoft/tui-test"
import { ZULONG_BIN, TERMINAL_WIDE } from "../helpers/constants.js"
import { zulongEnv } from "../helpers/env.js"
import { expectVisible } from "../helpers/terminal.js"

// ---------------------------------------------------------------------------
// zulong config --help
// ---------------------------------------------------------------------------
test.describe("zulong config --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["config", "--help"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("shows config help page", async ({ terminal }) => {
		await expectVisible(terminal, "Usage:")
		await expectVisible(terminal, "--config")
	})
})

// ---------------------------------------------------------------------------
// zulong config --config <dir>
// Shows interactive config view for the specified directory
// ---------------------------------------------------------------------------
test.describe("zulong config (default config)", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["config"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})
})

test.describe("zulong config --config (claude-sonnet-4.6)", () => {
	test.use({
		program: {
			file: ZULONG_BIN,
			args: ["config", "--config", "configs/claude-sonnet-4.6"],
		},
		...TERMINAL_WIDE,
		env: zulongEnv("claude-sonnet-4.6"),
	})
})
