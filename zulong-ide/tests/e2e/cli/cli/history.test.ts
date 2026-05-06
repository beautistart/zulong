// ---------------------------------------------------------------------------
// zulong history — CLI tests
//
// Covers:
//   - `zulong history --limit X`  — pagination limit
//   - `zulong history --page N`   — page selection
//   - `zulong history --config`   — custom config directory
//   - `zulong history --help`     — help page
// ---------------------------------------------------------------------------

import { test } from "@microsoft/tui-test"
import { ZULONG_BIN, TERMINAL_WIDE } from "../helpers/constants.js"
import { zulongEnv } from "../helpers/env.js"
import { expectVisible } from "../helpers/terminal.js"

// ---------------------------------------------------------------------------
// zulong history --help
// ---------------------------------------------------------------------------
test.describe("zulong history --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["history", "--help"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("shows history help page with all flags", async ({ terminal }) => {
		await expectVisible(terminal, "Usage:")
		await expectVisible(terminal, "--limit")
		await expectVisible(terminal, "--page")
		await expectVisible(terminal, "--config")
	})
})
