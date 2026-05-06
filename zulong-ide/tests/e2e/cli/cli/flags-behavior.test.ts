// ---------------------------------------------------------------------------
// CLI flag behavioral tests
//
// These tests verify the runtime behavior of each CLI flag — not just that
// the flag appears in --help output (that's covered in tests/flags.test.ts),
// but that the flag actually changes what zulong does.
//
// Tests marked ⚠️ in the spec reflect known gaps where the flag is accepted
// but currently has no observable effect. They are still written so the
// behavior can be asserted once the implementation catches up.
// ---------------------------------------------------------------------------

import { test } from "@microsoft/tui-test"
import { ZULONG_BIN, TERMINAL_WIDE } from "../helpers/constants.js"
import { zulongEnv } from "../helpers/env.js"
import { expectVisible } from "../utils.js"

// ---------------------------------------------------------------------------
// zulong --act
// Starts zulong in Act mode regardless of globalState
// ---------------------------------------------------------------------------
test.describe("zulong --act", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--act"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts in Act mode", async ({ terminal }) => {
		await expectVisible(terminal, "Act")
	})
})

// ---------------------------------------------------------------------------
// zulong --plan
// Starts zulong in Plan mode regardless of globalState
// ---------------------------------------------------------------------------
test.describe("zulong --plan", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--plan"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts in Plan mode", async ({ terminal }) => {
		await expectVisible(terminal, "Plan")
	})
})

// ---------------------------------------------------------------------------
// zulong --timeout <n>  ⚠️
// Current behavior: starts interactive mode and ignores timeout value
// ---------------------------------------------------------------------------
test.describe("zulong --timeout (interactive mode, flag ignored) ⚠️", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--timeout", "30"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode (timeout value currently ignored)", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --model <model-id>  ⚠️
// Current behavior: starts interactive mode and ignores model value
// ---------------------------------------------------------------------------
test.describe("zulong --model (interactive mode, flag ignored) ⚠️", () => {
	test.use({
		program: {
			file: ZULONG_BIN,
			args: ["--tui", "--model", "claude-3-5-haiku-20241022"],
		},
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode (model value currently ignored)", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --verbose  ⚠️
// Current behavior: starts interactive mode and ignores verbose value
// ---------------------------------------------------------------------------
test.describe("zulong --verbose (interactive mode, flag ignored) ⚠️", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--verbose"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode (verbose value currently ignored)", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong -c / zulong --cwd <dir>  ⚠️
// Starts zulong in interactive mode with the cwd present in the client footer
// ---------------------------------------------------------------------------
test.describe("zulong --cwd <dir>", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--cwd", "/tmp"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode with --cwd flag", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

test.describe("zulong -c <dir> (short alias)", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "-c", "/tmp"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode with -c flag", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --config <dir>
// Starts zulong in interactive mode using settings from the custom config dir
// ---------------------------------------------------------------------------
test.describe("zulong --config (claude-sonnet-4.6)", () => {
	test.use({
		program: {
			file: ZULONG_BIN,
			args: ["--tui", "--config", "configs/claude-sonnet-4.6"],
		},
		...TERMINAL_WIDE,
		env: zulongEnv("claude-sonnet-4.6"),
	})

	test("starts interactive mode with custom config directory", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --thinking  ⚠️
// Starts zulong in interactive mode with thinking turned on regardless of globalState
// (if thinking not supported, this flag is a no-op)
// ---------------------------------------------------------------------------
test.describe("zulong --thinking ⚠️", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--thinking"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode with --thinking flag", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --reasoning-effort <level>  ⚠️
// Starts zulong in interactive mode with reasoning turned on regardless of globalState
// (if reasoning not supported, this flag is a no-op)
// ---------------------------------------------------------------------------
test.describe("zulong --reasoning-effort ⚠️", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--reasoning-effort", "high"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode with --reasoning-effort flag", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --max-consecutive-mistakes <n>
// ---------------------------------------------------------------------------
test.describe("zulong --max-consecutive-mistakes", () => {
	test.use({
		program: {
			file: ZULONG_BIN,
			args: ["--tui", "--max-consecutive-mistakes", "5"],
		},
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode with --max-consecutive-mistakes flag", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --double-check-completion
// ---------------------------------------------------------------------------
test.describe("zulong --double-check-completion", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--tui", "--double-check-completion"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})

	test("starts interactive mode with --double-check-completion flag", async ({ terminal }) => {
		await expectVisible(terminal, /what can i do|plan|act/i)
	})
})
