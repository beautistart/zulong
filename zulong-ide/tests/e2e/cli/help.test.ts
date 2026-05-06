import { test } from "@microsoft/tui-test"
import { ZULONG_BIN } from "./helpers/constants.js"
import { expectVisible, testEnv } from "./utils.js"

const HELP_TERMINAL = { columns: 120, rows: 50 }

// ===========================================================================
// zulong --help  (root help)
// ===========================================================================
test.describe("zulong --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows Usage line and lists all subcommands", async ({ terminal }) => {
		await expectVisible(terminal, [
			"Usage:",
			"task|t",
			"history|h",
			"config [options]",
			"auth [options]",
			"version",
			"update [options]",
			"dev ",
		])
	})

	test("shows all root-level option flags", async ({ terminal }) => {
		await expectVisible(terminal, [
			"--act",
			"--plan",
			"--yolo",
			"--timeout",
			"--model",
			"--verbose",
			"--cwd",
			"--config",
			"--thinking",
			"--reasoning-effort",
			"--max-consecutive-mistakes",
			"--json",
			"--double-check-completion",
			"--acp",
			"--tui",
			"--taskId",
		])
	})
})

// ===========================================================================
// zulong -h  (short help flag)
// ===========================================================================
test.describe("zulong -h", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["-h"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows Usage line with short flag", async ({ terminal }) => {
		await expectVisible(terminal, "Usage:")
	})
})

// ===========================================================================
// zulong task --help
// ===========================================================================
test.describe("zulong task --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["task", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows task usage, prompt argument, and all flags", async ({ terminal }) => {
		await expectVisible(terminal, [
			"Usage:",
			"prompt",
			"--act",
			"--plan",
			"--yolo",
			"--timeout",
			"--model",
			"--verbose",
			"--cwd",
			"--config",
			"--thinking",
			"--reasoning-effort",
			"--max-consecutive-mistakes",
			"--json",
			"--double-check-completion",
			"--taskId",
		])
	})
})

// ===========================================================================
// zulong t --help  (task alias)
// ===========================================================================
test.describe("zulong t --help (task alias)", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["t", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows task usage and flags via alias", async ({ terminal }) => {
		await expectVisible(terminal, ["Usage:", "--yolo"])
	})
})

// ===========================================================================
// zulong history --help
// ===========================================================================
test.describe("zulong history --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["history", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows history usage and all flags", async ({ terminal }) => {
		await expectVisible(terminal, ["Usage:", "--limit", "--page", "--config"])
	})
})

// ===========================================================================
// zulong h --help  (history alias)
// ===========================================================================
test.describe("zulong h --help (history alias)", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["h", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows history usage and flags via alias", async ({ terminal }) => {
		await expectVisible(terminal, ["Usage:", "--limit"])
	})
})

// ===========================================================================
// zulong config --help
// ===========================================================================
test.describe("zulong config --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["config", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows config usage and --config flag", async ({ terminal }) => {
		await expectVisible(terminal, ["Usage:", "--config"])
	})
})

// ===========================================================================
// zulong auth --help
// ===========================================================================
test.describe("zulong auth --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["auth", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows auth usage and all flags", async ({ terminal }) => {
		await expectVisible(terminal, [
			"Usage:",
			"--provider",
			"--apikey",
			"--modelid",
			"--baseurl",
			"--verbose",
			"--cwd",
			"--config",
		])
	})
})

// ===========================================================================
// zulong version --help
// ===========================================================================
test.describe("zulong version --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["version", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows version command usage", async ({ terminal }) => {
		await expectVisible(terminal, "Usage:")
	})
})

// ===========================================================================
// zulong update --help
// ===========================================================================
test.describe("zulong update --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["update", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows update usage and --verbose flag", async ({ terminal }) => {
		await expectVisible(terminal, ["Usage:", "--verbose"])
	})
})

// ===========================================================================
// zulong dev --help
// ===========================================================================
test.describe("zulong dev --help", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["dev", "--help"] },
		env: testEnv("claude-sonnet-4.6"),
		...HELP_TERMINAL,
	})

	test("shows dev usage and lists log subcommand", async ({ terminal }) => {
		await expectVisible(terminal, ["Usage:", "log"])
	})
})
