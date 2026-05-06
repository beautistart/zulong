import { test } from "@microsoft/tui-test"
import { ZULONG_BIN } from "./helpers/constants.js"
import { expectVisible, testEnv } from "./utils.js"

// ---------------------------------------------------------------------------
// zulong --version  (root flag)
// ---------------------------------------------------------------------------
test.describe("zulong --version", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--version"] },
		env: testEnv("claude-sonnet-4.6"),
	})

	test("prints the version string", async ({ terminal }) => {
		await expectVisible(terminal, /\d+\.\d+\.\d+/g)
	})
})

// ---------------------------------------------------------------------------
// zulong -V  (short flag)
// ---------------------------------------------------------------------------
test.describe("zulong -V", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["-V"] },
		env: testEnv("claude-sonnet-4.6"),
	})

	test("prints the version string with short flag", async ({ terminal }) => {
		await expectVisible(terminal, /\d+\.\d+\.\d+/g)
	})
})

// ---------------------------------------------------------------------------
// zulong version  (subcommand)
// ---------------------------------------------------------------------------
test.describe("zulong version subcommand", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["version"] },
		env: testEnv("claude-sonnet-4.6"),
	})

	test("prints 'Zulong CLI version:' message", async ({ terminal }) => {
		await expectVisible(terminal, ["Zulong CLI version:", /\d+\.\d+\.\d+/g])
	})
})
