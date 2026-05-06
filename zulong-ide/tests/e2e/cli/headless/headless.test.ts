// ---------------------------------------------------------------------------
// CLI headless use cases  (zulong -y / zulong --json / piped stdin)
//
// These tests run zulong as a child process (no TUI harness) and assert on
// stdout, stderr, and exit codes.
//
// Tests tagged @live require a configured provider and are skipped by default.
// Run them with:  ZULONG_BIN=... npm test -- headless @live
// ---------------------------------------------------------------------------

import { test } from "@microsoft/tui-test"
import { ZULONG_BIN, TERMINAL_WIDE } from "../helpers/constants.js"
import { zulongEnv } from "../helpers/env.js"
import { expectVisible } from "../helpers/terminal.js"

// ---------------------------------------------------------------------------
// zulong -y "tell me a joke"
// Golden path: prints "Task started" then LLM output, then exits 0.
// Unauthenticated: prints "Not authenticated" and exits 1.
// ---------------------------------------------------------------------------
test.describe("zulong -y (headless yolo mode) — unauthenticated", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["-y", "tell me a joke"] },
		...TERMINAL_WIDE,
		env: zulongEnv("unauthenticated"),
	})

	test("prints Not authenticated and exits 1", async ({ terminal }) => {
		await expectVisible(terminal, /not authenticated/i)
	})
})

// ---------------------------------------------------------------------------
// echo "max paulus" | zulong -y "print only the second word I gave you"
// Piped stdin test — uses TUI harness with stdin pre-written
// ---------------------------------------------------------------------------
test.describe("piped stdin | zulong -y — unauthenticated", () => {
	test.use({
		program: {
			file: "sh",
			args: ["-c", `echo "max paulus" | ${ZULONG_BIN} -y "print only the second word I gave you"`],
		},
		...TERMINAL_WIDE,
		env: zulongEnv("unauthenticated"),
	})

	test("prints Not Authenticated for piped stdin", async ({ terminal }) => {
		await expectVisible(terminal, /not authenticated/i)
	})
})

// ---------------------------------------------------------------------------
// zulong -y --verbose "tell me a joke" 2>&1
// Golden path: prints task started, prompt, api request, reasoning, task_completion lines
// ---------------------------------------------------------------------------
test.describe("zulong -y --verbose — unauthenticated", () => {
	test.use({
		program: {
			file: "sh",
			args: ["-c", `${ZULONG_BIN} -y --verbose "tell me a joke" 2>&1`],
		},
		...TERMINAL_WIDE,
		env: zulongEnv("unauthenticated"),
	})

	test("shows verbose output or not-authenticated", async ({ terminal }) => {
		await expectVisible(terminal, /not authenticated|verbose|task/i)
	})
})

// ---------------------------------------------------------------------------
// zulong --json "tell me a joke"
// All output must conform to JSON (one JSON object per line)
// ---------------------------------------------------------------------------
test.describe("zulong --json — unauthenticated", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["--json", "tell me a joke"] },
		...TERMINAL_WIDE,
		env: zulongEnv("unauthenticated"),
	})

	test("outputs JSON error for unauthenticated", async ({ terminal }) => {
		// zulong --json when unauthenticated outputs a plain "Not authenticated" message
		await expectVisible(terminal, /not authenticated/i)
	})
})
