// ---------------------------------------------------------------------------
// zulong dev — CLI tests
//
// Covers:
//   - `zulong dev log`
// ---------------------------------------------------------------------------

import { test } from "@microsoft/tui-test"
import { ZULONG_BIN, TERMINAL_WIDE } from "../helpers/constants.js"
import { zulongEnv } from "../helpers/env.js"

test.describe("zulong dev log", () => {
	test.use({
		program: { file: ZULONG_BIN, args: ["dev", "log"] },
		...TERMINAL_WIDE,
		env: zulongEnv("default"),
	})
})
