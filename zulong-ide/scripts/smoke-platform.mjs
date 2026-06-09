#!/usr/bin/env node
import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"
import path from "node:path"

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
const node = process.execPath
const tscEntrypoint = path.join(root, "node_modules", "typescript", "bin", "tsc")
const mochaEntrypoint = path.join(root, "node_modules", "mocha", "bin", "mocha.js")

function run(name, command, args, options = {}) {
	return new Promise((resolve) => {
		console.log(`\n== ${name} ==`)
		console.log([command, ...args].join(" "))
		const child = spawn(command, args, {
			cwd: root,
			stdio: "inherit",
			shell: false,
			env: { ...process.env, ...(options.env || {}) },
		})
		child.on("close", (code) => {
			const ok = code === 0
			console.log(`[${ok ? "OK" : "FAIL"}] ${name}`)
			resolve({ name, command: [command, ...args], returncode: code ?? 1, ok })
		})
	})
}

const steps = [
	["typecheck", node, [tscEntrypoint, "--noEmit", "--project", "tsconfig.json"]],
	[
		"zulong-websocket-url-test",
		node,
		[
			mochaEntrypoint,
			"--no-config",
			"--extension",
			"ts",
			"--require",
			"ts-node/register",
			"--require",
			"tsconfig-paths/register",
			"--require",
			"source-map-support/register",
			"--require",
			"./src/test/requires.ts",
			"src/core/api/transport/__tests__/zulong-websocket.test.ts",
		],
		{ env: { TS_NODE_PROJECT: "./tsconfig.unit-test.json" } },
	],
]

const results = []
for (const [name, command, args, options] of steps) {
	results.push(await run(name, command, args, options))
}

if (process.argv.includes("--json")) {
	console.log("")
	console.log(JSON.stringify({ ok: results.every((item) => item.ok), results }, null, 2))
}

process.exit(results.every((item) => item.ok) ? 0 : 1)
