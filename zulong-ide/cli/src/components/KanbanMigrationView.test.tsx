import { render } from "ink-testing-library"
import { createElement } from "react"
import { describe, expect, it, vi } from "vitest"
import { KanbanMigrationView } from "./KanbanMigrationView"

describe("KanbanMigrationView", () => {
	it("renders the migration options", () => {
		const onSelect = vi.fn()
		const { lastFrame } = render(createElement(KanbanMigrationView, { isRawModeSupported: true, onSelect }))

		expect(lastFrame()).toContain("Introducing Zulong Kanban!")
		expect(lastFrame()).toContain("Open the new experience")
		expect(lastFrame()).toContain("Launch Zulong Kanban and start there by default.")
		expect(lastFrame()).toContain("zulong --tui")
		expect(lastFrame()).toContain("You can always run zulong --tui for the terminal experience.")
		expect(lastFrame()).toContain("Exit")
	})

	it("selects the highlighted option with Enter", () => {
		const onSelect = vi.fn()
		const { stdin } = render(createElement(KanbanMigrationView, { isRawModeSupported: true, onSelect }))

		stdin.write("\r")

		expect(onSelect).toHaveBeenCalledWith("kanban")
	})
})
