import { strict as assert } from "node:assert"
import { describe, it } from "mocha"
import { DEFAULT_ZULONG_IDE_WS_URL, normalizeZulongWebSocketUrl } from "../zulong-websocket"

describe("Zulong WebSocket URL normalization", () => {
	it("defaults to the IDE endpoint", () => {
		assert.equal(normalizeZulongWebSocketUrl(), DEFAULT_ZULONG_IDE_WS_URL)
	})

	it("adds /ide when only a host is configured", () => {
		assert.equal(normalizeZulongWebSocketUrl("ws://127.0.0.1:8090"), "ws://127.0.0.1:8090/ide")
		assert.equal(normalizeZulongWebSocketUrl("http://127.0.0.1:8090"), "ws://127.0.0.1:8090/ide")
	})

	it("preserves explicit custom paths", () => {
		assert.equal(normalizeZulongWebSocketUrl("wss://example.com/zulong/ide"), "wss://example.com/zulong/ide")
	})
})
