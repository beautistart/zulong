import type { ZulongMessage } from "@shared/ExtensionMessage"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import ErrorRow from "./ErrorRow"

// Mock the auth context
vi.mock("@/context/ZulongAuthContext", () => ({
	useZulongAuth: () => ({
		zulongUser: null,
	}),
	useZulongSignIn: () => ({
		isLoginLoading: false,
	}),
	handleSignOut: vi.fn(),
}))

// Mock CreditLimitError component
vi.mock("@/components/chat/CreditLimitError", () => ({
	default: ({ message }: { message: string }) => <div data-testid="credit-limit-error">{message}</div>,
}))

// Mock ZulongError
vi.mock("../../../../src/services/error/ZulongError", () => ({
	ZulongError: {
		parse: vi.fn(),
	},
	ZulongErrorType: {
		Balance: "balance",
		RateLimit: "rateLimit",
		Auth: "auth",
	},
}))

describe("ErrorRow", () => {
	const mockMessage: ZulongMessage = {
		ts: 123456789,
		type: "say",
		say: "error",
		text: "Test error message",
	}

	beforeEach(() => {
		vi.clearAllMocks()
	})

	it("renders basic error message", () => {
		render(<ErrorRow errorType="error" message={mockMessage} />)

		expect(screen.getByText("Test error message")).toBeInTheDocument()
	})

	it("renders mistake limit reached error", () => {
		const mistakeMessage = { ...mockMessage, text: "Mistake limit reached" }
		render(<ErrorRow errorType="mistake_limit_reached" message={mistakeMessage} />)

		expect(screen.getByText("Mistake limit reached")).toBeInTheDocument()
	})

	it("renders diff error", () => {
		render(<ErrorRow errorType="diff_error" message={mockMessage} />)

		expect(
			screen.getByText("The model used search patterns that don't match anything in the file. Retrying..."),
		).toBeInTheDocument()
	})

	it("renders zulongignore error", () => {
		const zulongignoreMessage = { ...mockMessage, text: "/path/to/file.txt" }
		render(<ErrorRow errorType="zulongignore_error" message={zulongignoreMessage} />)

		expect(screen.getByText(/Zulong tried to access/)).toBeInTheDocument()
		expect(screen.getByText("/path/to/file.txt")).toBeInTheDocument()
	})

	describe("API error handling", () => {
		it("renders credit limit error when balance error is detected", async () => {
			const mockZulongError = {
				message: "Insufficient credits",
				isErrorType: vi.fn((type) => type === "balance"),
				_error: {
					details: {
						current_balance: 0,
						total_spent: 10.5,
						total_promotions: 5.0,
						message: "You have run out of credits.",
						buy_credits_url: "https://app.zulong.ai/dashboard",
					},
				},
			}

			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(mockZulongError as any)

			render(<ErrorRow apiRequestFailedMessage="Insufficient credits error" errorType="error" message={mockMessage} />)

			expect(screen.getByTestId("credit-limit-error")).toBeInTheDocument()
			expect(screen.getByText("You have run out of credits.")).toBeInTheDocument()
		})

		it("renders rate limit error with request ID", async () => {
			const mockZulongError = {
				message: "Rate limit exceeded",
				isErrorType: vi.fn((type) => type === "rateLimit"),
				_error: {
					request_id: "req_123456",
				},
			}

			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(mockZulongError as any)

			render(<ErrorRow apiRequestFailedMessage="Rate limit exceeded" errorType="error" message={mockMessage} />)

			expect(screen.getByText("Rate limit exceeded")).toBeInTheDocument()
			expect(screen.getByText("Request ID: req_123456")).toBeInTheDocument()
		})

		it("renders quota exceeded error", async () => {
			const mockZulongError = {
				message: "Inference cap reached",
				isErrorType: vi.fn((type) => type === "quotaexceeded"),
			}

			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(mockZulongError as any)

			render(<ErrorRow apiRequestFailedMessage="The message" errorType="error" message="" />)
			expect(screen.getByText("Inference cap reached")).toBeInTheDocument()
		})

		it("renders friendly logged-out message and sign in button when user is not signed in", async () => {
			const mockZulongError = {
				message: "Authentication failed",
				isErrorType: vi.fn((type) => type === "auth"),
				providerId: "zulong",
				_error: {},
			}

			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(mockZulongError as any)

			render(<ErrorRow apiRequestFailedMessage="Authentication failed" errorType="error" message={mockMessage} />)

			expect(screen.queryByText("Authentication failed")).not.toBeInTheDocument()
			expect(screen.getByText(/Whoops looks like you're logged out/)).toBeInTheDocument()
			expect(screen.getByText("Sign in to Zulong")).toBeInTheDocument()
		})

		it("renders PowerShell troubleshooting link when error mentions PowerShell", async () => {
			const mockZulongError = {
				message: "PowerShell is not recognized as an internal or external command",
				isErrorType: vi.fn(() => false),
				_error: {},
			}

			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(mockZulongError as any)

			render(
				<ErrorRow
					apiRequestFailedMessage="PowerShell is not recognized as an internal or external command"
					errorType="error"
					message={mockMessage}
				/>,
			)

			expect(screen.getByText(/PowerShell is not recognized/)).toBeInTheDocument()
			expect(screen.getByText("troubleshooting guide")).toBeInTheDocument()
			expect(screen.getByRole("link", { name: "troubleshooting guide" })).toHaveAttribute(
				"href",
				"https://github.com/zulong/zulong/wiki/TroubleShooting-%E2%80%90-%22PowerShell-is-not-recognized-as-an-internal-or-external-command%22",
			)
		})

		it("handles apiReqStreamingFailedMessage instead of apiRequestFailedMessage", async () => {
			const mockZulongError = {
				message: "Streaming failed",
				isErrorType: vi.fn(() => false),
				_error: {},
			}

			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(mockZulongError as any)

			render(<ErrorRow apiReqStreamingFailedMessage="Streaming failed" errorType="error" message={mockMessage} />)

			expect(screen.getByText("Streaming failed")).toBeInTheDocument()
		})

		it("falls back to regular error message when ZulongError.parse returns null", async () => {
			const { ZulongError } = await import("../../../../src/services/error/ZulongError")
			vi.mocked(ZulongError.parse).mockReturnValue(undefined)

			render(<ErrorRow apiRequestFailedMessage="Some API error" errorType="error" message={mockMessage} />)

			// When ZulongError.parse returns null, we display the raw error message for non-Zulong providers
			// Since zulongError is undefined, isZulongProvider is false, so we show the raw apiRequestFailedMessage
			expect(screen.getByText("Some API error")).toBeInTheDocument()
		})

		it("renders regular error message when no API error messages are provided", () => {
			render(<ErrorRow errorType="error" message={mockMessage} />)

			expect(screen.getByText("Test error message")).toBeInTheDocument()
		})
	})
})
