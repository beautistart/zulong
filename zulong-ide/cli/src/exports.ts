/**
 * Zulong Library Exports
 *
 * This file exports the public API for programmatic use of Zulong.
 * Use these classes and types to embed Zulong into your applications.
 *
 * @example
 * ```typescript
 * import { ZulongAgent } from "zulong"
 *
 * const agent = new ZulongAgent()
 * await agent.initialize({ clientCapabilities: {} })
 * const session = await agent.newSession({ cwd: process.cwd() })
 * ```
 * @module zulong
 */

export { ZulongAgent } from "./agent/ZulongAgent.js"
export { ZulongSessionEmitter } from "./agent/ZulongSessionEmitter.js"
export type {
	AcpAgentOptions,
	AcpSessionState,
	AcpSessionStatus,
	Agent,
	AgentSideConnection,
	AudioContent,
	CancelNotification,
	ClientCapabilities,
	ZulongAcpSession,
	ZulongAgentCapabilities,
	ZulongAgentInfo,
	ZulongAgentOptions,
	ZulongPermissionOption,
	ZulongSessionEvents,
	ContentBlock,
	ImageContent,
	InitializeRequest,
	InitializeResponse,
	LoadSessionRequest,
	LoadSessionResponse,
	McpServer,
	ModelInfo,
	NewSessionRequest,
	NewSessionResponse,
	PermissionHandler,
	PermissionOption,
	PermissionOptionKind,
	PromptRequest,
	PromptResponse,
	RequestPermissionRequest,
	RequestPermissionResponse,
	SessionConfigOption,
	SessionModelState,
	SessionNotification,
	SessionUpdate,
	SessionUpdatePayload,
	SessionUpdateType,
	SetSessionConfigOptionRequest,
	SetSessionConfigOptionResponse,
	SetSessionModelRequest,
	SetSessionModelResponse,
	SetSessionModeRequest,
	SetSessionModeResponse,
	StopReason,
	TextContent,
	ToolCall,
	ToolCallStatus,
	ToolCallUpdate,
	ToolKind,
	TranslatedMessage,
} from "./agent/public-types.js"
