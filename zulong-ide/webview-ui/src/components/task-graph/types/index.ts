export type NodeType = "requirement" | "analysis" | "outline" | "task" | "subtask" | "note"

export type NodeStatus = "pending" | "in_progress" | "completed" | "blocked" | "skipped" | "needs_adjust" | "waiting_input" | "deleted"

export type EdgeType = "hierarchy" | "dependency" | "reference"

export type TaskDomain = "code" | "research" | "creative" | "data" | "general"

export type RenderMode = "full" | "full_optimized" | "virtual" | "degraded"

export type RenderStatus = "idle" | "loading" | "layouting" | "rendering" | "completed" | "degraded" | "error"

export type DegradationLevel = "none" | "skeleton_only" | "point_line" | "minimal"

export interface Position {
	x: number
	y: number
}

export interface FileRef {
	name: string
	path: string
}

export interface NodeAttribute {
	id: string
	label: string
	type: NodeType
	status: NodeStatus
	desc: string
	result: string
	files: FileRef[]
	taskDomain: TaskDomain
	parentId: string | null
	position: Position
	metadata: Record<string, unknown>
}

export interface EdgeAttribute {
	id: string
	source: string
	target: string
	type: EdgeType
	via: string
	cross: boolean
}

export interface ViewportState {
	offsetX: number
	offsetY: number
	zoom: number
	width: number
	height: number
}

export interface SelectionRect {
	x: number
	y: number
	width: number
	height: number
}

export interface RenderConfig {
	largeScaleThreshold: number
	virtualThreshold: number
	degradedThreshold: number
	viewportBufferRatio: number
	maxDOMNodes: number
	layoutTimeoutMs: number
	maxRetryCount: number
}

export const DEFAULT_RENDER_CONFIG: RenderConfig = {
	largeScaleThreshold: 200,
	virtualThreshold: 500,
	degradedThreshold: 2000,
	viewportBufferRatio: 0.2,
	maxDOMNodes: 500,
	layoutTimeoutMs: 10000,
	maxRetryCount: 2,
}

export interface RenderMetrics {
	firstRenderTime: number
	totalRenderTime: number
	domNodeCount: number
	fps: number
	memoryMB: number
	nodeCount: number
	renderMode: RenderMode
}

export interface ProgressInfo {
	totalNodes: number
	completedNodes: number
	percentage: number
	text: string
}

export interface SearchResult {
	nodeIds: string[]
	currentIndex: number
	total: number
}

export interface GraphUpdatePayload {
	action: string
	data: Record<string, unknown>
	sessionId: string
	timestamp: string
}

export interface GraphFullSyncPayload {
	id: string
	title: string
	nodes: Array<{
		id: string
		label: string
		type: string
		status: string
		desc: string
		result: string
		files: FileRef[]
		parentId: string | null
		taskDomain: TaskDomain
		metadata: Record<string, unknown>
	}>
	hEdges: Array<[string, string]>
	dEdges: Array<{
		s: string
		t: string
		via: string
		cross: boolean
	}>
}

export interface LayoutRequestMessage {
	type: "layout_request"
	nodes: Array<{ id: string; parentId: string | null }>
	edges: Array<{ source: string; target: string; type: string }>
	config: { algorithm: "hierarchy" | "force" | "force_incremental"; width: number; height: number }
}

export interface LayoutResultMessage {
	type: "layout_result"
	positions: Record<string, Position>
	durationMs: number
}

export interface LayoutPartialMessage {
	type: "layout_partial"
	positions: Record<string, Position>
	isPartial: boolean
}

export interface LayoutErrorMessage {
	type: "layout_error"
	error: string
}

export interface LayoutRefinedMessage {
	type: "layout_refined"
	positions: Record<string, Position>
}

export type LayoutWorkerMessage = LayoutResultMessage | LayoutPartialMessage | LayoutErrorMessage | LayoutRefinedMessage

export const NODE_STATUS_COLORS: Record<NodeStatus, string> = {
	pending: "#8b8b8b",
	in_progress: "#e2ab00",
	completed: "#388a34",
	blocked: "#e51400",
	skipped: "#6c6c6c",
	needs_adjust: "#ca6924",
	waiting_input: "#3774d8",
	deleted: "#4a4a4a",
}

export const NODE_TYPE_ICONS: Record<NodeType, string> = {
	requirement: "📋",
	analysis: "🔍",
	outline: "📝",
	task: "⚡",
	subtask: "🔹",
	note: "📌",
}

export const EDGE_TYPE_COLORS: Record<EdgeType, string> = {
	hierarchy: "#888888",
	dependency: "#6cb6ff",
	reference: "#4ec9b0",
}
