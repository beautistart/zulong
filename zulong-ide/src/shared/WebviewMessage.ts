export interface WebviewMessage {
	type: "grpc_request" | "grpc_request_cancel" | "ide_approval_result"
	grpc_request?: GrpcRequest
	grpc_request_cancel?: GrpcCancel
	ide_approval_result?: IdeApprovalResult
	text?: string
}

export type GrpcRequest = {
	service: string
	method: string
	message: any // JSON serialized protobuf message
	request_id: string // For correlating requests and responses
	is_streaming: boolean // Whether this is a streaming request
}

export type GrpcCancel = {
	request_id: string // ID of the request to cancel
}

export type IdeApprovalResult = {
	approval_id: string
	approvalId?: string
	interaction_id?: string
	pair_id?: string
	approved: boolean
	action?: "approve" | "reject"
	add_to_whitelist?: string
	tool_name?: string
	action_summary?: string
	risk_level?: string
	workspace_path?: string
	cwd?: string
}

export type ZulongAskResponse = "yesButtonClicked" | "noButtonClicked" | "messageResponse"

export type ZulongCheckpointRestore = "task" | "workspace" | "taskAndWorkspace"

export type TaskFeedbackType = "thumbs_up" | "thumbs_down"
