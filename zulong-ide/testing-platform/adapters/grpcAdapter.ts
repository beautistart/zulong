import { AccountServiceClient } from "@zulong-grpc/account"
import { BrowserServiceClient } from "@zulong-grpc/browser"
import { CheckpointsServiceClient } from "@zulong-grpc/checkpoints"
import { CommandsServiceClient } from "@zulong-grpc/commands"
import { FileServiceClient } from "@zulong-grpc/file"
import { McpServiceClient } from "@zulong-grpc/mcp"
import { ModelsServiceClient } from "@zulong-grpc/models"
import { SlashServiceClient } from "@zulong-grpc/slash"
import { StateServiceClient } from "@zulong-grpc/state"
import { TaskServiceClient } from "@zulong-grpc/task"
import { UiServiceClient } from "@zulong-grpc/ui"
import { WebServiceClient } from "@zulong-grpc/web"
import { credentials } from "@grpc/grpc-js"
import { promisify } from "util"

const serviceRegistry = {
	"zulong.AccountService": AccountServiceClient,
	"zulong.BrowserService": BrowserServiceClient,
	"zulong.CheckpointsService": CheckpointsServiceClient,
	"zulong.CommandsService": CommandsServiceClient,
	"zulong.FileService": FileServiceClient,
	"zulong.McpService": McpServiceClient,
	"zulong.ModelsService": ModelsServiceClient,
	"zulong.SlashService": SlashServiceClient,
	"zulong.StateService": StateServiceClient,
	"zulong.TaskService": TaskServiceClient,
	"zulong.UiService": UiServiceClient,
	"zulong.WebService": WebServiceClient,
} as const

export type ServiceClients = {
	-readonly [K in keyof typeof serviceRegistry]: InstanceType<(typeof serviceRegistry)[K]>
}

export class GrpcAdapter {
	private clients: Partial<ServiceClients> = {}

	constructor(address: string) {
		for (const [name, Client] of Object.entries(serviceRegistry)) {
			this.clients[name as keyof ServiceClients] = new (Client as any)(address, credentials.createInsecure())
		}
	}

	async call(service: keyof ServiceClients, method: string, request: any): Promise<any> {
		const client = this.clients[service]
		if (!client) {
			throw new Error(`No gRPC client registered for service: ${String(service)}`)
		}

		const fn = (client as any)[method]
		if (typeof fn !== "function") {
			throw new Error(`Method ${method} not found on service ${String(service)}`)
		}

		try {
			const fnAsync = promisify(fn).bind(client)
			const response = await fnAsync(request.message)
			return response?.toObject ? response.toObject() : response
		} catch (error) {
			console.error(`[GrpcAdapter] ${service}.${method} failed:`, error)
			throw error
		}
	}

	close(): void {
		for (const client of Object.values(this.clients)) {
			if (client && typeof (client as any).close === "function") {
				;(client as any).close()
			}
		}
	}
}
