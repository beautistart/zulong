import * as fs from "fs/promises"
import * as os from "os"
import * as path from "path"
import { Environment, type EnvironmentConfig } from "./shared/config-types"
import { Logger } from "./shared/services/Logger"

export { Environment, type EnvironmentConfig }

/**
 * Schema for the endpoints.json configuration file used in on-premise deployments.
 * All fields are required and must be valid URLs.
 */
interface EndpointsFileSchema {
	appBaseUrl: string
	apiBaseUrl: string
	mcpBaseUrl: string
}

/**
 * Error thrown when the Zulong configuration file exists but is invalid.
 * This error prevents Zulong from starting to avoid misconfiguration in enterprise environments.
 */
export class ZulongConfigurationError extends Error {
	constructor(message: string) {
		super(message)
		this.name = "ZulongConfigurationError"
	}
}

class ZulongEndpoint {
	private static _instance: ZulongEndpoint | null = null
	private static _initialized = false
	private static _extensionFsPath: string

	// On-premise config loaded from file (null if not on-premise)
	private onPremiseConfig: EndpointsFileSchema | null = null
	private environment: Environment = Environment.production
	// Track if config came from bundled file (enterprise distribution)
	private isBundled: boolean = false

	private constructor() {
		// Set environment at module load. Use override if provided.
		const _env = process?.env?.ZULONG_ENVIRONMENT_OVERRIDE || process?.env?.ZULONG_ENVIRONMENT
		if (_env && Object.values(Environment).includes(_env as Environment)) {
			this.environment = _env as Environment
		}
	}

	/**
	 * Initializes the ZulongEndpoint singleton.
	 * Must be called before any other methods.
	 * Reads the endpoints.json file if it exists and validates its schema.
	 *
	 * @param extensionFsPath Path to the extension installation directory (for checking bundled endpoints.json)
	 * @throws ZulongConfigurationError if the endpoints.json file exists but is invalid
	 */
	public static async initialize(extensionFsPath: string): Promise<void> {
		if (ZulongEndpoint._initialized) {
			return
		}

		ZulongEndpoint._extensionFsPath = extensionFsPath
		ZulongEndpoint._instance = new ZulongEndpoint()

		// Try to load on-premise config from file
		const endpointsConfig = await ZulongEndpoint.loadEndpointsFile()
		if (endpointsConfig) {
			ZulongEndpoint._instance.onPremiseConfig = endpointsConfig
			Logger.log("Zulong running in self-hosted mode with custom endpoints")
		}

		ZulongEndpoint._initialized = true
	}

	/**
	 * Returns true if the ZulongEndpoint has been initialized.
	 */
	public static isInitialized(): boolean {
		return ZulongEndpoint._initialized
	}

	/**
	 * Checks if Zulong is running in self-hosted/on-premise mode.
	 * @returns true if in selfHosted mode, or true if not initialized (safety fallback to prevent accidental external calls)
	 */
	public static isSelfHosted(): boolean {
		// Safety fallback: if not initialized, treat as selfHosted
		// to prevent accidental external service calls before configuration is loaded
		if (!ZulongEndpoint._initialized) {
			return true
		}
		return ZulongEndpoint.config.environment === Environment.selfHosted
	}

	/**
	 * Returns true if the current configuration was loaded from a bundled endpoints.json file.
	 * This indicates an enterprise distribution that should not auto-update.
	 * @throws Error if not initialized
	 */
	public static isBundledConfig(): boolean {
		if (!ZulongEndpoint._initialized || !ZulongEndpoint._instance) {
			throw new Error("ZulongEndpoint not initialized. Call ZulongEndpoint.initialize() first.")
		}
		return ZulongEndpoint._instance.isBundled
	}

	/**
	 * Returns the singleton instance.
	 * @throws Error if not initialized
	 */
	public static get instance(): ZulongEndpoint {
		if (!ZulongEndpoint._initialized || !ZulongEndpoint._instance) {
			throw new Error("ZulongEndpoint not initialized. Call ZulongEndpoint.initialize() first.")
		}
		return ZulongEndpoint._instance
	}

	/**
	 * Static getter for convenient access to the current configuration.
	 * @throws Error if not initialized
	 */
	public static get config(): EnvironmentConfig {
		return ZulongEndpoint.instance.config()
	}

	/**
	 * Returns the path to the endpoints.json configuration file.
	 * Located at ~/.zulong/endpoints.json
	 */
	private static getEndpointsFilePath(): string {
		return path.join(os.homedir(), ".zulong", "endpoints.json")
	}

	/**
	 * Returns the path to the bundled endpoints.json configuration file.
	 * Located in the extension installation directory.
	 */
	private static getBundledEndpointsFilePath(): string {
		return path.join(ZulongEndpoint._extensionFsPath, "endpoints.json")
	}

	/**
	 * Loads and validates the endpoints.json file.
	 * Checks bundled location first, then falls back to user directory.
	 * Priority: bundled endpoints.json → ~/.zulong/endpoints.json → null (standard mode)
	 * @returns The validated endpoints config, or null if no file exists
	 * @throws ZulongConfigurationError if a file exists but is invalid
	 */
	private static async loadEndpointsFile(): Promise<EndpointsFileSchema | null> {
		// 1. Try bundled file
		const bundledPath = ZulongEndpoint.getBundledEndpointsFilePath()
		try {
			await fs.access(bundledPath)
			// File exists, load and validate it
			const fileContent = await fs.readFile(bundledPath, "utf8")
			let data: unknown

			try {
				data = JSON.parse(fileContent)
			} catch (parseError) {
				throw new ZulongConfigurationError(
					`Invalid JSON in bundled endpoints configuration file (${bundledPath}): ${parseError instanceof Error ? parseError.message : String(parseError)}`,
				)
			}

			const config = ZulongEndpoint.validateEndpointsSchema(data, bundledPath)
			// Mark as bundled enterprise distribution
			ZulongEndpoint._instance!.isBundled = true
			return config
		} catch (error) {
			if (error instanceof ZulongConfigurationError) {
				throw error
			}
			// Bundled file doesn't exist or is not accessible, try user file
		}

		// 2. Try ~/.zulong/endpoints.json
		const userPath = ZulongEndpoint.getEndpointsFilePath()
		try {
			await fs.access(userPath)
		} catch {
			// File doesn't exist - not on-premise mode
			return null
		}

		// File exists, must be valid or we fail
		try {
			const fileContent = await fs.readFile(userPath, "utf8")
			let data: unknown

			try {
				data = JSON.parse(fileContent)
			} catch (parseError) {
				throw new ZulongConfigurationError(
					`Invalid JSON in user endpoints configuration file (${userPath}): ${parseError instanceof Error ? parseError.message : String(parseError)}`,
				)
			}

			return ZulongEndpoint.validateEndpointsSchema(data, userPath)
		} catch (error) {
			if (error instanceof ZulongConfigurationError) {
				throw error
			}
			throw new ZulongConfigurationError(
				`Failed to read user endpoints configuration file (${userPath}): ${error instanceof Error ? error.message : String(error)}`,
			)
		}
	}

	/**
	 * Validates that the provided data matches the EndpointsFileSchema.
	 * All fields must be present and be valid URLs.
	 *
	 * @param data The parsed JSON data to validate
	 * @param filePath The path to the file (for error messages)
	 * @returns The validated EndpointsFileSchema
	 * @throws ZulongConfigurationError if validation fails
	 */
	private static validateEndpointsSchema(data: unknown, filePath: string): EndpointsFileSchema {
		if (typeof data !== "object" || data === null) {
			throw new ZulongConfigurationError(`Endpoints configuration file (${filePath}) must contain a JSON object`)
		}

		const obj = data as Record<string, unknown>
		const requiredFields = ["appBaseUrl", "apiBaseUrl", "mcpBaseUrl"] as const
		const result: Partial<EndpointsFileSchema> = {}

		for (const field of requiredFields) {
			const value = obj[field]

			if (value === undefined || value === null) {
				throw new ZulongConfigurationError(
					`Missing required field "${field}" in endpoints configuration file (${filePath})`,
				)
			}

			if (typeof value !== "string") {
				throw new ZulongConfigurationError(
					`Field "${field}" in endpoints configuration file (${filePath}) must be a string`,
				)
			}

			if (!value.trim()) {
				throw new ZulongConfigurationError(
					`Field "${field}" in endpoints configuration file (${filePath}) cannot be empty`,
				)
			}

			// Validate URL format
			try {
				new URL(value)
			} catch {
				throw new ZulongConfigurationError(
					`Field "${field}" in endpoints configuration file (${filePath}) must be a valid URL. Got: "${value}"`,
				)
			}

			result[field] = value
		}

		return result as EndpointsFileSchema
	}

	/**
	 * Returns the current environment configuration.
	 */
	public config(): EnvironmentConfig {
		return this.getEnvironment()
	}

	/**
	 * Sets the current environment.
	 * @throws Error if in on-premise mode (environment switching is disabled)
	 */
	public setEnvironment(env: string) {
		if (this.onPremiseConfig) {
			throw new Error("Cannot change environment in on-premise mode. Endpoints are configured via ~/.zulong/endpoints.json")
		}

		switch (env.toLowerCase()) {
			case "staging":
				this.environment = Environment.staging
				break
			case "local":
				this.environment = Environment.local
				break
			default:
				this.environment = Environment.production
				break
		}
	}

	/**
	 * Returns the current environment configuration.
	 * If running in on-premise mode, returns the custom endpoints.
	 */
	public getEnvironment(): EnvironmentConfig {
		// On-premise mode: use custom endpoints from file
		if (this.onPremiseConfig) {
			return {
				environment: Environment.selfHosted,
				appBaseUrl: this.onPremiseConfig.appBaseUrl,
				apiBaseUrl: this.onPremiseConfig.apiBaseUrl,
				mcpBaseUrl: this.onPremiseConfig.mcpBaseUrl,
			}
		}

		// Standard mode: use built-in environment URLs
		switch (this.environment) {
			case Environment.staging:
				return {
					environment: Environment.staging,
					appBaseUrl: "https://staging-app.zulong.ai",
					apiBaseUrl: "https://core-api.staging.int.zulong.ai",
					mcpBaseUrl: "https://core-api.staging.int.zulong.ai/v1/mcp",
				}
			case Environment.local:
				return {
					environment: Environment.local,
					appBaseUrl: "http://localhost:3000",
					apiBaseUrl: "http://localhost:7777",
					mcpBaseUrl: "https://api.zulong.ai/v1/mcp",
				}
			default:
				return {
					environment: Environment.production,
					appBaseUrl: "https://app.zulong.ai",
					apiBaseUrl: "https://api.zulong.ai",
					mcpBaseUrl: "https://api.zulong.ai/v1/mcp",
				}
		}
	}
}

/**
 * Singleton instance to access the current environment configuration.
 * Usage:
 * - ZulongEnv.config() to get the current config.
 * - ZulongEnv.setEnvironment(Environment.local) to change the environment.
 *
 * IMPORTANT: ZulongEndpoint.initialize() must be called before using ZulongEnv.
 */
export const ZulongEnv = {
	config: () => ZulongEndpoint.config,
	setEnvironment: (env: string) => ZulongEndpoint.instance.setEnvironment(env),
	getEnvironment: () => ZulongEndpoint.instance.getEnvironment(),
}

// Export the class for initialization
export { ZulongEndpoint }
