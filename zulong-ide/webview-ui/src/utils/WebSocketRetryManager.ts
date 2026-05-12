/**
 * WebSocket传输超时重试机制
 * 
 * 实现单批次传输超时检测（阈值5秒）
 * - 超时自动重试，最多重试2次
 * - 重试失败后标记该批次为丢失
 * - 重试策略：指数退避，间隔时间 1s, 2s
 */

export interface BatchTransmissionResult {
	success: boolean
	data?: any
	error?: string
	retryCount: number
	lostBatch?: boolean
}

export interface BatchConfig {
	batchId: string
	data: any
	timeout: number
	maxRetries: number
}

const DEFAULT_TIMEOUT = 5000 // 5秒
const DEFAULT_MAX_RETRIES = 2
const RETRY_DELAYS = [1000, 2000] // 指数退避：1s, 2s

export class WebSocketRetryManager {
	private static instance: WebSocketRetryManager | null = null
	private lostBatches: Set<string> = new Set()
	private retryStats: Map<string, { attempts: number; lastAttempt: number }> = new Map()

	private constructor() {}

	static getInstance(): WebSocketRetryManager {
		if (!WebSocketRetryManager.instance) {
			WebSocketRetryManager.instance = new WebSocketRetryManager()
		}
		return WebSocketRetryManager.instance
	}

	/**
	 * 发送批次数据，支持超时重试
	 */
	async sendWithRetry(
		sendFunction: (data: any) => Promise<any>,
		config: Partial<BatchConfig> = {},
	): Promise<BatchTransmissionResult> {
		const batchId = config.batchId || this.generateBatchId()
		const timeout = config.timeout || DEFAULT_TIMEOUT
		const maxRetries = config.maxRetries ?? DEFAULT_MAX_RETRIES
		const data = config.data

		let retryCount = 0
		let lastError: string | undefined

		for (let attempt = 0; attempt <= maxRetries; attempt++) {
			try {
				if (attempt > 0) {
					retryCount = attempt
					const delay = RETRY_DELAYS[attempt - 1] || RETRY_DELAYS[RETRY_DELAYS.length - 1]
					console.log(`[WSRetry] Retry attempt ${attempt}/${maxRetries} for batch ${batchId}, waiting ${delay}ms`)
					await this.sleep(delay)
				}

				const result = await this.sendWithTimeout(sendFunction, data, timeout)

				if (retryCount > 0) {
					this.recordRetrySuccess(batchId, retryCount)
				}

				return {
					success: true,
					data: result,
					retryCount,
				}
			} catch (error) {
				lastError = error instanceof Error ? error.message : String(error)
				console.error(`[WSRetry] Attempt ${attempt + 1} failed for batch ${batchId}:`, lastError)

				this.recordRetryAttempt(batchId, attempt)
			}
		}

		this.markBatchAsLost(batchId)
		console.error(`[WSRetry] Batch ${batchId} marked as lost after ${maxRetries + 1} attempts`)

		return {
			success: false,
			error: lastError,
			retryCount,
			lostBatch: true,
		}
	}

	/**
	 * 带超时的发送
	 */
	private async sendWithTimeout(
		sendFunction: (data: any) => Promise<any>,
		data: any,
		timeout: number,
	): Promise<any> {
		return Promise.race([
			sendFunction(data),
			this.createTimeoutPromise(timeout),
		])
	}

	/**
	 * 创建超时Promise
	 */
	private createTimeoutPromise(timeout: number): Promise<never> {
		return new Promise((_, reject) => {
			setTimeout(() => {
				reject(new Error(`Transmission timeout after ${timeout}ms`))
			}, timeout)
		})
	}

	/**
	 * 记录重试尝试
	 */
	private recordRetryAttempt(batchId: string, attempt: number): void {
		const stats = this.retryStats.get(batchId) || { attempts: 0, lastAttempt: 0 }
		stats.attempts = attempt + 1
		stats.lastAttempt = Date.now()
		this.retryStats.set(batchId, stats)
	}

	/**
	 * 记录重试成功
	 */
	private recordRetrySuccess(batchId: string, retryCount: number): void {
		console.log(`[WSRetry] Batch ${batchId} succeeded after ${retryCount} retries`)
		this.retryStats.delete(batchId)
	}

	/**
	 * 标记批次为丢失
	 */
	private markBatchAsLost(batchId: string): void {
		this.lostBatches.add(batchId)
	}

	/**
	 * 检查批次是否丢失
	 */
	isBatchLost(batchId: string): boolean {
		return this.lostBatches.has(batchId)
	}

	/**
	 * 获取所有丢失的批次
	 */
	getLostBatches(): string[] {
		return Array.from(this.lostBatches)
	}

	/**
	 * 清除丢失的批次记录
	 */
	clearLostBatches(): void {
		this.lostBatches.clear()
		console.log("[WSRetry] Cleared all lost batches")
	}

	/**
	 * 获取重试统计
	 */
	getRetryStats(batchId: string): { attempts: number; lastAttempt: number } | undefined {
		return this.retryStats.get(batchId)
	}

	/**
	 * 获取所有重试统计
	 */
	getAllRetryStats(): Map<string, { attempts: number; lastAttempt: number }> {
		return new Map(this.retryStats)
	}

	/**
	 * 清理过期的重试统计
	 */
	cleanupExpiredStats(maxAge: number = 60000): void {
		const now = Date.now()
		for (const [batchId, stats] of this.retryStats) {
			if (now - stats.lastAttempt > maxAge) {
				this.retryStats.delete(batchId)
			}
		}
	}

	/**
	 * 生成批次ID
	 */
	private generateBatchId(): string {
		return `batch_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
	}

	/**
	 * 延迟函数
	 */
	private sleep(ms: number): Promise<void> {
		return new Promise((resolve) => setTimeout(resolve, ms))
	}

	/**
	 * 获取统计摘要
	 */
	getSummary(): {
		lostBatchCount: number
		activeRetryCount: number
	} {
		return {
			lostBatchCount: this.lostBatches.size,
			activeRetryCount: this.retryStats.size,
		}
	}

	/**
	 * 重置
	 */
	reset(): void {
		this.lostBatches.clear()
		this.retryStats.clear()
		console.log("[WSRetry] Reset")
	}
}

export const wsRetryManager = WebSocketRetryManager.getInstance()

/**
 * 包装WebSocket发送函数，添加超时重试
 */
export function createRetryableSendFunction(
	originalSend: (data: any) => Promise<any>,
): (data: any, batchId?: string) => Promise<BatchTransmissionResult> {
	return async (data: any, batchId?: string) => {
		return wsRetryManager.sendWithRetry(originalSend, {
			batchId,
			data,
		})
	}
}
