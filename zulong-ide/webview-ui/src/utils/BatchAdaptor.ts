/**
 * 批次大小自适应策略
 * 
 * 根据网络延迟动态调整批次大小，优化传输效率
 * - 网络探测：每30秒ping一次后端，测量往返延迟
 * - 自适应规则：
 *   - 延迟≤200ms → 500节点/批
 *   - 延迟201-500ms → 300节点/批
 *   - 延迟>500ms → 100节点/批
 * - 使用指数移动平均平滑延迟测量值
 * - 批次大小调整添加冷却时间（30秒内不重复调整）
 */

export interface NetworkMetrics {
	latency: number
	lastPingTime: number
	measurements: number[]
	smoothedLatency: number
}

export interface BatchConfig {
	batchSize: number
	minBatchSize: number
	maxBatchSize: number
	adjustmentCooldown: number
	lastAdjustmentTime: number
}

const DEFAULT_CONFIG: BatchConfig = {
	batchSize: 500,
	minBatchSize: 100,
	maxBatchSize: 500,
	adjustmentCooldown: 30000,
	lastAdjustmentTime: 0,
}

const PING_INTERVAL = 30000 // 30秒
const SMOOTHING_FACTOR = 0.3 // 指数移动平均平滑因子
const MAX_MEASUREMENTS = 10 // 保留最近10次测量

export class BatchAdaptor {
	private static instance: BatchAdaptor | null = null

	private metrics: NetworkMetrics = {
		latency: 0,
		lastPingTime: 0,
		measurements: [],
		smoothedLatency: 0,
	}

	private config: BatchConfig = { ...DEFAULT_CONFIG }
	private pingInterval: NodeJS.Timeout | null = null
	private pingCallback: (() => Promise<number>) | null = null

	private constructor() {}

	static getInstance(): BatchAdaptor {
		if (!BatchAdaptor.instance) {
			BatchAdaptor.instance = new BatchAdaptor()
		}
		return BatchAdaptor.instance
	}

	/**
	 * 初始化网络探测
	 * @param pingCallback - 执行ping测量延迟的回调函数
	 */
	initialize(pingCallback: () => Promise<number>): void {
		this.pingCallback = pingCallback
		this.startPingInterval()
		console.log('[BatchAdaptor] Initialized with automatic ping interval')
	}

	/**
	 * 停止网络探测
	 */
	dispose(): void {
		if (this.pingInterval) {
			clearInterval(this.pingInterval)
			this.pingInterval = null
		}
		this.pingCallback = null
		console.log('[BatchAdaptor] Disposed')
	}

	/**
	 * 获取当前批次大小
	 */
	getBatchSize(): number {
		return this.config.batchSize
	}

	/**
	 * 获取网络指标
	 */
	getMetrics(): NetworkMetrics {
		return { ...this.metrics }
	}

	/**
	 * 获取配置
	 */
	getConfig(): BatchConfig {
		return { ...this.config }
	}

	/**
	 * 手动触发一次延迟测量
	 */
	async measureLatency(): Promise<number> {
		if (!this.pingCallback) {
			console.warn('[BatchAdaptor] No ping callback configured')
			return this.metrics.latency
		}

		try {
			const start = performance.now()
			await this.pingCallback()
			const latency = performance.now() - start

			this.recordMeasurement(latency)
			return latency
		} catch (e) {
			console.error('[BatchAdaptor] Ping failed:', e)
			return this.metrics.latency
		}
	}

	/**
	 * 记录延迟测量值
	 */
	recordMeasurement(latency: number): void {
		this.metrics.measurements.push(latency)
		if (this.metrics.measurements.length > MAX_MEASUREMENTS) {
			this.metrics.measurements.shift()
		}

		this.metrics.latency = latency
		this.metrics.lastPingTime = Date.now()

		this.updateSmoothedLatency()
		this.adjustBatchSize()
	}

	/**
	 * 更新平滑延迟值（指数移动平均）
	 */
	private updateSmoothedLatency(): void {
		if (this.metrics.measurements.length === 0) {
			return
		}

		const latest = this.metrics.measurements[this.metrics.measurements.length - 1]

		if (this.metrics.smoothedLatency === 0) {
			this.metrics.smoothedLatency = latest
		} else {
			this.metrics.smoothedLatency =
				SMOOTHING_FACTOR * latest + (1 - SMOOTHING_FACTOR) * this.metrics.smoothedLatency
		}

		console.log(`[BatchAdaptor] Smoothed latency: ${Math.round(this.metrics.smoothedLatency)}ms`)
	}

	/**
	 * 根据延迟调整批次大小
	 */
	private adjustBatchSize(): void {
		const now = Date.now()
		const timeSinceLastAdjustment = now - this.config.lastAdjustmentTime

		if (timeSinceLastAdjustment < this.config.adjustmentCooldown) {
			console.log(`[BatchAdaptor] Cooldown active, skipping adjustment (${Math.round(timeSinceLastAdjustment)}ms < ${this.config.adjustmentCooldown}ms)`)
			return
		}

		const latency = this.metrics.smoothedLatency
		let newBatchSize: number

		if (latency <= 200) {
			newBatchSize = 500
		} else if (latency <= 500) {
			newBatchSize = 300
		} else {
			newBatchSize = 100
		}

		newBatchSize = Math.max(this.config.minBatchSize, Math.min(this.config.maxBatchSize, newBatchSize))

		if (newBatchSize !== this.config.batchSize) {
			const oldBatchSize = this.config.batchSize
			this.config.batchSize = newBatchSize
			this.config.lastAdjustmentTime = now
			console.log(`[BatchAdaptor] Batch size adjusted: ${oldBatchSize} → ${newBatchSize} (latency: ${Math.round(latency)}ms)`)
		}
	}

	/**
	 * 启动定时探测
	 */
	private startPingInterval(): void {
		if (this.pingInterval) {
			clearInterval(this.pingInterval)
		}

		this.pingInterval = setInterval(async () => {
			if (this.pingCallback) {
				try {
					await this.measureLatency()
				} catch (e) {
					console.error('[BatchAdaptor] Ping interval error:', e)
				}
			}
		}, PING_INTERVAL)

		console.log(`[BatchAdaptor] Ping interval started (${PING_INTERVAL}ms)`)
	}

	/**
	 * 设置批次大小范围
	 */
	setBatchSizeRange(min: number, max: number): void {
		this.config.minBatchSize = min
		this.config.maxBatchSize = max
		this.config.batchSize = Math.max(min, Math.min(max, this.config.batchSize))
		console.log(`[BatchAdaptor] Batch size range updated: [${min}, ${max}], current: ${this.config.batchSize}`)
	}

	/**
	 * 设置调整冷却时间
	 */
	setAdjustmentCooldown(cooldown: number): void {
		this.config.adjustmentCooldown = cooldown
		console.log(`[BatchAdaptor] Adjustment cooldown updated: ${cooldown}ms`)
	}

	/**
	 * 重置为默认配置
	 */
	reset(): void {
		this.config = { ...DEFAULT_CONFIG }
		this.metrics = {
			latency: 0,
			lastPingTime: 0,
			measurements: [],
			smoothedLatency: 0,
		}
		console.log('[BatchAdaptor] Reset to default configuration')
	}

	/**
	 * 获取推荐的批次大小（不修改配置）
	 */
	getRecommendedBatchSize(latency: number): number {
		let recommended: number

		if (latency <= 200) {
			recommended = 500
		} else if (latency <= 500) {
			recommended = 300
		} else {
			recommended = 100
		}

		return Math.max(this.config.minBatchSize, Math.min(this.config.maxBatchSize, recommended))
	}

	/**
	 * 状态摘要
	 */
	getSummary(): string {
		return `BatchAdaptor: batchSize=${this.config.batchSize}, latency=${Math.round(this.metrics.smoothedLatency)}ms, measurements=${this.metrics.measurements.length}`
	}
}

export const batchAdaptor = BatchAdaptor.getInstance()
