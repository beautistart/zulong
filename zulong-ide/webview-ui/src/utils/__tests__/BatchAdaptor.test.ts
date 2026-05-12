/**
 * BatchAdaptor 单元测试
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { BatchAdaptor, batchAdaptor } from '../BatchAdaptor'

describe('BatchAdaptor', () => {
	let adaptor: BatchAdaptor

	beforeEach(() => {
		adaptor = BatchAdaptor.getInstance()
		adaptor.reset()
	})

	afterEach(() => {
		adaptor.dispose()
	})

	describe('getInstance', () => {
		it('should return singleton instance', () => {
			const instance1 = BatchAdaptor.getInstance()
			const instance2 = BatchAdaptor.getInstance()
			expect(instance1).toBe(instance2)
		})

		it('should export default instance', () => {
			expect(batchAdaptor).toBeInstanceOf(BatchAdaptor)
		})
	})

	describe('getBatchSize', () => {
		it('should return default batch size 500', () => {
			expect(adaptor.getBatchSize()).toBe(500)
		})
	})

	describe('getMetrics', () => {
		it('should return initial metrics', () => {
			const metrics = adaptor.getMetrics()
			expect(metrics.latency).toBe(0)
			expect(metrics.lastPingTime).toBe(0)
			expect(metrics.measurements).toEqual([])
			expect(metrics.smoothedLatency).toBe(0)
		})
	})

	describe('getConfig', () => {
		it('should return default config', () => {
			const config = adaptor.getConfig()
			expect(config.batchSize).toBe(500)
			expect(config.minBatchSize).toBe(100)
			expect(config.maxBatchSize).toBe(500)
			expect(config.adjustmentCooldown).toBe(30000)
		})
	})

	describe('recordMeasurement', () => {
		it('should record latency measurement', () => {
			adaptor.recordMeasurement(100)
			const metrics = adaptor.getMetrics()
			expect(metrics.latency).toBe(100)
			expect(metrics.measurements).toHaveLength(1)
			expect(metrics.measurements[0]).toBe(100)
		})

		it('should keep only last 10 measurements', () => {
			for (let i = 0; i < 15; i++) {
				adaptor.recordMeasurement(i * 10)
			}
			const metrics = adaptor.getMetrics()
			expect(metrics.measurements).toHaveLength(10)
		})

		it('should update smoothed latency', () => {
			adaptor.recordMeasurement(100)
			adaptor.recordMeasurement(200)
			const metrics = adaptor.getMetrics()
			expect(metrics.smoothedLatency).toBeGreaterThan(0)
		})
	})

	describe('batch size adjustment', () => {
		it('should set batch size to 500 for latency <= 200ms', () => {
			adaptor.recordMeasurement(150)
			expect(adaptor.getBatchSize()).toBe(500)
		})

		it('should set batch size to 300 for latency 201-500ms', () => {
			adaptor.recordMeasurement(300)
			expect(adaptor.getBatchSize()).toBe(300)
		})

		it('should set batch size to 100 for latency > 500ms', () => {
			adaptor.recordMeasurement(600)
			expect(adaptor.getBatchSize()).toBe(100)
		})

		it('should not adjust during cooldown', () => {
			adaptor.recordMeasurement(600)
			expect(adaptor.getBatchSize()).toBe(100)

			adaptor.recordMeasurement(150)
			expect(adaptor.getBatchSize()).toBe(100)
		})
	})

	describe('measureLatency', () => {
		it('should call ping callback and record latency', async () => {
			const pingCallback = vi.fn().mockResolvedValue(undefined)
			adaptor.initialize(pingCallback)

			const latency = await adaptor.measureLatency()
			expect(latency).toBeGreaterThanOrEqual(0)
			expect(pingCallback).toHaveBeenCalled()
		})

		it('should handle ping failure', async () => {
			const pingCallback = vi.fn().mockRejectedValue(new Error('Ping failed'))
			adaptor.initialize(pingCallback)

			const latency = await adaptor.measureLatency()
			expect(latency).toBe(0)
		})
	})

	describe('setBatchSizeRange', () => {
		it('should update batch size range', () => {
			adaptor.setBatchSizeRange(50, 200)
			const config = adaptor.getConfig()
			expect(config.minBatchSize).toBe(50)
			expect(config.maxBatchSize).toBe(200)
		})

		it('should clamp current batch size to new range', () => {
			adaptor.recordMeasurement(150)
			expect(adaptor.getBatchSize()).toBe(500)

			adaptor.setBatchSizeRange(50, 200)
			expect(adaptor.getBatchSize()).toBe(200)
		})
	})

	describe('setAdjustmentCooldown', () => {
		it('should update adjustment cooldown', () => {
			adaptor.setAdjustmentCooldown(5000)
			const config = adaptor.getConfig()
			expect(config.adjustmentCooldown).toBe(5000)
		})
	})

	describe('reset', () => {
		it('should reset to default configuration', () => {
			adaptor.recordMeasurement(600)
			adaptor.reset()

			const config = adaptor.getConfig()
			const metrics = adaptor.getMetrics()

			expect(config.batchSize).toBe(500)
			expect(metrics.latency).toBe(0)
			expect(metrics.measurements).toEqual([])
		})
	})

	describe('getRecommendedBatchSize', () => {
		it('should recommend 500 for latency <= 200ms', () => {
			expect(adaptor.getRecommendedBatchSize(150)).toBe(500)
			expect(adaptor.getRecommendedBatchSize(200)).toBe(500)
		})

		it('should recommend 300 for latency 201-500ms', () => {
			expect(adaptor.getRecommendedBatchSize(300)).toBe(300)
			expect(adaptor.getRecommendedBatchSize(500)).toBe(300)
		})

		it('should recommend 100 for latency > 500ms', () => {
			expect(adaptor.getRecommendedBatchSize(600)).toBe(100)
			expect(adaptor.getRecommendedBatchSize(1000)).toBe(100)
		})

		it('should respect min and max batch size', () => {
			adaptor.setBatchSizeRange(150, 400)
			expect(adaptor.getRecommendedBatchSize(600)).toBe(150)
			expect(adaptor.getRecommendedBatchSize(150)).toBe(400)
		})
	})

	describe('getSummary', () => {
		it('should return summary string', () => {
			adaptor.recordMeasurement(300)
			const summary = adaptor.getSummary()
			expect(summary).toContain('BatchAdaptor')
			expect(summary).toContain('batchSize=300')
			expect(summary).toContain('latency=')
		})
	})

	describe('initialize', () => {
		it('should initialize with ping callback', () => {
			const pingCallback = vi.fn().mockResolvedValue(undefined)
			adaptor.initialize(pingCallback)
		})
	})

	describe('dispose', () => {
		it('should dispose and clear interval', () => {
			const pingCallback = vi.fn().mockResolvedValue(undefined)
			adaptor.initialize(pingCallback)
			adaptor.dispose()
		})
	})
})
