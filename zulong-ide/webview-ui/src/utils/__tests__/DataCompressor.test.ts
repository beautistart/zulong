/**
 * DataCompressor 单元测试
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { DataCompressor, dataCompressor, CompressedMessage } from '../DataCompressor'

describe('DataCompressor', () => {
	let compressor: DataCompressor

	beforeEach(() => {
		compressor = DataCompressor.getInstance()
	})

	describe('getInstance', () => {
		it('should return singleton instance', () => {
			const instance1 = DataCompressor.getInstance()
			const instance2 = DataCompressor.getInstance()
			expect(instance1).toBe(instance2)
		})

		it('should export default instance', () => {
			expect(dataCompressor).toBeInstanceOf(DataCompressor)
		})
	})

	describe('isCompressionSupported', () => {
		it('should return boolean', () => {
			const result = compressor.isCompressionSupported()
			expect(typeof result).toBe('boolean')
		})
	})

	describe('getCompressionThreshold', () => {
		it('should return 200KB', () => {
			const threshold = compressor.getCompressionThreshold()
			expect(threshold).toBe(200 * 1024)
		})
	})

	describe('shouldCompress', () => {
		it('should return false for small data', () => {
			const smallData = { test: 'small' }
			const result = compressor.shouldCompress(smallData)
			expect(result).toBe(false)
		})

		it('should return true for large data', () => {
			const largeData = {
				nodes: Array(10000).fill(null).map((_, i) => ({
					id: `node-${i}`,
					data: 'x'.repeat(100),
				})),
			}
			const result = compressor.shouldCompress(largeData)
			if (compressor.isCompressionSupported()) {
				expect(result).toBe(true)
			} else {
				expect(result).toBe(false)
			}
		})

		it('should return false when compression not supported', () => {
			if (!compressor.isCompressionSupported()) {
				const largeData = { data: 'x'.repeat(300000) }
				const result = compressor.shouldCompress(largeData)
				expect(result).toBe(false)
			}
		})
	})

	describe('compress', () => {
		it('should not compress small data', async () => {
			const data = { test: 'value' }
			const result = await compressor.compress(data)

			expect(result.success).toBe(true)
			expect(result.message.compressed).toBe(false)
			expect(typeof result.message.data).toBe('string')
			expect(result.compressionRatio).toBe(1)
		})

		it('should compress large data if supported', async () => {
			const largeData = {
				nodes: Array(5000).fill(null).map((_, i) => ({
					id: `node-${i}`,
					label: `Node ${i}`,
					data: 'x'.repeat(50),
				})),
			}

			const result = await compressor.compress(largeData)

			expect(result.success).toBe(true)
			expect(result.message.originalSize).toBeGreaterThan(0)

			if (compressor.isCompressionSupported()) {
				expect(result.message.compressed).toBe(true)
				expect(typeof result.message.data).toBe('string')
				expect(result.message.checksum).toBeDefined()
				expect(result.message.compressedSize).toBeLessThan(result.message.originalSize!)
				expect(result.compressionRatio).toBeLessThan(1)
			} else {
				expect(result.message.compressed).toBe(false)
			}
		})

		it('should include size information', async () => {
			const data = { test: 'value' }
			const result = await compressor.compress(data)

			expect(result.message.originalSize).toBeDefined()
			expect(result.message.originalSize).toBeGreaterThan(0)

			if (result.message.compressed) {
				expect(result.message.compressedSize).toBeDefined()
			}
		})
	})

	describe('decompress', () => {
		it('should decompress uncompressed data', async () => {
			const originalData = { test: 'value', number: 123 }
			const compressed = await compressor.compress(originalData)

			const decompressed = await compressor.decompress(compressed.message)
			expect(decompressed).toEqual(originalData)
		})

		it('should decompress compressed data', async () => {
			const originalData = {
				nodes: Array(5000).fill(null).map((_, i) => ({
					id: `node-${i}`,
					label: `Node ${i}`,
				})),
			}

			const compressed = await compressor.compress(originalData)
			const decompressed = await compressor.decompress(compressed.message)

			expect(decompressed).toEqual(originalData)
		})

		it('should fail on invalid uncompressed data', async () => {
			const message: CompressedMessage = {
				compressed: false,
				data: 'invalid json{',
			}

			await expect(compressor.decompress(message)).rejects.toThrow()
		})

		it('should fail on checksum mismatch', async () => {
			if (!compressor.isCompressionSupported()) {
				return
			}

			const originalData = { test: 'value' }
			const largeData = {
				...originalData,
				padding: 'x'.repeat(300000),
			}

			const compressed = await compressor.compress(largeData)

			if (compressed.message.compressed) {
				const tamperedMessage: CompressedMessage = {
					...compressed.message,
					checksum: 'invalidchecksum',
				}

				await expect(compressor.decompress(tamperedMessage)).rejects.toThrow('Data integrity check failed')
			}
		})
	})

	describe('round-trip compression', () => {
		it('should preserve data through compress/decompress cycle', async () => {
			const testCases = [
				{ simple: 'value' },
				{ array: [1, 2, 3, 4, 5] },
				{ nested: { a: { b: { c: 'deep' } } } },
				{ unicode: '中文测试 🎉 emoji' },
				{ special: '\n\t\r\\/"' },
			]

			for (const data of testCases) {
				const compressed = await compressor.compress(data)
				const decompressed = await compressor.decompress(compressed.message)
				expect(decompressed).toEqual(data)
			}
		})
	})

	describe('uint8ArrayToBase64 and base64ToUint8Array', () => {
		it('should convert Uint8Array to base64 and back', () => {
			const data = new Uint8Array([72, 101, 108, 108, 111]) // "Hello"
			const base64 = compressor.uint8ArrayToBase64(data)
			const result = compressor.base64ToUint8Array(base64)

			expect(result).toEqual(data)
		})

		it('should handle empty array', () => {
			const data = new Uint8Array([])
			const base64 = compressor.uint8ArrayToBase64(data)
			const result = compressor.base64ToUint8Array(base64)

			expect(result).toEqual(data)
		})

		it('should handle all byte values', () => {
			const data = new Uint8Array(256).map((_, i) => i)
			const base64 = compressor.uint8ArrayToBase64(data)
			const result = compressor.base64ToUint8Array(base64)

			expect(result).toEqual(data)
		})
	})

	describe('performance', () => {
		it('should compress 500KB data within 100ms', async () => {
			if (!compressor.isCompressionSupported()) {
				return
			}

			const data = {
				nodes: Array(10000).fill(null).map((_, i) => ({
					id: `node-${i}`,
					data: 'x'.repeat(50),
				})),
			}

			const start = performance.now()
			await compressor.compress(data)
			const duration = performance.now() - start

			expect(duration).toBeLessThan(100)
		})

		it('should decompress within 100ms', async () => {
			if (!compressor.isCompressionSupported()) {
				return
			}

			const data = {
				nodes: Array(10000).fill(null).map((_, i) => ({
					id: `node-${i}`,
					data: 'x'.repeat(50),
				})),
			}

			const compressed = await compressor.compress(data)

			const start = performance.now()
			await compressor.decompress(compressed.message)
			const duration = performance.now() - start

			expect(duration).toBeLessThan(100)
		})
	})
})
