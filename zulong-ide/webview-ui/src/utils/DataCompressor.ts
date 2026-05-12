/**
 * 数据压缩工具类
 * 
 * 用于WebSocket大数据传输时的gzip压缩和解压
 * 压缩阈值：200KB，超过阈值自动启用压缩
 * 
 * 压缩流程：数据序列化 → 检查大小 → gzip压缩 → 传输
 * 解压流程：接收数据 → 判断压缩标志 → gzip解压 → 数据反序列化
 */

export interface CompressedMessage {
	compressed: boolean
	checksum?: string
	data: string | Uint8Array
	originalSize?: number
	compressedSize?: number
}

export interface CompressionResult {
	success: boolean
	message: CompressedMessage
	compressionRatio?: number
}

const COMPRESSION_THRESHOLD = 200 * 1024 // 200KB

export class DataCompressor {
	private static instance: DataCompressor | null = null
	private compressionSupported: boolean = false

	private constructor() {
		this.checkCompressionSupport()
	}

	static getInstance(): DataCompressor {
		if (!DataCompressor.instance) {
			DataCompressor.instance = new DataCompressor()
		}
		return DataCompressor.instance
	}

	private checkCompressionSupport(): void {
		try {
			if (typeof CompressionStream !== 'undefined' && typeof DecompressionStream !== 'undefined') {
				this.compressionSupported = true
			} else {
				this.compressionSupported = false
				console.warn('[DataCompressor] CompressionStream API not available, compression disabled')
			}
		} catch (e) {
			this.compressionSupported = false
			console.warn('[DataCompressor] Compression support check failed:', e)
		}
	}

	isCompressionSupported(): boolean {
		return this.compressionSupported
	}

	async compress(data: any): Promise<CompressionResult> {
		const serialized = JSON.stringify(data)
		const originalSize = new Blob([serialized]).size

		if (originalSize < COMPRESSION_THRESHOLD || !this.compressionSupported) {
			return {
				success: true,
				message: {
					compressed: false,
					data: serialized,
					originalSize,
					compressedSize: originalSize,
				},
				compressionRatio: 1,
			}
		}

		try {
			const compressedData = await this.gzipCompress(serialized)
			const compressedSize = compressedData.byteLength
			const compressionRatio = compressedSize / originalSize
			const base64Data = this.uint8ArrayToBase64(compressedData)

			console.log(`[DataCompressor] Compressed: ${originalSize} → ${compressedSize} bytes (${Math.round((1 - compressionRatio) * 100)}% reduction)`)

			return {
				success: true,
				message: {
					compressed: true,
					checksum: this.calculateChecksum(serialized),
					data: base64Data,
					originalSize,
					compressedSize,
				},
				compressionRatio,
			}
		} catch (e) {
			console.error('[DataCompressor] Compression failed:', e)
			return {
				success: false,
				message: {
					compressed: false,
					data: serialized,
					originalSize,
				},
			}
		}
	}

	async decompress(message: CompressedMessage): Promise<any> {
		if (!message.compressed) {
			try {
				return JSON.parse(message.data as string)
			} catch (e) {
				console.error('[DataCompressor] Parse failed for uncompressed data:', e)
				throw new Error('Failed to parse uncompressed data')
			}
		}

		try {
			const base64Data = message.data as string
			const compressedData = this.base64ToUint8Array(base64Data)
			const decompressed = await this.gzipDecompress(compressedData)
			const parsed = JSON.parse(decompressed)

			if (message.checksum) {
				const calculatedChecksum = this.calculateChecksum(decompressed)
				if (calculatedChecksum !== message.checksum) {
					console.error('[DataCompressor] Checksum mismatch!')
					throw new Error('Data integrity check failed')
				}
			}

			return parsed
		} catch (e) {
			console.error('[DataCompressor] Decompression failed:', e)
			throw new Error('Failed to decompress data')
		}
	}

	private async gzipCompress(data: string): Promise<Uint8Array> {
		if (!this.compressionSupported) {
			throw new Error('Compression not supported')
		}

		const stream = new CompressionStream('gzip')
		const writer = stream.writable.getWriter()
		const reader = stream.readable.getReader()

		writer.write(new TextEncoder().encode(data))
		writer.close()

		const chunks: Uint8Array[] = []
		let totalLength = 0

		while (true) {
			const { done, value } = await reader.read()
			if (done) break
			chunks.push(value)
			totalLength += value.length
		}

		const result = new Uint8Array(totalLength)
		let offset = 0
		for (const chunk of chunks) {
			result.set(chunk, offset)
			offset += chunk.length
		}

		return result
	}

	uint8ArrayToBase64(data: Uint8Array): string {
		let binary = ''
		for (let i = 0; i < data.length; i++) {
			binary += String.fromCharCode(data[i])
		}
		return btoa(binary)
	}

	base64ToUint8Array(base64: string): Uint8Array {
		const binary = atob(base64)
		const result = new Uint8Array(binary.length)
		for (let i = 0; i < binary.length; i++) {
			result[i] = binary.charCodeAt(i)
		}
		return result
	}

	private async gzipDecompress(data: Uint8Array): Promise<string> {
		if (!this.compressionSupported) {
			throw new Error('Decompression not supported')
		}

		const stream = new DecompressionStream('gzip')
		const writer = stream.writable.getWriter()
		const reader = stream.readable.getReader()

		writer.write(new Uint8Array(data))
		writer.close()

		const chunks: Uint8Array[] = []
		let totalLength = 0

		while (true) {
			const { done, value } = await reader.read()
			if (done) break
			chunks.push(value)
			totalLength += value.length
		}

		const result = new Uint8Array(totalLength)
		let offset = 0
		for (const chunk of chunks) {
			result.set(chunk, offset)
			offset += chunk.length
		}

		return new TextDecoder().decode(result)
	}

	private calculateChecksum(data: string): string {
		let hash = 0
		for (let i = 0; i < data.length; i++) {
			const char = data.charCodeAt(i)
			hash = ((hash << 5) - hash) + char
			hash = hash & hash
		}
		return Math.abs(hash).toString(16)
	}

	getCompressionThreshold(): number {
		return COMPRESSION_THRESHOLD
	}

	shouldCompress(data: any): boolean {
		if (!this.compressionSupported) {
			return false
		}
		try {
			const serialized = JSON.stringify(data)
			const size = new Blob([serialized]).size
			return size >= COMPRESSION_THRESHOLD
		} catch {
			return false
		}
	}
}

export const dataCompressor = DataCompressor.getInstance()
