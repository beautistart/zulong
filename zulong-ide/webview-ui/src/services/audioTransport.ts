import { EventEmitter } from "events"

type AudioChunkHandler = (base64: string, format: string) => void
type AudioSignalHandler = () => void

class AudioTransportService extends EventEmitter {
	private static instance: AudioTransportService | null = null
	private _sendAudioChunk: AudioChunkHandler | null = null
	private _sendAudioStart: AudioSignalHandler | null = null
	private _sendAudioEnd: AudioSignalHandler | null = null

	static getInstance(): AudioTransportService {
		if (!AudioTransportService.instance) {
			AudioTransportService.instance = new AudioTransportService()
		}
		return AudioTransportService.instance
	}

	setTransport(sendAudioChunk: AudioChunkHandler, sendAudioStart: AudioSignalHandler, sendAudioEnd: AudioSignalHandler): void {
		this._sendAudioChunk = sendAudioChunk
		this._sendAudioStart = sendAudioStart
		this._sendAudioEnd = sendAudioEnd
	}

	clearTransport(): void {
		this._sendAudioChunk = null
		this._sendAudioStart = null
		this._sendAudioEnd = null
	}

	sendAudioChunk(base64: string, format: string = "webm"): void {
		if (this._sendAudioChunk) {
			this._sendAudioChunk(base64, format)
		} else {
			console.warn("[AudioTransport] 未连接 WebSocket")
		}
	}

	sendAudioStart(): void {
		if (this._sendAudioStart) {
			this._sendAudioStart()
		}
	}

	sendAudioEnd(): void {
		if (this._sendAudioEnd) {
			this._sendAudioEnd()
		}
	}

	get isConnected(): boolean {
		return this._sendAudioChunk !== null
	}
}

export const audioTransport = AudioTransportService.getInstance()
