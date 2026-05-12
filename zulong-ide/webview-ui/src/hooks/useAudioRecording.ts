import { useCallback, useEffect, useRef, useState } from "react"

export interface AudioChunk {
	data: ArrayBuffer
	timestamp: number
}

export interface UseAudioRecordingOptions {
	onChunk?: (chunk: AudioChunk) => void
	onTranscript?: (text: string) => void
	onError?: (error: Error) => void
	chunkInterval?: number
}

export interface UseAudioRecordingReturn {
	isRecording: boolean
	isSupported: boolean
	startRecording: () => Promise<void>
	stopRecording: () => void
	toggleRecording: () => Promise<void>
	error: Error | null
}

export function useAudioRecording(options: UseAudioRecordingOptions = {}): UseAudioRecordingReturn {
	const { onChunk, onError, chunkInterval = 100 } = options

	const [isRecording, setIsRecording] = useState(false)
	const [error, setError] = useState<Error | null>(null)

	const mediaRecorderRef = useRef<MediaRecorder | null>(null)
	const audioContextRef = useRef<AudioContext | null>(null)
	const streamRef = useRef<MediaStream | null>(null)
	const chunkTimerRef = useRef<NodeJS.Timeout | null>(null)
	const audioBufferRef = useRef<ArrayBuffer[]>([])

	const isSupported = typeof navigator !== "undefined" && 
		typeof navigator.mediaDevices !== "undefined" && 
		typeof MediaRecorder !== "undefined"

	const collectChunk = useCallback(() => {
		if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== "recording") {
			return
		}

		if (audioBufferRef.current.length > 0) {
			const chunks = audioBufferRef.current
			audioBufferRef.current = []

			const totalLength = chunks.reduce((acc, chunk) => acc + chunk.byteLength, 0)
			const combined = new Uint8Array(totalLength)
			let offset = 0
			for (const chunk of chunks) {
				combined.set(new Uint8Array(chunk), offset)
				offset += chunk.byteLength
			}

			onChunk?.({
				data: combined.buffer,
				timestamp: Date.now(),
			})
		}
	}, [onChunk])

	const startRecording = useCallback(async () => {
		if (!isSupported) {
			const err = new Error("当前浏览器不支持录音功能")
			setError(err)
			onError?.(err)
			return
		}

		if (isRecording) {
			return
		}

		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				audio: {
					echoCancellation: true,
					noiseSuppression: true,
					autoGainControl: true,
					sampleRate: 16000,
					channelCount: 1,
				},
			})
			streamRef.current = stream

			const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
				? "audio/webm;codecs=opus"
				: MediaRecorder.isTypeSupported("audio/webm")
					? "audio/webm"
					: "audio/mp4"

			const mediaRecorder = new MediaRecorder(stream, {
				mimeType,
				audioBitsPerSecond: 16000,
			})
			mediaRecorderRef.current = mediaRecorder

			audioBufferRef.current = []

			mediaRecorder.ondataavailable = (event) => {
				if (event.data && event.data.size > 0) {
					event.data.arrayBuffer().then((buffer) => {
						audioBufferRef.current.push(buffer)
					})
				}
			}

			mediaRecorder.onerror = (event) => {
				const err = new Error(`录音错误: ${event}`)
				setError(err)
				onError?.(err)
				stopRecording()
			}

			mediaRecorder.start(50)
			setIsRecording(true)
			setError(null)

			chunkTimerRef.current = setInterval(collectChunk, chunkInterval)

			console.log("[VoiceInput] 录音已启动")
		} catch (err) {
			const error = err instanceof Error ? err : new Error(String(err))
			setError(error)
			onError?.(error)
			console.error("[VoiceInput] 启动录音失败:", error)
		}
	}, [isSupported, isRecording, onError, collectChunk, chunkInterval])

	const stopRecording = useCallback(() => {
		if (chunkTimerRef.current) {
			clearInterval(chunkTimerRef.current)
			chunkTimerRef.current = null
		}

		collectChunk()

		if (mediaRecorderRef.current) {
			if (mediaRecorderRef.current.state !== "inactive") {
				mediaRecorderRef.current.stop()
			}
			mediaRecorderRef.current = null
		}

		if (streamRef.current) {
			streamRef.current.getTracks().forEach((track) => track.stop())
			streamRef.current = null
		}

		if (audioContextRef.current) {
			audioContextRef.current.close()
			audioContextRef.current = null
		}

		audioBufferRef.current = []
		setIsRecording(false)
		console.log("[VoiceInput] 录音已停止")
	}, [collectChunk])

	const toggleRecording = useCallback(async () => {
		if (isRecording) {
			stopRecording()
		} else {
			await startRecording()
		}
	}, [isRecording, startRecording, stopRecording])

	useEffect(() => {
		return () => {
			stopRecording()
		}
	}, [stopRecording])

	return {
		isRecording,
		isSupported,
		startRecording,
		stopRecording,
		toggleRecording,
		error,
	}
}
