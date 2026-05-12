import { VSCodeButton } from "@vscode/webview-ui-toolkit/react"
import { MicIcon, MicOffIcon } from "lucide-react"
import type React from "react"
import { useCallback, useEffect, useRef, useState } from "react"
import styled from "styled-components"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

interface VoiceInputButtonProps {
	onTranscript?: (text: string) => void
	onError?: (error: Error) => void
	disabled?: boolean
}

const ButtonContainer = styled.div`
	display: flex;
	align-items: center;
	justify-content: center;
	width: 16px;
	height: 16px;
`

const RecordingIndicator = styled.div`
	position: absolute;
	top: -2px;
	right: -2px;
	width: 6px;
	height: 6px;
	background-color: #f48771;
	border-radius: 50%;
	animation: pulse 1.5s ease-in-out infinite;

	@keyframes pulse {
		0%, 100% {
			opacity: 1;
			transform: scale(1);
		}
		50% {
			opacity: 0.5;
			transform: scale(1.2);
		}
	}
`

const VoiceInputButton: React.FC<VoiceInputButtonProps> = ({
	onTranscript,
	onError,
	disabled = false,
}) => {
	const [isRecording, setIsRecording] = useState(false)
	const [error, setError] = useState<Error | null>(null)
	const mediaRecorderRef = useRef<MediaRecorder | null>(null)
	const streamRef = useRef<MediaStream | null>(null)

	const isSupported = typeof navigator !== "undefined" && 
		typeof navigator.mediaDevices !== "undefined" && 
		typeof MediaRecorder !== "undefined"

	const startRecording = useCallback(async () => {
		try {
			const stream = await navigator.mediaDevices.getUserMedia({
				audio: {
					echoCancellation: true,
					noiseSuppression: true,
					sampleRate: 16000,
				},
			})
			streamRef.current = stream

			const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
				? "audio/webm;codecs=opus"
				: "audio/webm"

			const mediaRecorder = new MediaRecorder(stream, { mimeType })
			mediaRecorderRef.current = mediaRecorder

			const audioChunks: Blob[] = []
			mediaRecorder.ondataavailable = (event) => {
				if (event.data.size > 0) {
					audioChunks.push(event.data)
				}
			}

			mediaRecorder.onstop = async () => {
				const audioBlob = new Blob(audioChunks, { type: mimeType })
				console.log(`[VoiceInput] 录音完成: ${audioBlob.size} bytes`)
				
				const reader = new FileReader()
				reader.onloadend = () => {
					const base64 = reader.result as string
					console.log(`[VoiceInput] 音频数据: ${base64.substring(0, 50)}...`)
				}
				reader.readAsDataURL(audioBlob)

				if (streamRef.current) {
					streamRef.current.getTracks().forEach(track => track.stop())
					streamRef.current = null
				}
			}

			mediaRecorder.onerror = (event) => {
				const err = new Error(`录音错误: ${event}`)
				setError(err)
				onError?.(err)
			}

			mediaRecorder.start()
			setIsRecording(true)
			setError(null)
			console.log("[VoiceInput] 录音已启动")
		} catch (err) {
			const error = err instanceof Error ? err : new Error(String(err))
			setError(error)
			onError?.(error)
			console.error("[VoiceInput] 启动录音失败:", error)
		}
	}, [onError])

	const stopRecording = useCallback(() => {
		if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
			mediaRecorderRef.current.stop()
		}
		setIsRecording(false)
		console.log("[VoiceInput] 录音已停止")
	}, [])

	const handleToggleRecording = useCallback(async () => {
		if (disabled || !isSupported) {
			return
		}

		if (isRecording) {
			stopRecording()
		} else {
			await startRecording()
		}
	}, [disabled, isSupported, isRecording, startRecording, stopRecording])

	useEffect(() => {
		return () => {
			if (streamRef.current) {
				streamRef.current.getTracks().forEach(track => track.stop())
			}
		}
	}, [])

	if (!isSupported) {
		return null
	}

	return (
		<Tooltip>
			<TooltipContent>{isRecording ? "停止录音" : "语音输入"}</TooltipContent>
			<TooltipTrigger>
				<VSCodeButton
					appearance="icon"
					aria-label={isRecording ? "停止录音" : "语音输入"}
					className={`p-0 m-0 flex items-center relative ${isRecording ? "recording" : ""}`}
					data-testid="voice-input-button"
					disabled={disabled}
					onClick={handleToggleRecording}>
					<ButtonContainer>
						{isRecording ? <MicOffIcon size={14} /> : <MicIcon size={14} />}
					</ButtonContainer>
					{isRecording && <RecordingIndicator />}
				</VSCodeButton>
			</TooltipTrigger>
		</Tooltip>
	)
}

export default VoiceInputButton
