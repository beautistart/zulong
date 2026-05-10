import type { RenderMode, RenderMetrics } from "../types"

interface DegradationCheckResult {
	mode: RenderMode | null
	reason: string
}

export function checkPerformance(metrics: RenderMetrics): DegradationCheckResult {
	if (metrics.memoryMB > 512) {
		return { mode: "degraded", reason: `内存超限: ${metrics.memoryMB.toFixed(0)}MB > 512MB` }
	}
	if (metrics.fps > 0 && metrics.fps < 15) {
		return { mode: "degraded", reason: `帧率过低: ${metrics.fps.toFixed(0)} FPS < 15 FPS` }
	}
	return { mode: null, reason: "" }
}

export class PerformanceMonitor {
	private frameCount = 0
	private lastFrameTime = 0
	private fps = 60
	private isRunning = false

	start(): void {
		if (this.isRunning) return
		this.isRunning = true
		this.lastFrameTime = performance.now()
		this.frameCount = 0
		this.measureFrame()
	}

	private measureFrame = (): void => {
		if (!this.isRunning) return
		this.frameCount++
		const now = performance.now()
		const elapsed = now - this.lastFrameTime
		if (elapsed >= 1000) {
			this.fps = (this.frameCount / elapsed) * 1000
			this.frameCount = 0
			this.lastFrameTime = now
		}
		requestAnimationFrame(this.measureFrame)
	}

	stop(): void {
		this.isRunning = false
	}

	getFps(): number {
		return this.fps
	}

	getMemoryMB(): number {
		const perf = performance as unknown as { memory?: { usedJSHeapSize: number } }
		if (perf.memory) {
			return perf.memory.usedJSHeapSize / (1024 * 1024)
		}
		return 0
	}

	getMetrics(renderMode: RenderMode, nodeCount: number, domNodeCount: number, renderTime: number): RenderMetrics {
		return {
			firstRenderTime: renderTime,
			totalRenderTime: renderTime,
			domNodeCount,
			fps: this.fps,
			memoryMB: this.getMemoryMB(),
			nodeCount,
			renderMode,
		}
	}
}
