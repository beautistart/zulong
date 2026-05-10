import type { RenderMode } from "../types"

export function selectRenderMode(nodeCount: number): RenderMode {
	if (nodeCount <= 100) return "full"
	if (nodeCount <= 300) return "full_optimized"
	if (nodeCount <= 1500) return "virtual"
	return "degraded"  // 🔥 1500+节点直接进入降级模式
}

export function isSkeletonMode(renderMode: RenderMode, zoom: number): boolean {
	if (renderMode === "degraded") return true
	if (renderMode === "virtual") return true
	return zoom <= 0.2
}

export function isPointLineMode(zoom: number): boolean {
	return zoom <= 0.1
}
