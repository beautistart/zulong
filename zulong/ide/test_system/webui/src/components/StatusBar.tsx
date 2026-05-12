import { useConnectionStore } from "../stores/connectionStore"

export function StatusBar() {
  const status = useConnectionStore((s) => s.status)

  const statusConfig = {
    connected: { color: "bg-emerald-500", text: "已连接" },
    disconnected: { color: "bg-red-500", text: "未连接" },
    reconnecting: { color: "bg-amber-500", text: "重连中..." },
  }

  const config = statusConfig[status]

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-slate-800 border-t border-slate-700 text-xs">
      <div className="flex items-center gap-2">
        <span className={`inline-block w-2 h-2 rounded-full ${config.color} ${status === "reconnecting" ? "animate-pulse" : ""}`} />
        <span className="text-slate-300">WebSocket: {config.text}</span>
      </div>
      <span className="text-slate-500">祖龙测试监控系统 v0.1</span>
    </div>
  )
}
