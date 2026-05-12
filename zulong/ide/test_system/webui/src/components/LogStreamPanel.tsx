import { useRef, useEffect } from "react"
import { useLogStore } from "../stores/logStore"

const levelColors: Record<string, string> = {
  DEBUG: "text-slate-500",
  INFO: "text-blue-400",
  WARN: "text-amber-400",
  ERROR: "text-red-400",
}

const levelBg: Record<string, string> = {
  DEBUG: "bg-slate-800/30",
  INFO: "bg-slate-800/50",
  WARN: "bg-amber-900/20",
  ERROR: "bg-red-900/20",
}

export function LogStreamPanel() {
  const logs = useLogStore((s) => s.getVisibleLogs(200))
  const isPaused = useLogStore((s) => s.isPaused)
  const isAutoScroll = useLogStore((s) => s.isAutoScroll)
  const collapsedCount = useLogStore((s) => s.collapsedCount)
  const setPaused = useLogStore((s) => s.setPaused)
  const setAutoScroll = useLogStore((s) => s.setAutoScroll)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isAutoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [logs.length, isAutoScroll])

  return (
    <div className="flex flex-col h-full bg-slate-900 rounded-lg border border-slate-700">
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <span className="text-xs font-medium text-slate-300">日志流</span>
        <div className="flex items-center gap-2">
          {collapsedCount > 0 && (
            <span className="text-[10px] text-slate-500">已折叠 {collapsedCount} 条DEBUG日志</span>
          )}
          <button
            onClick={() => setAutoScroll(!isAutoScroll)}
            className={`text-[10px] px-2 py-0.5 rounded ${isAutoScroll ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-400"}`}
          >
            {isAutoScroll ? "自动滚动" : "手动滚动"}
          </button>
          <button
            onClick={() => setPaused(!isPaused)}
            className={`text-[10px] px-2 py-0.5 rounded ${isPaused ? "bg-amber-600 text-white" : "bg-slate-700 text-slate-400"}`}
          >
            {isPaused ? "已暂停" : "暂停"}
          </button>
        </div>
      </div>
      <div ref={containerRef} className="flex-1 overflow-y-auto p-2 space-y-0.5 font-mono text-[11px]">
        {logs.map((log, i) => (
          <div key={i} className={`px-2 py-0.5 rounded ${levelBg[log.level] || ""}`}>
            <span className="text-slate-600">{new Date(log.timestamp).toLocaleTimeString("zh-CN")}</span>
            <span className={`ml-2 ${levelColors[log.level] || "text-slate-400"}`}>[{log.level}]</span>
            <span className="ml-2 text-slate-500">{log.source}</span>
            <span className="ml-2 text-slate-300">{log.message}</span>
          </div>
        ))}
        {logs.length === 0 && (
          <div className="text-center text-slate-500 py-8">暂无日志</div>
        )}
      </div>
    </div>
  )
}
