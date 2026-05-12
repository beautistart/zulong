import { useTestStore } from "../stores/testStore"
import { AttentionRestoreStatus } from "../types/test"

export function TimelinePanel() {
  const execution = useTestStore((s) => s.currentExecution)

  if (!execution) return null

  const events: { time: string; type: string; data: Record<string, unknown> }[] = []

  execution.steps.forEach((s) => {
    if (s.started_at) events.push({ time: s.started_at, type: "step_start", data: { step_id: s.step_id, name: s.name } })
    if (s.completed_at) events.push({ time: s.completed_at, type: s.status === "failed" ? "step_fail" : "step_done", data: { step_id: s.step_id, name: s.name } })
  })

  execution.attention_switches.forEach((a) => {
    events.push({ time: a.timestamp, type: "attention", data: a as unknown as Record<string, unknown> })
  })

  if (execution.interrupt_point) {
    events.push({ time: execution.interrupt_point.timestamp, type: "interrupt", data: { step_id: execution.interrupt_point.step_id, reason: execution.interrupt_point.reason } })
  }

  events.sort((a, b) => a.time.localeCompare(b.time))

  const typeIcons: Record<string, { icon: string; color: string; label: string }> = {
    step_start: { icon: "▶", color: "text-blue-400", label: "步骤开始" },
    step_done: { icon: "✓", color: "text-emerald-400", label: "步骤完成" },
    step_fail: { icon: "✗", color: "text-red-400", label: "步骤失败" },
    interrupt: { icon: "⏸", color: "text-amber-400", label: "中断" },
    attention: { icon: "↔", color: "text-purple-400", label: "注意力切换" },
  }

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
      <h3 className="text-sm font-medium text-slate-200 mb-3">时间线</h3>
      <div className="space-y-2">
        {events.map((e, i) => {
          const cfg = typeIcons[e.type] || { icon: "•", color: "text-slate-400", label: e.type }
          return (
            <div key={i} className="flex items-start gap-3">
              <div className="flex flex-col items-center">
                <span className={`text-sm ${cfg.color}`}>{cfg.icon}</span>
                {i < events.length - 1 && <div className="w-px h-4 bg-slate-700 mt-1" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-300">{cfg.label}</span>
                  <span className="text-[10px] text-slate-500">
                    {new Date(e.time).toLocaleTimeString("zh-CN")}
                  </span>
                </div>
                {e.type === "attention" && (
                  <div className="text-xs text-slate-400 mt-0.5">
                    {String(e.data.from_session_id)} → {String(e.data.to_session_id)}
                    <span className={`ml-2 ${e.data.restore_status === AttentionRestoreStatus.RESTORED ? "text-emerald-400" : e.data.restore_status === AttentionRestoreStatus.COLD_START ? "text-amber-400" : "text-red-400"}`}>
                      {String(e.data.restore_status)}
                    </span>
                  </div>
                )}
                {e.type === "interrupt" && (
                  <div className="text-xs text-slate-400 mt-0.5">
                    步骤 {String(e.data.step_id)}: {String(e.data.reason)}
                  </div>
                )}
                {e.type.startsWith("step") && (
                  <div className="text-xs text-slate-400 mt-0.5">
                    {String(e.data.step_id)} - {String(e.data.name)}
                  </div>
                )}
              </div>
            </div>
          )
        })}
        {events.length === 0 && (
          <div className="text-xs text-slate-500 text-center py-4">暂无事件</div>
        )}
      </div>
    </div>
  )
}
