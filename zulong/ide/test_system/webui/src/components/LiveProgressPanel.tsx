import { useTestStore } from "../stores/testStore"
import { TestStatus, ProgressMode } from "../types/test"

export function LiveProgressPanel() {
  const execution = useTestStore((s) => s.currentExecution)

  if (!execution) {
    return (
      <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
        <div className="text-sm text-slate-400">暂无运行中的测试</div>
      </div>
    )
  }

  const latestReport = execution.progress_reports[execution.progress_reports.length - 1]
  const isRunning = execution.status === TestStatus.RUNNING

  const statusLabels: Record<string, string> = {
    pending: "等待中",
    initializing: "初始化",
    running: "运行中",
    interrupted: "已中断",
    completed: "已完成",
    failed: "失败",
    timeout: "超时",
  }

  const statusColors: Record<string, string> = {
    pending: "text-slate-400",
    initializing: "text-indigo-400",
    running: "text-blue-400",
    interrupted: "text-amber-400",
    completed: "text-emerald-400",
    failed: "text-red-400",
    timeout: "text-orange-400",
  }

  return (
    <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-200">{execution.name}</h3>
        <span className={`text-xs ${statusColors[execution.status]}`}>
          {isRunning && <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-400 mr-1 animate-pulse" />}
          {statusLabels[execution.status] || execution.status}
        </span>
      </div>

      {latestReport && (
        <>
          <div className="space-y-1">
            <div className="flex justify-between text-xs text-slate-400">
              <span>{latestReport.stage_description}</span>
              <span>{latestReport.progress_percent.toFixed(1)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${latestReport.mode === ProgressMode.EXACT ? "bg-indigo-500" : "bg-indigo-400 border-r-2 border-dashed border-indigo-300"}`}
                style={{ width: `${Math.min(latestReport.progress_percent, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-500">
              <span>
                {latestReport.completed_steps}/{latestReport.total_steps} 步骤
              </span>
              <span>{latestReport.mode === ProgressMode.EXACT ? "精确" : "估算"}</span>
            </div>
          </div>
        </>
      )}

      {execution.started_at && (
        <div className="text-[10px] text-slate-500">
          开始时间: {new Date(execution.started_at).toLocaleString("zh-CN")}
          {execution.completed_at && ` | 完成: ${new Date(execution.completed_at).toLocaleString("zh-CN")}`}
        </div>
      )}
    </div>
  )
}
