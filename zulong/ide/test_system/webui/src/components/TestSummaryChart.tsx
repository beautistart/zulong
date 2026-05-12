import { useTestStore } from "../stores/testStore"

export function TestSummaryChart() {
  const testCases = useTestStore((s) => s.testCases)

  const counts = testCases.reduce(
    (acc, tc) => {
      const st = tc.last_status || "pending"
      acc[st] = (acc[st] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  const total = testCases.length

  const segments = [
    { key: "completed", label: "通过", color: "#10b981", count: counts.completed || 0 },
    { key: "failed", label: "失败", color: "#ef4444", count: counts.failed || 0 },
    { key: "running", label: "运行中", color: "#3b82f6", count: counts.running || 0 },
    { key: "interrupted", label: "中断", color: "#f59e0b", count: counts.interrupted || 0 },
    { key: "other", label: "其他", color: "#64748b", count: total - (counts.completed || 0) - (counts.failed || 0) - (counts.running || 0) - (counts.interrupted || 0) },
  ]

  return (
    <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
      <h3 className="text-sm font-medium text-slate-200 mb-3">测试统计</h3>
      <div className="flex gap-2">
        {segments.map((s) => (
          <div key={s.key} className="flex-1 text-center">
            <div className="text-lg font-bold" style={{ color: s.color }}>
              {s.count}
            </div>
            <div className="text-[10px] text-slate-500">{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
