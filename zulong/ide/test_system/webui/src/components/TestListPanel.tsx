import { useTestStore } from "../stores/testStore"
import { TestCaseType } from "../types/test"

const typeLabels: Record<string, string> = {
  complex_task: "复杂任务",
  interrupt_resume: "中断恢复",
  long_task_report: "长任务汇报",
  attention_switch: "注意力切换",
}

const statusColors: Record<string, string> = {
  completed: "bg-emerald-600",
  running: "bg-blue-600",
  failed: "bg-red-600",
  interrupted: "bg-amber-600",
  timeout: "bg-orange-600",
  pending: "bg-slate-600",
  initializing: "bg-indigo-600",
}

export function TestListPanel() {
  const testCases = useTestStore((s) => s.getFilteredCases())
  const selectedId = useTestStore((s) => s.selectedCaseId)
  const selectCase = useTestStore((s) => s.selectCase)
  const typeFilter = useTestStore((s) => s.typeFilter)
  const statusFilter = useTestStore((s) => s.statusFilter)
  const searchQuery = useTestStore((s) => s.searchQuery)
  const setFilter = useTestStore((s) => s.setFilter)
  const setSearchQuery = useTestStore((s) => s.setSearchQuery)

  return (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-700">
      <div className="p-3 border-b border-slate-700">
        <input
          type="text"
          placeholder="搜索用例..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full px-3 py-1.5 bg-slate-800 border border-slate-600 rounded text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
        />
      </div>
      <div className="flex gap-1 px-3 py-2 border-b border-slate-700 flex-wrap">
        <button
          onClick={() => setFilter(null, null)}
          className={`px-2 py-0.5 text-xs rounded ${!typeFilter && !statusFilter ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-400"}`}
        >
          全部
        </button>
        {Object.entries(typeLabels).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setFilter(key as TestCaseType, null)}
            className={`px-2 py-0.5 text-xs rounded ${typeFilter === key ? "bg-indigo-600 text-white" : "bg-slate-700 text-slate-400"}`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex-1 overflow-y-auto">
        {testCases.map((tc) => (
          <div
            key={tc.test_case_id}
            onClick={() => selectCase(tc.test_case_id)}
            className={`px-3 py-2.5 cursor-pointer border-b border-slate-800 hover:bg-slate-800 transition-colors ${selectedId === tc.test_case_id ? "bg-slate-800 border-l-2 border-l-indigo-500" : ""}`}
          >
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-200 truncate">{tc.name}</span>
              {tc.last_status && (
                <span className={`px-1.5 py-0.5 text-[10px] rounded text-white ${statusColors[tc.last_status] || "bg-slate-600"}`}>
                  {tc.last_status}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="px-1.5 py-0.5 text-[10px] bg-slate-700 text-slate-400 rounded">
                {typeLabels[tc.type] || tc.type}
              </span>
              {tc.last_execution_time && (
                <span className="text-[10px] text-slate-500">{tc.last_execution_time}</span>
              )}
            </div>
          </div>
        ))}
        {testCases.length === 0 && (
          <div className="px-3 py-8 text-center text-slate-500 text-sm">暂无测试用例</div>
        )}
      </div>
    </div>
  )
}
