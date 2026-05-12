import { useState } from "react"
import { useTestStore } from "../stores/testStore"


const stepStatusConfig: Record<string, { color: string; label: string }> = {
  completed: { color: "bg-emerald-500", label: "PASS" },
  failed: { color: "bg-red-500", label: "FAIL" },
  running: { color: "bg-blue-500", label: "运行中" },
  pending: { color: "bg-slate-500", label: "等待" },
  skipped: { color: "bg-slate-600", label: "跳过" },
  interrupted: { color: "bg-amber-500", label: "中断" },
}

export function TestDetailPanel() {
  const execution = useTestStore((s) => s.currentExecution)
  const [expandedStep, setExpandedStep] = useState<string | null>(null)

  if (!execution) {
    return (
      <div className="p-6 text-center text-slate-500 text-sm">
        请从左侧选择测试用例查看详情
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4">
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <h2 className="text-base font-medium text-slate-200">{execution.name}</h2>
        <div className="flex items-center gap-3 mt-2 text-xs text-slate-400">
          <span>ID: {execution.execution_id}</span>
          <span>类型: {execution.type}</span>
          <span>状态: {execution.status}</span>
        </div>
      </div>

      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-4">
        <h3 className="text-sm font-medium text-slate-200 mb-3">步骤结果</h3>
        <div className="space-y-1">
          {execution.steps.map((step) => {
            const config = stepStatusConfig[step.status] || stepStatusConfig.pending
            const isExpanded = expandedStep === step.step_id
            return (
              <div key={step.step_id}>
                <div
                  onClick={() => setExpandedStep(isExpanded ? null : step.step_id)}
                  className="flex items-center gap-2 px-3 py-2 bg-slate-900/50 rounded cursor-pointer hover:bg-slate-700/50 transition-colors"
                >
                  <span className={`px-1.5 py-0.5 text-[10px] rounded text-white ${config.color}`}>
                    {config.label}
                  </span>
                  <span className="text-xs text-slate-300">{step.step_id}</span>
                  <span className="text-xs text-slate-400">{step.name}</span>
                  {step.duration_ms != null && (
                    <span className="text-[10px] text-slate-500 ml-auto">{step.duration_ms}ms</span>
                  )}
                </div>
                {isExpanded && (
                  <div className="ml-6 mt-1 space-y-2 px-3 py-2 border-l-2 border-slate-700">
                    {step.output != null && (
                      <div>
                        <span className="text-[10px] text-slate-500">输出:</span>
                        <pre className="text-xs text-slate-300 mt-1 bg-slate-900 p-2 rounded overflow-x-auto max-h-32">
                          {typeof step.output === "string" ? step.output.slice(0, 500) : JSON.stringify(step.output, null, 2)?.slice(0, 500)}
                        </pre>
                      </div>
                    )}
                    {step.error && (
                      <div className="text-xs text-red-400">错误: {step.error}</div>
                    )}
                    {step.assertion_results.length > 0 && (
                      <div>
                        <span className="text-[10px] text-slate-500">断言:</span>
                        {step.assertion_results.map((ar, i) => (
                          <div key={i} className="flex items-center gap-2 mt-1 text-xs">
                            <span className={ar.passed ? "text-emerald-400" : "text-red-400"}>
                              {ar.passed ? "✓" : "✗"}
                            </span>
                            <span className="text-slate-400">{ar.assertion_type}</span>
                            <span className="text-slate-500">{ar.field_path}</span>
                            {!ar.passed && (
                              <span className="text-red-400">
                                期望: {JSON.stringify(ar.expected)} | 实际: {JSON.stringify(ar.actual)}
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {execution.interrupt_point && (
        <div className="bg-amber-900/20 rounded-lg border border-amber-700/50 p-4">
          <h3 className="text-sm font-medium text-amber-300 mb-2">中断点</h3>
          <div className="text-xs text-slate-400 space-y-1">
            <p>步骤: {execution.interrupt_point.step_id}</p>
            <p>原因: {execution.interrupt_point.reason}</p>
            <p>时间: {execution.interrupt_point.timestamp}</p>
            {execution.interrupt_point.snapshot_id && (
              <p>快照: {execution.interrupt_point.snapshot_id}</p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
