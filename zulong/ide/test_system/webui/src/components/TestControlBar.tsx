import { useTestStore } from "../stores/testStore"
import { TestStatus } from "../types/test"

export function TestControlBar() {
  const execution = useTestStore((s) => s.currentExecution)
  const selectedId = useTestStore((s) => s.selectedCaseId)

  const isRunning = execution?.status === TestStatus.RUNNING
  const isInterrupted = execution?.status === TestStatus.INTERRUPTED
  const executionId = execution?.execution_id

  const handleStart = async () => {
    if (!selectedId) return
    try {
      const resp = await fetch(`/api/test/execute?test_case_id=${selectedId}`, { method: "POST" })
      const data = await resp.json()
      useTestStore.getState().setCurrentExecution(data)
    } catch (e) {
      console.error("启动测试失败:", e)
    }
  }

  const handleStop = async () => {
    if (!executionId) return
    try {
      await fetch(`/api/test/execute/${executionId}/stop`, { method: "POST" })
    } catch (e) {
      console.error("停止测试失败:", e)
    }
  }

  const handleResume = async () => {
    if (!executionId) return
    try {
      const resp = await fetch(`/api/test/execute/${executionId}/resume`, { method: "POST" })
      const data = await resp.json()
      useTestStore.getState().setCurrentExecution(data)
    } catch (e) {
      console.error("恢复测试失败:", e)
    }
  }

  return (
    <div className="flex items-center gap-2 px-4 py-2 bg-slate-800 border-b border-slate-700">
      <button
        onClick={handleStart}
        disabled={!selectedId || isRunning}
        className="px-3 py-1.5 text-xs font-medium rounded bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        启动测试
      </button>
      <button
        onClick={handleStop}
        disabled={!isRunning}
        className="px-3 py-1.5 text-xs font-medium rounded bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        停止测试
      </button>
      <button
        onClick={handleResume}
        disabled={!isInterrupted}
        className="px-3 py-1.5 text-xs font-medium rounded bg-emerald-600 text-white hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        恢复执行
      </button>
      {execution && (
        <span className="ml-4 text-xs text-slate-400">
          执行ID: {executionId}
        </span>
      )}
    </div>
  )
}
