import { useEffect, useRef } from "react"
import { TestMonitorConnection } from "./services/testMonitorConnection"
import { useConnectionStore } from "./stores/connectionStore"
import { useTestStore } from "./stores/testStore"
import { useLogStore } from "./stores/logStore"
import { StatusBar } from "./components/StatusBar"
import { TestListPanel } from "./components/TestListPanel"
import { TestControlBar } from "./components/TestControlBar"
import { LiveProgressPanel } from "./components/LiveProgressPanel"
import { LogStreamPanel } from "./components/LogStreamPanel"
import { TestDetailPanel } from "./components/TestDetailPanel"
import { TimelinePanel } from "./components/TimelinePanel"
import { TestSummaryChart } from "./components/TestSummaryChart"
import type { TestMonitorMessage } from "./types/test"
import { TestStatus, StepStatus, ProgressMode } from "./types/test"

function App() {
  const connRef = useRef<TestMonitorConnection | null>(null)
  const setStatus = useConnectionStore((s) => s.setStatus)

  useEffect(() => {
    const conn = new TestMonitorConnection("ws://127.0.0.1:8092/ws", (status) => {
      setStatus(status as "connected" | "disconnected" | "reconnecting")
    })

    conn.onAny((msg: TestMonitorMessage) => {
      const { type, payload } = msg

      switch (type) {
        case "TEST_WELCOME": {
          const p = payload as { test_cases: unknown[]; active_executions: unknown[] }
          useTestStore.getState().setTestCases(p.test_cases as any[])
          useTestStore.getState().setActiveExecutions(p.active_executions as any[])
          break
        }
        case "TEST_STARTED": {
          useTestStore.getState().setCurrentExecution({
            execution_id: (payload as any).execution_id,
            test_case_id: (payload as any).test_case_id,
            name: (payload as any).name || "",
            type: (payload as any).type,
            status: TestStatus.RUNNING,
            started_at: new Date().toISOString(),
            completed_at: null,
            steps: [],
            interrupt_point: null,
            progress_reports: [],
            attention_switches: [],
            error: null,
          })
          break
        }
        case "STEP_STARTED": {
          const p = payload as any
          useTestStore.getState().updateStepResult(p.step_id, {
            step_id: p.step_id,
            name: p.name || "",
            status: StepStatus.RUNNING,
            started_at: new Date().toISOString(),
            completed_at: null,
            output: null,
            assertion_results: [],
            error: null,
            duration_ms: null,
          })
          break
        }
        case "STEP_COMPLETED": {
          const p = payload as any
          useTestStore.getState().updateStepResult(p.step_id, {
            status: StepStatus.COMPLETED,
            completed_at: p.completed_at || new Date().toISOString(),
            output: p.output,
            assertion_results: p.assertion_results || [],
            duration_ms: p.duration_ms,
          })
          break
        }
        case "STEP_FAILED": {
          const p = payload as any
          useTestStore.getState().updateStepResult(p.step_id, {
            status: StepStatus.FAILED,
            completed_at: p.completed_at || new Date().toISOString(),
            error: p.error,
          })
          break
        }
        case "PROGRESS_UPDATE": {
          const p = payload as any
          useTestStore.getState().addProgressReport({
            execution_id: p.execution_id || "",
            progress_percent: p.progress_percent || 0,
            mode: p.mode === "estimated" ? ProgressMode.ESTIMATED : ProgressMode.EXACT,
            stage_description: p.stage_description || "",
            completed_steps: p.completed_steps || 0,
            total_steps: p.total_steps || 0,
            timestamp: p.timestamp || new Date().toISOString(),
            intermediate_result: p.intermediate_result,
          })
          break
        }
        case "TEST_COMPLETED": {
          const p = payload as any
          useTestStore.getState().setCurrentExecution({
            ...useTestStore.getState().currentExecution!,
            status: p.status || TestStatus.COMPLETED,
            completed_at: p.completed_at || new Date().toISOString(),
            error: p.error,
          })
          break
        }
        case "INTERRUPT_TRIGGERED":
        case "RESUME_STARTED":
        case "RESUME_COMPLETED":
        case "ATTENTION_SWITCH":
        case "STAGNATION_ALERT": {
          useLogStore.getState().addLog({
            timestamp: new Date().toISOString(),
            level: type === "STAGNATION_ALERT" ? "WARN" : "INFO",
            source: "TestSystem",
            message: `${type}: ${JSON.stringify(payload).slice(0, 200)}`,
            execution_id: (payload as any).execution_id || "",
          })
          break
        }
      }
    })

    conn.connect()
    connRef.current = conn

    return () => {
      conn.disconnect()
    }
  }, [setStatus])

  return (
    <div className="flex flex-col h-screen">
      <TestControlBar />
      <div className="flex flex-1 overflow-hidden">
        <div className="w-72 flex-shrink-0">
          <TestListPanel />
        </div>
        <div className="flex-1 flex flex-col overflow-hidden p-4 gap-4">
          <div className="flex gap-4">
            <div className="flex-1">
              <LiveProgressPanel />
            </div>
            <div className="w-64">
              <TestSummaryChart />
            </div>
          </div>
          <div className="flex-1 flex gap-4 overflow-hidden">
            <div className="flex-1 overflow-hidden">
              <TestDetailPanel />
            </div>
            <div className="w-80 flex-shrink-0">
              <TimelinePanel />
            </div>
          </div>
          <div className="h-48 flex-shrink-0">
            <LogStreamPanel />
          </div>
        </div>
      </div>
      <StatusBar />
    </div>
  )
}

export default App
