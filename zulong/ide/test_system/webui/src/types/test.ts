export enum TestCaseType {
  COMPLEX_TASK = "complex_task",
  INTERRUPT_RESUME = "interrupt_resume",
  LONG_TASK_REPORT = "long_task_report",
  ATTENTION_SWITCH = "attention_switch",
}

export enum TestStatus {
  PENDING = "pending",
  INITIALIZING = "initializing",
  RUNNING = "running",
  INTERRUPTED = "interrupted",
  COMPLETED = "completed",
  FAILED = "failed",
  TIMEOUT = "timeout",
}

export enum StepStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  SKIPPED = "skipped",
  INTERRUPTED = "interrupted",
}

export enum ProgressMode {
  EXACT = "exact",
  ESTIMATED = "estimated",
}

export enum AttentionRestoreStatus {
  RESTORED = "restored",
  COLD_START = "cold_start",
  FAILED = "failed",
}

export interface AssertionResultView {
  assertion_type: string
  field_path: string
  expected: unknown
  actual: unknown
  passed: boolean
  message: string
}

export interface StepResultView {
  step_id: string
  name: string
  status: StepStatus
  started_at: string | null
  completed_at: string | null
  output: unknown
  assertion_results: AssertionResultView[]
  error: string | null
  duration_ms: number | null
}

export interface InterruptPointView {
  step_id: string
  reason: string
  timestamp: string
  snapshot_id: string | null
}

export interface ProgressReportView {
  execution_id: string
  progress_percent: number
  mode: ProgressMode
  stage_description: string
  completed_steps: number
  total_steps: number
  timestamp: string
  intermediate_result: unknown
}

export interface AttentionSwitchView {
  from_session_id: string
  to_session_id: string
  restore_status: AttentionRestoreStatus
  elapsed_ms: number
  timestamp: string
}

export interface TestCaseListItem {
  test_case_id: string
  name: string
  type: TestCaseType
  description: string
  last_status: TestStatus | null
  last_execution_time: string | null
}

export interface TestExecutionLive {
  execution_id: string
  test_case_id: string
  name: string
  type: TestCaseType
  status: TestStatus
  started_at: string | null
  completed_at: string | null
  steps: StepResultView[]
  interrupt_point: InterruptPointView | null
  progress_reports: ProgressReportView[]
  attention_switches: AttentionSwitchView[]
  error: string | null
}

export interface LogEntry {
  timestamp: string
  level: "DEBUG" | "INFO" | "WARN" | "ERROR"
  source: string
  message: string
  execution_id: string
}

export type TestMonitorEventType =
  | "TEST_WELCOME"
  | "TEST_STARTED"
  | "STEP_STARTED"
  | "STEP_COMPLETED"
  | "STEP_FAILED"
  | "PROGRESS_UPDATE"
  | "STAGNATION_ALERT"
  | "INTERRUPT_TRIGGERED"
  | "RESUME_STARTED"
  | "RESUME_COMPLETED"
  | "ATTENTION_SWITCH"
  | "TEST_COMPLETED"

export interface TestMonitorMessage {
  type: TestMonitorEventType
  payload: Record<string, unknown>
  timestamp: string
}

export interface TestWelcomePayload {
  active_executions: TestExecutionLive[]
  test_cases: TestCaseListItem[]
}
