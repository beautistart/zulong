import { create } from "zustand"
import type { TestExecutionLive, TestCaseListItem, ProgressReportView } from "../types/test"
import { TestStatus, TestCaseType } from "../types/test"

interface TestState {
  testCases: TestCaseListItem[]
  selectedCaseId: string | null
  currentExecution: TestExecutionLive | null
  activeExecutions: TestExecutionLive[]
  typeFilter: TestCaseType | null
  statusFilter: TestStatus | null
  searchQuery: string

  setTestCases: (cases: TestCaseListItem[]) => void
  selectCase: (id: string | null) => void
  setCurrentExecution: (exec: TestExecutionLive | null) => void
  setActiveExecutions: (execs: TestExecutionLive[]) => void
  updateStepResult: (stepId: string, updates: Partial<import("../types/test").StepResultView>) => void
  addProgressReport: (report: ProgressReportView) => void
  setFilter: (type: TestCaseType | null, status: TestStatus | null) => void
  setSearchQuery: (query: string) => void
  getFilteredCases: () => TestCaseListItem[]
}

export const useTestStore = create<TestState>((set, get) => ({
  testCases: [],
  selectedCaseId: null,
  currentExecution: null,
  activeExecutions: [],
  typeFilter: null,
  statusFilter: null,
  searchQuery: "",

  setTestCases: (cases) => set({ testCases: cases }),
  selectCase: (id) => set({ selectedCaseId: id }),
  setCurrentExecution: (exec) => set({ currentExecution: exec }),
  setActiveExecutions: (execs) => set({ activeExecutions: execs }),

  updateStepResult: (stepId, updates) =>
    set((state) => {
      if (!state.currentExecution) return state
      const steps = state.currentExecution.steps.map((s) =>
        s.step_id === stepId ? { ...s, ...updates } : s,
      )
      return { currentExecution: { ...state.currentExecution, steps } }
    }),

  addProgressReport: (report) =>
    set((state) => {
      if (!state.currentExecution) return state
      const reports = [...state.currentExecution.progress_reports, report]
      return { currentExecution: { ...state.currentExecution, progress_reports: reports } }
    }),

  setFilter: (type, status) => set({ typeFilter: type, statusFilter: status }),
  setSearchQuery: (query) => set({ searchQuery: query }),

  getFilteredCases: () => {
    const { testCases, typeFilter, statusFilter, searchQuery } = get()
    return testCases.filter((c) => {
      if (typeFilter && c.type !== typeFilter) return false
      if (statusFilter && c.last_status !== statusFilter) return false
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        if (!c.name.toLowerCase().includes(q) && !c.test_case_id.toLowerCase().includes(q)) return false
      }
      return true
    })
  },
}))
