import { create } from "zustand"
import type { LogEntry } from "../types/test"

interface LogState {
  logs: LogEntry[]
  isPaused: boolean
  isAutoScroll: boolean
  collapsedCount: number

  addLog: (entry: LogEntry) => void
  addLogs: (entries: LogEntry[]) => void
  setPaused: (paused: boolean) => void
  setAutoScroll: (auto: boolean) => void
  clearLogs: () => void
  getVisibleLogs: (maxEntries?: number) => LogEntry[]
}

const MAX_LOGS = 1000
const COLLAPSE_THRESHOLD = 200

export const useLogStore = create<LogState>((set, get) => ({
  logs: [],
  isPaused: false,
  isAutoScroll: true,
  collapsedCount: 0,

  addLog: (entry) =>
    set((state) => {
      if (state.isPaused) return state
      const logs = [...state.logs, entry]
      if (logs.length > MAX_LOGS) {
        const removed = logs.length - MAX_LOGS
        return { logs: logs.slice(-MAX_LOGS), collapsedCount: state.collapsedCount + removed }
      }
      return { logs }
    }),

  addLogs: (entries) =>
    set((state) => {
      if (state.isPaused) return state
      const logs = [...state.logs, ...entries]
      if (logs.length > MAX_LOGS) {
        const removed = logs.length - MAX_LOGS
        return { logs: logs.slice(-MAX_LOGS), collapsedCount: state.collapsedCount + removed }
      }
      return { logs }
    }),

  setPaused: (paused) => set({ isPaused: paused }),
  setAutoScroll: (auto) => set({ isAutoScroll: auto }),
  clearLogs: () => set({ logs: [], collapsedCount: 0 }),

  getVisibleLogs: (maxEntries = 200) => {
    const { logs } = get()
    if (logs.length <= COLLAPSE_THRESHOLD) return logs
    const debugCount = logs.filter((l) => l.level === "DEBUG").length
    const nonDebug = logs.filter((l) => l.level !== "DEBUG")
    if (debugCount > 0 && logs.length > maxEntries) {
      return nonDebug.slice(-maxEntries)
    }
    return logs.slice(-maxEntries)
  },
}))
