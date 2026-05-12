import { create } from "zustand"

interface ConnectionState {
  status: "connected" | "disconnected" | "reconnecting"
  reconnectCount: number
  lastMessageTime: number

  setStatus: (status: ConnectionState["status"]) => void
  incrementReconnect: () => void
  resetReconnect: () => void
}

export const useConnectionStore = create<ConnectionState>((set) => ({
  status: "disconnected",
  reconnectCount: 0,
  lastMessageTime: 0,

  setStatus: (status) => set({ status, lastMessageTime: status === "connected" ? Date.now() : 0 }),
  incrementReconnect: () => set((s) => ({ reconnectCount: s.reconnectCount + 1 })),
  resetReconnect: () => set({ reconnectCount: 0 }),
}))
