import type { TestMonitorMessage, TestMonitorEventType } from "../types/test"

type MessageHandler = (msg: TestMonitorMessage) => void

export class TestMonitorConnection {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnect = 10
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private handlers: Map<TestMonitorEventType, MessageHandler[]> = new Map()
  private anyHandlers: MessageHandler[] = []
  private _status: "connected" | "disconnected" | "reconnecting" = "disconnected"
  private onStatusChange?: (status: string) => void
  private lastMsgTime = 0
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null

  constructor(
    private uri: string = "ws://127.0.0.1:8091",
    onStatusChange?: (status: string) => void,
  ) {
    this.onStatusChange = onStatusChange
  }

  connect(): void {
    try {
      this.ws = new WebSocket(this.uri)
      this.ws.onopen = () => {
        this.reconnectAttempts = 0
        this._status = "connected"
        this.onStatusChange?.(this._status)
        this.startHeartbeat()
      }
      this.ws.onmessage = (event) => {
        this.lastMsgTime = Date.now()
        try {
          const msg: TestMonitorMessage = JSON.parse(event.data)
          const typeHandlers = this.handlers.get(msg.type) || []
          typeHandlers.forEach((h) => h(msg))
          this.anyHandlers.forEach((h) => h(msg))
        } catch {
          // ignore
        }
      }
      this.ws.onclose = () => {
        this.stopHeartbeat()
        this._status = "disconnected"
        this.onStatusChange?.(this._status)
        this.scheduleReconnect()
      }
      this.ws.onerror = () => {
        this._status = "disconnected"
        this.onStatusChange?.(this._status)
      }
    } catch {
      this.scheduleReconnect()
    }
  }

  disconnect(): void {
    this.stopHeartbeat()
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.ws?.close()
    this.ws = null
  }

  on(type: TestMonitorEventType, handler: MessageHandler): void {
    const existing = this.handlers.get(type) || []
    existing.push(handler)
    this.handlers.set(type, existing)
  }

  onAny(handler: MessageHandler): void {
    this.anyHandlers.push(handler)
  }

  send(data: Record<string, unknown>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    }
  }

  get status(): string {
    return this._status
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnect) return
    this._status = "reconnecting"
    this.onStatusChange?.(this._status)
    const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 16000)
    this.reconnectAttempts++
    this.reconnectTimer = setTimeout(() => this.connect(), delay)
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (Date.now() - this.lastMsgTime > 30000) {
        this.send({ type: "ping" })
      }
    }, 15000)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer)
      this.heartbeatTimer = null
    }
  }
}
