// src/utils/websocket.ts
import type { LogMessage, BackendStatus } from '@/services/types/api'

export interface WebSocketMessage {
  type: 'log' | 'status' | 'image' | 'interaction' | 'error'
  module: string
  task?: string
  data: any
  timestamp: number
}

export class WebSocketService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 3000
  private messageHandlers: Map<string, ((data: any) => void)[]> = new Map()

  constructor(private url: string) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)
        
        this.ws.onopen = () => {
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }

        this.ws.onclose = () => {
          console.log('WebSocket disconnected')
          this.handleReconnection()
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          reject(error)
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  send(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
    }
  }

  on(messageType: string, handler: (data: any) => void): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, [])
    }
    this.messageHandlers.get(messageType)!.push(handler)
  }

  off(messageType: string, handler: (data: any) => void): void {
    const handlers = this.messageHandlers.get(messageType)
    if (handlers) {
      const index = handlers.indexOf(handler)
      if (index > -1) {
        handlers.splice(index, 1)
      }
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    const handlers = this.messageHandlers.get(message.type) || []
    handlers.forEach(handler => handler(message))

    // 同时触发模块特定的消息处理
    const moduleHandlers = this.messageHandlers.get(`${message.type}.${message.module}`) || []
    moduleHandlers.forEach(handler => handler(message))
  }

  private handleReconnection(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
      
      setTimeout(() => {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error)
        })
      }, this.reconnectInterval)
    } else {
      console.error('Max reconnection attempts reached')
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// 创建全局 WebSocket 实例
export const wsService = new WebSocketService('ws://localhost:8000/ws')

// Mock WebSocket 用于开发环境
export const createMockWebSocket = () => {
  let messageCount = 0
  
  const mockLogs = [
    'Initializing scriptor module...',
    'Configuration loaded successfully',
    'Starting task execution',
    'Processing data batch 1/10',
    'Task completed successfully'
  ]

  return {
    connect: (onMessage: (message: WebSocketMessage) => void) => {
      // 模拟实时日志输出
      const interval = setInterval(() => {
        if (messageCount < mockLogs.length) {
          const message: WebSocketMessage = {
            type: 'log',
            module: 'scriptor',
            task: 'start_scriptor',
            data: {
              level: 'info',
              message: mockLogs[messageCount]
            },
            timestamp: Date.now()
          }
          onMessage(message)
          messageCount++
        } else {
          clearInterval(interval)
        }
      }, 1000)

      return () => clearInterval(interval)
    }
  }
}