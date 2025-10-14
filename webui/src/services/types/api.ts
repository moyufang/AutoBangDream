// 通用类型定义
export interface ApiResponse<T = any> {
  success: boolean
  data: T
  message?: string
}

export interface TaskConfig {
  [key: string]: any
}

export interface LogMessage {
  module: string
  task: string
  level: 'info' | 'warn' | 'error'
  message: string
  timestamp: number
}

export interface BackendStatus {
  module: string
  status: 'idle' | 'running' | 'error'
  currentTask?: string
  progress?: number
}