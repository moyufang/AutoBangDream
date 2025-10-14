// src/services/api.ts
import axios from 'axios'
import type { ApiResponse, TaskConfig } from './types/api'
import type { 
  Mode, Event, Choose, Level, Performance, 
  FetchMode, WorkflowMode 
} from './types/enums'

const api = axios.create({
  baseURL: '/api',
  timeout: 10000
})

// 请求拦截器 - 添加认证等
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加 token 等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器 - 统一处理错误
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

// 通用任务控制API
export const taskAPI = {
  start(module: string, task: string, config?: TaskConfig) {
    return api.post<ApiResponse>(`/${module}/${task}/start`, config)
  },
  
  stop(module: string, task: string) {
    return api.post<ApiResponse>(`/${module}/${task}/stop`)
  },
  
  getStatus(module: string) {
    return api.get<ApiResponse>(`/${module}/status`)
  }
}

// Scriptor 模块 API
export const scriptorAPI = {
  getConfig() {
    return api.get<ApiResponse>('/scriptor/config')
  },
  
  updateConfig(config: any) {
    return api.post<ApiResponse>('/scriptor/config', config)
  },
  
  getAvailableStates() {
    return api.get<ApiResponse<string[]>>('/scriptor/available-states')
  },
  
  getWeightTitles() {
    return api.get<ApiResponse<string[]>>('/scriptor/weight-titles')
  },
  
  checkSongId(songId: number) {
    return api.get<ApiResponse<boolean>>(`/scriptor/check-song/${songId}`)
  }
}

// Song Recognition 模块 API
export const songRecognitionAPI = {
  getTrainConfig() {
    return api.get<ApiResponse>('/song-recognition/train-config')
  },
  
  updateTrainConfig(config: any) {
    return api.post<ApiResponse>('/song-recognition/train-config', config)
  },
  
  interact(action: 'Yes' | 'No' | 'Drop' | 'Stop') {
    return api.post<ApiResponse>('/song-recognition/interact', { action })
  },
  
  submitSongId(songId: number) {
    return api.post<ApiResponse>('/song-recognition/submit-song-id', { songId })
  }
}

// UI Recognition 模块 API
export const uiRecognitionAPI = {
  getTrainConfig() {
    return api.get<ApiResponse>('/ui-recognition/train-config')
  },
  
  updateTrainConfig(config: any) {
    return api.post<ApiResponse>('/ui-recognition/train-config', config)
  },
  
  interact(action: 'Yes' | 'No' | 'Drop' | 'Stop') {
    return api.post<ApiResponse>('/ui-recognition/interact', { action })
  },
  
  submitImgId(imgId: number) {
    return api.post<ApiResponse>('/ui-recognition/submit-img-id', { imgId })
  }
}

// Fetch 模块 API
export const fetchAPI = {
  getConfig() {
    return api.get<ApiResponse>('/fetch/config')
  },
  
  updateConfig(config: { fetch_mode: FetchMode }) {
    return api.post<ApiResponse>('/fetch/config', config)
  },
  
  getProgress() {
    return api.get<ApiResponse>('/fetch/progress')
  }
}

// Workflow 模块 API
export const workflowAPI = {
  getConfig() {
    return api.get<ApiResponse>('/workflow/config')
  },
  
  updateConfig(config: { workflow_mode: WorkflowMode }) {
    return api.post<ApiResponse>('/workflow/config', config)
  },
  
  getImage() {
    return api.get<ApiResponse>('/workflow/image')
  },
  
  getColor(x: number, y: number) {
    return api.get<ApiResponse>(`/workflow/color/${x}/${y}`)
  }
}

// 全局状态 API
export const statusAPI = {
  getGlobalStatus() {
    return api.get<ApiResponse>('/status')
  }
}

export default api