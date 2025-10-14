// src/services/mock/index.ts
import Mock from 'mockjs'
import { scriptorMock } from './scriptor'
import { songRecognitionMock } from './songRecognition'
import { uiRecognitionMock } from './uiRecognition'
import { fetchMock } from './fetch'
import { workflowMock } from './workflow'

// 模拟延迟
Mock.setup({
  timeout: '200-800'
})

// 注册所有mock
scriptorMock()
songRecognitionMock() 
uiRecognitionMock()
fetchMock()
workflowMock()

// 全局状态接口
Mock.mock('/api/status', 'get', () => {
  return Mock.mock({
    success: true,
    data: {
      currentTask: null,
      tasks: []
    }
  })
})

// 模拟 WebSocket 消息流
export const createMockWebSocket = (onMessage: (data: any) => void) => {
  const modules = ['scriptor', 'song-recognition', 'ui-recognition', 'fetch', 'workflow']
  const tasks = {
    'scriptor': ['start_scriptor'],
    'song-recognition': ['add-song', 'train'],
    'ui-recognition': ['add-img', 'train'], 
    'fetch': ['fetch'],
    'workflow': ['workflow']
  }

  // 模拟日志消息
  const logInterval = setInterval(() => {
    const module = Mock.Random.pick(modules)
    const moduleTasks = tasks[module as keyof typeof tasks]
    const task = Mock.Random.pick(moduleTasks)
    
    const logMessage = {
      type: 'log',
      module,
      task,
      data: {
        level: Mock.Random.pick(['info', 'warn', 'error']),
        message: Mock.Random.sentence(3, 8)
      },
      timestamp: Date.now()
    }
    
    onMessage(logMessage)
  }, 2000)

  // 模拟状态更新
  const statusInterval = setInterval(() => {
    const statusMessage = {
      type: 'status',
      module: 'system',
      data: {
        modules: modules.map(module => ({
          module,
          status: Mock.Random.pick(['idle', 'running', 'error']),
          currentTask: Mock.Random.boolean() ? tasks[module as keyof typeof tasks]?.[0] : null,
          progress: Mock.Random.float(0, 100, 2, 2)
        }))
      },
      timestamp: Date.now()
    }
    
    onMessage(statusMessage)
  }, 5000)

  return () => {
    clearInterval(logInterval)
    clearInterval(statusInterval)
  }
}