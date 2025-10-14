// src/services/mock/fetch.ts
import Mock from 'mockjs'
import { FetchMode } from '../types/enums'

export const fetchMock = () => {
  // 获取抓取配置
  Mock.mock('/api/fetch/config', 'get', () => {
    return Mock.mock({
      success: true,
      data: {
        fetch_mode: FetchMode.FetchLack
      }
    })
  })

  // 更新抓取配置
  Mock.mock('/api/fetch/config', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '配置已保存' }
    })
  })

  // 启动抓取任务
  Mock.mock('/api/fetch/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running'
      }
    })
  })

  // 停止抓取任务
  Mock.mock('/api/fetch/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: { status: 'stopped' }
    })
  })

  // 获取抓取进度
  Mock.mock('/api/fetch/progress', 'get', () => {
    return Mock.mock({
      success: true,
      data: {
        progress: Mock.Random.float(0, 100, 2, 2),
        current: Mock.Random.integer(1, 100),
        total: Mock.Random.integer(100, 500),
        status: 'fetching'
      }
    })
  })
}