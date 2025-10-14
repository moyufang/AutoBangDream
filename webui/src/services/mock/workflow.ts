// src/services/mock/workflow.ts
import Mock from 'mockjs'
import { WorkflowMode } from '../types/enums'

export const workflowMock = () => {
  // 获取工作流配置
  Mock.mock('/api/workflow/config', 'get', () => {
    return Mock.mock({
      success: true,
      data: {
        workflow_mode: WorkflowMode.Record
      }
    })
  })

  // 更新工作流配置
  Mock.mock('/api/workflow/config', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '配置已保存' }
    })
  })

  // 启动工作流任务
  Mock.mock('/api/workflow/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running'
      }
    })
  })

  // 停止工作流任务
  Mock.mock('/api/workflow/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: { status: 'stopped' }
    })
  })

  // 获取图片数据
  Mock.mock('/api/workflow/image', 'get', () => {
    // 生成模拟的 base64 图片数据（实际上是一个很小的透明图片）
    const mockImageData = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
    
    return Mock.mock({
      success: true,
      data: {
        image: mockImageData,
        width: 1280,
        height: 720
      }
    })
  })

  // 获取像素颜色
  Mock.mock(/\/api\/workflow\/color\/\d+\/\d+/, 'get', () => {
    return Mock.mock({
      success: true,
      data: {
        h: Mock.Random.integer(0, 360),
        s: Mock.Random.integer(0, 100),
        v: Mock.Random.integer(0, 100),
        x: Mock.Random.integer(0, 1279),
        y: Mock.Random.integer(0, 719)
      }
    })
  })
}