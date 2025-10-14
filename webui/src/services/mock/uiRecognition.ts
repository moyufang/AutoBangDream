// src/services/mock/uiRecognition.ts
import Mock from 'mockjs'

export const uiRecognitionMock = () => {
  // 获取训练配置
  Mock.mock('/api/ui-recognition/train-config', 'get', () => {
    return Mock.mock({
      success: true,
      data: {
        is_load_model: true,
        epoch: 20,
        batch_size: 128,
        learn_rate: 0.01
      }
    })
  })

  // 更新训练配置
  Mock.mock('/api/ui-recognition/train-config', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '配置已保存' }
    })
  })

  // 启动图片添加任务
  Mock.mock('/api/ui-recognition/add-img/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running',
        requiresInteraction: true
      }
    })
  })

  // 停止图片添加任务
  Mock.mock('/api/ui-recognition/add-img/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: { status: 'stopped' }
    })
  })

  // 启动训练任务
  Mock.mock('/api/ui-recognition/train/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running'
      }
    })
  })

  // 停止训练任务
  Mock.mock('/api/ui-recognition/train/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: { status: 'stopped' }
    })
  })

  // 交互响应 (Yes/No/Drop/Stop)
  Mock.mock('/api/ui-recognition/interact', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '交互已处理' }
    })
  })

  // 提交 img_id
  Mock.mock('/api/ui-recognition/submit-img-id', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '图片ID已提交' }
    })
  })
}