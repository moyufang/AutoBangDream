// src/services/mock/songRecognition.ts
import Mock from 'mockjs'

export const songRecognitionMock = () => {
  // 获取训练配置
  Mock.mock('/api/song-recognition/train-config', 'get', () => {
    return Mock.mock({
      success: true,
      data: {
        is_load_model: true,
        epoch: 20,
        num_batches: 32,
        batch_size: 64,
        learn_rate: 0.01
      }
    })
  })

  // 更新训练配置
  Mock.mock('/api/song-recognition/train-config', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '配置已保存' }
    })
  })

  // 启动歌曲添加任务
  Mock.mock('/api/song-recognition/add-song/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running',
        requiresInteraction: true
      }
    })
  })

  // 停止歌曲添加任务
  Mock.mock('/api/song-recognition/add-song/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: { status: 'stopped' }
    })
  })

  // 启动训练任务
  Mock.mock('/api/song-recognition/train/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running'
      }
    })
  })

  // 停止训练任务
  Mock.mock('/api/song-recognition/train/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: { status: 'stopped' }
    })
  })

  // 交互响应 (Yes/No/Drop/Stop)
  Mock.mock('/api/song-recognition/interact', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '交互已处理' }
    })
  })

  // 提交 song_id
  Mock.mock('/api/song-recognition/submit-song-id', 'post', () => {
    return Mock.mock({
      success: true,
      data: { message: '歌曲ID已提交' }
    })
  })
}