import Mock from 'mockjs'

export const scriptorMock = () => {
  // 获取可用状态
  Mock.mock('/api/scriptor/available-states', 'get', () => {
    return Mock.mock({
      success: true,
      data: ['join_wait', 'ready_done', 'ready', 'playing', 'finished', 'result']
    })
  })

  // 获取权重标题
  Mock.mock('/api/scriptor/weight-titles', 'get', () => {
    return Mock.mock({
      success: true,
      data: ['skilled', 'balanced', 'technical', 'rhythmic', 'challenging']
    })
  })

  // 检查歌曲ID
  Mock.mock(/\/api\/scriptor\/check-song\/\d+/, 'get', () => {
    return Mock.mock({
      success: true,
      data: Mock.Random.boolean()
    })
  })

  // 启动任务
  Mock.mock('/api/scriptor/start_scriptor/start', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        taskId: Mock.Random.guid(),
        status: 'running'
      }
    })
  })

  // 停止任务  
  Mock.mock('/api/scriptor/stop_scriptor/stop', 'post', () => {
    return Mock.mock({
      success: true,
      data: {
        status: 'stopped'
      }
    })
  })
}