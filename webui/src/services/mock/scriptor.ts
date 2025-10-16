import Mock from 'mockjs'
import {Mode, Event, Choose, Level, Performance} from '../types/enums'

export const scriptorMock = () => {
  Mock.mock('/api/scriptor/config', 'get', ()=>{
    return Mock.mock({
      sucess: true,
      data:{
          dilation_time: 1000000,
          correction_time: -45000,
          mumu_port: 7555,
          server_port: 31415,
          bangcheater_port: 12345,
          is_no_action: false,
          is_caliboration: false,
          play_one_song_id: 655,
          is_play_one_song: false,
          is_restart_play: true,
          is_checking_3d: true,
          is_repeat: true,
          MAX_SAME_STATE: 100,
          MAX_RE_READY: 10,
          is_allow_save: true,
          protected_state: ['join_wait', 'ready_done'],
          special_state_list: ['ready'],
          mode: Mode.Event,
          event: Event.Compete,
          choose: Choose.Loop,
          level: Level.Expert,
          performance: Performance.AllPerfect,
          weight_title: 'skilled',
          lobby: true
        }
    })
  })

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