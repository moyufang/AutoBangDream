import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { Mode, Event, Choose, Level, Performance } from '@/services/types/enums'

export const useScriptorStore = defineStore('scriptor', () => {
  // 配置状态
  const config = ref({
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
  })

  const updateConfig = (key:string, value:any) => {
    (config.value as any)[key] = value
  }

  // 批量更新配置
  const updateMultipleConfig = (newConfig:any) => {
    Object.assign(config.value, newConfig)
  }


  // 日志
  const logs = ref<string[]>([])
  
  // 可用选项
  const availableStates = ref<string[]>([])
  const availableWeightTitles = ref<string[]>([])

  // 计算属性 - 条件显示
  const showLobbyConfig = computed(() => {
    const { event } = config.value
    return [Event.Tour, Event.Compete, Event.Team].includes(event)
  })

  // Actions
  const addLog = (message: string) => {
    logs.value.push(`[${new Date().toLocaleTimeString()}] ${message}`)
    // 限制日志数量
    if (logs.value.length > 1000) {
      logs.value = logs.value.slice(-1000)
    }
  }

  const clearLogs = () => {
    logs.value = []
  }

  return {
    config,
    updateConfig,
    updateMultipleConfig,
    logs,
    availableStates,
    availableWeightTitles,
    showLobbyConfig,
    addLog,
    clearLogs
  }
})