import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { BackendStatus } from '@/services/types/api'

export const useAppStore = defineStore('app', () => {
  // 侧边栏状态
  const sidebarExpanded = ref(true)
  
  // 当前运行的任务（实现任务互斥）
  const currentRunningTask = ref<string | null>(null)
  
  // 后端状态信息
  const backendStatus = ref<BackendStatus[]>([])
  
  // 计算属性
  const isAnyTaskRunning = computed(() => currentRunningTask.value !== null)
  
  // Actions
  const toggleSidebar = () => {
    sidebarExpanded.value = !sidebarExpanded.value
  }
  
  const setRunningTask = (task: string | null) => {
    currentRunningTask.value = task
  }
  
  const updateBackendStatus = (status: BackendStatus[]) => {
    backendStatus.value = status
  }

  return {
    sidebarExpanded,
    currentRunningTask,
    backendStatus,
    isAnyTaskRunning,
    toggleSidebar,
    setRunningTask,
    updateBackendStatus
  }
})