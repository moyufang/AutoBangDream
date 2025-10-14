import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { BackendStatus } from '@/services/types/api'

export const useAppStore = defineStore('controller', () => {
  // bangcheater 启动状态
  const isBangcheaterRunning = ref(false)
  
  return {
    isBangcheaterRunning
  }
})