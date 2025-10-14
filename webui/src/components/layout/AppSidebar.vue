<template>
  <aside class="app-sidebar" :class="{ 'is-collapsed': !expanded }">
    <div class="sidebar-content">
      <div class="status-items">
        <div
          v-for="status in backendStatus"
          :key="status.module"
          class="status-item"
        >
          <div class="status-icon">
            <component :is="getStatusIcon(status.status)" />
          </div>
          <div v-if="expanded" class="status-text">
            <div class="module-name">{{ status.module }}</div>
            <div class="task-status">{{ status.currentTask || 'Idle' }}</div>
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const expanded = computed(() => appStore.sidebarExpanded)
const backendStatus = computed(() => appStore.backendStatus)

const getStatusIcon = (status: string) => {
  // 返回对应状态的图标组件
  return 'div'
}
</script>