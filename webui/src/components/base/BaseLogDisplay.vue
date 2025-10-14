<!-- src/components/base/BaseLogDisplay.vue -->
<template>
  <div class="base-log-display">
    <div class="log-header">
      <span class="log-title">{{ title }}</span>
      <div class="log-actions">
        <button 
          v-if="showClear"
          type="button"
          class="action-btn"
          @click="clearLogs"
        >
          <slot name="clear-icon">🗑️</slot>
          <span v-if="!compact">清空</span>
        </button>
        
        <button 
          type="button"
          class="action-btn"
          @click="toggleAutoScroll"
        >
          <slot name="scroll-icon" :autoScroll="autoScroll">
            {{ autoScroll ? '🔒' : '🔓' }}
          </slot>
          <span v-if="!compact">{{ autoScroll ? '锁定' : '解锁' }}</span>
        </button>
        
        <button 
          v-if="showCopy"
          type="button"
          class="action-btn"
          @click="copyLogs"
        >
          <slot name="copy-icon">📋</slot>
          <span v-if="!compact">复制</span>
        </button>
      </div>
    </div>
    
    <div 
      ref="logContainerRef"
      class="log-content"
      :class="{ 'compact': compact }"
    >
      <div
        v-for="(log, index) in logs"
        :key="index"
        class="log-entry"
        :class="getLogLevelClass(log.level)"
      >
        <span class="log-time">{{ formatTime(log.timestamp) }}</span>
        <span class="log-level">[{{ log.level.toUpperCase() }}]</span>
        <span class="log-message">{{ log.message }}</span>
      </div>
      
      <div v-if="logs.length === 0" class="log-empty">
        暂无日志
      </div>
    </div>
    
    <div v-if="showStats" class="log-stats">
      共 {{ logs.length }} 条日志
      <span v-if="errorCount > 0" class="error-count">({{ errorCount }} 个错误)</span>
    </div>
  </div>
</template>
 
<script setup lang="ts">
import { ref, computed, nextTick, watch} from 'vue'

interface LogEntry {
  message: string
  level: 'info' | 'warn' | 'error'
  timestamp: number
}

interface Props {
  logs: LogEntry[]
  title?: string
  maxLines?: number
  autoScroll?: boolean
  compact?: boolean
  showClear?: boolean
  showCopy?: boolean
  showStats?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  title: '日志输出',
  maxLines: 1000,
  autoScroll: true,
  compact: false,
  showClear: true,
  showCopy: true,
  showStats: true
})

const emit = defineEmits<{
  'clear': []
}>()

const logContainerRef = ref<HTMLDivElement>()
const autoScroll = ref(props.autoScroll)

const errorCount = computed(() => {
  return props.logs.filter(log => log.level === 'error').length
})

const getLogLevelClass = (level: string) => {
  return `level-${level}`
}

const formatTime = (timestamp: number) => {
  return new Date(timestamp).toLocaleTimeString()
}

const scrollToBottom = () => {
  if (logContainerRef.value && autoScroll.value) {
    nextTick(() => {
      logContainerRef.value!.scrollTop = logContainerRef.value!.scrollHeight
    })
  }
}

const clearLogs = () => {
  emit('clear')
}

const toggleAutoScroll = () => {
  autoScroll.value = !autoScroll.value
  if (autoScroll.value) {
    scrollToBottom()
  }
}

const copyLogs = async () => {
  const logText = props.logs
    .map(log => `[${formatTime(log.timestamp)}] [${log.level.toUpperCase()}] ${log.message}`)
    .join('\n')
  
  try {
    await navigator.clipboard.writeText(logText)
    // 可以添加成功提示
  } catch (err) {
    console.error('复制失败:', err)
  }
}

// 监听日志变化自动滚动
watch(() => props.logs.length, () => {
  scrollToBottom()
})

defineExpose({
  scrollToBottom,
  clear: clearLogs
})
</script>

<style scoped lang="scss">
@import '@/styles/mixins';

.base-log-display {
  border: 1px solid #e4e7ed;
  border-radius: 4px;
  background: white;
  
  .log-header {
    @include flex-between;
    padding: 12px 16px;
    border-bottom: 1px solid #e4e7ed;
    background: #f5f7fa;
  }

  .log-title {
    font-weight: 500;
    color: #303133;
  }

  .log-actions {
    display: flex;
    gap: 8px;
  }

  .action-btn {
    @include button-reset;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    border-radius: 3px;
    font-size: 12px;
    color: #606266;
    transition: all 0.2s;
    
    &:hover {
      background: #e4e7ed;
      color: #409eff;
    }
  }

  .log-content {
    height: 300px;
    overflow-y: auto;
    padding: 8px;
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 12px;
    line-height: 1.4;
    
    @include scrollbar;
    
    &.compact {
      height: 200px;
      font-size: 11px;
    }
  }

  .log-entry {
    margin-bottom: 2px;
    word-break: break-all;
    
    &.level-info {
      color: #606266;
    }
    
    &.level-warn {
      color: #e6a23c;
    }
    
    &.level-error {
      color: #f56c6c;
      font-weight: 500;
    }
  }

  .log-time {
    color: #909399;
    margin-right: 8px;
  }

  .log-level {
    margin-right: 8px;
    font-weight: 500;
  }

  .log-empty {
    text-align: center;
    color: #c0c4cc;
    padding: 20px;
  }

  .log-stats {
    padding: 8px 16px;
    border-top: 1px solid #e4e7ed;
    font-size: 12px;
    color: #909399;
    background: #fafafa;
  }

  .error-count {
    color: #f56c6c;
  }
}
</style>