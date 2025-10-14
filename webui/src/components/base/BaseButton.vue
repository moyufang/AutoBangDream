<!-- src/components/base/BaseButton.vue -->
<template>
  <button
    :type="nativeType"
    :disabled="disabled || loading"
    :class="buttonClasses"
    @click="handleClick"
  >
    <span v-if="loading" class="button-loading">
      <slot name="loading-icon">⏳</slot>
    </span>
    
    <span v-else class="button-icon">
      <slot name="icon"></slot>
    </span>
    
    <span class="button-text">
      <slot></slot>
    </span>
  </button>
</template>

<script setup lang="ts">
import {computed} from 'vue'
interface Props {
  type?: 'primary' | 'success' | 'warning' | 'danger' | 'info' | 'default'
  size?: 'large' | 'medium' | 'small' | 'mini'
  nativeType?: 'button' | 'submit' | 'reset'
  disabled?: boolean
  loading?: boolean
  round?: boolean
  circle?: boolean
  plain?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  type: 'default',
  size: 'medium',
  nativeType: 'button',
  disabled: false,
  loading: false,
  round: false,
  circle: false,
  plain: false
})

const emit = defineEmits<{
  'click': [event: MouseEvent]
}>()

const buttonClasses = computed(() => [
  'base-button',
  `button-${props.type}`,
  `button-${props.size}`,
  {
    'is-disabled': props.disabled,
    'is-loading': props.loading,
    'is-round': props.round,
    'is-circle': props.circle,
    'is-plain': props.plain
  }
])

const handleClick = (event: MouseEvent) => {
  if (!props.disabled && !props.loading) {
    emit('click', event)
  }
}
</script>

<style scoped lang="scss">
@import '@/styles/mixins';

.base-button {
  @include button-reset;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 1px solid;
  border-radius: 4px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  white-space: nowrap;
  
  // 尺寸
  &.button-large {
    padding: 12px 20px;
    font-size: 14px;
  }
  
  &.button-medium {
    padding: 10px 16px;
    font-size: 14px;
  }
  
  &.button-small {
    padding: 8px 12px;
    font-size: 12px;
  }
  
  &.button-mini {
    padding: 6px 10px;
    font-size: 12px;
  }
  
  // 圆角
  &.is-round {
    border-radius: 20px;
  }
  
  &.is-circle {
    border-radius: 50%;
    aspect-ratio: 1;
  }
  
  // 类型
  &.button-default {
    background: white;
    border-color: #dcdfe6;
    color: #606266;
    
    &:hover:not(.is-disabled) {
      border-color: #c0c4cc;
      color: #409eff;
    }
  }
  
  &.button-primary {
    background: #409eff;
    border-color: #409eff;
    color: white;
    
    &:hover:not(.is-disabled) {
      background: #66b1ff;
      border-color: #66b1ff;
    }
    
    &.is-plain {
      background: #ecf5ff;
      border-color: #b3d8ff;
      color: #409eff;
      
      &:hover:not(.is-disabled) {
        background: #409eff;
        border-color: #409eff;
        color: white;
      }
    }
  }
  
  &.button-success {
    background: #67c23a;
    border-color: #67c23a;
    color: white;
    
    &:hover:not(.is-disabled) {
      background: #85ce61;
      border-color: #85ce61;
    }
  }
  
  &.button-warning {
    background: #e6a23c;
    border-color: #e6a23c;
    color: white;
    
    &:hover:not(.is-disabled) {
      background: #ebb563;
      border-color: #ebb563;
    }
  }
  
  &.button-danger {
    background: #f56c6c;
    border-color: #f56c6c;
    color: white;
    
    &:hover:not(.is-disabled) {
      background: #f78989;
      border-color: #f78989;
    }
  }
  
  // 禁用状态
  &.is-disabled {
    opacity: 0.6;
    cursor: not-allowed;
    
    &:hover {
      opacity: 0.6;
    }
  }
  
  // 加载状态
  &.is-loading {
    pointer-events: none;
    opacity: 0.7;
  }
  
  .button-loading {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
}
</style>