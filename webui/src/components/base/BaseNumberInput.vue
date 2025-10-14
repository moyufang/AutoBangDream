<!-- src/components/base/BaseNumberInput.vue -->
<template>
  <div class="base-number-input" :class="{ 'has-error': hasError }">
    <label v-if="label" class="input-label">{{ label }}</label>
    <div class="input-container">
      <button 
        v-if="showControls"
        type="button"
        class="control-btn decrement"
        :disabled="disabled || isMin"
        @click="decrement"
      >
        <slot name="decrement-icon">-</slot>
      </button>
      
      <input
        ref="inputRef"
        type="number"
        :value="modelValue"
        :min="min"
        :max="max"
        :step="step"
        :disabled="disabled"
        :placeholder="placeholder"
        @input="handleInput"
        @blur="handleBlur"
        class="number-input"
      />
      
      <button 
        v-if="showControls"
        type="button"
        class="control-btn increment"
        :disabled="disabled || isMax"
        @click="increment"
      >
        <slot name="increment-icon">+</slot>
      </button>
    </div>
    
    <div v-if="hasError" class="error-message">
      {{ errorMessage }}
    </div>
    
    <div v-if="description" class="input-description">
      {{ description }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'

interface Props {
  modelValue: number
  label?: string
  min?: number
  max?: number
  step?: number
  disabled?: boolean
  placeholder?: string
  showControls?: boolean
  description?: string
}

const props = withDefaults(defineProps<Props>(), {
  min: -Infinity,
  max: Infinity,
  step: 1,
  showControls: true,
  disabled: false
})

const emit = defineEmits<{
  'update:modelValue': [value: number]
  'change': [value: number]
  'blur': [value: number]
}>()

const inputRef = ref<HTMLInputElement>()

const hasError = ref(false)
const errorMessage = ref('')

const isMin = computed(() => props.modelValue <= props.min)
const isMax = computed(() => props.modelValue >= props.max)

const validate = (value: number): boolean => {
  if (value < props.min) {
    hasError.value = true
    errorMessage.value = `值不能小于 ${props.min}`
    return false
  }
  
  if (value > props.max) {
    hasError.value = true
    errorMessage.value = `值不能大于 ${props.max}`
    return false
  }
  
  hasError.value = false
  errorMessage.value = ''
  return true
}

const handleInput = (event: Event) => {
  const target = event.target as HTMLInputElement
  const value = parseFloat(target.value) || 0
  
  if (validate(value)) {
    emit('update:modelValue', value)
    emit('change', value)
  }
}

const handleBlur = (event: Event) => {
  const target = event.target as HTMLInputElement
  const value = parseFloat(target.value) || 0
  
  // 确保值在范围内
  let clampedValue = value
  if (value < props.min) clampedValue = props.min
  if (value > props.max) clampedValue = props.max
  
  if (clampedValue !== value) {
    emit('update:modelValue', clampedValue)
    emit('change', clampedValue)
  }
  
  emit('blur', clampedValue)
  validate(clampedValue)
}

const increment = () => {
  const newValue = props.modelValue + props.step
  if (newValue <= props.max) {
    emit('update:modelValue', newValue)
    emit('change', newValue)
    validate(newValue)
  }
}

const decrement = () => {
  const newValue = props.modelValue - props.step
  if (newValue >= props.min) {
    emit('update:modelValue', newValue)
    emit('change', newValue)
    validate(newValue)
  }
}

// 监听外部值变化进行验证
watch(() => props.modelValue, (value) => {
  validate(value)
}, { immediate: true })

// 暴露方法供父组件调用
defineExpose({
  focus: () => inputRef.value?.focus(),
  validate: () => validate(props.modelValue)
})
</script>

<style scoped lang="scss">
@import '@/styles/mixins';

.base-number-input {
  .input-label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
    color: #333;
  }

  .input-container {
    display: flex;
    align-items: center;
    border: 1px solid #dcdfe6;
    border-radius: 4px;
    overflow: hidden;
    transition: border-color 0.2s;

    &:focus-within {
      border-color: #409eff;
      box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
    }
  }

  .number-input {
    @include input-base;
    border: none;
    text-align: center;
    flex: 1;
    min-width: 0;

    /* 隐藏数字输入框的上下箭头 */
    &::-webkit-outer-spin-button,
    &::-webkit-inner-spin-button {
      -webkit-appearance: none;
      margin: 0;
    }
    
    // &[type=number] {
    //   -moz-appearance: textfield;
    // }
  }

  .control-btn {
    @include button-reset;
    padding: 8px 12px;
    background: #f5f7fa;
    color: #606266;
    transition: all 0.2s;
    min-width: 40px;

    &:hover:not(:disabled) {
      background: #e4e7ed;
      color: #409eff;
    }

    &:disabled {
      background: #f5f7fa;
      color: #c0c4cc;
      cursor: not-allowed;
    }
  }

  &.has-error {
    .input-container {
      border-color: #f56c6c;
    }
  }

  .error-message {
    color: #f56c6c;
    font-size: 12px;
    margin-top: 4px;
  }

  .input-description {
    color: #909399;
    font-size: 12px;
    margin-top: 4px;
  }
}
</style>