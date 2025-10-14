<!-- src/components/base/BaseTupleInput.vue -->
<template>
  <div class="base-tuple-input" :class="{ 'has-error': hasError }">
    <label v-if="label" class="tuple-label">{{ label }}</label>
    
    <div class="tuple-container">
      <div 
        v-for="(item, index) in tupleValues"
        :key="index"
        class="tuple-item"
      >
        <input
          v-if="schema[index] === 'string'"
          v-model="tupleValues[index]"
          type="text"
          :placeholder="`元素 ${index + 1}`"
          @input="handleItemChange(index, $event)"
          class="tuple-input"
        />
        
        <BaseNumberInput
          v-else-if="schema[index] === 'number'"
          v-model="tupleValues[index]"
          :min="constraints[index]?.min"
          :max="constraints[index]?.max"
          :step="constraints[index]?.step || 0.01"
          :show-controls="false"
          :placeholder="`元素 ${index + 1}`"
          @update:modelValue="handleItemChange(index, $event)"
          class="tuple-number-input"
        />
        
        <span v-if="index < tupleValues.length - 1" class="tuple-separator">,</span>
      </div>
    </div>
    
    <div v-if="hasError" class="error-message">
      {{ errorMessage }}
    </div>
    
    <div v-if="description" class="tuple-description">
      {{ description }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import BaseNumberInput from './BaseNumberInput.vue'

interface TupleConstraint {
  min?: number
  max?: number
  step?: number
}

interface Props {
  modelValue: any[]
  schema: ('string' | 'number')[]
  constraints?: TupleConstraint[]
  label?: string
  description?: string
}

const props = withDefaults(defineProps<Props>(), {
  constraints: () => [],
  schema: () => []
})

const emit = defineEmits<{
  'update:modelValue': [value: any[]]
  'change': [value: any[]]
}>()

const tupleValues = ref<any[]>([])
const hasError = ref(false)
const errorMessage = ref('')

// 初始化值
const initializeValues = () => {
  if (props.modelValue && props.modelValue.length === props.schema.length) {
    tupleValues.value = [...props.modelValue]
  } else {
    // 使用默认值初始化
    tupleValues.value = props.schema.map(type => 
      type === 'number' ? 0 : ''
    )
  }
}

const handleItemChange = (index: number, value: any) => {
  // 类型转换
  if (props.schema[index] === 'number') {
    tupleValues.value[index] = parseFloat(value) || 0
  } else {
    tupleValues.value[index] = value
  }
  
  validateTuple()
  emit('update:modelValue', [...tupleValues.value])
  emit('change', [...tupleValues.value])
}

const validateTuple = (): boolean => {
  // 检查每个元素是否符合约束
  for (let i = 0; i < tupleValues.value.length; i++) {
    const value = tupleValues.value[i]
    const constraint = props.constraints[i]
    
    if (props.schema[i] === 'number' && constraint) {
      if (constraint.min !== undefined && value < constraint.min) {
        hasError.value = true
        errorMessage.value = `第 ${i + 1} 个元素不能小于 ${constraint.min}`
        return false
      }
      
      if (constraint.max !== undefined && value > constraint.max) {
        hasError.value = true
        errorMessage.value = `第 ${i + 1} 个元素不能大于 ${constraint.max}`
        return false
      }
    }
  }
  
  hasError.value = false
  errorMessage.value = ''
  return true
}

// 初始化
initializeValues()

// 监听外部值变化
watch(() => props.modelValue, (newValue) => {
  if (newValue && newValue.length === props.schema.length) {
    tupleValues.value = [...newValue]
    validateTuple()
  }
}, { deep: true })

defineExpose({
  validate: validateTuple
})
</script>

<style scoped lang="scss">
@import '@/styles/mixins';

.base-tuple-input {
  .tuple-label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
    color: #333;
  }

  .tuple-container {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px;
    border: 1px solid #dcdfe6;
    border-radius: 4px;
    background: #fafafa;
    
    &:focus-within {
      border-color: #409eff;
    }
  }

  .tuple-item {
    display: flex;
    align-items: center;
  }

  .tuple-input {
    @include input-base;
    width: 100px;
    text-align: center;
  }

  .tuple-number-input {
    width: 100px;
    
    :deep(.input-container) {
      border: none;
      background: white;
    }
    
    :deep(.number-input) {
      text-align: center;
      background: white;
    }
  }

  .tuple-separator {
    color: #909399;
    margin: 0 4px;
    font-weight: bold;
  }

  &.has-error {
    .tuple-container {
      border-color: #f56c6c;
    }
  }

  .error-message {
    color: #f56c6c;
    font-size: 12px;
    margin-top: 4px;
  }

  .tuple-description {
    color: #909399;
    font-size: 12px;
    margin-top: 4px;
  }
}
</style>