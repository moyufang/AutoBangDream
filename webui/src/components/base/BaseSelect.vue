<!-- src/components/base/BaseSelect.vue -->
<template>
  <div class="base-select" :class="{ 'is-open': isOpen, 'has-error': hasError }">
    <label v-if="label" class="select-label">{{ label }}</label>
    
    <div class="select-trigger" @click="toggleDropdown">
      <span class="selected-value">
        <slot name="selected" :value="selectedOption">
          {{ selectedOption?.label || placeholder }}
        </slot>
      </span>
      
      <span class="select-arrow">
        <slot name="arrow-icon">▼</slot>
      </span>
    </div>
    
    <div v-if="isOpen" class="select-dropdown">
      <div 
        v-for="option in options"
        :key="option.value"
        class="select-option"
        :class="{ 'is-selected': isSelected(option), 'is-disabled': option.disabled }"
        @click="selectOption(option)"
      >
        <slot name="option" :option="option">
          {{ option.label }}
        </slot>
      </div>
    </div>
    
    <div v-if="hasError" class="error-message">
      {{ errorMessage }}
    </div>
    
    <div v-if="description" class="select-description">
      {{ description }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch} from 'vue'

interface SelectOption {
  value: any
  label: string
  disabled?: boolean
}

interface Props {
  modelValue: any
  options: SelectOption[]
  label?: string
  placeholder?: string
  disabled?: boolean
  description?: string
}

const props = withDefaults(defineProps<Props>(), {
  placeholder: '请选择',
  disabled: false
})

const emit = defineEmits<{
  'update:modelValue': [value: any]
  'change': [value: any]
}>()

const isOpen = ref(false)
const hasError = ref(false)
const errorMessage = ref('')

const selectedOption = computed(() => {
  return props.options.find(opt => opt.value === props.modelValue)
})

const isSelected = (option: SelectOption) => {
  return option.value === props.modelValue
}

const toggleDropdown = () => {
  if (!props.disabled) {
    isOpen.value = !isOpen.value
  }
}

const selectOption = (option: SelectOption) => {
  if (option.disabled) return
  
  emit('update:modelValue', option.value)
  emit('change', option.value)
  isOpen.value = false
}

const handleClickOutside = (event: MouseEvent) => {
  const target = event.target as HTMLElement
  if (!target.closest('.base-select')) {
    isOpen.value = false
  }
}

// 验证函数
const validate = (value: any): boolean => {
  const isValid = props.options.some(opt => opt.value === value)
  if (!isValid && value !== undefined && value !== null) {
    hasError.value = true
    errorMessage.value = '选择的值不在选项范围内'
    return false
  }
  
  hasError.value = false
  errorMessage.value = ''
  return true
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  validate(props.modelValue)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})

// 监听值变化进行验证
watch(() => props.modelValue, (value) => {
  validate(value)
})

defineExpose({
  validate: () => validate(props.modelValue)
})
</script>

<style scoped lang="scss">
@import '@/styles/mixins';

.base-select {
  position: relative;
  
  .select-label {
    display: block;
    margin-bottom: 8px;
    font-weight: 500;
    color: #333;
  }

  .select-trigger {
    @include input-base;
    display: flex;
    justify-content: space-between;
    align-items: center;
    cursor: pointer;
    user-select: none;
    
    &:hover {
      border-color: #c0c4cc;
    }
  }

  .selected-value {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .select-arrow {
    transition: transform 0.2s;
    margin-left: 8px;
    color: #c0c4cc;
  }

  &.is-open {
    .select-arrow {
      transform: rotate(180deg);
    }
    
    .select-trigger {
      border-color: #409eff;
    }
  }

  .select-dropdown {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: white;
    border: 1px solid #e4e7ed;
    border-radius: 4px;
    box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
    z-index: 1000;
    max-height: 200px;
    overflow-y: auto;
    margin-top: 4px;
    
    @include scrollbar;
  }

  .select-option {
    padding: 8px 12px;
    cursor: pointer;
    transition: background-color 0.2s;
    
    &:hover:not(.is-disabled) {
      background-color: #f5f7fa;
    }
    
    &.is-selected {
      background-color: #ecf5ff;
      color: #409eff;
    }
    
    &.is-disabled {
      color: #c0c4cc;
      cursor: not-allowed;
    }
  }

  &.has-error {
    .select-trigger {
      border-color: #f56c6c;
    }
  }

  .error-message {
    color: #f56c6c;
    font-size: 12px;
    margin-top: 4px;
  }

  .select-description {
    color: #909399;
    font-size: 12px;
    margin-top: 4px;
  }
}
</style>