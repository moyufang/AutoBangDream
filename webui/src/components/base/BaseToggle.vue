<template>
  <div class="base-toggle" :class="{ 'is-checked': modelValue }">
    <button
      type="button"
      class="toggle-button"
      :disabled="disabled"
      @click="toggle"
    >
      <span class="toggle-track">
        <span class="toggle-thumb">
          <slot name="thumb">
            <!-- 默认thumb或自定义图标 -->
          </slot>
        </span>
      </span>
      <span v-if="$slots.default || label" class="toggle-label">
        <slot>{{ label }}</slot>
      </span>
    </button>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  modelValue: boolean
  label?: string
  disabled?: boolean
}>()

const emit = defineEmits<{
  'update:modelValue': [value: boolean]
}>()

const toggle = () => {
  emit('update:modelValue', !props.modelValue)
}
</script>

<style scoped lang="scss">
.base-toggle {
  // SCSS样式 - 逻辑与样式分离
}
</style>