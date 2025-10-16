<!-- src/views/ControllerView.vue -->
<template>
  <div class="controller-view">
    <h1>Controller Module</h1>
    <p>This is the Controller module view.</p>
    <h2>{{ appStore.currentRunningTask }}</h2>
  </div>
</template>

<script setup lang="ts">

import { useAppStore } from '@/stores/app';
const appStore = useAppStore();

// 模块逻辑将在后续添加
</script>

<style scoped lang="scss">
.controller-view {
  padding: 20px;
}
</style>