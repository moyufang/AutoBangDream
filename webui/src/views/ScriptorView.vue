<!-- src/views/ScriptorView.vue -->
<template>
  <div class="scriptor-view">
    <h1>Scriptor Module</h1>
    <BaseButton @click="()=>taskAPI.start('scriptor', 'cycle')">Start</BaseButton>
  </div>
</template>

<script setup lang="ts">
import BaseButton from '@/components/base/BaseButton.vue';

import { useAppStore } from '@/stores/app';
import { useScriptorStore } from '@/stores/scriptor';
import { taskAPI, scriptorAPI } from '@/services/api';

const appStore = useAppStore();
const scriptorStore = useScriptorStore();

scriptorStore.updateMultipleConfig(scriptorAPI.getConfig());
let cfg = scriptorStore.config;



</script>

<style scoped lang="scss">
.scriptor-view {
  padding: 20px;
}
</style>