<!-- src/components/layout/AppHeader.vue -->
<template>
  <header class="app-header">
    <div class="header-left">
      <button @click="toggleSidebar" class="sidebar-toggle">
        ☰
      </button>
      <h1 class="app-title">Backend Control Panel</h1>
    </div>
    <nav class="main-nav">
      <router-link 
        v-for="route in routes" 
        :key="route.name"
        :to="route.path" 
        class="nav-link"
      >
        {{ route.name }}
      </router-link>
    </nav>
  </header>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const routes = [
  { name: 'Scriptor', path: '/scriptor' },
  { name: 'Song Recognition', path: '/song-recognition' },
  { name: 'UI Recognition', path: '/ui-recognition' },
  { name: 'Fetch', path: '/fetch' },
  { name: 'Workflow', path: '/workflow' }
]

const toggleSidebar = () => {
  appStore.toggleSidebar()
}
</script>

<style scoped lang="scss">
.app-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
  height: 60px;
  background-color: #f5f5f5;
  border-bottom: 1px solid #ddd;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 15px;
}

.sidebar-toggle {
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  padding: 5px;
}

.app-title {
  margin: 0;
  font-size: 18px;
}

.main-nav {
  display: flex;
  gap: 15px;
}

.nav-link {
  text-decoration: none;
  color: #333;
  padding: 5px 10px;
  border-radius: 4px;
  
  &.router-link-active {
    background-color: #3498db;
    color: white;
  }
}
</style>