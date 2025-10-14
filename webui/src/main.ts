import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import pinia from './stores'

// 导入Mock服务（开发环境）
if (import.meta.env.DEV) {
  import('@/services/mock')
}

const app = createApp(App).use(pinia).use(router).mount('#app')