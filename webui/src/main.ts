import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import pinia from './stores'

// 导入Mock服务（开发环境）
if (import.meta.env.DEV) {
  import('@/services/mock')
}

// 全局样式
import '@/styles/main.scss'

const app = createApp(App)

app.use(pinia)
app.use(router)

app.mount('#app')