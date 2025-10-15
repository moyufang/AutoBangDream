import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'
import { resolve } from 'node:path' // 导入 resolve

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    vueDevTools(),
  ],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src')
    },
  },
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `@import "@/styles/main.scss";`
      }
    }
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:62358',
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: '../client/static',  // 构建到后端目录
    emptyOutDir: true,
  },
})