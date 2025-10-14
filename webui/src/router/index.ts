import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/scriptor'
    },
    {
      path: '/scriptor',
      name: 'Scriptor',
      component: () => import('@/views/ScriptorView.vue')
    },
    {
      path: '/song-recognition',
      name: 'SongRecognition',
      component: () => import('@/views/SongRecognitionView.vue')
    },
    {
      path: '/ui-recognition',
      name: 'UIRecognition', 
      component: () => import('@/views/UIRecognitionView.vue')
    },
    {
      path: '/fetch',
      name: 'Fetch',
      component: () => import('@/views/FetchView.vue')
    },
    {
      path: '/workflow',
      name: 'Workflow',
      component: () => import('@/views/WorkflowView.vue')
    }
  ],
})

export default router
