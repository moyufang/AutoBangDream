// src/utils/helpers.ts

/**
 * 防抖函数，适合实时搜索、输入验证
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: number | null = null
  return (...args: Parameters<T>) => {
    if (timeout !== null) {
      clearTimeout(timeout)
    }
    timeout = setTimeout(() => {
      func(...args)
    }, wait)
  }
}

/**
 * 节流函数，适合滚动加载、鼠标移动、窗口调整
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle = false
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args)
      inThrottle = true
      setTimeout(() => {
        inThrottle = false
      }, limit)
    }
  }
}

/**
 * 深拷贝函数
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj
  }
  
  if (obj instanceof Date) {
    return new Date(obj.getTime()) as unknown as T
  }
  
  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as unknown as T
  }
  
  if (typeof obj === 'object') {
    const cloned = {} as T
    for (const key in obj) {
      if (obj.hasOwnProperty(key)) {
        cloned[key] = deepClone(obj[key])
      }
    }
    return cloned
  }
  
  return obj
}

/**
 * 格式化文件大小
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

/**
 * 生成随机ID
 */
export function generateId(length: number = 8): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

/**
 * 验证数字范围
 */
export function validateNumberRange(
  value: number, 
  min: number, 
  max: number
): boolean {
  return value >= min && value <= max
}

/**
 * 验证端口号
 */
export function validatePort(port: number): boolean {
  return validateNumberRange(port, 1024, 65535)
}

/**
 * 颜色转换：HSV 转 RGB
 */
export function hsvToRgb(h: number, s: number, v: number): { r: number; g: number; b: number } {
  let r: number, g: number, b: number
  
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break
    case 1: r = q; g = v; b = p; break
    case 2: r = p; g = v; b = t; break
    case 3: r = p; g = q; b = v; break
    case 4: r = t; g = p; b = v; break
    case 5: r = v; g = p; b = q; break
    default: r = 0; g = 0; b = 0
  }
  
  return {
    r: Math.round(r * 255),
    g: Math.round(g * 255),
    b: Math.round(b * 255)
  }
}

/**
 * 获取像素颜色信息
 */
export function getPixelColorInfo(
  x: number, 
  y: number, 
  imageData: ImageData
): { h: number; s: number; v: number; rgb: string } | null {
  if (x < 0 || x >= imageData.width || y < 0 || y >= imageData.height) {
    return null
  }
  
  const index = (y * imageData.width + x) * 4
  const r = imageData.data[index]
  const g = imageData.data[index + 1]
  const b = imageData.data[index + 2]
  
  // RGB 转 HSV
  const max = Math.max(r!, g!, b!) / 255
  const min = Math.min(r!, g!, b!) / 255
  const delta = max - min
  
  let h = 0
  if (delta !== 0) {
    if (max === r! / 255) {
      h = ((g! - b!) / delta) % 6
    } else if (max === g! / 255) {
      h = (b! - r!) / delta + 2
    } else {
      h = (r! - g!) / delta + 4
    }
    h = (h * 60 + 360) % 360
  }
  
  const s = max === 0 ? 0 : delta / max
  const v = max
  
  return {
    h: Math.round(h),
    s: Math.round(s * 100),
    v: Math.round(v * 100),
    rgb: `rgb(${r}, ${g}, ${b})`
  }
}

/**
 * 下载数据为文件
 */
export function downloadData(data: any, filename: string, type: string = 'application/json'): void {
  const blob = new Blob([data], { type })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

/**
 * 读取文件内容
 */
export function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = (e) => resolve(e.target?.result as string)
    reader.onerror = (e) => reject(e)
    reader.readAsText(file)
  })
}

/**
 * 睡眠函数
 */
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}