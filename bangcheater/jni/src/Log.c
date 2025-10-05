#ifndef LOG_C
#define LOG_C

#include <stdio.h>
#include <stdarg.h>

// Use macro to ban unnecessary logs
// #define NO_INFO_LOG
// #define NO_ERROR_LOG
// #define NO_DEBUG_LOG
// #define NO_RUNNING_LOG

#ifdef NO_ERROR_LOG
#define LogE (void)
#else
void LogE(const char* format, ...) {
  
  // 添加前缀
  fprintf(stderr, "[ERROR @ bc ]: ");
  
  // 处理可变参数
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}
#endif

#ifdef NO_DEBUG_LOG
#define LogD (void)
#else
void LogD(const char* format, ...) {
  // 添加前缀
  fprintf(stderr, "[DEBUG @ bc ]: ");
  
  // 处理可变参数
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}
#endif

#ifdef NO_INFO_LOG
#define LogI (void)
#else
void LogI(const char* format, ...) {
  // 添加前缀
  fprintf(stderr, "[INFO  @ bc ]: ");
  
  // 处理可变参数
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}
#endif

#ifdef NO_RUNNING_LOG
#define LogR (void)
#else
void LogR(const char* format, ...) {
  // 添加前缀
  fprintf(stderr, "[RUN   @ bc ]: ");
  
  // 处理可变参数
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
}
#endif

#endif