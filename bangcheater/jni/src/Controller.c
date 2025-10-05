#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>  // 用于 TCP_NODELAY
#include <arpa/inet.h>
#include <errno.h>
#include <time.h>
#include "Log.c"

#define CONTROLLER_EXIT -1
#define CONTROLLER_QUIT_CONNECTION -2
#define CONTROLLER_READY 1
#define CONTROLLER_READY_HASH "BANGCHEATERCONTROLLERREADY"

struct LowLatencyController{
  int port;
  int buffer_size;
  int server_fd;
  int client_fd;
  int opt;
  socklen_t addrlen;
  struct sockaddr_in address;

  int (*process)(const char*, char*);
};

// 设置socket为非阻塞
void set_socket_nonblocking(int sockfd) {
  int flags = fcntl(sockfd, F_GETFL, 0);
  fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
}

// 设置TCP无延迟
void set_tcp_nodelay(int sockfd) {
  int flag = 1;
  setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(flag));
}

void init_controller(struct LowLatencyController *clr, int port, int buffer_size, int (*process)(const char*, char*)) {
  clr->port = port;
  clr->buffer_size = buffer_size;
  clr->opt = 1;
  clr->addrlen = sizeof(clr->address);
  clr->process = process;
}

int launch_controller(struct LowLatencyController *clr){
  // 创建 socket
  if ((clr->server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    LogE("TCP socket failed");
    exit(EXIT_FAILURE);
  }
  
  // 设置 socket 选项 - 重用地址
  if (setsockopt(clr->server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &(clr->opt), sizeof(clr->opt))) {
      LogE("TCP setsockopt");
      close(clr->server_fd);
      exit(EXIT_FAILURE);
  }
  
  clr->address.sin_family = AF_INET;
  clr->address.sin_addr.s_addr = INADDR_ANY;
  clr->address.sin_port = htons(clr->port);
  
  // 绑定端口
  if (bind(clr->server_fd, (struct sockaddr *)&clr->address, clr->addrlen) < 0) {
      LogE("TCP bind failed");
      close(clr->server_fd);
      exit(EXIT_FAILURE);
  }
  
  // 监听连接
  if (listen(clr->server_fd, 5) < 0) {
      LogE("TCP listen failed");
      close(clr->server_fd);
      exit(EXIT_FAILURE);
  }
  
  // 设置接收超时
  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  
  fd_set readfds;
  char *buffer = malloc(clr->port);
  char *rp_buffer = malloc(clr->port);
  
  LogI("TCP listening on port %d\n", clr->port);

  int is_running = 1;
  while (is_running) {
    FD_ZERO(&readfds);
    FD_SET(clr->server_fd, &readfds);
    
    // 使用select等待连接，带超时
    int activity = select(clr->server_fd + 1, &readfds, NULL, NULL, &tv);
    
    if (activity < 0 && errno != EINTR) {
        LogE("select error");
        break;
    }
    
    if (activity > 0 && FD_ISSET(clr->server_fd, &readfds)) {
      // 接受新连接
      if ((clr->client_fd = accept(clr->server_fd, (struct sockaddr *)&(clr->address), (socklen_t*)&(clr->addrlen))) < 0) {
        LogI("accept");
        continue;
      }
      
      // 优化socket设置
      set_tcp_nodelay(clr->client_fd);  // 禁用Nagle算法，减少延迟
      setsockopt(clr->client_fd, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv));
      
      LogI("TCP connected\n");
      
      // 处理客户端通信
      int is_waiting = 0;
      while (1) {
        memset(buffer, 0, clr->buffer_size);
        
        int bytes_read = recv(clr->client_fd, buffer, clr->buffer_size - 1, 0);
        if (bytes_read <= 0) {
          if (bytes_read == 0) {
              LogI("TCP disconnected(receving noting).\n");
              break;
          } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
              // 没有数据可读，可以继续等待
              // 注意：如果是非阻塞模式，那么可以继续循环，但可能需要进行适当的休眠避免忙等待
              // 如果是阻塞模式，则不会出现EAGAIN。
              if (!is_waiting){
                LogD("TCP wait.\n",bytes_read, buffer);
                is_waiting = 1;
              }
              continue;
          } else {
              LogE("TCP recv error: %s\n", strerror(errno));
              break;
          }
        }
        //LogD("TCP recv: bytes:%d content:%s\n",bytes_read, buffer);
        is_waiting = 0;
        
        // 处理命令（去掉换行符）
        if (buffer[bytes_read - 1] == '\n') {
            buffer[bytes_read - 1] = '\0';
        }
        
        int rp_code = clr->process(buffer, rp_buffer);
        if (rp_code == CONTROLLER_READY){
          LogI("TCP confirm been ready.");
          strcpy(rp_buffer, CONTROLLER_READY_HASH);
          send(clr->client_fd, rp_buffer, strlen(rp_buffer), 0);
          rp_buffer[0] = 0;
        }
        else if (rp_code == CONTROLLER_EXIT){
          is_running = 0;
          break;
        }
        else if (rp_code == CONTROLLER_QUIT_CONNECTION) break;
        
        // 发送响应
        if (rp_buffer[0] && send(clr->client_fd, rp_buffer, strlen(rp_buffer), 0) < 0) {
          LogE("TCP send failed");
          break;
        }
      }
      
      close(clr->client_fd);
      LogI("Client connection closed\n");
    }
    
    // 这里可以添加其他后台任务
  }

  LogI("TCP close");
  
  close(clr->server_fd);
  free(buffer);
  free(rp_buffer);
  return CONTROLLER_EXIT;
}

