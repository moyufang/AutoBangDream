import subprocess
import socket
import time
from utils.log import LogE, LogD, LogI, LogS
from configuration import *

class BangcheaterController:
  def __init__(self, adb_path="adb", device=f"127.0.0.1:{MUMU_PORT}", remote_port=12345):
    self.adb_path = adb_path
    self.device = device
    self.remote_port = remote_port
  
  def run_adb_command(self, command_args, check_success=False, timeout=10):
    try:
      result = subprocess.run(
        command_args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check_success  # 如果为True，非零退出码会抛出异常
      )
      return result
    except subprocess.TimeoutExpired as e:
      LogE(f"ADB command timeout: {' '.join(command_args)}")
      raise e
    except subprocess.CalledProcessError as e:
      LogE(f"ADB command failed: {' '.join(command_args)}, error: {e.stderr}")
      raise e
  
  def is_process_running(self, process_name):
    """检查进程是否正在运行"""
    check_cmd = [self.adb_path, "-s", self.device, "shell", "pidof", process_name]
    try:
      result = self.run_adb_command(check_cmd)
      return result.stdout.strip() != ""
    except Exception:
      return False
  
  def adb_connect(self):
    """连接设备"""
    connect_cmd = [self.adb_path, "connect", self.device]
    try:
      result = self.run_adb_command(connect_cmd)
      LogI(f"{result.stdout.strip()}")
      return True
    except Exception as e:
      LogE(f"failed to connect to {self.device}: {e}")
      return False
  
  def remove_forward(self):
    """移除端口转发"""
    remove_cmd = [self.adb_path, "-s", self.device, "forward", "--remove", f"tcp:{self.remote_port}"]
    try:
      result = self.run_adb_command(remove_cmd)
      # 移除转发可能失败（如果转发不存在），这通常是正常的
      LogI("removed old forward")
      return True
    except Exception as e:
      LogW(f"failed to remove forward (may not exist): {e}")
      return False
  
  def establish_forward(self):
    """建立新的端口转发"""
    forward_cmd = [self.adb_path, "-s", self.device, "forward", f"tcp:{self.remote_port}", f"tcp:{self.remote_port}"]
    try:
      result = self.run_adb_command(forward_cmd)
      LogI(f"established new forward: {result.stdout.strip()}")
      return True
    except Exception as e:
      LogE(f"failed to establish forward: {e}")
      return False
  
  def kill_bangcheater(self):
    """杀死进程，返回是否成功"""
    if not self.is_process_running("bangcheater"):
      LogI("'bangcheater' is not running, skip killing")
      return True
    
    kill_cmd = [self.adb_path, "-s", self.device, "shell", "pkill", "-9", "-f", "bangcheater"]
    try:
      result = self.run_adb_command(kill_cmd)
      
      # 等待并确认
      time.sleep(1)
      if not self.is_process_running("bangcheater"):
        LogI("killed 'bangcheater'")
        return True
      else:
        LogE("failed to kill 'bangcheater'")
        return False
    except Exception as e:
      LogE(f"failed to kill 'bangcheater': {e}")
      return False
  
  def start_bangcheater(self, remote_path=REMOTE_BANGCHEATER_PATH, commands_path=REMOTE_COMMANDS_PATH):
    try:
      # 连接和设备准备
      if not self.adb_connect(): return False
      self.remove_forward()
      if not self.establish_forward(): return False
      
      # 清理旧进程
      self.kill_bangcheater()
      
      # 启动新进程
      start_cmd = [self.adb_path, "-s", self.device, "shell", f"{remote_path}", f"{commands_path}", "-t"]
      self.p = subprocess.Popen(start_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
      LogI("started new bangcheater process")
      
      # 等待启动
      timeout = 3
      start_time = time.time()
      while time.time() - start_time < timeout:
        if self.p.poll() is None:  # 进程仍在运行
          # 额外检查进程是否真的在设备上运行
          if self.is_process_running("bangcheater"):
            LogI("'bangcheater' started")
            return True
        time.sleep(0.5)
      
      LogE("'bangcheater' failed to start or died quickly")
      return False
    except Exception as e:
      LogE(f"failed to start 'bangcheater': {e}")
      return False
  
  def __del__(self):
    """析构函数"""
    try:
      if hasattr(self, 'p') and self.p and self.p.poll() is None:
        self.p.terminate()
      if hasattr(self, 'is_process_running') and self.is_process_running("bangcheater"):
        self.kill_bangcheater()
    except Exception:
        pass  # 避免析构时抛出异常

class Controller:
  def __init__(self, remote_port=BANGCHEATER_PORT):
    self.remote_port = remote_port
    self.socket = None
  
  def _connect(self, max_retries=3):
    for attempt in range(max_retries):
      try:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置socket选项以减少延迟
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用Nagle算法
        self.socket.settimeout(5.0)
        self.socket.connect(('127.0.0.1', self.remote_port))
        return True
      except Exception as e:
        LogE(f"Connection attempt {attempt + 1} failed: {e}")
        if self.socket:
            self.socket.close()
            self.socket = None
        time.sleep(0.1 * (attempt + 1))  # 递增重试延迟
    return False
  
  def connect(self):
    while True:
      try:
        if not self._connect(): LogE("Failed to connect after retries"); exit(1)
        self.send_cmd("p")
        rp = self.recv()
        if rp == CONTROLLER_READY_HASH: break
      except Exception as e:
        LogI(f"Test connection failed: {e}")
      time.sleep(1.0)
    LogI("Connected to 'bangcheater'")
  
  def send_cmd(self, command):
    if not self.socket: return False
    if not command.endswith('\n'): command += '\n'
    # 发送命令
    self.socket.sendall(command.encode('utf-8'))
    return True
    
  def recv(self, recv_timeout:int=RECV_TIMEOUT):
    # 接收响应（带超时）
    try:
      self.socket.settimeout(recv_timeout)
      response = self.socket.recv(1024).decode('utf-8').strip()
    except socket.timeout:
      LogE(f"Receive timeout ({recv_timeout}s)")
      raise socket.timeout
    return response
  
  def __del__(self): self.cleanup()
  
  def cleanup(self):
    if self.socket:
      try: self.send_cmd("e")
      except: pass
      self.socket.close()

def click(x, y):
  global clr
  clr.send_cmd('d 0 %d %d\n'%(x,y))
  time.sleep(TCP_SEND_GAP)
  clr.send_cmd('c\n')
  time.sleep(0.05)
  clr.send_cmd('u\n')
  time.sleep(TCP_SEND_GAP)
  clr.send_cmd('c\n')

if __name__ == "__main__":
  bcc = BangcheaterController(
    "adb",
    f"127.0.0.1:{MUMU_PORT}",
    BANGCHEATER_PORT
  )
  clr = Controller(
    remote_port=bcc.remote_port
  )
  try:
    bcc.start_bangcheater()
    # 启动bangcheater
    time.sleep(0.0)
    # 连接
    clr.connect()
    
    # 通信
    clr.send_cmd("f")
    
    while True:
      cmd = input("type cmd:\n")
      x, y = cmd.split(' ')
      click(int(x), int(y))
      
  finally:
    clr.cleanup()