import time
import socket
import numpy as np
from abc import ABC, abstractmethod
from server.controller import Controller
from server.ADB import ADB
from utils.WinGrabber import *
from utils.log import LogE, LogD, LogI, LogS
from play.note_extractor import HealthExtractor, NoteExtractor
from play.predict import predict
from configuration import *

class PlayerInterface(ABC):
  @abstractmethod
  def click(self, touch:int, x:int, y:int)->None: pass
  @abstractmethod
  def set_scale(self, scale:int)->None: pass
  @abstractmethod
  def get_scale(self)->int: pass
  @abstractmethod
  def set_caliboration_parameters(self, dilation_time:int, correction_time:int)->None: pass
  @abstractmethod
  def start_playing(self, is_caliboration:bool)->int: pass
  @abstractmethod
  def stop_playing(self): pass
  @abstractmethod
  def wait_finish(self, timeout:int)->bool: pass

class WinPlayer(PlayerInterface):
  def __init__(self, communication_mode:str='tcp', init_scale=1, remote_port=BANGCHEATER_PORT):
    SCALE = init_scale

    self.track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], [TRACK_B_X1, TRACK_T_Y, TRACK_B_X2, TRACK_B_Y])
    self.full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None) 
    self.extractor     = NoteExtractor(self.track_grabber, True)
    self.health_extrator = HealthExtractor(self.full_grabber)
    
    self.communication_mode = communication_mode
    if self.communication_mode == 'stdio':
      self.adb = ADB()
      self.send_cmd = lambda cmd: self.adb.write(cmd+'\n')
      self.recv = lambda timeout: self.adb.read()
    elif self.communication_mode == 'tcp':
      self.clr = Controller(remote_port)
      self.clr.connect()
      self.send_cmd = lambda cmd: self.clr.socket.sendall(cmd.encode() if isinstance(cmd, str) else cmd)
      self.recv = lambda *arg: self.clr.recv(*arg)
    else:
      LogE("Unknown communication mode.")
      exit(1)
  def click(self, touch, x, y):
    self.send_cmd(f'd {touch} {x} {y}\n')
    time.sleep(TCP_SEND_GAP)
    self.send_cmd(f'c\n')
    time.sleep(CLICK_PERIOD)
    self.send_cmd(f'u {touch}\n')
    time.sleep(TCP_SEND_GAP)
    self.send_cmd(f'c\n')
  def set_scale(self, scale):
    if scale == self.full_grabber.scale: return
    self.full_grabber.set_window(scale)
    self.track_grabber.set_window(scale)
  def get_scale(self):
    return self.full_grabber.scale
  def set_caliboration_parameters(self, dilation_time, correction_time):
    self.dilation_time = dilation_time
    self.correction_time = correction_time
  def grab_full_img(self):
    return self.full_grabber.grab()[:,:,:3]
  def start_playing(self, is_caliboration:bool = False):
    
    LogS('playing', "Start detecting 'is_playing'")
    while not self.health_extrator.get_is_playing(): pass
    
    start_time = time.time()
    self.send_cmd(f't {int(start_time*1000000+0.5)} {self.correction_time} {self.dilation_time}')
    predict_time = 0.0
    predict_count = 0
    while True:
      img = self.extractor.grab()
      tim = self.extractor.get_grab_time() - start_time
      derive_img, t_s, is_edge = self.extractor.extract(img, NoteExtractor.DerivePara.NONE)
      if tim > 60:
        LogE("failed to start playing")
        return -1
      # if t_s[0] < 0.0:
      #   print(t_s)
      #   print("save false into opps.")
      #   cv.imwrite('./play/opps/img.png', cv.cvtColor(img, cv.COLOR_HSV2BGR))
      if t_s[0] < 0.0:
        predict_time = 0.0
        predict_count = 0
        continue
      
      predict_time += tim+predict(t_s[0])
      predict_count += 1
      
      if is_edge:break
    predict_time = predict_time/predict_count
    predict_time = int(predict_time*1000000+0.5)
    
    if is_caliboration:
      self.send_cmd(f"C {predict_time}")
      diff_time = self.recv()
      return diff_time
    else:
      self.send_cmd(f"s {predict_time}")
      # LogS('playing', f"First note info: t_s:{t_s} is_edge:{is_edge} predict_time:{predict_time}")
      return -1
  def stop_playing(self):
    self.clr.send_cmd("k")
  def wait_finish(self, timeout:int=300):
    try:
      return str(self.recv(timeout)) == "F"
    except socket.timeout:
      return False

class FakeServer:
  def __init__(self):
    self.player = WinPlayer('tcp', init_scale=1)
    self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 设置地址重用
    self.server_socket.bind(('localhost', SERVER_PORT))
  def __del__(self):
    self.server_socket.close()
  def launch(self):
    self.server_socket.listen(1)
    try:
      while True:
        LogI("Server wait connection.")
        client_socket, client_address = self.server_socket.accept() # 阻塞直到有客户端连接
        self.handle_client_connection(client_socket, client_address)
    except KeyboardInterrupt:
      LogI("Server close.")
    finally:
      self.server_socket.close()
  def handle_client_connection(self, client_socket, client_address):
    print(f"Server connect with {client_address}")
    try:
      while True:
        # 接收客户端消息
        data = client_socket.recv(1024).decode('utf-8')
        if not data:
          LogI(f"Server disconnect with {client_address}")
          break
        rp = self.parse_data(data)
        client_socket.send(rp if rp is bytes else rp.encode('utf-8'))
    except ConnectionResetError:
      LogI(f"Client RunTimeError: {client_address}\n")
    except Exception as e:
      LogI(f"Server Error: {e}")
    finally:
      client_socket.close()
      LogI(f"Client close: {client_address}\n")
  def parse_data(self, data:str):
    str_list = data.split(' ')
    if str_list[0] == 's':
      new_scale = int(str_list[1])
      self.player.set_scale(new_scale)
      rp = str(ServerResponse.OK)
    elif str_list[0] == 'g':
      rp = str(self.player.get_scale())
    elif str_list[0] == 't':
      self.player.set_caliboration_parameters(int(str_list[1]), int(str_list[2]))
      rp = str(ServerResponse.OK)
    elif str_list[0] == 'i':
      img = self.player.grab_full_img()
      rp = img.tobytes()
    elif str_list[0] == 'p':
      is_caliboration = bool(str_list[1])
      self.player.start_playing(is_caliboration)
      rp = str(ServerResponse.OK)
    elif str_list[0] == 'c':
      touch, x, y = int(str_list[1]), int(str_list[2]), int(str_list[3])
      self.player.click(touch, x, y)
      rp = str(ServerResponse.OK)
    elif str_list[0] == 'k':
      self.player.stop_playing()
      rp = str(ServerResponse.OK)
    elif str_list[0] == 'w':
      timeout = int(str_list[1])
      if self.player.wait_finish(timeout): rp = str(ServerResponse.OK)
      else: rp = str(ServerResponse.TIMEOUT_FAILED)
    else:
      rp = str(ServerResponse.UNKNOWN)
    return rp

class ClientPlayer:
  def __init__(self):
    self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  def __del__(self):
    self.client.close()
  def get_int_rp(self):
    return int(self.client.recv(1024).decode('utf-8'))
  def click(self, touch:int, x:int, y:int):
    self.client.send(f"c {touch} {x} {y}".encode('utf-8'))
    return self.get_int_rp()
  def set_scale(self, scale):
    self.client.send(f"s {scale}".encode('utf-8'))
    return self.get_int_rp()
  def get_scale(self):
    self.client.send(f"g".encode('utf-8'))
    return self.get_int_rp()
  def set_caliboration_parameters(self, dilation_time, correction_time):
    self.client.send(f"t {dilation_time} {correction_time}")
    return self.get_int_rp()
  def grab_full_img(self):
    self.client.send(f"i".encode('utf-8'))
    
    rp, expected_size = b'', 1280 * 720 * 3 
    while len(rp) < expected_size:
      remaining = expected_size - len(rp)
      chunk = self.client.recv(min(4096, remaining))
      if not chunk: break
      rp += chunk
    if len(rp) == expected_size:
      img_arr = np.frombuffer(rp, dtype=np.uint8)
      img = img_arr.reshape(1280, 720, 3)
    else:
      img = None
      raise ValueError(f"receive partial data: expect {expected_size}bytes but actual {len(rp)} bytes")
    return img
  def start_playing(self, is_caliboration:bool = False):
    self.client.send(f"p {is_caliboration}".encode('utf-8'))
    return self.get_int_rp()
  def stop_playing(self):
    self.client.send(f'k')
    return self.get_int_rp()
  def wait_finish(self, timeout:int):
    self.client.send(f"w {timeout}".encode('utf-8'))
    return self.get_int_rp()
  # Used to debug
  def parse_cmd(self, cmd):
    str_list = cmd.split(' ')
    if str_list[0] == 's':
      new_scale = int(str_list[1])
      self.set_scale(new_scale)
    elif str_list[0] == 'g':
      scale = self.get_scale()
      LogI("Got scale:", scale)
    elif str_list[0] == 'i':
      img = self.grab_full_img()
      return img
    elif str_list[0] == 'p':
      song_duration = 140
      self.start_playing(song_duration)
    elif str_list[0] == 'c':
      touch, x, y = int(str_list[1]), int(str_list[2]), int(str_list[3])
      self.click(touch, x, y)
    else:
      raise ValueError("Unknown cmd.")
  def connect(self):
    try:
      self.client.connect(('localhost', SERVER_PORT))
      LogI("Client connect to Server")
    except Exception as e:
      LogE(f"Client connect failed: {e}")
      self.client.close()
  
if __name__ == "__main__":
  client = ClientPlayer()
  client.connect()
    
  while True:
    cmd = input("> ")
    client.parse_cmd(cmd)

# if __name__ == '__main__':
#   player = WinPlayer('tcp', init_scale=1)
