import time
import socket
import numpy as np
from server.controller import LowLatencyController
from server.ADB import ADB
from utils.WinGrabber import *
from utils.log import LogE, LogD, LogI, LogS
from play.note_extractor import HealthExtractor, NoteExtractor
from play.predict import predict
from configuration import *

class Player:
  def __init__(self, communication_mode:str, init_scale=2):
    SCALE = init_scale

    self.track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], [TRACK_B_X1, TRACK_T_Y, TRACK_B_X2, TRACK_B_Y])
    self.full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None) 
    self.extractor     = NoteExtractor(self.track_grabber, True)
    self.health_extrator = HealthExtractor(self.full_grabber)
    
    self.communication_mode = communication_mode
    if self.communication_mode == 'stdio':
      adb = ADB()
      self.send_cmd = lambda cmd: adb.write(cmd+'\n')
    elif self.communication_mode == 'tcp':
      clr = LowLatencyController(
        adb_path="adb",
        device="127.0.0.1:7555",
        local_port=12345
      )
      clr.start_bangcheater()
      clr.connect()
    
      self.send_cmd = lambda cmd: clr.socket.sendall(cmd.encode() if isinstance(cmd, str) else cmd)
    else:
      LogE("Unknown communication mode.")
      exit(1)
  def set_scale(self, scale):
    if scale == self.full_grabber.scale: return
    self.full_grabber.set_window(scale)
    self.track_grabber.set_window(scale)
  def get_scale(self):
    return self.full_grabber.scale
  def set_caliboration_para(self, dilation_time, correction_time):
    self.dilation_time = dilation_time
    self.correction_time = correction_time
    
  def click(self, touch, x, y):
    self.send_cmd(f'd {touch} {x} {y}\n')
    time.sleep(TCP_SEND_GAP)
    self.send_cmd(f'c\n')
    time.sleep(CLICK_PERIOD)
    self.send_cmd(f'u {touch}\n')
    time.sleep(TCP_SEND_GAP)
    self.send_cmd(f'c\n')
    
  def start_playing(self, song_duration):
    
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
    self.send_cmd(f"s {predict_time}")
    
    LogS('playing', f"First note info: t_s:{t_s} is_edge:{is_edge} predict_time:{predict_time}")
    time.sleep(song_duration+5)

class PlayerClient:
  def __init__(self):
    self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  def __del__(self):
    self.client.close()
  def set_scale(self, scale):
    self.client.send(f"s {scale}".encode('utf-8'))
    rp = int(self.client.recv(1024).decode('utf-8'))
    return rp == SERVER_OK
  def get_scale(self):
    self.client.send(f"g".encode('utf-8'))
    rp = int(self.client.recv(1024).decode('utf-8'))
    return rp
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
  def start_playing(self, song_duration:int):
    self.client.send(f"p {song_duration}".encode('utf-8'))
    rp = int(self.client.recv(1024).decode('utf-8'))
    return rp
  def click(self, touch:int, x:int, y:int):
    self.client.send(f"c {touch} {x} {y}".encode('utf-8'))
    rp = int(self.client.recv(1024).decode('utf-8'))
    return rp
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
  client = PlayerClient()
  client.connect()
    
  while True:
    cmd = input("> ")
    client.parse_cmd(cmd)

if __name__ == '__main__':
  player = Player('tcp', init_scale=1)
