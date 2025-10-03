import time
from utils.controller import LowLatencyController
from utils.ADB import ADB
from utils.WinGrabber import *
from utils.log import LogE, LogD, LogI, LogS
from play.note_extractor import HealthExtractor, NoteExtractor
from play.predict import predict
from configuration import *

class Player:
  def __init__(self, communication_mode:str):
    SCALE = 2

    self.full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None) 
    self.track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], [TRACK_B_X1, TRACK_T_Y, TRACK_B_X2, TRACK_B_Y])
    self.extractor     = NoteExtractor(self.track_grabber, True)
    self.health_extrator = HealthExtractor(self.full_grabber)
    
    self.communication_mode = communication_mode
    if self.communication_mode == 'adb':
      adb = ADB()
      self.send_cmd = lambda cmd: adb.write(cmd+'\n')
    elif self.communication_mode == 'tcp':
      clr = clr = LowLatencyController(
        adb_path="adb",
        device="-s 127.0.0.1:7555",
        local_port=12345
      )
      clr.start_bangcheater()

      if not clr.connect_with_retry(): print("Failed to connect after retries"); exit(1)
    
      self.send_cmd = lambda cmd: clr.socket.sendall(cmd)
    else:
      LogE("Unknown communication mode.")
      exit(1)
      
  def start_playing(self, song_duration):
    dilation_time   = 1002000
    correction_time = 1100000
    
    LogS("Start playing:")
    LogS("detecting 'is_playing'.")
    while not self.health_extrator.get_is_playing(): pass
    LogS("Start tracing first note.")
    
    start_time = time.time()
    self.send_cmd(f't {int(start_time*1000000+0.5)} {correction_time} {dilation_time}')
    predict_time = 0.0
    predict_count = 0
    while True:
      img = self.extractor.grab()
      tim = self.extractor.get_grab_time() - start_time
      derive_img, t_s, is_edge = self.extractor.extract(img, NoteExtractor.DerivePara.NONE)
      
      predict_time += tim+predict(t_s[0])
      predict_count += 1
      
      if is_edge:break
    predict_time = predict_time/predict_count
    predict_time = int(predict_time*1000000+0.5)
    self.send_cmd(f"s {predict_time}")
    
    LogD("t_s:", t_s, "is_edge:", is_edge, "pred_time:", predict_time)
    time.sleep(song_duration+5)

player = Player('tcp')

