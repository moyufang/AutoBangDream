import time
from utils.controller import LowLatencyController
from utils.ADB import ADB
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
