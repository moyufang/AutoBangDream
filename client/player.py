import time
from utils.LowLatencyController import LowLatencyController
from utils.ADB import ADB
from utils.WinGrabber import *
from play.note_extractor import HealthExtractor, NoteExtractor
from play.predict import predict
from configuration import *

SCALE = 2

full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None) 
track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], [TRACK_B_X1, TRACK_T_Y, TRACK_B_X2, TRACK_B_Y])
extractor     = NoteExtractor(track_grabber, True)
health_extrator = HealthExtractor(full_grabber)

mode = 'adb'
if mode == 'tcp':
  clr = clr = LowLatencyController(
    adb_path="adb",
    device="-s 127.0.0.1:7555",
    local_port=12345
  )
  clr.start_bangcheater()

  if not clr.connect_with_retry(): print("Failed to connect after retries"); exit(1)
    
  def syn_time(): clr.socket.sendall(f"t {int(time.time()*1000000+0.5)}".encode('utf-8'))
  def send_cmd(cmd): clr.socket.sendall(cmd)
  
elif mode == 'adb':
  adb = ADB()
  def syn_time(): adb.write(f"t {int(time.time()*1000000+0.5)}\n")
  def send_cmd(cmd): adb.write(cmd+'\n')
  
else: print("Unknown mode"); exit(1)

class Print:
  def log_status(msg:str):
    print("Current status:", msg)
  
def start_playing():
  
  correction_time = 1100000
  
  print("Start playing:")
  Print.log_status("detecting 'is_playing'.")
  while not health_extrator.get_is_playing(): pass
  Print.log_status("Start tracing first note.")
  
  start_time = time.time()
  send_cmd(f't {int(start_time*1000000+0.5)} {correction_time}')
  predict_time = 0.0
  predict_count = 0
  while True:
    img = extractor.grab()
    tim = extractor.get_grab_time() - start_time
    derive_img, t_s, is_edge = extractor.extract(img, NoteExtractor.DerivePara.NONE)
    
    predict_time += tim+predict(t_s[0])
    predict_count += 1
    
    if is_edge:break
  predict_time = predict_time/predict_count
  predict_time = int(predict_time*1000000+0.5)
  print("t_s:", t_s, "is_edge:", is_edge, "pred_time:", predict_time)
  send_cmd(f"s {predict_time}")
  
start_playing()
time.sleep(180)
  
    
    