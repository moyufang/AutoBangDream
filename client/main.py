from UI_recognition.predict import UIRecognition
from song_recognition.predict import SongRecognition
from utils.WinGrabber import MumuGrabber
from utils.ADB import push_file
from utils.json_refiner import refine
from configuration import *
from client.player import Player
from client.script import Script
from client.sheet2commands import NoteSkewer, sheet2commands
from utils.log import LogE, LogD, LogI, LogS
import time
import cv2 as cv

#============ User Configuration ============#

custom_performance = CustomPerformance()

user_config = UserConfig()
user_config.set_config(
  Mode.Free,
  Event.Challenge,
  Choose.Loop,
  Level.Expert,
  Performance.AllPerfect,
  custom_performance,
  None,
  'new_bee'
)
uc = user_config

dilation_time   = 1002000
correction_time = 1100000

#============ main ============#

ui_recognition = UIRecognition()
song_recognition = SongRecognition(user_config)
player = Player('tcp')
script = Script(player, user_config)

player.set_caliboration_para(dilation_time, correction_time)
grabber = player.full_grabber

def to_torch_type(img):
  # img 应为 BGR 色彩的 (H,W,C) 维的图片，也即 opencv 导入图片的默认格式
  # 然后 img 将被转换成 RGB 色彩的 (C,H,W) 供图像预测
  return np.transpose(cv.cvtColor(img, cv.COLOR_BGR2RGB), (2, 0, 1))

log_imgs_path = './log_imgs/'
song_duration = None
commands_sheet_path = './client/commands.sheet'
commands_json_path = "./client/commands.json"
is_save_commands_json = True
last_state = None
same_state_count = 1
MAX_SAME_STATE_COUNT = 100

#============ Cycle ============#

while True:
  if grabber.scale == 2:
    grabber.set_window(1)
  
  img = grabber.grab()[:,:,:3]
  th_img = cv.resize(img, (STD_WINDOW_WIDTH//8,STD_WINDOW_HEIGHT//8,3),interpolation=cv.INTER_AREA)
  th_img = to_torch_type(th_img)
  
  label, state = ui_recognition.get_state(th_img)
  LogI(f"Recognition state:{state} label:{label}")
  
  if state == last_state:
    same_state_count += 1
    if same_state_count > MAX_SAME_STATE_COUNT:
      false_img_path = log_imgs_path+f'false_{state}.png'
      cv.imwrite(false_img_path, img)
      LogE(f"The state \"{state}\" occur to much, saving img to \"{false_img_path}\"")
      break
  else: last_state = state; same_state_count = 1; 
  
  if state == 'ready':
    song_id, song_name = song_recognition.get_song(th_img)
    LogS('ready', f'Recognition song: id:{song_id} name:{song_name}')
    
    sheet_path = f'./sheet/sheets/{song_id}_{user_config.level}.bestdori'
    commands, song_duration = sheet2commands(sheet_path, commands_sheet_path, user_config.note_skewer)
    LogS('ready', f'song_duration:{song_duration}')
    LogS('ready', f'Try to upload "{sheet_path}"')
    push_file(commands_sheet_path)
    
    if is_save_commands_json:
      with open(commands_json_path, "w", encoding="utf-8") as file: json.dump(commands, file)
      refine(commands_json_path)
      LogS('ready', f'Save commands_json to "{commands_json_path}"')
    
    script.act(state)
  elif state == 'playing':
    player.full_grabber.set_window(2)
    player.start_playing(song_duration)
  else:
    script.act(state)
  
  time.sleep(CYCLE_GAP)
  

  
  
  



