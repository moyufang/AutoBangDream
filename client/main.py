from configuration import *
from utils.WinGrabber import MumuGrabber
from utils.ADB import push_file
from utils.json_refiner import refine
from utils.log import LogE, LogD, LogI, LogS
from UI_recognition.predict import UIRecognition
from song_recognition.predict_TitleNet import SongRecognition
from client.player import Player
from client.script import Script
from client.sheet2commands import sheet2commands
import time
import cv2 as cv
import keyboard

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
  'newbee'
)
uc = user_config

dilation_time   = 1002000
correction_time = 1100000
play_one_song_id = 306

#============ main ============#

ui_recognition = UIRecognition()
song_recognition = SongRecognition(
  ckpt_path='./song_recognition/ckpt_triplet.pth',
  img_dir='./song_recognition/title_imgs',
  feature_json_path='./song_recognition/feature_vectors.json',
  is_load_library=True,
  user_config=user_config
)
player = Player('tcp')
script = Script(player, user_config)

player.set_caliboration_para(dilation_time, correction_time)
grabber = player.full_grabber

def to_torch_type(img):
  # img 应为 BGR 色彩的 (H,W,C) 维的图片，也即 opencv 导入图片的默认格式
  # 然后 img 将被转换成 RGB 色彩的 (C,H,W) 供图像预测
  return np.transpose(cv.cvtColor(img, cv.COLOR_BGR2RGB), (2, 0, 1))

log_imgs_path = './UI_recognition/log_imgs/'
song_duration = None
sheets_path = './sheet/sheets/'
commands_sheet_path = './client/commands.sheet'
commands_json_path = "./client/commands.json"
is_save_commands_json = True
is_checking_3d = True
is_play_one_song = False
is_restart_play = True
is_allow_save = True
last_state = None
same_state_count = 1
MAX_SAME_STATE_COUNT = 100
protected_state = ['join_wait', 'ready_done']

#============ calibration & play one song ============#

def create_and_push_commands(song_id:int, user_config:UserConfig):
  sheet_path = sheets_path+f'{song_id}_{user_config.level}.bestdori'
  commands, song_duration = sheet2commands(sheet_path, commands_sheet_path, user_config.note_skewer)
  LogS('ready', f'song_duration:{song_duration}')
  LogS('ready', f'Try to upload "{sheet_path}"')
  push_file(commands_sheet_path)
  
  if is_save_commands_json:
    with open(commands_json_path, "w", encoding="utf-8") as file: json.dump(commands, file)
    refine(commands_json_path)
    LogS('ready', f'Save commands_json to "{commands_json_path}"')
  return song_duration
    
if is_play_one_song:
  if is_restart_play:
    player.click(0, 0, 0)
    time.sleep(CLICK_GAP)
    player.click(0, 0, 0)
    time.sleep(CLICK_GAP)
    player.click(0, 0, 0)
    time.sleep(CLICK_GAP)
    pass
  
  
  song_duration = create_and_push_commands(play_one_song_id, user_config)
  
  player.full_grabber.set_window(2)
  player.start_playing(song_duration)
  
  exit(0)

#============ Cycle ============#
frame_id = 0; is_repeat = False
LogI("Cycle start ...")
while True:
  if grabber.scale == 2:
    grabber.set_window(1)
  
  img = grabber.grab()[:,:,:3]
  f_img = cv.resize(img, (STD_WINDOW_WIDTH//8,STD_WINDOW_HEIGHT//8),interpolation=cv.INTER_AREA)
  th_img = to_torch_type(f_img)
  
  label, state = ui_recognition.get_state(th_img)
  if state != last_state: LogI(f"{'\n'if is_repeat else ''}Recognition state:{state} label:{label}"); is_repeat = False
  else: print('.', end=''); is_repeat = True
  
  if is_allow_save and keyboard.is_pressed('s'):
    img_path = log_imgs_path+f"f%03d.png"%frame_id
    cv.imwrite(img_path, f_img)
    LogI("'s' pressed, save img to \"%s\""%img_path)
    frame_id += 1
    # 等待按键释放，避免重复触发
    while keyboard.is_pressed('s'): time.sleep(0.1)
  
  if state == last_state and False:
    same_state_count += 1
    if same_state_count > MAX_SAME_STATE_COUNT:
      false_img_path = log_imgs_path+f'false_{state}.png'
      cv.imwrite(false_img_path, img)
      LogE(f"The state \"{state}\" occur to much, saving img to \"{false_img_path}\"")
      if state not in protected_state: break
  else: last_state = state; same_state_count = 1; 
  
  state = 'loading' ########################################################################
  if state == 'ready':
    song_id, song_name = song_recognition.get_song_id(img)
    LogS('ready', f'Recognition song: id:{song_id} name:{song_name}')
    
    song_duration = create_and_push_commands(song_id, user_config)
      
    # 排除 3d 演出、3d cut in、mv 的情况
    while is_checking_3d:
      hsv_img = cv.cvtColor(grabber.grab()[:,:,:3], cv.COLOR_BGR2HSV)
      c1 = hsv_img[COLOR_1_POS[1], COLOR_1_POS[0]]
      c2 = hsv_img[COLOR_2_POS[1], COLOR_2_POS[0]]
      if (COLOR_1_LOW <= c1 and c1 <= COLOR_1_HIGH) and\
        (COLOR_2_LOW <= c2 or c2 <= COLOR_2_HIGH): break
      script.click(140, 652)
      time.sleep(CLICK_GAP*3)
      script.click(0, 0)
      time.sleep(CLICK_GAP*3)
    
    script.act(state)
  elif state == 'playing':
    player.full_grabber.set_window(2)
    player.start_playing(song_duration)
  else:
    script.act(state)
  
  time.sleep(CYCLE_GAP)
  
