from configuration import *
from utils.WinGrabber import MumuGrabber
from server.ADB import push_file
from utils.json_refiner import refine
from utils.log import LogE, LogD, LogI, LogS
from UI_recognition.predict import UIRecognition
from song_recognition.predict_TitleNet import SongRecognition
from server.player import Player
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
  Event.Compete,
  Choose.Loop,
  Level.Normal,
  Performance.AllPerfect,
  custom_performance,
  None,
  'skilled'
)
uc = user_config
dilation_time       =  1000000
correction_time     = -  45000

#============ Run Configuration ==ssssss==========#

is_no_action        = False

is_caliboration     = False

play_one_song_id    = 655
is_play_one_song    = False
is_restart_play     = True

is_checking_3d      = True

is_save_commands    = True

log_imgs_path       = './UI_recognition/log_imgs/'
sheets_path         = './sheet/sheets/'
commands_sheet_path = './client/commands.sheet'
commands_json_path  = './client/commands.json'

#============ declaration ============#

if True:
  ui_recognition = UIRecognition()
  song_recognition = SongRecognition(
    ckpt_path='./song_recognition/ckpt_triplet.pth',
    img_dir='./song_recognition/title_imgs',
    feature_json_path='./song_recognition/feature_vectors.json',
    is_load_library=True,
    user_config=user_config
  )

  player = Player('tcp', init_scale=1)
  script = Script(player, user_config)
  player.set_caliboration_para(dilation_time, correction_time)
  grabber = player.full_grabber

  def to_torch_type(img):
    # img 应为 BGR 色彩的 (H,W,C) 维的图片，也即 opencv 导入图片的默认格式
    # 然后 img 将被转换成 RGB 色彩的 (C,H,W) 供图像预测
    return np.transpose(cv.cvtColor(img, cv.COLOR_BGR2RGB), (2, 0, 1))

  def create_and_push_commands(song_id:int, user_config:UserConfig):
    sheet_name = f'{song_id}_{user_config.level}.bestdori'
    if not os.path.exists(sheets_path+sheet_name):
      if user_config.level == Level.Special:
        LogI(f"Song {song_id} has not level \"Specil\", using \"Expert\"")
        sheet_name = f'{song_id}_3.bestdori'
    sheet_path = sheets_path+sheet_name
    if not os.path.exists(sheet_path):
      raise ValueError(f"Sheet \"{sheet_path} doesn't exist\"")
    
    commands, song_duration = sheet2commands(sheet_path, commands_sheet_path, user_config.note_skewer)
    LogS('ready', f'song_duration:{song_duration}')
    LogS('ready', f'Try to upload "{sheet_path}"')
    push_file(commands_sheet_path)
    
    if is_save_commands:
      with open(commands_json_path, "w", encoding="utf-8") as file: json.dump(commands, file)
      refine(commands_json_path)
      LogS('ready', f'Save commands_json to "{commands_json_path}"')
    return song_duration

#============ play one song ============#

if is_caliboration:
  song_duration = -1
  
  if is_restart_play:
    player.click(0, 1248,  32)
    time.sleep(0.6)
    player.click(0,  650, 450)
    time.sleep(0.6)
    player.click(0,  760, 450)
    time.sleep(0.6)
    
  player.set_scale(2)
  diff_time = player.start_playing(song_duration, is_caliboration=True)
  LogI("Receive diff_time:", diff_time)
  
  # 使用 diff_time 去更新 correction_time
  # ...
  
  exit(0)

if is_play_one_song:
  song_duration = create_and_push_commands(play_one_song_id, user_config)
  
  player.send_cmd("f\n")
  
  if is_restart_play:
    player.click(0, 1248,  32)
    time.sleep(0.6)
    player.click(0,  650, 450)
    time.sleep(0.6)
    player.click(0,  760, 450)
    time.sleep(0.6)
    
  player.set_scale(2)
  player.start_playing(song_duration)
  
  exit(0)

#============ Cycle ============#

# 重复状态处理的临时变量
is_allow_save       = True
frame_id           = 0

is_repeat          = True
MAX_SAME_STATE     = 100
same_state_count   = 1

is_ready           = False
ready_count        = 0
MAX_RE_READY       = 10

protected_state    = ['join_wait', 'ready_done']
special_state_list = ['ready']
special_state_dict = {}

last_state          = None
song_duration       = 140

LogI("Cycle start ...")
while True:
  # 截图
  if player.get_scale() == 2: player.set_scale(1); time.sleep(CYCLE_GAP)
  img = grabber.grab()[:,:,:3]
  f_img = cv.resize(img, (STD_WINDOW_WIDTH//8,STD_WINDOW_HEIGHT//8),interpolation=cv.INTER_AREA)
  th_img = to_torch_type(f_img)
  
  # 检测按键，用于保存截图
  if is_allow_save and keyboard.is_pressed('s'):
    img_path = log_imgs_path+f"f%03d.png"%frame_id
    cv.imwrite(img_path, f_img)
    LogI("'s' pressed, save img to \"%s\""%img_path)
    frame_id += 1
    # 等待按键释放，避免重复触发
    while keyboard.is_pressed('s'): time.sleep(0.1)
  
  # 判断当前状态是否与上一个状态相同
  label, state = ui_recognition.get_state(th_img)
  if state != last_state: LogI(f"{'\n'if is_repeat else ''}Recognition state:{state} label:{label}"); is_repeat = False
  else: print('.', end=''); is_repeat = True
  
  # 重复状态处理逻辑，当重复次数大于阈值时，保存截图
  if is_repeat:
    same_state_count += 1
    if same_state_count > MAX_SAME_STATE:
      false_img_path = log_imgs_path+f'false_{state}.png'
      cv.imwrite(false_img_path, img)
      LogE(f"The state \"{state}\" occur to much, saving img to \"{false_img_path}\"")
      if state not in protected_state: break
  else: last_state = state; same_state_count = 1
  
  # 特殊图保存
  if state in special_state_list:
    if state not in special_state_dict: special_state_dict[state] = 0
    img_path = log_imgs_path+f"{state}-%03d.png"%special_state_dict[state]
    special_state_dict[state] += 1
    cv.imwrite(img_path, f_img)
    LogI(f"get state {state}, save img to \"%s\""%img_path)
  
  # 根据状态，执行操作
  if is_no_action: state = 'loading' # 不执行任何操作
  if state == 'ready' and is_ready: 
    if ready_count < MAX_RE_READY: ready_count += 1; state = 'loading'
    else: ready_count = 0
  if state == 'ready':
    song_id, song_name, similarity, song_safe = song_recognition.get_id_by_full_img(img)
    LogS('ready', f'Recognition song: id:{song_id} name:{song_name} similarity:{similarity}')
    
    if not song_safe:
      # 未来在特判掉不安全的情况
      pass
    
    if similarity < 0.95:
      img_path = log_imgs_path+"unknown_song.png"
      cv.imwrite(img_path, img)
      LogE(f'Unknown song, save img to "{img_path}"')
      break
    
    song_duration = create_and_push_commands(song_id, user_config)
      
    # 排除 3d 演出、3d cut in、mv 的情况
    while is_checking_3d:
      hsv_img = cv.cvtColor(grabber.grab()[:,:,:3], cv.COLOR_BGR2HSV)
      c1 = hsv_img[COLOR_1_POS[1], COLOR_1_POS[0]]
      c2 = hsv_img[COLOR_2_POS[1], COLOR_2_POS[0]]
      if ((COLOR_1_LOW <= c1).all() and (c1 <= COLOR_1_HIGH).all()) and\
        ((COLOR_2_LOW <= c2).all() and (c2 <= COLOR_2_HIGH).all()): break
      script.click(140, 652)
      time.sleep(CLICK_GAP*3)
      script.click(500, 650)
      time.sleep(CLICK_GAP*3)
    
    player.send_cmd("f")
    
    time.sleep(0.3)
    
    script.act(state)
    is_ready = True
  elif state == 'playing' and is_ready:
    player.set_scale(2)
    player.start_playing(song_duration)
    is_ready = False
  elif state == 'story_choose':
    hsv_img = cv.cvtColor(grabber.grab()[:,:,:3], cv.COLOR_BGR2HSV)
    story_color = hsv_img[STORY_POS[1], STORY_POS[0]]
    if (STORY_LOW <= story_color).all() and (story_color <= STORY_HIGH).all():
      script.act(state)
  else:
    script.act(state)
    
  # 防止截屏过快
  time.sleep(CYCLE_GAP)
  
