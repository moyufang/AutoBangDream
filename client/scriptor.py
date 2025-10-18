from multiprocessing import Queue

import time
import cv2 as cv
import keyboard
from configuration import *
from module_config.scriptor_config import *
from utils.json_refiner import refine
from utils.log import LogE, LogD, LogI, LogS
from server.ADB import push_file
from server.controller import BangcheaterController
from server.player import WinPlayer
from UI_recognition.predict import UIRecognition
from song_recognition.predict_TitleNet import SongRecognition
from client.script import Script
from client.sheet2commands import sheet2commands

is_config_by_code = __name__ == '__main__'

def start(rQ:Queue, sQ:Queue, logQ:Queue):
  
  class RunConfig:
    def __init__(self): pass
  rc = RunConfig()
  cfg = ScriptorConfig(SCRIPTOR_CONFIG_PATH)
  if not is_config_by_code:
    cfg.save = lambda *arg: None   # 禁用直接保存
    cfg.update = lambda *arg: None # 禁用直接更新
  def refresh():
    nonlocal cfg, rc
    cfg.load()
    for (k,v) in cfg.run_config.items():
      rc.__setattr__(k, v)

#============ User Configuration ============#
  # refresh()
  if is_config_by_code: # 代码里修改配置
    cfg.set_config(
      Mode.Free,
      Event.Compete,
      Choose.Loop,
      Diff.Expert,
      Performance.Custom,
      'master',
      {"lobby":True,"special_stage":True},
    )
    cfg.set_performance(Performance.Custom if cfg.mode == Mode.Event else Performance.AllPerfect)
    # Run Configuration

    rc.correction_time     = -  40000
    rc.target_diff_time    =    55000
    rc.dilation_time       =  1000000

    rc.is_no_action        = False
    rc.is_caliboration     = False

    rc.play_one_song_id    = 316
    rc.is_play_one_song    = False
    rc.is_restart_play     = True

    rc.is_checking_3d      = True

    rc.MAX_SAME_STATE     = 100
    rc.MAX_RE_READY       = 5

    rc.is_allow_save      = True
    rc.is_allow_suspend   = False

    rc.protected_state    = ['join_wait', 'ready_done']
    rc.record_state       = ['award']
    
    cfg.save()
    from utils.json_refiner import refine
    refine(cfg.config_path, [2, 2, 0])

  #============ declaration ============#

  if True:
    ui_recognition = UIRecognition()
    song_recognition = SongRecognition(
      ckpt_path=SONG_RECOGNITION_MODEL_PATH,
      img_dir=TITLE_IMGS_PATH,
      feature_vectors_path=FEATURE_VECTORS_PATH,
      is_load_library=True,
      user_config=cfg
    )
    
    player = WinPlayer('tcp', init_scale=1, remote_port=BANGCHEATER_PORT)
    script = Script(player, cfg)
    player.set_caliboration_parameters(rc.dilation_time, rc.correction_time)

    def to_torch_type(img):
      # img 应为 BGR 色彩的 (H,W,C) 维的图片，也即 opencv 导入图片的默认格式
      # 然后 img 将被转换成 RGB 色彩的 (C,H,W) 供图像预测
      return np.transpose(cv.cvtColor(img, cv.COLOR_BGR2RGB), (2, 0, 1))

    def create_and_push_commands(song_id:int, cfg:ScriptorConfig):
      sheet_name = f'{song_id}_{int(cfg.diff)}.bestdori'
      if not os.path.exists(SHEETS_PATH+sheet_name):
        if cfg.diff == Diff.Special:
          LogI(f"Song {song_id} has not diff \"Special\", using \"Expert\"")
          sheet_name = f'{song_id}_3.bestdori'
      sheet_path = SHEETS_PATH+sheet_name
      if not os.path.exists(sheet_path):
        raise ValueError(f"Sheet \"{sheet_path} doesn't exist\"")
      
      commands, song_duration = sheet2commands(sheet_path, COMMANDS_SHEET_PATH, cfg)
      LogS('ready', f'song_duration:{song_duration}')
      LogS('ready', f'Try to upload "{sheet_path}"')
      push_file(COMMANDS_SHEET_PATH)
      
      with open(COMMANDS_JSON_PATH, "w", encoding="utf-8") as file: json.dump(commands, file)
      refine(COMMANDS_JSON_PATH)
      LogS('ready', f'Save commands_json to "{COMMANDS_JSON_PATH}"')
      return song_duration
    
    def parse(data:dict):
      nonlocal player
      cmd = data['command']
      if cmd == 'refresh': refresh()
      elif cmd == 'stop':
        player.stop_playing()
        exit(0)
    
#============ play one song ============#

  if rc.is_caliboration:
    song_duration = -1
    
    if rc.is_restart_play:
      player.click(0, 1248,  32)
      time.sleep(0.6)
      player.click(0,  650, 450)
      time.sleep(0.6)
      player.click(0,  760, 450)
      time.sleep(0.6)
      
    player.set_scale(2)
    diff_time = player.start_playing(is_caliboration=True)
    # 使用 diff_time 去更新 correction_time
    new_correction_time = rc.correction_time + rc.target_diff_time-int(diff_time)
    LogI(f"Receive diff_time:{diff_time}, recommending 'correction_time':{new_correction_time}")
    
    exit(0)

  if rc.is_play_one_song:
    song_duration = create_and_push_commands(rc.play_one_song_id, cfg)
    
    player.send_cmd("f\n")
    
    if rc.is_restart_play:
      time.sleep(0.6)
      player.click(0, 1248,  32)
      time.sleep(0.6)
      player.click(0,  650, 450)
      time.sleep(0.6)
      player.click(0,  760, 450)
      time.sleep(0.6)
      
    player.set_scale(2)
    player.start_playing(is_caliboration=False)
    time.sleep(song_duration+5)
    
    exit(0)

#============ Cycle ============#

  # 循环时的临时变量
  frame_id           = 0
  false_img_id       = 0
  same_state_count   = 1
  is_ready           = False
  is_repeat          = False
  ready_count        = 0
  last_state         = None
  song_duration      = 140

  def save_false_img(img, state):
    nonlocal false_img_id
    false_img_path = LOG_IMGS_PATH+f'false-%03d-{state}.png'%false_img_id
    false_img_id += 1
    cv.imwrite(false_img_path, img)
    return false_img_path

  LogI("Cycle start ...")
  while True:
    while not sQ.empty(): parse(sQ.get())
    
    # 截图
    if player.get_scale() == 2: player.set_scale(1); time.sleep(CYCLE_GAP)
    img = player.grab_full_img()
    f_img = cv.resize(img, (STD_WINDOW_WIDTH//8,STD_WINDOW_HEIGHT//8),interpolation=cv.INTER_AREA)
    th_img = to_torch_type(f_img)
    
    # 暂停任务
    if rc.is_allow_suspend and keyboard.is_pressed('t'):
      LogI("Scriptor suspend")
      while True:
        time.sleep(0.5)
        if keyboard.is_pressed('c'):
          LogI("Scriptor continue")
          break
    
    # 检测按键，用于保存截图
    if rc.is_allow_save and keyboard.is_pressed('s'):
      img_path = LOG_IMGS_PATH+f"f%03d.png"%frame_id
      cv.imwrite(img_path, f_img)
      LogI("'s' pressed, save img to \"%s\""%img_path)
      frame_id += 1
      # 等待按键释放，避免重复触发
      while keyboard.is_pressed('s'): time.sleep(0.1)
    
    # 判断当前状态是否与上一个状态相同
    label, state = ui_recognition.get_state(th_img)
    if state != last_state: LogI(f"{'\n'if is_repeat else ''}Recognition state:{state} label:{label}"); is_repeat = False
    else: print('.', end=''); is_repeat = True
    
    if state in rc.record_state:
      save_false_img(img, state)
    
    # 重复状态处理逻辑，当重复次数大于阈值时，保存截图
    if state == last_state:
      if is_repeat:
        same_state_count += 1
        if same_state_count > rc.MAX_SAME_STATE:
          same_state_count = 0
          false_img_path = save_false_img(img, state)
          LogE(f"The state \"{state}\" occur too much, saving img to \"{false_img_path}\"")
          if state not in rc.protected_state: break
    else: last_state = state; same_state_count = 1
    
    if cfg.get_is_multiplayer():
      # 准备界面后，不应该出现错误识别
      if is_ready and state not in ['ready_done', 'playing', 'loading', 'join_wait']:
        false_img_path = save_false_img(img, state)
        LogE(f"Unexpected state {state} after 'ready', saveing img to \"{false_img_path}\"")
      # 如果重复次数太多，有问题
      if is_ready and state == 'ready':
        if ready_count < rc.MAX_RE_READY: ready_count += 1; state = 'loading'
        else: ready_count = 0
      
    # 根据状态，执行操作
    if rc.is_no_action: state = 'loading' # 不执行任何操作
    if state == 'ready':
      song_id, song_name, similarity, song_safe = song_recognition.get_id_by_full_img(img)
      LogS('ready', f'Recognition song: id:{song_id} name:{song_name} similarity:{similarity}')
      
      # 在 ready界面可选择难度的情况下，根据其它难度的level，特判掉不安全的情况
      if not song_safe:
        ori_level = cfg.diff
        if song_id in [316, 676]:
          cfg.diff = Diff.Hard
          player.click(0, 770, 580)
          time.sleep(CLICK_GAP_4)
          song_id, song_name, similarity, song_safe = song_recognition.get_id_by_full_img(player.grab_full_img())
        if song_id in [410, 467, 389, 462]:
          cfg.diff = Diff.Expert  
          player.click(0, 860, 580)
          time.sleep(CLICK_GAP_4)
          song_id, song_name, similarity, song_safe = song_recognition.get_id_by_full_img(player.grab_full_img())
        cfg.diff = ori_level
      if not song_safe: raise RuntimeError("Cannot distinguish this same title song")
      
      if similarity < 0.95:
        img_path = LOG_IMGS_PATH+"unknown_song.png"
        cv.imwrite(img_path, img)
        LogE(f'Unknown song, save img to "{img_path}"')
        time.sleep(CYCLE_GAP)
        continue
      
      song_duration = create_and_push_commands(song_id, cfg)
        
      # 排除 3d 演出、3d cut in、mv 的情况
      while rc.is_checking_3d:
        hsv_img = cv.cvtColor(player.grab_full_img(), cv.COLOR_BGR2HSV)
        c1 = hsv_img[COLOR_1_POS[1], COLOR_1_POS[0]]
        c2 = hsv_img[COLOR_2_POS[1], COLOR_2_POS[0]]
        if ((COLOR_1_LOW <= c1).all() and (c1 <= COLOR_1_HIGH).all()) and\
          ((COLOR_2_LOW <= c2).all() and (c2 <= COLOR_2_HIGH).all()): break
        script.click(140, 652)
        time.sleep(CLICK_GAP*3)
        script.click(500, 650)
        time.sleep(CLICK_GAP*3)
      
      script.act(state)
      time.sleep(CLICK_GAP_4)
      
      player.send_cmd("f")
      
      is_ready = True
      time.sleep(CYCLE_GAP)
    elif state == 'playing' and is_ready:
      player.set_scale(2)
      player.start_playing(is_caliboration=False)
      playing_time = time.time()
      while time.time()-playing_time < song_duration+5:
        while not sQ.empty(): parse(sQ.get())
      is_ready = False
    elif state == 'story_choose':
      hsv_img = cv.cvtColor(player.grab_full_img(), cv.COLOR_BGR2HSV)
      story_color = hsv_img[STORY_POS[1], STORY_POS[0]]
      if (STORY_LOW <= story_color).all() and (story_color <= STORY_HIGH).all():
        script.act(state)
    else:
      script.act(state)
      
    # 防止截屏过快
    time.sleep(CYCLE_GAP)


if __name__ == "__main__":
  if True: # 启动 bangcheater
    from server.controller import BangcheaterController
    bcc = BangcheaterController("adb", f"127.0.0.1:{MUMU_PORT}", BANGCHEATER_PORT)
    bcc.start_bangcheater()
  
  rQ, sQ, logQ = Queue(), Queue(), Queue()
  start(rQ, sQ, logQ)

