from server.player import Player
from configuration import  *
import cv2 as cv
import time
import re
from enum import Enum, auto


class Mode(Enum):
  BatchGrab = auto()
  AddNewSong = auto()
  AddNewLevel = auto()
  

init_scale = 1
player = Player('tcp', init_scale)
grabber = player.full_grabber
title_imgs_path = './song_recognition/title_imgs/'
is_fix = True
x1,y1,x2,y2 = STD_LEVEL_FIX_TITLE_REGION if is_fix else STD_LEVEL_UNFIX_TITLE_REGION

mode = Mode.AddNewLevel

if mode == Mode.BatchGrab:
  title_imgs_path = './song_recognition/title_imgs/'
  start_id = 655
  count = 1
  click_gap = 3.0
  for i in range(count):
    id = start_id+i
    # player.click(0, 1080, 620); time.sleep(2)
    
    img_path = title_imgs_path + "s%03d.png"%id
    print(f"save to \"{img_path}\"")
    cv.imwrite(img_path, grabber.grab()[y1:y2,x1:x2,:3])
    
    # player.click(0,  32,  32); time.sleep(click_gap+1)
    # player.click(0, 380, 272); time.sleep(1.5)
elif mode == Mode.AddNewSong:
  from song_recognition.predict_easyocr import SongRecognition
  recognition = SongRecognition(SHEETS_HEADER_PATH)
  
  while True:
    t_img = grabber.grab()[y1:y2,x1:x2,:3]
    
    pred_idx, pred_song_title, pattern, is_safe = recognition.parse_t_img(t_img)
    
    print(f"pred_song_id    : {pred_idx}")
    print(f"pred_song_title : {pred_song_title}")
    print(f"ocr_text        : {pattern}")
    print(f"is_safe         : {is_safe}")
    
    if is_safe:
      title_img_name = "t-%03d.png"%pred_idx
      y = input(f"Would you like to save meta data as \"{title_img_name}\"?\nPlease input 'y' or 'n': ")
      if y != 'y':
        if y == 'q': continue
        if y == 'e': break
        y = input(f"Please input a num or 'q': ")
        if y[0] == 'q': is_drop = True; print("Drop saving.")
      else: pred_idx = int(y)
      if not is_drop:
        title_img_name = "t-%03d.png"%pred_idx
        cv.imwrite(title_imgs_path+title_img_name, t_img)
        print(f"Save to \"{title_imgs_path+title_img_name}\"")
elif mode == Mode.AddNewLevel:
  from song_recognition.predict_easyocr import SongRecognition
  recognition = SongRecognition(SHEETS_HEADER_PATH)
  
  x1,y1,x2,y2 = STD_LEVEL_FIX_LEVEL_REGION if is_fix else STD_LEVEL_UNFIX_LEVEL_REGION
  
  while(True):
    o_img = grabber.grab()[y1:y2,x1:x2,:3]
    l_img = (np.ones_like(o_img)*255).astype(np.uint8)
    l_img[:, 0:64, :] = o_img[:, 0:64, :]
    pattern = recognition.parse_l_img(l_img)
    level_text = ''.join(filter(lambda x: x.isdigit(), pattern))
    level_id = 0 if level_text == '' else int(level_text)
    
    print(f"ocr_text        :{pattern}")
    
    is_drop = False
    title_img_name = "l-%03d.png"%level_id
    y = input(f"Would you like to save meta data as \"{title_img_name}\"?\nPlease input 'y' or 'n': ")
    if y != 'y':
      if y == 'q': continue
      if y == 'e': break
      y = input(f"Please input a num or 'q': ")
      if y[0] == 'q': is_drop = True; print("Drop saving.")
      else: level_id = int(y)
    if not is_drop:
      title_img_name = "l-%03d.png"%level_id
      cv.imwrite(title_imgs_path+title_img_name, l_img)
      print(f"Save to \"{title_imgs_path+title_img_name}\"")
    
# 更新 features 库
if mode == Mode.AddNewLevel or mode == Mode.AddNewSong:
  from song_recognition.predict_TitleNet import SongRecognition as SR
      
  recognizer = SR(
    ckpt_path=SONG_RECOGNITION_MODEL_PATH,
    img_dir='./song_recognition/title_imgs',
    feature_json_path='./song_recognition/feature_vectors.json',
    is_load_library=False
  )