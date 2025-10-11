from client.player import Player
from configuration import  *
import cv2 as cv
import time
from enum import Enum, auto


class Mode(Enum):
  BatchGrab = auto()
  AddNewSong = auto()
  

init_scale = 1
player = Player('tcp', init_scale)
grabber = player.full_grabber

x1,y1,x2,y2 = STD_LEVEL_FIX_TITLE_REGION

mode = Mode.AddNewSong

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
  
  title_imgs_path = './song_recognition/title_imgs/'
  
  t_img = grabber.grab()[y1:y2,x1:x2,:3]
  
  pred_idx, pred_song_title, pattern, is_safe = recognition.parse_t_img(t_img)
  
  print(f"pred_song_id    : {pred_idx}")
  print(f"pred_song_title : {pred_song_title}")
  print(f"ocr_text        :{pattern}")
  print(f"is_safe         :{is_safe}")
  
  if is_safe:
    title_img_name = "t-%03d.png"%pred_idx
    y = input(f"Would you like to save meta data as \"{title_img_name}\"?\nPlease input 'y' or 'n': ")
    if y == 'y':
      cv.imwrite(title_imgs_path+title_img_name, t_img)
      print(f"Save to \"{title_imgs_path+title_img_name}\"")
      
      from song_recognition.predict_TitleNet import SongRecognition as SR
      
      recognizer = SR(
        ckpt_path=SONG_RECOGNITION_MODEL_PATH,
        img_dir='./song_recognition/title_imgs',
        feature_json_path='./song_recognition/feature_vectors.json',
        is_load_library=False
      )
    else:
      print("Drop saving.")
    
  
  