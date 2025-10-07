import cv2 as cv
import time 
from pathlib import Path
from utils.Preview import Preview
import numpy as np

if False:
  start_time = time.time()
  from song_recognition.predict import *
  load_model_time = time.time() - start_time
  start_time = time.time()
  song_recognition = SongRecognition(None, None, './sheet/sheets_header.json')
  create_ocr_time = time.time()-start_time
  def recog(t_img):
    return song_recognition.parse_t_img(t_img)
  
  print(f"load_time:{load_model_time} create_time:{create_ocr_time}")
  
is_show = False
title_imgs_path = './song_recognition/title_imgs/'
title_imgs_list = []

special_list = ['s043.png', 's210.png', 's214.png', 's216.png','s517.png']
target_list = ['t-635.png', 't-462.png', 't-498.png', 't-513.png','t-149.png']
# for img_path in Path(title_imgs_path).rglob("*png"):
#   if img_path.name not in special_list:
#     title_imgs_list.append(img_path.name)
title_imgs_list.extend(special_list)

mode = 1
if mode == 1:
  if is_show:
    pv = Preview(1, is_mouse=False)
  i = 94
  for i in range(len(title_imgs_list)):
    t_img = cv.imread(title_imgs_path+title_imgs_list[i])
    # t_mask = (t_img < MASK_THRESHOLD).astype(np.uint8)*255

    print(f"\nimg_name:{title_imgs_list[i]}")
    start_time = time.time()
    # id, header, pattern = recog(t_img)
    recog_time = time.time()-start_time
    print(f"recog_time:{recog_time}")
    
    new_name = title_imgs_path+target_list[i]
    o_img = cv.cvtColor(t_img, cv.COLOR_BGR2GRAY)
    cv.imwrite(new_name, o_img)
    
    # if is_show:
    #   pv.load_img(t_mask, 'bgr')
    #   pv.show_img()
    #   k = cv.waitKey(0) & 0xFF
    #   if k == ord('q'): break
    #   elif k == ord('a'):
    #     if i == 0: i = len(title_imgs_list)-1
    #     else: i -= 1
    #   elif k == ord('b'):
    #     if i == len(title_imgs_list)-1: i = 0
    #     else: i += 1
        
    
elif mode == 2:
  for i in range(len(title_imgs_list)):
    t_img = cv.imread(title_imgs_path+title_imgs_list[i])
    t_mask = (t_img < MASK_THRESHOLD).astype(np.uint8)*255
    # rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    print(f"\nimg_name:{title_imgs_list[i]}")
    start_time = time.time()
    ret = recog(t_mask)
    recog_time = time.time()-start_time
    print(f"recog_time:{recog_time}")
  
  


