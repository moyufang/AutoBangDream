import cv2 as cv
import time 
from pathlib import Path
from utils.Preview import Preview

if True:
  start_time = time.time()
  
  load_model_time = time.time() - start_time
  start_time = time.time()
  
  create_ocr_time = time.time()-start_time
  def recog(bgr_img):
    return None

title_imgs_path = './song_recognition/title_imgs/'
title_imgs_list = []

for img_path in Path(title_imgs_path).rglob("*png"):
  title_imgs_list.append(img_path.name)

print(f"load_time:{load_model_time} create_time:{create_ocr_time}")

for i in range(len(title_imgs_list)):
  img = cv.imread(title_imgs_path+title_imgs_list[i])
  # rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  
  start_time = time.time()
  ret = recog(img)
  recog_time = time.time()-start_time
  print(f"recog_time:{recog_time} res_text:{ret} img_name:{title_imgs_list[i]}")
  
  
  
# pv = Preview(1)
  # pv.load_img(img, 'bgr')
  # pv.show_img()
  # k = cv.waitKey(0) & 0xFF
  # if k == ord('q'): break
  # elif k == ord('a'):
  #   if i == 0: i = len(title_imgs_list)-1
  #   else: i -= 1
  # elif k == ord('b'):
  #   if i == len(title_imgs_list)-1: i = 0
  #   else: i += 1
