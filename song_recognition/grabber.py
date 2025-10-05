from client.player import Player
from configuration import  *
import cv2 as cv
import time


init_scale = 1
player = Player('tcp', init_scale)
grabber = player.full_grabber

title_imgs_path = './song_recognition/title_imgs/'
start_id = 71
count = 30
click_gap = 4.0
for i in range(count):
  id = start_id+i
  player.click(0, 1080, 620); time.sleep(click_gap)
  
  img_path = title_imgs_path + "s%03d.png"%id
  x1,y1,x2,y2 = STD_LEVEL_FIX_TITLE_REGION
  cv.imwrite(img_path, grabber.grab()[y1:y2,x1:x2,:3])
  
  player.click(0,  32,  32); time.sleep(click_gap+2)
  player.click(0, 380, 272); time.sleep(1.5)
  