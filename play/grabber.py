from configuration import *
from utils.WinGrabber import *
import cv2 as cv
import cv2

SCALE = 2
full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [RAW_WIDTH, RAW_HEIGHT], None)
grabber = full_grabber
track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [RAW_WIDTH, RAW_HEIGHT],
                            [TRACK_BL_X, TRACK_T_Y, TRACK_BR_X, TRACK_B_Y])
grabber = track_grabber

while True:
  # 如果处于连续抓取模式，持续抓取和处理图像
  if continuous_grabbing:
    img = grabe()
    current_img = img.copy()  # 保存当前图像用于鼠标回调
    
    handle(img)
    
    if is_save:
      filename = f"{frames_path}/%04d.png" % frame_id
      # 注意：OpenCV保存图像通常使用BGR格式
      cv2.imwrite(filename, img)
      frame_id += 1
    
    # 显示图像（转换为BGR格式以便正确显示）
    derive_img = handle(img)
    cv2.imshow('Preview', derive_img)
  
  # 检测按键输入
  key = cv2.waitKey(1) & 0xFF
  
  if key == ord('q'):  # 退出程序
    print("退出程序")
    break
  elif key == ord('s'):  # 开始连续抓取
    if not continuous_grabbing:
      continuous_grabbing = True
      print("开始连续抓取...")
  elif key == ord('e'):  # 结束连续抓取
    if continuous_grabbing:
      continuous_grabbing = False
      print("结束连续抓取")
  elif key == ord('c'):  # 截取单帧
    img = grabe()
    current_img = img.copy()
    
    derive_img = handle(img)
    
    cv2.imshow('Preview', derive_img)
    print("截取单帧完成")

cv2.destroyAllWindows()


