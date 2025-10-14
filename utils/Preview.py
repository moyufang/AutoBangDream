import cv2 as cv
import numpy as np
import os

class Preview:
  img_type = {
    'bgr' : None,
    'bgra': cv.COLOR_BGRA2BGR,
    'hsv' : cv.COLOR_HSV2BGR,
    'rgb' : cv.COLOR_RGB2BGR,
    'gray': cv.COLOR_GRAY2BGR,
    'lab' : cv.COLOR_Lab2BGR,
    'yuv' : cv.COLOR_YUV2BGR,
    'hls' : cv.COLOR_HLS2BGR
  }
  
  def __init__(self, display_scale:int=1, window_name:str='Preview', is_mouse:bool =True):
    self.window_name = window_name
    self.img, self.mouse_x, self.mouse_y = None, -1, -1
    self.display_scale = display_scale
    self.is_mouse = is_mouse
    mouse_callback = lambda e, x, y, f, p: self.mouse_callback(e, x, y, f, p)
    cv.namedWindow(self.window_name)
    cv.setMouseCallback(self.window_name, mouse_callback)
    
    self.add_loop_func = lambda : None
    
  def __del__(self):
    cv.destroyWindow(self.window_name)
    
  def mouse_callback(self, event, x, y, flags, param):
    if not self.is_mouse: return
    self.mouse_x, self.mouse_y = x, y
    x, y = x//self.display_scale, y//self.display_scale
    if event == cv.EVENT_MOUSEMOVE and self.img is not None:
      # 确保坐标在图像范围内
      if 0 <= y < self.img.shape[0] and 0 <= x < self.img.shape[1]:
        color = self.img[y, x]
        print("(x,y):(%4d, %4d)|(%4d,%4d) color:" % (x, y, x*self.display_scale, y*self.display_scale) + str(color)+f" type:{self.type}")
        
  def load_img(self, img:str|cv.Mat, ty:str=None):
    self.type = ty
    
    if isinstance(img, str):
      assert(os.path.exists(img))
      self.img = cv.imread(img)
      print("img:", img, " self.img:", self.img.shape if img is not None else None)
      assert(self.img is not None)
      
      if self.type in Preview.img_type: pass
      elif img.ndim == 2: self.type = 'gray'
      elif img.ndim == 3 and img.shape[2] == 3: self.type = 'bgr'
      elif img.ndim == 3 and img.shape[2] == 4: self.type = 'bgra'
      else: self.type = 'unknown'
      
    elif isinstance(img, np.ndarray):
      self.img, self.type = img, ty
    else:
      print("Loading img failed: Unknown img.")
      
    if self.type not in Preview.img_type: self.type = 'unknown'
    # print(f"Loading img succeeded with type:\"{self.type}\" shape:{self.img.shape} type:{self.type}")
    
  def show_img(self):
    if self.display_scale != 1:
      img = cv.resize(src=self.img, dsize=None, fx=self.display_scale, fy=self.display_scale, interpolation=cv.INTER_CUBIC)
    else:
      img = self.img
    if self.type in Preview.img_type:
      cv.imshow(self.window_name, cv.cvtColor(img, Preview.img_type[self.type]))
    else:
      cv.imshow(self.window_name, img)
      
  def set_title(self, title:str=''):
    if title == '': title = self.window_name
    cv.setWindowTitle(self.window_name, title)