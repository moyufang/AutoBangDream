from enum import Enum, auto
from configuration import *
from utils.WinGrabber import *
import cv2 as cv
import cv2

class Mode(Enum):
  RecordNote   = auto()
  GetFirstNote = auto()
  Record       = auto()
  WalkThrough  = auto()

class Preview:
  img_type = {
    'bgr' : None,
    'bgra': cv.COLOR_BGRA2BGR
    'hsv' : cv.COLOR_HSV2BGR,
    'rgb' : cv.COLOR_RGB2BGR,
    'gray': cv.COLOR_GRAY2BGR,
    'lab' : cv.COLOR_Lab2BGR,
    'yuv' : cv.COLOR_YUV2BGR,
    'hls' : cv.COLOR_HLS2BGR
  }
  
  def __init__(self, mode:Mode, window_name:str='Preview'):
    self.window_name = window_name
    self.img, self.mouse_x, self.mouse_y = None, -1, -1
    mouse_callback = lambda e, x, y, f, p: self.mouse_callback(e, x, y, f, p)
    cv.namedWindow(self.window_name)
    cv.setMouseCallback(self.window_name, mouse_callback)
    
  def __del__(self):
    cv2.destroyWindow(self.window_name)
    
  def mouse_callback(event, x, y, flags, param):
    self.mouse_x, self.mouse_y = x, y
    if event == cv2.EVENT_MOUSEMOVE and self.img is not None:
      # 确保坐标在图像范围内
      if 0 <= y < self.img.shape[0] and 0 <= x < self.img.shape[1]:
        color = self.img[y, x]
        print("(x,y):(%4d, %4d) color:" % (x, y) + str(color))
        
  def load_img(self, img, ty:str=None):
    self.type = ty
    
    if isinstance(img, str):
      assert(os.path.exist(img))
      img_path = img
      self.img = cv.imread(img)
      assert(self.img != None)
      
      if self.type in Preview.img_type: pass
      elif img.ndim == 2: self.type = 'gray'
      elif img.ndim == 3 and img.shape[2] == 3: self.type = 'bgr'
      elif img.ndim == 3 and img.shape[2] == 4: self.type = 'bgra'
      
    else isinstance(img, np.ndarray):
      self.img, self.type = img, ty
      
    if self.type not in Preview.img_type: self.type = 'Unknown'
    print(f"Loading img succeeded with type:\"{type}\" shape:{self.img.shape} type:{self.type}")
    
  def show_img(self):
    cv.imshow(self.window_name, self.img, Preview.img_type[self.type])
    
  def parse_key(self, k):
    print("按键说明:")
    print("s - 开始连续抓取")
    print("e - 结束连续抓取") 
    print("c - 截取单帧")
    print("q - 退出程序")
    pass

class NoteExtractor:
  class DerivePara(Enum):
    NONE   = 0    # 不修改
    BLUE   = 1<<0 # 显示 single tap
    GREEN  = 1<<1 # 显示 slide 和 long 的 tap，但不包括 middle note
    PINK   = 1<<2 # 显示 flick，包括尾端 slide 和 long 尾端的 flick
    YELLOW = 1<<3 # 显示 skill tap，包括 single、slide、long
    PURPLE = 1<<4 # 显示 left directional tap
    ORANGE = 1<<5 # 显示 right directional tap
    ALL    = 1<<6 # 所有 note 都显示
    NOBG   = 1<<7 # 不显示背景
    TAG    = 1<<8 # 用圆点标记 note 的质心
    
  def __init__(self, mode:Mode, grabber, is_to_hsv:bool = True):
    self.grabber = grabber
    self.img = None
    self.is_to_hsv = is_to_hsv
  
  def grab(self):
    self.img = self.grabber.grab()[:,:,3]
    if self.is_to_hsv: self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
    
  def extract(self, hsv_img, derive_para):
    gbr          = self.grabber
    scale        = gbr.SCALE
    health_pos   = [(HEALTH_POS[i]-gbr.std_region[i])//scale, for i in range(2)]
    health_color = hsv_img[health_pos[1], health_pos[0]]
    is_playing   = ((HEALTH_LOW <= health_color) & (health_color <= HEALTH_HIGH)).all()
    if not is_playing:
      print("no playing health_color:", hsv_img[health_pos[1], health_pos[0]])
      return hsv_img
    
    mask1 = cv.inRange(hsv_img, BLUE_LOW   , BLUE_HIGH  )
    mask2 = cv.inRange(hsv_img, GREEN_LOW  , GREEN_HIGH )
    mask3 = cv.inRange(hsv_img, PINK_LOW   , PINK_HIGH  )
    mask4 = cv.inRange(hsv_img, YELLOW_LOW , YELLOW_HIGH)
    mask5 = cv.inRange(hsv_img, PURPLE_LOW , PURPLE_HIGH)
    mask6 = cv.inRange(hsv_img, ORANGE_LOW , ORANGE_HIGH)
    mask  = mask1 | mask2 | mask3 | mask4 | mask5 | mask6
    
    A1    = (TRACK_B_X1 - TRACK_B_X1)//scale
    A2    = (TRACK_B_X2 - TRACK_B_X1)//scale
    B1    = (TRACK_T_X1 - TRACK_B_X1)//scale
    B2    = (TRACK_T_X2 - TRACK_B_X1)//scale
    C1    = (TRACK_T_Y  - TRACK_T_Y )//scale
    C2    = (TRACK_B_Y  - TRACK_T_Y )//scale
    A,B   = (A1 + A2)//2,   (B1 + B2)//2
    a,b,h = (A2 -  A)   ,   (B2 -  B)   ,   (C2 - C1)
    
    print("(A1,A2):(%4d,%4d) (B1,B2):(%4d,%4d) (A,B):(%4d,%4d) (a,b,h):(%4d,%4d)",A1,A2,B1,B2,A,B,a,b,h)

    max_s = -0.1
    tagging_centroid_list = []
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1,retval):
      sta       = stats[i]
      cen       = np.int32(centroids[i])
      x,y       = cen[0]-A, cen[1]-C1
      t,s       = y/h     , x*h/((a-b)*y+b)
      fill_rate = sta[4]/(sta[2]*sta[3])
      if fill_rate < MIN_FILL_RATE or sta[4] < MIN_PIXEL_COUNT:
        mask = mask*np.uint8(labels != i)
      else: s >= MIN_RATE_S  and s <= MAX_RATE_S:
        #oimg = cv.circle(oimg,cen, 10, (0,0,255), -1)
        #print("rate_y", rate_y)
        max_s = max(max_s, s)
    #print("max_s:", max_s)
    
    self.mask1 = mask1 & mask; self.mask2 = mask2 & mask; self.mask3 = mask3 & mask
    self.mask4 = mask4 & mask; self.mask5 = mask5 & mask; self.mask6 = mask6 & mask
    self.mask  = mask ; self.rmask = cv.bitwise_not(self.mask)
    
    return is_playing, derive_img
      

# 自定义运行时参数
frames_path    = './play/frames/' # 帧保存路径
is_save        = False            # 是否保存图片
frame_id_start = 0                # 帧ID起始值
frame_id       = frame_id_start   # 帧ID
handle         = handle_nop       # 图像处理函数

# continuous_grabbing = False       # 连续抓取状态标志



