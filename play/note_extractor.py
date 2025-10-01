from enum import Enum, auto
from configuration import *
from utils.WinGrabber import *
import cv2 as cv
from typing import Callable
from pathlib import Path
import time

class NoteExtractor:
  class DerivePara(Enum):
    NONE   = 0    # 不修改
    ALL    = 1<<0 # 所有 note 都显示
    BLUE   = 1<<1 # 显示 single tap
    GREEN  = 1<<2 # 显示 slide 和 long 的 tap，但不包括 middle note
    PINK   = 1<<3 # 显示 flick，包括尾端 slide 和 long 尾端的 flick
    YELLOW = 1<<4 # 显示 skill tap，包括 single、slide、long
    PURPLE = 1<<5 # 显示 left directional tap
    ORANGE = 1<<6 # 显示 right directional tap
    NOBG   = 1<<7 # 不显示背景
    TAG    = 1<<8 # 用圆点标记 note 的质心
    
  def __init__(self, grabber:MumuGrabber, is_extract_first_note:bool, is_to_hsv:bool = True):
    self.grabber    = grabber
    self.img        = None
    self.is_to_hsv  = is_to_hsv
    self.reset_extractor(is_extract_first_note)
  
  def grab(self):
    self.img = self.grabber.grab()[:,:,3]
    if self.is_to_hsv: self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
    return self.img
  def get_grab_time(self):
    return self.grabber.grabber.grab_time
  
  def reset_extractor(self, is_extract_first_note):
    self.is_playing = False
    self.is_extract_first_note = is_extract_first_note
    
  # 提取 note，完成三个任务
  # （1）返回标记过后的图像，辅助开发和调试，应用时不执行该任务
  # （2）返回每个有效 note 的位置率 (t, s)，还原公式是 (x = (a-b)*s*t+b*s, y = h*h)
  #      其中 a, b, h 分别是下水平线的半长度、上水平线的半长度、上下水平线之间的距离
  #      辅助模型训练，应用时不执行该任务
  # （3）提取一首歌 first note 的位置轨迹 (t, s)，同时当 first note 足够接近判定线（下水平线时），通知外部
  #      应用时执行该任务，因排除其它任务的开销忽略不计，易复用代码 
  def extract(self, hsv_img, derive_para:DerivePara):
    gbr               = self.grabber
    scale             = gbr.scale
    health_pos        = [(HEALTH_POS[i]-gbr.std_region[i])//scale for i in range(2)]
    health_color      = hsv_img[health_pos[1], health_pos[0]]
    is_playing        = ((HEALTH_LOW <= health_color) & (health_color <= HEALTH_HIGH)).all()
    if not self.is_playing and is_playing:
      self.is_playing = True
      self.first_note_t_s = [(-1.0, -2.0)]
    if not is_playing: # 演出未开始，无效返回
      print("no playing, with health_pos's color:", hsv_img[health_pos[1], health_pos[0]])
      self.is_playing = False
      return None, None
    
    mask = [
      None,
      cv.inRange(hsv_img,   BLUE_LOW ,   BLUE_HIGH),
      cv.inRange(hsv_img,  GREEN_LOW ,  GREEN_HIGH),
      cv.inRange(hsv_img,   PINK_LOW ,   PINK_HIGH),
      cv.inRange(hsv_img, YELLOW_LOW , YELLOW_HIGH),
      cv.inRange(hsv_img, PURPLE_LOW , PURPLE_HIGH),
      cv.inRange(hsv_img, ORANGE_LOW , ORANGE_HIGH),
      None
    ]
    mask[0] = mask[1]
    for i in range(2, 7): mask[0] = mask[0] | mask[i]
    
    A1    = (TRACK_B_X1 - TRACK_B_X1)//scale
    A2    = (TRACK_B_X2 - TRACK_B_X1)//scale
    B1    = (TRACK_T_X1 - TRACK_B_X1)//scale
    B2    = (TRACK_T_X2 - TRACK_B_X1)//scale
    C1    = (TRACK_T_Y  - TRACK_T_Y )//scale
    C2    = (TRACK_B_Y  - TRACK_T_Y )//scale
    A,B   = (A1 + A2)//2,   (B1 + B2)//2
    a,b,h = (A2 -  A)   ,   (B2 -  B)   ,   (C2 - C1)
    
    print("(A1,A2):(%4d,%4d) (B1,B2):(%4d,%4d) (A,B):(%4d,%4d) (a,b,h):(%4d,%4d)",A1,A2,B1,B2,A,B,a,b,h)

    # 过滤出可靠 note，效果对参数(mask颜色、FILL_RATE、PIXEL_COUNT、RATE_S)和游戏渲染环境强依赖
    max_t_s = [-0.1, -0.2]
    t_s_list = []
    centroid_to_tag_list = []
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask[0], connectivity=8)
    for i in range(1,retval):
      sta       = stats[i]
      cen       = np.int32(centroids[i])
      x,y       = cen[0]-A, cen[1]-C1
      t,s       = y/h     , x*h/((a-b)*y+b)
      fill_rate = sta[4]/(sta[2]*sta[3])
      
      is_avail        = fill_rate >=  MIN_FILL_RATE and sta[4] >= MIN_PIXEL_COUNT
      is_considerable =         t >=     MIN_RATE_T and      t <=      MAX_RATE_T
      if is_avail and is_considerable:
        centroid_to_tag_list.append(cen)
        t_s_list.append([t, s])
        if t > max_t_s[0]: max_t_s = [t, s]
      elif not self.is_extract_first_note:
        mask[0] = mask[0]*np.uint8(labels != i)
        
    if self.is_extract_first_note: # 任务（3）出口
      return None, max_t_s, max_t_s[0] > EDGE_RATE_T
    
    # 根据 derive_para 生成 derive_img，便于开发者调试
    if derive_para == NoteExtractor.DerivePara.NONE:
      derive_img = hsv_img
    else:
      for i in range(1, 6): mask[i] = mask[i] | mask[0]
      mask[7] = cv.bitwise_not(mask)
      derive_mask = np.zeros_like(mask[0])
      if (derive_para & NoteExtractor.DerivePara.ALL) > 0:
          derive_mask |= mask[0]
      else:
        for i in range(1, 7):
          if ((derive_para & (1<<i)) > 0): derive_mask |= mask[i]

      if (derive_para & NoteExtractor.DerivePara.NOBG) == 0:
        derive_mask |= mask[7] # reverse mask[0]
    
      derive_img = cv.bitwise_and(hsv_img, hsv_img, mask=derive_mask)
    
      if (derive_para & NoteExtractor.DerivePara.TAG) > 0:
        for x,y in centroid_to_tag_list:
          derive_img = cv.circle(derive_img, [x,y], TAG_RADIUS, TAG_COLOR, -1)
    
    return derive_img, t_s_list # 任务（1）（2）出口

class Mode(Enum):
  Capture          = auto()
  Record           = auto()
  RecordNote       = auto()
  ExtractFirstNote = auto()
  WalkThrough      = auto()
  WalkThroughSheet = auto()
  
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
  
  def __init__(self, mode:Mode, key2func:dict, window_name:str='Preview'):
    self.key2func = key2func.copy()
    self.window_name = window_name
    self.img, self.mouse_x, self.mouse_y = None, -1, -1
    mouse_callback = lambda e, x, y, f, p: self.mouse_callback(e, x, y, f, p)
    cv.namedWindow(self.window_name)
    cv.setMouseCallback(self.window_name, mouse_callback)
    
    self.add_loop_func = lambda : None
    
  def __del__(self):
    cv.destroyWindow(self.window_name)
    
  def mouse_callback(self, event, x, y, flags, param):
    self.mouse_x, self.mouse_y = x, y
    if event == cv.EVENT_MOUSEMOVE and self.img is not None:
      # 确保坐标在图像范围内
      if 0 <= y < self.img.shape[0] and 0 <= x < self.img.shape[1]:
        color = self.img[y, x]
        print("(x,y):(%4d, %4d) color:" % (x, y) + str(color))
        
  def load_img(self, img:str|cv.Mat, ty:str=None):
    self.type = ty
    
    if isinstance(img, str):
      assert(os.path.exist(img))
      self.img = cv.imread(img)
      assert(self.img != None)
      
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
    print(f"Loading img succeeded with type:\"{type}\" shape:{self.img.shape} type:{self.type}")
    
  def show_img(self):
    cv.imshow(self.window_name, self.img, Preview.img_type[self.type])
  
  # def add_key_func_pair_on_mode(self, key:str, func:Callable, mode:Mode=None):
  #   if mode == None: mode = self.mode
  #   self.preview.key2func[(mode, key)] = func
    
  # def add_loop_func(self, func:Callable, mode:Mode=None):
  #   self.loop_func = func
  
  # def parse_key(self, key:chr):
  #   if (mode, key) in self.key2func: self.key2func[(mode, key)]
  
  # def loop(self):
  #   while True:
  #     if self.loop_func() == -1: break

# 自定义运行时参数
is_save        = False            # 是否保存帧
frame_id_start = 0                # 帧ID起始值
frame_id       = frame_id_start   # 帧ID
frames_path    = './play/frames/' # 帧保存路径
frame_name     = 'f%05d'
derive_para    = NoteExtractor.DerivePara.ALL |\
                 NoteExtractor.DerivePara.TAG
note_track_path = './play/note_track.json'
mode = Mode.WalkThrough
# 自定义运行时参数

full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None)
track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], [TRACK_B_X1, TRACK_T_Y, TRACK_B_X2, TRACK_B_Y])
extractor     = NoteExtractor(track_grabber, False)
pv = Preview(mode, {})
def q(): del pv
if mode == Mode.WalkThrough or mode == Mode.WalkThroughSheet:
  frame_list = []
  for file in Path(frames_path).rglob("*.png"):
    frame_list.append(file.__str__)
  frame_list_cur = 0
  
  if mode == Mode.WalkThroughSheet:
    def show_img():
      img = cv.imread(frame_list[frame_list_cur], cv.IMREAD_COLOR_RGB)
      img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
      pv.load_img(extractor.extract(img, derive_para), 'hsv')
      pv.show_img()
  else:
    def show_img():
      pv.load_img(frame_list[frame_list_cur], 'hsv')
      pv.show_img()
  def a():
    if frame_list_cur > 0: frame_list_cur -= 1
    else: frame_list_cur = len(frame_list) - 1
    show_img()
  def b():
    if frame_list_cur < len(frame_list)-1: frame_list_cur += 1
    else: frame_list_cur = 0
    show_img() 
  
  while True:
    k = cv.waitKey(0) & 0xFF
    if k == ord('q'): q(); break
    elif k == ord('a'): a()
    elif k == ord('b'): b()

elif mode == Mode.Record or mode == Mode.Capture:
  while True:
    is_recording = False
    if is_recording:
      img = full_grabber.grab()
      pv.load_img(cv.cvtColor(img, cv.COLOR_BGR2HSV), 'hsv')
      pv.show_img()
    def c():
      img = full_grabber.grab()
      pv.load_img(cv.cvtColor(img, cv.COLOR_BGR2HSV), 'hsv')
      pv.show_img()
      if is_save:
        img_path = frames_path + frame_name%frame_id
        cv.imwrite(img_path, full_grabber.grab())
    k = cv.waitKey(16) & 0xFF
    if k == ord('q'): q(); break
    elif k == ord('c'): c()
    elif k == ord('s'): is_recording = True
    elif k == ord('e'): is_recording = False
    
elif mode == Mode.RecordNote:
  is_recording = False
  def e():
    with open(note_track_path, 'w', 'utf-8') as file:
      json.dump(super_t_s_list, file)
    super_t_s_list = []
    
  while True:
    k = cv.waitKey(16) & 0xFF
    if is_recording:
      img = extractor.grab()
      tim = extractor.get_grab_time()
      derive_img, t_s_list = extractor.extract(img, derive_para)
      super_t_s_list.append([tim, t_s_list])
    if k == ord('q'): q(); break
    elif k == ord('s'): is_recording = True; super_t_s_list = []; extractor.reset_extractor(False)
    elif k == ord('e'): is_recording = False; e()
    
elif mode == Mode.ExtractFirstNote:
  is_recording = False
  while True:
    if is_recording:
      img = extractor.grab()
      tim = extractor.get_grab_time()
      derive_img, t_s, is_edge = extractor.extract(img, derive_para)
      if is_edge:
        is_recording = False
        print(first_note_t_s_list)
      else:
        first_note_t_s_list.append([tim, t_s])
    k = cv.waitKey(16) & 0xFF
    if k == ord('q'): q(); break
    elif k == ord('s'):
      is_recording = True
      first_note_t_s_list = []
      extractor.reset_extractor(True)
    elif k == ord('e'): is_recording = False
  
#pv.add_key_func_pair_on_mode('a')



