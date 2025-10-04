from configuration import *
from utils.WinGrabber import *
from play.note_extractor import NoteExtractor
import cv2 as cv
from pathlib import Path
import time

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
  
  def __init__(self, display_scale:int=1, window_name:str='Preview'):
    self.window_name = window_name
    self.img, self.mouse_x, self.mouse_y = None, -1, -1
    self.display_scale = display_scale
    mouse_callback = lambda e, x, y, f, p: self.mouse_callback(e, x, y, f, p)
    cv.namedWindow(self.window_name)
    cv.setMouseCallback(self.window_name, mouse_callback)
    
    self.add_loop_func = lambda : None
    
  def __del__(self):
    cv.destroyWindow(self.window_name)
    
  def mouse_callback(self, event, x, y, flags, param):
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

class Mode(Enum):
  Capture          = auto()
  Record           = auto()
  TraceNote        = auto()
  TraceFirstNote   = auto()
  WalkThrough      = auto()
  WalkThroughSheet = auto()

# 自定义运行时参数
SCALE = 2
is_save        = False                # 是否保存帧
frame_id_start = 0                    # 帧ID起始值
frame_id       = frame_id_start       # 帧ID
frames_path    = './UI_recognition/BangUINet_train_imgs/'     # 帧保存路径
frame_name     = 'f%05d.png'
frame_list     = []                   # 在 WalkThrough 和 WalkThroughSheet 模式下，指定待查看的图片程 frame_id
                                      # 为空列表时，则抓取 frames_path 下所有 png 图片

derive_para    = 0
for tag in [
  NoteExtractor.DerivePara.ALL,
  NoteExtractor.DerivePara.TAG,
  NoteExtractor.DerivePara.NOBG,
]: derive_para |= tag                 # 指定 derive_img 的样式

is_extractor_use_full = False         # 选择 extractor 是截取全屏，还是仅截取与音轨相关的区域

trace_note_path       = \
  './play/trace_note.json'            # ExtractNote 模式下，结果的保存地址
trace_first_note_path = \
  './play/trace_first_note.json'      # ExtractFirstNote 模式下，结果的保存地址

mode = Mode.WalkThrough            # 选择模式

# 计算得到的参数
is_extract_first_note = mode == Mode.TraceFirstNote  # 选择 提取第一个 note，ExtractFirstNote 专用

if mode != Mode.WalkThrough and mode != Mode.WalkThroughSheet:
  full_grabber  = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], None)
  track_grabber = MumuGrabber('Mumu安卓设备', SCALE, None, [STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT], [TRACK_B_X1, TRACK_T_Y, TRACK_B_X2, TRACK_B_Y])
  extractor     = NoteExtractor(full_grabber if is_extractor_use_full else track_grabber, is_extract_first_note)
else: full_grabber, track_grabber, extractor = None, None, None
pv = Preview(8)
def q(): global pv; del pv
def save(img):
  global frame_id
  img_path = frames_path + frame_name%frame_id
  frame_id += 1
  cv.imwrite(img_path, img)
  print(f"Save img to \"{img_path}\"")
def gss(img:None): # grab-show-save
  img = extractor.grab() if img is None else img
  pv.load_img(img, 'hsv')
  pv.show_img()
  save(cv.cvtColor(img, cv.COLOR_HSV2BGR))

# 工作流
if mode == Mode.WalkThrough or mode == Mode.WalkThroughSheet:
  
  if frame_list == []:
    frame_list = []
    for file in Path(frames_path).rglob("*.png"):
      frame_list.append(file.__str__())
    frame_list_cur = 0
  else:
    frame_id_list = frame_list
    frame_list = []
    for i in frame_id_list:
      frame_list.append(frames_path+frame_name%i)
    frame_list_cur = 0
  
  if mode == Mode.WalkThroughSheet:
    def show_img():
      cv.setWindowTitle(pv.window_name, frame_list[frame_list_cur])
      img = cv.imread(frame_list[frame_list_cur])
      img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
      pv.load_img(extractor.extract(img, derive_para)[0], 'hsv')
      pv.show_img()
  else:
    def show_img():
      cv.setWindowTitle(pv.window_name, frame_list[frame_list_cur])
      pv.load_img(frame_list[frame_list_cur], 'bgr')
      pv.show_img()
  def a():
    global frame_list_cur
    if frame_list_cur > 0: frame_list_cur -= 1
    else: frame_list_cur = len(frame_list) - 1
    show_img()
  def b():
    global frame_list_cur
    if frame_list_cur < len(frame_list)-1: frame_list_cur += 1
    else: frame_list_cur = 0
    show_img() 
  
  show_img()
  while True:
    k = cv.waitKey(16) & 0xFF
    if k == ord('q'): q(); break
    elif k == ord('a'): a()
    elif k == ord('b'): b()

elif mode == Mode.Record or mode == Mode.Capture:
  is_recording = False
  while True:
    img = full_grabber.grab()
    pv.load_img(cv.cvtColor(img, cv.COLOR_BGR2HSV), 'hsv')
    pv.show_img()
    if is_recording: save(img)
    
    k = cv.waitKey(16) & 0xFF
    if k == ord('q'): q(); break
    elif k == ord('c'):
      if not is_recording: save(img)
    elif k == ord('s'): print("Start recording"); is_recording = True
    elif k == ord('e'): print("End recording"); is_recording = False
    
elif mode == Mode.TraceNote:
  is_recording = False
  
  while True:
    k = cv.waitKey(16) & 0xFF
    if is_recording:
      img = extractor.grab()
      tim = extractor.get_grab_time()
      derive_img, t_s_list = extractor.extract(img, derive_para)
      if t_s_list != []:
        super_t_s_list.append([tim-start_time, t_s_list])
      if is_save: gss(img)
        
    if k == ord('q'): q(); break
    elif k == ord('c'):
      if not is_recording: gss()
    elif k == ord('s'):
      is_recording = True
      super_t_s_list = []
      extractor.reset_extractor(False)
      start_time = time.time()
      print("Start tracing note ...")
    elif k == ord('e'):
      is_recording = False; 
      with open(trace_note_path, 'w', encoding='utf-8') as file:
        json.dump(super_t_s_list, file)
      print(f"End tracing note, file saved to \"{trace_note_path}\"")
    
elif mode == Mode.TraceFirstNote:
  is_recording = False
  while True:
    if is_recording:
      img = extractor.grab()
      tim = extractor.get_grab_time() 
      derive_img, t_s, is_edge = extractor.extract(img, derive_para)
      if is_save: gss(img)
      if is_edge:
        is_recording = False
        print(first_note_t_s_list)
        with open(trace_first_note_path, 'w', encoding='utf-8') as file:
          json.dump(first_note_t_s_list, file)
        print(f"End tracing first note, file saved to \"{trace_first_note_path}\"")
      elif t_s != [] and t_s[0] >= 0.0:
        first_note_t_s_list.append([tim - start_time, t_s])
    k = cv.waitKey(16) & 0xFF
    if k == ord('q'): q(); break
    elif k == ord('s'):
      is_recording = True
      first_note_t_s_list = []
      extractor.reset_extractor(True)
      start_time = time.time()
      print("Start tracing first note ...")