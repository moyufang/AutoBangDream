from enum import IntFlag
from configuration import *
from utils.WinGrabber import *
import cv2 as cv

class HealthExtractor:
  def __init__(self, grabber:MumuGrabber, is_to_hsv:bool = True):
    self.grabber    = grabber
    self.scale      = grabber.scale
    self.is_to_hsv  = is_to_hsv
    self.is_playing = False
  def grab(self):
    self.img = self.grabber.grab()[:,:,:3]
    if self.is_to_hsv: self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
    return self.img
  def get_is_playing(self, hsv_img:cv.Mat|None=None):
    if hsv_img == None: hsv_img = self.grab()
    # 提取生命条位置的颜色，判断演出是否开始
    health_pos   = [(HEALTH_POS[i]-self.grabber.std_region[i])//self.scale for i in range(2)]
    health_color = hsv_img[health_pos[1], health_pos[0]]
    self.is_playing   = ((HEALTH_LOW <= health_color) & (health_color <= HEALTH_HIGH)).all()
    if not self.is_playing: # 演出未开始，无效返回
      #print("no playing, with health_pos's color:", hsv_img[health_pos[1], health_pos[0]])
      self.is_playing = False
    return self.is_playing

class NoteExtractor:
  class DerivePara(IntFlag):
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
    self.img = self.grabber.grab()[:,:,:3]
    if self.is_to_hsv: self.img = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
    return self.img
  
  def get_grab_time(self):
    return self.grabber.grabber.grab_time
  
  def reset_extractor(self, is_extract_first_note):
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
    
    A1,A2 = (TRACK_B_X1 - TRACK_B_X1)//scale, (TRACK_B_X2 - TRACK_B_X1)//scale
    B1,B2 = (TRACK_T_X1 - TRACK_B_X1)//scale, (TRACK_T_X2 - TRACK_B_X1)//scale
    C1,C2 = (TRACK_T_Y  - TRACK_T_Y )//scale, (TRACK_B_Y  - TRACK_T_Y )//scale
    a,b,h = (A2 - A1)//2   ,(B2 - B1)//2,   (C2 - C1)
    A,B,C = (A1 + A2)//2+(TRACK_B_X1-gbr.std_region[0])//scale,\
            (B1 + B2)//2+(TRACK_T_X1-gbr.std_region[0])//scale,\
            (TRACK_T_Y - gbr.std_region[1])//scale
    
    #print("(A1,A2):(%4d,%4d) (B1,B2):(%4d,%4d) (A,B,C):(%4d,%4d,%4d) (a,b,h):(%4d,%4d,%4d)"%(A1,A2,B1,B2,A,B,C,a,b,h))

    # 过滤出可靠 note，效果对参数(mask颜色、FILL_RATE、PIXEL_COUNT、RATE_S)和游戏渲染环境强依赖
    max_t_s = [-0.1, -0.2]
    t_s_list = []
    centroid_to_tag_list = []
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask[0], connectivity=8)
    for i in range(1,retval):
      sta       = stats[i]
      cen       = np.int32(centroids[i])
      x,y       = cen[0]-A, cen[1]-C
      t,s       = y/h     , x*h/((a-b)*y+b*h)
      fill_rate = sta[4]/(sta[2]*sta[3])
      
      is_avail        = fill_rate >=  MIN_FILL_RATE and sta[4] >= MIN_PIXEL_COUNT and fill_rate <= MAX_FILL_RATE
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
      mask[7] = cv.bitwise_not(mask[0])
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

# 具体用法，请参考 client/workflow
      
