from enum import Enum, auto
import json
import os
import numpy as np

#mumu模拟器设置的分辨率
SCALE, STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT = 1, 1280, 720 

# 选歌模式
class Choose(Enum):
  Loop           = auto() #单曲循环
  Random         = auto() #随机选曲
  ListUp         = auto() #列表循环，向上（目前未支持到头时自动回到底部）
  ListDowm       = auto() #列表循环，向下（目前未支持到底时自动回到顶部）

# 活动种类
class Event(Enum):
  Mission        = auto() # 任务活动
  Trial          = auto() # 试炼活动
  Challenge      = auto() # 挑战活动 
  Tour           = auto() # 巡回演出活动
  Team           = auto() # 团队竞演活动
  Compete        = auto() # 竞演活动

# 演出模式
class Mode(Enum):
  Free           = auto() # 自由演出
  Collaborate    = auto() # 协力演出
  Challenge      = auto() # 舞台挑战
  Event          = auto() # 活动演出(Mission   -> 协力演出
                          #         Trial     -> 协力演出
                          #         Challenge -> 挑战演出
                          #         Tour      -> 巡回演出
                          #         Team      -> 团队竞演
                          #         Compete   -> 竞演演出)

# 选曲难度
class Level(Enum):
  Easy           = 0
  Normal         = 1
  Hard           = 2
  Expert         = 3
  Special        = 4      #如果该歌曲不存在 Special 难度，则选择 Expert 难度
  
# 演出水平
class Performance(Enum):
  AllPerfect     = auto() # AP
  FullCombo      = auto() # FC
  Custom         = auto() # 自定义，选择 Perfect、Great、Good、Bad、Miss 各自的概率
  DropLastCustom = auto() # 自定义，同时 Miss 最后一个键（防 FC）

# 音符判定
class Note(Enum):
  Perfect        = auto()
  Great          = auto()
  Good           = auto()
  Bad            = auto()
  Miss           = auto()
  
class CustomPerformance:
  def __init__(self, is_load:bool=False, file_path:str = '', is_custom_level_list=False):
    self.is_load = is_load
    self.file_path = file_path
    self.is_custom_level_list = is_custom_level_list
    if is_load:
      assert(os.path.isfile(file_path))
    
    # (title,      [perfect,great,good , bad  , miss ])
    self.default_weights = [
      ("blind",    [0.450, 0.300, 0.050, 0.050, 0.150]),
      ('hell',     [0.600, 0.200, 0.050, 0.050, 0.100]),
      ('fool',     [0.700, 0.200, 0.050, 0.030, 0.020]),
      ('newbee',   [0.850, 0.095, 0.005, 0.015, 0.035]),
      ('skilled',  [0.930, 0.050, 0.005, 0.005, 0.010]),
      ('master',   [0.950, 0.040, 0.003, 0.003, 0.004]),
      ('top',      [0.980, 0.017, 0.001, 0.001, 0.001]),
      ('nohuman',  [0.985, 0.012, 0.001, 0.001, 0.001]),
      ('newworld', [0.995, 0.002, 0.001, 0.001, 0.001]),
      ('god',      [0.100, 0.000, 0.000, 0.000, 0.000]),
    ]
    # (title, level_num)
    self.default_level_list = [
      ("blind",    32),
      ("hell",     30),
      ("fool",     29),
      ("newbee",   28),
      ("skilled",  27),
      ("master",   26),
      ("top",      25),
      ("nohuman",  23),
      ("newworld", 18),
      ("god",      0 )
    ]
    
    self.weights_map = {}
    for k,l in self.default_weights:self.weights_map[k] = l
    if self.is_load: self.load_custom()
    if not self.is_custom_level_list: self.level_list = self.default_level_list
  
  # 根据难度选择 title, 进而根据 title 选择 weights
  def get_weights(self, level_num:int):
    for item in self.level_list:
      if level_num < item[1]: continue
      else: return self.weights_map[item[0]]
    return self.weights_map['blind']
  
  def set_level_list(self, level_list:list=[]):
    if not level_list: level_list = self.default_level_list.copy(); return False
    for item in level_list:
      assert(isinstance(item, tuple) and isinstance(item[0], str) and isinstance(item[1], int))
    self.set_level_list = level_list.copy()
    return True
  
  def add_weights(self, key:str, weights:list):
    assert(isinstance(key, str))
    assert(len(weights, 5))
    sum = 0.0
    for i in weights: assert(isinstance(weights, float)); sum += i
    EPS = 1e-6
    assert(sum > 1.0-EPS and sum < 1.0+EPS)
    assert(key != '')
    assert(key in self.weights_map)
    self.weights_map[key] = weights.copy()
    
  def save_custom(self, file_path:str=''):
    if file_path: self.file_path = file_path
    else: file_path = self.file_path
    custom_weights = []
    for k in self.weights_map:
      if k in self.default_weights: continue
      custom_weights.append = self.weights_map[k]
    json.dump({'custom_weights':custom_weights, 'level_list':self.level_list}, file_path)
  
  def load_custom(self, file_path:str=''):
    if file_path: self.file_path = file_path
    else: file_path = self.file_path
    dct = json.load(file_path)
    custom_weights = dct['custom_weights']
    self.level_list = dct['level_list']
    for k,l in custom_weights:self.weights_map[k] = l
    
  def clear_custom(self):
    self.weights_map = {}
    self.save_custom()

#============ POS ============#

TRACK_LT      = ( 474,  180)
TRACK_RT      = ( 804,  180)
TRACK_LB      = ( 120,  580)
TRACK_RB      = (1160,  580)

TRACK_T_X1    = TRACK_LT[0]
TRACK_T_X2    = TRACK_RT[0]
TRACK_T_Y     = TRACK_LT[1]
TRACK_T_LEN   = TRACK_T_X2 - TRACK_T_X1
TRACK_T_BLOCK = TRACK_T_LEN // 7
TRACK_T_X     = [int(TRACK_T_X1+(2*i+1)*TRACK_T_LEN/14) for i in range(7)]
TRACK_T       = [(TRACK_T_X[i], TRACK_T_Y) for i in range(7)]

TRACK_B_X1    = TRACK_LB[0]
TRACK_B_X2    = TRACK_RB[0]
TRACK_B_Y     = TRACK_LB[1]
TRACK_B_LEN   = TRACK_B_X2 - TRACK_B_X1
TRACK_B_BLOCK = TRACK_B_LEN // 7
TRACK_B_X     = [int(TRACK_B_X1+(2*i+1)*TRACK_B_LEN/14) for i in range(7)]
TRACK_B       = [(TRACK_B_X[i], TRACK_B_Y) for i in range(7)]

#============ commands ============#

# 最大触点数
MAX_TOUCH      = 10

SINGLE_PERIOD  = 0.02

FLICK_BIAS     = 0.0
FLICK_PERIOD   = 0.08
FLICK_COUNT    = 5
FLICK_DIS      = 100

DIRECT_BIAS    = 0.0
DIRECT_PERIOD  = 0.05
DIRECT_COUNT   = 7
DIRECT_DIS     = 100

MIDDLE_MIN_GAP = 0.005

LONG_BIAS      = 0.0
LONG_RELEASE   = 0.01
LFLICK_PERIOD  = 0.08
LFLICK_COUNT   = 5
LFLICK_DIS     = 100

SLIDE_BIAS     = -0.01
SLIDE_RELEASE   = 0.01
SFLICK_PERIOD  = 0.08
SFLICK_COUNT   = 5
SFLICK_DIS     = 100

#============ note color & health bar color ============#

HEALTH_POS = [1080, 32]

BLACK      , WHITE       = np.uint8([[  0,   0,   0], [255, 255, 255]])
HEALTH_LOW , HEALTH_HIGH = np.uint8([[ 53, 150, 150], [ 65, 160, 255]]) #
BLUE_LOW   , BLUE_HIGH   = np.uint8([[125,  70, 230], [140, 200, 255]]) #
GREEN_LOW  , GREEN_HIGH  = np.uint8([[ 70,  90, 220], [ 85, 180, 255]]) #
PINK_LOW   , PINK_HIGH   = np.uint8([[150,  80, 230], [170, 200, 255]]) #
YELLOW_LOW , YELLOW_HIGH = np.uint8([[ 20, 120, 230], [ 30, 255, 255]]) #
PURPLE_LOW , PURPLE_HIGH = np.uint8([[100, 150, 250], [110, 255, 255]]) #
ORANGE_LOW , ORANGE_HIGH = np.uint8([[  0,  85, 240], [ 10, 200, 255]]) #

MIN_FILL_RATE   = 0.40
MAX_FILL_RATE   = 0.95
MIN_PIXEL_COUNT = 45
MIN_RATE_T      = 0.025
MAX_RATE_T      = 0.9
EDGE_RATE_T     = 0.7  # 应小于 MAX_RATE_T 至少一帧程度（经验）的 dt
                        # 否则 ExtractFirstNote 任务会失效

TAG_COLOR       = [0, 255, 255]
TAG_RADIUS      = 3

#============ calibration ============#
CORECTION_TIME = 1900000
DILATION_TIME  = 1002000 # DILATION_TIME/1000000