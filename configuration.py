from enum import Enum, auto, IntFlag
import json
import os
import numpy as np
from numpy import random as rd
from utils.EnumRegistry import *

# 端口
MUMU_PORT = 7555
SERVER_PORT = 31415
BANGCHEATER_PORT = 12345
WARPER_PORT = 62358

CONTROLLER_READY_HASH = "BANGCHEATERCONTROLLERREADY"
RECV_TIMEOUT = 0.5
TCP_SEND_GAP = 0.005

#mumu模拟器设置的分辨率
SCALE, STD_WINDOW_WIDTH, STD_WINDOW_HEIGHT = 1, 1280, 720

# 项目文件(夹)地址
SHEETS_HEADER_PATH           = './sheet/sheets_header.json'
UI_RECOGNITION_MODEL_PATH    = './UI_recognition/BangUINet.pth'
SONG_RECOGNITION_MODEL_PATH  = './song_recognition/ckpt_triplet.pth'

LOG_IMGS_PATH                = './UI_recognition/log_imgs/'
UI_IMGS_PATH                 = './UI_recognition/UI_imgs/'
UI_LABEL_2_STR_PATH          = './UI_recognition/UI_label2str.json'
TITLE_IMGS_PATH              = './song_recognition/title_imgs/'
FEATURE_VECTORS_PATH         = './song_recognition/feature_vectors.json'

SHEETS_PATH                  = './sheet/sheets/'
COMMANDS_SHEET_PATH          = './client/commands.sheet'
COMMANDS_JSON_PATH           = './client/commands.json'

SCRIPTOR_CONFIG_PATH         = './module_config/scriptor_config.json'
SONG_RECOGNITION_CONFIG_PATH = './module_config/song_recognition_config.json'
UI_RECOGNITION_CONFIG_PATH   = './module_config/UI_recognition_config.json'
FETCH_CONFIG_PATH            = './module_config/fetch_config.json'
WORKFLOW_CONFIG_PATH         = './module_config/workflow_config.json'
PT_CALCULATOR_CONFIG_PATH    = './module_config/pt_calculator_config.json'

REMOTE_BANGCHEATER_PATH      = '/data/local/tmp/bangcheater'
REMOTE_COMMANDS_PATH         = '/data/local/tmp/commands.sheet'

# 演出模式
@enum_register
class Mode(Enum):
  Free           = auto() # 自由演出
  Collaborate    = auto() # 协力演出
  Stage          = auto() # 舞台挑战
  Event          = auto() # 活动演出(Mission   -> 协力演出
                          #         Trial     -> 协力演出
                          #         Challenge -> 挑战演出
                          #         Tour      -> 巡回演出
                          #         Team      -> 团队竞演
                          #         Compete   -> 竞演演出)
  Story          = auto() # 阅读故事

# 活动种类
@enum_register
class Event(Enum):
  Mission        = auto() # 任务活动
  Trial          = auto() # 试炼活动
  Challenge      = auto() # 挑战活动 
  Tour           = auto() # 巡回演出活动
  Team           = auto() # 团队竞演活动
  Compete        = auto() # 竞演活动

# 选歌模式
@enum_register
class Choose(Enum):
  Loop           = auto() # 单曲循环
  Random         = auto() # 随机选曲
  ListUp         = auto() # 列表循环，向上（目前未支持到头时自动回到底部）
  ListDown       = auto() # 列表循环，向下（目前未支持到底时自动回到顶部）
  No             = auto() # 不指定选曲

# 选曲难度
@enum_register
class Diff(IntFlag):
  Easy           = 0
  Normal         = 1
  Hard           = 2
  Expert         = 3
  Special        = 4      #如果该歌曲不存在 Special 难度，则选择 Expert 难度
  
# 演出水平
@enum_register
class Performance(Enum):
  AllPerfect     = auto() # AP
  FullCombo      = auto() # FC
  DropLastFC     = auto() # 伪FC，Miss最后一个键
  Custom         = auto() # 自定义，选择 Perfect、Great、Good、Bad、Miss 各自的概率
  DropLastCustom = auto() # 自定义，同时 Miss 最后一个键（防 FC）

# 音符判定
@enum_register
class Note(Enum):
  Perfect        = auto()
  Great          = auto()
  Good           = auto()
  Bad            = auto()
  Miss           = auto()

#============ POS ============#

TRACK_LT      = ( 480,  180)
TRACK_RT      = ( 800,  180)
TRACK_LB      = ( 120,  590)
TRACK_RB      = (1160,  590)

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

HEALTH_POS = [1080, 42]

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

#============ script ============#

TCP_SEND_GAP     = 0.01
CLICK_PERIOD     = 0.05
CLICK_GAP        = 0.05
CLICK_GAP_1      = 0.1
CLICK_GAP_2      = 0.2
CLICK_GAP_3      = 0.4
CLICK_GAP_4      = 0.8

# 用于检测是否开启了 3d 演出、mv、3d cut in
COLOR_1_LOW, COLOR_1_HIGH = np.uint8([[155,   6, 240], [170,  12, 255]]) #
COLOR_2_LOW, COLOR_2_HIGH = np.uint8([[  0,   0, 150], [  5,   5, 170]]) #
COLOR_1_POS = [640, 670]
COLOR_2_POS = [496, 640]

#用于检测故事是否读完了
STORY_LOW,     STORY_HIGH = np.uint8([[165, 240, 240], [170, 255, 255]])
STORY_POS   = [256, 200] 

#============ song recognition ============#

# 必须保证这四个区域都是 36 * 450
STD_LEVEL_FIX_TITLE_REGION   = [220, 540, 670, 576]
STD_LEVEL_FIX_LEVEL_REGION   = [220, 576, 670, 612]
STD_LEVEL_UNFIX_TITLE_REGION = [116, 540, 566, 576]
STD_LEVEL_UNFIX_LEVEL_REGION = [116, 576, 566, 612]

MASK_THRESHOLD = 170

#============ main ============#

CYCLE_GAP        = 0.8

#============ Server & Client ============#

class ServerResponse(IntFlag):
  OK = 0
  UNKNOWN = -1
  TIMEOUT_FAILED = -2

