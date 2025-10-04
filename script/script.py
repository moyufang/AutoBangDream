import cv2 as cv
from configuration import *
from play import Player
from utils.log import LogE, LogD, LogI, LogS
import time

class Script:
  def config(
    self,
    mode:Mode = None,
    event:Event = None,
    choose:Choose = None,
    level:Level = None
    ):
    self.mode = mode if mode is not None else Mode.Free
    self.event = event if event is not None else Event.Mission
    self.choose = choose if choose is not None else Choose.Loop
    self.level = level if level is not None else Level.Expert
    
    # performance:Performance = None
    # custom_performance:CustomPerformance = None
    # self.performance = performance if performance is not None else Performance.AllPerfect
    # self.custom_performance = custom_performance if custom_performance is not None else CustomPerformance
  def click(self, x, y):
    touch = 0
    self.player.send_cmd(f'd {touch} {x} {y}\nc\n')
    time.sleep(CLICK_PERIOD)
    self.player.send_cmd(f'u {touch}')
    return True
  def act(self, state:str):
    if state not in self.state2action:
      LogE(f"\"{state}\" is 't in state2actio.")
    else: return self.state2action[state]()
  def __init__(self, player:Player):
    self.player = player
    self.state2action = {
      'award': lambda : self.click(1080, 620),
      'award_again': lambda : self.click(800, 620),
      'award_back': lambda : self.click(760, 440),
      'award_dialog': lambda: (
        self.click(640, 620),
        self.click(640, 590),
        self.click(640, 580),
        self.click(640, 560),
        self.click(640, 520),
        self.click(640, 510),
      ),
      'award_level': lambda : self.click(1080, 620),
      'award_loading': lambda: True,
      'award_score': lambda : self.click(1080, 620),
      'choose': self._choose,
      'failed': lambda : self.click(640,  440),
      'failed_again': lambda: self.click(680, 440),
      'join': lambda : self.click(1080, 620),
      'join_loading': lambda : True,
      'join_wait': lambda: True,
      'loading': lambda: True,
      'main_page': lambda : self.click(1200, 640),
      'opps': lambda : (
        self.click(640, 420),
        self.click(640, 430),
        self.click(640, 440),
        self.click(640, 470),
        self.click(640, 480),
        self.click(640, 500),
      ),
      'opps_reconnect': lambda : self.click(780, 500),
      'playing': lambda: True,
      'play_mode': self._play_mode,
      'ready': self._ready,
      'ready_adjust': lambda: self.click(780, 500),
      'ready_done': lambda: True,
      'stage': lambda: self.click(1080, 620),
      'tour': lambda: self.click(440, 450) if self.event_msg['lobby'] == 'event' else self.click(840, 450),
      'tour_choose': lambda: self.click(1080, 620),
    }
  def _ready(self):
    if self.mode == Mode.Event and self.event == Event.Tour:
      for i in range(3): self.click(304+i*408, 370)
      time.sleep(CLICK_GAP)
      for i in range(3): self.click(64+self.level*80+i*408, 370)
      time.sleep(CLICK_GAP)
    
    # ready 界面有选择难度的情况
    elif not(self.mode == Mode.Free or (self.mode == Mode.Event and self.event == Event.Challenge)):
      self.click(852, 540)
      time.sleep(CLICK_GAP)
      self.click(600+self.level*84, 540)
      time.sleep(CLICK_GAP)
      
      time.sleep(INCASE_DELAY_GAP)
      
      self.click(852, 540)
      time.sleep(CLICK_GAP)
      self.click(600+self.level*84, 540)
      time.sleep(CLICK_GAP)
    
    self.click(1120, 600)
    return True
  def _play_mode(self):
    if self.mode == Mode.Free: self.click(800, 480)
    elif self.mode == Mode.Collaborate: self.click(1080, 480)
    elif self.mode == Mode.Stage: self.click(920, 600)
    elif self.mode == Mode.Event:
      if self.event in [Event.Mission, Event.Trial]: self.click(1080, 480) 
      elif self.event in [Event.Challenge, Event.Tour]: self.click(800, 240)
      elif self.event in [Event.Compete, Event.Team]: self.click(1080, 240)
    return True
  def _choose(self):
    
    if self.choose == Choose.Loop: pass
    elif self.choose == Choose.ListDowm:
      if self.mode == Mode.Stage: self.click(160, 310)
      else: self.click(380, 420)
    elif self.choose == Choose.ListUp:
      if self.mode == Mode.Stage: self.click(160, 150)
      else: self.click(380, 270)
    elif self.choose == Choose.Random: self.click(780, 540)
    elif self.choose == Choose.No: self.click(780, 610)
    time.sleep(CLICK_GAP)
    
    # 选歌界面有选择难度的情况
    if self.mode == Mode.Free or self.mode == Mode.Stage:
      self.click(944, 520)                 # Default diff is 'Expert'
      time.sleep(CLICK_GAP) 
      self.click(712+self.level*116, 520)  # Choose song diff
      time.sleep(CLICK_GAP)
    
    self.click(1080, 620)
    return True
        
  
  
  
    