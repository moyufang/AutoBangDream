import cv2 as cv
from configuration import *
from server.player import Player
from utils.log import LogE, LogD, LogI, LogS
import time

class Script:
  def click(self, x, y):
    touch = 0
    self.player.click(touch, x, y)
    return True
  def act(self, state:str):
    if state not in self.state2action:
      LogE(f"\"{state}\" isn't in state2action.")
    else: return self.state2action[state]()
  def __init__(self, player:Player, user_config:UserConfig):
    self.player = player
    self.user_config = user_config
    self.uc = self.user_config
    self.state2action = {
      'award':         lambda : self.click(1080, 640),
      'award_again':   lambda : self.click( 800, 640),
      'award_back':    lambda : self.click( 760, 440),
      'award_dialog':  lambda : (
                                self.click( 640, 640), time.sleep(CLICK_GAP),
                                self.click( 640, 590), time.sleep(CLICK_GAP),
                                self.click( 640, 580), time.sleep(CLICK_GAP),
                                self.click( 640, 560), time.sleep(CLICK_GAP),
                                self.click( 640, 520), time.sleep(CLICK_GAP),
                                self.click( 640, 510),
                                ),
      'award_level':   lambda : self.click(1080, 640),
      'award_loading': lambda: True,
      'award_score':   lambda : self.click(1080, 640),
      'choose':        self._choose,
      'choose_dialog': lambda : (
                                self.click( 876, 208), time.sleep(CLICK_GAP),
                                self.click( 876, 284), time.sleep(CLICK_GAP),
                                self.click( 876, 356), time.sleep(CLICK_GAP),
                                self.click( 876, 430), time.sleep(CLICK_GAP),
                                self.click( 780, 640)
      ),
      'failed':        lambda : self.click( 640, 440),
      'failed_again':  lambda : self.click( 680, 440),
      'join':          lambda : self.click(1080, 640),
      'join_loading':  lambda : True,
      'join_wait':     lambda : True,
      'join_exit':     lambda : self.click(  32,  32),
      'loading':       lambda : True,
      'main_page':     lambda : self.click(1120, 640),
      'opps':          lambda : (
                                self.click( 640, 420),
                                self.click( 640, 430),
                                self.click( 640, 440),
                                self.click( 640, 470),
                                self.click( 640, 480),
                                self.click( 640, 500),
                                ),
      'opps_reconnect':lambda : self.click( 780, 500),
      'playing':       lambda : True,
      'playmode':      self._play_mode,
      'ready':         self._ready,
      'ready_adjust':  lambda : self.click( 780, 500),
      'ready_done':    lambda : True,
      'stage':         lambda : self.click(1080, 640),
      'stage_choose':  lambda : self.click(1080, 640),
      'tour':          lambda : self.click( 440, 450) if self.uc.event_config['lobby'] == 'event' else \
                                self.click( 840, 450),
      'tour_choose':   lambda : self.click(1080, 640),
    }
    
  def _ready(self):
    if self.uc.mode == Mode.Event and self.uc.event == Event.Tour:
      if self.uc.level == 4:
        for i in range(3): self.click(304+i*408, 370)
        time.sleep(CLICK_GAP)
      for i in range(3): self.click(64+self.uc.level*80+i*408, 370)
      time.sleep(CLICK_GAP_3)
    
    # ready 界面有选择难度的情况
    elif not(self.uc.mode == Mode.Free or (self.uc.mode == Mode.Event and self.uc.event == Event.Challenge)):
      if self.uc.level == 4:
        self.click(852, 540)
        time.sleep(CLICK_GAP)
      self.click(600+self.uc.level*84, 540)
      time.sleep(CLICK_GAP_3)
      
    self.click(1120, 600)
    return True
  
  def _play_mode(self):
    if self.uc.mode == Mode.Free: self.click(800, 480)
    elif self.uc.mode == Mode.Collaborate: self.click(1080, 480)
    elif self.uc.mode == Mode.Stage: self.click(920, 600)
    elif self.uc.mode == Mode.Event:
      if self.uc.event in [Event.Mission, Event.Trial]: self.click(1080, 480) 
      elif self.uc.event in [Event.Challenge, Event.Tour]: self.click(800, 240)
      elif self.uc.event in [Event.Compete, Event.Team]: self.click(1080, 240)
    return True
  
  def _choose(self):
    if self.uc.choose == Choose.Loop: pass
    elif self.uc.choose == Choose.ListDowm:
      if self.uc.mode == Mode.Stage: self.click(160, 310)
      else: self.click(380, 420)
    elif self.uc.choose == Choose.ListUp:
      if self.uc.mode == Mode.Stage: self.click(160, 150)
      else: self.click(380, 270)
    elif self.uc.choose == Choose.Random: self.click(780, 540)
    elif self.uc.choose == Choose.No: self.click(780, 610)
    time.sleep(CLICK_GAP)
    
    # 选歌界面有选择难度的情况
    if self.uc.mode == Mode.Free or self.uc.mode == Mode.Stage:
      if self.uc.level == 4:
        self.click(1060, 520)                 # Default diff is 'Expert'
        time.sleep(CLICK_GAP) 
      self.click(715+self.uc.level*115, 520)  # Choose song diff
      time.sleep(CLICK_GAP_3)
    
    self.click(1080, 620)
    return True
        
