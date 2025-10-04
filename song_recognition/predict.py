import cv2 as cv
import numpy as np
from enum import Enum, auto
from configuration import *

class SongRecognition:
  class ReadyType(Enum):
    LevelFix = auto()
    LevelUnfix = auto()
  def __init__(self, user_config:UserConfig):
    self.user_config = user_config
  def get_song(self, img):
    uc = self.user_config
    if uc.mode == Mode.Collaborate or (uc.mode == Mode.Event and Event.Tour):
      ready_type = SongRecognition.ReadyType.LevelUnfix
    else:
      ready_type = SongRecognition.ReadyType.LevelFix
    
    song_id = 306
    pred_song_name = 'saviorofsong'
    
    return song_id, pred_song_name