from configuration import *
from utils.EnumRegistry import *
import json 

class CustomPerformance:
  def __init__(self):
    # (title,      [perfect,great,good , bad  , miss ])
    self.weights_map = {
      "blind":    [0.450, 0.300, 0.050, 0.050, 0.150],
      'hell':     [0.600, 0.200, 0.050, 0.050, 0.100],
      'fool':     [0.700, 0.200, 0.050, 0.030, 0.020],
      'newbee':   [0.850, 0.095, 0.005, 0.015, 0.035],
      'skilled':  [0.940, 0.050, 0.002, 0.003, 0.005],
      'master':   [0.960, 0.038, 0.000, 0.001, 0.001],
      'top':      [0.980, 0.018, 0.000, 0.001, 0.001],
      'nohuman':  [0.990, 0.008, 0.000, 0.001, 0.001],
      'newworld': [0.995, 0.004, 0.000, 0.000, 0.001],
      'god':      [0.100, 0.000, 0.000, 0.000, 0.000],
    }
    # (title, level_num)
    self.level_ladder = [
      ["blind",    32],
      ["hell",     30],
      ["fool",     29],
      ["newbee",   28],
      ["skilled",  27],
      ["master",   26],
      ["top",      25],
      ["nohuman",  23],
      ["newworld", 18],
      ["god",       0],
    ]
    
  # 根据难度选择 title, 进而根据 title 选择 weights
  def get_weights_by_level_ladder(self, level:int):
    for item in self.level_ladder:
      if level < item[1]: continue
      else: return self.weights_map[item[0]]
    return self.weights_map['blind']
  
  def set_level_ladder(self, level_ladder:list=[]):
    if not level_ladder: return False
    for item in level_ladder:
      assert(isinstance(item, tuple) and isinstance(item[0], str) and isinstance(item[1], int))
    self.level_ladder = level_ladder.copy()
    return True
  
  def set_weights_map(self, weights_map:dict):
    self.weights_map = weights_map.copy()
  
  def add_weights(self, key:str, weights:list):
    assert(isinstance(key, str))
    assert(len(weights, 5))
    sum = 0.0
    for i in weights: assert(isinstance(weights, float)); sum += i
    EPS = 1e-6
    assert(sum > 1.0-EPS and sum < 1.0+EPS)
    assert(key != '')
    self.weights_map[key] = weights.copy()
    
  def del_weights(self, key:str):
    if key not in self.weights_map: return False
    self.weights_map.__delitem__(key)
    return True

# 音符偏移器
# 根据 weights 和 performance，确定每个 note 是否产生时移，以及产生时移的幅度
# 关键函数 get_skew 用 _get_skew 实现了多态，便于后续更新策略
class NoteSkewer:
  def __init__(self,
               bias=[0.000,0.032,0.048,0.064,1.000],
               weights:list=None,
               performance:Performance=Performance.AllPerfect):
    self.set_bias(bias)
    self.set_performance(performance)
    self.set_weights(weights)
    
  def set_performance(self, performance:Performance):
    self.performance = performance
    if hasattr(self, 'ori_weights'): self.create_skewer()
    
  def set_weights(self, weights:list=None):
    if weights is not None:
      assert((len(weights) == 5 and sum([int(i<0.0) for i in weights]) == 0))
      self.ori_weights = weights.copy()
    else: self.ori_weights = None
    if hasattr(self, 'performance'): self.create_skewer()
    
  def set_bias(self, bias:list[int]=[0,0.032,0.048,0.064,1.0]):
    self.bias = [[0,0]for i in range(5)]
    for (i, v) in enumerate(bias):
      self.bias[i] = [-v, v]
      
  def create_skewer(self):
    weights = self.ori_weights.copy() if self.ori_weights is not None else None
    
    if self.performance == Performance.AllPerfect or weights == None:
      self._get_skew = lambda : 0.0
    else:
      if self.performance == Performance.FullCombo:
        for i in [2,3,4]: weights[i] = 0.0
      for i in range(1, len(weights)): weights[i] += weights[i-1]
      self._get_skew = lambda : rd.choice(self.bias[self.get_note()])
    self.weights = weights
    return True
  
  def get_note(self):
    weights = self.weights
    x = rd.random() * weights[-1]
    if x <= weights[0]: return 0   # Perfect
    elif x <= weights[1]: return 1 # Great
    elif x <= weights[2]: return 2 # Good
    elif x <= weights[3]: return 3 # Bad
    else: return 4                 # Miss
    
  def get_skew(self):
    return float(self._get_skew())
  
from module_config.config_manager import *

# @config_register('scriptor', SCRIPTOR_CONFIG_PATH)
class ScriptorConfig(Config, CustomPerformance, NoteSkewer):
  """
  在 scriptor.py 中，ScriptorConfig 中有部分属性需要set函数来维持一致性
  set_weights_map, add_weights, del_weights, set_level_ladder,
  set_bias,set_weights, set_performance
  """
  def __init__(self, config_path:str):
    Config.__init__(self, config_path)
  def set_config(
      self,
      mode:Mode = None,
      event:Event = None,
      choose:Choose = None,
      diff:Diff = None,
      performance:Performance = None,
      custom_performance:str = None,
      additional_config:dict = None, # 特殊活动特殊模式特殊考虑
    ):
    self.mode = mode if mode is not None else Mode.Free
    self.event = event if event is not None else Event.Mission
    self.choose = choose if choose is not None else Choose.Loop
    self.diff = diff if diff is not None else Diff.Expert
    self.additional_config = additional_config if additional_config is not None else {}
    self.custom_performance = custom_performance if custom_performance is not None else 'god'
    CustomPerformance.__init__(self)
    NoteSkewer.__init__(self,
      performance=performance if performance is not None else Performance.AllPerfect,
      weights=self.weights_map[self.custom_performance]
    )

    self.run_config = {
      "correction_time"    : -  40000,
      "target_diff_time"   :    55000,
      "dilation_time"      :  1000000,

      "is_no_action"       : False,
      "is_caliboration"    : False,

      "play_one_song_id"   : 316,
      "is_play_one_song"   : False,
      "is_restart_play"    : True,

      "is_checking_3d"     : True,

      "MAX_SAME_STATE"     : 100,
      "MAX_RE_READY"       : 5,

      "is_allow_save"      : True,
      "is_allow_suspend"   : True,
      
      "protected_state"    : ['join_wait', 'ready_done'], 
      "record_state"       : ['award'],
    }
    
  def get_is_fix(self)->bool:
    return self.mode in [Mode.Free,Mode.Stage] or (self.mode == Mode.Event and self.event in [Event.Challenge, Event.Tour])
  def get_is_multiplayer(self):
    return self.mode == Mode.Collaborate or (self.mode == Mode.Event and self.event in [Event.Compete, Event.Team, Event.Mission, Event.Trial])
  def save(self):
    self.cfg = {
      'mode': self.mode,
      'event': self.event,
      'choose': self.choose,
      'diff': self.diff,
      'performance': self.performance,
      'additional_config': self.additional_config,
      'custom_performance': self.custom_performance,
      'bias': self.bias,
      'weights_map': self.weights_map,
      'level_ladder': self.level_ladder,
      'run_config': self.run_config,
    }
    Config.save(self)
    del self.cfg
  def load(self):
    Config.load(self)
    for (k,v) in self.cfg.items(): self.__setattr__(k, v)
    del self.cfg
    self.set_weights(self.weights_map[self.custom_performance])
  def update(self, new_cfg:dict, note:dict={}):
    for k,v in new_cfg.items():
      if hasattr(self, k):
        self.__setattr__(k, v)
    if 'add_weights' in note:
      for (k,v) in note['add_weights']:
        try: self.add_weights(k, v)
        except Exception: pass
    if 'del_weights' in note:
      for (k,v) in note['del_weights']:
        self.del_weights(k, v)
    if 'custom_performance' in note:
      if note['custom_performance'] in self.weights_map:
        self.set_weights(self.weights_map[note['custom_performance']])
    if 'run_config' in note:
      self.run_config.update(note['run_config'])
    self.save()
    
if __name__ == '__main__':
  scfg = ScriptorConfig(SCRIPTOR_CONFIG_PATH)
  scfg.set_config()
  scfg.save()
  
  from utils.json_refiner import refine
  refine(scfg.config_path, max_depth=[2, 2, 0])