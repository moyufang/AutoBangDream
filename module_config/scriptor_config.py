from configuration import *
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
      'master':   [0.960, 0.035, 0.001, 0.001, 0.003],
      'top':      [0.980, 0.017, 0.001, 0.001, 0.001],
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
    if hasattr(self, 'ori_weight'): self.create_skewer()
    
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
  
class RunConfig():
  def __init__(self):
    self.run_config = {
      "dilation_time"      :  1000000,
      "correction_time"    : -  40000,

      "is_no_action"       : False,
      "is_caliboration"    : False,

      "play_one_song_id"   : 316,
      "is_play_one_song"   : False,
      "is_restart_play"    : True,

      "is_checking_3d"     : True,

      "is_repeat"          : True,
      "MAX_SAME_STATE"     : 100,
      "MAX_RE_READY"       : 5,

      "is_allow_save"      :True,
    }

class ScriptorConfig(CustomPerformance, NoteSkewer, RunConfig):
  """
  在 scriptor.py 中，ScriptorConfig 中有部分属性需要set函数来维持一致性
  set_weights_map, add_weights, del_weights, set_level_ladder,
  set_bias,set_weights, set_performance
  """
  def __init__(self, *arg):
    self.set_config(*arg)
  def set_config(
      self,
      mode:Mode = None,
      event:Event = None,
      choose:Choose = None,
      diff:Diff = None,
      performance:Performance = None,
      event_config:dict = None, # 特殊活动特殊考虑
    ):
    self.mode = mode if mode is not None else Mode.Free
    self.event = event if event is not None else Event.Mission
    self.choose = choose if choose is not None else Choose.Loop
    self.diff = diff if diff is not None else Diff.Expert
    self.event_config = event_config if event_config is not None else {}
    CustomPerformance.__init__(self)
    NoteSkewer.__init__(self, performance=performance if performance is not None else Performance.AllPerfect)
    RunConfig.__init__(self)
  def save(self, config_path:str):
    cfg = {
      'mode': f'Mode.{self.mode.name}',
      'event': f'Event.{self.event.name}',
      'choose': f'Choose.{self.choose.name}',
      'diff': f'Diff.{self.diff.name}',
      'performance': f'Performance.{self.performance.name}',
      'event_config': self.event_config,
      'bias': self.bias,
      'weights_map': self.weights_map,
      'level_ladder': self.level_ladder,
      'weights': self.ori_weights,
      'run_config': self.run_config,
    }
    with open(config_path, "w", encoding='utf-8') as file:
      json.dump(cfg, file)
  def load(self, config_path:str):
    with open(config_path, "r", encoding='utf-8') as file:
      cfg = json.load(file)
    for (k,v) in cfg.items():
      if k in ['mode', 'event', 'diff', 'performance']:
        e,v = v.split('.')
        self.__setattr__(k, str2enum[e][v])
      else:
        self.__setattr__(k, v)