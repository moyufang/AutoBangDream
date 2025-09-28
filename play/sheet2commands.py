from configuration import *
from numpy import random as rd

# 根据 weights 和 performance，确定每个 note 是否产生时移，以及产生时移的幅度
# 关键函数 get_skew 用 _get_skew 实现了多态，便于后续更新策略
class NoteSkewer:
  def __init__(self, weights:list=None, performance:Performance=Performance.AllPerfect):
    self.performance2bias = {
      Note.Perfect : [0.0],
      Note.Great   : [-0.032, 0.032],
      Note.Great   : [-0.048, 0.048],
      Note.Bad     : [-0.064, 0.064],
      Note.Miss    : [-1.000, 1.000]
    }
    self.weights = weights.copy()
    self.performance = performance
    
    if self.performance == Performance.AllPerfect or weights == None:
      self._get_skew = lambda : 0.0
    else:
      assert(len(weights) == 5 and sum([int(i<0.0) for i in weights]) == 0)
      if self.performance == Performance.FullCombo:
        for i in [2,3,4]: self.weights[i] = 0.0
      for i in range(1, len(weights)): weights[i] += weights[i-1]
      self._get_skew = lambda : rd.choice(self.performance2bias[self.get_note()])
  def get_note(self):
    weights = self.weights
    x = rd.random() * weights[-1]
    if x <= weights[0]: return Note.Perfect
    elif x <= weights[1]: return Note.Great
    elif x <= weights[2]: return Note.Good
    elif x <= weights[3]: return Note.Bad
    else: return Note.Miss
  def get_skew(self):
    return float(self._get_skew())

# 返回操作序列，其中 timestamp 递增
# commands = [
#   [timestamp, "{action_type} {touch_id} {x} {y}"],
#   ...
# ]

def sheet2commands(file_path:str, note_skewer:NoteSkewer = None):
  if note_skewer == None: note_skewer = NoteSkewer()
  commands = []
  
  avail = [-1 for i in range(MAX_TOUCH)]
  
  with open(file_path, "r", encoding='utf-8') as file:
    raw_data = json.load(file)
  if note_skewer.performance == Performance.DropLastCustom:
    big_delay = 5
    if raw_data[-1]['type'] in ['Single', 'Directional']:
      raw_data[-1]['beat'] += big_delay 
    elif raw_data[-1]['type'] in ['Long', 'Slide']:
      raw_data[-1]['connections'][-1]['beat'] += big_delay
  
  bpm, base_t, t, base_b, b = 1024*1024*1024*1024*1024*1024, 0.0, 0.0, 0.0, 0.0
  first_t = -1.0
  for k in range(len(raw_data)):
    item = raw_data[k]; ty = item['type']
    if ty == "System":
      continue
    if ty == "BPM":
      b = float(item['beat'])
      t = base_t + (b-base_b)*60/bpm
      bpm, base_t, base_b = float(item['bpm']), t, b
      continue
    
    if ty in ["Slide", "Long"]:
      seq = item['connections']
      l = int(seq[0]['lane'])
      b = float(seq[0]['beat'])
    else:
      l = int(item['lane'])
      b = float(item['beat'])
    t = base_t + (b-base_b)*60/bpm
    
    if ty == "BPM":
      bpm, base_t, base_b = float(item['bpm']), t, b
      continue
    
    if first_t < 0.0: first_t = t
    
    # pick idle touch
    touch = -1
    for i in range(len(avail)):
      if t > avail[i]: avail[i] = -1; touch = i; break
    assert(touch != -1)
    
    if ty == "Single" and "flick" in item and item["flick"]:
      t += note_skewer.get_skew()
      release_t = t+FLICK_PERIOD
      commands.append([t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      for i in range(1, FLICK_COUNT):
        commands.append([t+i*FLICK_PERIOD/FLICK_COUNT,
          f"m {touch} {TRACK_B_X[l]} {TRACK_B_Y+FLICK_DIS*i/(FLICK_COUNT-1)}"])
      commands.append([release_t, f"u {touch}"])
        
    elif ty == "Single":
      t += note_skewer.get_skew()
      release_t = t+SINGLE_PERIOD
      commands.append([t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      commands.append([release_t, f"u {touch}"])
        
    elif ty == "Directional":
      t += note_skewer.get_skew()
      sgn = 1 if item['direction'] == 'right' else -1 
      release_t = t+DIRECT_PERIOD
      commands.append([t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      for i in range(1, DIRECT_COUNT):
        commands.append([t+i*DIRECT_PERIOD/DIRECT_COUNT,
          f"m {touch} {TRACK_B_X[l]+sgn*i*DIRECT_DIS/(DIRECT_COUNT-1)} {TRACK_B_Y}"])
      commands.append([release_t, f"u {touch}"])
      
    elif ty == "Long":
      record_t = t+note_skewer.get_skew() 
      release_t = max(record_t+MIDDLE_MIN_GAP, t+(seq[-1]['beat']-b)*60/bpm + note_skewer.get_skew())
      commands.append([record_t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      if 'flick' in seq[-1] and seq[-1]['flick']:
        for i in range(1, len(LFLICK_COUNT)):
          commands.append([
            release_t+i*LFLICK_PERIOD/LFLICK_COUNT,
            f"m {touch} {TRACK_B_X[l]+i*LFLICK_DIS/(LFLICK_COUNT-1)} {TRACK_B_Y}"
          ])
        release_t += LFLICK_PERIOD
        commands.append([release_t, f"u {touch}"])
      else:
        commands.append([release_t, f"u {touch}"])
        
    elif ty == "Slide":
      record_t = t+note_skewer.get_skew()
      release_t = t+(seq[-1]['beat']-b)*60/bpm
      commands.append([record_t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      for i in range(1, len(seq)-1):
        seq_item = seq[i]
        record_t = max(record_t+MIDDLE_MIN_GAP, t + (seq_item['beat']-b)*60/bpm + note_skewer.get_skew())
        commands.append([
          record_t,
          f"m {touch} {TRACK_B_X[int(seq_item['lane'])]} {TRACK_B_Y}"
        ])
      release_t = max(record_t+MIDDLE_MIN_GAP, t+(seq[-1]['beat']-b)*60/bpm + note_skewer.get_skew())
      if 'flick' in seq[-1]  and seq[-1]['flick']:
        for i in range(1, SFLICK_COUNT):
          commands.append([
            release_t + i*SFLICK_PERIOD/SFLICK_COUNT,
            f"m {touch} {TRACK_B_X[int(seq[-1]['lane'])]} {TRACK_B_Y + int(i*SFLICK_DIS/(SFLICK_COUNT-1))}"
          ])
        release_t = release_t + SFLICK_PERIOD
        commands.append([release_t, f"u {touch}"])
      else:
        commands.append([release_t, f"u {touch}"])
    else:
      print(f"Unknown type of note with ty:{ty} item:{item}\n")
    avail[touch] = release_t
  commands.sort(key=lambda item:item[0])
  return commands
      
if __name__ == '__main__':
  file_path = './play/fetch_one_sheet_295_4.bestdori'
  custom_performance = CustomPerformance()
  note_skewer = NoteSkewer(custom_performance.weights_map['newbee'], Performance.DropLastCustom)
  commands = sheet2commands(file_path, note_skewer)
  
  commands_file_path = "./play/commands.json"
  with open(commands_file_path, "w", encoding="utf-8") as file:
    json.dump(commands, file)
  from utils.json_refiner import refine
  refine(commands_file_path)
  
  
  
      