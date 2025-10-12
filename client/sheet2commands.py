from configuration import *
from server.ADB import push_file

# 生成操作序列commands和保存为commands.sheet文件
# commands = [
#   [timestamp, "{action_type} {touch_id} {x} {y}"],
#   ...
# ]
# 其中 timestamp 递增
def sheet2commands(file_path:str, commands_path:str='./commands.sheet', note_skewer:NoteSkewer = None):
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
          f"m {touch} {TRACK_B_X[l]} {TRACK_B_Y-int(FLICK_DIS*i/(FLICK_COUNT-1))}"])
      commands.append([release_t, f"u {touch}"])
        
    elif ty == "Single":
      t += note_skewer.get_skew()
      release_t = t+SINGLE_PERIOD
      commands.append([t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      commands.append([release_t, f"u {touch}"])
        
    elif ty == "Directional":
      t += note_skewer.get_skew()
      sgn = 1 if item['direction'] == 'Right' else -1 
      release_t = t+DIRECT_PERIOD
      commands.append([t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      for i in range(1, DIRECT_COUNT):
        commands.append([t+i*DIRECT_PERIOD/DIRECT_COUNT,
          f"m {touch} {TRACK_B_X[l]+sgn*int(i*DIRECT_DIS/(DIRECT_COUNT-1))} {TRACK_B_Y}"])
      commands.append([release_t, f"u {touch}"])
      
    elif ty == "Long":
      record_t = t+note_skewer.get_skew() 
      release_t = max(record_t+MIDDLE_MIN_GAP, t+(seq[-1]['beat']-b)*60/bpm + note_skewer.get_skew())
      commands.append([record_t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      if 'flick' in seq[-1] and seq[-1]['flick']:
        for i in range(1, LFLICK_COUNT):
          commands.append([
            release_t+i*LFLICK_PERIOD/LFLICK_COUNT,
            f"m {touch} {TRACK_B_X[l]-int(i*LFLICK_DIS/(LFLICK_COUNT-1))} {TRACK_B_Y}"
          ])
        release_t += LFLICK_PERIOD
        commands.append([release_t, f"u {touch}"])
      else:
        release_t += LONG_RELEASE
        commands.append([release_t, f"u {touch}"])
        
    elif ty == "Slide":
      record_t = t+note_skewer.get_skew()
      release_t = t+(seq[-1]['beat']-b)*60/bpm
      commands.append([record_t, f"d {touch} {TRACK_B_X[l]} {TRACK_B_Y}"])
      for i in range(1, len(seq)):
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
            f"m {touch} {TRACK_B_X[int(seq[-1]['lane'])]} {TRACK_B_Y - int(i*SFLICK_DIS/(SFLICK_COUNT-1))}"
          ])
        release_t = release_t + SFLICK_PERIOD
        commands.append([release_t, f"u {touch}"])
      else:
        release_t += SLIDE_RELEASE
        commands.append([release_t, f"u {touch}"])
    else:
      print(f"Unknown type of note with ty:{ty} item:{item}\n")
    avail[touch] = release_t
  commands.sort(key=lambda item:item[0])
  
  song_duration = commands[-1][0]
  
  # print(commands_path)
  with open(commands_path, "w", encoding='utf-8') as file:
    file.write(f"b %d\n"%(int(1000000*commands[0][0]+0.5)))
    cur_t = -1
    for t, cmd in commands:
      t = int(float(t)*1000000+0.5)
      if cur_t >= 0 and t > cur_t:
        file.write("t %d\n"%(cur_t))
      cur_t = t
      file.write(cmd+'\n')
    file.write("t %d\n"%(cur_t))
    
  return commands, song_duration

# 306 Savior Of Songs
# 85 ハッピーシンセサイザ

if __name__ == '__main__':
  file_path = './sheet/sheets/689_4.bestdori'#'./play/sample.bestdori'#'./play/fetch_one_sheet_295_4.bestdori'
  commands_path = './client/commands.sheet'
  custom_performance = CustomPerformance()
  note_skewer = None#NoteSkewer(custom_performance.weights_map['newbee'], Performance.DropLastCustom)
  commands = sheet2commands(file_path, commands_path, note_skewer)
  
  commands_file_path = "./client/commands.json"
  with open(commands_file_path, "w", encoding="utf-8") as file:
    json.dump(commands, file)
  from utils.json_refiner import refine
  refine(commands_file_path)
  
  if True:
    push_file(commands_path)
  
  
  
      