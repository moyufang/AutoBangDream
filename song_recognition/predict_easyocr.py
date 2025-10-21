import cv2 as cv
import numpy as np
from enum import Enum, auto
import json
import easyocr
from configuration import *
from sheet.fetch import shave_str, special_char
from configuration import *
import torch as th

sheets_header_path = './sheet/sheets_header.json'

def get_edit_dis(s1, s2, eo=-1):
  l1, l2 = len(s1), len(s2)
  if l1 > l2: l1,l2 = l2,l1; s1,s2=s2,s1
  if eo < 0: eo = max(l1, l2)
  f = [min(eo,i) for i in range(l2+1)]
  for i in range(1, 1+l1):
    c = s1[i-1]
    for j in range(min(i+eo-1, l2), max(0, i-eo), -1):
      t = f[j-1]+int(c!=s2[j-1])
      f[j] = t if t <= f[j] else f[j]+1
    if i <= eo: f[0] += 1
    for j in range(1+max(0,i-eo), min(i+eo, l2+1)):
      f[j] = min(f[j], f[j-1]+1)
  return f[l2]

class SongRecognition:
  def __init__(self, sheets_header_path:str):
    with open(sheets_header_path, "r", encoding='utf-8') as file:
      self.sheets_header = json.load(file)
      
    self.title_map = {}
    for id in self.sheets_header:
      o = self.sheets_header[id]
      assert(isinstance(o[0], int))
      for c in o[1]:
        if c not in self.title_map:
          self.title_map[c] = {o[0]:1}
        else:
          if o[0] not in self.title_map[c]: self.title_map[c][o[0]] = 1
          else: self.title_map[c][o[0]] += 1
    
    self.reader = easyocr.Reader(['ja', 'en'], gpu=th.cuda.is_available())
  def patch(self, s):
    char_cnt = {}
    id_cnt = {}
    for c in s:
      if c not in char_cnt: char_cnt[c] = 1
      else: char_cnt[c] += 1
    for c in char_cnt:
      if c not in self.title_map: continue
      char_map = self.title_map[c]
      for idx in char_map:
        w = 1.0 if c.islower() else 1.0 
        # 每种字符的得分，根据数差递减
        t = w/(1+abs(char_map[idx] - char_cnt[c]))**2 
        if idx not in id_cnt: id_cnt[idx] = t
        else: id_cnt[idx] += t
    tar_id, score = -1, 0.0
    l = len(s)
    for idx in id_cnt:
      # 字符串长度得分
      t = id_cnt[idx]+1.0/(1+abs(l-len(self.sheets_header[str(idx)][1])))
      if t > score: tar_id, score = idx, t
    if tar_id != -1:
      # 预选目标，之后如果得分小于 score*0.8 的，都抛弃（即剪枝掉）
      eo = get_edit_dis(s, self.sheets_header[str(tar_id)][1])
    for idx in id_cnt:
      t = id_cnt[idx]+2.0/(1+abs(l-len(self.sheets_header[str(idx)][1])))
      if t > score*0.8 or l <= 4:
        eo_tmp = get_edit_dis(s, self.sheets_header[str(idx)][1], eo)
        if eo_tmp < eo: eo = eo_tmp; tar_id = idx
        # print("id:%d title:%s score:%lf eo:%d"%(
        # 	idx, self.sheets_header[idx][1], t, eo_tmp))
    if tar_id != -1:
      score = id_cnt[tar_id]+2.0/(1+abs(l-len(self.sheets_header[str(tar_id)][1])))
    #print("tar_id:%d title:%s score:%lf"%(tar_id, self.sheets_header[tar_id][1], score))
    return tar_id
  def parse_t_img(self, t_img):
    
    t_mask = (t_img < MASK_THRESHOLD).astype(np.uint8)*255
    t_recog = self.reader.readtext(t_mask, detail = 0, paragraph=True, text_threshold=0.8)
    pattern = shave_str("".join(t_recog))
    pre_idx = self.patch(pattern)

    print("[",pattern, pre_idx, self.sheets_header[str(pre_idx)][1], "]")
    if len(pattern) == 1 or pre_idx in [-1, 389, 410, 462, 467]:
      print(f"Unsafe pre_idx : {pre_idx} raw_str:{t_recog}")
      is_safe = False
    else: is_safe = True
    return pre_idx, self.sheets_header[str(pre_idx)], pattern, is_safe
  
  def parse_l_img(self, l_img):
    l_mask = (l_img < MASK_THRESHOLD).astype(np.uint8)*255
    l_recog = self.reader.readtext(l_mask, detail = 0, paragraph=True, text_threshold=0.8)
    pattern = shave_str("".join(l_recog))
    return pattern

if __name__ == '__main__':
  recognition = SongRecognition(SHEETS_HEADER_PATH)
  
  