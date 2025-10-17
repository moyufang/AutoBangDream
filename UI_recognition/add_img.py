from configuration import *
from utils.Preview import Preview
from pathlib import Path
import cv2 as cv
import re

def load_UI_imgs():
  classes_num, str2label, label2str, weight = 0, {}, [], []
  
  for file in Path(UI_IMGS_PATH).rglob('*.png'):
    # print(file._str)
    label_str,count,___ = re.split(r'[-.]', file.name)
    if label_str not in str2label:
      str2label[label_str] = classes_num
      label2str.append(label_str)
      weight.append(0)
      classes_num += 1
    img_id = weight[str2label[label_str]]
    if int(count) != img_id:
      file.rename(UI_IMGS_PATH+f"{label_str}-%03d.png"%img_id)
    weight[str2label[label_str]] += 1
    
  with open(UI_LABEL_2_STR_PATH, "w") as file:
    json.dump(label2str, file)
  
  return classes_num, str2label, label2str, weight

def start():
  pv = Preview(4)
  classes_num, str2label, label2str, weight = load_UI_imgs()
  for i in range((3+classes_num)//4):
    pt = ''
    for label in range(i*4, min(i*4+4, classes_num)):
      s = label2str[label]
      pt += '%20s:%2d '%(s, label)
    print(pt)
  
  for img_file in Path(LOG_IMGS_PATH).glob('*.png'):
    img = cv.imread(img_file.__str__(), cv.IMREAD_UNCHANGED)
    if img.shape[0] != 160:
      img = cv.resize(img, (160, 90))
      cv.imwrite(img_file.__str__(), img)
    pv.set_title(img_file.name)
    pv.load_img(img, 'bgr')
    pv.show_img()
    
    cv.waitKey(1)
    
    is_to_delete = False
    while True:
      s = input("Please input a lalel:")
      if s[0] == 'q': break
      if s[0] == 'e': exit(0)
      if s[0] == 'd': is_to_delete = True; break
      label = int(s)
      if label not in range(classes_num):
        print("illegal input:", label)
        continue
      else:
        idx = weight[label]
        weight[label] += 1
        
        new_path = UI_IMGS_PATH + f"{label2str[label]}-%03d.png"%idx
        img_file.rename(new_path)
        print(f"Save to \"{new_path}\"")
        break
    if is_to_delete:
      print(f"delete \"{img_file.__str__()}\"")
      img_file.unlink()
      
if __name__ == '__main__':
  start()
    