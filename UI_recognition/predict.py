import json
import torch as th

class BangUI:
  def __init__(self):
    with open("./UI_recognition/BangUINet_train_imgs/BangUINet_label2str.json", "r") as file:
      self.label2str = json.load(file)

    self.model_path = './UI_recognition/BangUiNet.pth'
    self.nnw = th.load(self.model_path)
    self.nnw.eval()
    
  def get_label(self, img):
    out = self.nnw((th.from_numpy(img).float()/255.0).unsqueeze(0))
    _, pred = out.max(1)
    return pred.item(), self.label2str[pred.item()]