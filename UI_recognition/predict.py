import json
import torch as th
import cv2 as cv
import numpy as np

class UIRecognition:
  def __init__(self):
    with open("./UI_recognition/BangUINet_label2str.json", "r") as file:
      self.label2str = json.load(file)

    self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
    self.model_path = './UI_recognition/BangUiNet.pth'
    self.nnw = th.load(self.model_path, weights_only=False, map_location=self.device)
    self.nnw.eval()
  
  def get_state(self, img):
    out = self.nnw((th.from_numpy(img).to(self.device).float()/255.0).unsqueeze(0))
    _, pred = out.max(1)
    return pred.item(), self.label2str[pred.item()]
 
 
