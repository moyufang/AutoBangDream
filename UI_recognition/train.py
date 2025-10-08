import time
import os
from PIL import Image
import cv2 as cv
import numpy as np
from glob import glob
import json
import pathlib
import re
import cv2 as cv
import bisect

import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.io as io
import torch.optim as opt

from UI_recognition.BangUINet import *
from utils.log import LogE, LogD, LogI, LogS

def get_batch_size(s:int, n:int):
  if n%s == 0: return s
  for i in range(1, n+1):
    for j in [1, -1]:
      t = s+j*i
      if t < 1 or t > n: continue
      if n%t == 0: return t
  return n

class ImgsDataset(Dataset):
	def __init__(self, model_name):
		self.model_name = model_name
		self.dir_path = './UI_recognition/'+model_name+'_train_imgs/'
		self.classes_num, self.height, self.width = 0, 0, 0
		self.str2label, self.label2str, self.weight, self.data_array, self.datasets = \
			{}, [], [], [], []
		for file in pathlib.Path(self.dir_path).rglob('*.png'):
			print(file._str)
			label_str,count,___ = re.split(r'[-.]', file.name)
			if label_str not in self.str2label:
				self.str2label[label_str] = self.classes_num
				self.label2str.append(label_str)
				self.weight.append(0)
				self.classes_num += 1
			self.weight[self.str2label[label_str]] += 1
		for o in range(self.classes_num):
			self.data_array.extend([(o, i) for i in range(self.weight[o])])
		#print(self.data_array)
		#print(len(self.data_array))
		for label, idx in self.data_array:
			label_str = self.label2str[label]
			img_path = self.dir_path+f'{label_str}-%03d.png'%(idx)
			img = io.read_image(img_path)

			if self.height == 0:
				self.height, self.width = img.size()[1:3]
				
			self.datasets.append(img.to(th.float32))

			# img_cv = cv.cvtColor(cv.imread(img_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
			# img_cv = np.transpose(img_cv, (2, 0, 1))
			# print((img.numpy()-img_cv).max())
			# print(type(img), img.shape)

		self.datasets = th.stack(self.datasets, dim=0)
		self.datasets.requires_grad_(False)
		self.datasets /= 255.0

		with open('./UI_recognition/'+model_name+r"_label2str.json", "w") as file:
			json.dump(self.label2str, file)

		LogI(f"Read image datasets from '%s' with shape: {self.datasets.size()} and classes_num: %d"%(
      self.dir_path, self.classes_num))
	def __len__(self):
		return len(self.data_array)
	def __getitem__(self, idx):
		return self.datasets[idx], self.data_array[idx][0]

def train(nnw, dataset:ImgsDataset, loader, epoches=20, up_labels:list=[]):
	learn_rate = 0.05
	momentum = 0.8
 
 # 用 wt 平衡训练样本中，各类图片数量不均衡的问题
	wt = [o for o in dataset.weight]
	for i in range(len(wt)):
		if dataset.label2str[i] in up_labels: wt[i] = wt[i]*3
	tot = sum(wt)
	for i in range(len(wt)): wt[i] = tot/wt[i]

	Loss = nn.CrossEntropyLoss(th.Tensor(wt))
	SGD = opt.SGD(nnw.parameters(), lr = learn_rate, momentum = momentum)#, weight_decay = weight_decay)

	LogI("Start Training")
	for epoch in range(epoches):
		train_loss, train_acc, test_loss, test_acc = 0, 0, 0, 0
		nnw.train()
		#if epoch >= 5 and train_losses[epoch-1]/(train_losses[epoch-2]) >= 0.99:
		#	SGD.param_groups[0]['lr'] *= 0.1
		if (epoch+1) % 5 == 0:
			th.save(nnw, model_path)
			#SGD.param_groups[0]['lr'] *= 0.1
		start_time = time.time()
		for img, label in loader:
			out = nnw(img)
			loss = Loss(out, label)
			#print(img.size(), label.size(), out.size())
			#exit()

			SGD.zero_grad()
			loss.backward()
			SGD.step()

			train_loss += loss.item()
			# 计算分类的准确率
			_, pred = out.max(1)
			mmin, mmax = _.min(), _.max()
			# LogD(f"mmin: {mmin} mmax: {mmax}")
			num_correct = (pred == label).sum().item()
			train_acc += num_correct / img.shape[0]
		end_time = time.time()
		LogI("OneBatchTrainTime:{}".format(end_time-start_time))
		train_loss /= len(loader)
		train_acc /= len(loader)

		LogI("epoch%d train_loss:%lf train_acc:%lf"%(epoch, train_loss, train_acc))
	
	th.save(nnw, model_path)

#============ train configuration ============#

Net = BangUiNet
model_name = "BangUINet"
batch_size = 64 
epoches = 40
up_labels = ['award', 'award_again', 'ready', 'ready_done'] 
is_load_new_model = False

#============ train ============#
 
dataset = ImgsDataset(model_name)
batch_size = get_batch_size(64, len(dataset))
LogD(f"batch_size:{batch_size}")
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model_path = './UI_recognition/'+model_name+".pth"
nnw = Net(dataset.classes_num)

if is_load_new_model:
  nnw = Net(num_classes = dataset.classes_num, keep_rate=0.2)
else:
  nnw = th.load(model_path, weights_only=False)

n = dataset.datasets.size(0)
weight_acum = dataset.weight.copy()
weight_acum = [0]+weight_acum
for i in range(1, len(weight_acum)):
	weight_acum[i] += weight_acum[i-1]
LogD(f"\n\tweight:{dataset.weight}")
LogD(f"\n\tweight_acum:{weight_acum}")

train(nnw, dataset, loader, epoches, up_labels)

def get_label(img):
	global nnw
	out = nnw(img.unsqueeze(0))
	_, pred = out.max(1)
	return pred.item(), dataset.label2str[pred.item()]

for i in range(n):
	nnw.eval()
	idx = i

	pid = bisect.bisect_left(weight_acum, i+1)

	img, label = dataset[idx]
	pred_label, pred_str = get_label(img)
	if pred_label != label:
		LogD("pred:(%d %s) label:(%d %s-%d)"%(
			pred_label, pred_str, label, dataset.label2str[label], i-weight_acum[pid-1]))

