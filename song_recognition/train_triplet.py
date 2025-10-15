import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import time

from song_recognition.TitleNet import TitleNet, SongTitleDataset

class BatchMaker:
  def __init__(self, dataset:SongTitleDataset, t=32, a=4, k=3, is_semi_hard:bool=False):
    self.dataset = dataset
    self.t = t  # 每批类别数
    self.a = a  # 每个类别的增强数
    self.k = k  # 每个anchor的负样本数
    self.is_semi_hard = is_semi_hard
      
  def get_batch(self):
    """生成一个批次的增强数据和三元组标签"""
    # 随机选择t个类别
    selected_indices = random.sample(range(len(self.dataset)), self.t)
    
    batch_images = []
    batch_labels = []
    
    # 对每个选中的类别生成增强数据
    for idx in selected_indices:
      original_img, label = self.dataset[idx]
      
      # 添加原始图片
      batch_images.append(original_img)
      batch_labels.append(label)
      
      # 生成a个增强版本
      for _ in range(self.a):
        augmented_img = self.dataset.augment_image(original_img)
        batch_images.append(augmented_img)
        batch_labels.append(label)
    
    batch_tensor = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    
    assert(len(batch_tensor) == self.t*(1+self.a))
    
    return batch_tensor, batch_labels, selected_indices
  
  def create_triplets(self, features, labels):
    """创建三元组"""
    n = len(features)
    anchor, positive, negative = [], [], []
    # 计算所有样本间的距离矩阵
    
    # 对于每个增强图片作为anchor
    for i in range(0, n, (1+self.a)):  # 前t个是原始图片，后面是增强图片
      # 正样本：同类的原始图片
      original_idx = i
      anchor_label = labels[original_idx]
      
      # 寻找负样本
      negative_candidates = []
      for j in range(n):
        if labels[j] != anchor_label:  # 不同类别
          d_an = F.pairwise_distance(features[i], features[j], p=2)
          negative_candidates.append((j, d_an))
      negative_candidates.sort(key=lambda x: x[1])
      
      for j in range(1, self.a+1):
        anchor_idx = i+j
        assert(labels[anchor_idx] == anchor_label)
        d_ap = F.pairwise_distance(features[original_idx], features[anchor_idx], p=2)
      
        if self.is_semi_hard:      
          # 选择semi-hard负样本
          semi_hard_negatives = []
          hard_negatives = []
          
          for neg_idx, d_an in negative_candidates:
            if d_an > d_ap:  # semi-hard
              semi_hard_negatives.append(neg_idx)
            else:  # hard
              hard_negatives.append(neg_idx)
          
          # 优先选择semi-hard，不够再用hard补足
          selected_negatives = semi_hard_negatives[:self.k]
          if len(selected_negatives) < self.k:
            needed = self.k - len(selected_negatives)
            selected_negatives.extend(hard_negatives[:needed])
        else:
          selected_negatives = [x[0] for x in negative_candidates[:self.k]]
        # 创建三元组
        for neg in selected_negatives:
          anchor.append(features[anchor_idx])
          positive.append(features[original_idx])
          negative.append(features[neg])
    
    return anchor, positive, negative

class TripletLoss(nn.Module):
  def __init__(self, margin=0.2, scale_factor=64.0):
    super(TripletLoss, self).__init__()
    self.margin = margin * scale_factor
    self.scale_factor = scale_factor
    
  def normalize(self, x):
    x = x*self.scale_factor
    return x
      
  def forward(self, anchor, positive, negative):
    anchor = self.normalize(anchor)
    positive = self.normalize(positive)
    negative = self.normalize(negative)
    
    d_ap = torch.pairwise_distance(anchor, positive, p=2)
    d_an = torch.pairwise_distance(anchor, negative, p=2)
    
    losses = torch.relu(d_ap - d_an + self.margin)
    return losses.mean()

def train():
  # 设备配置
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  
  title_imgs_path = './song_recognition/title_imgs/'
  ckpt_path = f'./song_recognition/ckpt_triplet.pth'
  
  # 训练参数
  is_load = True
  num_epochs = 60
  num_batches = 16
  print_batch = num_batches // 4
  batch_classes = 64
  is_semi_hard = False
  learning_rate = 1e-2
  triplet_margin = 0.5
  accumulation_steps = 16  # 梯度累积
  scale_factor = 64.0
  feature_dim = 128
  num_augmented = 4
  k_hard = 3
  backbone_type = 3
  min_loss = 1e6
  
  if is_load: ckpt = torch.load(ckpt_path)
  
  dataset = SongTitleDataset(title_imgs_path)
  batch_maker = BatchMaker(dataset, t=batch_classes, a=num_augmented, k=k_hard, is_semi_hard=is_semi_hard)
  
  # 创建模型
  if not is_load:
    model = TitleNet(feature_dim, backbone_type).to(device)
  else:
    model = TitleNet(ckpt['feature_dim'], ckpt['backbone_type'],pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    min_loss = ckpt['loss']
    min_loss = 1e6
  
  # 创建损失函数
  criterion = TripletLoss(margin=triplet_margin, scale_factor=scale_factor)
  if is_load: criterion.load_state_dict(ckpt['criterion_state_dict'])
    
  #优化器和调度器
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
     
  # 训练循环
  print("开始训练...")
  for epoch in range(num_epochs):
    epoch_time = time.time()
    model.train()
    running_loss = 0.0
      
    for batch_idx in range(num_batches):
      batch_time = time.time()
      # 获取批次数据
      batch_imgs, batch_labels, selected_indices = batch_maker.get_batch()
      batch_imgs = batch_imgs.to(device)
      
      # 前向传播
      features = model(batch_imgs)
      
      # 创建三元组
      triplets = batch_maker.create_triplets(features.cpu(), batch_labels)
      
      if not triplets:continue
      
      anchor = torch.stack(triplets[0]).to(device)
      positive = torch.stack(triplets[1]).to(device)
      negative = torch.stack(triplets[2]).to(device)
      avg_loss = criterion(anchor, positive, negative)
      
      batch_time = time.time()-batch_time
      
      # 梯度累积
      total_loss = avg_loss / accumulation_steps
      total_loss.backward()
      
      if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
      
      if (batch_idx+1)%print_batch == 0:
        print(f"Batch: [{batch_idx}/{num_batches}], Loss: {total_loss.item()*accumulation_steps:.6f}, Batch Time per: {batch_time:.4f}")
      
      running_loss += avg_loss.item()
    
    scheduler.step()
    
    # 打印统计信息
    epoch_time = time.time()-epoch_time
    epoch_loss = running_loss / num_batches
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Epoch Time: {epoch_time}')
    
    # 每1个epoch保存一次模型(如果更优)
    if epoch_loss < min_loss and (epoch + 1) % 1 == 0:
      min_loss = epoch_loss
      torch.save({
        'backbone_type':model.backbone_type,
        'feature_dim': model.feature_dim,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'loss': avg_loss,
      }, ckpt_path)
      print(f'模型已保存: {ckpt_path}')

if __name__ == "__main__":
    train()