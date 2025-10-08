import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import random
import cv2
import time

from song_recognition.TitleNet import TitleNet

class SongTitleDataset(Dataset):
  def __init__(self, img_dir, transform=None):
    self.img_dir = Path(img_dir)
    self.transform = transform
    
    # 获取所有t-*.png文件并打乱顺序
    self.img_paths = sorted(list(self.img_dir.glob("t-*.png")))
    random.shuffle(self.img_paths)
    
    # 创建索引到歌曲ID的映射
    self.idx_to_song_id = {}
    for idx, img_path in enumerate(self.img_paths):
      song_id = int(img_path.stem.split('-')[1])
      self.idx_to_song_id[idx] = song_id
        
    print(f"Loaded {len(self.img_paths)} song title images")
        
  def __len__(self):
    return len(self.img_paths)
    
  def __getitem__(self, idx):
    img_path = self.img_paths[idx]
    
    # 读取图片并转换为numpy数组
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
    
    # 白底黑字反转为黑底白字
    img_array = np.array(img).astype(np.float32)
    img_array = 255 - img_array  # 反转
    
    img_array = transforms.ToTensor()(img_array)
        
    return img_array, idx  # 返回图片和索引

class BatchMaker:
  def __init__(self, dataset:SongTitleDataset, t=32, a=4, k=3):
    self.dataset = dataset
    self.t = t  # 每批类别数
    self.a = a  # 每个类别的增强数
    self.k = k  # 每个anchor的负样本数
      
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
        augmented_img = self.augment_image(original_img)
        batch_images.append(augmented_img)
        batch_labels.append(label)
    
    batch_tensor = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    
    assert(len(batch_tensor) == self.t*(1+self.a))
    
    return batch_tensor, batch_labels, selected_indices
  
  def augment_image(self, img_tensor):
    """对单张图片进行增强"""
    # 转换为numpy进行增强操作
    img_np = img_tensor.numpy().squeeze(0) * 255.0
    img_np = img_np.astype(np.uint8)
    
    # 1. 微小平移
    img_a = self.random_shift(img_np)
    
    # 2. 随机噪声
    img_b = self.add_noise(img_a)
    
    # 3. 亮度变化
    img_c = self.brightness_change(img_b)
    
    # 转换回tensor
    img_tensor_aug = torch.from_numpy(img_c.astype(np.float32) / 255.0).unsqueeze(0)
    
    return img_tensor_aug
  
  def random_shift(self, img):
    """随机平移"""
    h, w = img.shape
    dy = random.randint(-3, 3)
    dx = random.randint(-10, 10)
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h), borderValue=0)
    
    return shifted
  
  def add_noise(self, img):
    """添加高斯噪声和椒盐噪声"""
    # 高斯噪声
    gaussian_noise = np.random.normal(0, 255.0 * 0.01, img.shape)
    img_with_gaussian = img.astype(np.float32) + gaussian_noise
    
    # 椒盐噪声
    salt_pepper_prob = 0.01
    salt_mask = np.random.random(img.shape) < salt_pepper_prob / 2
    pepper_mask = np.random.random(img.shape) < salt_pepper_prob / 2
    img_with_gaussian[salt_mask] = 255
    img_with_gaussian[pepper_mask] = 0
    
    # 限制像素范围
    img_noisy = np.clip(img_with_gaussian, 0, 255)
    
    return img_noisy.astype(np.uint8)
  
  def brightness_change(self, img):
    """亮度变化"""
    thresholds = [150, 155, 160, 165, 170]
    threshold = random.choice(thresholds)
    
    # 创建mask：低于threshold的像素设为0，高于的设为255
    mask = (img > threshold).astype(np.uint8) * 255
    
    return mask
  
  def create_triplets(self, features, labels):
    """创建三元组"""
    n = len(features)
    triplets = []
    
    # 计算所有样本间的距离矩阵
    distances = torch.cdist(features, features, p=2)
    
    # 对于每个增强图片作为anchor
    for i in range(0, n, (1+self.a)):  # 前t个是原始图片，后面是增强图片
      # 正样本：同类的原始图片
      original_idx = i
      for j in range(1, self.a+1):
        anchor_idx = i+j
        anchor_label = labels[anchor_idx]
        d_ap = distances[anchor_idx, original_idx]
      
        # 寻找负样本
        negative_candidates = []
        for j in range(n):
          if labels[j] != anchor_label:  # 不同类别
            d_an = distances[i, j]
            negative_candidates.append((j, d_an))
      
        # 按距离排序
        negative_candidates.sort(key=lambda x: x[1])
        
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
        
        # 创建三元组
        for neg_idx in selected_negatives:
          triplets.append((anchor_idx, original_idx, neg_idx))
      
    return triplets

class TripletLoss(nn.Module):
  def __init__(self, margin=0.2, scale_factor=64.0):
    super(TripletLoss, self).__init__()
    self.margin = margin * scale_factor
    self.scale_factor = scale_factor
    
  def normalize(self, x):
    x = F.normalize(x, p=2, dim=1) * self.scale_factor  # L2归一化
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
  is_load = False
  num_epochs = 60
  num_batches = 100
  batch_classes = 32
  learning_rate = 1e-2
  triplet_margin = 0.2
  accumulation_steps = 4  # 梯度累积
  scale_factor = 64.0
  feature_dim = 128
  num_augmented = 4
  k_hard = 3
  backbone_type = 4
  min_loss = 1e6
  
  if is_load: ckpt = torch.load(ckpt_path)
  
  dataset = SongTitleDataset(title_imgs_path)
  batch_maker = BatchMaker(dataset, t=batch_classes, a=num_augmented, k=k_hard)
  
  # 创建模型
  if not is_load:
    model = TitleNet(feature_dim, backbone_type).to(device)
  else:
    model = TitleNet(ckpt['feature_dim'], ckpt['backbone_type'])
    model.load_state_dict(ckpt['model_state_dict'])
    min_loss = ckpt_path['loss']
  
  # 创建损失函数
  criterion = TripletLoss(margin=triplet_margin, scale_factor=scale_factor)
  if is_load: criterion.load_state_dict(ckpt['criterion_state_dict'])
    
  #优化器和调度器
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
     
  # 训练循环
  print("开始训练...")
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    print_batch = num_batches//4
    
    for batch_idx in range(num_batches):
      start_time = time.time()
      # 获取批次数据
      batch_imgs, batch_labels, selected_indices = batch_maker.get_batch()
      batch_imgs = batch_imgs.to(device)
      
      # 前向传播
      features = model(batch_imgs)
      
      # 创建三元组
      triplets = batch_maker.create_triplets(features.cpu(), batch_labels)
      
      if not triplets:continue
      
      anchor, positive, negative = [], [], []
      for anchor_idx, positive_idx, negative_idx in triplets:
        anchor.append(features[anchor_idx])
        positive.append(features[positive_idx])
        negative.append(features[negative_idx])
      anchor, positive, negative = torch.stack(anchor), torch.stack(positive), torch.stack(negative)
      
      avg_loss = criterion(anchor, positive, negative)
      
      batch_time = time.time()-start_time
      
      if len(triplets) > 0:
        # 梯度累积
        total_loss = avg_loss / accumulation_steps
        total_loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
          optimizer.step()
          optimizer.zero_grad()
        
        if (batch_idx+1)%print_batch == 0:
          print(f"Batch: [{batch_idx}/{num_batches}] Loss: {total_loss.item()*accumulation_steps:.6f} Batch Time per: {batch_time:.4f}")
        
        running_loss += avg_loss.item()
    
    scheduler.step()
    
    # 打印统计信息
    epoch_loss = running_loss / num_batches
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    
    # 每1个epoch保存一次模型(如果更优)
    if avg_loss < min_loss and (epoch + 1) % 1 == 0:
      min_loss = avg_loss
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