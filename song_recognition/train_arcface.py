import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import random
import time

from song_recognition.TitleNet import TitleNet

class SongTitleDataset(Dataset):
    def __init__(self, img_dir, img_size=(36, 450)):
        """
        歌曲标题数据集
        Args:
            img_dir: 图片目录路径
            img_size: 图片尺寸 (height, width)
        """
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        
        # 收集所有图片路径并打乱顺序
        self.img_paths = sorted(list(self.img_dir.glob('t-*.png')))
        random.shuffle(self.img_paths)
        
        # 创建标签映射 (文件名 -> 标签ID)
        self.label_dict = {path.stem: idx for idx, path in enumerate(self.img_paths)}
        
        print(f"加载了 {len(self.img_paths)} 张图片")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.label_dict[img_path.stem]
        
        # 加载灰度图像
        img = Image.open(img_path).convert('L')
        img = img.resize((self.img_size[1], self.img_size[0]))  # (width, height)
        
        # 转换为numpy数组并归一化到[0,1]
        img = np.array(img, dtype=np.float32) / 255.0
        
        # 添加通道维度
        img = np.expand_dims(img, axis=0)
        
        return torch.from_numpy(img), label

class BatchMaker:
    def __init__(self, dataset, thresholds=[150, 155, 160, 165, 170]):
        """
        Batch生成器
        Args:
            dataset: 歌曲标题数据集
            thresholds: 亮度阈值列表
        """
        self.dataset = dataset
        self.thresholds = thresholds
        self.num_classes = len(dataset)
        self.samples_per_class = len(thresholds)
    
    def create_batch(self, batch_classes=32):
        """
        创建一个batch的数据
        Args:
            batch_classes: 每个batch的类别数
        Returns:
            batch_imgs: 增强后的图像 [batch_classes * samples_per_class, 1, H, W]
            batch_labels: 对应的标签 [batch_classes * samples_per_class]
        """
        # 随机选择batch_classes个类别
        selected_classes = random.sample(range(self.num_classes), batch_classes)
        
        batch_imgs = []
        batch_labels = []
        
        for class_idx in selected_classes:
            # 获取原始图像
            original_img, label = self.dataset[class_idx]
            
            # 对每个阈值生成增强图像
            for threshold in self.thresholds:
                augmented_img = self._apply_threshold_augmentation(original_img, threshold)
                batch_imgs.append(augmented_img)
                batch_labels.append(label)
        
        batch_imgs = torch.stack(batch_imgs)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        
        return batch_imgs, batch_labels
    
    def _apply_threshold_augmentation(self, img, threshold):
        """
        应用阈值增强
        Args:
            img: 原始图像 [1, H, W], 值在[0,1]
            threshold: 阈值 (0-255)
        Returns:
            增强后的图像
        """
        # 将图像转换到0-255范围
        img_255 = img * 255.0
        
        # 应用阈值二值化
        threshold_normalized = threshold / 255.0
        augmented = (img_255 > threshold_normalized).float()
        
        # 添加少量噪声
        noise = torch.randn_like(augmented) * 0.01
        augmented = torch.clamp(augmented + noise, 0, 1)
        
        return augmented

class ArcFaceLoss(nn.Module):
    def __init__(self, feature_dim, num_classes, margin=0.5, scale=64.0):
        """
        ArcFace损失函数
        Args:
            feature_dim: 特征维度
            num_classes: 类别数量
            margin: 角度边界
            scale: 特征缩放因子
        """
        super(ArcFaceLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.margin = torch.tensor(margin)
        self.scale = scale
        
        # 权重矩阵
        self.W = nn.Parameter(torch.FloatTensor(num_classes, feature_dim))
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, features, labels):
        """
        Args:
            features: 输入特征 [batch_size, feature_dim]
            labels: 真实标签 [batch_size]
        Returns:
            loss: ArcFace损失
        """
        # 归一化特征和权重
        self.F = features.shape[-1]
        features = F.normalize(features*self.F, p=2, dim=1)
        W = F.normalize(self.W*self.F, p=2, dim=1)
        
        # 计算余弦相似度
        cosine = F.linear(features, W)  # [batch_size, num_classes]
        
        # 计算目标角度的余弦值
        sine = torch.sqrt(1.0000001 - torch.pow(cosine, 2))
        phi = cosine * torch.cos(self.margin) - sine * torch.sin(self.margin)
        
        # 应用角度边界
        one_hot = torch.zeros(cosine.size(), device=features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # 缩放
        output *= self.scale
        
        # 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        return loss

def get_batch_size(s:int, n:int):
  if n%s == 0: return s
  for i in range(1, n+1):
    for j in [1, -1]:
      t = s+j*i
      if t < 1 or t > n: continue
      if n%t == 0: return t
  return n

def train_epoch(model, batch_maker, criterion, optimizer, device, batch_classes=32, num_batches=32):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    print_idx = num_batches//4
    
    for batch_idx in range(num_batches):
        start_time = time.time()
        # 生成一个batch
        batch_imgs, batch_labels = batch_maker.create_batch(batch_classes)
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        features = model(batch_imgs)
        loss = criterion(features, batch_labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_time = time.time()-start_time
        if (batch_idx+1)%print_idx == 0 or batch_idx == num_batches-1:
          print(f'Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f} batch_time:{batch_time:.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    ckpt_path = f'./song_recognition/ckpt_arcface_train.pth'
    img_dir = './song_recognition/title_imgs'
    
    is_load = False
    num_classes = -1
    feature_dim = 128
    batch_classes = 64
    num_epochs = 20
    learning_rate = 1e-2
    arcface_margin = 0.5
    
    # 创建数据集和batch生成器
    dataset = SongTitleDataset(img_dir)
    batch_maker = BatchMaker(dataset)
    num_classes = len(dataset)
    num_batches = 72
    # batch_classes = get_batch_size(batch_classes, num_classes)
    print(f"num_classes:{num_classes} batch_classes:{batch_classes}")
    
    # 创建模型
    if is_load:
      ckpt = torch.load(ckpt_path)
      model = TitleNet(num_classes=num_classes)
      model.load_state_dict(ckpt['model_state_dict'])
    else:
      model = TitleNet(num_classes=num_classes).to(device)
    
    # 创建损失函数和优化器
    criterion = ArcFaceLoss(feature_dim, num_classes, margin=arcface_margin)
    if is_load: criterion.load_state_dict(ckpt['criterion_state_dict'])
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': criterion.parameters()}
    ], lr=learning_rate, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    min_loss = 10000
    
    # 训练循环
    print("开始训练...")
    for epoch in range(1, num_epochs + 1):
      avg_loss = train_epoch(model, batch_maker, criterion, optimizer, device, batch_classes, num_batches)
      
      # 更新学习率
      scheduler.step()
      
      print(f'Epoch [{epoch}/{num_epochs}], Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
      
      # 每1个epoch保存一次模型
      if avg_loss < min_loss:
        min_loss = avg_loss
        torch.save({
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'criterion_state_dict': criterion.state_dict(),
          'loss': avg_loss,
        }, ckpt_path)
        print(f'模型已保存: {ckpt_path}')

if __name__ == '__main__':
    main()