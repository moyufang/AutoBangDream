import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import cv2
import numpy as np
import random

def get_batch_size(s:int, n:int):
  if n%s == 0: return s
  for i in range(1, n+1):
    for j in [1, -1]:
      t = s+j*i
      if t < 1 or t > n: continue
      if n%t == 0: return t
  return n

def prepocess_img(img):
  img = (255-img).astype(np.float32)/255.0
  return torch.from_numpy(np.expand_dims(img, axis=0))

class SongTitleDataset(Dataset):
  def __init__(self, img_dir):
    self.img_dir = Path(img_dir)
    
    # 获取所有t-*.png文件并打乱顺序
    self.img_paths = sorted(list(self.img_dir.glob("?-*.png")))

    # 创建索引到歌曲ID的映射
    self.idx_to_song_id = {}
    for idx, img_path in enumerate(self.img_paths):
      song_id = int(img_path.stem.split('-')[1])
      self.idx_to_song_id[idx] = song_id
      
    self.imgs = []
    for img_path in self.img_paths:
      img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
      self.imgs.append(prepocess_img(img))
        
    print(f"Loaded {len(self.img_paths)} song title images")
        
  def __len__(self):
    return len(self.img_paths)
  
  def __getitem__(self, idx):
    return self.imgs[idx], self.idx_to_song_id[idx]
  
  def augment_image(self, img_tensor):
    """对单张图片进行增强"""
    # 转换为numpy进行增强操作
    img_np = img_tensor.numpy().squeeze(0) * 255.0
    img_np = img_np.astype(np.uint8)
    
    # 1. 微小平移
    img_a = self.random_shift(img_np)
    
    # 2. 随机噪声
    img_b = self.add_noise(img_a)
    
    # 转换回tensor
    img_tensor_aug = torch.from_numpy(img_b.astype(np.float32) / 255.0).unsqueeze(0)
    
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

class TitleNet(nn.Module):
  def __init__(self, feature_dim=128, backbone_type:int=0, pretrained:bool=False):
    """
    Args:
        feature_dim: 特征向量维度
        backbone_type:  backbone 模型选择
                        0 -> efficientnet_b0     | 4,170,940
                        1 -> mobilenet_v3_small  | 1,648,768
                        2 -> shufflenet_v2       | 1,384,372
                        3 -> squeezenet          |   787,008
                        4 -> ghostnet            |    23,352 
    """
    
    super(TitleNet, self).__init__()
    if backbone_type == 0: model = self.create_efficientnet_b0(feature_dim,pretrained=pretrained)
    elif backbone_type == 1: model = self.create_mobilenet_v3_small(feature_dim,pretrained=pretrained)
    elif backbone_type == 2: model = self.create_shufflenet_v2(feature_dim,pretrained=pretrained)
    elif backbone_type == 3: model = self.create_squeezenet(feature_dim,pretrained=pretrained)
    elif backbone_type == 4: model = self.create_ghost(feature_dim,pretrained=pretrained)
    
    self.backbone = model
    self.feature_dim = feature_dim
    self.backbone_type = backbone_type
      
  def forward(self, x):
    features = self.backbone(x)
    normalized_features = F.normalize(features, p=2, dim=1)
    return normalized_features

  def extract_features(self, x):
    """提取L2归一化后的特征向量"""
    return self.forward(x)
  
  def create_mobilenet_v3_small(self, num_classes=128, in_channels=1, pretrained:bool=False):
    # 加载预训练模型
    
    if pretrained:
      model = models.mobilenet_v3_small(pretrained=pretrained)
    else:
      model = models.mobilenet_v3_small(weights=None)
    
    # 修改第一层卷积，适应灰度图像 (1通道)
    original_first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
      in_channels=in_channels,  # 改为1通道输入
      out_channels=original_first_conv.out_channels,
      kernel_size=original_first_conv.kernel_size,
      stride=original_first_conv.stride,
      padding=original_first_conv.padding,
      bias=original_first_conv.bias is not None
    )
    
    # 初始化新卷积层的权重（使用原RGB通道的均值）
    with torch.no_grad():
      # 对原3通道权重取平均，扩展到1通道
      new_weight = original_first_conv.weight.mean(dim=1, keepdim=True)
      model.features[0][0].weight.copy_(new_weight)
      if original_first_conv.bias is not None:
          model.features[0][0].bias.copy_(original_first_conv.bias)
    
    # 修改分类器，输出128分类
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model

  def create_efficientnet_b0(self, num_classes=128, in_channels=1, pretrained:bool=False):
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # 修改第一层适应灰度输入
    original_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    
    # 初始化新权重
    with torch.no_grad():
        new_weight = original_conv.weight.mean(dim=1, keepdim=True)
        model.features[0][0].weight.copy_(new_weight)
    
    # 修改分类器
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

  def create_shufflenet_v2(self, num_classes=128, in_channels=1, pretrained:bool=False):
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=pretrained)
    
    # 修改第一层
    original_conv = model.conv1[0]
    model.conv1[0] = nn.Conv2d(
      in_channels=in_channels,
      out_channels=original_conv.out_channels,
      kernel_size=original_conv.kernel_size,
      stride=original_conv.stride,
      padding=original_conv.padding,
      bias=original_conv.bias is not None
    )
    
    # 初始化权重
    with torch.no_grad():
      new_weight = original_conv.weight.mean(dim=1, keepdim=True)
      model.conv1[0].weight.copy_(new_weight)
    
    # 修改分类器
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
  
  def create_squeezenet(self, num_classes=128, in_channels=1, pretrained:bool=False):
    if pretrained:
      model = torchvision.models.squeezenet1_1(pretrained=pretrained)
    else:
      model = torchvision.models.squeezenet1_1(weights=None)
    # 修改第一层
    model.features[0] = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=2, padding=1
    )
    
    # 修改分类器
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    
    return model

  def create_ghost(self, num_classes=128, in_channels=1, pretrained:bool=False):
    from song_recognition.GhostNet import GhostNet
    return GhostNet(num_classes, in_channels)

def load_TitleNet(ckpt_path:str):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = torch.load(ckpt_path, map_location=device)
  model = TitleNet(ckpt['feature_dim'], ckpt['backbone_type'],pretrained=False).to(device)
  model.load_state_dict(ckpt['model_state_dict'])
  model.eval()
  return model

if __name__ == "__main__":
  """测试模型结构和输出维度"""
  model = TitleNet(backbone_type=3)
  
  # 创建符合输入尺寸的测试数据 (36x450 灰度图)
  batch_size = 4
  test_input = torch.randn(batch_size, 1, 36, 450)
  
  # print("模型结构:")
  # print(model)
  print(f"\n输入尺寸: {test_input.shape}")
  
  with torch.no_grad():
    start_time = time.time()
    output = model(test_input)
    inference_time = time.time()-start_time
    print(f"输出特征维度: {output.shape}")
    print(f"推理时间: {inference_time} 特征范数: {torch.norm(output, dim=1)}")  # 应该接近1.0

  # 统计参数量
  total_params = sum(p.numel() for p in model.parameters())
  print(f"\n总参数量: {total_params:,}")
  

