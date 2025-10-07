import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class hswish(nn.Module):
  def forward(self, x):
    out = x * F.relu6(x + 3, inplace=True) / 6
    return out

class hsigmoid(nn.Module):
  def forward(self, x):
    out = F.relu6(x + 3, inplace=True) / 6
    return out

class SeModule(nn.Module):
  def __init__(self, in_size, reduction=4):
    super(SeModule, self).__init__()
    self.se = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(in_size // reduction),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(in_size),
      hsigmoid()
    )

  def forward(self, x):
    return x * self.se(x)

class Block(nn.Module):
  def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
    super(Block, self).__init__()
    self.stride = stride
    self.se = semodule

    self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(expand_size)
    self.nolinear1 = nolinear
    self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, 
                          padding=kernel_size//2, groups=expand_size, bias=False)
    self.bn2 = nn.BatchNorm2d(expand_size)
    self.nolinear2 = nolinear
    self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn3 = nn.BatchNorm2d(out_size)

    self.shortcut = nn.Sequential()
    if stride == 1 and in_size != out_size:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_size),
      )

  def forward(self, x):
    out = self.nolinear1(self.bn1(self.conv1(x)))
    out = self.nolinear2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.se is not None:
      out = self.se(out)
    out = out + self.shortcut(x) if self.stride==1 else out
    return out

class MobileNetV3_Small(nn.Module):
  def __init__(self, num_classes=1000):
    super(MobileNetV3_Small, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.hs1 = hswish()

    self.bneck = nn.Sequential(
      Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
      Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
      Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
      Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
      Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
      Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
      Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
      Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
      Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
      Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
      Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
    )

    self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn2 = nn.BatchNorm2d(576)
    self.hs2 = hswish()
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # 特征提取部分
    self.feature = nn.Sequential(
      nn.Linear(576, 256),
      nn.BatchNorm1d(256),
      hswish(),
      nn.Dropout(0.2),
      nn.Linear(256, 128),
    )
    
    self._initialize_weights()

  def forward(self, x):
    out = self.hs1(self.bn1(self.conv1(x)))
    out = self.bneck(out)
    out = self.hs2(self.bn2(self.conv2(out)))
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    out = self.feature(out)
    return out

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

class L2Norm(nn.Module):
  def __init__(self, dim=1):
    super(L2Norm, self).__init__()
    self.dim = dim
      
  def forward(self, x):
    return F.normalize(x, p=2, dim=self.dim)

class TitleNet(nn.Module):
  def __init__(self, num_classes=1000):
    super(TitleNet, self).__init__()
    self.backbone = MobileNetV3_Small(num_classes)
    self.l2_norm = L2Norm(dim=1)
      
  def forward(self, x):
    features = self.backbone(x)
    normalized_features = self.l2_norm(features)
    return normalized_features

  def extract_features(self, x):
    """提取L2归一化后的特征向量"""
    return self.forward(x)

if __name__ == "__main__":
  """测试模型结构和输出维度"""
  model = TitleNet()
  
  # 创建符合输入尺寸的测试数据 (36x450 灰度图)
  batch_size = 4
  test_input = torch.randn(batch_size, 1, 36, 450)
  
  # print("模型结构:")
  # print(model)
  print(f"\n输入尺寸: {test_input.shape}")
  
  with torch.no_grad():
    output = model(test_input)
    print(f"输出特征维度: {output.shape}")
    print(f"特征范数: {torch.norm(output, dim=1)}")  # 应该接近1.0

  # 统计参数量
  total_params = sum(p.numel() for p in model.parameters())
  print(f"\n总参数量: {total_params:,}")
  
def get_batch_size(s:int, n:int):
  if n%s == 0: return s
  for i in range(1, n+1):
    for j in [1, -1]:
      t = s+j*i
      if t < 1 or t > n: continue
      if n%t == 0: return t
  return n
