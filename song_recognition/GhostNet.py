import torch
import torch.nn as nn
import math

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class GhostNet(nn.Module):
    def __init__(self, num_classes=128, in_channels=1, width_mult=1.0):
        super().__init__()
        
        # 简化的GhostNet结构
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),  # (16, 18, 225)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            GhostModule(16, 16, 3, 2, 3),  # (16, 18, 225)
            nn.MaxPool2d(2, 2),  # (16, 9, 112)
            
            GhostModule(16, 24, 3, 2, 3),  # (24, 9, 112)
            GhostModule(24, 24, 3, 2, 3),  # (24, 9, 112)
            nn.MaxPool2d(2, 2),  # (24, 4, 56)
            
            GhostModule(24, 40, 3, 2, 3),  # (40, 4, 56)
            GhostModule(40, 40, 3, 2, 3),  # (40, 4, 56)
            nn.AdaptiveAvgPool2d((1, 1))  # (40, 1, 1)
        )
        
        self.classifier = nn.Linear(40, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
  model = GhostNet(num_classes=128)
  print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")  # ~0.2M