import torch as th
import torch.nn as nn

def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor #最小通道数,若为None,则为8,所以 最小为8
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor) #趋近于8的整数倍,四舍五入
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLU6(nn.Sequential):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
		super(ConvBNReLU6, self).__init__(
			nn.Conv2d(in_channels, out_channels, kernel_size, 
				stride, (kernel_size-1)//2, groups=groups, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU6(inplace=True)
		)

class InvertedResidual(nn.Module):
	def __init__(self, in_channels, out_channels, stride, expend_ratio):
		super(InvertedResidual, self).__init__()
		hidden_channel = in_channels*expend_ratio
		self.use_shortcut = stride == 1 and in_channels == out_channels

		layers = []
		if expend_ratio != 1:
			layers.append(ConvBNReLU6(in_channels, hidden_channel, kernel_size=1))
		layers.extend([
			ConvBNReLU6(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
			nn.Conv2d(hidden_channel, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels)
		])
		self.ir = nn.Sequential(*layers)
	def forward(self, x):
		return x + self.ir(x) if self.use_shortcut else self.ir(x)

class BangUiNet(nn.Module):
	def __init__(self, num_classes, alpha=1.0, round_nearest=8, keep_rate=0.2):
		super(BangUiNet, self).__init__()
		in_channels = _make_divisible(32 * alpha, round_nearest)#调整到8的整数倍
		last_channels = _make_divisible(256 * alpha, round_nearest)
		inverted_residual_setting = [
			#t扩展因子,c输出特征矩阵深度channel,n是bottleneck的重复次数,s是步距
			[1, 16, 1, 1],  #90*160
			[6, 24, 2, 2],  #45*80
			[6, 32, 3, 2],  #23*40
			[6, 64, 3, 2],  #12*20
			[6, 96, 2, 2],  #6*10
			[6, 160, 1, 1], #6*10
		]
		features = []
		features.append(ConvBNReLU6(3, in_channels, stride=1))
		for t, c, n, s in inverted_residual_setting:
			out_channels = _make_divisible(c*alpha, round_nearest)
			for i in range(n):
				stride = s if i == 0 else 1
				features.append(InvertedResidual(in_channels, out_channels, stride, t))
				in_channels = out_channels
		features.append(ConvBNReLU6(in_channels, last_channels, 1))
		self.features = nn.Sequential(*features)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Sequential(
			nn.Dropout(keep_rate),
			nn.Linear(last_channels, num_classes)
		)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias != None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, (2/(m.weight.size(0)+m.weight.size(1)))**0.5)
				nn.init.zeros_(m.bias)
	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

if __name__ == "__main__":
    # 创建模型实例
    model = BangUiNet(num_classes=26)
    
    # 测试前向传播
    batch_size = 4
    dummy_input = th.randn(batch_size, 3, 160, 90)
    features = model(dummy_input)
    
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出特征尺寸: {features.shape}")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 验证模型组件
    # print("\n模型结构:")
    # print(model)
