import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, num_features=64, reduction=4):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1) 
        return self.sigmoid(x1)*x

class RCSAB(nn.Module):
    def __init__(self, num_features, reduction):
        super(RCSAB, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.ca1    = ChannelAttention(num_features, reduction)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.sa1    = SpatialAttention()
    def forward(self, x):
        x1 = self.conv1(x)
        residual1 = x1
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.ca1(x1)
        x1 = self.conv3(x1)
        residual2 = x1
        x1 = self.sa1(x1 + residual1)
        return  x1 + residual2

class RG(nn.Module):
    def __init__(self, num_features=64, num_rcab=20, reduction=16):
        super(RG, self).__init__()
        self.module = [RCSAB(num_features, reduction) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x+self.module(x)


class my_srnet(nn.Module):
    def __init__(self):
        super(my_srnet, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.rgs = nn.Sequential(*[RG(64, 20, 4) for _ in range(6)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64 * (4 ** 2), kernel_size=3, padding=1)
        self.upscale = nn.PixelShuffle(4)
        self.conv4  = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self,x):
        x = self.conv1(x)
        residual = x
        x = self.rgs(x)
        x = self.conv2(x)
        x += residual
        x = self.conv3(x)
        x = self.upscale(x)
        x = self.conv4(x)
        return x