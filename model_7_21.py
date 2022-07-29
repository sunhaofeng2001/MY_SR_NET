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
        #'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x1 = self.conv1(x1) 
        return x * self.sigmoid(x1)

class CORSS_SR(nn.Module):
    def __init__(self, num_features, reduction,kernel_size):
        super(CORSS_SR, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.ca1    = ChannelAttention(num_features, reduction)
        self.ca2    = ChannelAttention(num_features, reduction)
        self.sa1    = SpatialAttention(kernel_size)
        self.sa2    = SpatialAttention(kernel_size)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x1 = self.ca1(x)
        x2 = self.sa1(x)
        cross1 = x1
        cross2 = x2
        x1 = self.sa1(x1 + cross1)
        x2 = self.sa2(x2 + cross2)
        x3 = self.conv2(x1 + x2)
        return  x3


class RG(nn.Module):
    def __init__(self, num_features=64, num_rcab=20, reduction=16,kernel_size=7):
        super(RG, self).__init__()
        self.module = [CORSS_SR(num_features, reduction,kernel_size) for _ in range(num_rcab)]
        self.module.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return x+self.module(x)

class my_srnet(nn.Module):
    def __init__(self):
        super(my_srnet, self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,1,1)
        self.rgs1_1 = nn.Sequential(*[RG(64,20,8,3) for _ in range(4)])
        
        self.conv1_2 = nn.Conv2d(3,64,7,1,3)
        self.rgs1_2 = nn.Sequential(*[RG(64,10,16,7) for _ in range(3)])

        self.conv2 = nn.Conv2d(64, 64 * (4 ** 2), kernel_size=3, padding=1)
        self.upscale = nn.PixelShuffle(4)
        self.conv3  = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self,x):
        x1 = self.conv1_1(x)
        x1 = self.rgs1_1(x1)

        #x2 = self.conv1_2(x)
        #x2 = self.rgs1_2(x2)

        x3 = self.conv2(x1)
        x3 = self.upscale(x3)
        x3 = self.conv3(x3)
        return x3