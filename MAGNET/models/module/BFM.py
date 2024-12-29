import torch
import torch.nn as nn


class BFM(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        if resnet:
            self.num_channel = 640
            self.fiter_g2 = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fiter_g3 = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            )
            
            self.fc_2 = fc(resnet, in_c=160)
            self.fc_3 = fc(resnet, in_c=320)
            
        else:
            self.num_channel = 64
            self.fiter_g2 = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fiter_g3 = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fc_2 = fc(resnet, in_c=64)
            self.fc_3 = fc(resnet, in_c=64)
            

    def forward(self, F_2, F_3, F_4, way, shot):
        heat_map_2 = nn.functional.interpolate(F_4, size=(F_2.shape[-1], F_2.shape[-1]), mode='bilinear', align_corners=False)
        fiter_2 = nn.Sigmoid()(self.fiter_g2(heat_map_2))

        heat_map_3 = nn.functional.interpolate(F_4, size=(F_3.shape[-1], F_3.shape[-1]), mode='bilinear', align_corners=False)
        fiter_3 = nn.Sigmoid()(self.fiter_g3(heat_map_3))
        
        F_2 = F_2 * fiter_2 
        F_3 = F_3 * fiter_3
        
        f_2 = self.fc_2(F_2)
        f_3 = self.fc_3(F_3)
        
        support_f2 = f_2[:way * shot].view(way, shot, -1).mean(1)
        query_f2 = f_2[way * shot:]

        support_f3 = f_3[:way * shot].view(way, shot, -1).mean(1)
        query_f3 = f_3[way * shot:]

        return support_f3, query_f3, support_f2, query_f2

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, max=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.max = nn.MaxPool2d(2) if max else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.max is not None:
            x = self.max(x)
        return x

class fc(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        self.max = nn.AdaptiveMaxPool2d((1, 1))
        self.alpha = 0.3
        if resnet:
            self.feature_size = 640
            self.conv_block = nn.Sequential(
                BasicConv(in_c, self.feature_size, kernel_size=1, stride=1, padding=0, relu=True)
            )
            self.mlp = nn.Sequential(
                nn.BatchNorm1d(self.feature_size),
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )
        else:
            self.feature_size = 64
            self.conv_block = nn.Sequential(
                BasicConv(in_c, self.feature_size, kernel_size=3, stride=1, padding=1, relu=True, bias=True)
            )
            self.mlp = nn.Sequential(
                nn.Linear(self.feature_size, self.feature_size),
                nn.ELU(inplace=True)
            )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.alpha * x + (1 - self.alpha) * self.mlp(x)
        return x