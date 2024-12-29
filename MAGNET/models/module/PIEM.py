import torch
import torch.nn as nn
import torch.nn.functional as F

class structure_capture(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(structure_capture, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
 
        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
 
        # 在两个维度上应用注意力
        return x * x_h * x_w


class PIEM(nn.Module):
    def __init__(self, num_channel, num_channel2, disturb_num, weight, resnet):
        super(PIEM, self).__init__()
        self.num_channel = num_channel
        self.num_channel2 = num_channel2
        self.disturb_num = disturb_num
        self.weight = weight
        self.structure_capture = structure_capture(channel=self.num_channel, kernel_size=7)

        if resnet:
            self.mask_branch = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, self.disturb_num, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.mask_branch = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, self.disturb_num, kernel_size=1, stride=1, padding=0)
            )

        if resnet:
            self.both_mlp1 = nn.Sequential(
                nn.BatchNorm1d(self.num_channel2 * self.disturb_num),
                nn.Linear(self.num_channel2 * self.disturb_num, self.num_channel2 * self.disturb_num),
                nn.ELU(inplace=True)
            )
        else:
            self.both_mlp1 = nn.Sequential(
                nn.Linear(self.num_channel2 * self.disturb_num, self.num_channel2 * self.disturb_num),
                nn.ELU(inplace=True)
            )

    def integration(self, layer1, layer2):
        batch_size = layer1.size(0)
        channel_num = layer1.size(1)
        disturb_num = layer2.size(1)
        layer1 = layer1.unsqueeze(2)
        layer2 = layer2.unsqueeze(1)

        sum_of_weight = layer2.view(batch_size, disturb_num, -1).sum(-1) + 1e-5
        vec = (layer1 * layer2).view(batch_size, channel_num, disturb_num, -1).sum(-1)
        vec = vec / sum_of_weight.unsqueeze(1)
        vec = vec.view(batch_size, channel_num * disturb_num)
        return vec

    def forward(self, F1, F_L, way, shot):
        F_L = self.structure_capture(F_L)
        heat_map = F.interpolate(F_L, size=(F1.shape[-1], F1.shape[-1]), mode='bilinear', align_corners=False)
        heat_map = self.mask_branch(heat_map)
        mask = torch.sigmoid(heat_map)
        layer1_vec = self.integration(F1, mask)
        layer1_vec = (1 - self.weight) * self.both_mlp1(layer1_vec) + self.weight * layer1_vec
        support_f1 = layer1_vec[:way * shot].view(way, shot, self.num_channel2 * self.disturb_num).mean(1)
        query_f1 = layer1_vec[way * shot:]
        return support_f1, query_f1