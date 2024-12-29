import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math
from MAGNET.models.module.PIEM import PIEM
from MAGNET.models.module.WAKG import WAKG
from MAGNET.models.module.BFM import BFM
from utils.l2_norm import l2_norm


class MAGNET(nn.Module):

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet
        self.disturb_num = self.args.disturb_num
        self.weight = self.args.weight

        if self.resnet:
            self.num_channel = 640
            self.num_channel2 = 64
            self.feature_extractor = ResNet.resnet12(drop=True)
        else:
            self.num_channel = 64
            self.num_channel2 = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.PIEM = PIEM(
            num_channel=self.num_channel,
            num_channel2=self.num_channel2,
            disturb_num=self.disturb_num,
            weight=self.weight,
            resnet=self.resnet
        )
        self.wakg = WAKG(in_channels=self.num_channel, out_channels=self.num_channel, kernel_size=5, stride=1, wt_levels=1, wt_type='db1', resnet=self.resnet)
        self.bfm = BFM(self.resnet)
        
        self.scale_1 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_2 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_3 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_4 = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_cosine_dist(self, inp, way, shot):
        F1, F_2, F_3, F_4 = self.feature_extractor(inp)  # t:2层   s:3层    l:4层
        
        support_f1, query_f1 = self.PIEM(F1, F_4, way, shot)
        support_f3, query_f3, support_f2, query_f2 = self.bfm(F_2, F_3, F_4, way, shot)
        support_f4, query_f4 = self.wakg(F_4, way, shot)

        cos_f1 = F.linear(l2_norm(query_f1), l2_norm(support_f1))
        cos_f2 = F.linear(l2_norm(query_f2), l2_norm(support_f2))
        cos_f3 = F.linear(l2_norm(query_f3), l2_norm(support_f3))
        cos_f4 = F.linear(l2_norm(query_f4), l2_norm(support_f4))

        return cos_f1, cos_f2, cos_f3, cos_f4

    def meta_test(self, inp, way, shot):
        cos_f1, cos_f2, cos_f3, cos_f4 = self.get_cosine_dist(inp=inp, way=way, shot=shot)
        scores = cos_f1 + cos_f2 + cos_f3 + cos_f4

        _, max_index = torch.max(scores, 1)
        return max_index

    def forward(self, inp):
        cos_f1, cos_f2, cos_f3, cos_f4 = self.get_cosine_dist(inp=inp, way=self.way, shot=self.shots[0])
        
        cos_f1 = cos_f1 * self.scale_1
        cos_f2 = cos_f2 * self.scale_2
        cos_f3 = cos_f3 * self.scale_3
        cos_f4 = cos_f4 * self.scale_4

        return cos_f1, cos_f2, cos_f3, cos_f4