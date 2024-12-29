import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from functools import partial
from einops import rearrange


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)
    ], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
    ], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

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


class WAKG(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1', resnet=False):
        super(WAKG, self).__init__()
        if resnet:
            self.num_channel = 640
            self.TG_prompt = nn.Parameter(torch.randn((1, self.num_channel, 5, 5)))
            self.attn = Attention(self.num_channel, num_heads=1)
            self.fc_4 = fc(resnet=True, in_c=640)
        else:
            self.num_channel = 64
            self.TG_prompt = nn.Parameter(torch.randn((1, self.num_channel, 5, 5)))
            self.attn = Attention(self.num_channel, num_heads=1)
            self.fc_4 = fc(resnet=False, in_c=64)

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels * 4, bias=False) 
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) 
            for _ in range(self.wt_levels)
        ])

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride, groups=in_channels)
        else:
            self.do_stride = None
        self.lamda = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.max = nn.AdaptiveMaxPool2d((1, 1))

        self.feature_compression = fc(resnet=resnet, in_c=out_channels) 
    
    def forward(self, x, way, shot):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)
        
        b, c, h, w = x.shape
        m = h * w
        support_4 = x[:way * shot].view(way, shot, c, m)
        centroid_4 = support_4.mean(dim=1).unsqueeze(dim=1).view(-1, 1, c, m)
        query_4 = x[way * shot:].view(1, -1, c, m)
        query_num = query_4.shape[1]
        zero_c = torch.zeros([1, query_num, c, m]).cuda()
        zero_q = torch.zeros([way, 1, c, m]).cuda()
        centroid_4 = (centroid_4 + zero_c).view(-1, c, h, w)
        query_4 = (query_4 + zero_q).view(-1, c, h, w)
        query_4 = self.attn(query_4, centroid_4, self.TG_prompt).view(-1, query_num, c, h, w).mean(0)
        query_4 = self.max(query_4).view(query_4.size(0), -1)

        f_4 = self.fc_4(x)

        support_f4 = f_4[:way * shot].view(way, shot, -1).mean(1)
        query_f4 = f_4[way * shot:] + self.lamda * query_4


        return support_f4, query_f4

class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, head_dim_ratio=1., qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = round(dim // num_heads * head_dim_ratio)
        self.head_dim = head_dim
        qk_scale_factor = qk_scale if qk_scale is not None else -0.25
        self.scale = head_dim ** qk_scale_factor

        self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 3, stride=1, padding=1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.TG_prompt = nn.Parameter(torch.randn((1, dim, 5, 5)))

    def forward(self, q, s, TG_prompt):
        B, C, H, W = q.shape

        t = s + TG_prompt  # [B, C, H, W]

        x = self.qkv(q)  # [B, 3C, H, W]
        qkv = rearrange(x, 'b (x y z) h w -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, head, HW, d]

        x1 = self.qkv(t)  # [B, 3C, H, W]
        qkv1 = rearrange(x1, 'b (x y z) h w -> x b y (h w) z', x=3, y=self.num_heads, z=self.head_dim)
        qs, ks, vs = qkv1[0], qkv1[1], qkv1[2]  # [B, head, HW, c]

        q = torch.cat([q, qs], dim=2)  # [B, head, 2HW, c]
        k = torch.cat([k, ks], dim=2)  # [B, head, 2HW, c]
        v = torch.cat([v, vs], dim=2)  # [B, head, 2HW, c]

        attn = ((q * self.scale) @ (k.transpose(-2, -1) * self.scale))  # [B, head, 2HW, 2HW]
        attn = attn.softmax(dim=-1)  # [B, head, 2HW, 2HW]
        attn = self.attn_drop(attn)
        x = attn @ v  # [B, head, 2HW, c]

        x = x[:, :, :H * W]  # [B, head, HW ,c]
        x = rearrange(x, 'b y (h w) z -> b (y z) h w', h=H, w=W)  # [B, C, H, W]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x