import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import logging
import math
from torch import einsum
from torch.nn.parameter import Parameter
from torch.nn import init
from torch._jit_internal import Optional
from torch.nn.modules.module import Module
from torch.autograd import Function

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super(Downsample, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LIE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LIE, self).__init__()
        self.DSConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1)
        )
        ########## add #########
        self.activation = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        dwconv_result = self.DSConv(x)  # 8,48,64,64
        dwconv_result = self.batchnorm(dwconv_result)
        dwconv_result = self.activation(dwconv_result)
        result = dwconv_result + x
        return result


#                         new                       #

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


##########################################################################
# Multi-Scale Flow Gating Network
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.dwconv_2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, padding='same',
                                  groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x2 = self.dwconv_2(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x = x[:, :, :h, :w]

        return x


class Enhanced_SPR_SA(nn.Module):
    """
    增强版空间像素细化自注意力模块
    结合了多种注意力机制和特征增强技术
    """

    def __init__(self, dim, growth_rate=2.0, num_heads=4):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.dim = dim
        self.num_heads = num_heads

        # 多尺度特征提取
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim // 4, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, hidden_dim // 4, 5, 1, 2, groups=dim),
            nn.Conv2d(dim, hidden_dim // 4, 7, 1, 3, groups=dim),
            nn.Conv2d(dim, hidden_dim // 4, 1, 1, 0)
        ])

        # 特征融合
        self.fusion_conv = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 8, 1, 1),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 残差连接权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        # 多尺度特征提取
        multi_scale_feats = []
        for conv in self.multi_scale_conv:
            multi_scale_feats.append(conv(x))
        x = torch.cat(multi_scale_feats, dim=1)

        # 特征融合
        x = self.fusion_conv(x)

        # 通道注意力
        ca_weight = self.channel_att(x)
        x = x * ca_weight

        # 空间注意力
        sa_weight = self.spatial_att(x)
        x = x * sa_weight

        # 多头自注意力
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attn_out, _ = self.multi_head_att(x_flat, x_flat, x_flat)
        attn_out = attn_out.transpose(1, 2).reshape(B, -1, H, W)

        # 特征融合
        x = x + attn_out

        # 输出投影
        out = self.output_conv(x)

        # 残差连接
        return residual + self.alpha * out + self.beta * x.mean(dim=1, keepdim=True)


# 保持原始spr_sa类作为备选
class spr_sa(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        )
        self.act = nn.GELU()
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x1 = F.adaptive_avg_pool2d(x, (1, 1))
        x1 = F.softmax(x1, dim=1)
        x = x1 * x
        x = self.act(x)
        x = self.conv_1(x)
        return x


class Enhanced_Spatial_Attention(nn.Module):
    """
    简化版增强空间注意力模块 - 比原始spr_sa更强但更简洁
    保留核心的多尺度空间注意力机制
    """

    def __init__(self, dim, growth_rate=3.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        self.dim = dim

        # 多尺度空间特征提取 (3x3, 5x5, 1x1)
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(dim, hidden_dim // 3, 3, 1, 1, groups=dim),
            nn.Conv2d(dim, hidden_dim // 3, 5, 1, 2, groups=dim),
            nn.Conv2d(dim, hidden_dim // 3, 1, 1, 0)
        ])

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

        # 全局空间注意力
        self.global_spatial_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid()
        )

        # 局部空间注意力
        self.local_spatial_att = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 8, 1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 8, 1, 1),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 残差连接权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        # 1. 多尺度空间特征提取
        multi_scale_feats = []
        for conv in self.multi_scale_conv:
            multi_scale_feats.append(conv(x))
        multi_scale_x = torch.cat(multi_scale_feats, dim=1)

        # 2. 特征融合
        fused_x = self.fusion_conv(multi_scale_x)

        # 3. 全局空间注意力
        global_weight = self.global_spatial_att(fused_x)
        global_attended = fused_x * global_weight

        # 4. 局部空间注意力
        local_weight = self.local_spatial_att(global_attended)
        local_attended = global_attended * local_weight

        # 5. 输出投影
        out = self.output_conv(local_attended)

        # 6. 残差连接
        return residual + self.alpha * out + self.beta * local_weight


# 第一种改进版spr_sa
class LightAware_SPRSA(nn.Module):
    """
    Light-Aware Spatial Pixel Refinement Self-Attention
    SPR-SA 改进版
    """

    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)

        # 局部空间卷积
        self.local_conv = nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim)
        self.mix_conv = nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)
        # 光照引导分支（近似 Retinex 光照图）
        self.illum_branch = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=1),  # 提取亮度
            nn.Sigmoid()  # 归一化为 [0,1]
        )
        # 全局上下文建模
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 1),
            nn.Sigmoid()
        )
        # 光照门控机制（illumination-aware gating）
        self.gate_conv = nn.Sequential(
            nn.Conv2d(1, hidden_dim, 3, 1, 1),
            nn.Sigmoid()
        )
        # 输出卷积
        self.out_conv = nn.Conv2d(hidden_dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        # 局部空间建模
        feat = self.local_conv(x)
        feat = self.mix_conv(feat)
        # 提取光照引导图
        illum_map = self.illum_branch(x)  # [B,1,H,W]
        # 全局上下文（通道维重标定）
        g_weight = self.global_context(feat)
        # 光照门控（空间维调节）
        illum_gate = self.gate_conv(illum_map)
        # 综合门控 (通道 × 空间)
        feat = feat * g_weight * illum_gate
        # 激活 & 输出映射
        feat = self.act(feat)
        out = self.out_conv(feat)
        # 残差 + 亮度跳连
        out = out + residual + illum_map * 0.1  # 亮度残差稳定训练

        return out


# 第二种改进版spr_sa
class DDPR_SA(nn.Module):
    """
    Dynamic Direction-aware Pixel Refinement Self-Attention (改进版 SPR-SA)
    """

    def __init__(self, dim):
        super(DDPR_SA, self).__init__()

        # 多方向卷积（模拟方向感知）
        self.conv0 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=1, dilation=1)
        self.conv45 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=(1, 2), dilation=(1, 2))
        self.conv90 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=(2, 1), dilation=(2, 1))
        self.conv135 = nn.Conv2d(dim, dim // 4, kernel_size=3, padding=2, dilation=2)

        # 融合方向特征
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)

        # 全局引导
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        # 残差输出
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # 多方向特征提取
        f0 = self.conv0(x)
        f45 = self.conv45(x)
        f90 = self.conv90(x)
        f135 = self.conv135(x)

        # 拼接方向特征
        f_cat = torch.cat([f0, f45, f90, f135], dim=1)
        f_dir = self.fuse(f_cat)

        # 全局感知门控
        g = self.global_pool(f_dir).view(B, C)
        gate = self.gate(g).view(B, C, 1, 1)

        # 动态像素加权
        f_weighted = f_dir * gate

        # 残差增强
        out = self.out_conv(f_weighted) + x

        return out


# 第三种改进版spr_sa
class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAB, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        out = self.sigmoid(x) * res
        return out


# 第四种改进版spr_sa
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 8,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = True
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


# 第五种改进版spr_sa
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.5):
        super().__init__()
        dw_channel = c * DW_Expand
        # print('dw_channel ', dw_channel)

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # ablation SSA
        # donot use attention
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2 * 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.ssa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.sg(x)

        x_channel = x * self.sca(x)

        maxpool_channel, _ = torch.max(x, dim=1, keepdim=True)

        x_spatial = x * self.ssa(maxpool_channel)

        x_merge = torch.cat((x_channel, x_spatial), dim=1)

        x = self.conv3(x_merge)

        x = self.dropout1(x)

        y = res + x * self.beta

        return y


# 第六种改进版spr_sa
class SMSA(nn.Module):  # 第六种改法
    def __init__(self, in_channels, gate_layer='sigmoid'):
        super(SMSA, self).__init__()
        self.dim = in_channels
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=3, padding=1, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=5, padding=2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=7, padding=3, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=9, padding=4, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, in_channels)
        self.norm_w = nn.GroupNorm(4, in_channels)

        # 多尺度空间特征提取 (3x3, 5x5, 1x1)
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        ])

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, 1, 0),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

    def forward(self, x):
        b, c, h_, w_ = x.size()
        # 新加
        multi_scale_feats = []
        for conv in self.multi_scale_conv:
            multi_scale_feats.append(conv(x))
        multi_scale_x = torch.cat(multi_scale_feats, dim=1)

        # 特征融合
        fused_x = self.fusion_conv(multi_scale_x)

        # (B, C, H)
        x_h = fused_x.mean(dim=3)  # x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        # (B, C, W)
        x_w = fused_x.mean(dim=2)  # x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn
        return x


class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class GlobalContext(nn.Module):

    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            act_layer(),
            nn.Linear(dim // 8, dim)
        )
        self.head = 8
        self.scale = (dim // self.head) ** -0.5
        self.rescale_weight = nn.Parameter(torch.ones(self.head))
        self.rescale_bias = nn.Parameter(torch.zeros(self.head))
        self.epsilon = 1e-5

    def _get_gc(self, gap):
        return self.fc(gap)

    def forward(self, x):
        b, c, w, h = x.size()
        x = rearrange(x, "b c x y -> b c (x y)")
        gap = x.mean(dim=-1, keepdim=True)
        q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h=self.head), [x, gap])
        sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale
        std, mean = torch.std_mean(sim, dim=[1, 2], keepdim=True)
        sim = (sim - mean) / (std + self.epsilon)
        sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(
            dim=0).unsqueeze(dim=-1)
        sim = sim.reshape(b, self.head, 1, w, h)
        gc = self._get_gc(gap.squeeze(dim=-1)).reshape(b, self.head, -1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        gc = rearrange(sim * gc, "b h d x y -> b (h d) x y")
        return gc


class HAM(nn.Module):
    def __init__(self, dim, act_layer=nn.GELU):
        super().__init__()
        self.act = act_layer()

        self.gc1 = GlobalContext(dim, act_layer=act_layer)
        self.dw1 = nn.Conv2d(dim, dim, 11, stride=1, padding=5, groups=dim, bias=True)
        self.eca_layer1 = eca_layer(dim)

        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, max(dim // 8, 16), 1),
            self.act,
            nn.Conv2d(max(dim // 8, 16), dim, 1)
        )

        self.gc2 = GlobalContext(dim, act_layer=act_layer)
        self.dw2 = nn.Conv2d(dim, dim, 11, stride=1, padding=5, groups=dim, bias=True)

        self.fc2 = nn.Sequential(
            nn.Conv2d(dim, max(dim // 8, 16), 1),
            self.act,
            nn.Conv2d(max(dim // 8, 16), dim, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        gc1 = self.gc1(x)
        eca1 = self.eca_layer1(x)
        x = eca1 + gc1
        x = self.act(self.fc1(self.dw1(x)))
        return x


#
class SoftPooling2D(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(SoftPooling2D, self).__init__()

        self.avgpool = torch.nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)

    def forward(self, x):
        # return self.avgpool(x)
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class LIA(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        f = f
        self.body = nn.Sequential(
            # sample importance
            nn.Conv2d(channels, f, 1),
            SoftPooling2D(7, stride=3),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(f, channels, 3, padding=1),
            # to heatmap
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, x):
        # interpolate the heat map
        g = self.gate(x[:, :1])
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        return x * w * g


class ComplexFFT(nn.Module):
    def __init__(self):
        super(ComplexFFT, self).__init__()

    def forward(self, x):
        # 对输入进行复数FFT变换
        x_fft = torch.fft.fft2(x, dim=(-2, -1))  # 2D FFT，沿最后两个维度进行
        real = x_fft.real  # 提取实部
        imag = x_fft.imag  # 提取虚部
        return real, imag


class ComplexIFFT(nn.Module):
    def __init__(self):
        super(ComplexIFFT, self).__init__()

    def forward(self, real, imag):
        # 复合实部和虚部，作为复杂信号
        x_complex = torch.complex(real, imag)
        # 进行复数IFFT逆变换
        x_ifft = torch.fft.ifft2(x_complex, dim=(-2, -1))  # 2D IFFT
        return x_ifft.real  # 输出结果的实部


class Conv1x1(nn.Module):
    def __init__(self, in_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0,
                              groups=in_channels * 2)

    def forward(self, x):
        return self.conv(x)


class Stage2_fft(nn.Module):
    def __init__(self, in_channels):
        super(Stage2_fft, self).__init__()
        self.c_fft = ComplexFFT()
        self.conv1x1 = Conv1x1(in_channels)
        self.c_ifft = ComplexIFFT()

    def forward(self, x):
        real, imag = self.c_fft(x)

        combined = torch.cat([real, imag], dim=1)
        conv_out = self.conv1x1(combined)

        out_channels = conv_out.shape[1] // 2
        real_out = conv_out[:, :out_channels, :, :]
        imag_out = conv_out[:, out_channels:, :, :]

        output = self.c_ifft(real_out, imag_out)

        return output


class SAI2E(nn.Module):
    def __init__(self, in_channels=3, train_patch=256, eps=0, ):
        super(SAI2E, self).__init__()
        self.in_channels = in_channels
        self.eps = eps
        if not isinstance(train_patch, list):
            self.train_patch = [train_patch, train_patch]
        else:
            self.train_patch = train_patch
        self.offset_predict = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, 4, 1, padding=0, bias=True),
            )
        self.modulation_predict = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, padding=0, bias=True),
            )

    def get_center_grid(self, x):
        B, _, H, W = x.shape
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing='xy'), dim=-1).type(x.dtype).to(x.device)
        norm_coords = coords / torch.tensor([W, H], dtype=x.dtype).to(x.device) * 2 - 1
        return norm_coords

    def forward(self, x, train_mode=False):
        Batch, _, H, W = x.shape

        # 生成积分图（各通道独立）
        integrated_x = torch.cumsum(x, dim=-1)
        integrated_x = torch.cumsum(integrated_x, dim=-2)

        # 获得中心坐标
        center_grid = self.get_center_grid(x).unsqueeze(0)
        normalizer = torch.tensor(
            [self.train_patch[0] / W, self.train_patch[1] / H], dtype=x.dtype, device=x.device).view(1, 1, 1, 2)

        subnet_output = self.offset_predict(x).permute(0, 2, 3, 1)

        off_w, off_h = torch.split(subnet_output, 2, dim=3)
        off_w = off_w - off_w.mean(dim=-1, keepdim=True)
        off_h = off_h - off_h.mean(dim=-1, keepdim=True)
        minimum_patch = 2
        off_w_min = torch.minimum(off_w.min(dim=-1, keepdim=True)[0],
                                  torch.zeros_like(off_w.min(dim=-1, keepdim=True)[0]) - minimum_patch /
                                  self.train_patch[0])
        off_w_max = torch.maximum(off_w.max(dim=-1, keepdim=True)[0],
                                  torch.zeros_like(off_w.max(dim=-1, keepdim=True)[0]) + minimum_patch /
                                  self.train_patch[0])
        off_h_min = torch.minimum(off_h.min(dim=-1, keepdim=True)[0],
                                  torch.zeros_like(off_h.min(dim=-1, keepdim=True)[0]) - minimum_patch /
                                  self.train_patch[1])
        off_h_max = torch.maximum(off_h.max(dim=-1, keepdim=True)[0],
                                  torch.zeros_like(off_h.max(dim=-1, keepdim=True)[0]) + minimum_patch /
                                  self.train_patch[1])

        area = (off_h_max - off_h_min) * (off_w_max - off_w_min) * self.train_patch[0] * self.train_patch[1] / 4
        area = area.view(Batch, 1, H, W).clip(1, H * W)
        scale = self.modulation_predict(x)
        if self.eps != 0:
            mask = (scale.abs() < self.eps)
            safe_sign = torch.where(scale >= 0, 1.0, -1.0)
            scale = torch.where(mask, safe_sign * self.eps, scale)
        area = area * scale

        off_tl = (torch.cat([off_w_min, off_h_min], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_tr = (torch.cat([off_w_max, off_h_min], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_bl = (torch.cat([off_w_min, off_h_max], dim=-1) * normalizer + center_grid).clip(-1, 1)
        off_br = (torch.cat([off_w_max, off_h_max], dim=-1) * normalizer + center_grid).clip(-1, 1)

        # 采样
        A = F.grid_sample(integrated_x, off_tl, align_corners=True, padding_mode='border', mode='bilinear')
        B = F.grid_sample(integrated_x, off_tr, align_corners=True, padding_mode='border', mode='bilinear')
        C = F.grid_sample(integrated_x, off_bl, align_corners=True, padding_mode='border', mode='bilinear')
        D = F.grid_sample(integrated_x, off_br, align_corners=True, padding_mode='border', mode='bilinear')

        res = (A + D - B - C) / area
        return res


# Axis-based Multi-head Self-Attention
class NextAttentionImplZ(nn.Module):
    def __init__(self, num_dims, num_heads, bias):
        super(NextAttentionImplZ, self).__init__()
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.q1 = nn.Conv2d(num_dims, num_dims * 3, kernel_size=1, bias=bias)
        self.q2 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)
        self.q3 = nn.Conv2d(num_dims * 3, num_dims * 3, kernel_size=3, padding=1, groups=num_dims * 3, bias=bias)

        self.fac = nn.Parameter(torch.ones(1))
        self.fin = nn.Conv2d(num_dims, num_dims, kernel_size=1, bias=bias)

        self.gate = nn.Sequential(
            nn.Conv2d(num_dims, num_dims // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_dims // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.size()
        n_heads, dim_head = self.num_heads, c // self.num_heads

        qkv = self.q3(self.q2(self.q1(x)))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'n (head c) h w -> n (head h) w c', head=n_heads, c=dim_head)
        k = rearrange(k, 'n (head c) h w -> n (head h) w c', head=n_heads, c=dim_head)
        v = rearrange(v, 'n (head c) h w -> n (head h) w c', head=n_heads, c=dim_head)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        _, _, C1, _ = q.shape

        gate_output = self.gate(x).view(n, -1).mean().clamp(0.1, 0.9)
        if torch.isnan(gate_output):
            gate_output = 0.5  # 默认值替换
        dynamic_k = max(int(C1 / 2), min(C1, int(C1 * gate_output)))
        mask1 = torch.zeros(n, self.num_heads*h, C1, C1, device=x.device, requires_grad=False)

        # fac = dim_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.fac
        index1 = torch.topk(attn.detach(), k=dynamic_k, dim=-1, largest=True)[1]
        mask1.scatter_(-1, index1, 1.)
        attn = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn = torch.softmax(attn, dim=-1)

        out1 = (attn @ v)
        out_att = rearrange(out1, 'n (nh h) w dh -> n (nh dh) h w', nh=n_heads, dh=dim_head, n=n, h=h)
        out_att = self.fin(out_att)
        del q, k, v, attn, out1, mask1, index1

        return out_att


# Axis-based Multi-head Self-Attention (row and col attention)
class NextAttentionZ(nn.Module):
    def __init__(self, num_dims, num_heads=8, bias=False):
        super(NextAttentionZ, self).__init__()
        assert num_dims % num_heads == 0
        self.num_dims = num_dims
        self.num_heads = num_heads
        self.row_att = NextAttentionImplZ(num_dims, num_heads, bias)
        self.col_att = NextAttentionImplZ(num_dims, num_heads, bias)

    def forward(self, x):
        x = self.row_att(x) + x
        x = x.transpose(-2, -1)
        x = self.col_att(x) + x
        x = x.transpose(-2, -1)

        return x


class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.spr_sa = spr_sa(dim // 2, 2)

        # self.light_aware_sprsa = LightAware_SPRSA(dim // 2, 2)  # 新加Light-Aware SPR-SA
        # self.ddpr_sa = DDPR_SA(dim // 2)                        # 新加ddpr_sa
        # self.sab = SAB()                                        # 新加SAB
        # self.sru = SRU(dim // 2)                                # 新加SRU
        # self.naf = NAFBlock(dim // 2)                           # 新加NAFBlock
        # self.enhanced_spatial_att = Enhanced_Spatial_Attention(dim // 2, 3)  # 新的增强空间注意力
        # self.smsa = SMSA(dim // 2)  # 新加
        # self.ham = HAM(dim // 2)  # 新加
        self.lia = LIA(dim // 2)  # 新加
        self.nextattentionz = NextAttentionZ(dim // 2, num_heads)

        self.linear_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim // 2, dim // 2 * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim // 2 * 3, dim // 2 * 3, kernel_size=3, stride=1, padding=1, groups=dim // 2 * 3,
                                    bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1),
            # nn.BatchNorm2d(dim // 8),
            nn.GroupNorm(1, dim // 4),  # 新加
            nn.GELU(),
            nn.Conv2d(dim // 4, dim // 2, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 8, kernel_size=1),
            # nn.BatchNorm2d(dim // 16),
            nn.GroupNorm(1, dim // 8),  # 新加
            nn.GELU(),
            nn.Conv2d(dim // 8, 1, kernel_size=1),
        )
        self.fft = Stage2_fft(in_channels=dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim // 2, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y, x = self.linear_0(x).chunk(2, dim=1)

        y_d = self.spr_sa(y)

        # y_d = self.light_aware_sprsa(y)  # 新加
        # y_d = self.ddpr_sa(y)            # 新加
        # y_d = self.sab(y)                # 新加
        # y_d = self.sru(y)                # 新加
        # y_d = self.naf(y)                # 新加
        # y_d = self.enhanced_spatial_att(y)  # 使用新的增强空间注意力
        # y_d = self.smsa(y)  # 新加
        # y_d = self.ham(y)
        # y_d = self.lia(y)
        # out_att = self.nextattentionz(x)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)

        _, _, C1, _ = q.shape
        _, _, C2, _ = k.shape

        # dynamic_k = max(int(C2 / 2), min(C2, int(C2 * self.gate(x).view(b, -1).mean().clamp(0.1, 0.9))))

        gate_output = self.gate(x).view(b, -1).mean().clamp(0.1, 0.9)

        if torch.isnan(gate_output):
            gate_output = 0.5  # 默认值替换
            dynamic_k = C2
        else:
            dynamic_k = max(int(C2 / 2), min(C2, int(C2 * gate_output)))

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask1 = torch.zeros(b, self.num_heads, C1, C2, device=x.device, requires_grad=False)
        index1 = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask1.scatter_(-1, index1, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn1 = torch.nan_to_num(attn1, nan=0.0, posinf=0.0, neginf=0.0)

        out1 = (attn1 @ v)
        out2 = (attn1 @ v)
        out3 = (attn1 @ v)
        out4 = (attn1 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_att = self.adjust_conv(out_att)  # 新加

        # Frequency Adaptive Interaction Module (FAIM)
        # stage1
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(out_att)
        # S-Map (before sigmoid)
        spatial_map = self.spatial_interaction(y_d)

        # S-I
        attened_x = out_att * torch.sigmoid(spatial_map)
        # C-I
        conv_x = y_d * torch.sigmoid(channel_map)

        x = torch.cat([attened_x, conv_x], dim=1)
        # x = torch.cat([out_att, y_d], dim=1)
        out = self.project_out(x)
        # out = self.project_out(out_att)
        # stage 2
        out = self.fft(out)
        return out


class Attention2(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention2, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.nextattentionz = NextAttentionZ(dim, num_heads)

        self.linear_0 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)

        # self.svp_q_dwconv = nn.Conv2d(3, 3 * self.num_heads, kernel_size=1, bias=bias)  # 新加
        # self.adjust_conv = nn.Conv2d(dim + 3*self.num_heads, dim, kernel_size=1, bias=bias)  # 新加
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, kernel_size=1),  # 输出动态 K
            nn.Sigmoid()
        )

    def forward(self, x, svp_img):
        b, c, h, w = x.shape

        # out_att = self.nextattentionz(x)
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # svp_q = self.svp_q_dwconv(svp_img)  # 新加
        # q = torch.concat((q, svp_q), dim=1)  # 新加

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)

        _, _, C1, _ = q.shape
        _, _, C2, _ = k.shape

        gate_output = self.gate(x).view(b, -1).mean().clamp(0.1, 0.9)

        if torch.isnan(gate_output):
            dynamic_k = C2  # 默认值替换
        else:
            dynamic_k = max(int(C2 / 2), min(C2, int(C2 * gate_output)))

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        mask1 = torch.zeros(b, self.num_heads, C1, C2, device=x.device, requires_grad=False)
        index1 = torch.topk(attn, k=dynamic_k, dim=-1, largest=True)[1]
        mask1.scatter_(-1, index1, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn1 = torch.nan_to_num(attn1, nan=0.0, posinf=0.0, neginf=0.0)

        out1 = (attn1 @ v)
        out2 = (attn1 @ v)
        out3 = (attn1 @ v)
        out4 = (attn1 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out_att = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_att = self.adjust_conv(out_att)  # 新加

        out = self.project_out(out_att)

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.svp_q_dwconv = nn.Conv2d(3, 3 * self.num_heads, kernel_size=1, bias=bias)  # 新加
        self.adjust_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 新加

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.adjust_conv(out)  # 新加

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, in_channels):
        super(TransformerBlock, self).__init__()

        # self.lie = LIE(in_channels=in_channels, out_channels=in_channels)
        self.lie = spr_sa(dim=in_channels)  # 新加

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        # self.attn = NextAttentionZ(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        # self.ffn = EDFFN(dim, ffn_expansion_factor, bias)  # 新加

    def forward(self, x):
        lie = self.lie(x)
        x = x + lie

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


# dim,heads,exp,in_channels
class Conv_Transformer(nn.Module):

    def __init__(self, dim, num_heads, expand_ratio, in_channels):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.doubleconv = DoubleConv(in_channels=in_channels, out_channels=in_channels)
        self.Transformer = TransformerBlock(dim, num_heads, expand_ratio, bias=False, in_channels=in_channels)
        self.channel_reduce = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1,
                                        stride=1)
        self.Conv_out = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1))

    def forward(self, x):
        conv = self.lrelu(self.doubleconv(x))
        trans = self.Transformer(x)
        x = torch.cat([conv, trans], 1)
        x = self.channel_reduce(x)
        x = self.lrelu(self.Conv_out(x))
        return x


class simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size=3, D_mul=None, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros', simam=False):
        super(DOConv2d, self).__init__()

        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))
        self.simam = simam
        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            D_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.D_diag = Parameter(torch.cat([D_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.D_diag = Parameter(D_diag, requires_grad=False)
        ##################################################################################################
        if simam:
            self.simam_block = simam_module()
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.D_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            DoW = torch.reshape(self.W, DoW_shape)
        if self.simam:
            DoW_h1, DoW_h2 = torch.chunk(DoW, 2, dim=2)
            DoW = torch.cat([self.simam_block(DoW_h1), DoW_h2], dim=2)

        return self._conv_forward(input, DoW)


class BasicConv_do(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bias=False, norm=False, relu=True,
                 transpose=False, relu_method=nn.ReLU, groups=1, norm_method=nn.BatchNorm2d):
        super(BasicConv_do, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
                         groups=groups))
        if norm:
            layers.append(norm_method(out_channel))
        if relu:
            if relu_method == nn.ReLU:
                layers.append(nn.ReLU(inplace=True))
            elif relu_method == nn.LeakyReLU:
                layers.append(nn.LeakyReLU(inplace=True))
            else:
                layers.append(relu_method())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class SEAttention(nn.Module):

    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(2, 1, 5, 1, 2),
            nn.Sigmoid())
        self.processpha = nn.Sequential(
            nn.Conv2d(2, 1, 5, 1, 2),
            nn.Sigmoid())

        self.process1 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True))
        self.process2 = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.BatchNorm2d(nc),
            nn.ReLU(inplace=True))

    def forward(self, x):
        _, _, H, W = x.shape
        # x = self.proj(x)
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        pha = self.process2(pha)
        mag_avg = torch.mean(mag, dim=1, keepdim=True)
        mag_max = torch.max(mag, dim=1, keepdim=True)[0]
        pha_avg = torch.mean(pha, dim=1, keepdim=True)
        pha_max = torch.max(pha, dim=1, keepdim=True)[0]
        mag = mag * self.processmag(torch.cat([mag_avg, mag_max], dim=1))
        pha = pha * self.processpha(torch.cat([pha_avg, pha_max], dim=1))
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        return x_freq_spatial


class ResBlock_do_fft_bench(nn.Module):
    def __init__(self, out_channel, norm='backward'):
        super(ResBlock_do_fft_bench, self).__init__()

        self.main = nn.Sequential(
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv_do(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

        self.main_fft = nn.Sequential(
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=True),
            BasicConv_do(out_channel * 2, out_channel * 2, kernel_size=1, stride=1, relu=False)
        )
        self.CA = SEAttention(channel=out_channel * 2, reduction=8)
        self.conv1 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1)
        self.dim = out_channel
        self.norm = norm

        self.conv33conv11 = FreBlock(out_channel)  # 新加

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        # y = torch.fft.rfft2(x, norm=self.norm)
        # y_imag = y.imag
        # y_real = y.real
        # y_f = torch.cat([y_real, y_imag], dim=dim)
        # y = self.main_fft(y_f)
        # y_real, y_imag = torch.chunk(y, 2, dim=dim)
        # y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.conv33conv11(x)  # 新加

        conv = self.main(x)

        ft = torch.cat([y, conv], 1)
        res = torch.cat([y, conv], 1)

        # ft = self.CA(ft)
        ft = self.conv1(ft)

        res = self.conv2(res)
        return ft + x + res


class ADF(nn.Module):  # 原始
    def __init__(self, in_channels):
        super(ADF, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        x_down = self.down(x)
        g_down = self.down(depth_map)

        g = F.max_pool2d(self.g(g_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)

        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)

        Ra = F.softmax(torch.bmm(theta, phi), 2)
        y = torch.bmm(Ra, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.interpolate(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


# 第一种改法
class Mlpgated(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(Mlpgated, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        if x.size(1) != self.project_in.in_channels:
            raise ValueError(f"Expected input channels {self.project_in.in_channels}, but got {x.size(1)}")
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# Cross-Attention
class Mutual_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape, 'The shape of feature maps from image and fourier branch are not equal!'

        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Cross_ChannelAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(Cross_ChannelAttentionTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Mutual_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = LayerNorm(dim)

        self.ffn = Mlpgated(dim, ffn_expansion_factor=2, bias=bias)

    def forward(self, image, event):
        assert image.shape == event.shape, 'the shape of image doesnt equal to event'
        b, c, h, w = image.shape
        fused = image + self.attn(self.norm1(image), self.norm1(event))
        # print(fused.shape)
        # mlp
        # print(fused.shape)
        fused_norm = self.norm2(fused)
        ffn_fused = self.ffn(fused_norm, h, w)
        fused = fused + ffn_fused

        return fused


# 第二种改法（gpt)
class EADF(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(EADF, self).__init__()

        inter_channels = in_channels // reduction

        # 空间域映射
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        # 频域映射（支持多通道）
        self.freq_proj = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        # 通道融合注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inter_channels, inter_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels // 4, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Conv2d(inter_channels, 1, kernel_size=7, padding=3)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        # 输出映射
        self.out_conv = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x, freq):
        n, c, h, w = x.size()
        x_down = self.down(x)
        g_down = self.down(freq)
        # 投影
        theta = self.theta(x_down)  # [N, C/2, H, W]
        phi = self.phi(x_down)
        g = self.g(x_down)

        # 频率引导投影
        freq_embed = self.freq_proj(g_down)
        # freq_embed = F.interpolate(freq_embed, size=(h, w), mode='bilinear', align_corners=False)

        # 空间注意力（基于频率引导）
        attn_spatial = torch.sigmoid(self.spatial_att(freq_embed))
        g_weighted = g * attn_spatial  # 频率引导空间特征

        # 自注意力计算（像素关系）
        theta_flat = theta.view(n, c // 2, -1).transpose(1, 2)  # [N, HW, C/2]

        phi_flat = phi.view(n, c // 2, -1)  # [N, C/2, HW]
        attn = torch.bmm(theta_flat, phi_flat) / phi_flat.shape[-1] ** 0.5
        attn = F.softmax(attn, dim=-1)

        y = torch.bmm(attn, g_weighted.view(n, c // 2, -1).transpose(1, 2))
        y = y.transpose(1, 2).contiguous().view(n, c // 2, h // 4, -1)

        # 通道注意力（自适应频率调制）
        ch_att = self.channel_att(y)
        y = y * ch_att

        # 输出融合
        out = self.out_conv(y)
        return x + F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)


# 第三种改法
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        height, width = ffted.shape[-2:]
        coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
        coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
        ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)
        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])
        ifft_shape_slice = x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm='ortho')
        return output


class SpectralTransform(nn.Module):
    def __init__(self, ch):
        super(SpectralTransform, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.fu = FourierUnit(ch, ch)
        self.conv2 = nn.Conv2d(ch * 2, ch, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        x = self.conv2(torch.cat([x, x2], dim=1))
        return x


class FFC(nn.Module):
    def __init__(self, ch):
        super(FFC, self).__init__()
        self.convl2l = nn.Conv2d(ch, ch, 3, 1, 1)
        self.convl2g = nn.Conv2d(ch, ch, 3, 1, 1)
        self.convg2l = nn.Conv2d(ch, ch, 3, 1, 1)
        self.convg2g = SpectralTransform(ch)

    def forward(self, x_l, x_g):
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class SFIB(nn.Module):
    def __init__(self, ch):
        super(SFIB, self).__init__()
        self.ffc = FFC(ch)
        self.bn_l = nn.BatchNorm2d(ch)
        self.bn_g = nn.BatchNorm2d(ch)
        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x_l, x_g = self.ffc(x, y)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class SFIADF(nn.Module):
    def __init__(self, in_channels):
        super(SFIADF, self).__init__()
        self.sfib = SFIB(in_channels)

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x, depth_map):
        n, c, h, w = x.size()

        # x, depth_map = self.sfib(x, depth_map)

        x_down = self.down(x)
        g_down = self.down(depth_map)
        x_down, g_down = self.sfib(x_down, g_down)

        g = F.max_pool2d(self.g(g_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)

        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)

        Ra = F.softmax(torch.bmm(theta, phi), 2)
        y = torch.bmm(Ra, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.interpolate(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


# 第四种改法
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class InvBlock(nn.Module):
    def __init__(self, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = UNetConvBlock(self.split_len2, self.split_len1)
        self.G = UNetConvBlock(self.split_len1, self.split_len2)
        self.H = UNetConvBlock(self.split_len1, self.split_len2)

        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x):
        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = InvBlock(nc, nc // 2)

    def forward(self, x):
        yy = self.block(x)

        return x + yy


class FreBlock1(nn.Module):
    def __init__(self, nc):
        super(FreBlock1, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class ProcessBlock(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlock, self).__init__()
        self.spatial_process = SpaBlock(in_nc)
        self.frequency_process = FreBlock1(in_nc)

    def forward(self, x, depth_map):
        _, _, H, W = depth_map.shape
        x_freq = torch.fft.rfft2(depth_map, norm='backward')
        x = self.spatial_process(x)
        x_freq = self.frequency_process(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        return x, x_freq_spatial


class SFDADF(nn.Module):
    def __init__(self, in_channels):
        super(SFDADF, self).__init__()
        self.sfd = ProcessBlock(in_channels)

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x, depth_map):
        n, c, h, w = x.size()

        x, depth_map = self.sfd(x, depth_map)

        x_down = self.down(x)
        g_down = self.down(depth_map)
        # x_down, g_down = self.sfib(x_down, g_down)

        g = F.max_pool2d(self.g(g_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)

        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)

        d_k = theta.shape[-1]
        Ra = F.softmax(torch.bmm(theta, phi) / (math.sqrt(d_k) + 1e-8), 2)
        y = torch.bmm(Ra, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.interpolate(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


# 第五种改法
class MDAF(nn.Module):
    def __init__(self, dim, num_heads):
        super(MDAF, self).__init__()
        self.num_heads = num_heads

        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv1_1_1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv1_1_2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1_1_3 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        # self.conv1_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        # self.conv1_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        # self.conv1_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv2_1_1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv2_1_3 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        # self.conv2_2_1 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        # self.conv2_2_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        # self.conv2_2_3 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        attn_111 = self.conv1_1_1(x1)
        attn_112 = self.conv1_1_2(x1)
        attn_113 = self.conv1_1_3(x1)
        # attn_121 = self.conv1_2_1(x1)
        # attn_122 = self.conv1_2_2(x1)
        # attn_123 = self.conv1_2_3(x1)

        attn_211 = self.conv2_1_1(x2)
        attn_212 = self.conv2_1_2(x2)
        attn_213 = self.conv2_1_3(x2)
        # attn_221 = self.conv2_2_1(x2)
        # attn_222 = self.conv2_2_2(x2)
        # attn_223 = self.conv2_2_3(x2)

        out1 = (attn_111 + attn_112 + attn_113
                # + attn_121 + attn_122 + attn_123
                )
        out2 = (attn_211 + attn_212 + attn_213
                # + attn_221 + attn_222 + attn_223
                )
        out1 = self.project_out(out1)
        out2 = self.project_out(out2)
        k1 = rearrange(out1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1 + x2

        return out


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x
        # H = torch.cat([x_lh, x_hl, x_hh], dim=1)
        # return (x_ll, H)
        # return x_ll, x_lh, x_hl, x_hh
    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H // 2, W // 2)

            dx = dx.transpose(1, 2).reshape(B, -1, H // 2, W // 2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)


class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
        # (L, H) = DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
        # return (L, H)


class FDCFormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_heads=8, expand_ratio=4, ):
        super(FDCFormer, self).__init__()

        self.pixelunshuffle = nn.PixelUnshuffle(2)  # 1,12,64,64
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

        self.net = nn.Conv2d(inp_channels * 4, dim, kernel_size=3, stride=1, padding=1)
        self.ft = ResBlock_do_fft_bench(out_channel=dim, )

        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, dim, 4, stride=2, padding=1),  # kernel_size由4改为3
            nn.GroupNorm(num_groups=dim, num_channels=dim),
            # nn.SELU(inplace=True)
            nn.SiLU(inplace=True)  # 新加
        )

        self.ft1 = ResBlock_do_fft_bench(out_channel=dim)
        self.down1_1 = Downsample(dim)
        self.ft2 = ResBlock_do_fft_bench(out_channel=dim * 2)
        self.dowm2_2 = Downsample(dim * 2)
        self.ft3 = ResBlock_do_fft_bench(out_channel=dim * 4)
        self.dowm3_3 = Downsample(dim * 4)
        self.ft4 = ResBlock_do_fft_bench(out_channel=dim * 8)
        self.conv_fuss = nn.Conv2d(int(dim * 3), int(dim), kernel_size=1)

        self.ft5 = ResBlock_do_fft_bench(out_channel=dim * 8)
        self.u1 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.ft6 = ResBlock_do_fft_bench(out_channel=dim * 4)
        self.u2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.ft7 = ResBlock_do_fft_bench(out_channel=dim * 2)
        self.u3 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.ft8 = ResBlock_do_fft_bench(out_channel=dim)
        self.depth_pred = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            #     nn.Conv2d(dim, 1, kernel_size=1, stride=1),
            #     # nn.BatchNorm2d(1),
            #     # nn.GELU()
            #     nn.Sigmoid()
        )

        # self.svp = SAI2E(in_channels=inp_channels, train_patch=128, eps=0)  # 新加
        # self.svp_down1_2 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=False, groups=inp_channels)
        # self.svp_down2_3 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=False, groups=inp_channels)
        # self.svp_down3_4 = nn.Conv2d(inp_channels, inp_channels, 4, 2, 1, bias=False, groups=inp_channels)

        self.conv_tran1 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.down1 = Downsample(dim)
        self.conv_tran2 = Conv_Transformer(dim * 2, num_heads, expand_ratio, in_channels=dim * 2,
                                           )  # h/2
        self.down2 = Downsample(dim * 2)
        self.conv_tran3 = Conv_Transformer(dim * 4, num_heads, expand_ratio, in_channels=dim * 4,
                                           )  # h/4
        self.down3 = Downsample(dim * 4)
        self.conv_tran4 = Conv_Transformer(dim * 8, num_heads, expand_ratio, in_channels=dim * 8, )  # h/8

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.channel_reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.conv_tran5 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.up2 = nn.ConvTranspose2d(dim * 4, dim, kernel_size=4, stride=4)
        self.channel_reduce2 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.conv_tran6 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.up3 = nn.ConvTranspose2d(dim * 8, dim, kernel_size=8, stride=8)
        self.channel_reduce3 = nn.Conv2d(dim * 2, dim, kernel_size=1, stride=1)
        self.conv_tran7 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        self.conv_tran8 = Conv_Transformer(dim, num_heads, expand_ratio, in_channels=dim,
                                           )  # h
        # self.adf = ADF(dim)  # 原始

        # self.cross_attention_block = Cross_ChannelAttentionTransformerBlock(
        #     dim=dim, num_heads=num_heads, bias=False)  # 新加
        # self.eadf = EADF(dim)  # 新加
        # self.sfiadf = SFIADF(dim)  # 新加
        # self.sfd = SFDADF(dim)  # 新加
        self.mdaf = MDAF(dim, num_heads)  # 新加

        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, stride=1, padding=1)

        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x1 = self.pixelunshuffle(x)

        x1 = self.lrelu(self.net(x1))  # 8 48 64 64

        # svp_img_1 = self.svp(x)

        # x = (x - self.mean) / self.std
        # x = self.conv1(x)
        # d_f1 = self.ft1(x)  # d
        #
        # d_f2 = self.down1_1(d_f1)  # 2d
        # d_f2 = self.ft2(d_f2)
        #
        # d_f3 = self.dowm2_2(d_f2)  # 4d
        # d_f3 = self.ft3(d_f3)
        #
        # # d_f4 = self.dowm3_3(d_f3)  # 8d
        # # d_f4 = self.ft4(d_f4)
        # d_f4 = self.ft3(d_f3)  # 新加
        #
        # # d_f5 = self.ft5(d_f4)
        # d_f5 = self.ft3(d_f4)  # 新加
        #
        # # d_f6 = self.u1(d_f5)  # 4d
        # # d_f6 = self.ft6(d_f6 + d_f3)
        # d_f6 = self.ft6(d_f5 + d_f3)  # 新加
        #
        # d_f7 = self.u2(d_f6)  # 2d
        # d_f7 = self.ft7(d_f7 + d_f2)
        #
        # d_f8 = self.u3(d_f7)  # d
        #
        # depth_pred = self.depth_pred(d_f8 + d_f1)
        # svp_img_2 = self.svp_down1_2(svp_img_1)  # 新加
        conv_tran1 = self.conv_tran1(x1)
        pool1 = self.down1(conv_tran1)
        # svp_img_3 = self.svp_down2_3(svp_img_2)  # 新加
        conv_tran2 = self.conv_tran2(pool1)
        pool2 = self.down2(conv_tran2)
        # svp_img_4 = self.svp_down3_4(svp_img_3)  # 新加
        conv_tran3 = self.conv_tran3(pool2)
        # pool3 = self.down3(conv_tran3)

        # conv_tran4 = self.conv_tran4(pool3)
        conv_tran4 = self.conv_tran3(conv_tran3)  # 新加

        up1 = self.up1(conv_tran2)
        concat1 = torch.cat([up1, conv_tran1, ], 1)
        concat1_1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1_1)

        up2 = self.up2(conv_tran3)
        concat2 = torch.cat([up2, conv_tran5], 1)
        concat2_2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2_2)

        # up3 = self.up3(conv_tran4)
        up3 = self.up2(conv_tran4)  # 新加

        concat3 = torch.cat([up3, conv_tran6], 1)
        concat3_3 = self.channel_reduce2(concat3)
        conv_tran7 = self.conv_tran7(concat3_3)

        conv_tran8 = self.conv_tran8(conv_tran7)

        # f = self.adf(conv_tran8, depth_pred)

        # f = self.cross_attention_block(conv_tran8, depth_pred)  # 新加
        # f = self.eadf(conv_tran8, depth_pred)  # 新加
        # f = self.sfiadf(conv_tran8, depth_pred)  # 新加
        # f = self.sfd(conv_tran8, depth_pred)  # 新加
        # f = self.mdaf(conv_tran8, depth_pred)  # 新加

        # conv_out = self.lrelu(self.conv_out(f))

        conv_out = self.lrelu(self.conv_out(conv_tran8))  # 新加

        out = self.pixelshuffle(conv_out)

        return out
