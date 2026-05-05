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


class LFE(nn.Module):
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


# DGFFN
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
        x1 = F.gelu(x1) * x1
        x2 = F.gelu(x2) * x2
        x = self.project_out(x1+x2)
        return x
        

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


# Axis-based Dynamic Sparse Attention 
class ADSA(nn.Module):
    def __init__(self, num_dims, num_heads=8, bias=False):
        super(ADSA, self).__init__()
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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, in_channels):
        super(TransformerBlock, self).__init__()
        self.lfe = LFE(dim=in_channels)  

        self.norm1 = LayerNorm(dim)
        self.attn = ADSA(dim, num_heads, bias)
        
        self.norm2 = LayerNorm(dim)
        self.ffn = DGFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        lfe = self.lfe(x)
        x = x + lfe

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
        
        self.conv1 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=1)
        self.dim = out_channel
        self.norm = norm

        self.conv33conv11 = FreBlock(out_channel) 

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
    
        y = self.conv33conv11(x)  

        conv = self.main(x)

        ft = torch.cat([y, conv], 1)
        res = torch.cat([y, conv], 1)

        ft = self.conv1(ft)

        res = self.conv2(res)
        return ft + x + res


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

        self.conv2_1_1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv2_1_2 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv2_1_3 = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

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


class FDSA(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=48, num_heads=8, expand_ratio=4, ):
        super(FDSA, self).__init__()

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
            nn.SELU(inplace=True)
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
        
        self.mdaf = MDAF(dim, num_heads)  

        self.conv_out = nn.Conv2d(dim, out_channels * 4, kernel_size=3, stride=1, padding=1)

        self.pixelshuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x1 = self.pixelunshuffle(x)

        x1 = self.lrelu(self.net(x1))  

        x = (x - self.mean) / self.std
        x = self.conv1(x)
        d_f1 = self.ft1(x)  # d
        
        d_f2 = self.down1_1(d_f1)  # 2d
        d_f2 = self.ft2(d_f2)
        
        d_f3 = self.dowm2_2(d_f2)  # 4d
        d_f3 = self.ft3(d_f3)
        
        d_f4 = self.dowm3_3(d_f3)  # 8d
        d_f4 = self.ft4(d_f4)
        
        d_f5 = self.ft5(d_f4)
        
        d_f6 = self.u1(d_f5)  # 4d
        d_f6 = self.ft6(d_f6 + d_f3)
        
        d_f7 = self.u2(d_f6)  # 2d
        d_f7 = self.ft7(d_f7 + d_f2)
        
        d_f8 = self.u3(d_f7)  # d
        
        depth_pred = self.depth_pred(d_f8 + d_f1)
        
        conv_tran1 = self.conv_tran1(x1)
        pool1 = self.down1(conv_tran1)
        
        conv_tran2 = self.conv_tran2(pool1)
        pool2 = self.down2(conv_tran2)
       
        conv_tran3 = self.conv_tran3(pool2)
        pool3 = self.down3(conv_tran3)

        conv_tran4 = self.conv_tran4(pool3)

        up1 = self.up1(conv_tran2)
        concat1 = torch.cat([up1, conv_tran1, ], 1)
        concat1_1 = self.channel_reduce1(concat1)
        conv_tran5 = self.conv_tran5(concat1_1)

        up2 = self.up2(conv_tran3)
        concat2 = torch.cat([up2, conv_tran5], 1)
        concat2_2 = self.channel_reduce2(concat2)
        conv_tran6 = self.conv_tran6(concat2_2)

        up3 = self.up3(conv_tran4)

        concat3 = torch.cat([up3, conv_tran6], 1)
        concat3_3 = self.channel_reduce3(concat3)
        conv_tran7 = self.conv_tran7(concat3_3)

        conv_tran8 = self.conv_tran8(conv_tran7)

        f = self.mdaf(conv_tran8, depth_pred)  

        conv_out = self.lrelu(self.conv_out(f))

        out = self.pixelshuffle(conv_out)

        return out
