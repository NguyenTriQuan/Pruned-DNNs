# from tkinter.tix import Tree
# from turtle import forward
# from regex import D
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal
import numpy as np
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, dropout
from layers.sparse_layer import SparseLinear, SparseConv2D, _SparseLayer

from utils import *
import sys
from arguments import get_args
args = get_args()
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class _SparseModel(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(_SparseModel, self).__init__()

    def forward(self, x):
        for module in self.layers:
            x = module(x)
        return x

    def count_params(self):
        count = 0
        total = 0
        print('num neurons:', end=' ')
        for m in self.SL:
            c, t = m.count_params()
            count += c
            total += t
            print(m.out_features, end=' ')

        return count, total

    def PGD_lasso(self, lr, lamb):
        for m in self.SL:
            m.PGD_lasso(lr, lamb, self.total_strength)

        self.get_strength()

    def PGD_group_lasso(self, lr, lamb):
        for m in self.SL:
            m.PGD_group_lasso(lr, lamb, self.total_strength)

    def squeeze(self, optim_state):
        mask_in = None
        self.total_strength = 1
        for i, m in enumerate(self.SL[:-1]):
            mask_out = self.SL[i].mask_out * self.SL[i+1].mask_in
            m.squeeze(optim_state, mask_in, mask_out)
            mask_in = mask_out
            self.total_strength += self.SL[i].strength + self.SL[i+1].strength
        self.SL[-1].squeeze(optim_state, mask_in, None)

    def get_strength(self):
        self.total_strength = 1
        for i, m in enumerate(self.SL[:-1]):
            self.total_strength += self.SL[i].strength + self.SL[i+1].strength
    
    def report(self):
        for m in self.SL:
            print(m.__class__.__name__, m.in_features, m.out_features)
            
class MLP(_SparseModel):

    def __init__(self, input_size, output_size, mul=1, norm_type=None):
        super(MLP, self).__init__()
        self.mul = mul
        self.input_size = input_size
        N = 400
       
        self.layers = nn.ModuleList([
            nn.Flatten(),
            SparseLinear(np.prod(input_size), N, first_layer=True, bias=True, norm_type=norm_type),
            nn.ReLU(),
            SparseLinear(N, N, bias=True, norm_type=norm_type),
            nn.ReLU(),
            SparseLinear(N, output_size, bias=True, last_layer=True),
            ])
        
        self.SL = [m for m in self.layers if isinstance(m, _SparseLayer)]
        self.get_strength()

class VGG(_SparseModel):
    '''
    VGG model 
    '''
    def __init__(self, input_size, output_size, cfg, norm_type=None, mul=1):
        super(VGG, self).__init__()

        nchannels, size, _ = input_size

        self.layers = make_layers(cfg, nchannels, norm_type=norm_type, mul=mul)

        self.p = 0.1
        s = size
        for m in self.layers:
            if isinstance(m, SparseConv2D):
                s = compute_conv_output_size(s, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
            elif isinstance(m, nn.MaxPool2d):
                s = compute_conv_output_size(s, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            SparseLinear(int(512*s*s*mul), int(4096*mul), s=s),
            nn.ReLU(True),
            SparseLinear(int(4096*mul), int(4096*mul)),
            nn.ReLU(True),
            SparseLinear(int(4096*mul), output_size, last_layer=True),
        ])

        self.SL = [m for m in self.modules() if isinstance(m, _SparseLayer)]
        self.get_strength()


def make_layers(cfg, nchannels, norm_type=None, bias=True, mul=1):
    layers = []
    in_channels = nchannels
    layers += [SparseConv2D(in_channels, int(cfg[0]*mul), kernel_size=3, padding=1, norm_type=norm_type, bias=bias, first_layer=True), nn.ReLU(inplace=True)]
    in_channels = int(cfg[0]*mul)
    for v in cfg[1:]:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v*mul)
            layers += [SparseConv2D(in_channels, v, kernel_size=3, padding=1, norm_type=norm_type, bias=bias), nn.ReLU(inplace=True)]
            in_channels = v

    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def VGG11(input_size, output_size, norm_type=None):
    """VGG 11-layer model (configuration "A")"""
    return VGG(input_size, output_size, cfg['A'], norm_type=norm_type)

def VGG11_small(input_size, output_size, norm_type):
    return VGG(input_size, output_size, cfg['A'], norm_type=norm_type, mul=0.5)

def VGG13(input_size, output_size, norm_type):
    """VGG 13-layer model (configuration "B")"""
    return VGG(input_size, output_size, cfg['B'], norm_type=norm_type)

def VGG16(input_size, output_size, norm_type):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_size, output_size, cfg['C'], norm_type=norm_type)

def VGG16_small(input_size, output_size, norm_type):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_size, output_size, cfg['C'], norm_type=norm_type, mul=0.5)

def VGG19(input_size, output_size, norm_type):
    """VGG 19-layer model (configuration "E")"""
    return VGG(input_size, output_size, cfg['D'], norm_type=norm_type)


'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return SparseConv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return SparseConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(_SparseModel):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.ModuleList([
            SparseConv2D(in_planes, planes, kernel_size=3, 
                                stride=stride, padding=1, bias=False, norm_type=norm_type),
            nn.ReLU(),
            SparseConv2D(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, norm_type=norm_type)
        ])

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = SparseConv2D(in_planes, self.expansion*planes,
                            kernel_size=1, stride=stride, bias=False, norm_type=norm_type)
        else:
            self.shortcut = None

    def forward(self, x):
        out = x.clone()
        for module in self.layers:
            out = module(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x

        return F.relu(out)

class Bottleneck(_SparseModel):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type=None):
        super(Bottleneck, self).__init__()

        self.layers = nn.ModuleList([
            SparseConv2D(in_planes, planes, kernel_size=1, bias=False, norm_type=norm_type),
            nn.ReLU(),
            SparseConv2D(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, norm_type=norm_type),
            nn.ReLU(),
            SparseConv2D(planes, self.expansion * planes, 
                                kernel_size=1, bias=False, norm_type=norm_type)
        ])

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = SparseConv2D(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, norm_type=norm_type)
        else:
            self.shortcut = None


    def forward(self, x):
        out = x.clone()
        for module in self.layers:
            out = module(out)

        if self.shortcut:
            out += self.shortcut(x)
        else:
            out += x

        return F.relu(out)

class ResNet(_SparseModel):
    def __init__(self, block, num_blocks, norm_type, input_size, output_size, nf=64):
        super(ResNet, self).__init__()
        n_channels, in_size, _ = input_size
        s_mid = 1

        self.in_planes = nf

        self.conv1 = SparseConv2D(n_channels, nf*1, kernel_size=3,
                               stride=1, padding=1, bias=False, norm_type=norm_type, first_layer=True)
        self.blocks = self._make_layer(block, nf*1, num_blocks[0], stride=1, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*2, num_blocks[1], stride=2, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*4, num_blocks[2], stride=2, norm_type=norm_type)
        self.blocks += self._make_layer(block, nf*8, num_blocks[3], stride=2, norm_type=norm_type)
        self.linear = SparseLinear(nf*8*block.expansion*s_mid*s_mid, output_size, last_layer=True, s=s_mid)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.SL = [m for m in self.modules() if isinstance(m, _SparseLayer)]
        self.get_strength()


    def _make_layer(self, block, planes, num_blocks, stride, norm_type):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(block(self.in_planes, planes, stride, norm_type))
            self.in_planes = planes * block.expansion
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool(out)
        for block in self.blocks:
            out = block(out)

        out = self.avgpool(out)
        # out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def ResNet18(input_size, output_size, norm_type=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_type, input_size, output_size, nf=64)

def ResNet34(input_size, output_size, norm_type=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_type, input_size, output_size, nf=64)

def ResNet50(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], norm_type, input_size, output_size, nf=64)

def ResNet101(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], norm_type, input_size, output_size, nf=64)

def ResNet152(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], norm_type, input_size, output_size, nf=64)

