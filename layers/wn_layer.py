from asyncio import current_task
from unittest import makeSuite
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli, LogNormal, Normal
import numpy as np
import random
from torch.nn.modules.utils import _single, _pair, _triple
from torch import Tensor, device, isin, seed
from typing import Optional, Any
from torch.nn import init
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from utils import *
from typing import Optional, List, Tuple, Union
import sys
from arguments import get_args
args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

class _WeightNormLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=False, activation='leaky_relu', norm_type=None):
        super(_WeightNormLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features       
        self.bias = nn.Parameter(torch.Tensor(self.out_features).uniform_(0, 0).to(device)) if bias else None
        self.activation = activation

        if norm_type:
            self.norm_layer = nn.BatchNorm2d(out_features, track_running_stats=False)
        else:
            self.norm_layer = None

    def initialize(self):
        
        fan_in, fan_out = _calculate_fan_in_and_fan_out(self.weight)
        if self.activation == 'leaky_relu':
            self.gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope)
            self.negative_slope = args.negative_slope
            # self.gain = math.sqrt(fan_in/self.weight.numel())
            # self.negative_slope = math.sqrt((2/(self.gain**2))-1)
            self.activation = nn.LeakyReLU(self.negative_slope, inplace=False)
            print(self.gain, self.negative_slope)
        elif self.activation == 'sigmoid':
            self.gain = torch.nn.init.calculate_gain('sigmoid', args.negative_slope)
            self.activation = nn.Sigmoid()
        else:
            self.gain = 1
            self.activation = nn.Identity()
            print(self.gain)

        fan_mode = fan_in if args.fan_mode == 'fan_in' else fan_out
        self.bound = self.gain / math.sqrt(fan_mode)
        nn.init.normal_(self.weight, 0, self.bound)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def normalize(self):
        mean = self.weight.mean(dim=self.norm_dim).detach().view(self.norm_view)
        std = self.weight.std(dim=self.norm_dim, unbiased=False).detach().view(self.norm_view)
        self.weight.data = self.bound * (self.weight.data - mean) / std


class WeightNormLinear(_WeightNormLayer):

    def __init__(self, in_features, out_features, bias=False, activation='leaky_relu', norm_type=None):
        super(WeightNormLinear, self).__init__(in_features, out_features, bias, activation, norm_type)

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features).to(device))
        self.initialize()
        self.norm_dim = (1)
        self.norm_view = (-1, 1)

    def forward(self, x):  
        out = self.activation(x)
        out = F.linear(x, self.weight, self.bias)
        return out
            
        
class _WeightNormConvNd(_WeightNormLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, bias, activation, norm_type):
        super(_WeightNormConvNd, self).__init__(in_features, out_features, bias, activation, norm_type)
        if in_features % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_features % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups


class WeightNormConv2D(_WeightNormConvNd):
    def __init__(self, in_features, out_features, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=False, activation='leaky_relu', norm_type=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(WeightNormConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, bias, activation, norm_type)

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features // self.groups, *self.kernel_size).to(device))
        self.initialize()
        self.norm_dim = (1, 2, 3)
        self.norm_view = (-1, 1, 1, 1)

    def forward(self, x): 
        out = self.activation(x)
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm_layer:
            out = self.norm_layer(out)
        return out

