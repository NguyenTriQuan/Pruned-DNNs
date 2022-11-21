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

def apply_mask_out(param, mask_out, optim_state):
    param.data = param.data[mask_out].clone()
    param.grad = None
    param_states = optim_state[param]
    for name, state in param_states.items():
        if isinstance(state, torch.Tensor):
            if len(state.shape) > 0:
                param_states[name] = state[mask_out].clone()

def apply_mask_in(param, mask_in, optim_state):
    param.data = param.data[:, mask_in].clone()
    param.grad = None
    param_states = optim_state[param]
    for name, state in param_states.items():
        if isinstance(state, torch.Tensor):
            if len(state.shape) > 0:
                param_states[name] = state[:, mask_in].clone()

class _SparseLayer(nn.Module):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(_SparseLayer, self).__init__()

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.in_features = in_features
        self.out_features = out_features
            
        self.weight = []
        self.bias = nn.Parameter(torch.Tensor(self.out_features).uniform_(0, 0).to(device)) if bias else None

        self.scale = []

        self.num_in = []
        self.num_out = []

        self.shape_out = [self.out_features]
        self.shape_in = [self.in_features]

        self.norm_type = norm_type

        if norm_type:
            self.norm_layer = nn.BatchNorm2d(self.out_features)
        else:
            self.norm_layer = None

        self.s = s

    def count_params(self):
        count = (self.weight != 0).sum() + (self.bias != 0).sum()
        total = self.weight.numel() + self.bias.numel()

        if self.bias is not None:
            count += (self.bias != 0).sum()
            total += self.bias.numel()
        if self.norm_layer:
            count += (self.norm_layer.weight != 0).sum() + (self.norm_layer.bias != 0).sum()
            total += self.norm_layer.weight.numel() + self.norm_layer.bias.numel()
        return count.item(), total

    def norm_in(self):
        norm = self.weight.norm(2, dim=self.dim_in)
        if self.bias is not None:
            norm = (norm ** 2 + self.bias ** 2) ** 0.5
        return norm

    def norm_out(self):   
        weight = self.weight
        if self.s != 1:
            weight = weight.view(self.out_features, int(self.in_features/self.s/self.s), self.s, self.s)
        return weight.norm(2, dim=self.dim_out)

    def PGD_group_lasso(self, lr, lamb, total_strength):
        eps = 0
        with torch.no_grad():
            strength = self.strength/total_strength
            if not self.last_layer:
                # group lasso weights in
                norm = self.norm_in()
                aux = 1 - lamb * lr * strength / norm
                aux = F.threshold(aux, 0, eps, False)
                self.mask_out = (aux > eps)
                self.weight.data *= aux.view(self.view_in)
                if self.bias is not None:
                    self.bias.data *= aux

                # group lasso affine weights
                if self.norm_layer:
                    if self.norm_layer.affine:
                        norm = (self.norm_layer.weight ** 2 + self.norm_layer.bias ** 2) ** 0.5
                        aux = 1 - lamb * lr * strength / norm
                        aux = F.threshold(aux, 0, eps, False)
                        self.mask_out *= (aux > eps)
                        self.norm_layer.weight.data *= aux
                        self.norm_layer.bias.data *= aux

            if not self.first_layer:
                # group lasso weights out
                norm = self.norm_out()
                aux = 1 - lamb * lr * strength / norm
                aux = F.threshold(aux, 0, eps, False)
                self.mask_in = (aux > eps)
                if self.s != 1:
                    aux = aux.view(-1, 1, 1).expand(aux.size(0), self.s, self.s).contiguous().view(-1)
                self.weight.data *= aux.view(self.view_out)  

    def PGD_lasso(self, lr, lamb, total_strength):
        def proximal_operator(param, strength):
            return torch.sign(param) * F.threshold(param.abs()-lr*lamb*strength, 0, eps, False)
        eps = 0
        with torch.no_grad():
            strength = self.strength/total_strength
            self.weight.data = proximal_operator(self.weight.data, strength)
            self.mask = (self.weight.data != 0)
            # self.scale = self.bound / self.weight.std().detach()
            std = self.weight.std().detach()
            mean = self.weight.mean().detach()
            self.weight.data = self.bound * (self.weight.data - mean) / std
            # self.weight.data = self.bound * (self.weight.data) / std
            # if self.bias is not None:
            #     self.bias.data = proximal_operator(self.bias.data, strength)

            # if self.norm_layer:
            #     if self.norm_layer.affine:
            #         self.norm_layer.weight.data = proximal_operator(self.norm_layer.weight.data, strength)
            #         self.norm_layer.bias.data = proximal_operator(self.norm_layer.bias.data, strength)

        self.strength = (self.weight != 0).sum().item()

    def squeeze(self, optim_state, mask_in=None, mask_out=None):

        if mask_out is not None:
            apply_mask_out(self.weight, mask_out, optim_state)
            self.out_features = self.weight.shape[0]
            if self.bias is not None:
                apply_mask_out(self.bias, mask_out, optim_state)

            if self.norm_layer:
                if self.norm_layer.affine:
                    apply_mask_out(self.norm_layer.weight, mask_out, optim_state)
                    apply_mask_out(self.norm_layer.bias, mask_out, optim_state)

                if self.norm_layer.track_running_stats:
                    self.norm_layer.running_mean = self.norm_layer.running_mean[mask_out]
                    self.norm_layer.running_var = self.norm_layer.running_var[mask_out]

                self.norm_layer.num_features = self.out_features
        
        if mask_in is not None:
            if self.s != 1:
                mask_in = mask_in.view(-1,1,1).expand(mask_in.size(0), self.s, self.s).contiguous().view(-1)
            apply_mask_in(self.weight, mask_in, optim_state)
            self.in_features = self.weight.shape[1]

        self.strength = self.weight.numel()


class SparseLinear(_SparseLayer):

    def __init__(self, in_features, out_features, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        super(SparseLinear, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)

        self.view_in = [-1, 1]
        self.view_out = [1, -1]
        self.dim_in = [1]
        self.dim_out = [0]

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features).to(device))
        gain = torch.nn.init.calculate_gain('relu', math.sqrt(5))
        self.bound = gain / math.sqrt(self.in_features)
        nn.init.normal_(self.weight, 0, self.bound)
        self.scale = self.bound / self.weight.std().detach()
        # nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        self.strength = self.weight.numel()
        self.mask = (self.weight.data != 0).detach()

    def forward(self, x):    
        x = F.linear(x, self.weight*self.mask, self.bias)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x
            
        
class _SparseConvNd(_SparseLayer):
    def __init__(self, in_features, out_features, kernel_size, 
                stride, padding, dilation, transposed, output_padding, groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout):
        super(_SparseConvNd, self).__init__(in_features, out_features, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)
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


class SparseConv2D(_SparseConvNd):
    def __init__(self, in_features, out_features, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, next_layers=[], bias=True, norm_type=None, s=1, first_layer=False, last_layer=False, dropout=0.0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SparseConv2D, self).__init__(in_features, out_features, kernel_size, 
                                            stride, padding, dilation, False, _pair(0), groups, next_layers, bias, norm_type, s, first_layer, last_layer, dropout)

        self.view_in = [-1, 1, 1, 1]
        self.view_out = [1, -1, 1, 1]
        self.dim_in = [1, 2, 3]
        self.dim_out = [0, 2, 3]

        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features // self.groups, *self.kernel_size).to(device))

        gain = torch.nn.init.calculate_gain('relu', math.sqrt(5))
        self.bound = gain / math.sqrt(self.in_features * np.prod(self.kernel_size))
        nn.init.normal_(self.weight, 0, self.bound)
        self.scale = self.bound / self.weight.std().detach()

        # nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        self.strength = self.weight.numel()
        self.mask = (self.weight.data != 0).detach()

    def forward(self, x):    
        x = F.conv2d(x, self.weight*self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x

