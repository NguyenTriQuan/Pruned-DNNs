import sys
import torch
import torch.nn.functional as F
from utils import *

from arguments import get_args
args = get_args()

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

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, input_size, output_size, cfg, mul=1, batch_norm=False, bias=True):
        super(VGG, self).__init__()
        mul = args.mul
        n_channels, size, _ = input_size
        self.layers = make_layers(cfg, n_channels, mul=mul, batch_norm=batch_norm, bias=bias)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(int(512*self.smid*self.smid*mul), int(4096*mul), bias=bias),
            # nn.ReLU(True),
            nn.LeakyReLU(math.sqrt(5), inplace=True),
            nn.Linear(int(4096*mul), int(4096*mul), bias=bias),
            # nn.ReLU(True),
            nn.LeakyReLU(math.sqrt(5), inplace=True),
            nn.Linear(int(4096*mul), output_size),
        ])
        self.initialize()

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def initialize(self):
        for m in self.layers:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                # gain = torch.nn.init.calculate_gain('relu')
                gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                bound = gain / math.sqrt(fan_in)
                nn.init.normal_(m.weight, 0, bound)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def normalize(self):
        for m in self.layers:
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                # gain = torch.nn.init.calculate_gain('relu')
                gain = torch.nn.init.calculate_gain('leaky_relu', math.sqrt(5))
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                bound = gain / math.sqrt(fan_in)
                mean = m.weight.mean().detach()
                std = m.weight.std(unbiased=False).detach()
                m.weight.data = bound * (m.weight.data - mean) / std
        
    def compute_norm(self):
        return


def make_layers(cfg, n_channels, mul=1, batch_norm=False, bias=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v*mul)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=bias)
            if batch_norm:
                # layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                layers += [conv2d, nn.BatchNorm2d(v), nn.LeakyReLU(math.sqrt(5), inplace=True)]
            else:
                # layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [conv2d, nn.LeakyReLU(math.sqrt(5), inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
    'E': [64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 'M', 256, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 512, 'M'],
}


def VGG11(input_size,output_size,mul=1):
    return VGG(input_size, output_size, cfg['A'], batch_norm=False)

def VGG11_BN(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['A'], batch_norm=True)

def VGG13(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['B'], batch_norm=False)

def VGG13_BN(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['B'], batch_norm=True)

def VGG16(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['C'], batch_norm=False)

def VGG16_BN(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['C'], batch_norm=True)

def VGG19(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['D'], batch_norm=False)

def VGG19_BN(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['D'], batch_norm=True)

def CustomVGG(input_size,output_size,mul=1):
    return VGG(input_size,output_size, cfg['E'], batch_norm=False)