import sys
import torch
import torch.nn.functional as F
from utils import *

from arguments import get_args
args = get_args()

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, input_size, output_size, cfg, mul=1, batch_norm=False):
        super(VGG, self).__init__()
        mul = args.mul
        n_channels, size, _ = input_size
        self.layers = make_layers(cfg, n_channels, mul=mul, batch_norm=batch_norm)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            nn.Linear(int(512*self.smid*self.smid*mul), int(4096*mul)),
            nn.ReLU(True),
            nn.Linear(int(4096*mul), int(4096*mul)),
            nn.ReLU(True),
            nn.Linear(int(4096*mul), output_size),
        ])

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


def make_layers(cfg, n_channels, mul=1, batch_norm=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v*mul)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def VGG11(input_size,output_size,mul=1):
    """VGG 11-layer model (configuration "A")"""
    return VGG(input_size, output_size, cfg['A'], batch_norm=False)


def VGG11_BN(input_size,output_size,mul=1):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(input_size,output_size, cfg['A'], batch_norm=True)


def VGG13(input_size,output_size,mul=1):
    """VGG 13-layer model (configuration "B")"""
    return VGG(input_size,output_size, cfg['B'], batch_norm=False)


def VGG13_BN(input_size,output_size,mul=1):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(input_size,output_size, cfg['B'], batch_norm=True)


def VGG16(input_size,output_size,mul=1):
    """VGG 16-layer model (configuration "D")"""
    return VGG(input_size,output_size, cfg['C'], batch_norm=False)


def VGG16_BN(input_size,output_size,mul=1):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(input_size,output_size, cfg['C'], batch_norm=True)


def VGG19(input_size,output_size,mul=1):
    """VGG 19-layer model (configuration "E")"""
    return VGG(input_size,output_size, cfg['D'], batch_norm=False)


def VGG19_BN(input_size,output_size,mul=1):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(input_size,output_size, cfg['D'], batch_norm=True)

