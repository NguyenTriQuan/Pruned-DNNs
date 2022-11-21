import sys
import torch
import torch.nn.functional as F
from utils import *

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, inputsize, taskcla, cfg, batch_norm=False):
        super(VGG, self).__init__()

        n_channels, size, _ = inputsize
        self.taskcla = taskcla
        self.layers = make_layers(cfg, n_channels, batch_norm=batch_norm)

        self.smid = size
        for m in self.layers:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            # nn.Dropout(),
            nn.Linear(512//2*self.smid*self.smid, 4096//2),
            nn.ReLU(True),
            # nn.Dropout(),
            nn.Linear(4096//2, 4096//2),
            nn.ReLU(True),
            # nn.Linear(4096, output_dim),
        ])

        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(4096//2,n))
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, t):
        for m in self.layers:
            x = m(x)
        return self.last[t](x)


def make_layers(cfg, n_channels, batch_norm=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = v // 2
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


def VGG11(inputsize,taskcla,mul=1):
    """VGG 11-layer model (configuration "A")"""
    return VGG(inputsize,taskcla, cfg['A'], batch_norm=False)


def VGG11_BN(inputsize,taskcla,mul=1):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(inputsize,taskcla, cfg['A'], batch_norm=True)


def VGG13(inputsize,taskcla,mul=1):
    """VGG 13-layer model (configuration "B")"""
    return VGG(inputsize,taskcla, cfg['B'], batch_norm=False)


def VGG13_BN(inputsize,taskcla,mul=1):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(inputsize,taskcla, cfg['B'], batch_norm=True)


def VGG16(inputsize,taskcla,mul=1):
    """VGG 16-layer model (configuration "D")"""
    return VGG(inputsize,taskcla, cfg['C'], batch_norm=False)


def VGG16_BN(inputsize,taskcla,mul=1):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(inputsize,taskcla, cfg['C'], batch_norm=True)


def VGG19(inputsize,taskcla,mul=1):
    """VGG 19-layer model (configuration "E")"""
    return VGG(inputsize,taskcla, cfg['D'], batch_norm=False)


def VGG19_BN(inputsize,taskcla,mul=1):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(inputsize,taskcla, cfg['D'], batch_norm=True)

