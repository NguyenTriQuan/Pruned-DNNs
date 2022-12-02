import sys
import torch
import torch.nn.functional as F
from utils import *
from layers.wn_layer import _WeightNormLayer, WeightNormLinear, WeightNormConv2D
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
    def __init__(self, input_size, output_size, cfg, mul=1, batch_norm=False, bias=False):
        super(VGG, self).__init__()
        mul = args.mul
        n_channels, size, _ = input_size
        self.layers = make_layers(cfg, n_channels, mul=mul, batch_norm=batch_norm, bias=bias)
        self.smid = size
        for m in self.layers:
            if isinstance(m, WeightNormConv2D) or isinstance(m, nn.MaxPool2d):
                try:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size[0], m.stride[0], m.padding[0], m.dilation[0])
                except:
                    self.smid = compute_conv_output_size(self.smid, m.kernel_size, m.stride, m.padding, m.dilation)

        self.layers += nn.ModuleList([
            nn.Flatten(),
            WeightNormLinear(int(512*self.smid*self.smid*mul), int(4096*mul), bias=bias, activation=args.activation),
            WeightNormLinear(int(4096*mul), int(4096*mul), bias=bias, activation=args.activation),
            # nn.Linear(int(4096*mul), output_size),
            WeightNormLinear(int(4096*mul), output_size, bias=True, activation='identity'),
        ])
        # gain = torch.nn.init.calculate_gain('leaky_relu', args.negative_slope)
        # fan_in, fan_out = _calculate_fan_in_and_fan_out(self.layers[-1].weight)
        # bound = gain / math.sqrt(fan_in)
        # nn.init.normal_(self.layers[-1].weight, 0, bound)
        # nn.init.constant_(self.layers[-1].bias, 0)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x

    def normalize(self):
        for m in self.layers[:-1]:
            if isinstance(m, _WeightNormLayer):
                m.normalize()

def make_layers(cfg, n_channels, mul=1, batch_norm=False, bias=False):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v*mul)
            for i in range(1):
                conv2d = WeightNormConv2D(in_channels, v, kernel_size=3, padding=1, bias=bias, activation=args.activation, norm_type=batch_norm)
                layers += [conv2d]
                in_channels = v
    return nn.ModuleList(layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
            512, 512, 512, 512, 'M'],
    'E': [64, 64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 128, 'M', 256, 256, 256, 256, 256, 256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 512, 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512, 512, 512, 512, 512, 'M'],
    'F': [64, 64, 64, 64, 64 ,'M', 64, 64, 64, 64, 64, 'M', 128, 128, 128, 128, 128, 'M', 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 'M', 
            512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 'M'],
    'G': [64, 'M', 64, 64, 64, 64 ,'M', 64, 64, 64, 64,'M', 64, 64, 64, 64, 64 ,'M', 64, 64, 64, 64, 64 ,'M'],
}


def VGG11(input_size,output_size,mul=1,norm_type=False):
    return VGG(input_size, output_size, cfg['A'], batch_norm=norm_type)

def VGG13(input_size,output_size,mul=1,norm_type=False):
    return VGG(input_size,output_size, cfg['B'], batch_norm=norm_type)

def VGG16(input_size,output_size,mul=1,norm_type=False):
    return VGG(input_size,output_size, cfg['C'], batch_norm=norm_type)

def VGG19(input_size,output_size,mul=1,norm_type=False):
    return VGG(input_size,output_size, cfg['D'], batch_norm=norm_type)

def CustomVGG(input_size,output_size,mul=1,norm_type=False):
    return VGG(input_size,output_size, cfg['G'], batch_norm=norm_type)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, norm_type=None):
    """3x3 convolution with padding"""
    return WeightNormConv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
        activation=args.activation,
        norm_type=norm_type
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, norm_type=None):
    """1x1 convolution"""
    return WeightNormConv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, activation=args.activation, norm_type=norm_type)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_type = None,
    ) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, norm_type=norm_type)
        self.conv2 = conv3x3(planes, planes, norm_type=norm_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if args.res:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_type = None,
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, norm_type=norm_type)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, norm_type=norm_type)
        self.conv3 = conv1x1(width, planes * self.expansion, norm_type=norm_type)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if args.res:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        norm_type,
        input_size,
        output_size,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation = None,
    ) -> None:
        super().__init__()
        
        self._norm_layer = norm_type

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = WeightNormConv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, norm_type=norm_type, activation='identity')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = WeightNormLinear(512 * block.expansion, output_size, bias=True, activation=args.activation)
        self.WN = [m for m in self.modules() if isinstance(m, _WeightNormLayer)]
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_type = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, norm_type=norm_type),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_type
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_type=norm_type,
                )
            )

        return nn.Sequential(*layers)

    def normalize(self):
        for m in self.WN[:-1]:
            m.normalize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ResNet18(input_size, output_size, norm_type=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_type, input_size, output_size)

def ResNet34(input_size, output_size, norm_type=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_type, input_size, output_size)

def ResNet50(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], norm_type, input_size, output_size)

def ResNet101(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 4, 23, 3], norm_type, input_size, output_size)

def ResNet152(input_size, output_size, norm_type=None):
    return ResNet(Bottleneck, [3, 8, 36, 3], norm_type, input_size, output_size)