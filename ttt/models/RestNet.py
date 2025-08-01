import math

import torch
import torch.nn as nn
from torchvision.models.resnet import conv3x3

class ResNetMNIST(nn.Module):
    def __init__(self, depth: int, width=1, class_num=10, channels=1, norm_layer=nn.BatchNorm2d, fc_in=384):
        assert (depth - 2) % 6 == 0         # depth is 6N+2
        self.N = (depth - 2) // 6
        super(ResNetMNIST, self).__init__()

        # Following the Wide ResNet convention, we fix the very first convolution
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inplanes = 16
        self.layer1 = self._make_layer(norm_layer=norm_layer, planes=16 * width, N=self.N)
        self.layer2 = self._make_layer(norm_layer=norm_layer, planes=32 * width, stride=2, N=self.N)
        self.layer3 = self._make_layer(norm_layer=norm_layer, planes=64 * width, stride=2, N=self.N)
        self.bn = norm_layer(64 * width)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(in_features=fc_in, out_features=class_num)
        self.soft_max = nn.Softmax(dim=1)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0, std=math.sqrt(2. / n))
    
    def _make_layer(self, planes: int, N: int, norm_layer=nn.BatchNorm2d, stride=1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = Downsample(nIn=self.inplanes, nOut=planes, stride=stride)
        layers = [BasicBlock(inplanes=self.inplanes, planes=planes, norm_layer=norm_layer, stride=stride, downsample=downsample)]
        self.inplanes = planes
        for i in range(N - 1):
            layers.append(BasicBlock(inplanes=self.inplanes, planes=planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, nIn: int, nOut: int, stride: int):
        super(Downsample, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=stride)
        assert nOut % nIn == 0
        self.expand_ratio = nOut // nIn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg(x)
        return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), dim=1)

class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, norm_layer: nn.Module, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.conv2(residual)

        if self.downsample is not None:
            x = self.downsample(x)
        return x + residual