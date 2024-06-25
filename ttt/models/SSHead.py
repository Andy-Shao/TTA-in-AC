import copy

import torch 
import torch.nn as nn

class ExtractorHead(nn.Module):
    def __init__(self, ext: nn.Module, head: nn.Module):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ext(x))

def extractor_from_layer3(net: nn.Module) -> nn.Module:
    layers = [net.conv1, net.layer1, net.layer2, net.layer3, net.bn, net.relu, net.avgpool, ViewFlatten()]
    return nn.Sequential(*layers)

def extractor_from_layer2(net: nn.Module) -> nn.Module:
    layers = [net.conv1, net.layer1, net.layer2]
    return nn.Sequential(*layers)

def head_on_layer2(net: nn.Module, width: int, class_num: int, fc_in: int) -> nn.Module:
    head = copy.deepcopy([net.layer3, net.bn, net.relu, net.avgpool])
    head.append(ViewFlatten())
    head.append(nn.Linear(in_features=fc_in, out_features=class_num))
    return nn.Sequential(*head)

class ViewFlatten(nn.Module):
    def __init__(self):
        super(ViewFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)