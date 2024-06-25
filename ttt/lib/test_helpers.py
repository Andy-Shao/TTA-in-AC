import argparse

import torch.nn as nn

def build_model(args: argparse.Namespace) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
    from ttt.models.RestNet import ResNetMNIST as ResNet
    from ttt.models.SSHead import ExtractorHead

    print('Building model...')
    