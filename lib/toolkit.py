import argparse

import torch 
import torch.nn as nn

def print_argparse(args: argparse.Namespace) -> None:
    for arg in vars(args):
        print(f'--{arg} = {getattr(args, arg)}')

class BatchTransform(object):
    def __init__(self, transforms: nn.Module) -> None:
        super().__init__()
        self.transforms = transforms
    
    def transf(self, ls: torch.Tensor) -> torch.Tensor:
        ret = []
        for i in range(ls.size(0)):
            tmp = torch.unsqueeze(self.transforms(ls[i]), dim=0)
            ret.append(tmp)
        return torch.cat(ret, dim=0)
    
    def tran_one(self, x: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(self.transforms(x), dim=0)
    
    def inner_transforms(self) -> nn.Module:
        return self.transforms