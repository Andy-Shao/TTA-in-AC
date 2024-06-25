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
        ret = torch.ones(2,3)
        for i in range(ls.size(0)):
            if i == 0:
                ret = self.transforms(ls[i])
                ret = torch.unsqueeze(ret, dim=0)
            else:
                tmp = torch.unsqueeze(self.transforms(ls[i]), dim=0)
                ret = torch.cat((ret, tmp), dim=0)
        return ret
    
    def tran_one(self, x: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(self.transforms(x), dim=0)