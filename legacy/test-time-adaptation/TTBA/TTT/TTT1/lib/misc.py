import os

import torch

def my_makedir(name: str):
    try:
        os.makedirs(name=name)
    except OSError:
        pass

def mean(ls: list) -> float:
	return sum(ls) / len(ls)

def flat_grad(grad_tuple: torch.Tensor) -> torch.Tensor:
    return torch.cat([p.view(-1) for p in grad_tuple])