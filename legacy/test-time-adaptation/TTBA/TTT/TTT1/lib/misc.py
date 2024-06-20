import os
from colorama import Fore

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

def normalize(v: torch.Tensor) -> torch.Tensor:
	return (v - v.mean()) / v.std()

def print_color(color: str, string: str):
	print(getattr(Fore, color) + string + Fore.RESET)