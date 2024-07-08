import torch.nn as nn
import torch

class ExpandChannel(nn.Module):
    def __init__(self, out_channel: int) -> None:
        super().__init__()
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat((self.out_channel, 1, 1))