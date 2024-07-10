import torch.nn as nn
import torch
from torch.utils.data import Dataset

class ExpandChannel(nn.Module):
    def __init__(self, out_channel: int) -> None:
        super().__init__()
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat((self.out_channel, 1, 1))
    
class Dataset_Idx(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int, int]:
        feature, label = self.dataset[index]
        return feature, label, index