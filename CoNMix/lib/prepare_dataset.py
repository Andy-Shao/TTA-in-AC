import argparse

import torch.nn as nn
import torch
from torch.utils.data import Dataset
from lib.scDataset import SpeechCommandsDataset, RandomSpeechCommandsDataset

def build_dataset(args: argparse.Namespace, mode:str, data_tfs:nn.Module) -> Dataset:
    if args.dataset == 'speech-commands-random':
        return RandomSpeechCommandsDataset(root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=data_tfs, data_type=args.dataset_type)
    else:
        return SpeechCommandsDataset(root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=data_tfs, data_type=args.dataset_type)

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