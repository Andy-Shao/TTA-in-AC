import os
from typing import Any

from torch.utils.data import Dataset

class ESC50(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __len__(self):
        return -1
    
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)