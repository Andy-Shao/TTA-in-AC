import numpy as np
from typing import Any 
from PIL import Image
 
from torch.utils.data import Dataset

class CIFAR_New(Dataset):
    def __init__(self, root: str, transform=None, target_transform=None, version='v6'):
        self.data = np.load(f'{root}/cifar10.1_{version}_data.npy')
        self.targets = np.load(f'{root}cifar10.1_%s_labels.npy').astype('long')
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index) -> Any:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target