from typing import Any, List, Tuple
import os
import os.path
import numpy as np
from PIL import Image

import torch 
import torch.nn as nn
from torch.utils.data import Dataset

def make_dataset(image_list, labels) -> List:
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list: List, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError('Without image'))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index) -> Any:
        path, target = self.imgs[index]
        img = self.loader(path=path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class ImageList_idx(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError('Without image'))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index) -> Any:
        path, target = self.imgs[index]
        img = self.loader(path=path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

def make_dataset_MixUp(image_list: List[str], labels) -> List[Tuple[str, str, str, str]]:
    ## Image_path, Actual_lbl, pseudo_label, domain
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        images = [(val.split(',')[1], int(val.split(',')[3]), int(val.split(',')[2]), int(val.split(',')[0])) for val in image_list]
    return images
    
class ImageList_MixUp(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset_MixUp(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError('Without data'))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __len__(self) -> int:
        return len(self.imgs)
    
    def __getitem__(self, index) -> Tuple:
        path, pseudo_label, target, domain = self.imgs[index]
        img = self.loader(path=path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, pseudo_label, target, domain