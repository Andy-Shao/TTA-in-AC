import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Any

from torch.utils.data import Dataset
import torchaudio
import torch

class AudioMINST(Dataset):
    def __init__(self, data_paths: list[str], data_trainsforms=None, include_rate=True):
        super(AudioMINST, self).__init__()
        self.data_paths = data_paths
        self.data_trainsforms = data_trainsforms
        self.include_rate = include_rate

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index) -> tuple[tuple, float]:
        (wavform, sample_rate) = torchaudio.load(self.data_paths[index])
        label = self.data_paths[index].split('/')[-1].split('_')[0]
        if self.data_trainsforms is not None:
            wavform = self.data_trainsforms(wavform)
        if self.include_rate:
            return (wavform, sample_rate), int(label)
        else:
            return wavform, int(label)
    
def load_datapath(root_path: str, filter_fn) -> list[str]:
    dataset_list = []
    meta_file_path = root_path + '/audioMNIST_meta.txt'
    with open(meta_file_path) as f:
        meta_data = json.load(f)
    for k,v in meta_data.items():
        if filter_fn(v):
            data_path = f'{root_path}/{k}'
            for it in os.listdir(data_path):
                if it.endswith('wav'):
                    dataset_list.append(f'{data_path}/{it}')
    return dataset_list

class FormateDataset(Dataset):
    def __init__(self, dataset: Dataset, data_transf=None, label_transf=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.ready_for_read = False
        self.data_index = pd.DataFrame(columns=['data_path', 'label'])
        self.data_transf = data_transf
        self.label_transf = label_transf
    
    def __len__(self) -> int:
        if not self.ready_for_read:
            return 0
        return len(self.data_index)
        
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        if not self.ready_for_read:
            return None
        data_path = self.data_index['data_path'][index]
        feature, label = torch.load(data_path), self.data_index['label'][index]
        if self.data_transf is not None:
            feature = self.data_transf(feature)
        if self.label_transf is not None:
            label = self.label_transf(label)
        return feature, int(label)
    
    def store_to(self, root_path: str, index_file_name: str, data_transf=None, label_transf=None) -> None:
        try: 
            if os.path.exists(root_path): os.removedirs(root_path)
            os.makedirs(root_path)
        except:
            pass
        for index, (feature, label) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            if data_transf is not None:
                feature = data_transf(feature)
            if label_transf is not None:
                label = data_transf(feature)
            data_path = f'{index}_{label}.dt'
            self.data_index.loc[len(self.data_index)] = [data_path, label]
            torch.save(feature, os.path.join(root_path, data_path))
        self.data_index.to_csv(os.path.join(root_path, index_file_name))
        self.load_from(root_path=root_path, index_file_name=index_file_name)

    def load_from(self, root_path: str, index_file_name: str) -> None:
        index_file_path = os.path.join(root_path, index_file_name)
        self.data_index = pd.read_csv(index_file_path, index_col=0)
        data_pathes = []
        for pth in self.data_index['data_path']:
            data_pathes.append(os.path.join(root_path, pth))
        self.data_index['data_path'] = data_pathes
        self.ready_for_read = True  

class ClipDataset(Dataset):
    def __init__(self, dataset: Dataset, rate: float) -> None:
        super().__init__()      
        assert rate > 0. and rate <= 1., 'rate is the out range'
        self.dataset = dataset
        self.data_size = int(len(dataset) * rate)
        self.indexes = np.random.randint(len(dataset), size=self.data_size)

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index) -> Any:
        return self.dataset[self.indexes[index]]