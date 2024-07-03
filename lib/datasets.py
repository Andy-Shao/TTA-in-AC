import os
import json
from tqdm import tqdm

from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn as nn

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

class StoredDataset(Dataset):
    def __init__(self, dataset: Dataset, stored_path: str, data_transfs=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_transfs = data_transfs
        self.stored_path = stored_path
        self.is_stored = False

    def store(self):
        print('stored the dataset into hard disk')
        if os.path.exists(self.stored_path):
            os.removedirs(self.stored_path)
        os.makedirs(self.stored_path)
        for idx, (feature, label) in tqdm(enumerate(self.dataset)):
            if self.data_transfs is not None:
                feature = self.data_transfs(feature)
            item_path = os.path.join(self.stored_path, f'{idx}_{str(label)}.dt')
            torch.save(feature, item_path)
        self.stored_list = self.__read_stored_file__()
        self.is_stored = True

    def __read_stored_file__(self) -> list[str]:
        dataset_list = []
        for pth in os.listdir(self.stored_path):
            if pth.endswith('.dt'):
                dataset_list.append(os.path.join(self.stored_path, pth))
        return dataset_list
    
    def __len__(self):
        assert self.is_stored, 'You should store the file first'
        return len(self.stored_list)
    
    def __getitem__(self, index) -> torch.Tensor:
        assert self.is_stored, 'You should store the file first'
        file_path = self.stored_list[index]
        label = file_path.split('_')[-1]
        return torch.load(file_path), int(label)
    
class FormateDataset(Dataset):
    def __init__(self, dataset: Dataset, data_transf: nn.Module) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_transf = data_transf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        feature = self.dataset[index]
        return self.data_transf(feature)