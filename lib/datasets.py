import os
import json

from torch.utils.data import Dataset
import torchaudio

class AudioMINST(Dataset):
    def __init__(self, data_paths: list[str]):
        super(AudioMINST, self).__init__()
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index) -> tuple[tuple, float]:
        audio = torchaudio.load(self.data_paths[index])
        label = self.data_paths[index].split('/')[-1].split('_')[0]
        return audio, float(label)
    
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