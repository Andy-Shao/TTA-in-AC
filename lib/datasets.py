import os
import json

from torch.utils.data import Dataset
import torchaudio

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