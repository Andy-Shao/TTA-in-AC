import os
from typing import Any

from torch.utils.data import Dataset
import torch
import torchaudio

class SpeechCommandsDataset(Dataset):
    test_meta_file = 'testing_list.txt'
    val_meta_file = 'validation_list.txt'
    label_dic = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9., 'bed': 10., 'dog': 11., 'happy': 12., 'marvin': 13., 'off': 14., 
        'right': 15., 'up': 16., 'yes': 17., 'bird': 18., 'down': 19., 'house': 20., 'on': 21., 
        'stop': 22., 'tree': 23., 'cat': 24., 'go': 25., 'left': 26., 'no': 27., 'sheila': 28., 
        'wow': 29.
    }

    def __init__(self, root_path: str, mode: str, include_rate=True, data_tfs=None) -> None:
        super().__init__()
        self.root_path = root_path
        assert mode in ['train', 'validation', 'test', 'full'], 'mode type is incorrect'
        self.mode = mode
        self.include_rate = include_rate
        self.data_list = self.__cal_data_list__(mode=mode)
        self.data_tfs = data_tfs

    def __cal_data_list__(self, mode: str) -> list[str]:
        if mode == 'validation':
            with open(os.path.join(self.root_path, self.val_meta_file), 'rt', newline='\n') as f:
                val_meta_data = f.readlines()
            return [line.rstrip('\n') for line in val_meta_data]
        elif mode == 'test':
            with open(os.path.join(self.root_path, self.test_meta_file), 'rt', newline='\n') as f:
                test_meta_data = f.readlines()
            return [line.rstrip('\n') for line in test_meta_data]
        elif mode == 'full':
            full_meta_data = []
            for k,v in self.label_dic.items():
                base_path = os.path.join(self.root_path, k)
                for p in os.listdir(base_path):
                    if p.endswith('.wav'):
                        full_meta_data.append(f'{k}/{p}')
            return full_meta_data
        else:
            val_meta_data = self.__cal_data_list__(mode='validation')
            test_meta_data = self.__cal_data_list__(mode='test')
            full_meta_data = self.__cal_data_list__(mode='full')
            train_meta_data = []
            for it in full_meta_data:
                if it in val_meta_data:
                    continue
                if it in test_meta_data:
                    continue
                train_meta_data.append(it)
            return train_meta_data

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> torch.Tensor:
        audio_path, label = self.__cal_audio_path_label__(self.data_list[index])
        audio, sample_rate = torchaudio.load(audio_path)
        if self.data_tfs is not None:
            audio = self.data_tfs(audio)
        if self.include_rate:
            return audio, label, sample_rate
        return audio, int(label)

    def __cal_audio_path_label__(self, meta_data: str) -> tuple[str, float]:
        label = meta_data.strip().split('/')[0]
        label = self.label_dic[label]
        audio_path = os.path.join(self.root_path, meta_data)
        return audio_path, label
    
class BackgroundNoise(Dataset):
    base_path = '_background_noise_'

    def __init__(self, root_path: str) -> None:
        super().__init__()
        self.root_path = root_path 
        self.data_list = self.__cal_data_list__()

    def __cal_data_list__(self) -> list[str]:
        data_list = []
        for it in os.listdir(os.path.join(self.root_path, self.base_path)):
            if it.endswith('.wav'):
                data_list.append(it)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> Any:
        noise_path = os.path.join(self.root_path, self.base_path, self.data_list[index])
        noise, sample_rate = torchaudio.load(noise_path)
        noise_type = self.data_list[index][:-(len('.wav'))]
        return noise_type, noise, sample_rate