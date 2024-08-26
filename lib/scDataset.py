import os
from typing import Any

from torch.utils.data import Dataset
import torch
import torchaudio

class RandomSpeechCommandsDataset(Dataset):
    test_meta_file = 'rand_testing_list.txt'
    val_meta_file = 'rand_validation_list.txt'

    def __init__(self, root_path: str, mode: str, include_rate=True, data_tfs=None, data_type='all', seed:int = 2024) -> None:
        super().__init__()
        self.dataset = SpeechCommandsDataset(root_path=root_path, mode='train', include_rate=include_rate, data_tfs=data_tfs, data_type=data_type)
        self.seed = seed
        assert mode in ['train', 'validation', 'test', 'full', 'test+val'], 'mode type is incorrect'
        self.mode = mode
        self.__generate_random_meta_file__(seed=seed)

        if mode == 'train':
            self.data_list = self.train_indexes
        elif mode == 'validation':
            self.data_list = self.val_indexes
        elif mode == 'full':
            self.data_list = [it for it in range(self.dataset)]
        elif mode == 'test+val':
            self.data_list = []
            self.data_list.extend(self.test_indexes)
            self.data_list.extend(self.val_indexes)
        elif mode == 'test':
            self.data_list = self.test_indexes

    def __generate_random_meta_file__(self, seed:int) -> None:
        from numpy.random import MT19937, RandomState, SeedSequence
        rs = RandomState(MT19937(SeedSequence(seed)))
        self.test_indexes = rs.choice(len(self.dataset), size=int(.3*len(self.dataset)), replace=False)
        residua = []
        for i in range(len(self.dataset)):
            if i in self.test_indexes:
                continue
            residua.append(i)
        self.train_indexes = rs.choice(residua, size=int(.9*len(residua)), replace=False)
        self.val_indexes = []
        for i in residua:
            if i in self.train_indexes:
                continue
            self.val_indexes.append(i)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index) -> Any:
        return self.dataset[self.data_list[index]]

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
    commands = {'yes':0., 'no':1., 'up':2., 'down':3., 'left':4., 'right':5., 'on':6., 'off':7., 'stop':8., 'go':9.}
    no_commands = {
        'zero':0., 'one':1., 'two':2., 'three':3., 'four':4., 'five':5., 'six':6., 'seven':7., 'eight':8., 'nine':9., 
        'bed':10., 'dog':11., 'happy':12., 'marvin':13., 'bird':14., 'house':15., 'tree':16., 'cat':17., 'sheila':18., 
        'wow':19.
    }
    numbers = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9.
    }

    def __init__(self, root_path: str, mode: str, include_rate=True, data_tfs=None, data_type='all') -> None:
        super().__init__()
        self.root_path = root_path
        assert mode in ['train', 'validation', 'test', 'full', 'test+val'], 'mode type is incorrect'
        self.mode = mode
        assert data_type in ['all', 'commands', 'no_commands', 'numbers']
        self.data_type = data_type
        self.include_rate = include_rate
        data_list = self.__cal_data_list__(mode=mode)
        self.data_list = self.__filter_data_list__(data_list=data_list)
        self.data_tfs = data_tfs
    
    def __filter_data_list__(self, data_list:list[str]) -> list[str]:
        if self.data_type == 'all':
            return data_list
        elif self.data_type == 'commands':
            filter_list = self.commands.keys()
        elif self.data_type == 'no_commands':
            filter_list = self.no_commands.keys()
        elif self.data_type == 'numbers':
            filter_list = self.numbers.keys()
        else:
            raise Exception('No support')
        new_data_list = []
        for it in data_list:
            label = it.strip().split('/')[0]
            if label in filter_list:
                new_data_list.append(it)
        return new_data_list


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
        elif mode == 'test+val':
            val_meta_data = self.__cal_data_list__(mode='validation')
            test_meta_data = self.__cal_data_list__(mode='test')
            for it in val_meta_data:
                test_meta_data.append(it)
            return test_meta_data
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
        if self.data_type == 'all':
            label = self.label_dic[label]
        elif self.data_type == 'commands':
            label = self.commands[label]
        elif self.data_type == 'no_commands':
            label = self.no_commands[label]
        elif self.data_type == 'numbers':
            label = self.numbers[label]
        else:
            raise Exception('No support')
        audio_path = os.path.join(self.root_path, meta_data)
        return audio_path, label
    
class BackgroundNoiseDataset(Dataset):
    base_path = '_background_noise_'

    def __init__(self, root_path: str, data_tf=None, label_tf=None) -> None:
        super().__init__()
        self.root_path = root_path 
        self.data_list = self.__cal_data_list__()
        self.data_tf = data_tf
        self.label_tf = label_tf

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
        if self.data_tf is not None:
            noise = self.data_tf(noise)
        if self.label_tf is not None:
            sample_rate = self.label_tf(sample_rate)
        return noise_type, noise, sample_rate