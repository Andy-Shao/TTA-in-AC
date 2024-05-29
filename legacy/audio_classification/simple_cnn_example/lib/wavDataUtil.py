from typing import Any
import random
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from .wavUtil import WavOps
import torch

class WavDataset(Dataset):
    def __init__(self, df, data_path) -> None:
        super().__init__()
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sample_rate = 44100
        self.channel = 2
        self.shift_precentage = .4

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Any:
        absolute_audio_path = self.data_path + self.df.loc[index, 'path']
        class_id = self.df.loc[index, 'classID']
        class_id = torch.eye(10)[class_id]

        audio = WavOps.open(absolute_audio_path)
        audio = WavOps.resampleRate(audio=audio, new_sample_rate=self.sample_rate)
        audio = WavOps.rechannel(audio=audio, channel_num=self.channel)
        audio = WavOps.pad_trunc(audio=audio, max_ms=self.duration)
        audio = WavOps.time_shift(audio=audio, shift_limit=self.shift_precentage)
        audio = WavOps.spectro_gram(audio=audio, n_mels=64, n_fft=1024, hop_len=None)
        audio = WavOps.spectro_augment(spec=audio, max_mask_perctage=.03, freq_mask_num=2, time_mask_num=2)
        
        return (audio, class_id)

def formatAudio(audio, sample_rate, channel, duration, shift_precentage) :
    audio = WavOps.resampleRate(audio=audio, new_sample_rate=sample_rate)
    audio = WavOps.rechannel(audio=audio, channel_num=channel)
    audio = WavOps.pad_trunc(audio=audio, max_ms=duration)
    if random.random() < .5: audio = WavOps.time_shift(audio=audio, shift_limit=shift_precentage)
    audio = WavOps.spectro_gram(audio=audio, n_mels=64, n_fft=1024, hop_len=None)
    if random.random() >= .5 :audio = WavOps.spectro_augment(spec=audio, max_mask_perctage=.03, freq_mask_num=2, time_mask_num=2)
    return audio