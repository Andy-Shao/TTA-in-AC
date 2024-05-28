from typing import Any
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio

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
        absolute_audio_path = self.data_path + self.df.loc[index, 'relative_path']
        class_id = self.df.loc[index, 'classID']
        
        pass