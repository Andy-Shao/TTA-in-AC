
from torch.utils.data import Dataset

class AudioMINST(Dataset):
    def __init__(self, data_paths: str):
        pass

    def __len__(self):
        return 0
    
    def __getitem__(self, index) -> tuple:
        audio = 0
        label = 0
        return audio, label