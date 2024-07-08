import random

import torch 
import torch.nn as nn

class GuassianNoise(nn.Module):
    def __init__(self, noise_level=.05):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        ## Guassian Noise
        noise = torch.rand_like(wavform) * self.noise_level
        noise_wavform = wavform + noise
        return noise_wavform

class pad_trunc(nn.Module):
    """
    Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    """
    def __init__(self, max_ms: float, sample_rate: int) -> None:
        super().__init__()
        assert max_ms > 0, 'max_ms must be greater than zero'
        assert sample_rate > 0, 'sample_rate must be greater than zeror'
        self.max_ms = max_ms
        self.sample_rate = sample_rate

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        channel_num, wav_len = wavform.shape
        max_len = self.sample_rate // 1000 * self.max_ms

        if (wav_len > max_len):
            wavform = wavform[:, :max_len]
        elif wav_len < max_len:
            head_len = random.randint(0, max_len - wav_len)
            tail_len = max_len - wav_len - head_len

            head_pad = torch.zeros((channel_num, head_len))
            tail_pad = torch.zeros((channel_num, tail_len))

            wavform = torch.cat((head_pad, wavform, tail_pad), dim=1)
        return wavform

class Components(nn.Module):
    def __init__(self, transforms: list) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.transforms is None:
            return None
        for transform in self.transforms:
            wavform = transform(wavform)
        return wavform

class time_shift(nn.Module):
    def __init__(self, shift_limit: float, is_random=True, is_bidirection=False) -> None:
        """
        Time shift data augmentation

        :param shift_limit: shift_limit -> (-1, 1), shift_limit < 0 is left shift
        """
        super().__init__()
        self.shift_limit = shift_limit
        self.is_random = is_random
        self.is_bidirection = is_bidirection

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.is_random:
            shift_arg = int(random.random() * self.shift_limit * wavform.shape[1])
            if self.is_bidirection:
                shift_arg = int((random.random() * 2 - 1) * self.shift_limit * wavform.shape[1])
        else:
            shift_arg = int(self.shift_limit * wavform.shape[1])
        return wavform.roll(shifts=shift_arg)
    
def display_wavform(waveform: torch.Tensor):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(waveform.numpy().T)
    plt.title('Audio Waveform')
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.show()

def display_spectro_gram(waveform: torch.Tensor):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,4))
    plt.imshow(waveform[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.title('Mel Spectrogram in channel 0')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()