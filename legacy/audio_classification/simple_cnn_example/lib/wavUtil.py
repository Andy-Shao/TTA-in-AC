import math,random
import torch 
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class WavOps():
    @staticmethod
    def open(wav_path):
        signal, sample_rate = torchaudio.load(wav_path) # a signal tensor and sample rate
        return (signal, sample_rate)