import math,random
import torch 
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

class WavOps():
    @staticmethod
    def open(wav_path):
        """
        Open and read a wav audio file

        :param wav_path: the location of the wav file
        :return: a signal tensor and sample rate
        """
        signal, sample_rate = torchaudio.load(wav_path)
        return (signal, sample_rate)
    
    @staticmethod
    def rechannel(audio, channel_num):
        """ 
        Some of the sound files are mono (ie. 1 audio channel) while most of them are stereo (ie. 2 audio channels). 
        Since our model expects all items to have the same dimensions, we will convert the mono files to stereo, by 
        duplicating the first channel to the second. 
        """
        signal, sample_rate = audio

        if(signal.shape[0] == channel_num):
            return audio # Return directly
        
        if(channel_num == 1):
            return ((signal[:1,:], sample_rate))
        else: 
            return ((torch.cat[signal, signal], sample_rate))
        
    @staticmethod
    def resampleRate(audio, new_sample_rate):
        """
        Some of the sound files are sampled at a sample rate of 48000Hz, while most are sampled at a rate of 44100Hz.
        we must standardize and convert all audio to the same sampling rate so that all arrays have the same dimensions.
        """
        signal, sample_rate = audio

        if(sample_rate == new_sample_rate):
            return audio
        
        new_signal = transforms.Resample(sample_rate, new_sample_rate)(signal[:1, :])
        if(signal.shape[0] > 1):
            signal_two = transforms.Resample(sample_rate, new_sample_rate)(signal[1:, :])
            new_signal = torch.cat([new_signal, signal_two])
        
        return ((new_signal, new_sample_rate))

    @staticmethod
    def pad_trunc(audio, max_ms):
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
        """
        signal, sample_rate = audio
        channel_num, signal_len = signal.shape
        max_len = sample_rate//1000 * max_ms

        if(signal_len > max_len):
            signal = signal[:, :max_len]
        elif(signal_len < max_len):
            head_len = random.randint(0, max_len - signal_len)
            tail_len = max_len - signal_len - head_len

            head_pad = torch.zeros((channel_num, head_len))
            tail_pad = torch.zeros((channel_num, tail_len))

            signal = torch.cat((head_pad, signal, tail_pad),1)

        return (signal, sample_rate)

    @staticmethod
    def time_shift(audio, shift_limit):
        """
        Time shift data augmentation

        :param shift_limit: shift_limit -> (-1, 1), shift_limit < 0 is left shift
        """
        signal, sample_rate = audio
        shift_arg = int(random.random() * shift_limit * signal.shape[1])

        return (signal.roll(shift_arg), sample_rate)

    @staticmethod
    def spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None):
        """
        Cover the audio to a Mel Spectrogram

        :param n_mels: Number of mel filterbanks
        :param n_fft: Size of FFT, creates n_fft // 2 + 1 bins
        :param hop_len: Length of hop between STFT windows
        :return: the shape is [channel, n_mels, time]
        """
        signal, sample_rate = audio
        top_db = 80 # minimum negative cut-off in decibels

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_len)(signal)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        return (spec)
    
    @staticmethod
    def spectro_augment(spec, max_mask_perctage=.1, freq_mask_num=1, time_mask_num=1):
        """
        Do data augmentation under Mel Spectrogram

        Frequency mask : adding horizontal bars on the spectrogram
        Time mask : adding vertical bars on the spectrogram
        """
        _, n_mels, n_times = spec.shape
        mask_value = spec.mean()
        augment_spec = spec

        freq_mask_param = max_mask_perctage * n_mels
        for _ in range(freq_mask_num):
            augment_spec = transforms.FrequencyMasking(freq_mask_param=freq_mask_param)(augment_spec, mask_value)

        time_mask_param = max_mask_perctage * n_times
        for _ in range(time_mask_num):
            augment_spec = transforms.TimeMasking(time_mask_param=time_mask_param)(augment_spec, mask_value)

        return augment_spec