import argparse

import torch
import torch.nn as nn
import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from lib.datasets import AudioMINST, load_datapath
from lib.wavUtils import Components, pad_trunc, GuassianNoise, time_shift
from lib.toolkit import BatchTransform

common_corruptions = ['gaussian_noise']

class TimeShiftOps:
    LEFT = 'left'
    RIGHT = 'right'
    ORIGIN = 'origin'
    def __init__(self) -> None:
        pass

def train_transforms(args: argparse.Namespace) -> dict[str, BatchTransform]:
    if args.dataset == 'audio-mnist':
        ret = dict()
        transf = Components(transforms=[
                pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                time_shift(shift_limit=-.25, is_random=False),
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80),
                transforms.FrequencyMasking(freq_mask_param=.1),
                transforms.TimeMasking(time_mask_param=.1)
            ])
        ret[TimeShiftOps.LEFT] = BatchTransform(transforms=transf)
        transf = Components(transforms=[
                pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                time_shift(shift_limit=-.25, is_random=False),
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80),
                transforms.FrequencyMasking(freq_mask_param=.1),
                transforms.TimeMasking(time_mask_param=.1)
            ])
        ret[TimeShiftOps.RIGHT] = BatchTransform(transforms=transf)
        transf = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
            transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
            transforms.AmplitudeToDB(top_db=80),
            transforms.FrequencyMasking(freq_mask_param=.1),
            transforms.TimeMasking(time_mask_param=.1)
        ])
        ret[TimeShiftOps.ORIGIN] = BatchTransform(transforms=transf)
        return ret
    raise Exception('No support')

def test_transforms(args: argparse.Namespace) -> dict:
    if args.dataset == 'audio-mnist':
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            print('Test on the original test set')
            ret = dict()
            ret[TimeShiftOps.LEFT] = Components(transforms=[
                    pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                    time_shift(shift_limit=-.25, is_random=False),
                    transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mel, hop_length=args.hop_length),
                    transforms.AmplitudeToDB(top_db=80)
                ])
            ret[TimeShiftOps.RIGHT] = Components(transforms=[
                    pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                    time_shift(shift_limit=.25, is_random=False),
                    transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mel, hop_length=args.hop_length),
                    transforms.AmplitudeToDB(top_db=80)
                ])
            ret[TimeShiftOps.ORIGIN] = Components(transforms=[
                pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mel, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80)
            ])
            return ret
        elif args.corruption in common_corruptions:
            print('Test on %s serverity_level %d' %(args.corruption, args.serverity_level))
            ret = dict()
            ret[TimeShiftOps.LEFT] = Components(transforms=[
                    pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                    time_shift(shift_limit=-.25, is_random=False),
                    GuassianNoise(noise_level=.0025), # .025
                    transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mel, hop_length=args.hop_length),
                    transforms.AmplitudeToDB(top_db=80)
                ])
            ret[TimeShiftOps.RIGHT] = Components(transforms=[
                    pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                    time_shift(shift_limit=.25, is_random=False),
                    GuassianNoise(noise_level=.0025), # .025
                    transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mel, hop_length=args.hop_length),
                    transforms.AmplitudeToDB(top_db=80)
                ])
            ret[TimeShiftOps.ORIGIN] = Components(transforms=[
                pad_trunc(max_ms=1000, sample_rate=args.sample_rate),
                GuassianNoise(noise_level=.0025), # .025
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mel, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80)
            ])
            return ret
    raise Exception('No support')

def prepare_train_data(args: argparse.Namespace) -> tuple[Dataset, DataLoader]:
    if args.dataset == 'audio-mnist':
        dataset = AudioMINST(data_paths=load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent']== 'German'), 
                             include_rate=False, data_trainsforms=pad_trunc(max_ms=1000, sample_rate=args.sample_rate))
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        return dataset, data_loader
    raise Exception('No support')

def prepare_test_data(args: argparse.Namespace) -> tuple[Dataset, DataLoader]:
    if args.dataset == 'audio-mnist':
        dataset = AudioMINST(data_paths=load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent']!= 'German'), 
                             include_rate=False, data_trainsforms=pad_trunc(max_ms=1000, sample_rate=args.sample_rate))
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        return dataset, data_loader
    raise Exception('No support')