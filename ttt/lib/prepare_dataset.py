import argparse

import torchaudio.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

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

def fill_default_args(args: argparse.Namespace) -> argparse.Namespace:
    if not hasattr(args, 'shift_limit'):
        args.shift_limit = .25
    elif args.shift_limit <= 0.:
        raise Exception('shift_limit should be greater than zero')
    if not hasattr(args, 'severity_level'):
        args.severity_level = .0025
    return args

def train_transforms(args: argparse.Namespace) -> dict[str, BatchTransform]:
    fill_default_args(args=args)
    if args.dataset == 'audio-mnist':
        ret = dict()
        ret[TimeShiftOps.LEFT] = BatchTransform(transforms=Components(transforms=[
                time_shift(shift_limit=-args.shift_limit, is_random=False),
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80),
                transforms.FrequencyMasking(freq_mask_param=.1),
                transforms.TimeMasking(time_mask_param=.1)
            ]))
        ret[TimeShiftOps.RIGHT] = BatchTransform(transforms=Components(transforms=[
                time_shift(shift_limit=args.shift_limit, is_random=False),
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80),
                transforms.FrequencyMasking(freq_mask_param=.1),
                transforms.TimeMasking(time_mask_param=.1)
            ]))
        ret[TimeShiftOps.ORIGIN] = BatchTransform(transforms=Components(transforms=[
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80),
                transforms.FrequencyMasking(freq_mask_param=.1),
                transforms.TimeMasking(time_mask_param=.1)
            ]))
        return ret
    raise Exception('No support')

def test_transforms(args: argparse.Namespace) -> dict[str, BatchTransform]:
    fill_default_args(args=args)
    if args.dataset == 'audio-mnist':
        ret = dict()
        ret[TimeShiftOps.LEFT] = BatchTransform(transforms=Components(transforms=[
                    time_shift(shift_limit=-args.shift_limit, is_random=False),
                    transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                    transforms.AmplitudeToDB(top_db=80)
                ]))
        ret[TimeShiftOps.RIGHT] = BatchTransform(transforms=Components(transforms=[
                    time_shift(shift_limit=args.shift_limit, is_random=False),
                    transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                    transforms.AmplitudeToDB(top_db=80)
                ]))
        ret[TimeShiftOps.ORIGIN] = BatchTransform(transforms=Components(transforms=[
                transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
                transforms.AmplitudeToDB(top_db=80)
            ]))
        return ret
    raise Exception('No support')

def prepare_train_data(args: argparse.Namespace, data_transforms=None) -> tuple[Dataset, DataLoader]:
    if args.dataset == 'audio-mnist':
        if data_transforms is None:
            data_transforms = pad_trunc(max_ms=1000, sample_rate=args.sample_rate)
        dataset = AudioMINST(data_paths=load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent']== 'German'), 
                             include_rate=False, data_trainsforms=data_transforms)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        return dataset, data_loader
    raise Exception('No support')

def prepare_test_data(args: argparse.Namespace, data_transforms=None) -> tuple[Dataset, DataLoader]:
    if args.dataset == 'audio-mnist':
        if data_transforms is None:
            data_transforms = pad_trunc(max_ms=1000, sample_rate=args.sample_rate)
        dataset = AudioMINST(data_paths=load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent']!= 'German'), 
                             include_rate=False, data_trainsforms=data_transforms)
        data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        return dataset, data_loader
    raise Exception('No support')