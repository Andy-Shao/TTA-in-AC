import argparse

from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
import torch.nn as nn

from lib.wavUtils import pad_trunc, time_shift, Components
from lib.scDataset import SpeechCommandsDataset, RandomSpeechCommandsDataset
from lib.toolkit import BatchTransform, cal_norm
from ttt.lib.prepare_dataset import TimeShiftOps
from lib.datasets import TransferDataset

def add_normalization(args:argparse.Namespace, tsf_dict:dict[str, BatchTransform], dataset:Dataset) -> dict[str, BatchTransform]:
    if hasattr(args, 'normalized') and args.normalized:
        result = {}
        for key, tsf in tsf_dict.items():
            tsf = tsf.transforms
            data_loader = DataLoader(dataset=TransferDataset(dataset=dataset, data_tf=tsf), batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
            data_mean, data_std = cal_norm(loader=data_loader)
            result[key] = BatchTransform(transforms=Components(transforms=[tsf, v_transforms.Normalize(mean=data_mean, std=data_std)]))
        return result
    else:
        return tsf_dict

def build_dataset(args:argparse.Namespace, data_tsf:nn.Module=None, mode:str='train', data_type:str='all') -> Dataset:
    if args.dataset == 'speech-commands-random':
        return RandomSpeechCommandsDataset(root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=data_tsf, data_type=data_type)
    else:
        return SpeechCommandsDataset(root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=data_tsf, data_type=data_type)

def prepare_data(args: argparse.Namespace, data_transforms=None, mode='train', data_type='all') -> tuple[Dataset, DataLoader]:
    if data_transforms is None:
        data_transforms = pad_trunc(max_ms=1000, sample_rate=args.sample_rate)
    dataset = build_dataset(args=args, data_tsf=data_transforms, mode=mode, data_type=data_type)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    return dataset, data_loader

def train_transforms(args: argparse.Namespace) -> dict[str, BatchTransform]:
    ret = dict()
    ret[TimeShiftOps.LEFT] = BatchTransform(transforms=Components(transforms=[
        time_shift(shift_limit=-args.shift_limit, is_random=False),
        a_transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        a_transforms.FrequencyMasking(freq_mask_param=.1),
        a_transforms.TimeMasking(time_mask_param=.1)
    ]))
    ret[TimeShiftOps.RIGHT] = BatchTransform(transforms=Components(transforms=[
        time_shift(shift_limit=args.shift_limit, is_random=False),
        a_transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        a_transforms.FrequencyMasking(freq_mask_param=.1),
        a_transforms.TimeMasking(time_mask_param=.1)
    ]))
    ret[TimeShiftOps.ORIGIN] = BatchTransform(transforms=Components(transforms=[
        a_transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        a_transforms.FrequencyMasking(freq_mask_param=.1),
        a_transforms.TimeMasking(time_mask_param=.1)
    ]))
    return ret

def val_transforms(args: argparse.Namespace) -> dict[str, BatchTransform]:
    ret = dict()
    ret[TimeShiftOps.LEFT] = BatchTransform(transforms=Components(transforms=[
        time_shift(shift_limit=-args.shift_limit, is_random=False),
        a_transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
    ]))
    ret[TimeShiftOps.RIGHT] = BatchTransform(transforms=Components(transforms=[
        time_shift(shift_limit=args.shift_limit, is_random=False),
        a_transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
    ]))
    ret[TimeShiftOps.ORIGIN] = BatchTransform(transforms=Components(transforms=[
        a_transforms.MelSpectrogram(sample_rate=args.sample_rate, n_fft=1024, n_mels=args.n_mels, hop_length=args.hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
    ]))
    return ret