import argparse

from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as a_transforms

from lib.wavUtils import pad_trunc, time_shift, Components
from lib.scDataset import SpeechCommandsDataset
from lib.toolkit import BatchTransform
from ttt.lib.prepare_dataset import TimeShiftOps

def prepare_data(args: argparse.Namespace, data_transforms=None, mode='train') -> tuple[Dataset, DataLoader]:
    if data_transforms is None:
        data_transforms = pad_trunc(max_ms=1000, sample_rate=args.sample_rate)
    dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=data_transforms)
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