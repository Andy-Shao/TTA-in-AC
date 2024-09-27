import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as v_transforms
from torchaudio import transforms as a_transforms

from lib.toolkit import cal_norm
from lib.datasets import FilterAudioMNIST
from lib.wavUtils import Components, pad_trunc, time_shift
from lib.scDataset import SpeechCommandsDataset, RandomSpeechCommandsDataset
from CoNMix.lib.prepare_dataset import ExpandChannel

if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--datasets', type=str)
    arg_parse.add_argument('--dataset_root_pathes', type=str)
    arg_parse.add_argument('--num_workers', type=int, default=16)
    arg_parse.add_argument('--output_root_path', type=str, default='./result')
    arg_parse.add_argument('--batch_size', type=int, default=256)
    arg_parse.add_argument('--output_file', type=str, default='mean_std_evaluation.csv')

    args = arg_parse.parse_args()
    args.deivce = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.datasets = [it.strip() for it in args.datasets.strip().split(',')]
    args.dataset_root_pathes = [it.strip() for it in args.dataset_root_pathes.strip().split(',')]
    output_full_root_path = os.path.join(args.output_root_path, 'dataset_analysis')
    try:
        if not os.path.exists(output_full_root_path):
            os.makedirs(output_full_root_path)
    except:
        pass
    ######################################

    records = pd.DataFrame(columns=['dataset', 'type', 'mean', 'standard deviaiton', 'dataset size'])

    for index, dataset in enumerate(args.datasets):
        dataset_root_path = args.dataset_root_pathes[index]
        if dataset == 'audio-mnist':
            class_num = 10
            sample_rate = 48000
            n_mels=128
            hop_length=377
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                time_shift(shift_limit=.25, is_bidirection=True),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
                a_transforms.AmplitudeToDB(top_db=80),
                a_transforms.FrequencyMasking(freq_mask_param=.1),
                a_transforms.TimeMasking(time_mask_param=.1),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
                v_transforms.RandomHorizontalFlip(),
            ]
            train_dataset = FilterAudioMNIST(
                root_path=dataset_root_path, include_rate=False, data_tsf=Components(transforms=[pad_trunc(max_ms=1000, sample_rate=sample_rate)]), 
                filter_fn=lambda x: x['accent'] == 'German'
            )
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((224, 224), antialias=False),
            ]
            test_dataset = FilterAudioMNIST(
                root_path=dataset_root_path, include_rate=False, data_tsf=Components(transforms=[pad_trunc(max_ms=1000, sample_rate=sample_rate)]),
                filter_fn=lambda x: x['accent'] != 'German'
            )
            val_dateset = None
        elif dataset == 'speech-commands':
            class_num = 30
            sample_rate = 16000
            n_mels=129
            hop_length=125
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                time_shift(shift_limit=.25, is_bidirection=True),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                a_transforms.FrequencyMasking(freq_mask_param=.1),
                a_transforms.TimeMasking(time_mask_param=.1),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
                v_transforms.RandomHorizontalFlip(),
            ]
            train_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='train', data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((224, 224), antialias=False),
            ]
            test_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='test', data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
            val_dateset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='validation', data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
        elif dataset == 'speech-commands-norm':
            class_num = 30
            sample_rate = 16000
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate)
            ]
            train_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='train',
                normalized=True
            )
            test_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='test',
                normalized=True
            )
            val_dateset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='validation',
                normalized=True
            )
        elif dataset == 'speech-commands-numbers':
            class_num = 10
            sample_rate = 16000
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate)
            ]
            train_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='train',
                data_type='numbers', data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((224, 224), antialias=False),
            ]
            test_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='test',
                data_type='numbers', data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
            val_dateset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='validation',
                data_type='numbers', data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
        elif dataset == 'speech-commands-random':
            class_num = 30
            sample_rate = 16000
            n_mels=129
            hop_length=125
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                time_shift(shift_limit=.25, is_bidirection=True),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                a_transforms.FrequencyMasking(freq_mask_param=.1),
                a_transforms.TimeMasking(time_mask_param=.1),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
                v_transforms.RandomHorizontalFlip(),
            ]
            train_dataset = RandomSpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='train', 
                data_type='all', seed=2024, data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((224, 224), antialias=False),
            ]
            test_dataset = RandomSpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='test',
                data_type='all', seed=2024, data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
            val_dateset = RandomSpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, mode='validation',
                data_type='all', seed=2024, data_tfs=pad_trunc(max_ms=1000, sample_rate=sample_rate)
            )
        else:
            raise Exception('No support')
    
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

        train_mean, train_std = cal_norm(loader=train_loader)
        print(f'{dataset} Train mean:{train_mean}, std: {train_std}')
        records.loc[len(records)] = [dataset, 'train', train_mean, train_std, len(train_dataset)]

        test_mean, test_std = cal_norm(loader=test_loader)
        print(f'{dataset} Test mean:{test_mean}, std:{test_std}')
        records.loc[len(records)] = [dataset, 'test', test_mean, test_std, len(test_dataset)]

        if val_dateset is not None:
            val_mean, val_std = cal_norm(DataLoader(dataset=val_dateset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers))
            records.loc[len(records)] = [dataset, 'validation', val_mean, val_std, len(val_dateset)]
            print(f'{dataset} validation mean:{val_mean}, std: {val_std}')
    records.to_csv(os.path.join(output_full_root_path, args.output_file))
