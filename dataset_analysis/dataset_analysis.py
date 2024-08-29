import argparse
import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, cal_norm
from lib.datasets import FilterAudioMNIST
from lib.wavUtils import Components, pad_trunc
from lib.scDataset import SpeechCommandsDataset, RandomSpeechCommandsDataset

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
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate)
            ]
            train_dataset = FilterAudioMNIST(
                root_path=dataset_root_path, include_rate=False, data_tsf=Components(transforms=tsf), 
                filter_fn=lambda x: x['accent'] == 'German'
            )
            test_dataset = FilterAudioMNIST(
                root_path=dataset_root_path, include_rate=False, data_tsf=Components(transforms=tsf),
                filter_fn=lambda x: x['accent'] != 'German'
            )
            val_dateset = None
        elif dataset == 'speech-commands':
            class_num = 30
            sample_rate = 16000
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate)
            ]
            train_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='train'
            )
            test_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='test'
            )
            val_dateset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='validation'
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
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='train',
                data_type='numbers'
            )
            test_dataset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='test',
                data_type='numbers'
            )
            val_dateset = SpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='validation',
                data_type='numbers'
            )
        elif dataset == 'speech-commands-random':
            class_num = 30
            sample_rate = 16000
            tsf = [
                pad_trunc(max_ms=1000, sample_rate=sample_rate)
            ]
            train_dataset = RandomSpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='train', 
                data_type='all', seed=2024
            )
            test_dataset = RandomSpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='test',
                data_type='all', seed=2024
            )
            val_dateset = RandomSpeechCommandsDataset(
                root_path=dataset_root_path, include_rate=False, data_tfs=Components(transforms=tsf), mode='validation',
                data_type='all', seed=2024
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
