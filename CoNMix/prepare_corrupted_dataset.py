import argparse
import os
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms

from lib.toolkit import print_argparse, cal_norm
from lib.wavUtils import Components, pad_trunc, GuassianNoise, time_shift
from lib.datasets import AudioMINST, load_datapath, store_to, load_from
from CoNMix.lib.prepare_dataset import ExpandChannel

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)
    ap.add_argument('--cal_norm', action='store_true')
    ap.add_argument('--cal_strong', action='store_true')

    # ap.add_argument('--seed', type=int, default=2024, help='random seed')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    #####################################

    print(f'Generate analysis dataset by {args.corruption} corruption:')
    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        corrupted_test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            GuassianNoise(noise_level=args.severity_level),
        ])
        audio_minst_load_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] != 'German')
        corrupted_test_dataset = AudioMINST(data_trainsforms=corrupted_test_tf, include_rate=False, data_paths=audio_minst_load_pathes)
    else:
        raise Exception('No support')
    meta_file_name = 'audio_minst_meta.csv'

    if args.data_type == 'final':
        store_to(dataset=corrupted_test_dataset, root_path=args.output_path, index_file_name=meta_file_name)
        corrupted_test_dataset = load_from(root_path=args.output_path, index_file_name=meta_file_name)

    weak_tf = Components(transforms=[
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
        a_transforms.AmplitudeToDB(top_db=80),
        ExpandChannel(out_channel=3),
        v_transforms.Resize((256, 256), antialias=False),
        v_transforms.RandomCrop(224)
    ])
    print('low augmentation')
    weak_path = f'{args.output_path}_weak'
    if args.data_type == 'raw':
        store_to(dataset=corrupted_test_dataset, root_path=weak_path, index_file_name=meta_file_name)
    elif args.data_type == 'final':
        store_to(dataset=corrupted_test_dataset, root_path=weak_path, index_file_name=meta_file_name, data_transf=weak_tf)
    else:
        raise Exception('No support')
    weak_aug_dataset = load_from(root_path=weak_path, index_file_name=meta_file_name)

    print(f'Foreach checking, datasize: {len(weak_aug_dataset)}')
    for feature, label in tqdm(weak_aug_dataset):
        pass

    if args.cal_norm:
        data_loader = DataLoader(dataset=weak_aug_dataset, batch_size=256, shuffle=False, drop_last=False)
        mean, std = cal_norm(data_loader)
        result = f'mean: {mean}, std: {std}'
        print(result)
        with open(os.path.join(weak_path, 'mean_std.txt'), 'w') as f:
            f.write(result)
            f.flush()

    if args.cal_strong:
        print('strong augmentation')
        strong_path = f'{args.output_path}_strong'
        if args.data_type == 'raw':
            strong_tf = Components(transforms=[
                a_transforms.PitchShift(sample_rate=sample_rate, n_steps=4, n_fft=512),
                time_shift(shift_limit=.25, is_random=True, is_bidirection=True)
            ])
            store_to(dataset=weak_aug_dataset, root_path=strong_path, index_file_name=meta_file_name, data_transf=strong_tf)
        elif args.data_type == 'final':
            strong_tf = Components(transforms=[
                a_transforms.PitchShift(sample_rate=sample_rate, n_steps=4, n_fft=512),
                time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224)
            ])
            store_to(dataset=corrupted_test_dataset, root_path=strong_path, index_file_name=meta_file_name, data_transf=strong_tf)
        else:
            raise Exception('No support')
        strong_aug_dataset = load_from(root_path=strong_path, index_file_name=meta_file_name)

        print(f'Foreach checking, datasize: {len(strong_aug_dataset)}')
        for feature, label in tqdm(strong_aug_dataset):
            pass

        if args.cal_norm:
            data_loader = DataLoader(dataset=strong_aug_dataset, batch_size=256, shuffle=False, drop_last=False)
            mean, std = cal_norm(data_loader)
            result = f'mean: {mean}, std: {std}'
            print(result)
            with open(os.path.join(strong_path, 'mean_std.txt'), 'w') as f:
                f.write(result)
                f.flush()