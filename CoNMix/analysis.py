import argparse
import random
import numpy as np
import os
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.wavUtils import pad_trunc, GuassianNoise, Components
from CoNMix.lib.prepare_dataset import ExpandChannel
from lib.datasets import AudioMINST, load_datapath, StoredDataset

def cal_norm(loader: DataLoader) -> tuple[int, int]:
    mean = 0.
    std = 0.
    for features, _ in tqdm(loader):
        mean += features.mean()
        std += features.std()
    mean /= len(loader)
    std /= len(loader)
    return mean, std

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--cal_norm', type=str)

    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    ap.add_argument('--smooth', type=float, default=.1)
    # ap.add_argument('--da', type=str, default='uda', choices=['uda', 'pda', 'oda'])
    ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    # ap.add_argument('--bsp', type=bool, default=False)
    # ap.add_argument('--se', type=bool, default=False)
    # ap.add_argument('--nl', type=bool, default=False)
    # ap.add_argument('--worker', type=int, default=16)

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'analysis')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    #####################################

    print(f'Generate analysis dataset by {args.corruption} corruption:')
    if args.dataset == 'audio-mnist':
        max_ms=1000
        sample_rate=48000
        n_mels=128
        hop_length=377
        test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.Normalize(mean=[-51.2592887878418, -51.2592887878418, -51.2592887878418], std=[19.16661834716797, 19.16661834716797, 19.16661834716797])
        ])
        audio_minst_load_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] != 'German')
        test_dataset = AudioMINST(data_trainsforms=test_tf, include_rate=False, data_paths=audio_minst_load_pathes)
        
        corrupted_test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            GuassianNoise(noise_level=args.severity_level),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.Normalize(mean=[-30.82235336303711, -30.82235336303711, -30.82235336303711], std=[7.744304656982422, 7.744304656982422, 7.744304656982422])
        ])
        corrupted_test_dataset = AudioMINST(data_trainsforms=corrupted_test_tf, include_rate=False, data_paths=audio_minst_load_pathes)
        corrupted_test_dataset = StoredDataset(dataset=corrupted_test_dataset)
        corrupted_test_dataset.store_to(root_path=args.temporary_path, index_file_name='audio_minst_meta.csv')
    else:
        raise Exception('No support')
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    corrupted_test_loader = DataLoader(dataset=corrupted_test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    if hasattr(args, 'cal_norm'):
        if args.cal_norm == 'original':
            mean, std = cal_norm(test_loader)
        else:
            mean, std = cal_norm(corrupted_test_loader)
        print(f'mean is: {mean}, std is: {std}')
        exit()

    print('Original Test')

    