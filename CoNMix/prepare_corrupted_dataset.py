import argparse
import os
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, cal_norm
from lib.prepare_dataset import ExpandChannel
from lib.wavUtils import Components, pad_trunc, GuassianNoise
from lib.datasets import AudioMINST, load_datapath, store_to, load_from

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')

    ap.add_argument('--corruption', type=str, default='gaussian_noise')
    ap.add_argument('--severity_level', type=float, default=.0025)
    ap.add_argument('--cal_norm', action='store_true')

    # ap.add_argument('--seed', type=int, default=2024, help='random seed')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'CoNMix', 'analysis')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
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
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224)
        ])
        audio_minst_load_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] != 'German')
        corrupted_test_dataset = AudioMINST(data_trainsforms=corrupted_test_tf, include_rate=False, data_paths=audio_minst_load_pathes)
    else:
        raise Exception('No support')
    
    store_to(dataset=corrupted_test_dataset, root_path=args.temporary_path, index_file_name='audio_minst_meta.csv')
    sd = load_from(root_path=args.temporary_path, index_file_name='audio_minst_meta.csv')

    print(f'Foreach checking, datasize: {len(sd)}')
    for feature, label in tqdm(sd):
        pass

    if args.cal_norm:
        data_loader = DataLoader(dataset=sd, batch_size=256, shuffle=False, drop_last=False)
        mean, std = cal_norm(data_loader)
        print(f'mean: {mean}, std: {std}')