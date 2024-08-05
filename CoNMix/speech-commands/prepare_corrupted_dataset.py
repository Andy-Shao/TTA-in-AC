import argparse
from tqdm import tqdm

import torch
from torchvision import transforms as v_transforms
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse
from lib.wavUtils import pad_trunc, Components, BackgroundNoise, DoNothing, time_shift, GuassianNoise
from lib.scDataset import SpeechCommandsDataset, BackgroundNoiseDataset
from lib.datasets import store_to, load_from
from CoNMix.lib.prepare_dataset import ExpandChannel

def find_background_noise(args: argparse.Namespace) -> tuple[str, torch.Tensor]:
    background_noise_dataset = BackgroundNoiseDataset(root_path=args.dataset_root_path)
    for noise_type, noise, _ in background_noise_dataset:
        if args.corruption == noise_type:
            return noise_type, noise
    return ()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')

    ap.add_argument('--corruption', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'gaussian_noise'])
    ap.add_argument('--severity_level', type=float, default=20)
    # ap.add_argument('--cal_norm', action='store_true')
    ap.add_argument('--cal_strong', action='store_true')

    # ap.add_argument('--seed', type=int, default=2024, help='random seed')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    #####################################

    max_ms = 1000
    sample_rate = 16000
    n_mels=129
    hop_length=125
    meta_file_name = 'speech_commands_meta.csv'

    if args.corruption == 'gaussian_noise':
        is_background_noise = False
        noise_type = args.corruption
        corrupted_test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            GuassianNoise(noise_level=args.severity_level),
        ])
    else:
        is_background_noise = True
        noise_type, noise = find_background_noise(args)
        corrupted_test_tf = Components(transforms=[
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            BackgroundNoise(noise_level=args.severity_level, noise=noise, is_random=True),
        ])

    print(f'Generate the corrupted dataset by {noise_type}')
    output_path = f'{args.output_path}-{noise_type}'
    
    corrupted_test_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=corrupted_test_tf)
    store_to(dataset=corrupted_test_dataset, root_path=output_path, index_file_name=meta_file_name)
    corrupted_test_dataset = load_from(root_path=output_path, index_file_name=meta_file_name)

    print('low augmentation')
    weak_path = f'{output_path}-weak'
    if args.data_type == 'raw':
        weak_tf = DoNothing()
    elif args.data_type == 'final':
        weak_tf = Components(transforms=[
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            # v_transforms.Resize((256, 256), antialias=False),
            # v_transforms.RandomCrop(224),
            v_transforms.Resize((224, 224), antialias=False),
        ])
    store_to(dataset=corrupted_test_dataset, root_path=weak_path, index_file_name=meta_file_name, data_transf=weak_tf)
    weak_aug_dataset = load_from(root_path=weak_path, index_file_name=meta_file_name)
    print(f'Foreach checking, datasize: {len(weak_aug_dataset)}')
    for feature, label in tqdm(weak_aug_dataset):
        pass

    if args.cal_strong:
        print('strong augmentation')
        strong_path = f'{output_path}-strong'
        if args.data_type == 'raw':
            strong_tf = Components(transforms=[
                a_transforms.PitchShift(sample_rate=sample_rate, n_steps=4, n_fft=512),
                time_shift(shift_limit=.25, is_random=True, is_bidirection=True)
            ])
        elif args.data_type == 'final':
            strong_tf = Components(transforms=[
                a_transforms.PitchShift(sample_rate=sample_rate, n_steps=4, n_fft=512),
                time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
                a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
                a_transforms.AmplitudeToDB(top_db=80),
                # a_transforms.FrequencyMasking(freq_mask_param=.1),
                # a_transforms.TimeMasking(time_mask_param=.1),
                ExpandChannel(out_channel=3),
                v_transforms.Resize((256, 256), antialias=False),
                v_transforms.RandomCrop(224),
                # v_transforms.Resize((224, 224), antialias=False),
            ])
        store_to(dataset=corrupted_test_dataset, root_path=strong_path, index_file_name=meta_file_name, data_transf=strong_tf)
        strong_aug_dataset = load_from(root_path=strong_path, index_file_name=meta_file_name)
        print(f'Foreach checking, datasize: {len(strong_aug_dataset)}')
        for feature, label in tqdm(strong_aug_dataset):
            pass
