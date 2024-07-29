import argparse
import os

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.scDataset import BackgroundNoise, SpeechCommandsDataset
from lib.wavUtils import Components, pad_trunc, BackgroundNoise

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn', 'restnet50'])
    ap.add_argument('--model_weight_file_path', type=str)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--severity_level', default=1., type=float)
    ap.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    ap.add_argument('--test_mean', type=str, default='0., 0., 0.')
    ap.add_argument('--test_std', type=str, default='1., 1., 1.')
    ap.add_argument('--noise_type', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise'])
    ap.add_argument('--corrupted_test_mean', type=str, default='0., 0., 0.')
    ap.add_argument('--corrupted_test_std', type=str, default='1., 1., 1.')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--tent_batch_size', type=int, default=200)
    ap.add_argument('--cal_norm', action='store_true')
    ap.add_argument('--normalized', action='store_true')

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'tent', 'analysis')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass

    print_argparse(args=args)

    sample_rate = 16000
    background_noise_dataset = BackgroundNoise(root_path=args.dataset_root_path)
    noise_audio = None
    for noise_type, noise, _ in background_noise_dataset:
        if noise_type == args.noise_type:
            noise_audio = noise
    assert noise_audio is None, 'without choiced noise'
    if args.model == 'cnn':
        n_mels = 64
        data_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
        ])
        test_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tfs=data_transforms)
        corrupted_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            BackgroundNoise(noise_level=args.severity_level, noise=noise_audio, is_random=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
        ])
        corrupted_dataset = SpeechCommandsDataset(root_path=args.data_root_path, mode='test', include_rate=False, data_tfs=corrupted_transforms)
    else: 
        raise Exception('No support')
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    corrupted_loader = DataLoader(dataset=corrupted_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print('Original test')