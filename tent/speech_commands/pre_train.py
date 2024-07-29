import argparse
import os
import numpy as np
import random

import torch
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, store_model_structure_to_txt
from lib.scDataset import SpeechCommandsDataset
from lib.wavUtils import Components, pad_trunc, time_shift
from lib.models import WavClassifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn', 'restnet50'])
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_name', type=str, default='model_weights.pt')
    ap.add_argument('--train_mean', type=str, default='0.,0.,0.')
    ap.add_argument('--train_std', type=str, default='1.,1.,1.')
    ap.add_argument('--val_mean', type=str, default='0.,0.,0.')
    ap.add_argument('--val_std', type=str, default='1.,1.,1.')
    ap.add_argument('--cal_norm', action='store_true')
    ap.add_argument('--normalized', action='store_true')
    ap.add_argument('--seed', type=int, default=2024)

    args = ap.parse_args()
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'tent', 'pre_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass
        
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print_argparse(args=args)

    if args.dataset == 'speech-commands' and args.model == 'cnn':
        sample_rate = 16000
        n_mels = 64
        audio_len = 1000
        class_num = 30
        train_transforms = Components(transforms=[
            pad_trunc(max_ms=audio_len, sample_rate=sample_rate),
            time_shift(shift_limit=.1, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
            a_transforms.FrequencyMasking(freq_mask_param=.1),
            a_transforms.TimeMasking(time_mask_param=.1)
        ])
        val_transforms = Components(transforms=[
            pad_trunc(max_ms=audio_len, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
        ])
        train_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=train_transforms)
        val_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='validation', include_rate=False, data_tfs=val_transforms)
        model = WavClassifier(class_num=30, l1_in_features=64, c1_in_channels=1).to(device=args.device)
    else:
        raise Exception('No support')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f'train dataset size: {len(train_dataset)}, number of batches: {len(train_loader)}')
    print(f'val dataset size: {len(val_dataset)}, number of batches: {len(val_loader)}')
    store_model_structure_to_txt(model=model, output_path=os.path.join(args.output_full_path, f'{args.model}.txt'))

    for features, labels in train_loader:
        features, labels = features.to(args.device), labels.to(args.device)
        outputs = model(features)
        break