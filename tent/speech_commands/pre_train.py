import argparse
import os
import numpy as np
import random
from tqdm import tqdm
import wandb

import torch
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from lib.toolkit import print_argparse, store_model_structure_to_txt, cal_norm, count_ttl_params, parse_mean_std
from lib.scDataset import SpeechCommandsDataset
from lib.wavUtils import Components, pad_trunc, time_shift, DoNothing
from lib.models import WavClassifier, ElasticRestNet50
from CoNMix.lib.prepare_dataset import ExpandChannel

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

    class_num = 30
    sample_rate = 16000
    if args.model == 'cnn':
        n_mels = 64
        train_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            time_shift(shift_limit=.1, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
            a_transforms.FrequencyMasking(freq_mask_param=.1),
            a_transforms.TimeMasking(time_mask_param=.1)
        ])
        val_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
        ])
        train_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=train_transforms)
        val_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='validation', include_rate=False, data_tfs=val_transforms)
        model = WavClassifier(class_num=30, l1_in_features=64, c1_in_channels=1).to(device=args.device)
    elif args.model == 'restnet50':
        n_mels=129
        hop_length=125
        train_tf = [
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
        if args.normalized:
            print('calculate the train mean and standard deviation')
            train_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=Components(transforms=train_tf))
            train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, drop_last=True)
            train_mean, train_std = cal_norm(loader=train_loader)
            train_tf.append(v_transforms.Normalize(mean=train_mean, std=train_std))
        train_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tfs=Components(transforms=train_tf))
        val_tf = [
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((224, 224), antialias=False),
        ]
        if args.normalized:
            print('calculate the validation mean and standard deviation')
            val_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='validation', include_rate=False, data_tfs=Components(transforms=val_tf))
            val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, drop_last=True)
            val_mean, val_std = cal_norm(loader=val_loader)
            val_tf.append(v_transforms.Normalize(mean=val_mean, std=val_std))
        val_dataset = SpeechCommandsDataset(root_path=args.dataset_root_path, mode='validation', include_rate=False, data_tfs=Components(transforms=val_tf))
        model = ElasticRestNet50(class_num=class_num).to(device=args.device)
    else:
        raise Exception('No support')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f'train dataset size: {len(train_dataset)}, number of batches: {len(train_loader)}')
    print(f'val dataset size: {len(val_dataset)}, number of batches: {len(val_loader)}')
    store_model_structure_to_txt(model=model, output_path=os.path.join(args.output_full_path, f'{args.model}.txt'))

    wandb_run = wandb.init(
        project=f'Audio Classification Pre-Training ({args.dataset})', name=f'Tent/Norm {args.model}', mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])

    print(f'model weight number is: {count_ttl_params}')
    loss_fn = nn.CrossEntropyLoss().to(device=args.device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    ttl_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch: {epoch+1}/{args.max_epoch}')

        print('Training...')
        ttl_corr = 0
        ttl_num = 0
        ttl_loss = 0.
        model.train()
        for features, labels in tqdm(train_loader):
            optimizer.zero_grad()
            features, labels = features.to(args.device), labels.to(args.device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_num += features.shape[0]
            ttl_loss += loss.detach().cpu().item()
        ttl_accu = ttl_corr/ttl_num * 100.
        avg_loss = ttl_loss/ttl_num
        print(f'Training accuracy: {ttl_accu:.2f}%, loss: {avg_loss:.4f}')
        wandb_run.log({'Train/accuracy': ttl_accu, 'Train/loss': avg_loss}, step=epoch)

        print('Evaluation...')
        ttl_corr = 0
        ttl_num = 0
        model.eval()
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                outputs = model(features)
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_num += features.shape[0]
        curr_accu = ttl_corr/ttl_num * 100.
        wandb_run.log({'Validation/accuracy': curr_accu}, step=epoch)
        print(f'Validation accuracy: {curr_accu:.2f}%')
        if ttl_accu < curr_accu:
            ttl_accu = curr_accu
            torch.save(model.state_dict(), f=os.path.join(args.output_full_path, args.output_weight_name))