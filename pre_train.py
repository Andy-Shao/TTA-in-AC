import argparse
import os
from tqdm import tqdm
import pandas as pd

import torch 
import torchaudio.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

from lib.datasets import AudioMINST, load_datapath
from lib.wavUtils import Components, pad_trunc, time_shift
from lib.models import WavClassifier

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn', 'rnn', 'restnet'])
    ap.add_argument('--output_path', type=str)

    args = ap.parse_args()
    if not os.path.exists(f'{args.output_path}/pre_train'):
        os.makedirs(f'{args.output_path}/pre_train')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.dataset == 'audio-mnist':
        data_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        sample_rate = 48000
        n_mels = 40
    else: 
        raise Exception('configure cannot be satisfied')
    
    data_transforms = Components(transforms=[
        pad_trunc(max_ms=1000, sample_rate=sample_rate),
        time_shift(shift_limit=.1),
        transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
        transforms.AmplitudeToDB(top_db=80),
        transforms.FrequencyMasking(freq_mask_param=.1),
        transforms.TimeMasking(time_mask_param=.1)
    ])
    dataset = AudioMINST(data_paths=data_pathes, data_trainsforms=data_transforms, include_rate=False)
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=[.7, .3])
    print(f'train dataset size: {len(train_dataset)}, validation dataset size: {len(val_dataset)}')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.model == 'cnn':
        model = WavClassifier(class_num=10, l1_in_features=64, c1_in_channels=1).to(device=device)
    else:
        raise Exception('Configure cannot be satisfied')
    loss_fn = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    train_step = 0
    val_step = 0
    record = pd.DataFrame(columns=['type', 'step', 'accuracy', 'loss'])
    for epoch in tqdm(range(args.max_epoch)):

        # training
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_accu = (preds == labels).sum().cpu().item()
            record.loc[len(record)] = ['train', train_step, train_accu/labels.shape[0] * 100., loss.cpu().item()]
            train_step += 1 

        # validation
        model.eval()
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)
            val_accu = (preds == labels).sum().cpu().item()
            record.loc[len(record)] = ['validation', val_step, val_accu/labels.shape[0] * 100., loss.cpu().item()]
            val_step += 1 
        torch.save(model.state_dict(), f'{args.output_path}/pre_train/{args.model}_{args.dataset}.pt')
    record.to_csv(f'{args.output_path}/pre_train/{args.model}_{args.dataset}_record.csv')