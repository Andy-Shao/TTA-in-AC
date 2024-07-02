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
from lib.toolkit import print_argparse, count_ttl_params

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--model', type=str, default='cnn', choices=['cnn', 'rnn', 'restnet'])
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_name', type=str, default='model_weights.pt')

    args = ap.parse_args()
    print_argparse(args=args)
    args.output_full_path = os.path.join(args.output_path, args.dataset, args.model, 'pre_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'The device is: {device}')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    if args.dataset == 'audio-mnist':
        data_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        sample_rate = 48000
        n_mels = 64
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
    print(f'model weight number is: {count_ttl_params(model)}')
    loss_fn = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    train_step = 0
    val_step = 0
    record = pd.DataFrame(columns=['type', 'step', 'accuracy', 'loss'])
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch}')

        print('training...')
        model.train()
        train_loader_iter = iter(train_loader)
        for i in tqdm(range(len(train_loader))):
            inputs, labels = next(train_loader_iter)
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
        print(f'train accuracy: {record.iloc[-1, 2]:.2f}%, train loss: {record.iloc[-1, 3]:.2f}')

        print('validation')
        model.eval()
        val_loader_iter = iter(val_loader)
        for i in tqdm(range(len(val_loader))):
            inputs, labels = next(val_loader_iter)
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)
            val_accu = (preds == labels).sum().cpu().item()
            record.loc[len(record)] = ['validation', val_step, val_accu/labels.shape[0] * 100., loss.cpu().item()]
            val_step += 1 
        print(f'validation accuracy: {record.iloc[-1, 2]:.2f}%, validation loss: {record.iloc[-1, 3]:.2f}')
        torch.save(model.state_dict(), os.path.join(args.output_full_path, args.output_weight_name))
    record.to_csv(os.path.join(args.output_full_path, args.output_csv_name))