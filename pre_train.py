import argparse
import os
from tqdm import tqdm
import pandas as pd

import torch 
import torchaudio.transforms as a_transforms
import torchvision.transforms as v_transforms
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

from lib.datasets import AudioMINST, load_datapath, ClipDataset
from lib.wavUtils import Components, pad_trunc, time_shift
from lib.models import WavClassifier, ElasticRestNet50
from lib.toolkit import print_argparse, count_ttl_params, parse_mean_std, cal_norm, store_model_structure_to_txt
from CoNMix.lib.prepare_dataset import ExpandChannel

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='SHOT')
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
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

    args = ap.parse_args()
    print_argparse(args=args)
    args.output_full_path = os.path.join(args.output_path, args.dataset, 'tent', 'pre_train')
    try:
        os.makedirs(args.output_full_path)
    except:
        pass
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'The device is: {device}')
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    if args.dataset == 'audio-mnist' and args.model == 'cnn':
        data_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        sample_rate = 48000
        n_mels = 64
        data_transforms = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            time_shift(shift_limit=.1),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
            a_transforms.FrequencyMasking(freq_mask_param=.1),
            a_transforms.TimeMasking(time_mask_param=.1)
        ])
        dataset = AudioMINST(data_paths=data_pathes, data_trainsforms=data_transforms, include_rate=False)
        train_dataset, val_dataset = random_split(dataset=dataset, lengths=[.7, .3])
        model = WavClassifier(class_num=10, l1_in_features=64, c1_in_channels=1).to(device=device)
    elif args.dataset == 'audio-mnist' and args.model == 'restnet50':
        data_pathes = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        sample_rate = 48000
        n_mels=128
        hop_length=377
        train_tf = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            time_shift(shift_limit=.25, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
            a_transforms.FrequencyMasking(freq_mask_param=.1),
            a_transforms.TimeMasking(time_mask_param=.1),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.RandomHorizontalFlip(),
            v_transforms.Normalize(mean=parse_mean_std(args.train_mean), std=parse_mean_std(args.train_std))
        ])
        train_dataset = AudioMINST(data_paths=data_pathes, data_trainsforms=train_tf, include_rate=False)
        val_tf = Components(transforms=[
            pad_trunc(max_ms=1000, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, n_mels=n_mels),
            a_transforms.AmplitudeToDB(top_db=80),
            ExpandChannel(out_channel=3),
            v_transforms.Resize((256, 256), antialias=False),
            v_transforms.RandomCrop(224),
            v_transforms.Normalize(mean=parse_mean_std(args.val_mean), std=parse_mean_std(args.val_std))
        ])
        val_dataset = ClipDataset(dataset=AudioMINST(data_paths=data_pathes, data_trainsforms=val_tf, include_rate=False), rate=.3)
        model = ElasticRestNet50(class_num=10).to(device=device)
    else: 
        raise Exception('configure cannot be satisfied')
    
    print(f'train dataset size: {len(train_dataset)}, validation dataset size: {len(val_dataset)}')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.cal_norm:
        print('calculate the mean and standard deviation')
        mean, std = cal_norm(loader=train_loader)
        print(f'train: mean: {mean}, std: {std}')
        mean, std = cal_norm(loader=val_loader)
        print(f'validation: mean: {mean}, std: {std}')
        exit()

    print(f'model weight number is: {count_ttl_params(model)}')
    store_model_structure_to_txt(model=model, output_path=os.path.join(args.output_full_path, f'{args.model}.txt'))
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